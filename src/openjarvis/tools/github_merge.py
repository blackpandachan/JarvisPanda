"""GitHub PR merge tool — lets local agents merge PRs programmatically.

Used by agents after verifying that a ClaudeCodeAgent PR is ready to merge.
Part of the autonomous review-and-merge workflow:

  claude-worker creates PR + posts implementation guide
       ↓
  local agent (or human) runs tests
       ↓
  if tests pass: agent calls github_merge(pr_number=N)
       ↓
  if merge fails: agent uses gemini_escalate to diagnose, then cuts follow-up issue

Required env vars: GITHUB_TOKEN, GITHUB_REPO
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

_GITHUB_API = "https://api.github.com"


def _gh_headers() -> dict[str, str]:
    return {
        "Authorization": f"token {os.environ.get('GITHUB_TOKEN', '')}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


@ToolRegistry.register("github_merge")
class GitHubMergeTool(BaseTool):
    """Merge a GitHub pull request after verifying the implementation is correct.

    Use this after checking out a ClaudeCodeAgent branch, running tests, and
    confirming the changes are safe to merge.  Performs a squash merge and
    deletes the source branch.
    """

    tool_id = "github_merge"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="github_merge",
            description=(
                "Merge a GitHub pull request. Use this after verifying the "
                "implementation is correct (tests pass, changes look good). "
                "Performs squash merge and optionally deletes the source branch. "
                "Only use on PRs created by ClaudeCodeAgent (labelled claude-code-done)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pr_number": {
                        "type": "integer",
                        "description": "The pull request number to merge.",
                    },
                    "merge_method": {
                        "type": "string",
                        "enum": ["squash", "merge", "rebase"],
                        "description": "Merge strategy (default: squash).",
                        "default": "squash",
                    },
                    "delete_branch": {
                        "type": "boolean",
                        "description": "Delete the source branch after merge (default: true).",
                        "default": True,
                    },
                },
                "required": ["pr_number"],
            },
            category="github",
            requires_confirmation=True,  # Ask before executing
        )

    def execute(self, **params: Any) -> ToolResult:
        pr_number: int      = int(params.get("pr_number", 0))
        merge_method: str   = params.get("merge_method", "squash")
        delete_branch: bool = bool(params.get("delete_branch", True))

        if not pr_number:
            return ToolResult(
                tool_name="github_merge",
                content="Error: pr_number is required.",
                success=False,
            )

        token = os.environ.get("GITHUB_TOKEN", "")
        repo  = os.environ.get("GITHUB_REPO", "")
        if not token or not repo:
            return ToolResult(
                tool_name="github_merge",
                content="GITHUB_TOKEN and GITHUB_REPO must be set.",
                success=False,
            )

        try:
            # Get PR details first
            pr_resp = httpx.get(
                f"{_GITHUB_API}/repos/{repo}/pulls/{pr_number}",
                headers=_gh_headers(), timeout=10,
            )
            if pr_resp.status_code == 404:
                return ToolResult(
                    tool_name="github_merge",
                    content=f"PR #{pr_number} not found in {repo}.",
                    success=False,
                )
            pr_data   = pr_resp.json()
            pr_state  = pr_data.get("state", "")
            pr_url    = pr_data.get("html_url", "")
            pr_title  = pr_data.get("title", "")
            head_ref  = pr_data.get("head", {}).get("ref", "")
            mergeable = pr_data.get("mergeable")

            if pr_state != "open":
                return ToolResult(
                    tool_name="github_merge",
                    content=f"PR #{pr_number} is {pr_state}, not open. Nothing to merge.",
                    success=False,
                )

            if mergeable is False:
                return ToolResult(
                    tool_name="github_merge",
                    content=(
                        f"PR #{pr_number} has merge conflicts and cannot be merged automatically.\n"
                        f"Use gemini_escalate to get a conflict resolution plan, or resolve manually."
                    ),
                    success=False,
                )

            # Merge
            merge_resp = httpx.put(
                f"{_GITHUB_API}/repos/{repo}/pulls/{pr_number}/merge",
                headers=_gh_headers(),
                json={
                    "merge_method": merge_method,
                    "commit_title": f"[JarvisPanda] {pr_title} (#{pr_number})",
                },
                timeout=15,
            )

            if merge_resp.status_code == 200:
                sha = merge_resp.json().get("sha", "?")[:8]
                result_msg = (
                    f"✅ PR #{pr_number} merged successfully ({merge_method}) → "
                    f"commit `{sha}`\n{pr_url}"
                )

                # Delete branch if requested
                if delete_branch and head_ref:
                    try:
                        httpx.delete(
                            f"{_GITHUB_API}/repos/{repo}/git/refs/heads/{head_ref}",
                            headers=_gh_headers(), timeout=10,
                        )
                        result_msg += f"\nBranch `{head_ref}` deleted."
                    except Exception as exc:
                        result_msg += f"\nNote: branch deletion failed: {exc}"

                return ToolResult(
                    tool_name="github_merge",
                    content=result_msg,
                    success=True,
                    metadata={"pr_number": pr_number, "sha": sha, "branch": head_ref},
                )

            if merge_resp.status_code == 405:
                return ToolResult(
                    tool_name="github_merge",
                    content=f"PR #{pr_number} is not mergeable (GitHub 405). Check CI status.",
                    success=False,
                )

            return ToolResult(
                tool_name="github_merge",
                content=f"Merge failed (HTTP {merge_resp.status_code}): {merge_resp.text[:300]}",
                success=False,
            )

        except httpx.RequestError as exc:
            return ToolResult(
                tool_name="github_merge",
                content=f"Network error merging PR: {exc}",
                success=False,
            )


__all__ = ["GitHubMergeTool"]
