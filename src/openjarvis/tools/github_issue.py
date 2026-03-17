"""GitHub issue tool — create and label GitHub issues for agent escalation.

Used by local agents to escalate tasks they cannot complete to the
ClaudeCodeAgent worker queue.  Set ``GITHUB_TOKEN`` and ``GITHUB_REPO``
(``owner/repo``) in the environment before use.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

_GITHUB_API = "https://api.github.com"
_ESCALATION_LABEL = "claude-code-work"
_ESCALATION_LABEL_COLOR = "e11d48"  # red


def _ensure_label(repo: str, token: str) -> None:
    """Create the escalation label on the repo if it does not exist."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    resp = httpx.get(
        f"{_GITHUB_API}/repos/{repo}/labels/{_ESCALATION_LABEL}",
        headers=headers,
        timeout=10,
    )
    if resp.status_code == 404:
        httpx.post(
            f"{_GITHUB_API}/repos/{repo}/labels",
            headers=headers,
            json={
                "name": _ESCALATION_LABEL,
                "color": _ESCALATION_LABEL_COLOR,
                "description": "Queued for ClaudeCodeAgent autonomous resolution",
            },
            timeout=10,
        )


@ToolRegistry.register("github_issue")
class GitHubIssueTool(BaseTool):
    """Create a GitHub issue to escalate a task to the ClaudeCodeAgent worker.

    When a local agent cannot resolve a task it should call this tool with
    a detailed description of what was tried and what is blocking progress.
    The ClaudeCodeAgent worker polls for ``claude-code-work`` labelled issues
    and works them autonomously.
    """

    tool_id = "github_issue"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="github_issue",
            description=(
                "Escalate a task you cannot complete by creating a GitHub issue "
                "labelled 'claude-code-work'. The ClaudeCodeAgent worker will pick "
                "it up and attempt to resolve it autonomously. Use this when you "
                "have exhausted your available approaches and need a more capable "
                "agent to continue."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short, descriptive issue title (< 80 chars).",
                    },
                    "body": {
                        "type": "string",
                        "description": (
                            "Full issue body. Include: the original task, all "
                            "approaches attempted, specific blocker, relevant file "
                            "paths, and any partial output. Markdown is supported."
                        ),
                    },
                    "extra_labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional additional labels to apply.",
                    },
                },
                "required": ["title", "body"],
            },
            category="github",
        )

    def execute(self, **params: Any) -> ToolResult:
        title: str = params.get("title", "").strip()
        body: str = params.get("body", "").strip()
        extra_labels: list[str] = params.get("extra_labels", [])

        if not title or not body:
            return ToolResult(
                tool_name="github_issue",
                content="Error: title and body are required.",
                success=False,
            )

        token = os.environ.get("GITHUB_TOKEN", "")
        repo = os.environ.get("GITHUB_REPO", "")

        if not token or not repo:
            return ToolResult(
                tool_name="github_issue",
                content=(
                    "Error: GITHUB_TOKEN and GITHUB_REPO environment variables "
                    "must be set to create issues."
                ),
                success=False,
            )

        try:
            _ensure_label(repo, token)

            labels = [_ESCALATION_LABEL] + [l for l in extra_labels if l]
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github+json",
            }
            resp = httpx.post(
                f"{_GITHUB_API}/repos/{repo}/issues",
                headers=headers,
                json={"title": title, "body": body, "labels": labels},
                timeout=15,
            )

            if resp.status_code == 201:
                data = resp.json()
                url = data.get("html_url", "")
                number = data.get("number", "?")
                return ToolResult(
                    tool_name="github_issue",
                    content=(
                        f"Issue #{number} created and labelled '{_ESCALATION_LABEL}'.\n"
                        f"The ClaudeCodeAgent worker will pick this up shortly.\n"
                        f"Track it at: {url}"
                    ),
                    success=True,
                    metadata={"issue_number": number, "url": url},
                )

            return ToolResult(
                tool_name="github_issue",
                content=f"GitHub API error {resp.status_code}: {resp.text[:500]}",
                success=False,
            )

        except httpx.RequestError as exc:
            return ToolResult(
                tool_name="github_issue",
                content=f"Network error creating issue: {exc}",
                success=False,
            )


__all__ = ["GitHubIssueTool"]
