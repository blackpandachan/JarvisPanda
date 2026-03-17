"""Jarvis Claude Code Issue Worker.

Polls GitHub for issues labelled ``claude-code-work``, claims them, runs
the ClaudeCodeAgent, then depending on MERGE_STRATEGY:

  manual (default) — push branch + open PR, post implementation guide on both
                     the PR and the original issue, notify Discord with merge
                     instructions.  Human or local agent merges via !merge <pr>.

  auto             — same as manual, but auto-merges after MERGE_DELAY_MINUTES
                     if no one has closed or declined the PR.

  comment          — legacy: post agent output as issue comment, no branch/PR.

After a PR is created the worker posts a structured implementation guide that
tells local agents exactly how to verify and merge the work.  If the local
agent encounters problems merging, it can escalate via gemini_escalate or cut
a follow-up issue.

Label lifecycle:
  claude-code-work        → queued (created by local qwen3 or Discord !propose)
  claude-code-in-progress → worker claimed it
  claude-code-done        → PR opened (or comment posted); awaiting merge/review
  claude-code-failed      → worker errored; issue stays open for manual triage

Required env vars:
  GITHUB_TOKEN       PAT with repo scope (read + write issues + PRs)
  GITHUB_REPO        owner/repo
  ANTHROPIC_API_KEY  For ClaudeCodeAgent

Optional env vars:
  WORKSPACE           Repo path inside container (default /workspace)
  POLL_INTERVAL       Seconds between polls (default 300)
  MAX_ISSUE_AGE_DAYS  Skip stale issues (default 14)
  WORKER_LABEL_WORK   Override work label (default claude-code-work)
  MERGE_STRATEGY      manual | auto | comment (default manual)
  MERGE_DELAY_MINUTES Minutes to wait before auto-merge (default 15, auto only)
  DISCORD_DIGEST_WEBHOOK  Webhook for PR-ready notifications
  DEFAULT_BRANCH      Base branch to create PRs against (default main)
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("claude-worker")

# ── Config ────────────────────────────────────────────────────────────────────

GITHUB_TOKEN: str      = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO: str       = os.environ.get("GITHUB_REPO", "")
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
WORKSPACE: str         = os.environ.get("WORKSPACE", "/workspace")
POLL_INTERVAL: int     = int(os.environ.get("POLL_INTERVAL", "300"))
MAX_ISSUE_AGE_DAYS: int = int(os.environ.get("MAX_ISSUE_AGE_DAYS", "14"))
MERGE_STRATEGY: str    = os.environ.get("MERGE_STRATEGY", "manual")  # manual|auto|comment
MERGE_DELAY: int       = int(os.environ.get("MERGE_DELAY_MINUTES", "15")) * 60
DISCORD_WEBHOOK: str   = os.environ.get("DISCORD_DIGEST_WEBHOOK", "")
DEFAULT_BRANCH: str    = os.environ.get("DEFAULT_BRANCH", "main")

LABEL_WORK   = os.environ.get("WORKER_LABEL_WORK", "claude-code-work")
LABEL_WIP    = "claude-code-in-progress"
LABEL_DONE   = "claude-code-done"
LABEL_FAILED = "claude-code-failed"

GITHUB_API = "https://api.github.com"
LABEL_COLORS = {
    LABEL_WORK:   "e11d48",
    LABEL_WIP:    "f59e0b",
    LABEL_DONE:   "10b981",
    LABEL_FAILED: "6b7280",
}

# ── GitHub helpers ────────────────────────────────────────────────────────────

def _gh(timeout: int = 15) -> httpx.Client:
    return httpx.Client(
        headers={
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=timeout,
    )


def ensure_labels() -> None:
    with _gh() as http:
        for name, color in LABEL_COLORS.items():
            r = http.get(f"{GITHUB_API}/repos/{GITHUB_REPO}/labels/{name}")
            if r.status_code == 200:
                continue
            if r.status_code == 404:
                cr = http.post(
                    f"{GITHUB_API}/repos/{GITHUB_REPO}/labels",
                    json={"name": name, "color": color},
                )
                if cr.status_code == 201:
                    log.info("Created label: %s", name)
                elif cr.status_code == 403:
                    log.warning("Cannot create label %r — token needs repo write scope (403).", name)
                else:
                    log.warning("Unexpected %d creating label %r: %s", cr.status_code, name, cr.text[:200])


def get_queued_issues() -> list[dict]:
    with _gh() as http:
        r = http.get(
            f"{GITHUB_API}/repos/{GITHUB_REPO}/issues",
            params={"labels": LABEL_WORK, "state": "open", "per_page": 20},
        )
        r.raise_for_status()
        issues = r.json()

    cutoff = int(datetime.now(timezone.utc).timestamp()) - MAX_ISSUE_AGE_DAYS * 86400
    result = []
    for issue in issues:
        labels = {lbl["name"] for lbl in issue.get("labels", [])}
        if LABEL_WIP in labels:
            continue
        try:
            ts = int(datetime.fromisoformat(
                issue.get("created_at", "").replace("Z", "+00:00")
            ).timestamp())
        except Exception:
            ts = 0
        if ts and ts < cutoff:
            log.info("Skipping stale issue #%d", issue["number"])
            continue
        result.append(issue)
    return result


def set_labels(issue_number: int, add: list[str], remove: list[str]) -> None:
    with _gh() as http:
        r = http.get(f"{GITHUB_API}/repos/{GITHUB_REPO}/issues/{issue_number}")
        current   = {lbl["name"] for lbl in r.json().get("labels", [])}
        new_labels = list((current | set(add)) - set(remove))
        http.patch(
            f"{GITHUB_API}/repos/{GITHUB_REPO}/issues/{issue_number}",
            json={"labels": new_labels},
        )


def post_comment(issue_number: int, body: str) -> None:
    with _gh() as http:
        http.post(
            f"{GITHUB_API}/repos/{GITHUB_REPO}/issues/{issue_number}/comments",
            json={"body": body},
        )


def post_pr_comment(pr_number: int, body: str) -> None:
    with _gh() as http:
        http.post(
            f"{GITHUB_API}/repos/{GITHUB_REPO}/issues/{pr_number}/comments",
            json={"body": body},
        )


def merge_pr(pr_number: int, method: str = "squash") -> tuple[bool, str]:
    """Merge a PR. Returns (success, message)."""
    with _gh() as http:
        r = http.put(
            f"{GITHUB_API}/repos/{GITHUB_REPO}/pulls/{pr_number}/merge",
            json={
                "merge_method": method,
                "commit_title": f"Auto-merge PR #{pr_number} (ClaudeCodeAgent)",
            },
        )
    if r.status_code == 200:
        return True, r.json().get("message", "Merged successfully")
    return False, f"HTTP {r.status_code}: {r.text[:300]}"


def get_pr_state(pr_number: int) -> str | None:
    """Return 'open', 'closed', or None on error."""
    with _gh() as http:
        r = http.get(f"{GITHUB_API}/repos/{GITHUB_REPO}/pulls/{pr_number}")
    if r.status_code == 200:
        return r.json().get("state")
    return None

# ── Git helpers ───────────────────────────────────────────────────────────────

def _git(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args, cwd=WORKSPACE, capture_output=True, text=True, check=check,
    )


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50]


def prepare_branch(issue_number: int, title: str) -> str:
    branch = f"claude-code/issue-{issue_number}-{_slugify(title)}"
    try:
        _git(["fetch", "origin"])
        _git(["checkout", "-B", branch, f"origin/{DEFAULT_BRANCH}"])
    except subprocess.CalledProcessError:
        _git(["checkout", "-B", branch], check=False)
    log.info("Branch ready: %s", branch)
    return branch


def push_branch(branch: str) -> bool:
    try:
        _git(["push", "origin", branch, "--force-with-lease", "--set-upstream"])
        return True
    except subprocess.CalledProcessError as exc:
        log.error("git push failed: %s", exc.stderr)
        return False


def get_changed_files(branch: str) -> str:
    """Return a formatted list of changed files vs DEFAULT_BRANCH."""
    try:
        r = _git(["diff", "--name-status", f"origin/{DEFAULT_BRANCH}...{branch}"])
        lines = r.stdout.strip().splitlines()
        if not lines:
            return "_No file changes detected._"
        # Format: M src/foo.py → "modified: src/foo.py"
        labels = {"M": "modified", "A": "added", "D": "deleted", "R": "renamed"}
        out = []
        for ln in lines[:30]:  # cap at 30
            parts = ln.split("\t", 1)
            if len(parts) == 2:
                action = labels.get(parts[0][0], parts[0])
                out.append(f"- **{action}**: `{parts[1]}`")
            else:
                out.append(f"- `{ln}`")
        if len(lines) > 30:
            out.append(f"- _...and {len(lines) - 30} more files_")
        return "\n".join(out)
    except Exception as exc:
        log.debug("get_changed_files: %s", exc)
        return "_Could not determine changed files._"


def get_commit_summary(branch: str) -> str:
    """Return recent commits on the branch relative to DEFAULT_BRANCH."""
    try:
        r = _git([
            "log", f"origin/{DEFAULT_BRANCH}..{branch}",
            "--oneline", "--no-merges", "--max-count=10",
        ])
        lines = r.stdout.strip().splitlines()
        if not lines:
            return "_No commits yet._"
        return "\n".join(f"- `{ln}`" for ln in lines)
    except Exception:
        return "_Could not determine commits._"

# ── PR creation ───────────────────────────────────────────────────────────────

def create_pr(issue_number: int, title: str, branch: str, agent_summary: str) -> tuple[int | None, str | None]:
    """Create a GitHub PR. Returns (pr_number, pr_url) or (None, None)."""
    body = (
        f"Closes #{issue_number}\n\n"
        f"## Agent Summary\n\n{agent_summary[:3000]}\n\n"
        f"---\n"
        f"*Implemented by [JarvisPanda](https://github.com/blackpandachan/JarvisPanda) "
        f"ClaudeCodeAgent (`claude-opus-4-6`). Review the changes, then merge with "
        f"`!merge {'{pr_number}'}` in Discord or via GitHub UI.*"
    )
    with _gh() as http:
        r = http.post(
            f"{GITHUB_API}/repos/{GITHUB_REPO}/pulls",
            json={
                "title": f"[Claude] {title}",
                "body": body,
                "head": branch,
                "base": DEFAULT_BRANCH,
            },
        )
    if r.status_code == 201:
        data   = r.json()
        pr_num = data.get("number")
        pr_url = data.get("html_url", "")
        log.info("PR #%d created: %s", pr_num, pr_url)
        return pr_num, pr_url
    log.error("PR creation failed (%d): %s", r.status_code, r.text[:300])
    return None, None


def post_implementation_guide(
    issue_number: int,
    pr_number: int,
    pr_url: str,
    branch: str,
    agent_summary: str,
) -> None:
    """Post a structured guide on the PR and original issue for local agents."""
    changed = get_changed_files(branch)
    commits = get_commit_summary(branch)

    # Extract a clean summary — first 1500 chars of agent output
    summary_excerpt = agent_summary[:1500].strip()
    if len(agent_summary) > 1500:
        summary_excerpt += "\n\n_(full output in PR description)_"

    guide = f"""## 🤖 Implementation Complete — Action Required

**PR:** {pr_url}
**Branch:** `{branch}`

---

### What Changed

{changed}

### Commits

{commits}

### Implementation Summary

{summary_excerpt}

---

### Verification & Merge Instructions

**For local agents (automated):**
```bash
# Fetch and check out the branch
git fetch origin
git checkout {branch}

# Run tests (adjust command to match project)
uv run pytest tests/ -v

# If tests pass, merge via Discord:
# Post in #jarvis:  !merge {pr_number}
```

**For humans:**
- Review the PR at {pr_url}
- Merge via Discord: `!merge {pr_number}`
- Or merge via GitHub UI

**If the verification fails:**
1. Post `!propose <what failed and why>` in Discord to cut a follow-up issue
2. The agent will consult Gemini and open a new issue with a fix plan

---
*Implemented autonomously by ClaudeCodeAgent · [JarvisPanda](https://github.com/blackpandachan/JarvisPanda)*"""

    # Post on the original issue
    post_comment(issue_number, guide)
    # Post on the PR too
    post_pr_comment(pr_number, guide)


def notify_discord_pr(
    issue_number: int,
    issue_title: str,
    pr_number: int,
    pr_url: str,
    branch: str,
    auto_merge: bool,
) -> None:
    if not DISCORD_WEBHOOK:
        return
    merge_note = (
        f"⏱️ Auto-merging in {MERGE_DELAY // 60} minutes unless declined."
        if auto_merge
        else f"React **👍** on the issue comment or run `!merge {pr_number}` to merge."
    )
    payload = {
        "embeds": [{
            "title": f"🤖 PR Ready — Issue #{issue_number}",
            "description": (
                f"**{issue_title}**\n\n"
                f"ClaudeCodeAgent has implemented this issue and opened a PR.\n\n"
                f"**PR:** [{pr_url}]({pr_url})\n"
                f"**Branch:** `{branch}`\n\n"
                f"{merge_note}\n\n"
                f"Implementation guide posted on the issue and PR."
            ),
            "color": 0x10B981,
            "footer": {"text": f"JarvisPanda ClaudeCodeAgent · {MERGE_STRATEGY} merge strategy"},
        }]
    }
    try:
        httpx.post(DISCORD_WEBHOOK, json=payload, timeout=10)
    except Exception as exc:
        log.warning("Discord notification failed: %s", exc)


def notify_discord_merged(pr_number: int, pr_url: str) -> None:
    if not DISCORD_WEBHOOK:
        return
    try:
        httpx.post(
            DISCORD_WEBHOOK,
            json={"content": f"✅ PR #{pr_number} auto-merged: {pr_url}"},
            timeout=10,
        )
    except Exception:
        pass

# ── ClaudeCodeAgent runner ────────────────────────────────────────────────────

def build_task_prompt(issue: dict) -> str:
    number = issue["number"]
    title  = issue["title"]
    body   = issue.get("body") or "(no description)"
    url    = issue.get("html_url", "")
    author = issue.get("user", {}).get("login", "unknown")

    return f"""You are working on GitHub Issue #{number} from the JarvisPanda repository.

## Issue Details
**Title:** {title}
**URL:** {url}
**Reported by:** {author}

## Description
{body}

## Your Task
1. Understand the issue fully — read the description and any referenced files.
2. Explore the relevant code in the workspace at {WORKSPACE}.
3. Implement a complete solution (code, tests, config — whatever is needed).
4. Verify your changes (run tests if available).
5. Commit your changes with a clear message referencing issue #{number}.
6. Write a concise structured summary of:
   - What you changed and why
   - Files modified
   - How to verify the fix
   - Any caveats or follow-up work needed

## Rules
- Work only within {WORKSPACE}.
- Make commits — they will be pushed and a PR opened automatically.
- Do NOT push yourself.
- If the issue is ambiguous or needs human judgment, explain clearly and stop.
- Be thorough in your summary — local agents will use it to verify and merge.
"""


def run_claude_code_agent(task_prompt: str, issue_number: int) -> tuple[bool, str]:
    if not ANTHROPIC_API_KEY:
        return False, "ANTHROPIC_API_KEY is not set."

    log.info("Running ClaudeCodeAgent for issue #%d...", issue_number)
    try:
        result = subprocess.run(
            ["jarvis", "ask", "--agent", "claude_code", "--no-context", task_prompt],
            capture_output=True, text=True, cwd=WORKSPACE, timeout=1800,
            env={**os.environ, "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY},
        )
        output  = ((result.stdout or "") + (result.stderr or "")).strip()
        success = result.returncode == 0
        if not success:
            log.warning("ClaudeCodeAgent exited %d for #%d", result.returncode, issue_number)
        return success, output
    except subprocess.TimeoutExpired:
        log.error("ClaudeCodeAgent timed out on issue #%d", issue_number)
        return False, "ClaudeCodeAgent exceeded the 30-minute time limit."
    except Exception as exc:
        log.exception("Unexpected error running ClaudeCodeAgent for #%d", issue_number)
        return False, f"Unexpected error: {exc}"


def format_failure_comment(output: str, elapsed: float) -> str:
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    if len(output) > 10_000:
        output = output[:10_000] + "\n\n_(output truncated)_"
    return (
        f"❌ **ClaudeCodeAgent failed** — {minutes}m {seconds}s\n\n"
        f"<details><summary>Agent output</summary>\n\n```\n{output}\n```\n\n</details>\n\n"
        f"---\n"
        f"**Next steps for local agents:**\n"
        f"1. Review the error above.\n"
        f"2. Use `gemini_escalate` with the full error context to get a fix plan.\n"
        f"3. If Gemini can solve it, implement the fix and push to a branch manually.\n"
        f"4. If not, cut a new issue with the combined error + Gemini analysis.\n\n"
        f"*[JarvisPanda](https://github.com/blackpandachan/JarvisPanda) ClaudeCodeAgent*"
    )

# ── Main processing ───────────────────────────────────────────────────────────

def process_issue(issue: dict) -> None:
    number = issue["number"]
    title  = issue["title"]
    log.info("Processing issue #%d: %s (strategy=%s)", number, title, MERGE_STRATEGY)

    set_labels(number, add=[LABEL_WIP], remove=[LABEL_WORK])

    # comment-only legacy mode
    if MERGE_STRATEGY == "comment":
        start          = time.monotonic()
        success, output = run_claude_code_agent(build_task_prompt(issue), number)
        elapsed        = time.monotonic() - start
        post_comment(number, format_failure_comment(output, elapsed) if not success else
                     f"✅ **ClaudeCodeAgent completed** — {int(elapsed//60)}m {int(elapsed%60)}s\n\n{output[:5000]}")
        set_labels(number, add=[LABEL_DONE if success else LABEL_FAILED], remove=[LABEL_WIP])
        return

    # PR-based strategies (manual / auto)
    branch: str | None = None
    try:
        branch = prepare_branch(number, title)
    except Exception as exc:
        log.error("Branch preparation failed: %s", exc)

    start          = time.monotonic()
    success, output = run_claude_code_agent(build_task_prompt(issue), number)
    elapsed        = time.monotonic() - start

    if not success:
        post_comment(number, format_failure_comment(output, elapsed))
        set_labels(number, add=[LABEL_FAILED], remove=[LABEL_WIP])
        log.warning("Issue #%d failed (%.0fs)", number, elapsed)
        return

    # Push branch
    if branch and not push_branch(branch):
        post_comment(number,
            "⚠️ ClaudeCodeAgent completed but `git push` failed. "
            "Check worker logs. The implementation may exist as uncommitted local changes."
        )
        set_labels(number, add=[LABEL_FAILED], remove=[LABEL_WIP])
        return

    # Create PR
    pr_number, pr_url = create_pr(number, title, branch, output)
    if not pr_number:
        post_comment(number,
            "⚠️ Agent completed and pushed the branch but PR creation failed. "
            f"Branch: `{branch}`. Please open a PR manually."
        )
        set_labels(number, add=[LABEL_FAILED], remove=[LABEL_WIP])
        return

    # Post implementation guide to both issue and PR
    post_implementation_guide(number, pr_number, pr_url, branch, output)

    # Discord notification
    notify_discord_pr(number, title, pr_number, pr_url, branch, MERGE_STRATEGY == "auto")

    set_labels(number, add=[LABEL_DONE], remove=[LABEL_WIP])
    log.info("Issue #%d → PR #%d created (%.0fs)", number, pr_number, elapsed)

    # Auto-merge
    if MERGE_STRATEGY == "auto":
        log.info("Auto-merge: waiting %dm before merging PR #%d...", MERGE_DELAY // 60, pr_number)
        time.sleep(MERGE_DELAY)
        state = get_pr_state(pr_number)
        if state == "open":
            ok, msg = merge_pr(pr_number)
            if ok:
                notify_discord_merged(pr_number, pr_url)
                log.info("PR #%d auto-merged: %s", pr_number, msg)
            else:
                log.warning("PR #%d auto-merge failed: %s", pr_number, msg)
                notify_discord_merged_failed(pr_number, pr_url, msg)
        else:
            log.info("PR #%d is %s — skipping auto-merge", pr_number, state)


def notify_discord_merged_failed(pr_number: int, pr_url: str, reason: str) -> None:
    if not DISCORD_WEBHOOK:
        return
    try:
        httpx.post(DISCORD_WEBHOOK, json={
            "content": (
                f"⚠️ PR #{pr_number} auto-merge failed: {reason}\n"
                f"Merge manually: `!merge {pr_number}` or {pr_url}"
            )
        }, timeout=10)
    except Exception:
        pass

# ── Entry point ───────────────────────────────────────────────────────────────

def validate_config() -> bool:
    ok = True
    if not GITHUB_TOKEN:
        log.error("GITHUB_TOKEN is not set")
        ok = False
    if not GITHUB_REPO:
        log.error("GITHUB_REPO is not set")
        ok = False
    if not ANTHROPIC_API_KEY:
        log.warning("ANTHROPIC_API_KEY not set — worker will poll but cannot run ClaudeCodeAgent.")
    if MERGE_STRATEGY not in ("manual", "auto", "comment"):
        log.error("MERGE_STRATEGY must be manual, auto, or comment (got %r)", MERGE_STRATEGY)
        ok = False
    return ok


def main() -> None:
    if not validate_config():
        sys.exit(1)

    log.info(
        "Claude Code Worker starting — repo: %s | poll: %ds | strategy: %s",
        GITHUB_REPO, POLL_INTERVAL, MERGE_STRATEGY,
    )
    ensure_labels()

    while True:
        try:
            issues = get_queued_issues()
            if issues:
                log.info("Found %d queued issue(s)", len(issues))
                for issue in issues:
                    process_issue(issue)
            else:
                log.debug("No queued issues")
        except httpx.HTTPStatusError as exc:
            log.error("GitHub API error: %s", exc)
        except Exception as exc:
            log.exception("Main loop error: %s", exc)

        log.info("Sleeping %ds...", POLL_INTERVAL)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
