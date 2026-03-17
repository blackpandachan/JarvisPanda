"""Jarvis Claude Code Issue Worker.

Polls GitHub for issues labelled ``claude-code-work``, claims them, runs
the OpenJarvis ClaudeCodeAgent against each one (with the repo as workspace),
then posts the result as an issue comment and swaps labels so humans can
review the outcome.

When CLAUDE_WORK_MODE=pr (default), the worker creates a branch, lets
ClaudeCodeAgent make commits, then pushes and opens a GitHub PR linked to the
issue.  Discord is notified via DISCORD_DIGEST_WEBHOOK when the PR is ready.

Label lifecycle:
  claude-code-work        → issue is queued (created by local qwen3 agent)
  claude-code-in-progress → worker has claimed it and is running
  claude-code-done        → worker completed; awaiting human review
  claude-code-failed      → worker errored; issue stays open for manual triage

Required environment variables:
  GITHUB_TOKEN       Personal access token with repo scope
  GITHUB_REPO        owner/repo  (e.g. "open-jarvis/OpenJarvis")
  ANTHROPIC_API_KEY  For the ClaudeCodeAgent (claude-opus-4-6)

Optional:
  WORKSPACE          Path to the repo clone inside the container (default /workspace)
  POLL_INTERVAL      Seconds between GitHub polls (default 300 = 5 min)
  MAX_ISSUE_AGE_DAYS Skip issues older than this (default 14)
  WORKER_LABEL_WORK  Override the "work" label (default claude-code-work)
  CLAUDE_WORK_MODE   "pr" (default) or "comment" — pr creates a branch+PR,
                     comment just posts the agent output on the issue
  DISCORD_DIGEST_WEBHOOK  If set, posts a PR-ready notification to Discord
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

GITHUB_TOKEN: str = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO: str  = os.environ.get("GITHUB_REPO", "")
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
WORKSPACE: str    = os.environ.get("WORKSPACE", "/workspace")
POLL_INTERVAL: int = int(os.environ.get("POLL_INTERVAL", "300"))
MAX_ISSUE_AGE_DAYS: int = int(os.environ.get("MAX_ISSUE_AGE_DAYS", "14"))
WORK_MODE: str    = os.environ.get("CLAUDE_WORK_MODE", "pr")   # "pr" | "comment"
DISCORD_WEBHOOK: str = os.environ.get("DISCORD_DIGEST_WEBHOOK", "")

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

def _gh_headers() -> dict[str, str]:
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def ensure_labels() -> None:
    """Create worker lifecycle labels on the repo if they don't exist."""
    with httpx.Client(headers=_gh_headers(), timeout=15) as http:
        for name, color in LABEL_COLORS.items():
            r = http.get(f"{GITHUB_API}/repos/{GITHUB_REPO}/labels/{name}")
            if r.status_code == 200:
                continue
            if r.status_code == 404:
                create = http.post(
                    f"{GITHUB_API}/repos/{GITHUB_REPO}/labels",
                    json={"name": name, "color": color},
                )
                if create.status_code == 201:
                    log.info("Created label: %s", name)
                elif create.status_code == 403:
                    log.warning(
                        "Cannot create label %r — GitHub token needs 'repo' write scope "
                        "(got 403). Issues will still be polled but labels won't be managed.",
                        name,
                    )
                else:
                    log.warning(
                        "Unexpected status %d creating label %r: %s",
                        create.status_code, name, create.text[:200],
                    )


def get_queued_issues() -> list[dict]:
    """Return open issues labelled LABEL_WORK but NOT LABEL_WIP."""
    with httpx.Client(headers=_gh_headers(), timeout=15) as http:
        r = http.get(
            f"{GITHUB_API}/repos/{GITHUB_REPO}/issues",
            params={"labels": LABEL_WORK, "state": "open", "per_page": 20},
        )
        r.raise_for_status()
        issues = r.json()

    cutoff_ts = int(datetime.now(timezone.utc).timestamp()) - MAX_ISSUE_AGE_DAYS * 86400
    result = []
    for issue in issues:
        labels = {lbl["name"] for lbl in issue.get("labels", [])}
        if LABEL_WIP in labels:
            continue
        created = issue.get("created_at", "")
        try:
            created_ts = int(
                datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
            )
        except Exception:
            created_ts = 0
        if created_ts and created_ts < cutoff_ts:
            log.info("Skipping stale issue #%d", issue["number"])
            continue
        result.append(issue)
    return result


def set_labels(issue_number: int, add: list[str], remove: list[str]) -> None:
    """Add and remove labels on an issue atomically."""
    with httpx.Client(headers=_gh_headers(), timeout=15) as http:
        r = http.get(f"{GITHUB_API}/repos/{GITHUB_REPO}/issues/{issue_number}")
        current = {lbl["name"] for lbl in r.json().get("labels", [])}
        new_labels = list((current | set(add)) - set(remove))
        http.patch(
            f"{GITHUB_API}/repos/{GITHUB_REPO}/issues/{issue_number}",
            json={"labels": new_labels},
        )


def post_comment(issue_number: int, body: str) -> None:
    with httpx.Client(headers=_gh_headers(), timeout=15) as http:
        http.post(
            f"{GITHUB_API}/repos/{GITHUB_REPO}/issues/{issue_number}/comments",
            json={"body": body},
        )


# ── Git / PR helpers ──────────────────────────────────────────────────────────

def _git(args: list[str], cwd: str = WORKSPACE, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check,
    )


def _slugify(text: str) -> str:
    """Convert text to a safe branch-name slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:50]


def prepare_branch(issue_number: int, title: str) -> str:
    """Create and checkout a fresh branch for the issue. Returns branch name."""
    branch = f"claude-code/issue-{issue_number}-{_slugify(title)}"
    # Ensure we start from the latest main
    try:
        _git(["fetch", "origin"])
        _git(["checkout", "-B", branch, "origin/main"])
    except subprocess.CalledProcessError:
        # Fallback: branch from current HEAD
        _git(["checkout", "-B", branch])
    log.info("Checked out branch: %s", branch)
    return branch


def push_branch(branch: str) -> bool:
    """Push the branch to origin. Returns True on success."""
    try:
        _git([
            "push", "origin", branch, "--force-with-lease",
            "--set-upstream",
        ])
        return True
    except subprocess.CalledProcessError as exc:
        log.error("git push failed: %s", exc.stderr)
        return False


def create_pr(issue_number: int, title: str, branch: str, agent_summary: str) -> str | None:
    """Create a GitHub PR for the branch. Returns PR URL or None."""
    body = (
        f"Closes #{issue_number}\n\n"
        f"## Agent Summary\n\n"
        f"{agent_summary[:3000]}\n\n"
        f"---\n"
        f"*Opened automatically by the [OpenJarvis](https://github.com/blackpandachan/JarvisPanda) "
        f"ClaudeCodeAgent worker (`claude-opus-4-6`). Please review before merging.*"
    )
    with httpx.Client(headers=_gh_headers(), timeout=15) as http:
        r = http.post(
            f"{GITHUB_API}/repos/{GITHUB_REPO}/pulls",
            json={
                "title": f"[Claude] {title}",
                "body": body,
                "head": branch,
                "base": "main",
            },
        )
    if r.status_code == 201:
        url = r.json().get("html_url", "")
        log.info("PR created: %s", url)
        return url
    log.error("PR creation failed (%d): %s", r.status_code, r.text[:300])
    return None


def notify_discord(issue_number: int, issue_title: str, pr_url: str) -> None:
    """Post a PR-ready notification to Discord."""
    if not DISCORD_WEBHOOK:
        return
    payload = {
        "embeds": [{
            "title": f"🤖 PR Ready for Review",
            "description": (
                f"**Issue #{issue_number}:** {issue_title}\n\n"
                f"ClaudeCodeAgent has finished working this issue and opened a PR.\n"
                f"[View Pull Request]({pr_url})"
            ),
            "color": 0x10B981,  # green
            "footer": {"text": "OpenJarvis ClaudeCodeAgent Worker"},
        }]
    }
    try:
        httpx.post(DISCORD_WEBHOOK, json=payload, timeout=10)
    except Exception as exc:
        log.warning("Discord notification failed: %s", exc)


# ── Claude Code runner ────────────────────────────────────────────────────────

def build_task_prompt(issue: dict) -> str:
    """Convert a GitHub issue into a detailed ClaudeCodeAgent task prompt."""
    number = issue["number"]
    title  = issue["title"]
    body   = issue.get("body") or "(no description provided)"
    url    = issue.get("html_url", "")
    author = issue.get("user", {}).get("login", "unknown")

    return f"""You are working on GitHub Issue #{number} from the OpenJarvis repository.

## Issue Details
**Title:** {title}
**URL:** {url}
**Reported by:** {author}

## Description
{body}

## Your Task
1. Understand the issue fully.
2. Explore the relevant code in the workspace at {WORKSPACE}.
3. Implement a solution (write code, tests, config changes — whatever is needed).
4. Verify your changes make sense (run tests if available).
5. Commit your changes with a descriptive message referencing issue #{number}.
6. Summarise everything you did in a clear, structured response.

## Important Rules
- Work only within the repo at {WORKSPACE}.
- Commit your changes — they will be pushed to a branch and a PR will be opened.
- Do not push to GitHub yourself — the worker handles that.
- If the issue is unclear or requires human judgment, explain why and stop.
- Be thorough but concise in your final summary.
"""


def run_claude_code_agent(task_prompt: str, issue_number: int) -> tuple[bool, str]:
    """Run the ClaudeCodeAgent via the jarvis CLI. Returns (success, output)."""
    if not ANTHROPIC_API_KEY:
        return False, "ANTHROPIC_API_KEY is not set — ClaudeCodeAgent cannot run."

    cmd = [
        "jarvis", "ask",
        "--agent", "claude_code",
        "--no-context",
        task_prompt,
    ]

    log.info("Running ClaudeCodeAgent for issue #%d...", issue_number)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=WORKSPACE,
            timeout=1800,  # 30-min hard limit per issue
            env={**os.environ, "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY},
        )
        output  = (result.stdout or "") + (result.stderr or "")
        success = result.returncode == 0
        if not success:
            log.warning(
                "ClaudeCodeAgent exited %d for issue #%d",
                result.returncode,
                issue_number,
            )
        return success, output.strip()
    except subprocess.TimeoutExpired:
        log.error("ClaudeCodeAgent timed out on issue #%d", issue_number)
        return False, "ClaudeCodeAgent exceeded the 30-minute time limit."
    except Exception as exc:
        log.exception("Unexpected error running ClaudeCodeAgent for issue #%d", issue_number)
        return False, f"Unexpected error: {exc}"


def format_result_comment(
    success: bool,
    output: str,
    elapsed: float,
    pr_url: str | None = None,
) -> str:
    status_emoji = "✅" if success else "❌"
    status_text  = "completed successfully" if success else "encountered an error"
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    max_output = 60_000
    if len(output) > max_output:
        output = output[:max_output] + f"\n\n*(output truncated at {max_output} chars)*"

    pr_line = f"\n\n**Pull Request:** {pr_url}" if pr_url else ""

    return f"""{status_emoji} **ClaudeCodeAgent {status_text}** — {minutes}m {seconds}s{pr_line}

<details>
<summary>Full agent output (click to expand)</summary>

```
{output}
```

</details>

---
*This issue was worked autonomously by the [OpenJarvis](https://github.com/blackpandachan/JarvisPanda) \
ClaudeCodeAgent worker using `claude-opus-4-6`. \
Please review the changes before merging.*
"""


# ── Main loop ─────────────────────────────────────────────────────────────────

def process_issue(issue: dict) -> None:
    number = issue["number"]
    title  = issue["title"]
    log.info("Processing issue #%d: %s (mode=%s)", number, title, WORK_MODE)

    # Claim: swap to in-progress
    set_labels(number, add=[LABEL_WIP], remove=[LABEL_WORK])

    branch: str | None = None
    pr_url: str | None = None

    # PR mode: create a branch before running the agent
    if WORK_MODE == "pr":
        try:
            branch = prepare_branch(number, title)
        except Exception as exc:
            log.error("Branch preparation failed: %s", exc)
            branch = None

    start          = time.monotonic()
    task_prompt    = build_task_prompt(issue)
    success, output = run_claude_code_agent(task_prompt, number)
    elapsed        = time.monotonic() - start

    # PR mode: push branch and open PR
    if WORK_MODE == "pr" and branch and success:
        pushed = push_branch(branch)
        if pushed:
            pr_url = create_pr(number, title, branch, output)
            if pr_url:
                notify_discord(number, title, pr_url)

    comment = format_result_comment(success, output, elapsed, pr_url)
    post_comment(number, comment)

    if success:
        set_labels(number, add=[LABEL_DONE], remove=[LABEL_WIP])
        log.info("Issue #%d complete (%.0fs)%s", number, elapsed, f" — PR: {pr_url}" if pr_url else "")
    else:
        set_labels(number, add=[LABEL_FAILED], remove=[LABEL_WIP])
        log.warning("Issue #%d failed (%.0fs)", number, elapsed)


def validate_config() -> bool:
    ok = True
    if not GITHUB_TOKEN:
        log.error("GITHUB_TOKEN is not set")
        ok = False
    if not GITHUB_REPO:
        log.error("GITHUB_REPO is not set (expected: owner/repo)")
        ok = False
    if not ANTHROPIC_API_KEY:
        log.warning(
            "ANTHROPIC_API_KEY is not set — worker will start but cannot run ClaudeCodeAgent. "
            "Set it when you have quota."
        )
    return ok


def main() -> None:
    if not validate_config():
        log.error("Required config missing — exiting.")
        sys.exit(1)

    log.info(
        "Claude Code Issue Worker starting — repo: %s | poll: %ds | workspace: %s | mode: %s",
        GITHUB_REPO,
        POLL_INTERVAL,
        WORKSPACE,
        WORK_MODE,
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
                log.debug("No queued issues found")
        except httpx.HTTPStatusError as exc:
            log.error("GitHub API error: %s", exc)
        except Exception as exc:
            log.exception("Unexpected error in main loop: %s", exc)

        log.info("Sleeping %ds until next poll...", POLL_INTERVAL)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
