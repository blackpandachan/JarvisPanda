"""Jarvis Claude Code Issue Worker.

Polls GitHub for issues labelled ``claude-code-work``, claims them, runs
the OpenJarvis ClaudeCodeAgent against each one (with the repo as workspace),
then posts the result as an issue comment and swaps labels so humans can
review the outcome.

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
"""

from __future__ import annotations

import logging
import os
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
GITHUB_REPO: str = os.environ.get("GITHUB_REPO", "")
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
WORKSPACE: str = os.environ.get("WORKSPACE", "/workspace")
POLL_INTERVAL: int = int(os.environ.get("POLL_INTERVAL", "300"))
MAX_ISSUE_AGE_DAYS: int = int(os.environ.get("MAX_ISSUE_AGE_DAYS", "14"))

LABEL_WORK = os.environ.get("WORKER_LABEL_WORK", "claude-code-work")
LABEL_WIP = "claude-code-in-progress"
LABEL_DONE = "claude-code-done"
LABEL_FAILED = "claude-code-failed"

GITHUB_API = "https://api.github.com"
LABEL_COLORS = {
    LABEL_WORK: "e11d48",
    LABEL_WIP: "f59e0b",
    LABEL_DONE: "10b981",
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
            if r.status_code == 404:
                http.post(
                    f"{GITHUB_API}/repos/{GITHUB_REPO}/labels",
                    json={"name": name, "color": color},
                )
                log.info("Created label: %s", name)


def get_queued_issues() -> list[dict]:
    """Return open issues labelled LABEL_WORK but NOT LABEL_WIP."""
    with httpx.Client(headers=_gh_headers(), timeout=15) as http:
        r = http.get(
            f"{GITHUB_API}/repos/{GITHUB_REPO}/issues",
            params={"labels": LABEL_WORK, "state": "open", "per_page": 20},
        )
        r.raise_for_status()
        issues = r.json()

    # Filter out already-claimed and stale issues
    cutoff_ts = int(datetime.now(timezone.utc).timestamp()) - MAX_ISSUE_AGE_DAYS * 86400
    result = []
    for issue in issues:
        labels = {lbl["name"] for lbl in issue.get("labels", [])}
        if LABEL_WIP in labels:
            continue  # already being worked
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
        # Get current labels
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


# ── Claude Code runner ────────────────────────────────────────────────────────

def build_task_prompt(issue: dict) -> str:
    """Convert a GitHub issue into a detailed ClaudeCodeAgent task prompt."""
    number = issue["number"]
    title = issue["title"]
    body = issue.get("body") or "(no description provided)"
    url = issue.get("html_url", "")
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
5. Summarise everything you did in a clear, structured response.

## Important Rules
- Work only within the repo at {WORKSPACE}.
- Do not push to GitHub — the human reviewer will handle that.
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
        output = (result.stdout or "") + (result.stderr or "")
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


def format_result_comment(success: bool, output: str, elapsed: float) -> str:
    status_emoji = "✅" if success else "❌"
    status_text = "completed successfully" if success else "encountered an error"
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    # Truncate very long output to avoid GitHub's comment size limits
    max_output = 60_000
    if len(output) > max_output:
        output = output[:max_output] + f"\n\n*(output truncated at {max_output} chars)*"

    return f"""{status_emoji} **ClaudeCodeAgent {status_text}** — {minutes}m {seconds}s

<details>
<summary>Full agent output (click to expand)</summary>

```
{output}
```

</details>

---
*This issue was worked autonomously by the [OpenJarvis](https://github.com/open-jarvis/OpenJarvis) \
ClaudeCodeAgent worker using `claude-opus-4-6`. \
Please review the changes before merging.*
"""


# ── Main loop ─────────────────────────────────────────────────────────────────

def process_issue(issue: dict) -> None:
    number = issue["number"]
    title = issue["title"]
    log.info("Processing issue #%d: %s", number, title)

    # Claim: swap to in-progress
    set_labels(number, add=[LABEL_WIP], remove=[LABEL_WORK])

    start = time.monotonic()
    task_prompt = build_task_prompt(issue)
    success, output = run_claude_code_agent(task_prompt, number)
    elapsed = time.monotonic() - start

    comment = format_result_comment(success, output, elapsed)
    post_comment(number, comment)

    if success:
        set_labels(number, add=[LABEL_DONE], remove=[LABEL_WIP])
        log.info("Issue #%d complete (%.0fs)", number, elapsed)
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
        "Claude Code Issue Worker starting — repo: %s | poll: %ds | workspace: %s",
        GITHUB_REPO,
        POLL_INTERVAL,
        WORKSPACE,
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
