#!/usr/bin/env python3
"""
Gemini Flash implementation script for JarvisPanda GitHub Actions.

Reads a GitHub issue from env vars, sends relevant codebase context to
Gemini Flash, parses its JSON response, applies file changes, and writes
commit metadata to /tmp/ for the workflow to use.

Required env vars:
  GEMINI_API_KEY    Gemini API key
  ISSUE_TITLE       Issue title
  ISSUE_BODY        Issue body
  ISSUE_NUMBER      Issue number
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ISSUE_TITLE    = os.environ.get("ISSUE_TITLE", "")
ISSUE_BODY     = os.environ.get("ISSUE_BODY", "")
ISSUE_NUMBER   = os.environ.get("ISSUE_NUMBER", "0")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

MAX_FILE_SIZE   = 40_000   # bytes — skip files larger than this
MAX_FILES       = 50       # max source files to include
MAX_CONTEXT     = 120_000  # max chars of code context sent to Gemini


def gather_context() -> str:
    """Collect CLAUDE.md + relevant source files as a single context string.

    Search order: issue keywords → deploy/ → src/ so the most relevant files
    (e.g. the Discord bot for Discord issues) are included first.
    """
    parts: list[str] = []

    claude_md = Path("CLAUDE.md")
    if claude_md.exists():
        parts.append(f"=== CLAUDE.md (project guide) ===\n{claude_md.read_text()[:4000]}")

    # Keyword-based priority: if issue mentions discord/bot, include bot.py first
    issue_text = (ISSUE_TITLE + " " + ISSUE_BODY).lower()
    priority_files: list[Path] = []

    if any(k in issue_text for k in ("discord", "bot", "command", "slash", "!version", "!status", "!help")):
        bot_py = Path("deploy/discord/bot.py")
        if bot_py.exists():
            priority_files.append(bot_py)

    if any(k in issue_text for k in ("worker", "observer", "docker", "deploy")):
        for p in Path("deploy").rglob("*.py"):
            if p not in priority_files:
                priority_files.append(p)

    # Add priority files first (full content up to size limit)
    for path in priority_files:
        if path.stat().st_size > MAX_FILE_SIZE * 3:  # allow larger for key files
            content = path.read_text()[:MAX_FILE_SIZE * 3]
        else:
            content = path.read_text()
        parts.append(f"=== {path} ===\n{content}")

    # Then fill with src/ files, smallest first
    py_files = sorted(Path("src").rglob("*.py"), key=lambda p: p.stat().st_size)
    included = len(priority_files)
    for path in py_files:
        if included >= MAX_FILES:
            break
        if path in priority_files:
            continue
        if path.stat().st_size > MAX_FILE_SIZE:
            continue
        try:
            parts.append(f"=== {path} ===\n{path.read_text()}")
            included += 1
        except Exception:
            pass

    context = "\n\n".join(parts)
    return context[:MAX_CONTEXT]


def call_gemini(prompt: str) -> str:
    """Call Gemini Flash and return the text response."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 8192,
        },
    }
    resp = httpx.post(
        GEMINI_URL,
        params={"key": GEMINI_API_KEY},
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def extract_json(text: str) -> dict:
    """Extract JSON from Gemini's response (handles markdown code blocks)."""
    text = text.strip()
    for marker in ("```json", "```"):
        if marker in text:
            text = text.split(marker, 1)[1]
            text = text.rsplit("```", 1)[0]
            break
    return json.loads(text.strip())


def main() -> None:
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    print(f"Gathering codebase context for issue #{ISSUE_NUMBER}: {ISSUE_TITLE}")
    context = gather_context()
    print(f"Context size: {len(context):,} chars")

    prompt = f"""You are a senior engineer working on the JarvisPanda / OpenJarvis codebase.

## Project Context
{context}

## GitHub Issue #{ISSUE_NUMBER}
**Title:** {ISSUE_TITLE}

**Body:**
{ISSUE_BODY}

## Your Task
Implement a complete fix for this issue.

Respond with ONLY a valid JSON object in this exact format — no prose, no markdown outside the JSON:

{{
  "summary": "1-3 sentence description of what you changed and why",
  "files": {{
    "relative/path/to/file.py": "COMPLETE new content of this file"
  }},
  "new_files": {{
    "relative/path/to/new_file.py": "content for new file (omit if no new files needed)"
  }},
  "commit_message": "fix: short description (fixes #{ISSUE_NUMBER})"
}}

Rules:
- Provide COMPLETE file contents (not diffs) for every file you modify
- Keep changes minimal and focused on the issue
- Follow existing code style and conventions from CLAUDE.md
- Commit message must follow Conventional Commits (fix:, feat:, etc.)
- If the issue is unclear or unfixable, set files to {{}} and explain in summary
"""

    print("Calling Gemini Flash...")
    raw = call_gemini(prompt)
    print(f"Response length: {len(raw):,} chars")

    try:
        result = extract_json(raw)
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        print(f"ERROR: Failed to parse Gemini JSON response: {exc}", file=sys.stderr)
        print(f"Raw response (first 2000 chars):\n{raw[:2000]}", file=sys.stderr)
        sys.exit(1)

    # Apply modifications
    changed: list[str] = []

    for rel_path, content in result.get("files", {}).items():
        p = Path(rel_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        changed.append(rel_path)
        print(f"  modified: {rel_path}")

    for rel_path, content in result.get("new_files", {}).items():
        p = Path(rel_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        changed.append(rel_path)
        print(f"  created:  {rel_path}")

    if not changed:
        print("ERROR: Gemini made no file changes.", file=sys.stderr)
        print(f"Summary: {result.get('summary', 'n/a')}", file=sys.stderr)
        sys.exit(1)

    commit_msg = result.get("commit_message", f"fix: address issue #{ISSUE_NUMBER}")
    summary    = result.get("summary", "No summary provided.")

    Path("/tmp/commit_message.txt").write_text(commit_msg)
    Path("/tmp/summary.txt").write_text(summary)

    print(f"\nCommit: {commit_msg}")
    print(f"Summary: {summary}")
    print(f"Changed {len(changed)} file(s): {', '.join(changed)}")


if __name__ == "__main__":
    main()
