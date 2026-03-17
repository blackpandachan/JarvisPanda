"""Jarvis HN AI Digest — scheduled scraper and report generator.

Runs every DIGEST_INTERVAL_DAYS days (default: 3).
Fetches the top 15 AI-related Hacker News stories via the Algolia HN API,
formats a rich markdown report, saves it to /reports/, and optionally posts
a summary to Discord via webhook.
"""

from __future__ import annotations

import logging
import os
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("jarvis-digest")

# ── Config ────────────────────────────────────────────────────────────────────

REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", "/reports"))
DISCORD_WEBHOOK = os.environ.get("DISCORD_DIGEST_WEBHOOK", "")
INTERVAL_DAYS = float(os.environ.get("DIGEST_INTERVAL_DAYS", "3"))
TOP_N = int(os.environ.get("DIGEST_TOP_N", "15"))
MIN_SCORE = int(os.environ.get("DIGEST_MIN_SCORE", "30"))

HN_ALGOLIA = "https://hn.algolia.com/api/v1/search"
HN_ITEM_URL = "https://news.ycombinator.com/item?id={}"
DISCORD_MAX = 1990

# Search terms that reliably surface AI/ML content
AI_QUERIES = [
    "LLM language model",
    "artificial intelligence machine learning",
    "GPT Claude Gemini model",
    "AI agent autonomous",
    "deep learning neural network",
]


# ── Data fetching ─────────────────────────────────────────────────────────────

def _fetch_stories_for_query(query: str, n: int = 30) -> list[dict]:
    try:
        resp = httpx.get(
            HN_ALGOLIA,
            params={
                "query": query,
                "tags": "story",
                "hitsPerPage": n,
                "numericFilters": f"points>{MIN_SCORE}",
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("hits", [])
    except Exception as exc:
        log.warning("Failed to fetch HN stories for query %r: %s", query, exc)
        return []


def fetch_top_ai_stories(n: int = TOP_N) -> list[dict]:
    """Return the top-N AI-related HN stories by score, deduplicated."""
    seen: set[str] = set()
    all_hits: list[dict] = []

    for query in AI_QUERIES:
        for hit in _fetch_stories_for_query(query, 30):
            oid = hit.get("objectID", "")
            if oid and oid not in seen:
                seen.add(oid)
                all_hits.append(hit)

    # Sort by score descending, then recency as tiebreaker
    all_hits.sort(key=lambda h: (h.get("points", 0), h.get("created_at_i", 0)), reverse=True)
    return all_hits[:n]


# ── Report formatting ─────────────────────────────────────────────────────────

def _age_label(created_at_i: int) -> str:
    """Return a human-readable age string (e.g. '2 days ago')."""
    if not created_at_i:
        return "unknown"
    now = int(datetime.now(timezone.utc).timestamp())
    delta = now - created_at_i
    if delta < 3600:
        return f"{delta // 60}m ago"
    if delta < 86400:
        return f"{delta // 3600}h ago"
    return f"{delta // 86400}d ago"


def format_report(stories: list[dict]) -> str:
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%B %d, %Y")
    lines = [
        f"# 🤖 Jarvis AI Digest — {date_str}",
        f"*Top {len(stories)} AI & ML stories from Hacker News*",
        f"*Generated at {now.strftime('%H:%M UTC')} · Next digest in {int(INTERVAL_DAYS)} days*",
        "",
        "---",
        "",
    ]

    for i, story in enumerate(stories, 1):
        title = story.get("title", "Untitled")
        url = story.get("url") or HN_ITEM_URL.format(story.get("objectID", ""))
        hn_url = HN_ITEM_URL.format(story.get("objectID", ""))
        points = story.get("points", 0)
        comments = story.get("num_comments", 0)
        author = story.get("author", "unknown")
        age = _age_label(story.get("created_at_i", 0))

        lines += [
            f"## {i}. {title}",
            f"",
            f"| | |",
            f"|---|---|",
            f"| **Score** | {points} pts |",
            f"| **Comments** | [{comments} comments]({hn_url}) |",
            f"| **Author** | {author} |",
            f"| **Posted** | {age} |",
            f"| **Link** | [{url[:60]}{'...' if len(url) > 60 else ''}]({url}) |",
            f"",
        ]

        # If the story has a snippet, include it
        snippet = story.get("story_text") or story.get("comment_text") or ""
        if snippet:
            trimmed = textwrap.shorten(snippet, width=300, placeholder="…")
            lines += [f"> {trimmed}", ""]

    lines += [
        "---",
        "",
        f"*Powered by [OpenJarvis](https://github.com/open-jarvis/OpenJarvis) · "
        f"Source: [Hacker News](https://news.ycombinator.com)*",
    ]
    return "\n".join(lines)


def format_discord_summary(stories: list[dict]) -> str:
    """Compact Discord-friendly version of the digest."""
    now = datetime.now(timezone.utc)
    lines = [
        f"**🤖 Jarvis AI Digest — {now.strftime('%B %d, %Y')}**",
        f"*Top {len(stories)} AI stories from Hacker News*",
        "",
    ]
    for i, story in enumerate(stories, 1):
        title = story.get("title", "Untitled")
        url = story.get("url") or HN_ITEM_URL.format(story.get("objectID", ""))
        points = story.get("points", 0)
        comments = story.get("num_comments", 0)
        hn_url = HN_ITEM_URL.format(story.get("objectID", ""))
        lines.append(
            f"**{i}.** [{title}]({url}) — {points}pts · [{comments} comments]({hn_url})"
        )
    return "\n".join(lines)


# ── Output ────────────────────────────────────────────────────────────────────

def save_report(content: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = REPORTS_DIR / f"ai-digest-{date_str}.md"
    path.write_text(content, encoding="utf-8")
    log.info("Report saved to %s", path)
    return path


def post_to_discord(summary: str) -> None:
    if not DISCORD_WEBHOOK:
        log.info("No DISCORD_DIGEST_WEBHOOK set — skipping Discord post")
        return

    # Split into chunks respecting Discord's limit
    chunks: list[str] = []
    remaining = summary
    while remaining:
        if len(remaining) <= DISCORD_MAX:
            chunks.append(remaining)
            break
        # Find last newline before the limit
        cut = remaining.rfind("\n", 0, DISCORD_MAX)
        if cut == -1:
            cut = DISCORD_MAX
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")

    try:
        with httpx.Client(timeout=15) as http:
            for chunk in chunks:
                resp = http.post(DISCORD_WEBHOOK, json={"content": chunk})
                if resp.status_code not in (200, 204):
                    log.warning("Discord webhook returned %d: %s", resp.status_code, resp.text[:200])
                time.sleep(0.5)  # Rate-limit friendly
        log.info("Digest posted to Discord (%d chunks)", len(chunks))
    except Exception as exc:
        log.error("Failed to post digest to Discord: %s", exc)


# ── Scheduler loop ────────────────────────────────────────────────────────────

def run_digest() -> None:
    log.info("Starting digest run — fetching top %d AI stories from HN...", TOP_N)
    stories = fetch_top_ai_stories(TOP_N)
    if not stories:
        log.warning("No stories fetched — check network connectivity")
        return

    log.info("Fetched %d stories, formatting report...", len(stories))
    report = format_report(stories)
    path = save_report(report)
    log.info("Report written to %s (%d bytes)", path, len(report))

    summary = format_discord_summary(stories)
    post_to_discord(summary)
    log.info("Digest run complete.")


def main() -> None:
    interval_seconds = int(INTERVAL_DAYS * 86400)
    log.info(
        "Jarvis HN Digest scheduler starting — interval: every %g days (%d seconds)",
        INTERVAL_DAYS,
        interval_seconds,
    )

    # Run immediately on startup so we don't wait days for first report
    run_digest()

    while True:
        log.info("Next digest in %g days. Sleeping...", INTERVAL_DAYS)
        time.sleep(interval_seconds)
        run_digest()


if __name__ == "__main__":
    main()
