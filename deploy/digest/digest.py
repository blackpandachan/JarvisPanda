"""Jarvis HN AI Digest — 72-hour window, ranked by discussion heat.

Fetches the top N AI/ML stories from Hacker News posted in the last 72 hours,
ranked by comment count (discussion heat) with score as tiebreaker.
Posts a single cohesive Discord embed and saves a local markdown report.

Environment variables:
  DISCORD_DIGEST_WEBHOOK   Webhook URL for the digest post
  DIGEST_INTERVAL_DAYS     How often to run (default: 3, supports decimals)
  DIGEST_TOP_N             Stories to include (default: 15)
  DIGEST_WINDOW_HOURS      Age cutoff in hours (default: 72)
  DIGEST_MIN_SCORE         Minimum HN score to consider (default: 10)
  REPORTS_DIR              Where to save markdown reports (default: /reports)
"""

from __future__ import annotations

import logging
import os
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

REPORTS_DIR   = Path(os.environ.get("REPORTS_DIR", "/reports"))
DISCORD_WEBHOOK = os.environ.get("DISCORD_DIGEST_WEBHOOK", "")
INTERVAL_DAYS = float(os.environ.get("DIGEST_INTERVAL_DAYS", "3"))
TOP_N         = int(os.environ.get("DIGEST_TOP_N", "15"))
WINDOW_HOURS  = int(os.environ.get("DIGEST_WINDOW_HOURS", "72"))
MIN_SCORE     = int(os.environ.get("DIGEST_MIN_SCORE", "10"))

HN_ALGOLIA  = "https://hn.algolia.com/api/v1/search"
HN_ITEM_URL = "https://news.ycombinator.com/item?id={}"

# Discord embed: description max 4096, title max 256
EMBED_DESC_MAX = 4000
EMBED_COLOR    = 0xFF6600  # HN orange

# Search terms broad enough to catch all relevant AI/ML content
AI_QUERIES = [
    "LLM language model AI",
    "artificial intelligence machine learning",
    "GPT Claude Gemini Llama model",
    "AI agent autonomous reasoning",
    "deep learning neural network",
    "OpenAI Anthropic Google DeepMind",
]


# ── Data fetching ─────────────────────────────────────────────────────────────

def _window_start() -> int:
    """Unix timestamp for WINDOW_HOURS ago."""
    return int(datetime.now(timezone.utc).timestamp()) - WINDOW_HOURS * 3600


def _fetch_query(query: str, since_ts: int, n: int = 40) -> list[dict]:
    try:
        resp = httpx.get(
            HN_ALGOLIA,
            params={
                "query": query,
                "tags": "story",
                "hitsPerPage": n,
                "numericFilters": f"created_at_i>{since_ts},points>{MIN_SCORE}",
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("hits", [])
    except Exception as exc:
        log.warning("HN fetch failed for %r: %s", query, exc)
        return []


def fetch_top_ai_stories() -> list[dict]:
    """Return the top-N AI stories from the last WINDOW_HOURS, by comment count."""
    since = _window_start()
    seen: set[str] = set()
    pool: list[dict] = []

    for query in AI_QUERIES:
        for hit in _fetch_query(query, since):
            oid = hit.get("objectID", "")
            if oid and oid not in seen:
                seen.add(oid)
                pool.append(hit)

    if not pool:
        log.warning("No stories found in the last %dh — widening window to 7 days", WINDOW_HOURS)
        since = int(datetime.now(timezone.utc).timestamp()) - 7 * 86400
        for query in AI_QUERIES[:3]:
            for hit in _fetch_query(query, since, 20):
                oid = hit.get("objectID", "")
                if oid and oid not in seen:
                    seen.add(oid)
                    pool.append(hit)

    # Primary sort: comment count (discussion heat). Secondary: score.
    pool.sort(
        key=lambda h: (h.get("num_comments", 0), h.get("points", 0)),
        reverse=True,
    )
    return pool[:TOP_N]


# ── Formatting ────────────────────────────────────────────────────────────────

def _age(ts: int) -> str:
    if not ts:
        return "?"
    delta = int(datetime.now(timezone.utc).timestamp()) - ts
    if delta < 3600:
        return f"{delta // 60}m"
    if delta < 86400:
        return f"{delta // 3600}h"
    return f"{delta // 86400}d"


def build_embed_description(stories: list[dict]) -> str:
    """Build the Discord embed description — one cohesive block, all stories."""
    lines: list[str] = []
    for i, s in enumerate(stories, 1):
        title    = s.get("title", "Untitled")
        url      = s.get("url") or HN_ITEM_URL.format(s.get("objectID", ""))
        hn_url   = HN_ITEM_URL.format(s.get("objectID", ""))
        comments = s.get("num_comments", 0)
        pts      = s.get("points", 0)
        author   = s.get("author", "?")
        age      = _age(s.get("created_at_i", 0))

        # Truncate very long titles so the embed stays readable
        display_title = title if len(title) <= 80 else title[:77] + "…"

        lines.append(
            f"**{i}. [{display_title}]({url})**\n"
            f"💬 {comments} comments · ⬆️ {pts} pts · `{author}` · {age} ago"
            f"  [[HN]]({hn_url})"
        )

    return "\n\n".join(lines)


def build_embed(stories: list[dict]) -> dict:
    """Build a single Discord embed payload."""
    now      = datetime.now(timezone.utc)
    date_str = now.strftime("%B %d, %Y")
    desc     = build_embed_description(stories)

    # Trim if somehow over the limit (shouldn't happen with 15 stories)
    if len(desc) > EMBED_DESC_MAX:
        desc = desc[:EMBED_DESC_MAX - 3] + "…"

    return {
        "embeds": [
            {
                "title": f"🤖 Jarvis AI Digest — {date_str}",
                "description": (
                    f"**Top {len(stories)} most-discussed AI & ML stories "
                    f"on Hacker News · last {WINDOW_HOURS}h**\n"
                    f"*Ranked by comment count (discussion heat)*\n\n"
                    + desc
                ),
                "color": EMBED_COLOR,
                "footer": {
                    "text": (
                        f"OpenJarvis · Next digest in {int(INTERVAL_DAYS)}d · "
                        f"{now.strftime('%H:%M UTC')}"
                    )
                },
            }
        ]
    }


def format_markdown_report(stories: list[dict]) -> str:
    """Full markdown report saved to disk."""
    now      = datetime.now(timezone.utc)
    date_str = now.strftime("%B %d, %Y")
    lines    = [
        f"# 🤖 Jarvis AI Digest — {date_str}",
        f"*Top {len(stories)} most-discussed AI stories on HN · last {WINDOW_HOURS}h*",
        f"*Generated {now.strftime('%H:%M UTC')}*",
        "",
        "---",
        "",
    ]
    for i, s in enumerate(stories, 1):
        title    = s.get("title", "Untitled")
        url      = s.get("url") or HN_ITEM_URL.format(s.get("objectID", ""))
        hn_url   = HN_ITEM_URL.format(s.get("objectID", ""))
        comments = s.get("num_comments", 0)
        pts      = s.get("points", 0)
        author   = s.get("author", "?")
        age      = _age(s.get("created_at_i", 0))
        lines += [
            f"## {i}. {title}",
            f"",
            f"💬 **{comments} comments** · ⬆️ {pts} pts · by {author} · {age} ago",
            f"",
            f"- **Article:** {url}",
            f"- **HN thread:** {hn_url}",
            f"",
        ]
    lines += [
        "---",
        f"*Powered by [OpenJarvis](https://github.com/blackpandachan/JarvisPanda) · "
        f"Source: [Hacker News](https://news.ycombinator.com)*",
    ]
    return "\n".join(lines)


# ── Output ────────────────────────────────────────────────────────────────────

def save_report(content: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path     = REPORTS_DIR / f"ai-digest-{date_str}.md"
    path.write_text(content, encoding="utf-8")
    log.info("Report saved → %s (%d bytes)", path, len(content))
    return path


def post_to_discord(stories: list[dict]) -> None:
    if not DISCORD_WEBHOOK:
        log.info("No DISCORD_DIGEST_WEBHOOK set — skipping Discord post")
        return
    payload = build_embed(stories)
    try:
        resp = httpx.post(DISCORD_WEBHOOK, json=payload, timeout=15)
        if resp.status_code in (200, 204):
            log.info("Digest posted to Discord as single embed ✓")
        else:
            log.warning("Discord webhook returned %d: %s", resp.status_code, resp.text[:300])
    except Exception as exc:
        log.error("Failed to post digest to Discord: %s", exc)


# ── Scheduler loop ────────────────────────────────────────────────────────────

def run_digest() -> None:
    log.info(
        "Digest run starting — fetching top %d AI stories from last %dh...",
        TOP_N, WINDOW_HOURS,
    )
    stories = fetch_top_ai_stories()
    if not stories:
        log.warning("No stories fetched — check network / HN API")
        return

    log.info(
        "Fetched %d stories | top by comments: %s (%d 💬)",
        len(stories),
        stories[0].get("title", "?")[:60],
        stories[0].get("num_comments", 0),
    )

    report = format_markdown_report(stories)
    save_report(report)
    post_to_discord(stories)
    log.info("Digest run complete.")


def main() -> None:
    interval_s = int(INTERVAL_DAYS * 86400)
    log.info(
        "Jarvis HN Digest starting — window: %dh | interval: %gd | top: %d",
        WINDOW_HOURS, INTERVAL_DAYS, TOP_N,
    )
    run_digest()
    while True:
        log.info("Next digest in %gd. Sleeping...", INTERVAL_DAYS)
        time.sleep(interval_s)
        run_digest()


if __name__ == "__main__":
    main()
