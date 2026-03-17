"""Jarvis Observer Agent — autonomous system watchdog with local model analysis.

Runs on a configurable interval, observes the health and activity of the full
Jarvis stack, and posts actionable summaries to Discord.  When it detects
recurring failures it autonomously cuts GitHub issues for the ClaudeCodeAgent
worker to investigate.

What it observes:
  - Trace DB: queries that hit turn limits, tool errors, escalation patterns
  - Telemetry DB: inference latency, token usage, error rates
  - GitHub: PRs awaiting review, stale issues, open claude-code-work issues
  - arXiv: daily AI paper digest (posted once per day at configured hour)

What it does with findings:
  - Posts a health digest to Discord at each observation interval
  - If a tool fails 3+ times in 24h → opens a self-repair GitHub issue
  - If the turn-limit hit rate is >20% → opens an issue to improve the system prompt
  - Daily arXiv digest at OBSERVER_ARXIV_HOUR (UTC, default 8am)

Environment variables:
  DISCORD_DIGEST_WEBHOOK   Webhook URL for observer posts
  GITHUB_TOKEN             Repo read+write scope
  GITHUB_REPO              owner/repo
  OBSERVER_INTERVAL        Seconds between health checks (default 1800 = 30m)
  OBSERVER_ARXIV_HOUR      UTC hour for daily arXiv digest (default 8, -1 to disable)
  TRACE_DB_PATH            Path to traces SQLite (default /data/traces.db)
  TELEMETRY_DB_PATH        Path to telemetry SQLite (default /data/telemetry.db)
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("jarvis-observer")

# ── Config ────────────────────────────────────────────────────────────────────

DISCORD_WEBHOOK: str   = os.environ.get("DISCORD_DIGEST_WEBHOOK", "")
GITHUB_TOKEN: str      = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO: str       = os.environ.get("GITHUB_REPO", "")
OBSERVER_INTERVAL: int = int(os.environ.get("OBSERVER_INTERVAL", "1800"))
ARXIV_HOUR: int        = int(os.environ.get("OBSERVER_ARXIV_HOUR", "8"))
TRACE_DB: Path         = Path(os.environ.get("TRACE_DB_PATH", "/data/traces.db"))
TELEMETRY_DB: Path     = Path(os.environ.get("TELEMETRY_DB_PATH", "/data/telemetry.db"))

GITHUB_API = "https://api.github.com"
ARXIV_API  = "https://export.arxiv.org/api/query"
HN_API     = "https://hn.algolia.com/api/v1/search"

# Track which arXiv days we've already posted to avoid dupes across restarts
_arxiv_posted: set[str] = set()
# Track which auto-issues we've already opened (tool_name → day string)
_auto_issues_opened: set[str] = set()

# ── SQLite helpers ────────────────────────────────────────────────────────────

def _db_query(path: Path, sql: str, params: tuple = ()) -> list[dict]:
    """Run a SELECT on a SQLite DB; return list of row dicts. Never raises."""
    if not path.exists():
        return []
    try:
        conn = sqlite3.connect(str(path), timeout=5)
        conn.row_factory = sqlite3.Row
        cur  = conn.cursor()
        cur.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as exc:
        log.debug("DB query error (%s): %s", path.name, exc)
        return []


def _db_tables(path: Path) -> list[str]:
    rows = _db_query(path, "SELECT name FROM sqlite_master WHERE type='table'")
    return [r["name"] for r in rows]


def _db_columns(path: Path, table: str) -> list[str]:
    rows = _db_query(path, f"PRAGMA table_info({table})")
    return [r["name"] for r in rows]


# ── Trace analysis ────────────────────────────────────────────────────────────

def analyse_traces(window_hours: int = 24) -> dict:
    """Return observation metrics from the trace DB for the last window_hours."""
    since_ts = int(datetime.now(timezone.utc).timestamp()) - window_hours * 3600
    result   = {
        "available": False,
        "total_queries": 0,
        "hit_turn_limit": 0,
        "escalated": 0,
        "tool_errors": defaultdict(int),   # tool_name → error count
        "common_topics": [],
    }

    tables = _db_tables(TRACE_DB)
    if not tables:
        return result

    result["available"] = True

    # Find the main traces table — could be "traces", "agent_traces", "trace_steps"
    trace_table = next((t for t in tables if "trace" in t.lower()), None)
    if not trace_table:
        return result

    cols = _db_columns(TRACE_DB, trace_table)

    # Total recent queries
    ts_col = next((c for c in cols if c in ("created_at", "timestamp", "started_at")), None)
    if ts_col:
        rows = _db_query(
            TRACE_DB,
            f"SELECT COUNT(*) as cnt FROM {trace_table} WHERE {ts_col} >= ?",
            (since_ts,),
        )
        result["total_queries"] = rows[0]["cnt"] if rows else 0

    # Turn-limit hits (turns column at or above threshold)
    turns_col = next((c for c in cols if "turn" in c.lower()), None)
    max_turns = 12  # matches config.persistent.toml
    if turns_col and ts_col:
        rows = _db_query(
            TRACE_DB,
            f"SELECT COUNT(*) as cnt FROM {trace_table} "
            f"WHERE {ts_col} >= ? AND {turns_col} >= ?",
            (since_ts, max_turns),
        )
        result["hit_turn_limit"] = rows[0]["cnt"] if rows else 0

    # Escalations: look for github_issue tool calls in step data
    step_table = next((t for t in tables if "step" in t.lower()), None)
    if step_table:
        step_cols = _db_columns(TRACE_DB, step_table)
        tool_col  = next((c for c in step_cols if "tool" in c.lower()), None)
        err_col   = next((c for c in step_cols if c in ("error", "success", "failed")), None)
        ts_col2   = next((c for c in step_cols if c in ("created_at", "timestamp")), None)

        if tool_col and ts_col2:
            # Escalations
            rows = _db_query(
                TRACE_DB,
                f"SELECT COUNT(*) as cnt FROM {step_table} "
                f"WHERE {ts_col2} >= ? AND {tool_col} = 'github_issue'",
                (since_ts,),
            )
            result["escalated"] = rows[0]["cnt"] if rows else 0

            # Tool errors
            if err_col:
                err_condition = (
                    f"{err_col} = 0" if err_col == "success"
                    else f"{err_col} IS NOT NULL AND {err_col} != ''"
                )
                rows = _db_query(
                    TRACE_DB,
                    f"SELECT {tool_col}, COUNT(*) as cnt FROM {step_table} "
                    f"WHERE {ts_col2} >= ? AND {err_condition} "
                    f"GROUP BY {tool_col} ORDER BY cnt DESC LIMIT 10",
                    (since_ts,),
                )
                for r in rows:
                    result["tool_errors"][r[tool_col]] = r["cnt"]

    return result


# ── Telemetry analysis ────────────────────────────────────────────────────────

def analyse_telemetry(window_hours: int = 24) -> dict:
    """Return inference performance metrics from the telemetry DB."""
    since_ts = int(datetime.now(timezone.utc).timestamp()) - window_hours * 3600
    result   = {
        "available": False,
        "total_calls": 0,
        "avg_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "total_tokens": 0,
        "error_count": 0,
    }

    tables = _db_tables(TELEMETRY_DB)
    if not tables:
        return result
    result["available"] = True

    telem_table = next((t for t in tables if "telemetry" in t.lower() or "record" in t.lower()), None)
    if not telem_table:
        return result

    cols    = _db_columns(TELEMETRY_DB, telem_table)
    ts_col  = next((c for c in cols if c in ("created_at", "timestamp")), None)
    lat_col = next((c for c in cols if "latency" in c.lower()), None)
    tok_col = next((c for c in cols if "token" in c.lower() and "total" in c.lower()), None)
    err_col = next((c for c in cols if "error" in c.lower()), None)

    if not ts_col:
        return result

    rows = _db_query(TELEMETRY_DB, f"SELECT COUNT(*) as cnt FROM {telem_table} WHERE {ts_col} >= ?", (since_ts,))
    result["total_calls"] = rows[0]["cnt"] if rows else 0

    if lat_col and result["total_calls"] > 0:
        rows = _db_query(
            TELEMETRY_DB,
            f"SELECT AVG({lat_col}) as avg_lat, {lat_col} FROM {telem_table} "
            f"WHERE {ts_col} >= ? ORDER BY {lat_col}",
            (since_ts,),
        )
        if rows:
            result["avg_latency_ms"] = round(rows[0].get("avg_lat") or 0, 1)
            p95_idx = int(len(rows) * 0.95)
            result["p95_latency_ms"] = round(rows[p95_idx].get(lat_col) or 0, 1)

    if tok_col:
        rows = _db_query(
            TELEMETRY_DB,
            f"SELECT SUM({tok_col}) as total FROM {telem_table} WHERE {ts_col} >= ?",
            (since_ts,),
        )
        result["total_tokens"] = int(rows[0]["total"] or 0) if rows else 0

    if err_col:
        rows = _db_query(
            TELEMETRY_DB,
            f"SELECT COUNT(*) as cnt FROM {telem_table} "
            f"WHERE {ts_col} >= ? AND {err_col} IS NOT NULL AND {err_col} != ''",
            (since_ts,),
        )
        result["error_count"] = rows[0]["cnt"] if rows else 0

    return result


# ── GitHub analysis ───────────────────────────────────────────────────────────

def _gh_headers() -> dict[str, str]:
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def analyse_github() -> dict:
    """Return pending PR / issue counts from GitHub."""
    result = {
        "available": False,
        "open_prs": [],
        "work_queue": 0,
        "in_progress": 0,
        "done_unmerged": 0,
    }
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return result

    try:
        with httpx.Client(headers=_gh_headers(), timeout=15) as http:
            # Open PRs
            r = http.get(
                f"{GITHUB_API}/repos/{GITHUB_REPO}/pulls",
                params={"state": "open", "per_page": 20},
            )
            if r.status_code == 200:
                result["available"] = True
                prs = r.json()
                result["open_prs"] = [
                    {"number": p["number"], "title": p["title"][:60], "url": p["html_url"]}
                    for p in prs
                ]

            # Issue counts by label
            for label, key in [
                ("claude-code-work", "work_queue"),
                ("claude-code-in-progress", "in_progress"),
                ("claude-code-done", "done_unmerged"),
            ]:
                r2 = http.get(
                    f"{GITHUB_API}/repos/{GITHUB_REPO}/issues",
                    params={"labels": label, "state": "open", "per_page": 1},
                )
                if r2.status_code == 200:
                    result["available"] = True
                    # GitHub returns total count in Link header or we count items
                    result[key] = len(r2.json())

    except Exception as exc:
        log.debug("GitHub analysis failed: %s", exc)

    return result


# ── arXiv daily digest ────────────────────────────────────────────────────────

def fetch_arxiv_papers(query: str = "LLM AI agent reasoning", n: int = 8) -> list[dict]:
    """Return recent arXiv papers, newest first."""
    import xml.etree.ElementTree as ET
    NS = {"atom": "http://www.w3.org/2005/Atom"}

    def _t(el):
        return (el.text or "").strip() if el is not None else ""

    try:
        resp = httpx.get(
            ARXIV_API,
            params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": n,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
            timeout=20,
        )
        resp.raise_for_status()
        root    = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", NS)
        papers  = []
        for e in entries:
            title   = _t(e.find("atom:title", NS)).replace("\n", " ")
            summary = _t(e.find("atom:summary", NS)).replace("\n", " ")[:180] + "…"
            authors = [_t(a.find("atom:name", NS)) for a in e.findall("atom:author", NS)]
            link    = ""
            for lnk in e.findall("atom:link", NS):
                if lnk.get("type") == "text/html":
                    link = lnk.get("href", "")
                    break
            link = link or _t(e.find("atom:id", NS))
            published = _t(e.find("atom:published", NS))[:10]
            papers.append({"title": title, "summary": summary, "authors": authors[:2], "link": link, "published": published})
        return papers
    except Exception as exc:
        log.warning("arXiv fetch failed: %s", exc)
        return []


def build_arxiv_embed(papers: list[dict]) -> dict:
    today = datetime.now(timezone.utc).strftime("%B %d, %Y")
    lines = []
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p["authors"]) or "?"
        lines.append(
            f"**{i}. [{p['title']}]({p['link']})**\n"
            f"{authors} · {p['published']}\n"
            f"{p['summary']}"
        )
    desc = "\n\n".join(lines)
    if len(desc) > 3900:
        desc = desc[:3897] + "…"
    return {
        "embeds": [{
            "title": f"📄 arXiv AI/ML Daily — {today}",
            "description": f"**Newest AI & ML papers submitted today**\n\n{desc}",
            "color": 0xB31B1B,  # arXiv red
            "footer": {"text": "OpenJarvis Observer · arXiv.org"},
        }]
    }


# ── Auto-issue creation ───────────────────────────────────────────────────────

def open_auto_issue(title: str, body: str) -> str | None:
    """Open a self-repair GitHub issue; return URL or None."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return None
    try:
        with httpx.Client(headers=_gh_headers(), timeout=15) as http:
            r = http.post(
                f"{GITHUB_API}/repos/{GITHUB_REPO}/issues",
                json={
                    "title": title,
                    "body": body,
                    "labels": ["claude-code-work", "observer-auto"],
                },
            )
        if r.status_code == 201:
            url = r.json().get("html_url", "")
            log.info("Auto-issue opened: %s", url)
            return url
    except Exception as exc:
        log.warning("Auto-issue creation failed: %s", exc)
    return None


def maybe_open_auto_issues(trace_data: dict) -> list[str]:
    """Open GitHub issues for recurring failures. Returns list of new issue URLs."""
    today   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    new_urls: list[str] = []

    for tool_name, count in trace_data.get("tool_errors", {}).items():
        key = f"{tool_name}:{today}"
        if count >= 3 and key not in _auto_issues_opened:
            url = open_auto_issue(
                title=f"[Auto] Tool '{tool_name}' failing: {count} errors in 24h",
                body=(
                    f"## Automated Report from Jarvis Observer\n\n"
                    f"The `{tool_name}` tool has failed **{count} times** in the last 24 hours.\n\n"
                    f"### Requested Action\n"
                    f"1. Investigate the `{tool_name}` tool implementation.\n"
                    f"2. Check for dependency issues, API changes, or config problems.\n"
                    f"3. Fix the root cause and add a test if appropriate.\n"
                    f"4. Verify the fix by running the affected tool directly.\n\n"
                    f"*This issue was opened automatically by the Jarvis Observer agent.*"
                ),
            )
            if url:
                _auto_issues_opened.add(key)
                new_urls.append(url)

    # High turn-limit hit rate
    total = trace_data.get("total_queries", 0)
    stuck = trace_data.get("hit_turn_limit", 0)
    key   = f"turn_limit:{today}"
    if total >= 10 and stuck / total > 0.2 and key not in _auto_issues_opened:
        url = open_auto_issue(
            title=f"[Auto] High turn-limit hit rate: {stuck}/{total} queries ({int(100*stuck/total)}%)",
            body=(
                f"## Automated Report from Jarvis Observer\n\n"
                f"**{stuck} out of {total} queries ({int(100*stuck/total)}%) hit the agent turn limit** "
                f"in the last 24 hours.\n\n"
                f"### Requested Action\n"
                f"1. Review the system prompt in `deploy/docker/config.persistent.toml`.\n"
                f"2. Identify query patterns that are hitting the limit (check trace DB).\n"
                f"3. Either improve the system prompt, add new tools, or raise `max_turns`.\n"
                f"4. Consider adding more specific guidance for the most common stuck patterns.\n\n"
                f"*This issue was opened automatically by the Jarvis Observer agent.*"
            ),
        )
        if url:
            _auto_issues_opened.add(key)
            new_urls.append(url)

    return new_urls


# ── Discord posting ───────────────────────────────────────────────────────────

def post_discord(payload: dict) -> bool:
    if not DISCORD_WEBHOOK:
        return False
    try:
        resp = httpx.post(DISCORD_WEBHOOK, json=payload, timeout=15)
        ok   = resp.status_code in (200, 204)
        if not ok:
            log.warning("Discord post returned %d: %s", resp.status_code, resp.text[:200])
        return ok
    except Exception as exc:
        log.error("Discord post failed: %s", exc)
        return False


def build_health_embed(
    trace: dict,
    telem: dict,
    github: dict,
    auto_issues: list[str],
    window_hours: int = 24,
) -> dict:
    """Build a Discord embed summarising system health."""
    now      = datetime.now(timezone.utc)
    date_str = now.strftime("%H:%M UTC")

    # Determine overall health colour
    problems = 0
    if trace.get("hit_turn_limit", 0) > 3:
        problems += 1
    if sum(trace.get("tool_errors", {}).values()) > 5:
        problems += 1
    if telem.get("error_count", 0) > 5:
        problems += 1
    color = 0x10B981 if problems == 0 else (0xF59E0B if problems == 1 else 0xEF4444)

    lines: list[str] = []

    # Trace section
    if trace["available"]:
        total   = trace.get("total_queries", 0)
        stuck   = trace.get("hit_turn_limit", 0)
        escaped = trace.get("escalated", 0)
        stuck_pct = f"{int(100*stuck/total)}%" if total else "—"
        lines.append(
            f"**Traces (last {window_hours}h)**\n"
            f"Queries: {total} · Stuck: {stuck} ({stuck_pct}) · Escalated: {escaped}"
        )
        errs = trace.get("tool_errors", {})
        if errs:
            top_errors = sorted(errs.items(), key=lambda x: -x[1])[:4]
            err_str = " · ".join(f"`{t}` ×{n}" for t, n in top_errors)
            lines.append(f"Tool errors: {err_str}")
    else:
        lines.append("**Traces**: no data (DB not found)")

    # Telemetry section
    if telem["available"]:
        lines.append(
            f"\n**Inference (last {window_hours}h)**\n"
            f"Calls: {telem['total_calls']} · "
            f"Avg: {telem['avg_latency_ms']}ms · "
            f"p95: {telem['p95_latency_ms']}ms · "
            f"Tokens: {telem['total_tokens']:,} · "
            f"Errors: {telem['error_count']}"
        )
    else:
        lines.append("\n**Inference**: no data (DB not found)")

    # GitHub section
    if github["available"]:
        queue = github.get("work_queue", 0)
        wip   = github.get("in_progress", 0)
        done  = github.get("done_unmerged", 0)
        lines.append(
            f"\n**GitHub**\n"
            f"Work queue: {queue} · In progress: {wip} · Done (unmerged): {done}"
        )
        open_prs = github.get("open_prs", [])
        if open_prs:
            pr_lines = [f"  • [#{p['number']} {p['title']}]({p['url']})" for p in open_prs[:5]]
            lines.append("Open PRs:\n" + "\n".join(pr_lines))
    else:
        lines.append("\n**GitHub**: not configured")

    # Auto-issues
    if auto_issues:
        issue_lines = [f"  • {u}" for u in auto_issues]
        lines.append(f"\n⚠️ **Auto-issues opened ({len(auto_issues)})**\n" + "\n".join(issue_lines))

    desc = "\n".join(lines)
    if len(desc) > 3900:
        desc = desc[:3897] + "…"

    return {
        "embeds": [{
            "title": "🔭 Jarvis Observer — Health Report",
            "description": desc,
            "color": color,
            "footer": {"text": f"OpenJarvis Observer · {date_str} · interval {OBSERVER_INTERVAL//60}m"},
        }]
    }


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_observation() -> None:
    log.info("Running observation cycle...")

    trace  = analyse_traces(24)
    telem  = analyse_telemetry(24)
    github = analyse_github()

    # Auto-open issues for recurring failures
    auto_issues = maybe_open_auto_issues(trace)

    # Build and post health report
    embed = build_health_embed(trace, telem, github, auto_issues)
    posted = post_discord(embed)
    log.info(
        "Health report posted=%s | queries=%d stuck=%d errors=%d prs=%d",
        posted,
        trace.get("total_queries", 0),
        trace.get("hit_turn_limit", 0),
        sum(trace.get("tool_errors", {}).values()),
        len(github.get("open_prs", [])),
    )


def maybe_run_arxiv_digest() -> None:
    """Post arXiv digest once per day at ARXIV_HOUR UTC."""
    if ARXIV_HOUR < 0:
        return
    now  = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    if now.hour != ARXIV_HOUR or today in _arxiv_posted:
        return

    log.info("Running daily arXiv digest...")
    papers = fetch_arxiv_papers(
        "LLM language model AI agent reasoning alignment",
        n=8,
    )
    if papers:
        embed  = build_arxiv_embed(papers)
        posted = post_discord(embed)
        if posted:
            _arxiv_posted.add(today)
            log.info("arXiv digest posted (%d papers)", len(papers))
    else:
        log.warning("arXiv: no papers fetched")


def main() -> None:
    if not DISCORD_WEBHOOK:
        log.warning(
            "DISCORD_DIGEST_WEBHOOK not set — observer will run but cannot post to Discord. "
            "Set it in deploy/.env to enable notifications."
        )

    log.info(
        "Jarvis Observer starting — interval: %ds | arXiv hour: %d UTC | "
        "traces: %s | telemetry: %s",
        OBSERVER_INTERVAL,
        ARXIV_HOUR,
        TRACE_DB,
        TELEMETRY_DB,
    )

    # Initial run immediately on startup
    try:
        run_observation()
    except Exception as exc:
        log.exception("Observation cycle failed: %s", exc)

    while True:
        try:
            maybe_run_arxiv_digest()
        except Exception as exc:
            log.exception("arXiv digest failed: %s", exc)

        log.info("Sleeping %ds until next observation...", OBSERVER_INTERVAL)
        time.sleep(OBSERVER_INTERVAL)

        try:
            run_observation()
        except Exception as exc:
            log.exception("Observation cycle failed: %s", exc)


if __name__ == "__main__":
    main()
