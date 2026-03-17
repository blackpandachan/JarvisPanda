"""Jarvis Discord bot — full OpenJarvis SDK in-process, per-user memory.

Architecture:
- Runs the local qwen3:8b orchestrator directly (no API proxy)
- Per-user memory: every interaction is tagged with the Discord user ID so
  Jarvis builds a persistent model of each person across sessions
- Thread-based conversations: replies in a thread carry the last N messages
  as conversation context so multi-turn research actually works
- Reaction-based proposal approval: 👍 on a proposal message queues it for
  Claude Code; 👎 closes the issue

Triggers:
  - Any message in DISCORD_CHANNEL_NAME (default: #jarvis)
  - @mention anywhere in the server
  - DISCORD_PREFIX (default: !ask) in any channel

Commands:
  !status              — engine, model, memory stats
  !help                — full command reference
  !weather [city]      — current conditions + 3-day forecast
  !research <topic>    — multi-source synthesis, stored to memory
  !summarize <url>     — fetch + summarise any web page
  !memory <query>      — search your personal memory store
  !digest now          — trigger an on-demand HN AI digest
  !arxiv <query>       — search recent arXiv papers
  !sysinfo             — host CPU / RAM / GPU stats
  !skills              — list available Jarvis skills
  !propose <idea>      — draft + submit a feature proposal to GitHub
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import discord
import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("jarvis-discord")

# ── Config ────────────────────────────────────────────────────────────────────

DISCORD_BOT_TOKEN = os.environ["DISCORD_BOT_TOKEN"]
CHANNEL_NAME      = os.environ.get("DISCORD_CHANNEL_NAME", "jarvis")
COMMAND_PREFIX    = os.environ.get("DISCORD_PREFIX", "!ask")
DEFAULT_CITY      = os.environ.get("DEFAULT_WEATHER_CITY", "")
GITHUB_TOKEN      = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO       = os.environ.get("GITHUB_REPO", "")
MEMORY_DB_PATH    = os.environ.get("MEMORY_DB_PATH", "/data/memory.db")
MAX_MSG_LEN       = 1990
THREAD_HISTORY    = 12   # messages of context to pass for thread conversations

# Thread pool — synchronous Jarvis calls run here, never on the async event loop
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jarvis-ask")

# ── Jarvis SDK init ───────────────────────────────────────────────────────────

log.info("Initialising Jarvis SDK...")

from openjarvis import Jarvis  # noqa: E402

_jarvis: Jarvis | None = None
_jarvis_lock = threading.Lock()


def _get_jarvis() -> Jarvis:
    global _jarvis
    with _jarvis_lock:
        if _jarvis is None:
            _jarvis = Jarvis(config_path="/root/.openjarvis/config.toml")
            log.info("Jarvis SDK ready")
    return _jarvis


threading.Thread(target=_get_jarvis, daemon=True).start()

AGENT_TOOLS = [
    "think",
    "calculator",
    "retrieval",
    "web_search",
    "url_fetch",
    "arxiv_search",
    "weather",
    "system_monitor",
    "file_read",
    "github_issue",
]

# ── Discord client ────────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True
intents.reactions       = True
client = discord.Client(intents=intents)

# ── Per-user memory helpers ───────────────────────────────────────────────────

def _user_prefix(user_id: int, display_name: str) -> str:
    """Build the context preamble injected into every Jarvis query."""
    return (
        f"[Discord User: {display_name} | ID: {user_id}]\n"
        f"You are talking to this specific user. When storing memories use "
        f"'user:{user_id}:' as a metadata prefix so you can retrieve only their "
        f"memories when they ask. Tailor your response to their history and preferences.\n\n"
    )


def _search_user_memory(query: str, user_id: int, limit: int = 8) -> list[dict]:
    """Direct SQLite query against the memory DB, filtered to this user."""
    db_path = Path(MEMORY_DB_PATH)
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        # Try FTS5 table first, fall back to plain content search
        try:
            cur.execute(
                "SELECT content, source, metadata FROM memories_fts "
                "WHERE memories_fts MATCH ? AND (source LIKE ? OR metadata LIKE ?) "
                "LIMIT ?",
                (query, f"%user:{user_id}%", f"%user:{user_id}%", limit),
            )
        except sqlite3.OperationalError:
            # FTS table might be named differently — try a plain LIKE query
            cur.execute(
                "SELECT content, source, metadata FROM memories "
                "WHERE (content LIKE ? OR source LIKE ?) "
                "AND (source LIKE ? OR metadata LIKE ?) "
                "LIMIT ?",
                (f"%{query}%", f"%{query}%",
                 f"%user:{user_id}%", f"%user:{user_id}%", limit),
            )
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        log.debug("Memory search error: %s", exc)
        return []


# ── Core ask wrapper ─────────────────────────────────────────────────────────

def _run_ask(query: str, user_id: int | None = None, display_name: str = "User") -> str:
    """Synchronous Jarvis call — runs in thread pool."""
    j = _get_jarvis()
    full_query = query
    if user_id is not None:
        full_query = _user_prefix(user_id, display_name) + query
    try:
        result = j.ask(full_query, agent="orchestrator", tools=AGENT_TOOLS)
        # j.ask returns str per SDK signature
        if isinstance(result, dict):
            return result.get("content") or "(no response)"
        return str(result) or "(no response)"
    except Exception as exc:
        log.exception("Jarvis.ask failed: %s", exc)
        return (
            f"⚠️ Jarvis encountered an error: `{exc}`\n"
            "If this persists, I'll open a GitHub issue for the ClaudeCode worker to investigate."
        )


def _run_ask_with_thread_context(
    query: str,
    thread_history: str,
    user_id: int | None,
    display_name: str,
) -> str:
    """Like _run_ask but prepends thread conversation history."""
    context_block = ""
    if thread_history:
        context_block = (
            f"[Thread conversation so far — use this for context]\n"
            f"{thread_history}\n\n"
            f"[New message from {display_name}]\n"
        )
    return _run_ask(context_block + query, user_id, display_name)


# ── Specialised command handlers ──────────────────────────────────────────────

def _run_weather(city: str) -> str:
    try:
        from openjarvis.core.registry import ToolRegistry
        tool = ToolRegistry.get("weather")
        if tool is not None:
            return tool.execute(city=city).content
    except Exception:
        pass
    return _run_ask(
        f"Current weather and 3-day forecast for {city}. "
        "Use the weather tool to get accurate data."
    )


def _run_arxiv(query: str) -> str:
    try:
        from openjarvis.core.registry import ToolRegistry
        tool = ToolRegistry.get("arxiv_search")
        if tool is not None:
            return tool.execute(query=query, max_results=8).content
    except Exception:
        pass
    return _run_ask(f"Search arXiv for recent papers about: {query}")


def _run_sysinfo(detail: str = "summary") -> str:
    try:
        from openjarvis.core.registry import ToolRegistry
        tool = ToolRegistry.get("system_monitor")
        if tool is not None:
            return tool.execute(detail=detail).content
    except Exception:
        pass
    return "⚠️ system_monitor tool unavailable."


def _run_url_summarize(url: str, user_id: int | None, display_name: str) -> str:
    """Fetch a URL then ask Jarvis to summarise it."""
    try:
        from openjarvis.core.registry import ToolRegistry
        tool = ToolRegistry.get("url_fetch")
        if tool is not None:
            fetch_result = tool.execute(url=url, max_chars=6000)
            if fetch_result.success:
                prompt = (
                    f"Please summarise the following web page content concisely. "
                    f"Extract key points, main argument, and any important facts.\n\n"
                    f"URL: {url}\n\n"
                    f"{fetch_result.content}"
                )
                return _run_ask(prompt, user_id, display_name)
            return f"⚠️ Could not fetch URL: {fetch_result.content}"
    except Exception as exc:
        log.exception("url_summarize failed: %s", exc)
    return _run_ask(
        f"Please fetch and summarise this URL: {url}. Use the url_fetch tool.",
        user_id, display_name,
    )


def _run_research(topic: str, user_id: int | None, display_name: str) -> str:
    prefix = _user_prefix(user_id, display_name) if user_id else ""
    return _run_ask(
        prefix
        + f"Research the following topic thoroughly.\n"
        f"1. Use web_search for current information (at least 2 searches from different angles).\n"
        f"2. Use arxiv_search if it's a technical/scientific topic.\n"
        f"3. Check retrieval for anything already in memory.\n"
        f"4. Synthesise a well-structured report with key findings.\n"
        f"5. Store the synthesis to memory tagged with user:{user_id}:research if a user_id is provided.\n\n"
        f"Topic: {topic}"
    )


def _run_propose(idea: str, user_id: int | None, display_name: str) -> str:
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return (
            "⚠️ GITHUB_TOKEN and GITHUB_REPO must be set to submit proposals.\n"
            "Add them to `deploy/.env` and restart."
        )
    prefix = _user_prefix(user_id, display_name) if user_id else ""
    return _run_ask(
        prefix
        + f"A Discord user ({display_name}) has proposed this feature for the JarvisPanda project:\n\n"
        f"```\n{idea}\n```\n\n"
        f"Your job:\n"
        f"1. Think through feasibility, requirements, and tradeoffs.\n"
        f"2. Draft a GitHub issue with: clear title (< 80 chars), description, "
        f"technical approach, acceptance criteria.\n"
        f"3. Use the github_issue tool to submit it labelled 'claude-code-work'.\n"
        f"4. Return the issue URL so the user can track it and react with 👍 to approve "
        f"or 👎 to withdraw it.\n\n"
        f"Be technical and specific. This will be worked by the ClaudeCodeAgent."
    )


def _run_memory_search(query: str, user_id: int, display_name: str) -> str:
    rows = _search_user_memory(query, user_id)
    if not rows:
        return (
            f"No memories found for '{query}' for user {display_name}.\n"
            "Start chatting and Jarvis will build up your memory store over time."
        )
    lines = [f"**Memory results for '{query}' ({len(rows)} found)**\n"]
    for i, r in enumerate(rows, 1):
        content = r.get("content", "")[:300]
        lines.append(f"**{i}.** {content}")
    return "\n\n".join(lines)


def _run_digest_now() -> str:
    """Trigger an on-demand HN AI digest via the agent."""
    return _run_ask(
        "Fetch the top 10 most-discussed AI and machine learning stories on Hacker News "
        "from the last 72 hours. For each story include: rank, title (as a link), "
        "comment count, score, and a one-sentence description of what it's about. "
        "Format as a numbered Discord-friendly list. Use web_search to find the data."
    )


def _list_skills() -> str:
    try:
        from openjarvis.skills.loader import SkillLoader
        loader = SkillLoader()
        skills = loader.list_skills()
        if skills:
            lines = ["**Available Jarvis Skills**", "```"]
            for s in sorted(skills, key=lambda x: x.get("name", "")):
                name = s.get("name", "?")
                desc = (s.get("description", ""))[:60]
                lines.append(f"{name:<28} {desc}")
            lines.append("```")
            return "\n".join(lines)
    except Exception as exc:
        log.debug("SkillLoader failed: %s", exc)
    return (
        "**Available Jarvis Skills**\n```\n"
        "arxiv-daily              Latest AI papers from arXiv\n"
        "deep-research            Multi-search synthesis with citations\n"
        "morning-brief            Weather + top AI story + insight\n"
        "topic-research           Research + store to memory\n"
        "web-summarize            Summarise a web page\n"
        "daily-digest             Summarise recent activity\n"
        "code-lint                Lint and review code\n"
        "data-analyze             Analyse and interpret data\n"
        "```\n"
        "*Use `!research <topic>` or `!summarize <url>` for quick access.*"
    )


def _run_status() -> str:
    try:
        j   = _get_jarvis()
        cfg = j._config  # type: ignore[attr-defined]
        model      = getattr(cfg.intelligence, "default_model", "?")
        engine     = getattr(cfg.intelligence, "preferred_engine", "?")
        mem_cfg    = getattr(cfg, "memory", None)
        mem_backend = getattr(mem_cfg, "default_backend", "?") if mem_cfg else "?"
    except Exception:
        model, engine, mem_backend = "?", "?", "?"

    sys_info = _run_sysinfo("summary")
    return (
        f"**Jarvis Status**\n"
        f"```\n"
        f"Model  : {model}\n"
        f"Engine : {engine}\n"
        f"Memory : {mem_backend}\n"
        f"Tools  : {len(AGENT_TOOLS)} loaded\n"
        f"```\n"
        f"{sys_info}"
    )


def _help_text() -> str:
    return (
        f"**Jarvis — local AI assistant**\n"
        f"```\n"
        f"In #{CHANNEL_NAME}           — just type, no prefix\n"
        f"@Jarvis <query>         — mention me anywhere\n"
        f"{COMMAND_PREFIX} <query>      — explicit prefix\n"
        f"\n"
        f"Commands:\n"
        f"  !status               — engine, model, system stats\n"
        f"  !help                 — this message\n"
        f"  !weather [city]       — weather + 3-day forecast\n"
        f"  !research <topic>     — multi-source research synthesis\n"
        f"  !summarize <url>      — fetch and summarise a web page\n"
        f"  !memory <query>       — search your personal memory\n"
        f"  !digest now           — on-demand HN AI digest\n"
        f"  !arxiv <query>        — search recent arXiv papers\n"
        f"  !sysinfo              — host CPU/RAM/GPU stats\n"
        f"  !skills               — list available skills\n"
        f"  !propose <idea>       — submit a feature proposal\n"
        f"```\n"
        f"Runs **qwen3:8b** locally via Ollama. "
        f"React 👍 to approve proposals, 👎 to withdraw. "
        f"Stuck tasks auto-escalate to ClaudeCode."
    )


# ── GitHub label helpers (for reaction approval) ──────────────────────────────

def _gh_headers() -> dict[str, str]:
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _add_work_label(issue_number: int) -> None:
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return
    try:
        r = httpx.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/issues/{issue_number}",
            headers=_gh_headers(), timeout=10,
        )
        current = {lbl["name"] for lbl in r.json().get("labels", [])}
        current.add("claude-code-work")
        httpx.patch(
            f"https://api.github.com/repos/{GITHUB_REPO}/issues/{issue_number}",
            headers=_gh_headers(),
            json={"labels": list(current)},
            timeout=10,
        )
    except Exception as exc:
        log.warning("Could not add work label to #%d: %s", issue_number, exc)


def _close_issue(issue_number: int) -> None:
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return
    try:
        httpx.patch(
            f"https://api.github.com/repos/{GITHUB_REPO}/issues/{issue_number}",
            headers=_gh_headers(),
            json={"state": "closed"},
            timeout=10,
        )
    except Exception as exc:
        log.warning("Could not close issue #%d: %s", issue_number, exc)


def _extract_issue_number(text: str) -> int | None:
    """Parse a GitHub issue number from a message that contains an issue URL."""
    m = re.search(r"github\.com/[^/\s]+/[^/\s]+/issues/(\d+)", text)
    return int(m.group(1)) if m else None


# ── Message helpers ───────────────────────────────────────────────────────────

def extract_query(message: discord.Message) -> str | None:
    content = message.content.strip()

    if client.user and client.user.mentioned_in(message):
        return (
            content
            .replace(f"<@{client.user.id}>", "")
            .replace(f"<@!{client.user.id}>", "")
            .strip() or None
        )

    if content.lower().startswith(COMMAND_PREFIX.lower()):
        return content[len(COMMAND_PREFIX):].strip() or None

    if hasattr(message.channel, "name") and message.channel.name == CHANNEL_NAME:
        return content or None

    # Also respond inside threads that were started in the #jarvis channel
    if isinstance(message.channel, discord.Thread):
        parent = getattr(message.channel, "parent", None)
        if parent and getattr(parent, "name", "") == CHANNEL_NAME:
            return content or None

    return None


async def build_thread_context(channel: discord.Thread, exclude_id: int) -> str:
    """Return the last THREAD_HISTORY non-bot messages as a conversation string."""
    lines: list[str] = []
    try:
        async for msg in channel.history(limit=THREAD_HISTORY + 1, oldest_first=False):
            if msg.id == exclude_id:
                continue
            if msg.author.bot:
                continue
            lines.append(f"{msg.author.display_name}: {msg.content.strip()}")
            if len(lines) >= THREAD_HISTORY:
                break
        lines.reverse()
    except Exception as exc:
        log.debug("Thread history failed: %s", exc)
    return "\n".join(lines)


async def send_chunked(dest: discord.abc.Messageable, text: str) -> None:
    if len(text) <= MAX_MSG_LEN:
        await dest.send(text)
        return
    paragraphs = text.split("\n\n")
    chunk = ""
    for para in paragraphs:
        candidate = (chunk + "\n\n" + para).lstrip("\n") if chunk else para
        if len(candidate) <= MAX_MSG_LEN:
            chunk = candidate
        else:
            if chunk:
                await dest.send(chunk)
            while len(para) > MAX_MSG_LEN:
                await dest.send(para[:MAX_MSG_LEN])
                para = para[MAX_MSG_LEN:]
            chunk = para
    if chunk:
        await dest.send(chunk)


# ── Events ────────────────────────────────────────────────────────────────────

@client.event
async def on_ready() -> None:
    log.info(
        "Jarvis Discord bot online: %s (id=%s) | #%s | prefix=%s",
        client.user, getattr(client.user, "id", "?"), CHANNEL_NAME, COMMAND_PREFIX,
    )


@client.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent) -> None:
    """Handle 👍/👎 reactions on Jarvis proposal messages."""
    if payload.user_id == (client.user.id if client.user else None):
        return

    emoji = str(payload.emoji)
    if emoji not in ("👍", "👎"):
        return

    try:
        channel = client.get_channel(payload.channel_id)
        if channel is None:
            channel = await client.fetch_channel(payload.channel_id)
        message = await channel.fetch_message(payload.message_id)
    except Exception:
        return

    # Only act on messages the bot sent
    if not client.user or message.author.id != client.user.id:
        return

    issue_number = _extract_issue_number(message.content)
    if issue_number is None:
        return

    loop = asyncio.get_event_loop()
    if emoji == "👍":
        await loop.run_in_executor(_executor, _add_work_label, issue_number)
        await channel.send(
            f"✅ Issue #{issue_number} approved — added to Claude's work queue. "
            f"The ClaudeCodeAgent will pick it up on the next poll."
        )
        log.info("Issue #%d approved via reaction by user %d", issue_number, payload.user_id)
    elif emoji == "👎":
        await loop.run_in_executor(_executor, _close_issue, issue_number)
        await channel.send(
            f"❌ Issue #{issue_number} withdrawn and closed."
        )
        log.info("Issue #%d withdrawn via reaction by user %d", issue_number, payload.user_id)


@client.event
async def on_message(message: discord.Message) -> None:
    if message.author == client.user or message.author.bot:
        return

    content      = message.content.strip()
    lower        = content.lower()
    user_id      = message.author.id
    display_name = message.author.display_name
    loop         = asyncio.get_event_loop()

    # ── Built-in commands ─────────────────────────────────────────────────────

    if lower == "!status":
        async with message.channel.typing():
            reply = await loop.run_in_executor(_executor, _run_status)
        await send_chunked(message.channel, reply)
        return

    if lower == "!help":
        await message.channel.send(_help_text())
        return

    if lower == "!skills":
        async with message.channel.typing():
            reply = await loop.run_in_executor(_executor, _list_skills)
        await send_chunked(message.channel, reply)
        return

    if lower == "!sysinfo" or lower == "!sysinfo full":
        detail = "full" if "full" in lower else "summary"
        async with message.channel.typing():
            reply = await loop.run_in_executor(_executor, _run_sysinfo, detail)
        await send_chunked(message.channel, reply)
        return

    if lower.startswith("!weather"):
        city = content[8:].strip() or DEFAULT_CITY
        if not city:
            await message.channel.send(
                "Usage: `!weather <city>` — e.g. `!weather London`"
            )
            return
        async with message.channel.typing():
            reply = await loop.run_in_executor(_executor, _run_weather, city)
        await send_chunked(message.channel, reply)
        return

    if lower.startswith("!arxiv "):
        query = content[7:].strip()
        if not query:
            await message.channel.send("Usage: `!arxiv <search query>`")
            return
        async with message.channel.typing():
            reply = await loop.run_in_executor(_executor, _run_arxiv, query)
        await send_chunked(message.channel, reply)
        return

    if lower.startswith("!research "):
        topic = content[10:].strip()
        if not topic:
            await message.channel.send("Usage: `!research <topic>`")
            return
        async with message.channel.typing():
            reply = await loop.run_in_executor(
                _executor, _run_research, topic, user_id, display_name
            )
        if (
            isinstance(message.channel, (discord.TextChannel, discord.Thread))
            and len(reply) > MAX_MSG_LEN
        ):
            try:
                thread = await message.create_thread(
                    name=f"Research: {topic[:45]}",
                    auto_archive_duration=60,
                )
                await send_chunked(thread, reply)
                return
            except discord.Forbidden:
                pass
        await send_chunked(message.channel, reply)
        return

    if lower.startswith("!summarize "):
        url = content[11:].strip()
        if not url:
            await message.channel.send("Usage: `!summarize <url>`")
            return
        async with message.channel.typing():
            reply = await loop.run_in_executor(
                _executor, _run_url_summarize, url, user_id, display_name
            )
        await send_chunked(message.channel, reply)
        return

    if lower.startswith("!memory "):
        query = content[8:].strip()
        if not query:
            await message.channel.send("Usage: `!memory <search query>`")
            return
        async with message.channel.typing():
            reply = await loop.run_in_executor(
                _executor, _run_memory_search, query, user_id, display_name
            )
        await send_chunked(message.channel, reply)
        return

    if lower == "!digest now":
        await message.channel.send("Fetching HN AI digest… this may take a moment.")
        async with message.channel.typing():
            reply = await loop.run_in_executor(_executor, _run_digest_now)
        if (
            isinstance(message.channel, (discord.TextChannel, discord.Thread))
            and len(reply) > MAX_MSG_LEN
        ):
            try:
                thread = await message.create_thread(
                    name="HN AI Digest (on-demand)",
                    auto_archive_duration=60,
                )
                await send_chunked(thread, reply)
                return
            except discord.Forbidden:
                pass
        await send_chunked(message.channel, reply)
        return

    if lower.startswith("!propose "):
        idea = content[9:].strip()
        if not idea:
            await message.channel.send("Usage: `!propose <your feature idea>`")
            return
        await message.channel.send(
            "Drafting your proposal and opening a GitHub issue… "
            "React 👍 to approve it for Claude's queue, or 👎 to withdraw."
        )
        async with message.channel.typing():
            reply = await loop.run_in_executor(
                _executor, _run_propose, idea, user_id, display_name
            )
        await send_chunked(message.channel, reply)
        return

    # ── Regular Jarvis query (with thread context if in a thread) ─────────────

    query = extract_query(message)
    if query is None:
        return

    log.info("[%s] %s (id=%d): %s", message.channel, display_name, user_id, query[:100])

    # Gather thread history if we're in a thread
    thread_ctx = ""
    if isinstance(message.channel, discord.Thread):
        thread_ctx = await build_thread_context(message.channel, message.id)

    async with message.channel.typing():
        if thread_ctx:
            reply = await loop.run_in_executor(
                _executor,
                _run_ask_with_thread_context,
                query,
                thread_ctx,
                user_id,
                display_name,
            )
        else:
            reply = await loop.run_in_executor(
                _executor, _run_ask, query, user_id, display_name
            )

    # Long responses → thread to keep channel clean
    if (
        isinstance(message.channel, (discord.TextChannel, discord.Thread))
        and len(reply) > MAX_MSG_LEN
        and not isinstance(message.channel, discord.Thread)
    ):
        try:
            thread = await message.create_thread(
                name=f"Jarvis: {query[:50]}",
                auto_archive_duration=60,
            )
            await send_chunked(thread, reply)
            return
        except discord.Forbidden:
            pass

    await send_chunked(message.channel, reply)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client.run(DISCORD_BOT_TOKEN, log_handler=None)
