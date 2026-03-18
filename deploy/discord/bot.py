"""Jarvis Discord bot — personal local AI assistant built on OpenJarvis.

Design:
  1. Slash commands primary  — /ask, /research, /code, /pdf, etc.
  2. Free-form text          — type anything in #jarvis (or @mention) → agent responds
  3. Local model primary     — qwen3:8b via Ollama, no cloud needed for most tasks
  4. Gemini Flash mid-tier   — agent escalates when stuck (via gemini_escalate tool)
  5. Per-user memory         — every interaction tagged with Discord user ID
  6. Thread context          — replies in threads carry conversation history
  7. Interactive agents      — agents can post polls, ask follow-up questions, etc.
  8. Scheduler               — background tasks posted to Discord webhook

Adding commands: add a @tree.command() function below the "Slash commands" section.
Adding tools:    create src/openjarvis/tools/my_tool.py, register with @ToolRegistry.register,
                 import in tools/__init__.py — appears automatically in /tools and /status.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import discord
import httpx

# Scheduler — optional, gracefully disabled if unavailable
try:
    from openjarvis.scheduler import SchedulerStore, TaskScheduler
    _SCHEDULER_AVAILABLE = True
except Exception:
    _SCHEDULER_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("jarvis-discord")

# ── Config ────────────────────────────────────────────────────────────────────

DISCORD_BOT_TOKEN = os.environ["DISCORD_BOT_TOKEN"]
CHANNEL_NAME      = os.environ.get("DISCORD_CHANNEL_NAME", "jarvis")
DEFAULT_CITY      = os.environ.get("DEFAULT_WEATHER_CITY", "")
MEMORY_DB_PATH    = os.environ.get("MEMORY_DB_PATH", "/data/memory.db")
DISCORD_WEBHOOK   = os.environ.get("DISCORD_DIGEST_WEBHOOK", "")
OLLAMA_HOST       = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")

_GUILD_ID_STR    = os.environ.get("DISCORD_GUILD_ID", "")
DISCORD_GUILD_ID: int | None = int(_GUILD_ID_STR) if _GUILD_ID_STR.isdigit() else None

MAX_MSG_LEN   = 1990
THREAD_HISTORY = 12

# ── Thread pool ───────────────────────────────────────────────────────────────

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jarvis-ask")

# ── Jarvis SDK + tool discovery ───────────────────────────────────────────────

log.info("Initialising Jarvis SDK...")
from openjarvis import Jarvis  # noqa: E402

_jarvis: Jarvis | None = None
_jarvis_lock = threading.Lock()

_BASELINE_TOOLS = [
    "think", "calculator", "retrieval", "web_search",
    "url_fetch", "arxiv_search", "weather", "system_monitor",
    "file_read", "memory_store", "memory_search",
    "gemini_escalate", "pdf_tool", "code_interpreter",
]
_EXTRA_TOOLS = [
    t.strip() for t in os.environ.get("AGENT_EXTRA_TOOLS", "").split(",") if t.strip()
]


def _discover_agent_tools() -> list[str]:
    try:
        from openjarvis.core.registry import ToolRegistry
        registered = set(ToolRegistry.keys())
    except Exception:
        registered = set()
    _BLOCKED = {"shell_exec", "code_interpreter_docker", "repl"}
    combined = set(_BASELINE_TOOLS) | set(_EXTRA_TOOLS) | registered
    return sorted(combined - _BLOCKED)


def _get_jarvis() -> Jarvis:
    global _jarvis
    with _jarvis_lock:
        if _jarvis is None:
            _jarvis = Jarvis(config_path="/root/.openjarvis/config.toml")
            log.info("Jarvis SDK ready")
    return _jarvis


threading.Thread(target=_get_jarvis, daemon=True).start()

AGENT_TOOLS: list[str] = _BASELINE_TOOLS


def _refresh_agent_tools() -> None:
    global AGENT_TOOLS
    AGENT_TOOLS = _discover_agent_tools()
    log.info("Agent tools: %s", ", ".join(AGENT_TOOLS))


threading.Thread(target=_refresh_agent_tools, daemon=True).start()

# ── Scheduler ─────────────────────────────────────────────────────────────────

SCHEDULER_DB_PATH = os.environ.get("SCHEDULER_DB_PATH", "/data/scheduler.db")
_scheduler: "TaskScheduler | None" = None


def _post_task_result_to_discord(payload: dict) -> None:
    if not DISCORD_WEBHOOK:
        return
    task_id = payload.get("task_id", "?")[:8]
    success = payload.get("success", False)
    result  = (payload.get("result") or "").strip()
    error   = (payload.get("error") or "").strip()
    if not result and not error:
        return
    icon    = "✅" if success else "❌"
    content = result[:1800] if success else f"Task failed:\n```\n{error[:500]}\n```"
    body    = f"{icon} **Scheduled task `{task_id}` complete**\n\n{content}"
    try:
        httpx.post(DISCORD_WEBHOOK, json={"content": body}, timeout=10)
    except Exception as exc:
        log.debug("Webhook post failed for task %s: %s", task_id, exc)


def _init_scheduler() -> None:
    global _scheduler
    if not _SCHEDULER_AVAILABLE:
        log.warning("TaskScheduler not available — skipping scheduler init")
        return
    try:
        bus = None
        try:
            from openjarvis.core.events import EventBus
            bus = EventBus()
            bus.subscribe("scheduler_task_end", _post_task_result_to_discord)
        except Exception:
            pass

        store = SchedulerStore(SCHEDULER_DB_PATH)
        _scheduler = TaskScheduler(store, system=_get_jarvis(), poll_interval=60, bus=bus)
        _scheduler.start()
        log.info("TaskScheduler started (db=%s)", SCHEDULER_DB_PATH)

        existing = {t.metadata.get("seed_id") for t in _scheduler.list_tasks()}

        def _seed(seed_id: str, **kwargs: Any) -> None:
            if seed_id not in existing:
                meta = kwargs.pop("metadata", {})
                meta["seed_id"] = seed_id
                _scheduler.create_task(metadata=meta, **kwargs)  # type: ignore[union-attr]
                log.info("Seeded task: %s", seed_id)

        _seed(
            "morning-brief-daily",
            prompt=(
                "Run the morning-brief skill. City: " + (DEFAULT_CITY or "New York") + ". "
                "Summarise today's weather, top AI news, and one insight. "
                "Store the result tagged 'morning_brief:daily' in memory."
            ),
            schedule_type="cron",
            schedule_value="0 9 * * *",
            agent="orchestrator",
            tools="weather,web_search,think,retrieval",
            metadata={"system_prompt": "Produce a concise morning brief. Lead with weather, then one AI story, close with an insight."},
        )
        _seed(
            "weekly-ai-digest",
            prompt=(
                "Research and synthesise the most significant AI/ML developments from the past week. "
                "Use web_search and arxiv_search. Produce a digest: breakthroughs, notable papers, "
                "tooling updates, and one practical takeaway. Store tagged 'weekly_digest:ai'."
            ),
            schedule_type="cron",
            schedule_value="0 10 * * 1",
            agent="orchestrator",
            tools="web_search,arxiv_search,think,retrieval",
            metadata={"system_prompt": "Produce a weekly AI research digest. Be comprehensive but concise. Cite sources."},
        )
    except Exception as exc:
        log.error("Scheduler init failed: %s", exc)


# ── Discord client ────────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree   = discord.app_commands.CommandTree(client)

# ── Per-user memory ───────────────────────────────────────────────────────────

def _user_prefix(user_id: int, display_name: str) -> str:
    return (
        f"[Discord User: {display_name} | ID: {user_id}]\n"
        f"Tag any memory stores with 'user:{user_id}:' in the source/metadata field.\n\n"
    )


def _search_user_memory(query: str, user_id: int) -> list[dict]:
    db = Path(MEMORY_DB_PATH)
    if not db.exists():
        return []
    try:
        conn = sqlite3.connect(str(db), timeout=5)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        uid_filter = f"%user:{user_id}%"
        try:
            if query:
                cur.execute(
                    "SELECT content, source FROM memories_fts "
                    "WHERE memories_fts MATCH ? AND (source LIKE ? OR metadata LIKE ?) LIMIT 8",
                    (query, uid_filter, uid_filter),
                )
            else:
                cur.execute(
                    "SELECT content, source FROM memories "
                    "WHERE source LIKE ? OR metadata LIKE ? LIMIT 20",
                    (uid_filter, uid_filter),
                )
        except sqlite3.OperationalError:
            cur.execute(
                "SELECT content, source FROM memories "
                "WHERE (content LIKE ? OR source LIKE ?) "
                "AND (source LIKE ? OR metadata LIKE ?) LIMIT 8",
                (f"%{query}%", f"%{query}%", uid_filter, uid_filter),
            )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as exc:
        log.debug("Memory search error: %s", exc)
        return []


# ── Core agent call ───────────────────────────────────────────────────────────

def _current_date_prefix() -> str:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return (
        f"[Current date/time: {now.strftime('%A, %B %d, %Y')} at {now.strftime('%H:%M UTC')}]\n"
        "For any question about current events, prices, news, or recent data: "
        "use web_search with today's date in the query to get up-to-date results.\n\n"
    )


def _run_ask(
    query: str,
    user_id: int | None = None,
    display_name: str = "User",
    thread_history: str = "",
    agent: str = "orchestrator",
) -> str:
    j = _get_jarvis()
    parts: list[str] = [_current_date_prefix()]
    if user_id is not None:
        parts.append(_user_prefix(user_id, display_name))
    if thread_history:
        parts.append(
            f"[Thread history]\n{thread_history}\n\n"
            f"[Current message from {display_name}]\n"
        )
    parts.append(query)
    full_query = "".join(parts)
    try:
        result = j.ask(full_query, agent=agent, tools=AGENT_TOOLS)
        if isinstance(result, dict):
            return result.get("content") or "(no response)"
        return str(result) or "(no response)"
    except Exception as exc:
        log.exception("Jarvis.ask failed: %s", exc)
        return f"⚠️ Agent error: `{exc}`"


def _call_tool(name: str, **kwargs: Any) -> str | None:
    try:
        from openjarvis.core.registry import ToolRegistry
        tool = ToolRegistry.get(name)
        if tool is None:
            return None
        result = tool.execute(**kwargs)
        return result.content if result.success else None
    except Exception as exc:
        log.debug("Direct tool call %r failed: %s", name, exc)
        return None


# ── Discord helpers ───────────────────────────────────────────────────────────

def extract_query(message: discord.Message) -> str | None:
    """Extract query from a message in #jarvis or @mention."""
    content = message.content.strip()
    if client.user and client.user.mentioned_in(message):
        return (
            content
            .replace(f"<@{client.user.id}>", "")
            .replace(f"<@!{client.user.id}>", "")
            .strip() or None
        )
    if hasattr(message.channel, "name") and message.channel.name == CHANNEL_NAME:
        return content or None
    if isinstance(message.channel, discord.Thread):
        parent = getattr(message.channel, "parent", None)
        if parent and getattr(parent, "name", "") == CHANNEL_NAME:
            return content or None
    return None


async def build_thread_context(channel: discord.Thread, exclude_id: int) -> str:
    lines: list[str] = []
    try:
        async for msg in channel.history(limit=THREAD_HISTORY + 1, oldest_first=False):
            if msg.id == exclude_id or msg.author.bot:
                continue
            lines.append(f"{msg.author.display_name}: {msg.content.strip()}")
            if len(lines) >= THREAD_HISTORY:
                break
        lines.reverse()
    except Exception as exc:
        log.debug("Thread history: %s", exc)
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


async def reply_long(message: discord.Message, text: str, thread_name: str) -> None:
    if isinstance(message.channel, discord.TextChannel) and len(text) > MAX_MSG_LEN:
        try:
            thread = await message.create_thread(name=thread_name[:99], auto_archive_duration=60)
            await send_chunked(thread, text)
            return
        except discord.Forbidden:
            pass
    await send_chunked(message.channel, text)


async def _slash_reply(
    interaction: discord.Interaction,
    fn: "Callable[[], str]",
    thread_name: str = "",
) -> None:
    await interaction.response.defer(thinking=True)
    loop  = asyncio.get_event_loop()
    reply = await loop.run_in_executor(_executor, fn)
    if (
        thread_name
        and len(reply) > MAX_MSG_LEN
        and isinstance(interaction.channel, discord.TextChannel)
    ):
        try:
            seed   = await interaction.followup.send(f"*{thread_name}*", wait=True)
            thread = await seed.create_thread(name=thread_name[:99], auto_archive_duration=60)
            await send_chunked(thread, reply)
            return
        except discord.Forbidden:
            pass
    chunks = [reply[i : i + MAX_MSG_LEN] for i in range(0, max(len(reply), 1), MAX_MSG_LEN)]
    for chunk in chunks:
        await interaction.followup.send(chunk)


def _resolve_task(task_id: str) -> "tuple[Any | None, str]":
    if _scheduler is None:
        return None, "Scheduler is not running."
    matches = [t for t in _scheduler.list_tasks() if t.id.startswith(task_id)]
    if not matches:
        return None, f"No task found with id starting `{task_id}`. Run `/tasks` to list."
    if len(matches) > 1:
        ids = ", ".join(t.id[:8] for t in matches)
        return None, f"Multiple tasks match `{task_id}`: {ids}"
    return matches[0], ""


# ── Slash commands ────────────────────────────────────────────────────────────
# Everything is a /slash command. Free-form text in #jarvis also works.

_Choice = discord.app_commands.Choice


# ── Conversation ──────────────────────────────────────────────────────────────

@tree.command(name="ask", description="Ask Jarvis anything")
@discord.app_commands.describe(query="Your question or task")
async def slash_ask(interaction: discord.Interaction, query: str) -> None:
    uid, name = interaction.user.id, interaction.user.display_name
    await _slash_reply(
        interaction,
        lambda: _run_ask(query, uid, name),
        thread_name=f"Jarvis: {query[:50]}",
    )


@tree.command(name="agent", description="Run a task with the full OrchestratorAgent and all tools")
@discord.app_commands.describe(task="What you want the agent to do")
async def slash_agent(interaction: discord.Interaction, task: str) -> None:
    uid, name = interaction.user.id, interaction.user.display_name
    await _slash_reply(
        interaction,
        lambda: _run_ask(task, uid, name),
        thread_name=f"Agent: {task[:50]}",
    )


@tree.command(name="react", description="Run a task with NativeReActAgent — shows Thought/Action/Observation reasoning")
@discord.app_commands.describe(task="Task to reason through step-by-step")
async def slash_react(interaction: discord.Interaction, task: str) -> None:
    uid, name = interaction.user.id, interaction.user.display_name
    await _slash_reply(
        interaction,
        lambda: _run_ask(task, uid, name, agent="native_react"),
        thread_name=f"ReAct: {task[:50]}",
    )


@tree.command(name="ask-gemini", description="Ask Google Gemini Flash directly, bypassing the local model")
@discord.app_commands.describe(question="Question for Gemini")
async def slash_ask_gemini(interaction: discord.Interaction, question: str) -> None:
    def _gemini() -> str:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return "⚠️ GEMINI_API_KEY is not configured."
        try:
            from openjarvis.tools.gemini_escalate import gemini_generate
            return f"**Gemini Flash** says:\n\n{gemini_generate(question)}"
        except Exception as exc:
            return f"⚠️ Gemini call failed: {exc}"

    await _slash_reply(interaction, _gemini, thread_name=f"Gemini: {question[:50]}")


# ── Research & Information ────────────────────────────────────────────────────

@tree.command(name="research", description="Multi-source research synthesis stored to your memory")
@discord.app_commands.describe(topic="What to research")
async def slash_research(interaction: discord.Interaction, topic: str) -> None:
    uid, name = interaction.user.id, interaction.user.display_name

    def _research() -> str:
        return _run_ask(
            f"Research '{topic}' thoroughly.\n"
            "1. Use web_search with at least 2 different angles.\n"
            "2. Use arxiv_search if technical/scientific.\n"
            "3. Check retrieval for prior memory context.\n"
            "4. Synthesise: Overview, Key Concepts, Recent Developments, "
            "Practical Applications, Key Takeaways.\n"
            f"5. Store findings tagged 'user:{uid}:research:{topic[:30]}'.",
            uid, name,
        )

    await _slash_reply(interaction, _research, thread_name=f"Research: {topic[:50]}")


@tree.command(name="deep-dive", description="Thorough 15-turn research with arXiv + web, saved to memory")
@discord.app_commands.describe(topic="Topic for deep research")
async def slash_deep_dive(interaction: discord.Interaction, topic: str) -> None:
    uid, name = interaction.user.id, interaction.user.display_name
    await interaction.response.defer(thinking=True)
    await interaction.followup.send(
        f"Starting deep research on **{topic}**…\n"
        "Running up to 15 agent turns with web search + arXiv. Results saved to your memory."
    )
    loop  = asyncio.get_event_loop()
    reply = await loop.run_in_executor(
        _executor,
        lambda: _run_ask(
            f"Comprehensive deep research on: '{topic}'\n\n"
            "Strategy: web_search (3+ angles), arxiv_search, url_fetch (2-3 key sources), "
            f"think to synthesise, memory_store tagged 'user:{uid}:deepdive:{topic[:25]}'.\n\n"
            "Output: Executive Summary, Key Findings (with sources), Technical Details, "
            "Recent Developments, Open Questions, Key Sources.",
            uid, name,
        ),
    )
    chunks = [reply[i : i + MAX_MSG_LEN] for i in range(0, max(len(reply), 1), MAX_MSG_LEN)]
    for chunk in chunks:
        await interaction.followup.send(chunk)


@tree.command(name="summarize", description="Fetch and summarise a web page")
@discord.app_commands.describe(url="URL to summarise")
async def slash_summarize(interaction: discord.Interaction, url: str) -> None:
    def _summarize() -> str:
        raw = _call_tool("url_fetch", url=url, max_chars=6000)
        if raw:
            return _run_ask(f"Summarise this page. Key points and main argument.\n\nURL: {url}\n\n{raw}")
        return _run_ask(f"Fetch and summarise: {url}")

    await _slash_reply(interaction, _summarize, thread_name=f"Summary: {url[:60]}")


@tree.command(name="pdf", description="Extract and summarise a PDF from a URL")
@discord.app_commands.describe(url="URL of the PDF")
async def slash_pdf(interaction: discord.Interaction, url: str) -> None:
    uid, name = interaction.user.id, interaction.user.display_name

    def _pdf() -> str:
        result = _call_tool("pdf_tool", url=url)
        if result:
            return _run_ask(
                f"Summarise this PDF content from {url}:\n\n{result[:6000]}\n\n"
                "Provide: key points, main argument, important figures/tables if any.",
                uid, name,
            )
        return _run_ask(
            f"Use pdf_tool to extract and summarise the PDF at: {url}",
            uid, name,
        )

    await _slash_reply(interaction, _pdf, thread_name=f"PDF: {url[:60]}")


@tree.command(name="arxiv", description="Search recent arXiv papers")
@discord.app_commands.describe(query="Search query")
async def slash_arxiv(interaction: discord.Interaction, query: str) -> None:
    await _slash_reply(
        interaction,
        lambda: _call_tool("arxiv_search", query=query, max_results=8)
                or _run_ask(f"Search arXiv for: {query}"),
        thread_name=f"arXiv: {query[:50]}",
    )


@tree.command(name="papers", description="Latest AI/ML arXiv papers (or specify a topic)")
@discord.app_commands.describe(topic="Topic (leave blank for general AI/ML)")
async def slash_papers(interaction: discord.Interaction, topic: str = "") -> None:
    query = topic or "LLM AI agent reasoning alignment"
    await _slash_reply(
        interaction,
        lambda: _call_tool("arxiv_search", query=query, max_results=8, sort_by="submittedDate")
                or _run_ask(f"Latest arXiv papers on: {query}"),
        thread_name=f"arXiv: {query[:50]}",
    )


@tree.command(name="weather", description="Current conditions + 3-day forecast")
@discord.app_commands.describe(city="City name (leave blank for default)")
async def slash_weather(interaction: discord.Interaction, city: str = "") -> None:
    city = city or DEFAULT_CITY
    if not city:
        await interaction.response.send_message(
            "Set a default city in `.env` (`DEFAULT_WEATHER_CITY`) or provide one: `/weather city:London`",
            ephemeral=True,
        )
        return
    await _slash_reply(
        interaction,
        lambda: _call_tool("weather", city=city) or _run_ask(f"Weather in {city}?"),
    )


@tree.command(name="brief", description="Morning brief: weather + top AI story + daily insight")
async def slash_brief(interaction: discord.Interaction) -> None:
    uid, name = interaction.user.id, interaction.user.display_name

    def _brief() -> str:
        city    = DEFAULT_CITY or "New York"
        weather = _call_tool("weather", city=city) or ""
        return _run_ask(
            f"Give me a concise morning brief:\n"
            f"1. **Weather** — summarise: {weather[:300]}\n"
            f"2. **Top AI Story** — use web_search, 3 sentences.\n"
            f"3. **Insight** — one thought-provoking question for today.\n"
            f"Keep it concise and useful.",
            uid, name,
        )

    await _slash_reply(interaction, _brief, thread_name="Morning Brief")


@tree.command(name="digest", description="On-demand Hacker News AI digest (last 72h)")
async def slash_digest(interaction: discord.Interaction) -> None:
    def _digest() -> str:
        return _run_ask(
            "Fetch the top 10 most-discussed AI/ML stories on Hacker News from the last 72 hours. "
            "Use web_search. For each: rank, title (as a link), comment count, score, one-sentence description. "
            "Format as a numbered Discord-friendly list."
        )

    await _slash_reply(interaction, _digest, thread_name="HN AI Digest")


@tree.command(name="trending", description="What's trending in AI and tech right now")
async def slash_trending(interaction: discord.Interaction) -> None:
    def _trending() -> str:
        return _run_ask(
            "What's trending in AI and tech right now? Use web_search (multiple queries).\n\n"
            "**Top stories this week** — 3-4 items with link + 1-sentence description.\n"
            "**Hot tools/models** — 2-3 notable releases or demos.\n"
            "**Worth watching** — 1-2 early-stage things gaining traction."
        )

    await _slash_reply(interaction, _trending, thread_name="Trending in AI/Tech")


# ── Knowledge ─────────────────────────────────────────────────────────────────

@tree.command(name="explain", description="Explain a concept at three levels: simple, intermediate, expert")
@discord.app_commands.describe(concept="What to explain")
async def slash_explain(interaction: discord.Interaction, concept: str) -> None:
    uid, name = interaction.user.id, interaction.user.display_name

    def _explain() -> str:
        return _run_ask(
            f"Explain '{concept}' at three levels:\n\n"
            "**Simple (ELI5):** 2-3 sentences anyone can understand.\n\n"
            "**Intermediate:** 3-5 sentences with key concepts and how they connect.\n\n"
            "**Expert:** 3-5 sentences with technical depth, edge cases, and nuance.\n\n"
            "Use web_search if recent or technical. Be concise and concrete.",
            uid, name,
        )

    await _slash_reply(interaction, _explain, thread_name=f"Explain: {concept[:50]}")


@tree.command(name="compare", description="Structured comparison of two things")
@discord.app_commands.describe(items="What to compare, e.g. 'PyTorch vs JAX'")
async def slash_compare(interaction: discord.Interaction, items: str) -> None:
    uid, name = interaction.user.id, interaction.user.display_name
    if " vs " not in items.lower():
        await interaction.response.send_message(
            "Format: `/compare items:X vs Y` — e.g. `/compare items:PyTorch vs JAX`",
            ephemeral=True,
        )
        return

    def _compare() -> str:
        return _run_ask(
            f"Compare {items}. Use web_search for current info.\n\n"
            "**Overview:** one sentence each.\n"
            "**Key differences:** 4-6 bullet points.\n"
            "**When to use each:** 2-3 scenarios each.\n"
            "**Verdict:** direct recommendation.",
            uid, name,
        )

    await _slash_reply(interaction, _compare, thread_name=f"Compare: {items[:60]}")


@tree.command(name="debate", description="Structured pros and cons analysis")
@discord.app_commands.describe(topic="Topic or claim to analyse")
async def slash_debate(interaction: discord.Interaction, topic: str) -> None:
    uid, name = interaction.user.id, interaction.user.display_name

    def _debate() -> str:
        return _run_ask(
            f"Analyse: '{topic}'. Use web_search for real arguments.\n\n"
            "**For:** 3-4 bullet points with evidence.\n"
            "**Against:** 3-4 bullet points with evidence.\n"
            "**Nuance:** 1-2 caveats the debate often misses.\n"
            "**Bottom line:** honest synthesis in 2-3 sentences.",
            uid, name,
        )

    await _slash_reply(interaction, _debate, thread_name=f"Debate: {topic[:50]}")


@tree.command(name="code", description="Run Python code or get a code answer with a live example")
@discord.app_commands.describe(query="Code question or Python snippet to run")
async def slash_code(interaction: discord.Interaction, query: str) -> None:
    uid, name = interaction.user.id, interaction.user.display_name

    def _code() -> str:
        return _run_ask(
            f"Answer this coding question or run this code using code_interpreter:\n\n"
            f"```\n{query}\n```\n\n"
            "If it's runnable Python, execute it and show the output. "
            "If it's a question, write and run a minimal example. Show both code and result.",
            uid, name,
        )

    await _slash_reply(interaction, _code, thread_name=f"Code: {query[:50]}")


# ── Memory ────────────────────────────────────────────────────────────────────

@tree.command(name="memory", description="Search your personal memory store")
@discord.app_commands.describe(query="Search terms")
async def slash_memory(interaction: discord.Interaction, query: str) -> None:
    uid = interaction.user.id

    def _mem() -> str:
        rows = _search_user_memory(query, uid)
        if not rows:
            return f"No memories found for '{query}'.\nChat with Jarvis to build your memory store."
        lines = [f"**Memory results for '{query}'** ({len(rows)})"]
        for i, r in enumerate(rows, 1):
            lines.append(f"**{i}.** {r.get('content', '')[:280]}")
        return "\n\n".join(lines)

    await _slash_reply(interaction, _mem)


@tree.command(name="remember", description="Store a fact or note to your personal long-term memory")
@discord.app_commands.describe(fact="The fact, note, or information to remember")
async def slash_remember(interaction: discord.Interaction, fact: str) -> None:
    uid, name = interaction.user.id, interaction.user.display_name

    def _remember() -> str:
        result = _call_tool(
            "memory_store",
            content=fact,
            source=f"user:{uid}:manual",
            metadata={"user_id": str(uid), "display_name": name},
        )
        if result is not None:
            return f"✅ Remembered: _{fact[:200]}_"
        return _run_ask(
            f"Store this to long-term memory using memory_store. "
            f"Source: 'user:{uid}:manual'.\n\nFact: {fact}",
            uid, name,
        )

    await _slash_reply(interaction, _remember)


@tree.command(name="forget", description="Delete memories matching a query")
@discord.app_commands.describe(query="Search terms for the memories to delete")
async def slash_forget(interaction: discord.Interaction, query: str) -> None:
    uid = interaction.user.id

    def _forget() -> str:
        rows = _search_user_memory(query, uid)
        if not rows:
            return f"No memories found matching '{query}'."
        db = Path(MEMORY_DB_PATH)
        deleted = 0
        try:
            conn = sqlite3.connect(str(db), timeout=5)
            uid_filter = f"%user:{uid}%"
            cur = conn.cursor()
            for row in rows:
                content = row.get("content", "")
                try:
                    cur.execute(
                        "DELETE FROM memories WHERE content = ? AND (source LIKE ? OR metadata LIKE ?)",
                        (content, uid_filter, uid_filter),
                    )
                    deleted += cur.rowcount
                except Exception:
                    pass
            conn.commit()
            conn.close()
        except Exception as exc:
            return f"⚠️ Failed to delete memories: {exc}"
        return f"🗑️ Deleted {deleted} memor{'y' if deleted == 1 else 'ies'} matching '{query}'."

    await _slash_reply(interaction, _forget)


@tree.command(name="whoami", description="Show everything Jarvis knows about you from memory")
async def slash_whoami(interaction: discord.Interaction) -> None:
    uid, name = interaction.user.id, interaction.user.display_name

    def _whoami() -> str:
        all_rows = _search_user_memory("", uid)
        if not all_rows:
            return (
                f"I don't have any memories about you yet, {name}.\n"
                "Chat with me or use `/remember` to build your memory store."
            )
        lines = [f"**What I know about {name}** ({len(all_rows)} memories)"]
        for i, r in enumerate(all_rows[:15], 1):
            src = r.get("source", "")
            tag = src.split(":")[-1] if ":" in src else ""
            content = r.get("content", "")[:200]
            lines.append(f"**{i}.** {content}" + (f"\n   _[{tag}]_" if tag else ""))
        return "\n\n".join(lines)

    await _slash_reply(interaction, _whoami, thread_name=f"Memories: {name}")


# ── Skills & Tools ────────────────────────────────────────────────────────────

@tree.command(name="run-skill", description="Execute a registered Jarvis skill pipeline")
@discord.app_commands.describe(
    skill_name="Skill name (see /skills for list)",
    args="Optional key=value arguments",
)
async def slash_run_skill(interaction: discord.Interaction, skill_name: str, args: str = "") -> None:
    uid, name = interaction.user.id, interaction.user.display_name
    await _slash_reply(
        interaction,
        lambda: _run_ask(
            f"Run the Jarvis skill '{skill_name}'"
            + (f" with parameters: {args}" if args else "")
            + ". Use sensible defaults for missing parameters. Report the result clearly.",
            uid, name,
        ),
        thread_name=f"Skill: {skill_name}",
    )


@tree.command(name="skills", description="List available Jarvis skill pipelines")
async def slash_skills(interaction: discord.Interaction) -> None:
    def _skills() -> str:
        try:
            from openjarvis.skills.loader import SkillLoader
            skills = SkillLoader().list_skills()
            if skills:
                lines = ["**Jarvis Skills**", "```"]
                for s in sorted(skills, key=lambda x: x.get("name", "")):
                    lines.append(f"  {s.get('name','?'):<28} {s.get('description','')[:55]}")
                lines.append("```")
                return "\n".join(lines)
        except Exception:
            pass
        return (
            "**Jarvis Skills**\n```\n"
            "  deep-research            Multi-search synthesis with citations\n"
            "  morning-brief            Weather + AI story + daily insight\n"
            "  topic-research           Research + store to memory\n"
            "  web-summarize            Summarise a web page\n"
            "  pdf-summarize            Extract and summarise a PDF\n"
            "  code-lint                Lint and review code\n"
            "  code-test-gen            Generate unit tests from code\n"
            "  data-analyze             Analyse and interpret data\n"
            "  translate-doc            Translate a document\n"
            "  meeting-notes            Summarise meeting notes/transcripts\n"
            "```"
        )

    await _slash_reply(interaction, _skills)


@tree.command(name="tools", description="List all registered agent tools")
async def slash_tools(interaction: discord.Interaction) -> None:
    loop  = asyncio.get_event_loop()
    tools = await loop.run_in_executor(_executor, _discover_agent_tools)
    lines = ["**Registered Agent Tools**", "```"]
    for t in tools:
        lines.append(f"  {t}")
    lines.append("```")
    await interaction.response.send_message("\n".join(lines))


@tree.command(name="models", description="List Ollama models available locally")
async def slash_models(interaction: discord.Interaction) -> None:
    def _models() -> str:
        try:
            resp   = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            if not models:
                return f"No Ollama models found at `{OLLAMA_HOST}`."
            lines = [f"**Ollama models** ({len(models)} at `{OLLAMA_HOST}`)", "```"]
            for m in sorted(models, key=lambda x: x.get("name", "")):
                name_m   = m.get("name", "?")
                size     = m.get("size", 0)
                size_str = f"{size / 1e9:.1f}GB" if size > 1e8 else ""
                modified = (m.get("modified_at") or "")[:10]
                lines.append(f"  {name_m:<35} {size_str:<8} {modified}")
            lines.append("```")
            return "\n".join(lines)
        except Exception as exc:
            return f"⚠️ Could not reach Ollama at `{OLLAMA_HOST}`: {exc}"

    await _slash_reply(interaction, _models)


# ── System ────────────────────────────────────────────────────────────────────

@tree.command(name="status", description="Show engine, model, Gemini/Tavily config, and host stats")
async def slash_status(interaction: discord.Interaction) -> None:
    def _status() -> str:
        try:
            j      = _get_jarvis()
            cfg    = j._config  # type: ignore[attr-defined]
            model  = getattr(cfg.intelligence, "default_model", "?")
            engine = getattr(cfg.intelligence, "preferred_engine", "?")
        except Exception:
            model, engine = "?", "?"
        sys_info   = _call_tool("system_monitor") or "system_monitor unavailable"
        gemini_ok  = bool(os.environ.get("GEMINI_API_KEY"))
        tavily_ok  = bool(os.environ.get("TAVILY_API_KEY"))
        ollama_ok  = False
        try:
            r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
            ollama_ok = r.status_code == 200
        except Exception:
            pass
        return (
            f"**Jarvis Status**\n```\n"
            f"Local model : {model} ({engine})\n"
            f"Ollama      : {'✓ reachable' if ollama_ok else '✗ unreachable'} ({OLLAMA_HOST})\n"
            f"Gemini      : {'✓ configured' if gemini_ok else '✗ not set'}\n"
            f"Tavily      : {'✓ configured' if tavily_ok else '✗ not set (using DuckDuckGo)'}\n"
            f"Tools loaded: {len(AGENT_TOOLS)}\n"
            f"```\n{sys_info}"
        )

    await _slash_reply(interaction, _status)


@tree.command(name="health", description="System health: CPU/RAM/GPU + Ollama status")
async def slash_health(interaction: discord.Interaction) -> None:
    def _health() -> str:
        lines    = ["**System Health**"]
        sys_info = _call_tool("system_monitor", detail="full")
        if sys_info:
            lines.append(sys_info)
        try:
            resp   = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            models = resp.json().get("models", [])
            lines.append(f"Ollama: ✓ {len(models)} model(s) at `{OLLAMA_HOST}`")
        except Exception as exc:
            lines.append(f"Ollama: ✗ unreachable — `{exc}`")
        return "\n".join(lines)

    await _slash_reply(interaction, _health)


@tree.command(name="mcp", description="Show MCP server status")
async def slash_mcp(interaction: discord.Interaction) -> None:
    def _mcp() -> str:
        try:
            j       = _get_jarvis()
            cfg     = j._config  # type: ignore[attr-defined]
            mcp_cfg = getattr(getattr(cfg, "tools", None), "mcp", None)
            if not mcp_cfg or not getattr(mcp_cfg, "enabled", False):
                return "MCP is disabled. Enable with `[tools.mcp] enabled = true` in config."
            servers = getattr(mcp_cfg, "servers", []) or []
            if not servers:
                return "MCP is enabled but no servers are configured."
            lines = ["**MCP Servers**", "```"]
            for s in servers:
                name = getattr(s, "name", "?")
                cmd  = " ".join(getattr(s, "command", []))
                lines.append(f"  {name:<20} {cmd[:55]}")
            lines.append("```")
            return "\n".join(lines)
        except Exception as exc:
            return f"⚠️ MCP status error: {exc}"

    await _slash_reply(interaction, _mcp)


@tree.command(name="sysinfo", description="Detailed host CPU/RAM/GPU/disk stats")
async def slash_sysinfo(interaction: discord.Interaction) -> None:
    await _slash_reply(
        interaction,
        lambda: _call_tool("system_monitor", detail="full") or "system_monitor unavailable",
    )


# ── Scheduler ─────────────────────────────────────────────────────────────────

@tree.command(name="tasks", description="List all scheduled background tasks")
async def slash_tasks(interaction: discord.Interaction) -> None:
    def _list_tasks() -> str:
        if _scheduler is None:
            return "Scheduler is not running."
        tasks = _scheduler.list_tasks()
        if not tasks:
            return "No scheduled tasks. Use `/schedule` to create one."
        lines = ["**Scheduled Tasks**", "```"]
        for t in tasks:
            sched  = f"{t.schedule_type}:{t.schedule_value}"
            next_r = (t.next_run or "")[:16].replace("T", " ")
            icon   = {"active": "▶", "paused": "⏸", "cancelled": "✗", "completed": "✓"}.get(t.status, "?")
            lines.append(f"  {icon} [{t.id[:8]}] {sched:<28} next:{next_r}")
            lines.append(f"     {t.prompt[:70]}")
        lines.append("```")
        return "\n".join(lines)

    await _slash_reply(interaction, _list_tasks)


@tree.command(name="schedule", description="Create a recurring background task")
@discord.app_commands.describe(
    type="cron (e.g. 0 9 * * *) or interval (seconds)",
    value="Cron expression or number of seconds",
    prompt="What the agent should do each run",
)
@discord.app_commands.choices(type=[
    _Choice(name="cron",     value="cron"),
    _Choice(name="interval", value="interval"),
])
async def slash_schedule(interaction: discord.Interaction, type: str, value: str, prompt: str) -> None:
    if _scheduler is None:
        await interaction.response.send_message("Scheduler is not running.", ephemeral=True)
        return

    def _create() -> str:
        try:
            t = _scheduler.create_task(  # type: ignore[union-attr]
                prompt=prompt,
                schedule_type=type,
                schedule_value=value,
                agent="orchestrator",
                tools=",".join(_BASELINE_TOOLS),
                metadata={"created_by": str(interaction.user.id)},
            )
            next_r = (t.next_run or "")[:16].replace("T", " ")
            return (
                f"Task created: `{t.id[:8]}`\n"
                f"Schedule: `{type}:{value}`\n"
                f"Next run: `{next_r} UTC`\n"
                f"Prompt: {prompt[:100]}\n"
                f"Cancel: `/cancel-task id:{t.id[:8]}`"
            )
        except Exception as exc:
            return f"⚠️ Failed to create task: {exc}"

    await _slash_reply(interaction, _create)


@tree.command(name="pause-task", description="Pause a scheduled task temporarily")
@discord.app_commands.describe(id="Task ID prefix (from /tasks)")
async def slash_pause_task(interaction: discord.Interaction, id: str) -> None:
    def _pause() -> str:
        t, err = _resolve_task(id)
        if err:
            return err
        try:
            _scheduler.pause_task(t.id)  # type: ignore[union-attr]
            return f"Task `{t.id[:8]}` paused. Resume with `/resume-task id:{t.id[:8]}`"
        except Exception as exc:
            return f"⚠️ {exc}"

    await _slash_reply(interaction, _pause)


@tree.command(name="resume-task", description="Resume a paused scheduled task")
@discord.app_commands.describe(id="Task ID prefix (from /tasks)")
async def slash_resume_task(interaction: discord.Interaction, id: str) -> None:
    def _resume() -> str:
        t, err = _resolve_task(id)
        if err:
            return err
        try:
            _scheduler.resume_task(t.id)  # type: ignore[union-attr]
            updated = [x for x in _scheduler.list_tasks() if x.id == t.id]  # type: ignore[union-attr]
            next_r  = (updated[0].next_run if updated else "?")[:16].replace("T", " ")
            return f"Task `{t.id[:8]}` resumed. Next run: `{next_r} UTC`"
        except Exception as exc:
            return f"⚠️ {exc}"

    await _slash_reply(interaction, _resume)


@tree.command(name="cancel-task", description="Permanently cancel a scheduled task")
@discord.app_commands.describe(id="Task ID prefix (from /tasks)")
async def slash_cancel_task(interaction: discord.Interaction, id: str) -> None:
    def _cancel() -> str:
        t, err = _resolve_task(id)
        if err:
            return err
        try:
            _scheduler.cancel_task(t.id)  # type: ignore[union-attr]
            return f"Task `{t.id[:8]}` cancelled.\nWas: {t.prompt[:80]}"
        except Exception as exc:
            return f"⚠️ {exc}"

    await _slash_reply(interaction, _cancel)


@tree.command(name="task-log", description="Show recent run history for a scheduled task")
@discord.app_commands.describe(id="Task ID prefix (from /tasks)")
async def slash_task_log(interaction: discord.Interaction, id: str) -> None:
    def _logs() -> str:
        t, err = _resolve_task(id)
        if err:
            return err
        try:
            logs = _scheduler._store.get_run_logs(t.id, limit=5)  # type: ignore[union-attr]
            if not logs:
                return f"No run history for `{t.id[:8]}`."
            lines = [f"**Run log — `{t.id[:8]}`**", "```"]
            for run in logs:
                ok   = "✓" if run.get("success") else "✗"
                ts   = (run.get("started_at") or "")[:16].replace("T", " ")
                body = (run.get("result") or run.get("error") or "")[:120]
                lines.append(f"  {ok} {ts} UTC")
                if body:
                    lines.append(f"     {body}")
            lines.append("```")
            return "\n".join(lines)
        except Exception as exc:
            return f"⚠️ {exc}"

    await _slash_reply(interaction, _logs)


# ── Help ──────────────────────────────────────────────────────────────────────

@tree.command(name="help", description="Show all Jarvis commands and capabilities")
async def slash_help(interaction: discord.Interaction) -> None:
    lines = [
        "**Jarvis** — local AI assistant on OpenJarvis + qwen3:8b",
        "",
        "Just type in <#" + CHANNEL_NAME + "> or `@Jarvis <question>` — no command needed.",
        "",
        "**Conversation**",
        "`/ask <query>` — ask anything",
        "`/agent <task>` — explicit OrchestratorAgent run with all tools",
        "`/react <task>` — NativeReActAgent: shows Thought → Action → Observation trace",
        "`/ask-gemini <question>` — ask Google Gemini Flash directly",
        "",
        "**Research & Info**",
        "`/research <topic>` — multi-source synthesis, saved to memory",
        "`/deep-dive <topic>` — thorough 15-turn research with arXiv + web",
        "`/arxiv <query>` — search arXiv papers",
        "`/papers [topic]` — latest AI/ML arXiv papers",
        "`/summarize <url>` — fetch and summarise a web page",
        "`/pdf <url>` — extract and summarise a PDF",
        "`/weather [city]` — current conditions + forecast",
        "`/brief` — morning brief: weather + top AI story + insight",
        "`/digest` — on-demand HN AI digest",
        "`/trending` — what's hot in AI/tech right now",
        "",
        "**Knowledge**",
        "`/explain <concept>` — three-level explanation: simple / intermediate / expert",
        "`/compare <X vs Y>` — structured comparison with web research",
        "`/debate <topic>` — pros/cons analysis with evidence",
        "`/code <question>` — run Python or get a live coded example",
        "",
        "**Memory**",
        "`/memory <query>` — search your personal memory store",
        "`/remember <fact>` — store something explicitly",
        "`/forget <query>` — delete matching memories",
        "`/whoami` — show everything Jarvis knows about you",
        "",
        "**Skills & Tools**",
        "`/run-skill <name> [args]` — run a skill pipeline",
        "`/skills` — list available skill pipelines",
        "`/tools` — list all registered agent tools",
        "`/models` — list Ollama models available locally",
        "",
        "**Scheduler**",
        "`/schedule cron|interval <value> <prompt>` — create a background task",
        "`/tasks` — list scheduled tasks",
        "`/pause-task` · `/resume-task` · `/cancel-task` · `/task-log`",
        "",
        "**System**",
        "`/status` — model, Ollama, Gemini, Tavily, tools loaded",
        "`/health` — CPU/RAM/GPU + Ollama connectivity",
        "`/sysinfo` — detailed host stats",
        "`/mcp` — MCP server status",
        "",
        "*Stack: qwen3:8b (local) → Gemini Flash (stuck) → that's it, no cloud needed*",
    ]
    full = "\n".join(lines)
    if len(full) <= 1900:
        await interaction.response.send_message(full, ephemeral=True)
    else:
        mid = len(lines) // 2
        await interaction.response.send_message("\n".join(lines[:mid]), ephemeral=True)
        await interaction.followup.send("\n".join(lines[mid:]), ephemeral=True)


# ── Events ────────────────────────────────────────────────────────────────────

@client.event
async def on_ready() -> None:
    log.info(
        "Jarvis bot online: %s (id=%s) | #%s | tools=%d",
        client.user, getattr(client.user, "id", "?"),
        CHANNEL_NAME, len(AGENT_TOOLS),
    )
    if DISCORD_GUILD_ID:
        guild_obj = discord.Object(id=DISCORD_GUILD_ID)
        # Push current tree to guild first
        try:
            tree.copy_global_to(guild=guild_obj)
            synced = await tree.sync(guild=guild_obj)
            log.info("Slash commands synced to guild %d (%d commands)", DISCORD_GUILD_ID, len(synced))
        except discord.Forbidden:
            log.warning(
                "Slash sync failed (403). Re-invite: "
                "https://discord.com/api/oauth2/authorize"
                "?client_id=%s&permissions=2147830336&scope=bot+applications.commands",
                getattr(client.application, "id", client.user.id if client.user else "?"),
            )
        except Exception as exc:
            log.warning("Slash sync failed: %s", exc)
        # Clear stale global commands
        try:
            tree.clear_commands(guild=None)
            await tree.sync()
            log.info("Stale global commands cleared")
        except Exception as exc:
            log.warning("Global clear failed (non-fatal): %s", exc)
    else:
        try:
            synced = await tree.sync()
            log.info("Slash commands synced globally (%d commands)", len(synced))
        except Exception as exc:
            log.warning("Slash sync failed: %s", exc)
    threading.Thread(target=_init_scheduler, daemon=True).start()


@client.event
async def on_message(message: discord.Message) -> None:
    if message.author == client.user or message.author.bot:
        return

    query = extract_query(message)
    if query is None:
        return

    user_id      = message.author.id
    display_name = message.author.display_name
    loop         = asyncio.get_event_loop()

    log.info("[%s] %s (id=%d): %s", message.channel, display_name, user_id, query[:100])

    thread_ctx = ""
    if isinstance(message.channel, discord.Thread):
        thread_ctx = await build_thread_context(message.channel, message.id)

    async with message.channel.typing():
        reply = await loop.run_in_executor(
            _executor, _run_ask, query, user_id, display_name, thread_ctx
        )

    if isinstance(message.channel, discord.TextChannel) and len(reply) > MAX_MSG_LEN:
        try:
            thread = await message.create_thread(
                name=f"Jarvis: {query[:50]}", auto_archive_duration=60
            )
            await send_chunked(thread, reply)
            return
        except discord.Forbidden:
            pass

    await send_chunked(message.channel, reply)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client.run(DISCORD_BOT_TOKEN, log_handler=None)
