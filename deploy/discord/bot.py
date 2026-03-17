"""Jarvis Discord bot — extensible local-first AI assistant.

Design principles:
  1. Local models primary  — qwen3:8b via Ollama handles everything it can
  2. Gemini Flash mid-tier — called when the local agent gets stuck (via the
     gemini_escalate tool the agent has access to)
  3. ClaudeCode final tier — GitHub issue cut with Gemini's plan attached,
     worked asynchronously by the claude-worker container
  4. Plugin commands       — CommandRegistry lets you add new !commands as
     decorated async functions without touching the event loop
  5. Slash commands        — All major commands also registered as /slash-commands
     via discord.app_commands.CommandTree; synced to Discord on startup
  6. Dynamic tools         — AGENT_TOOLS is built from ToolRegistry at startup;
     add a tool to src/openjarvis/tools/ and it appears automatically
  7. MCP ready             — configure MCP servers in config.persistent.toml;
     their tools are auto-discovered and added to the agent's tool list
  8. Per-user memory       — every query is tagged with the Discord user ID so
     memories are namespaced per person across sessions
  9. Thread context        — replies in threads carry conversation history

Triggers:
  /slash-command anywhere — slash commands registered with Discord
  Any message in DISCORD_CHANNEL_NAME (default: #jarvis)
  @mention anywhere | DISCORD_PREFIX (default: !ask) in any channel
  Threads whose parent is the #jarvis channel

Adding commands:
  !prefix  — decorate with @cmd.exact / @cmd.prefix in this file
  /slash   — decorate with @tree.command() and add to the slash section below
To add a new tool: create src/openjarvis/tools/my_tool.py and import it
  in src/openjarvis/tools/__init__.py — it appears in !tools automatically.
To add MCP tools: add [[tools.mcp.servers]] to config.persistent.toml.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sqlite3
import threading
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import discord
import httpx

# Scheduler imports — optional, gracefully disabled if not available
try:
    from openjarvis.scheduler import SchedulerStore, TaskScheduler
    _SCHEDULER_AVAILABLE = True
except Exception:  # noqa: BLE001
    _SCHEDULER_AVAILABLE = False

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
# Optional: set to your server's ID for instant slash-command sync on startup.
# Leave blank to use global sync (takes up to 1 hour to propagate).
_GUILD_ID_STR     = os.environ.get("DISCORD_GUILD_ID", "")
DISCORD_GUILD_ID: int | None = int(_GUILD_ID_STR) if _GUILD_ID_STR.isdigit() else None
MAX_MSG_LEN       = 1990
THREAD_HISTORY    = 12

# ── Thread pool ───────────────────────────────────────────────────────────────

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jarvis-ask")

# ── Command registry ──────────────────────────────────────────────────────────

class _Entry:
    __slots__ = ("fn", "desc", "usage")
    def __init__(self, fn: Callable, desc: str, usage: str):
        self.fn    = fn
        self.desc  = desc
        self.usage = usage


class CommandRegistry:
    """
    Decorator-based Discord command dispatcher.

    Usage:
        @cmd.exact("!status", "show engine stats")
        async def _(message): ...

        @cmd.prefix("!weather", "!weather <city>", "weather forecast")
        async def _(message, arg): ...

    Adding a command requires only a decorated function — no changes to the
    event loop or dispatch table.
    """
    def __init__(self) -> None:
        self._exact:  dict[str, _Entry] = {}
        self._prefix: dict[str, _Entry] = {}

    def exact(self, command: str, description: str = "") -> Callable:
        """Register a handler that matches a message exactly."""
        def decorator(fn: Callable) -> Callable:
            self._exact[command.lower()] = _Entry(fn, description, command)
            return fn
        return decorator

    def prefix(self, command: str, usage: str = "", description: str = "") -> Callable:
        """Register a handler that matches messages starting with command."""
        def decorator(fn: Callable) -> Callable:
            self._prefix[command.lower()] = _Entry(fn, description, usage or command)
            return fn
        return decorator

    async def dispatch(
        self, message: discord.Message, content: str, lower: str
    ) -> bool:
        """Try to dispatch message to a registered command. Returns True if handled."""
        if lower in self._exact:
            await self._exact[lower].fn(message)
            return True
        for key, entry in self._prefix.items():
            if lower.startswith(key):
                arg = content[len(key):].strip()
                await entry.fn(message, arg)
                return True
        return False

    def help_lines(self) -> list[tuple[str, str]]:
        """Return (usage, description) pairs for all registered commands."""
        lines = [(e.usage, e.desc) for e in self._exact.values()]
        lines += [(e.usage, e.desc) for e in self._prefix.values()]
        return sorted(lines, key=lambda x: x[0])


cmd = CommandRegistry()

# ── Jarvis SDK + dynamic tool discovery ──────────────────────────────────────

log.info("Initialising Jarvis SDK...")
from openjarvis import Jarvis  # noqa: E402

_jarvis: Jarvis | None = None
_jarvis_lock = threading.Lock()

# Baseline tools — always included even if registry discovery fails
_BASELINE_TOOLS = [
    "think", "calculator", "retrieval", "web_search",
    "url_fetch", "arxiv_search", "weather", "system_monitor",
    "file_read", "gemini_escalate", "github_issue",
]

# Extra tools from AGENT_EXTRA_TOOLS env var (comma-separated) let operators
# add tools without rebuilding the image:  AGENT_EXTRA_TOOLS=my_tool,another_tool
_EXTRA_TOOLS = [
    t.strip() for t in os.environ.get("AGENT_EXTRA_TOOLS", "").split(",") if t.strip()
]


def _discover_agent_tools() -> list[str]:
    """
    Build the agent tool list by unioning:
      1. Baseline tools (hardcoded safe set)
      2. Extra tools from AGENT_EXTRA_TOOLS env var
      3. All tools currently registered in ToolRegistry

    Any tool in the registry but not in the baseline/extra lists is still
    included — this means MCP adapter tools and any future tools appear
    automatically once registered.
    """
    try:
        from openjarvis.core.registry import ToolRegistry
        registered = set(ToolRegistry.keys())
    except Exception:
        registered = set()

    combined = set(_BASELINE_TOOLS) | set(_EXTRA_TOOLS) | registered
    # Remove known-dangerous tools from the agent's reach in Discord context
    _BLOCKED = {"shell_exec", "code_interpreter_docker", "repl"}
    return sorted(combined - _BLOCKED)


def _get_jarvis() -> Jarvis:
    global _jarvis
    with _jarvis_lock:
        if _jarvis is None:
            _jarvis = Jarvis(config_path="/root/.openjarvis/config.toml")
            log.info("Jarvis SDK ready")
    return _jarvis


# Pre-warm
threading.Thread(target=_get_jarvis, daemon=True).start()

# Discover tools at startup (re-evaluated each time _discover_agent_tools is called)
AGENT_TOOLS: list[str] = _BASELINE_TOOLS  # set properly after SDK warms up


def _refresh_agent_tools() -> None:
    global AGENT_TOOLS
    AGENT_TOOLS = _discover_agent_tools()
    log.info("Agent tools: %s", ", ".join(AGENT_TOOLS))


threading.Thread(target=_refresh_agent_tools, daemon=True).start()

# ── Task Scheduler ────────────────────────────────────────────────────────────

SCHEDULER_DB_PATH = os.environ.get("SCHEDULER_DB_PATH", "/data/scheduler.db")

_scheduler: "TaskScheduler | None" = None


DISCORD_WEBHOOK = os.environ.get("DISCORD_DIGEST_WEBHOOK", "")


def _post_task_result_to_discord(payload: dict) -> None:
    """EventBus subscriber — posts scheduler task results to the Discord webhook."""
    if not DISCORD_WEBHOOK:
        return
    task_id  = payload.get("task_id", "?")[:8]
    success  = payload.get("success", False)
    result   = (payload.get("result") or "").strip()
    error    = (payload.get("error") or "").strip()

    if not result and not error:
        return  # Nothing worth posting

    icon    = "✅" if success else "❌"
    content = result[:1800] if success else f"Task failed:\n```\n{error[:500]}\n```"
    body    = f"{icon} **Scheduled task `{task_id}` complete**\n\n{content}"

    try:
        httpx.post(DISCORD_WEBHOOK, json={"content": body}, timeout=10)
    except Exception as exc:
        log.debug("Webhook post failed for task %s: %s", task_id, exc)


def _init_scheduler() -> None:
    """Start the TaskScheduler, wire EventBus posting, and seed default tasks."""
    global _scheduler
    if not _SCHEDULER_AVAILABLE:
        log.warning("TaskScheduler not available — skipping scheduler init")
        return

    try:
        # Try to get the EventBus from the Jarvis instance for event-driven posting
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

        # Seed default tasks — deduplicate by seed_id in metadata
        existing = {t.metadata.get("seed_id") for t in _scheduler.list_tasks()}

        def _seed(seed_id: str, **kwargs: Any) -> None:
            if seed_id not in existing:
                meta = kwargs.pop("metadata", {})
                meta["seed_id"] = seed_id
                _scheduler.create_task(metadata=meta, **kwargs)  # type: ignore[union-attr]
                log.info("Seeded scheduled task: %s", seed_id)

        # Daily morning brief at 09:00 UTC
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
            metadata={
                "system_prompt": (
                    "You are producing a concise morning brief. Be direct and informative. "
                    "Lead with weather, follow with one key AI story, close with one insight."
                ),
            },
        )

        # Weekly AI research digest — Monday 10:00 UTC
        _seed(
            "weekly-ai-digest",
            prompt=(
                "Research and synthesise the most significant AI/ML developments from the past week. "
                "Use web_search and arxiv_search. Produce a structured digest: breakthroughs, "
                "notable papers, tooling updates, and one practical takeaway. "
                "Store tagged 'weekly_digest:ai' in memory."
            ),
            schedule_type="cron",
            schedule_value="0 10 * * 1",
            agent="orchestrator",
            tools="web_search,arxiv_search,think,retrieval",
            metadata={
                "system_prompt": (
                    "You are producing a weekly AI research digest. "
                    "Be comprehensive but concise. Cite sources where possible."
                ),
            },
        )

    except Exception as exc:
        log.error("Scheduler init failed: %s", exc)


# ── Discord client ────────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True
intents.reactions       = True
client = discord.Client(intents=intents)
tree   = discord.app_commands.CommandTree(client)

# ── Per-user memory ───────────────────────────────────────────────────────────

def _user_prefix(user_id: int, display_name: str) -> str:
    return (
        f"[Discord User: {display_name} | ID: {user_id}]\n"
        f"Tag any memory stores with 'user:{user_id}:' in the metadata/source "
        f"field so they can be retrieved specifically for this user.\n\n"
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
            cur.execute(
                "SELECT content, source FROM memories_fts "
                "WHERE memories_fts MATCH ? AND (source LIKE ? OR metadata LIKE ?) LIMIT 8",
                (query, uid_filter, uid_filter),
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

def _run_ask(
    query: str,
    user_id: int | None = None,
    display_name: str = "User",
    thread_history: str = "",
) -> str:
    """Synchronous Jarvis call in thread pool. Includes user prefix + thread context."""
    j = _get_jarvis()

    parts: list[str] = []
    if user_id is not None:
        parts.append(_user_prefix(user_id, display_name))
    if thread_history:
        parts.append(
            f"[Thread conversation history — use for context]\n{thread_history}\n\n"
            f"[Current message from {display_name}]\n"
        )
    parts.append(query)
    full_query = "".join(parts)

    try:
        result = j.ask(full_query, agent="orchestrator", tools=AGENT_TOOLS)
        if isinstance(result, dict):
            return result.get("content") or "(no response)"
        return str(result) or "(no response)"
    except Exception as exc:
        log.exception("Jarvis.ask failed: %s", exc)
        return (
            f"⚠️ Jarvis encountered an error: `{exc}`\n"
            "The local agent will open a GitHub issue if this is a recurring problem."
        )

# ── Direct tool shortcuts (bypass agent for speed-sensitive commands) ─────────

def _call_tool(name: str, **kwargs: Any) -> str | None:
    """Call a registered tool directly. Returns content string or None on failure."""
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

# ── GitHub helpers (reaction approval) ───────────────────────────────────────

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
            headers=_gh_headers(), json={"labels": list(current)}, timeout=10,
        )
    except Exception as exc:
        log.warning("Label add failed for #%d: %s", issue_number, exc)


def _close_issue(issue_number: int) -> None:
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return
    try:
        httpx.patch(
            f"https://api.github.com/repos/{GITHUB_REPO}/issues/{issue_number}",
            headers=_gh_headers(), json={"state": "closed"}, timeout=10,
        )
    except Exception as exc:
        log.warning("Issue close failed for #%d: %s", issue_number, exc)


def _extract_issue_number(text: str) -> int | None:
    m = re.search(r"github\.com/[^/\s]+/[^/\s]+/issues/(\d+)", text)
    return int(m.group(1)) if m else None

# ── Discord helpers ───────────────────────────────────────────────────────────

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
    """Send a long reply, creating a thread if in a text channel."""
    if (
        isinstance(message.channel, discord.TextChannel)
        and len(text) > MAX_MSG_LEN
    ):
        try:
            thread = await message.create_thread(
                name=thread_name[:99], auto_archive_duration=60
            )
            await send_chunked(thread, text)
            return
        except discord.Forbidden:
            pass
    await send_chunked(message.channel, text)


# ── Slash-command helper ─────────────────────────────────────────────────────

async def _slash_reply(
    interaction: discord.Interaction,
    fn: "Callable[[], str]",
    thread_name: str = "",
) -> None:
    """Defer an interaction, run fn in the thread pool, and send the reply.

    For long replies in a text channel, creates a thread to avoid flooding.
    Always safe to call regardless of reply length.
    """
    await interaction.response.defer(thinking=True)
    loop  = asyncio.get_event_loop()
    reply = await loop.run_in_executor(_executor, fn)

    if (
        thread_name
        and len(reply) > MAX_MSG_LEN
        and isinstance(interaction.channel, discord.TextChannel)
    ):
        try:
            seed = await interaction.followup.send(f"*{thread_name}*", wait=True)
            thread = await seed.create_thread(
                name=thread_name[:99], auto_archive_duration=60
            )
            await send_chunked(thread, reply)
            return
        except discord.Forbidden:
            pass

    # Chunked plain reply
    chunks = [reply[i : i + MAX_MSG_LEN] for i in range(0, max(len(reply), 1), MAX_MSG_LEN)]
    for chunk in chunks:
        await interaction.followup.send(chunk)


# ── Registered commands ───────────────────────────────────────────────────────

@cmd.exact("!status", "engine, model, and host stats")
async def _(message: discord.Message) -> None:
    loop = asyncio.get_event_loop()

    def _status() -> str:
        try:
            j   = _get_jarvis()
            cfg = j._config  # type: ignore[attr-defined]
            model  = getattr(cfg.intelligence, "default_model", "?")
            engine = getattr(cfg.intelligence, "preferred_engine", "?")
        except Exception:
            model, engine = "?", "?"
        sys_info = _call_tool("system_monitor") or "system_monitor unavailable"
        gemini_ok = bool(os.environ.get("GEMINI_API_KEY"))
        github_ok = bool(GITHUB_TOKEN and GITHUB_REPO)
        return (
            f"**Jarvis Status**\n```\n"
            f"Local model : {model} ({engine})\n"
            f"Gemini      : {'✓ configured' if gemini_ok else '✗ not set'}\n"
            f"GitHub      : {'✓ ' + GITHUB_REPO if github_ok else '✗ not set'}\n"
            f"Tools loaded: {len(AGENT_TOOLS)}\n"
            f"```\n{sys_info}"
        )

    async with message.channel.typing():
        reply = await loop.run_in_executor(_executor, _status)
    await send_chunked(message.channel, reply)


@cmd.exact("!help", "show all commands")
async def _(message: discord.Message) -> None:
    lines = ["**Jarvis commands**", "```"]
    lines.append(f"  {'In #' + CHANNEL_NAME:<26} just type — no prefix needed")
    lines.append(f"  {'@Jarvis <query>':<26} mention anywhere")
    lines.append(f"  {COMMAND_PREFIX + ' <query>':<26} explicit prefix")
    lines.append("")
    for usage, desc in cmd.help_lines():
        if desc:
            lines.append(f"  {usage:<26} {desc}")
    lines.append("```")
    lines.append(
        "Local model primary · Gemini Flash mid-tier · Claude Code final escalation.\n"
        "React **👍** to approve proposals · **👎** to withdraw."
    )
    await message.channel.send("\n".join(lines))


@cmd.exact("!tools", "list all registered agent tools")
async def _(message: discord.Message) -> None:
    loop  = asyncio.get_event_loop()
    tools = await loop.run_in_executor(_executor, _discover_agent_tools)
    lines = ["**Registered Agent Tools**", "```"]
    for t in tools:
        lines.append(f"  {t}")
    lines.append("```")
    lines.append(
        "*Add a tool: create `src/openjarvis/tools/my_tool.py`, register with "
        "`@ToolRegistry.register(\"name\")`, import in `tools/__init__.py`.*"
    )
    await send_chunked(message.channel, "\n".join(lines))


@cmd.exact("!mcp", "show MCP server status")
async def _(message: discord.Message) -> None:
    """List configured MCP servers and whether they're reachable."""
    try:
        j   = _get_jarvis()
        cfg = j._config  # type: ignore[attr-defined]
        mcp_cfg = getattr(cfg, "tools", None)
        mcp_cfg = getattr(mcp_cfg, "mcp", None) if mcp_cfg else None
        enabled = getattr(mcp_cfg, "enabled", False) if mcp_cfg else False
        servers = getattr(mcp_cfg, "servers", []) if mcp_cfg else []
    except Exception:
        enabled, servers = False, []

    if not enabled:
        await message.channel.send(
            "MCP is disabled. To enable: set `[tools.mcp] enabled = true` in "
            "`config.persistent.toml` and add `[[tools.mcp.servers]]` entries."
        )
        return

    if not servers:
        await message.channel.send(
            "MCP is enabled but no servers are configured.\n"
            "Add servers to `config.persistent.toml`:\n"
            "```toml\n[[tools.mcp.servers]]\n"
            "name = \"filesystem\"\n"
            "command = [\"npx\", \"-y\", \"@modelcontextprotocol/server-filesystem\", \"/data\"]\n```"
        )
        return

    lines = ["**MCP Servers**", "```"]
    for s in servers:
        name = getattr(s, "name", str(s))
        lines.append(f"  {name}")
    lines.append("```")
    await message.channel.send("\n".join(lines))


@cmd.exact("!skills", "list available Jarvis skills")
async def _(message: discord.Message) -> None:
    loop = asyncio.get_event_loop()

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
            "  code-lint                Lint and review code\n"
            "  data-analyze             Analyse and interpret data\n"
            "```"
        )

    async with message.channel.typing():
        reply = await loop.run_in_executor(_executor, _skills)
    await send_chunked(message.channel, reply)


@cmd.exact("!sysinfo", "host CPU/RAM/GPU stats (summary)")
async def _(message: discord.Message) -> None:
    loop = asyncio.get_event_loop()
    async with message.channel.typing():
        reply = await loop.run_in_executor(
            _executor, lambda: _call_tool("system_monitor") or "system_monitor unavailable"
        )
    await send_chunked(message.channel, reply)


@cmd.exact("!sysinfo full", "host CPU/RAM/GPU stats (detailed)")
async def _(message: discord.Message) -> None:
    loop = asyncio.get_event_loop()
    async with message.channel.typing():
        reply = await loop.run_in_executor(
            _executor,
            lambda: _call_tool("system_monitor", detail="full") or "system_monitor unavailable",
        )
    await send_chunked(message.channel, reply)


@cmd.prefix("!weather", "!weather [city]", "current conditions + 3-day forecast")
async def _(message: discord.Message, city: str) -> None:
    city = city or DEFAULT_CITY
    if not city:
        await message.channel.send("Usage: `!weather <city>` — e.g. `!weather London`")
        return
    loop = asyncio.get_event_loop()
    async with message.channel.typing():
        reply = await loop.run_in_executor(
            _executor,
            lambda: _call_tool("weather", city=city) or _run_ask(f"Weather in {city}?"),
        )
    await send_chunked(message.channel, reply)


@cmd.prefix("!arxiv ", "!arxiv <query>", "search recent arXiv papers")
async def _(message: discord.Message, query: str) -> None:
    if not query:
        await message.channel.send("Usage: `!arxiv <search query>`")
        return
    loop = asyncio.get_event_loop()
    async with message.channel.typing():
        reply = await loop.run_in_executor(
            _executor,
            lambda: _call_tool("arxiv_search", query=query, max_results=8)
                    or _run_ask(f"Search arXiv for: {query}"),
        )
    await reply_long(message, reply, f"arXiv: {query[:50]}")


@cmd.prefix("!summarize ", "!summarize <url>", "fetch and summarise a web page")
async def _(message: discord.Message, url: str) -> None:
    if not url:
        await message.channel.send("Usage: `!summarize <url>`")
        return

    loop = asyncio.get_event_loop()

    def _summarize() -> str:
        raw = _call_tool("url_fetch", url=url, max_chars=6000)
        if raw:
            return _run_ask(
                f"Summarise this web page concisely. Extract key points and main argument.\n\n"
                f"URL: {url}\n\n{raw}"
            )
        return _run_ask(
            f"Fetch and summarise this URL using the url_fetch tool: {url}"
        )

    async with message.channel.typing():
        reply = await loop.run_in_executor(_executor, _summarize)
    await reply_long(message, reply, f"Summary: {url[:60]}")


@cmd.prefix("!research ", "!research <topic>", "multi-source research synthesis")
async def _(message: discord.Message, topic: str) -> None:
    if not topic:
        await message.channel.send("Usage: `!research <topic>`")
        return
    user_id      = message.author.id
    display_name = message.author.display_name
    loop         = asyncio.get_event_loop()

    def _research() -> str:
        return _run_ask(
            f"Research '{topic}' thoroughly.\n"
            "1. Use web_search (at least 2 angles).\n"
            "2. Use arxiv_search if it's technical/scientific.\n"
            "3. Check retrieval for prior memory context.\n"
            "4. Synthesise a structured report: Overview, Key Concepts, "
            "Recent Developments, Practical Applications, Key Takeaways.\n"
            f"5. Store findings tagged with 'user:{user_id}:research:{topic[:30]}'.",
            user_id, display_name,
        )

    async with message.channel.typing():
        reply = await loop.run_in_executor(_executor, _research)
    await reply_long(message, reply, f"Research: {topic[:50]}")


@cmd.prefix("!memory ", "!memory <query>", "search your personal memory store")
async def _(message: discord.Message, query: str) -> None:
    if not query:
        await message.channel.send("Usage: `!memory <search terms>`")
        return
    user_id = message.author.id
    loop    = asyncio.get_event_loop()

    def _mem() -> str:
        rows = _search_user_memory(query, user_id)
        if not rows:
            return (
                f"No memories found for '{query}'.\n"
                "Chat with Jarvis and it will build up your memory store over time."
            )
        lines = [f"**Memory results for '{query}'** ({len(rows)})\n"]
        for i, r in enumerate(rows, 1):
            lines.append(f"**{i}.** {r.get('content','')[:280]}")
        return "\n\n".join(lines)

    async with message.channel.typing():
        reply = await loop.run_in_executor(_executor, _mem)
    await send_chunked(message.channel, reply)


@cmd.exact("!digest now", "on-demand HN AI digest")
async def _(message: discord.Message) -> None:
    await message.channel.send("Fetching HN AI digest…")
    loop = asyncio.get_event_loop()

    def _digest() -> str:
        return _run_ask(
            "Fetch the top 10 most-discussed AI/ML stories on Hacker News from "
            "the last 72 hours. Use web_search. For each: rank, title (as a link), "
            "comment count, score, one-sentence description. "
            "Format as a numbered Discord-friendly list."
        )

    async with message.channel.typing():
        reply = await loop.run_in_executor(_executor, _digest)
    await reply_long(message, reply, "HN AI Digest (on-demand)")


@cmd.prefix("!propose ", "!propose <idea>", "submit a feature proposal to GitHub")
async def _(message: discord.Message, idea: str) -> None:
    if not idea:
        await message.channel.send("Usage: `!propose <your feature idea>`")
        return
    if not GITHUB_TOKEN or not GITHUB_REPO:
        await message.channel.send(
            "⚠️ GITHUB_TOKEN and GITHUB_REPO must be set to submit proposals."
        )
        return

    user_id      = message.author.id
    display_name = message.author.display_name
    await message.channel.send(
        "Analysing proposal, consulting Gemini if needed, and opening a GitHub issue…\n"
        "React **👍** to approve it for Claude's queue, or **👎** to withdraw."
    )
    loop = asyncio.get_event_loop()

    def _propose() -> str:
        return _run_ask(
            f"Discord user {display_name} proposed this feature:\n\n```\n{idea}\n```\n\n"
            "Steps:\n"
            "1. Use gemini_escalate (mode=plan) to get a structured implementation plan "
            "   for this feature. Pass the idea as the problem.\n"
            "2. Use github_issue to submit the issue with:\n"
            "   - A clear title (< 80 chars)\n"
            "   - Gemini's implementation plan as the body\n"
            "   - Acceptance criteria\n"
            "3. Return the issue URL so the user can track and approve it.",
            user_id, display_name,
        )

    async with message.channel.typing():
        reply = await loop.run_in_executor(_executor, _propose)
    await send_chunked(message.channel, reply)


# ── Events ────────────────────────────────────────────────────────────────────

@client.event
async def on_ready() -> None:
    log.info(
        "Jarvis bot online: %s (id=%s) | #%s | prefix=%s | tools=%d",
        client.user, getattr(client.user, "id", "?"),
        CHANNEL_NAME, COMMAND_PREFIX, len(AGENT_TOOLS),
    )
    # Sync slash commands — guild sync is instant, global takes ~1 hour
    try:
        if DISCORD_GUILD_ID:
            guild_obj = discord.Object(id=DISCORD_GUILD_ID)
            tree.copy_global_to(guild=guild_obj)
            await tree.sync(guild=guild_obj)
            log.info("Slash commands synced to guild %d", DISCORD_GUILD_ID)
        await tree.sync()
        log.info("Slash commands synced globally")
    except Exception as exc:
        log.warning("Slash command sync failed: %s", exc)
    # Start scheduler in background thread after bot is ready
    threading.Thread(target=_init_scheduler, daemon=True).start()


@client.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent) -> None:
    """👍 approves a proposal issue, 👎 closes it."""
    if client.user and payload.user_id == client.user.id:
        return
    emoji = str(payload.emoji)
    if emoji not in ("👍", "👎"):
        return

    try:
        channel = client.get_channel(payload.channel_id) or \
                  await client.fetch_channel(payload.channel_id)
        message = await channel.fetch_message(payload.message_id)
    except Exception:
        return

    if not client.user or message.author.id != client.user.id:
        return

    issue_number = _extract_issue_number(message.content)
    if issue_number is None:
        return

    loop = asyncio.get_event_loop()
    if emoji == "👍":
        await loop.run_in_executor(_executor, _add_work_label, issue_number)
        await channel.send(
            f"✅ Issue #{issue_number} approved and added to Claude's work queue."
        )
    else:
        await loop.run_in_executor(_executor, _close_issue, issue_number)
        await channel.send(f"❌ Issue #{issue_number} withdrawn and closed.")


@client.event
async def on_message(message: discord.Message) -> None:
    if message.author == client.user or message.author.bot:
        return

    content = message.content.strip()
    lower   = content.lower()

    # Try registered commands first
    if await cmd.dispatch(message, content, lower):
        return

    # Regular Jarvis query
    query = extract_query(message)
    if query is None:
        return

    user_id      = message.author.id
    display_name = message.author.display_name
    loop         = asyncio.get_event_loop()

    log.info("[%s] %s (id=%d): %s", message.channel, display_name, user_id, query[:100])

    # Thread context
    thread_ctx = ""
    if isinstance(message.channel, discord.Thread):
        thread_ctx = await build_thread_context(message.channel, message.id)

    async with message.channel.typing():
        reply = await loop.run_in_executor(
            _executor, _run_ask, query, user_id, display_name, thread_ctx
        )

    # Long non-thread replies → create a thread
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


# ── GitHub workflow commands ──────────────────────────────────────────────────

def _gh_api(path: str, method: str = "GET", json: dict | None = None) -> dict | list | None:
    """Minimal GitHub REST helper — returns parsed JSON or None on error."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return None
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url = f"https://api.github.com/repos/{GITHUB_REPO}/{path}"
    try:
        resp = httpx.request(method, url, headers=headers, json=json, timeout=15)
        if resp.status_code in (200, 201):
            return resp.json()
        return {"_error": resp.status_code, "_text": resp.text[:300]}
    except Exception as exc:
        return {"_error": str(exc)}


@cmd.prefix("!merge ", "!merge <pr#>", "merge a PR (squash + delete branch)")
async def _(message: discord.Message, arg: str) -> None:
    if not arg.isdigit():
        await message.channel.send("Usage: `!merge <pr_number>` — e.g. `!merge 42`")
        return
    pr_number = int(arg)
    loop      = asyncio.get_event_loop()

    def _do_merge() -> str:
        # Get PR details
        pr = _gh_api(f"pulls/{pr_number}")
        if not pr or "_error" in pr:
            return f"⚠️ Could not fetch PR #{pr_number}: {pr}"
        state = pr.get("state", "")
        if state != "open":
            return f"PR #{pr_number} is `{state}` — nothing to merge."
        mergeable = pr.get("mergeable")
        if mergeable is False:
            return (
                f"⚠️ PR #{pr_number} has merge conflicts.\n"
                "Run `!ask-gemini <conflict details>` to get a resolution plan."
            )
        title    = pr.get("title", "")
        head_ref = pr.get("head", {}).get("ref", "")
        pr_url   = pr.get("html_url", "")

        merge_data = _gh_api(
            f"pulls/{pr_number}/merge",
            method="PUT",
            json={"merge_method": "squash", "commit_title": f"[JarvisPanda] {title} (#{pr_number})"},
        )
        if merge_data and "_error" not in merge_data:
            sha = (merge_data.get("sha") or "")[:8]
            # Delete branch
            if head_ref:
                _gh_api(f"git/refs/heads/{head_ref}", method="DELETE")
            return f"✅ PR #{pr_number} merged (squash) → `{sha}`\n{pr_url}\nBranch `{head_ref}` deleted."
        return f"⚠️ Merge failed: {merge_data}"

    async with message.channel.typing():
        reply = await loop.run_in_executor(_executor, _do_merge)
    await send_chunked(message.channel, reply)


@cmd.exact("!prs", "list open pull requests")
async def _(message: discord.Message) -> None:
    loop = asyncio.get_event_loop()

    def _prs() -> str:
        data = _gh_api("pulls?state=open&per_page=15")
        if not data or "_error" in data:
            return f"⚠️ Could not fetch PRs: {data}"
        if not data:
            return "No open pull requests."
        lines = ["**Open Pull Requests**"]
        for pr in data:
            n     = pr["number"]
            title = pr["title"][:60]
            url   = pr["html_url"]
            user  = pr.get("user", {}).get("login", "?")
            lines.append(f"• **[#{n}]({url})** {title} — `{user}`  →  `!merge {n}`")
        return "\n".join(lines)

    async with message.channel.typing():
        reply = await loop.run_in_executor(_executor, _prs)
    await send_chunked(message.channel, reply)


@cmd.exact("!queue", "show claude-code-work issue queue")
async def _(message: discord.Message) -> None:
    loop = asyncio.get_event_loop()

    def _queue() -> str:
        if not GITHUB_TOKEN or not GITHUB_REPO:
            return "⚠️ GITHUB_TOKEN / GITHUB_REPO not configured."
        lines = ["**Issue Worker Queue**"]
        for label, emoji in [
            ("claude-code-work",        "📥 queued"),
            ("claude-code-in-progress", "⚙️ in progress"),
            ("claude-code-done",        "✅ done (unmerged)"),
            ("claude-code-failed",      "❌ failed"),
        ]:
            data = _gh_api(f"issues?labels={label}&state=open&per_page=10")
            if not data or "_error" in data:
                lines.append(f"{emoji}: (error fetching)")
                continue
            if not data:
                lines.append(f"{emoji}: none")
            else:
                for issue in data[:5]:
                    n     = issue["number"]
                    title = issue["title"][:55]
                    url   = issue["html_url"]
                    lines.append(f"{emoji}: **[#{n}]({url})** {title}")
        return "\n".join(lines)

    async with message.channel.typing():
        reply = await loop.run_in_executor(_executor, _queue)
    await send_chunked(message.channel, reply)


@cmd.prefix("!pr ", "!pr <number>", "show PR details and what changed")
async def _(message: discord.Message, arg: str) -> None:
    if not arg.isdigit():
        await message.channel.send("Usage: `!pr <pr_number>`")
        return
    pr_number = int(arg)
    loop      = asyncio.get_event_loop()

    def _pr_detail() -> str:
        pr = _gh_api(f"pulls/{pr_number}")
        if not pr or "_error" in pr:
            return f"⚠️ Could not fetch PR #{pr_number}"
        title  = pr.get("title", "")
        state  = pr.get("state", "")
        url    = pr.get("html_url", "")
        body   = (pr.get("body") or "")[:600]
        branch = pr.get("head", {}).get("ref", "")
        user   = pr.get("user", {}).get("login", "?")
        files  = _gh_api(f"pulls/{pr_number}/files?per_page=20")
        file_lines = []
        if files and "_error" not in files:
            for f in files[:15]:
                fname = f.get("filename", "")
                status = f.get("status", "")
                adds  = f.get("additions", 0)
                dels  = f.get("deletions", 0)
                file_lines.append(f"  `{fname}` ({status} +{adds}/-{dels})")
        files_str = "\n".join(file_lines) or "  _(no file data)_"
        return (
            f"**PR #{pr_number}** — `{state}`\n"
            f"**{title}**\n{url}\n"
            f"Author: `{user}` · Branch: `{branch}`\n\n"
            f"**Files changed:**\n{files_str}\n\n"
            f"**Description:**\n{body}\n\n"
            f"Merge: `!merge {pr_number}`"
        )

    async with message.channel.typing():
        reply = await loop.run_in_executor(_executor, _pr_detail)
    await reply_long(message, reply, f"PR #{pr_number}")


# ── Content & research commands ───────────────────────────────────────────────

@cmd.exact("!brief", "morning brief: weather + AI news + insight")
async def _(message: discord.Message) -> None:
    user_id      = message.author.id
    display_name = message.author.display_name
    loop         = asyncio.get_event_loop()

    def _brief() -> str:
        city = DEFAULT_CITY or "New York"
        weather = _call_tool("weather", city=city) or ""
        return _run_ask(
            f"Give me a concise morning brief with three sections:\n"
            f"1. **Weather** — summarise this in 1-2 sentences: {weather[:300]}\n"
            f"2. **Top AI Story** — use web_search to find the most-discussed AI news today, "
            f"summarise in 3 sentences.\n"
            f"3. **Insight** — one thought-provoking question or insight relevant to AI/tech "
            f"I should think about today.\n\n"
            f"Keep it concise and useful.",
            user_id, display_name,
        )

    async with message.channel.typing():
        reply = await loop.run_in_executor(_executor, _brief)
    await reply_long(message, reply, "Morning Brief")


@cmd.prefix("!papers", "!papers [topic]", "latest arXiv AI/ML papers (or custom topic)")
async def _(message: discord.Message, topic: str) -> None:
    query = topic or "LLM AI agent reasoning alignment"
    loop  = asyncio.get_event_loop()

    async with message.channel.typing():
        reply = await loop.run_in_executor(
            _executor,
            lambda: _call_tool("arxiv_search", query=query, max_results=8, sort_by="submittedDate")
                    or _run_ask(f"Find the latest arXiv papers on: {query}"),
        )
    await reply_long(message, reply, f"arXiv: {query[:50]}")


@cmd.prefix("!ask-gemini ", "!ask-gemini <question>", "ask Gemini Flash directly")
async def _(message: discord.Message, question: str) -> None:
    if not question:
        await message.channel.send("Usage: `!ask-gemini <question>`")
        return
    loop = asyncio.get_event_loop()

    def _gemini() -> str:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return "⚠️ GEMINI_API_KEY is not configured."
        try:
            from openjarvis.tools.gemini_escalate import gemini_generate
            resp = gemini_generate(question)
            return f"**Gemini Flash** says:\n\n{resp}"
        except Exception as exc:
            return f"⚠️ Gemini call failed: {exc}"

    async with message.channel.typing():
        reply = await loop.run_in_executor(_executor, _gemini)
    await reply_long(message, reply, f"Gemini: {question[:50]}")


# ── Agent workflow commands ───────────────────────────────────────────────────

@cmd.prefix("!agent ", "!agent <task>", "run a one-shot local agent task")
async def _(message: discord.Message, task: str) -> None:
    if not task:
        await message.channel.send("Usage: `!agent <what you want the agent to do>`")
        return
    user_id      = message.author.id
    display_name = message.author.display_name
    loop         = asyncio.get_event_loop()

    async with message.channel.typing():
        reply = await loop.run_in_executor(
            _executor, _run_ask, task, user_id, display_name, ""
        )
    await reply_long(message, reply, f"Agent: {task[:50]}")


@cmd.prefix("!build-tool ", "!build-tool <description>", "spec, plan, and propose a new Jarvis tool")
async def _(message: discord.Message, description: str) -> None:
    if not description:
        await message.channel.send("Usage: `!build-tool <what the tool should do>`")
        return
    user_id      = message.author.id
    display_name = message.author.display_name

    await message.channel.send(
        f"Designing tool: _{description}_\n"
        "Consulting Gemini for a spec, then opening a GitHub issue…\n"
        "React **👍** to queue for Claude Code, **👎** to withdraw."
    )
    loop = asyncio.get_event_loop()

    def _build() -> str:
        return _run_ask(
            f"A user wants to build a new OpenJarvis tool:\n\n```\n{description}\n```\n\n"
            "Steps:\n"
            "1. Use gemini_escalate(mode=plan) to design a complete tool spec including:\n"
            "   - Tool name (snake_case)\n"
            "   - What it does and why it's useful\n"
            "   - Parameters schema (JSON Schema)\n"
            "   - Implementation approach (API used, dependencies)\n"
            "   - Example usage\n"
            "   - How it fits into the escalation chain\n"
            "2. Use github_issue to create an issue with:\n"
            "   - Title: 'feat: add {tool_name} tool'\n"
            "   - Body: Gemini's full spec + implementation plan\n"
            "   - The tool should follow BaseTool / ToolSpec patterns from existing tools\n"
            "3. Return the issue URL so the user can approve or withdraw it.\n\n"
            "Reference implementation: src/openjarvis/tools/weather.py is a clean example.",
            user_id, display_name,
        )

    async with message.channel.typing():
        reply = await loop.run_in_executor(_executor, _build)
    await send_chunked(message.channel, reply)


@cmd.prefix("!run-skill ", "!run-skill <name> [args]", "execute a registered Jarvis skill")
async def _(message: discord.Message, arg: str) -> None:
    if not arg:
        await message.channel.send(
            "Usage: `!run-skill <skill-name> [key=value ...]`\n"
            "Run `!skills` to see available skills."
        )
        return
    # Parse: first word is skill name, rest are args
    parts      = arg.split(None, 1)
    skill_name = parts[0]
    skill_args = parts[1] if len(parts) > 1 else ""
    user_id      = message.author.id
    display_name = message.author.display_name
    loop         = asyncio.get_event_loop()

    def _skill() -> str:
        return _run_ask(
            f"Run the Jarvis skill named '{skill_name}'"
            + (f" with these parameters: {skill_args}" if skill_args else "")
            + ".\n\nIf the skill requires parameters not provided, use sensible defaults "
            "or ask the user what they want. Report the result clearly.",
            user_id, display_name,
        )

    async with message.channel.typing():
        reply = await loop.run_in_executor(_executor, _skill)
    await reply_long(message, reply, f"Skill: {skill_name}")


@cmd.exact("!health", "on-demand system health check (same as observer)")
async def _(message: discord.Message) -> None:
    loop = asyncio.get_event_loop()

    def _health() -> str:
        lines = ["**On-demand Health Check**"]
        # System
        sys_info = _call_tool("system_monitor", detail="summary")
        if sys_info:
            lines.append(sys_info)
        # GitHub queue summary
        if GITHUB_TOKEN and GITHUB_REPO:
            for label, emoji in [
                ("claude-code-work", "📥 queued"),
                ("claude-code-in-progress", "⚙️ in progress"),
                ("claude-code-done", "✅ done (unmerged)"),
            ]:
                data = _gh_api(f"issues?labels={label}&state=open&per_page=5")
                count = len(data) if isinstance(data, list) else 0
                lines.append(f"{emoji}: {count}")
            prs = _gh_api("pulls?state=open&per_page=5")
            pr_count = len(prs) if isinstance(prs, list) else 0
            lines.append(f"🔀 open PRs: {pr_count}")
        return "\n".join(lines)

    async with message.channel.typing():
        reply = await loop.run_in_executor(_executor, _health)
    await send_chunked(message.channel, reply)


# ── Scheduler commands ────────────────────────────────────────────────────────

@cmd.exact("!tasks", "list all scheduled background tasks")
async def _(message: discord.Message) -> None:
    loop = asyncio.get_event_loop()

    def _list_tasks() -> str:
        if _scheduler is None:
            return "Scheduler is not running." + ("" if _SCHEDULER_AVAILABLE else " (openjarvis.scheduler unavailable)")
        tasks = _scheduler.list_tasks()
        if not tasks:
            return "No scheduled tasks. Use `!schedule` to create one."
        lines = ["**Scheduled Tasks**", "```"]
        for t in tasks:
            sched = f"{t.schedule_type}:{t.schedule_value}"
            next_r = (t.next_run or "")[:16].replace("T", " ")
            status_icon = {"active": "▶", "paused": "⏸", "cancelled": "✗", "completed": "✓"}.get(t.status, "?")
            lines.append(f"  {status_icon} [{t.id[:8]}] {sched:<28} next:{next_r}")
            lines.append(f"     {t.prompt[:70]}")
        lines.append("```")
        lines.append("Use `!cancel-task <id>` to cancel · `!task-log <id>` for run history")
        return "\n".join(lines)

    async with message.channel.typing():
        reply = await loop.run_in_executor(_executor, _list_tasks)
    await send_chunked(message.channel, reply)


@cmd.prefix("!schedule ", "!schedule cron|interval <value> <prompt>", "create a recurring background task")
async def _(message: discord.Message, arg: str) -> None:
    """
    Usage:
      !schedule cron 0 9 * * * Daily morning weather check
      !schedule interval 3600 Check HN for AI news and summarise
    """
    if not arg:
        await message.channel.send(
            "Usage: `!schedule <cron|interval> <value> <prompt>`\n"
            "Examples:\n"
            "  `!schedule cron 0 9 * * * Check weather and store in memory`\n"
            "  `!schedule interval 7200 Search arXiv for new LLM papers and store results`"
        )
        return
    if _scheduler is None:
        await message.channel.send("Scheduler is not running." + ("" if _SCHEDULER_AVAILABLE else " (install openjarvis[server] for scheduler support)"))
        return

    parts = arg.split(None, 1)
    if len(parts) < 2:
        await message.channel.send("Usage: `!schedule <cron|interval> <value> <prompt>`")
        return

    sched_type = parts[0].lower()
    rest = parts[1]

    if sched_type not in ("cron", "interval"):
        await message.channel.send("Schedule type must be `cron` or `interval`.")
        return

    if sched_type == "cron":
        # cron expressions have 5 parts: split the first 5 words as the expression
        cron_parts = rest.split(None, 5)
        if len(cron_parts) < 6:
            await message.channel.send("Cron format: `!schedule cron <min> <hr> <dom> <mon> <dow> <prompt>`\nExample: `!schedule cron 0 9 * * * Do morning research`")
            return
        schedule_value = " ".join(cron_parts[:5])
        prompt = cron_parts[5]
    else:
        # interval: first word is seconds, rest is prompt
        iv_parts = rest.split(None, 1)
        if len(iv_parts) < 2:
            await message.channel.send("Interval format: `!schedule interval <seconds> <prompt>`\nExample: `!schedule interval 3600 Search for AI news`")
            return
        schedule_value = iv_parts[0]
        prompt = iv_parts[1]

    loop = asyncio.get_event_loop()

    def _create() -> str:
        try:
            t = _scheduler.create_task(  # type: ignore[union-attr]
                prompt=prompt,
                schedule_type=sched_type,
                schedule_value=schedule_value,
                agent="orchestrator",
                tools=",".join(_BASELINE_TOOLS),
                metadata={"created_by": str(message.author.id)},
            )
            next_r = (t.next_run or "")[:16].replace("T", " ")
            return (
                f"Task created: `{t.id[:8]}`\n"
                f"Schedule: `{sched_type}:{schedule_value}`\n"
                f"Next run: `{next_r} UTC`\n"
                f"Prompt: {prompt[:100]}\n"
                f"Cancel with: `!cancel-task {t.id[:8]}`"
            )
        except Exception as exc:
            return f"⚠️ Failed to create task: {exc}"

    async with message.channel.typing():
        reply = await loop.run_in_executor(_executor, _create)
    await send_chunked(message.channel, reply)


def _resolve_task(task_id: str) -> "tuple[Any | None, str]":
    """Find a task by ID prefix. Returns (task, error_msg)."""
    if _scheduler is None:
        return None, "Scheduler is not running."
    all_tasks = _scheduler.list_tasks()
    matches = [t for t in all_tasks if t.id.startswith(task_id)]
    if not matches:
        return None, f"No task found with id starting with `{task_id}`. Run `!tasks` to list."
    if len(matches) > 1:
        ids = ", ".join(t.id[:8] for t in matches)
        return None, f"Multiple tasks match `{task_id}`: {ids}. Provide more characters."
    return matches[0], ""


@cmd.prefix("!pause-task ", "!pause-task <id>", "pause a scheduled task (resume with !resume-task)")
async def _(message: discord.Message, task_id: str) -> None:
    if not task_id:
        await message.channel.send("Usage: `!pause-task <task-id>`")
        return
    loop = asyncio.get_event_loop()

    def _pause() -> str:
        t, err = _resolve_task(task_id)
        if err:
            return err
        try:
            _scheduler.pause_task(t.id)  # type: ignore[union-attr]
            return f"Task `{t.id[:8]}` paused. Resume with `!resume-task {t.id[:8]}`\nWas: {t.prompt[:80]}"
        except Exception as exc:
            return f"⚠️ Pause failed: {exc}"

    async with message.channel.typing():
        reply = await loop.run_in_executor(_executor, _pause)
    await send_chunked(message.channel, reply)


@cmd.prefix("!resume-task ", "!resume-task <id>", "resume a paused scheduled task")
async def _(message: discord.Message, task_id: str) -> None:
    if not task_id:
        await message.channel.send("Usage: `!resume-task <task-id>`")
        return
    loop = asyncio.get_event_loop()

    def _resume() -> str:
        t, err = _resolve_task(task_id)
        if err:
            return err
        try:
            _scheduler.resume_task(t.id)  # type: ignore[union-attr]
            # Fetch updated task to get new next_run
            updated = [x for x in _scheduler.list_tasks() if x.id == t.id]  # type: ignore[union-attr]
            next_r = (updated[0].next_run if updated else "?")[:16].replace("T", " ")
            return f"Task `{t.id[:8]}` resumed. Next run: `{next_r} UTC`\n{t.prompt[:80]}"
        except Exception as exc:
            return f"⚠️ Resume failed: {exc}"

    async with message.channel.typing():
        reply = await loop.run_in_executor(_executor, _resume)
    await send_chunked(message.channel, reply)


@cmd.prefix("!cancel-task ", "!cancel-task <id>", "permanently cancel a scheduled task")
async def _(message: discord.Message, task_id: str) -> None:
    if not task_id:
        await message.channel.send("Usage: `!cancel-task <task-id>` — get IDs from `!tasks`")
        return
    if _scheduler is None:
        await message.channel.send("Scheduler is not running.")
        return

    loop = asyncio.get_event_loop()

    def _cancel() -> str:
        t, err = _resolve_task(task_id)
        if err:
            return err
        try:
            _scheduler.cancel_task(t.id)  # type: ignore[union-attr]
            return f"Task `{t.id[:8]}` cancelled (permanent).\nWas: {t.prompt[:80]}"
        except Exception as exc:
            return f"⚠️ Cancel failed: {exc}"

    async with message.channel.typing():
        reply = await loop.run_in_executor(_executor, _cancel)
    await send_chunked(message.channel, reply)


@cmd.prefix("!task-log ", "!task-log <id>", "show recent run history for a scheduled task")
async def _(message: discord.Message, task_id: str) -> None:
    if not task_id:
        await message.channel.send("Usage: `!task-log <task-id>` — get IDs from `!tasks`")
        return
    if _scheduler is None:
        await message.channel.send("Scheduler is not running.")
        return

    loop = asyncio.get_event_loop()

    def _logs() -> str:
        t, err = _resolve_task(task_id)
        if err:
            return err
        try:
            logs = _scheduler._store.get_run_logs(t.id, limit=5)  # type: ignore[union-attr]
            if not logs:
                return f"No run history for task `{t.id[:8]}`."
            lines = [f"**Run log for `{t.id[:8]}`** ({len(logs)} most recent runs)", "```"]
            for run in logs:
                ok   = "✓" if run.get("success") else "✗"
                ts   = (run.get("started_at") or "")[:16].replace("T", " ")
                dur  = ""
                if run.get("started_at") and run.get("finished_at"):
                    try:
                        from datetime import datetime, timezone
                        s = datetime.fromisoformat(run["started_at"])
                        f = datetime.fromisoformat(run["finished_at"])
                        dur = f" ({int((f - s).total_seconds())}s)"
                    except Exception:
                        pass
                lines.append(f"  {ok} {ts} UTC{dur}")
                body = (run.get("result") or run.get("error") or "")[:120]
                if body:
                    lines.append(f"     {body}")
            lines.append("```")
            return "\n".join(lines)
        except Exception as exc:
            return f"⚠️ Log fetch failed: {exc}"

    async with message.channel.typing():
        reply = await loop.run_in_executor(_executor, _logs)
    await send_chunked(message.channel, reply)


# ── Slash commands ────────────────────────────────────────────────────────────
# All major !commands are mirrored here so they appear in Discord's / menu
# with autocomplete, type hints, and inline descriptions.
# The underlying logic is identical — both layers call the same helpers.

_Choice = discord.app_commands.Choice


@tree.command(name="ask", description="Ask Jarvis anything")
@discord.app_commands.describe(query="Your question or task")
async def slash_ask(interaction: discord.Interaction, query: str) -> None:
    uid  = interaction.user.id
    name = interaction.user.display_name
    await _slash_reply(
        interaction,
        lambda: _run_ask(query, uid, name),
        thread_name=f"Jarvis: {query[:50]}",
    )


@tree.command(name="brief", description="Morning brief: weather + top AI story + daily insight")
async def slash_brief(interaction: discord.Interaction) -> None:
    uid  = interaction.user.id
    name = interaction.user.display_name

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


@tree.command(name="weather", description="Current weather + 3-day forecast")
@discord.app_commands.describe(city="City name (leave blank for default)")
async def slash_weather(interaction: discord.Interaction, city: str = "") -> None:
    city = city or DEFAULT_CITY
    if not city:
        await interaction.response.send_message(
            "Usage: `/weather city:London` — or set `DEFAULT_WEATHER_CITY` in `.env`",
            ephemeral=True,
        )
        return
    await _slash_reply(
        interaction,
        lambda: _call_tool("weather", city=city) or _run_ask(f"Weather in {city}?"),
    )


@tree.command(name="research", description="Multi-source research synthesis stored to your memory")
@discord.app_commands.describe(topic="What to research")
async def slash_research(interaction: discord.Interaction, topic: str) -> None:
    uid  = interaction.user.id
    name = interaction.user.display_name

    def _research() -> str:
        return _run_ask(
            f"Research '{topic}' thoroughly.\n"
            "1. Use web_search (at least 2 angles).\n"
            "2. Use arxiv_search if technical/scientific.\n"
            "3. Check retrieval for prior memory context.\n"
            "4. Synthesise: Overview, Key Concepts, Recent Developments, "
            "Practical Applications, Key Takeaways.\n"
            f"5. Store findings tagged 'user:{uid}:research:{topic[:30]}'.",
            uid, name,
        )

    await _slash_reply(interaction, _research, thread_name=f"Research: {topic[:50]}")


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
@discord.app_commands.describe(topic="Topic to search (leave blank for general AI/ML)")
async def slash_papers(interaction: discord.Interaction, topic: str = "") -> None:
    query = topic or "LLM AI agent reasoning alignment"
    await _slash_reply(
        interaction,
        lambda: _call_tool("arxiv_search", query=query, max_results=8, sort_by="submittedDate")
                or _run_ask(f"Latest arXiv papers on: {query}"),
        thread_name=f"arXiv: {query[:50]}",
    )


@tree.command(name="summarize", description="Fetch and summarise a web page")
@discord.app_commands.describe(url="URL to summarise")
async def slash_summarize(interaction: discord.Interaction, url: str) -> None:
    def _summarize() -> str:
        raw = _call_tool("url_fetch", url=url, max_chars=6000)
        if raw:
            return _run_ask(
                f"Summarise this page. Key points and main argument.\n\nURL: {url}\n\n{raw}"
            )
        return _run_ask(f"Fetch and summarise: {url}")

    await _slash_reply(interaction, _summarize, thread_name=f"Summary: {url[:60]}")


@tree.command(name="memory", description="Search your personal memory store")
@discord.app_commands.describe(query="Search terms")
async def slash_memory(interaction: discord.Interaction, query: str) -> None:
    uid = interaction.user.id

    def _mem() -> str:
        rows = _search_user_memory(query, uid)
        if not rows:
            return (
                f"No memories found for '{query}'.\n"
                "Chat with Jarvis and it will build your memory store over time."
            )
        lines = [f"**Memory results for '{query}'** ({len(rows)})"]
        for i, r in enumerate(rows, 1):
            lines.append(f"**{i}.** {r.get('content', '')[:280]}")
        return "\n\n".join(lines)

    await _slash_reply(interaction, _mem)


@tree.command(name="digest", description="On-demand HN AI digest (top stories, last 72h)")
async def slash_digest(interaction: discord.Interaction) -> None:
    def _digest() -> str:
        return _run_ask(
            "Fetch the top 10 most-discussed AI/ML stories on Hacker News from "
            "the last 72 hours. Use web_search. For each: rank, title (as a link), "
            "comment count, score, one-sentence description. "
            "Format as a numbered Discord-friendly list."
        )

    await _slash_reply(interaction, _digest, thread_name="HN AI Digest")


@tree.command(name="ask-gemini", description="Ask Gemini Flash directly, bypassing the local model")
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


@tree.command(name="status", description="Show engine, model, Gemini/GitHub config, and host stats")
async def slash_status(interaction: discord.Interaction) -> None:
    def _status() -> str:
        try:
            j   = _get_jarvis()
            cfg = j._config  # type: ignore[attr-defined]
            model  = getattr(cfg.intelligence, "default_model", "?")
            engine = getattr(cfg.intelligence, "preferred_engine", "?")
        except Exception:
            model, engine = "?", "?"
        sys_info  = _call_tool("system_monitor") or "system_monitor unavailable"
        gemini_ok = bool(os.environ.get("GEMINI_API_KEY"))
        github_ok = bool(GITHUB_TOKEN and GITHUB_REPO)
        return (
            f"**Jarvis Status**\n```\n"
            f"Local model : {model} ({engine})\n"
            f"Gemini      : {'✓ configured' if gemini_ok else '✗ not set'}\n"
            f"GitHub      : {'✓ ' + GITHUB_REPO if github_ok else '✗ not set'}\n"
            f"Tools loaded: {len(AGENT_TOOLS)}\n"
            f"```\n{sys_info}"
        )

    await _slash_reply(interaction, _status)


@tree.command(name="health", description="On-demand health check: system + GitHub queue")
async def slash_health(interaction: discord.Interaction) -> None:
    def _health() -> str:
        lines = ["**On-demand Health Check**"]
        sys_info = _call_tool("system_monitor", detail="summary")
        if sys_info:
            lines.append(sys_info)
        if GITHUB_TOKEN and GITHUB_REPO:
            for label, emoji in [
                ("claude-code-work", "📥 queued"),
                ("claude-code-in-progress", "⚙️ in progress"),
                ("claude-code-done", "✅ done (unmerged)"),
            ]:
                data  = _gh_api(f"issues?labels={label}&state=open&per_page=5")
                count = len(data) if isinstance(data, list) else 0
                lines.append(f"{emoji}: {count}")
            prs = _gh_api("pulls?state=open&per_page=5")
            lines.append(f"🔀 open PRs: {len(prs) if isinstance(prs, list) else 0}")
        return "\n".join(lines)

    await _slash_reply(interaction, _health)


@tree.command(name="queue", description="Show the Claude Code issue queue")
async def slash_queue(interaction: discord.Interaction) -> None:
    def _queue() -> str:
        if not GITHUB_TOKEN or not GITHUB_REPO:
            return "⚠️ GITHUB_TOKEN / GITHUB_REPO not configured."
        lines = ["**Issue Worker Queue**"]
        for label, emoji in [
            ("claude-code-work",        "📥 queued"),
            ("claude-code-in-progress", "⚙️ in progress"),
            ("claude-code-done",        "✅ done (unmerged)"),
            ("claude-code-failed",      "❌ failed"),
        ]:
            data = _gh_api(f"issues?labels={label}&state=open&per_page=10")
            if not isinstance(data, list):
                lines.append(f"{emoji}: (error)")
                continue
            for issue in (data or [])[:5]:
                n, title, url = issue["number"], issue["title"][:55], issue["html_url"]
                lines.append(f"{emoji}: **[#{n}]({url})** {title}")
            if not data:
                lines.append(f"{emoji}: none")
        return "\n".join(lines)

    await _slash_reply(interaction, _queue)


@tree.command(name="prs", description="List open pull requests")
async def slash_prs(interaction: discord.Interaction) -> None:
    def _prs() -> str:
        data = _gh_api("pulls?state=open&per_page=15")
        if not isinstance(data, list):
            return f"⚠️ Could not fetch PRs: {data}"
        if not data:
            return "No open pull requests."
        lines = ["**Open Pull Requests**"]
        for pr in data:
            n, title = pr["number"], pr["title"][:60]
            url, user = pr["html_url"], pr.get("user", {}).get("login", "?")
            lines.append(f"• **[#{n}]({url})** {title} — `{user}` → `/merge {n}`")
        return "\n".join(lines)

    await _slash_reply(interaction, _prs)


@tree.command(name="pr", description="Show PR details: files changed and description")
@discord.app_commands.describe(number="Pull request number")
async def slash_pr(interaction: discord.Interaction, number: int) -> None:
    def _pr_detail() -> str:
        pr = _gh_api(f"pulls/{number}")
        if not pr or "_error" in pr:
            return f"⚠️ Could not fetch PR #{number}"
        title  = pr.get("title", "")
        state  = pr.get("state", "")
        url    = pr.get("html_url", "")
        body   = (pr.get("body") or "")[:600]
        branch = pr.get("head", {}).get("ref", "")
        user   = pr.get("user", {}).get("login", "?")
        files  = _gh_api(f"pulls/{number}/files?per_page=20")
        file_lines = []
        if isinstance(files, list):
            for f in files[:15]:
                fname  = f.get("filename", "")
                status = f.get("status", "")
                adds   = f.get("additions", 0)
                dels   = f.get("deletions", 0)
                file_lines.append(f"  `{fname}` ({status} +{adds}/-{dels})")
        files_str = "\n".join(file_lines) or "  _(no file data)_"
        return (
            f"**PR #{number}** — `{state}`\n**{title}**\n{url}\n"
            f"Author: `{user}` · Branch: `{branch}`\n\n"
            f"**Files changed:**\n{files_str}\n\n"
            f"**Description:**\n{body}\n\nMerge: `/merge {number}`"
        )

    await _slash_reply(interaction, _pr_detail, thread_name=f"PR #{number}")


@tree.command(name="merge", description="Squash-merge a PR and delete the source branch")
@discord.app_commands.describe(number="Pull request number")
async def slash_merge(interaction: discord.Interaction, number: int) -> None:
    def _do_merge() -> str:
        pr = _gh_api(f"pulls/{number}")
        if not pr or "_error" in pr:
            return f"⚠️ Could not fetch PR #{number}: {pr}"
        if pr.get("state") != "open":
            return f"PR #{number} is `{pr.get('state')}` — nothing to merge."
        if pr.get("mergeable") is False:
            return (
                f"⚠️ PR #{number} has merge conflicts.\n"
                "Use `/ask-gemini` with the conflict details to get a resolution plan."
            )
        title    = pr.get("title", "")
        head_ref = pr.get("head", {}).get("ref", "")
        pr_url   = pr.get("html_url", "")
        result   = _gh_api(
            f"pulls/{number}/merge", method="PUT",
            json={"merge_method": "squash",
                  "commit_title": f"[JarvisPanda] {title} (#{number})"},
        )
        if result and "_error" not in result:
            sha = (result.get("sha") or "")[:8]
            if head_ref:
                _gh_api(f"git/refs/heads/{head_ref}", method="DELETE")
            return f"✅ PR #{number} merged → `{sha}`\n{pr_url}\nBranch `{head_ref}` deleted."
        return f"⚠️ Merge failed: {result}"

    await _slash_reply(interaction, _do_merge)


@tree.command(name="propose", description="Draft a feature proposal and submit it to GitHub")
@discord.app_commands.describe(idea="Describe the feature you want")
async def slash_propose(interaction: discord.Interaction, idea: str) -> None:
    uid  = interaction.user.id
    name = interaction.user.display_name
    await interaction.response.defer(thinking=True)
    await interaction.followup.send(
        "Analysing proposal, consulting Gemini, and opening a GitHub issue…\n"
        "React **👍** to approve for Claude's queue, **👎** to withdraw."
    )
    loop  = asyncio.get_event_loop()
    reply = await loop.run_in_executor(
        _executor,
        lambda: _run_ask(
            f"Discord user {name} proposed:\n\n```\n{idea}\n```\n\n"
            "1. Use gemini_escalate(mode=plan) for an implementation plan.\n"
            "2. Use github_issue with the plan, a clear title (<80 chars), "
            "and acceptance criteria.\n"
            "3. Return the issue URL.",
            uid, name,
        ),
    )
    await interaction.followup.send(reply[:MAX_MSG_LEN])


@tree.command(name="build-tool", description="Design, spec, and propose a new Jarvis tool via GitHub issue")
@discord.app_commands.describe(description="What the tool should do")
async def slash_build_tool(interaction: discord.Interaction, description: str) -> None:
    uid  = interaction.user.id
    name = interaction.user.display_name
    await interaction.response.defer(thinking=True)
    await interaction.followup.send(
        f"Designing tool: _{description}_\n"
        "Consulting Gemini for a spec, then opening a GitHub issue…\n"
        "React **👍** to queue for Claude Code, **👎** to withdraw."
    )
    loop  = asyncio.get_event_loop()
    reply = await loop.run_in_executor(
        _executor,
        lambda: _run_ask(
            f"Build a new OpenJarvis tool:\n\n```\n{description}\n```\n\n"
            "1. gemini_escalate(mode=plan) for a complete spec: name, params, "
            "implementation, example usage.\n"
            "2. github_issue with the spec (title: 'feat: add {tool_name} tool').\n"
            "3. Return the issue URL.\n\n"
            "Reference: src/openjarvis/tools/weather.py",
            uid, name,
        ),
    )
    await interaction.followup.send(reply[:MAX_MSG_LEN])


@tree.command(name="agent", description="Run a one-shot local agent task")
@discord.app_commands.describe(task="What you want the agent to do")
async def slash_agent(interaction: discord.Interaction, task: str) -> None:
    uid  = interaction.user.id
    name = interaction.user.display_name
    await _slash_reply(
        interaction,
        lambda: _run_ask(task, uid, name),
        thread_name=f"Agent: {task[:50]}",
    )


@tree.command(name="run-skill", description="Execute a registered Jarvis skill")
@discord.app_commands.describe(
    skill_name="Skill name (see /skills for list)",
    args="Optional key=value arguments",
)
async def slash_run_skill(
    interaction: discord.Interaction, skill_name: str, args: str = ""
) -> None:
    uid  = interaction.user.id
    name = interaction.user.display_name
    await _slash_reply(
        interaction,
        lambda: _run_ask(
            f"Run the Jarvis skill '{skill_name}'"
            + (f" with parameters: {args}" if args else "")
            + ". Use sensible defaults for missing parameters.",
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
            "  code-lint                Lint and review code\n"
            "  data-analyze             Analyse and interpret data\n"
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


# ── Slash: scheduler commands ─────────────────────────────────────────────────

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
async def slash_schedule(
    interaction: discord.Interaction,
    type: str,
    value: str,
    prompt: str,
) -> None:
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
                ok  = "✓" if run.get("success") else "✗"
                ts  = (run.get("started_at") or "")[:16].replace("T", " ")
                body = (run.get("result") or run.get("error") or "")[:120]
                lines.append(f"  {ok} {ts} UTC")
                if body:
                    lines.append(f"     {body}")
            lines.append("```")
            return "\n".join(lines)
        except Exception as exc:
            return f"⚠️ {exc}"

    await _slash_reply(interaction, _logs)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client.run(DISCORD_BOT_TOKEN, log_handler=None)
