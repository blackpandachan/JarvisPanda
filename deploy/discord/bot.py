"""Jarvis Discord bot — runs the full OpenJarvis SDK in-process.

Architecture: this container IS the Jarvis agent, not a proxy.
- Uses the local qwen3:8b (via Ollama) through the OpenJarvis orchestrator
- Memory context injection, web search, retrieval, weather — all tools available
- When the agent gets stuck after its turn limit, it automatically calls the
  github_issue tool to cut an escalation issue for the ClaudeCodeAgent worker
- No cloud models involved unless you explicitly add one to the config

Triggers:
  - Any message in DISCORD_CHANNEL_NAME (default: #jarvis)
  - @mention anywhere in the server
  - DISCORD_PREFIX prefix (default: !ask) in any channel

Special commands:
  !status              — show engine health, model, memory stats
  !help                — show usage and available commands
  !weather [city]      — current conditions + 3-day forecast
  !research <topic>    — multi-source research synthesis stored to memory
  !skills              — list available Jarvis skills
  !propose <idea>      — submit a feature proposal for agent review
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor

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
MAX_MSG_LEN       = 1990

# Thread pool for running synchronous Jarvis.ask without blocking the event loop
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jarvis-ask")

# ── Jarvis SDK init ───────────────────────────────────────────────────────────

log.info("Initialising Jarvis SDK (loading config + engine)...")

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


# Pre-warm on startup (avoids first-message latency)
threading.Thread(target=_get_jarvis, daemon=True).start()

AGENT_TOOLS = [
    "think",
    "calculator",
    "retrieval",
    "web_search",
    "file_read",
    "weather",
    "github_issue",  # escalation path → ClaudeCodeAgent worker
]

# ── Discord client ────────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_query(message: discord.Message) -> str | None:
    """Return the query, or None if this message isn't for Jarvis."""
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

    return None


def _run_ask(query: str) -> str:
    """Synchronous Jarvis call — runs in thread pool."""
    j = _get_jarvis()
    try:
        result = j.ask(
            query,
            agent="orchestrator",
            tools=AGENT_TOOLS,
        )
        return result.get("content") or "(no response)"
    except Exception as exc:
        log.exception("Jarvis.ask failed: %s", exc)
        return (
            f"⚠️ Jarvis encountered an error: `{exc}`\n"
            "If this persists, I'll open a GitHub issue for the ClaudeCode worker to fix it."
        )


def _run_weather(city: str) -> str:
    """Direct weather tool call — bypasses full agent for speed."""
    from openjarvis.core.registry import ToolRegistry
    try:
        tool = ToolRegistry.get("weather")
        if tool is None:
            # Fall back to agent
            return _run_ask(f"What is the current weather in {city}? Give me current conditions and a 3-day forecast.")
        result = tool.execute(city=city)
        return result.content
    except Exception as exc:
        log.exception("Weather tool failed: %s", exc)
        return f"⚠️ Weather lookup failed: {exc}"


def _run_research(topic: str) -> str:
    """Research via agent with explicit research framing."""
    return _run_ask(
        f"Research the following topic thoroughly. Use web_search to find current information, "
        f"check retrieval for anything in memory, then synthesize a comprehensive summary. "
        f"Store key findings to memory when done.\n\nTopic: {topic}"
    )


def _run_propose(idea: str) -> str:
    """Have the agent draft and submit a feature proposal as a GitHub issue."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return (
            "⚠️ GITHUB_TOKEN and GITHUB_REPO must be set to submit proposals.\n"
            "Set them in your `.env` file and restart the bot."
        )
    return _run_ask(
        f"A user has proposed the following new feature or improvement for the OpenJarvis/JarvisPanda project:\n\n"
        f"```\n{idea}\n```\n\n"
        f"Your job:\n"
        f"1. Think through the proposal: is it technically feasible? What would it require? "
        f"What are the benefits and tradeoffs?\n"
        f"2. Draft a well-structured GitHub issue with:\n"
        f"   - A clear title (< 80 chars)\n"
        f"   - Description of the feature\n"
        f"   - Technical approach / implementation plan\n"
        f"   - Acceptance criteria\n"
        f"3. Use the github_issue tool to submit it with the label 'claude-code-work'.\n"
        f"4. Report the issue URL back so the user can track it.\n\n"
        f"Be concise and technical. This will be worked by the ClaudeCodeAgent."
    )


def _list_skills() -> str:
    """Return a formatted list of available Jarvis skills."""
    try:
        from openjarvis.skills.loader import SkillLoader
        loader = SkillLoader()
        skills = loader.list_skills()
        if not skills:
            return "No skills loaded."
        lines = ["**Available Jarvis Skills**", "```"]
        for s in sorted(skills, key=lambda x: x.get("name", "")):
            name = s.get("name", "?")
            desc = s.get("description", "")[:60]
            lines.append(f"{name:<25} {desc}")
        lines.append("```")
        return "\n".join(lines)
    except Exception as exc:
        log.warning("Could not list skills: %s", exc)
        return (
            "**Available Jarvis Skills**\n"
            "```\n"
            "topic-research           Research any topic, store findings\n"
            "web-summarize            Summarize web page content\n"
            "daily-digest             Summarize recent activity\n"
            "code-lint                Lint and review code\n"
            "data-analyze             Analyze and interpret data\n"
            "knowledge-extract        Extract key facts from documents\n"
            "```\n"
            "*Use `!research <topic>` for research or just ask me directly.*"
        )


def _run_status() -> str:
    """Return a status string from the running Jarvis instance."""
    try:
        j = _get_jarvis()
        cfg = j._config  # type: ignore[attr-defined]
        model   = getattr(cfg.intelligence, "default_model", "?")
        engine  = getattr(cfg.intelligence, "preferred_engine", "?")
        mem_cfg = getattr(cfg, "memory", None)
        mem_backend = getattr(mem_cfg, "default_backend", "?") if mem_cfg else "?"
        return (
            f"**Jarvis Status**\n"
            f"```\n"
            f"Model  : {model}\n"
            f"Engine : {engine}\n"
            f"Memory : {mem_backend}\n"
            f"Tools  : {', '.join(AGENT_TOOLS)}\n"
            f"```"
        )
    except Exception as exc:
        return f"⚠️ Status check failed: {exc}"


def _help_text() -> str:
    return (
        f"**Jarvis — local AI assistant**\n"
        f"```\n"
        f"In #{CHANNEL_NAME}         — just type, no prefix needed\n"
        f"@Jarvis <query>       — mention me anywhere\n"
        f"{COMMAND_PREFIX} <query>    — explicit prefix\n"
        f"\n"
        f"Special commands:\n"
        f"  !status             — engine & model info\n"
        f"  !help               — this message\n"
        f"  !weather [city]     — weather + 3-day forecast\n"
        f"  !research <topic>   — multi-source research synthesis\n"
        f"  !skills             — list available skills\n"
        f"  !propose <idea>     — submit a feature proposal to GitHub\n"
        f"```\n"
        f"Runs **{os.environ.get('OLLAMA_MODEL', 'qwen3:8b')}** locally. "
        f"Stuck tasks auto-escalate to ClaudeCode via GitHub issues."
    )


async def send_chunked(dest: discord.abc.Messageable, text: str) -> None:
    """Send text, splitting on newlines within the Discord character limit."""
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
async def on_message(message: discord.Message) -> None:
    if message.author == client.user or message.author.bot:
        return

    content = message.content.strip()
    lower   = content.lower()

    # ── Built-in commands ─────────────────────────────────────────────────────

    if lower == "!status":
        await message.channel.send(_run_status())
        return

    if lower == "!help":
        await message.channel.send(_help_text())
        return

    if lower == "!skills":
        loop  = asyncio.get_event_loop()
        reply = await loop.run_in_executor(_executor, _list_skills)
        await send_chunked(message.channel, reply)
        return

    if lower.startswith("!weather"):
        city = content[8:].strip() or DEFAULT_CITY
        if not city:
            await message.channel.send(
                "Usage: `!weather <city>` — e.g. `!weather London`\n"
                "Set `DEFAULT_WEATHER_CITY` in your `.env` to skip specifying a city."
            )
            return
        async with message.channel.typing():
            loop  = asyncio.get_event_loop()
            reply = await loop.run_in_executor(_executor, _run_weather, city)
        await send_chunked(message.channel, reply)
        return

    if lower.startswith("!research "):
        topic = content[10:].strip()
        if not topic:
            await message.channel.send("Usage: `!research <topic>`")
            return
        async with message.channel.typing():
            loop  = asyncio.get_event_loop()
            reply = await loop.run_in_executor(_executor, _run_research, topic)
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

    if lower.startswith("!propose "):
        idea = content[9:].strip()
        if not idea:
            await message.channel.send("Usage: `!propose <your feature idea>`")
            return
        await message.channel.send(
            f"Analysing your proposal and drafting a GitHub issue… this may take a moment."
        )
        async with message.channel.typing():
            loop  = asyncio.get_event_loop()
            reply = await loop.run_in_executor(_executor, _run_propose, idea)
        await send_chunked(message.channel, reply)
        return

    # ── Regular Jarvis query ──────────────────────────────────────────────────

    query = extract_query(message)
    if query is None:
        return

    log.info("[%s] %s: %s", message.channel, message.author, query[:100])

    async with message.channel.typing():
        loop  = asyncio.get_event_loop()
        reply = await loop.run_in_executor(_executor, _run_ask, query)

    # Long responses go in a thread to keep the channel clean
    if (
        isinstance(message.channel, (discord.TextChannel, discord.Thread))
        and len(reply) > MAX_MSG_LEN
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
