"""Jarvis Discord bot — runs the full OpenJarvis SDK in-process.

Architecture: this container IS the Jarvis agent, not a proxy.
- Uses the local qwen3:8b (via Ollama) through the OpenJarvis orchestrator
- Memory context injection, web search, retrieval — all tools available
- When the agent gets stuck after its turn limit, it automatically calls the
  github_issue tool to cut an escalation issue for the ClaudeCodeAgent worker
- No cloud models involved unless you explicitly add one to the config

Triggers:
  - Any message in DISCORD_CHANNEL_NAME (default: #jarvis)
  - @mention anywhere in the server
  - DISCORD_PREFIX prefix (default: !ask) in any channel

Special commands:
  !status  — show engine health, model, memory stats
  !help    — show usage
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import discord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("jarvis-discord")

# ── Config ────────────────────────────────────────────────────────────────────

DISCORD_BOT_TOKEN = os.environ["DISCORD_BOT_TOKEN"]
CHANNEL_NAME      = os.environ.get("DISCORD_CHANNEL_NAME", "jarvis")
COMMAND_PREFIX    = os.environ.get("DISCORD_PREFIX", "!ask")
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
        f"In #{CHANNEL_NAME}     — just type, no prefix needed\n"
        f"@Jarvis <query>   — mention me anywhere\n"
        f"{COMMAND_PREFIX} <query>  — explicit prefix\n"
        f"!status           — engine & model info\n"
        f"!help             — this message\n"
        f"```\n"
        f"Runs **{os.environ.get('OLLAMA_MODEL', 'qwen3:8b')}** locally. "
        f"Stuck tasks auto-escalate to ClaudeCode via GitHub issues."
    )


async def send_chunked(dest: discord.abc.Messageable, text: str) -> None:
    """Send text, splitting on newlines within the Discord character limit."""
    if len(text) <= MAX_MSG_LEN:
        await dest.send(text)
        return

    # Try to split on paragraph breaks first
    paragraphs = text.split("\n\n")
    chunk = ""
    for para in paragraphs:
        candidate = (chunk + "\n\n" + para).lstrip("\n") if chunk else para
        if len(candidate) <= MAX_MSG_LEN:
            chunk = candidate
        else:
            if chunk:
                await dest.send(chunk)
            # Para itself might be too long — hard split
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

    # Built-in commands (no Jarvis call needed)
    if content.lower() == "!status":
        await message.channel.send(_run_status())
        return
    if content.lower() == "!help":
        await message.channel.send(_help_text())
        return

    query = extract_query(message)
    if query is None:
        return

    log.info("[%s] %s: %s", message.channel, message.author, query[:100])

    async with message.channel.typing():
        loop = asyncio.get_event_loop()
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
