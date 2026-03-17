"""Jarvis Discord bot.

Listens for:
  • Messages in the configured DISCORD_CHANNEL_NAME (default: jarvis)
  • Direct @mentions anywhere in the server
  • !ask <query> prefix in any channel

Forwards queries to the Jarvis OpenAI-compatible API and replies inline.
Splits responses that exceed Discord's 2000-char limit into threaded chunks.
"""

from __future__ import annotations

import asyncio
import logging
import os
import textwrap

import discord
import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("jarvis-discord")

DISCORD_BOT_TOKEN: str = os.environ["DISCORD_BOT_TOKEN"]
JARVIS_URL: str = os.environ.get("JARVIS_URL", "http://jarvis-server:8000")
CHANNEL_NAME: str = os.environ.get("DISCORD_CHANNEL_NAME", "jarvis")
COMMAND_PREFIX: str = os.environ.get("DISCORD_PREFIX", "!ask")
REQUEST_TIMEOUT: int = int(os.environ.get("JARVIS_TIMEOUT", "120"))
MAX_MSG_LEN: int = 1990  # Discord hard limit is 2000; keep margin


# ── Discord client setup ──────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def query_jarvis(prompt: str) -> str:
    """Call the Jarvis /v1/chat/completions endpoint and return the reply."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as http:
            resp = await http.post(f"{JARVIS_URL}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except httpx.TimeoutException:
        return "⏱️ Jarvis timed out — the model may be loading. Try again in a moment."
    except httpx.HTTPStatusError as exc:
        log.error("Jarvis API error: %s", exc)
        return f"⚠️ Jarvis returned an error ({exc.response.status_code}). Check server logs."
    except Exception as exc:
        log.exception("Unexpected error querying Jarvis")
        return f"⚠️ Unexpected error: {exc}"


async def send_chunked(channel: discord.abc.Messageable, text: str) -> None:
    """Send text to a Discord channel, splitting at MAX_MSG_LEN boundaries."""
    # Prefer splitting on paragraph/sentence boundaries within the limit
    chunks = textwrap.wrap(
        text,
        width=MAX_MSG_LEN,
        break_long_words=False,
        break_on_hyphens=False,
        replace_whitespace=False,
    )
    if not chunks:
        chunks = [text[i : i + MAX_MSG_LEN] for i in range(0, len(text), MAX_MSG_LEN)]
    for chunk in chunks:
        await channel.send(chunk)


def extract_query(message: discord.Message) -> str | None:
    """Return the query string from a message, or None if not addressed to Jarvis."""
    content = message.content.strip()

    # @mention anywhere
    if client.user and client.user.mentioned_in(message):
        query = content.replace(f"<@{client.user.id}>", "").replace(
            f"<@!{client.user.id}>", ""
        ).strip()
        return query or None

    # !ask prefix
    if content.lower().startswith(COMMAND_PREFIX.lower()):
        query = content[len(COMMAND_PREFIX):].strip()
        return query or None

    # Any message in the designated #jarvis channel
    if hasattr(message.channel, "name") and message.channel.name == CHANNEL_NAME:
        return content or None

    return None


# ── Events ────────────────────────────────────────────────────────────────────

@client.event
async def on_ready() -> None:
    log.info("Jarvis Discord bot online as %s (id=%s)", client.user, client.user.id if client.user else "?")
    log.info("Listening in #%s | prefix: %s | API: %s", CHANNEL_NAME, COMMAND_PREFIX, JARVIS_URL)


@client.event
async def on_message(message: discord.Message) -> None:
    # Ignore own messages
    if message.author == client.user:
        return

    # Ignore bots
    if message.author.bot:
        return

    query = extract_query(message)
    if query is None:
        return

    log.info("Query from %s: %s", message.author, query[:120])

    async with message.channel.typing():
        reply = await query_jarvis(query)

    # Reply in a thread if the channel supports it and the response is long
    if isinstance(message.channel, (discord.TextChannel, discord.Thread)) and len(reply) > MAX_MSG_LEN:
        try:
            thread = await message.create_thread(
                name=f"Jarvis: {query[:50]}",
                auto_archive_duration=60,
            )
            await send_chunked(thread, reply)
            return
        except discord.Forbidden:
            pass  # No thread permissions — fall through to normal reply

    await send_chunked(message.channel, reply)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client.run(DISCORD_BOT_TOKEN, log_handler=None)
