"""URL fetch tool — retrieve and clean web page content.

Fetches a URL with a browser-like User-Agent, strips HTML tags and boilerplate,
and returns the clean readable text.  No external deps beyond httpx.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Tags whose content we drop entirely (scripts, styles, nav boilerplate)
_DROP_TAG = re.compile(
    r"<(script|style|nav|footer|header|aside|noscript|iframe|svg|form)"
    r"[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
# Remaining HTML tags
_TAG      = re.compile(r"<[^>]+>")
# Collapse whitespace
_WS       = re.compile(r"\n{3,}")
_NBSP     = re.compile(r"&[a-z]+;|&#\d+;")


def _clean(html: str, max_chars: int = 8000) -> str:
    text = _DROP_TAG.sub(" ", html)
    text = _TAG.sub("", text)
    text = _NBSP.sub(" ", text)
    # Normalise whitespace
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    text  = "\n".join(lines)
    text  = _WS.sub("\n\n", text)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n…[truncated at {max_chars} chars]"
    return text.strip()


@ToolRegistry.register("url_fetch")
class UrlFetchTool(BaseTool):
    """Fetch a web page and return its readable text content."""

    tool_id = "url_fetch"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="url_fetch",
            description=(
                "Fetch the text content of any public web page or article. "
                "Returns clean, readable text stripped of HTML, scripts, and navigation. "
                "Use this when you have a specific URL to read, such as an article, "
                "documentation page, or blog post."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to fetch (must start with http:// or https://).",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters of content to return (default 8000, max 20000).",
                        "default": 8000,
                    },
                },
                "required": ["url"],
            },
            category="web",
        )

    def execute(self, **params: Any) -> ToolResult:
        url: str      = params.get("url", "").strip()
        max_chars: int = min(int(params.get("max_chars", 8000)), 20000)

        if not url:
            return ToolResult(tool_name="url_fetch", content="Error: url is required.", success=False)

        if not url.startswith(("http://", "https://")):
            return ToolResult(
                tool_name="url_fetch",
                content="Error: url must start with http:// or https://",
                success=False,
            )

        try:
            resp = httpx.get(
                url,
                headers=_HEADERS,
                timeout=20,
                follow_redirects=True,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                tool_name="url_fetch",
                content=f"HTTP {exc.response.status_code} fetching {url}",
                success=False,
            )
        except Exception as exc:
            return ToolResult(tool_name="url_fetch", content=f"Fetch error: {exc}", success=False)

        content_type = resp.headers.get("content-type", "")
        if "text/html" in content_type or "text/plain" in content_type or not content_type:
            text = _clean(resp.text, max_chars)
        else:
            text = f"Non-HTML content ({content_type}). Raw preview:\n{resp.text[:500]}"

        final_url = str(resp.url)
        header    = f"**{final_url}**\n\n"

        return ToolResult(
            tool_name="url_fetch",
            content=header + text,
            success=True,
            metadata={"url": final_url, "status": resp.status_code, "chars": len(text)},
        )


__all__ = ["UrlFetchTool"]
