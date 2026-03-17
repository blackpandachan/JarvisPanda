"""arXiv search tool — free search over recent academic papers.

Uses the arXiv Atom API (no key required).  Returns title, authors,
abstract snippet, category, and link for the top N matching papers.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import httpx

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

_API = "https://export.arxiv.org/api/query"
_NS  = {"atom": "http://www.w3.org/2005/Atom"}


def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return (el.text or "").strip()


@ToolRegistry.register("arxiv_search")
class ArxivSearchTool(BaseTool):
    """Search arXiv for recent academic papers on any topic."""

    tool_id = "arxiv_search"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="arxiv_search",
            description=(
                "Search arXiv for academic papers. Returns titles, authors, "
                "abstract snippets, and links. Useful for cutting-edge research "
                "on AI/ML, physics, math, CS, and other fields. Free, no API key."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g. 'LLM reasoning chain-of-thought').",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max papers to return (1–20, default 8).",
                        "default": 8,
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                        "description": "Sort order (default: submittedDate = newest first).",
                        "default": "submittedDate",
                    },
                },
                "required": ["query"],
            },
            category="research",
        )

    def execute(self, **params: Any) -> ToolResult:
        query: str = params.get("query", "").strip()
        max_results: int = min(int(params.get("max_results", 8)), 20)
        sort_by: str = params.get("sort_by", "submittedDate")

        if not query:
            return ToolResult(tool_name="arxiv_search", content="Error: query is required.", success=False)

        try:
            resp = httpx.get(
                _API,
                params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": sort_by,
                    "sortOrder": "descending",
                },
                timeout=20,
            )
            resp.raise_for_status()
        except Exception as exc:
            return ToolResult(tool_name="arxiv_search", content=f"arXiv API error: {exc}", success=False)

        try:
            root    = ET.fromstring(resp.text)
            entries = root.findall("atom:entry", _NS)
        except ET.ParseError as exc:
            return ToolResult(tool_name="arxiv_search", content=f"XML parse error: {exc}", success=False)

        if not entries:
            return ToolResult(
                tool_name="arxiv_search",
                content=f"No papers found for '{query}'.",
                success=True,
            )

        lines = [f"**arXiv results for '{query}'** ({len(entries)} papers)\n"]
        for i, entry in enumerate(entries, 1):
            title   = _text(entry.find("atom:title", _NS)).replace("\n", " ")
            summary = _text(entry.find("atom:summary", _NS)).replace("\n", " ")
            if len(summary) > 200:
                summary = summary[:197] + "…"

            authors = [
                _text(a.find("atom:name", _NS))
                for a in entry.findall("atom:author", _NS)
            ]
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += f" +{len(authors) - 3} more"

            # Prefer the abs link
            link = ""
            for lnk in entry.findall("atom:link", _NS):
                if lnk.get("type") == "text/html":
                    link = lnk.get("href", "")
                    break
            if not link:
                link = _text(entry.find("atom:id", _NS))

            published = _text(entry.find("atom:published", _NS))[:10]

            # Primary category
            cat_el = entry.find("{http://arxiv.org/schemas/atom}primary_category")
            cat = cat_el.get("term", "") if cat_el is not None else ""

            lines.append(
                f"**{i}. [{title}]({link})**\n"
                f"{author_str} · {published} · `{cat}`\n"
                f"{summary}"
            )

        return ToolResult(
            tool_name="arxiv_search",
            content="\n\n".join(lines),
            success=True,
            metadata={"count": len(entries), "query": query},
        )


__all__ = ["ArxivSearchTool"]
