"""Gemini escalation tool — second-opinion analysis via Google Gemini Flash.

This is the middle tier of the JarvisPanda escalation chain:

  Local qwen3:8b  →  gemini_escalate  →  github_issue (ClaudeCodeAgent)

Use this tool when:
  - You have tried several approaches and are still stuck
  - You need a more capable model to help plan or solve a technical problem
  - You want a second opinion before cutting a GitHub issue

The tool calls the Gemini API and returns either:
  - A solution the agent can use directly
  - A detailed implementation plan to include in a github_issue body

Required env var: GEMINI_API_KEY
Optional env var: GEMINI_MODEL (default: gemini-3-flash-preview)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

log = logging.getLogger("gemini_escalate")

_GEMINI_API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_DEFAULT_MODEL = "gemini-3-flash-preview"

# Phrases that indicate Gemini believes code changes are required
_NEEDS_CODE_PHRASES = [
    "requires code changes",
    "requires modifying",
    "needs to be implemented",
    "should be added to the codebase",
    "i cannot execute",
    "i cannot access the repository",
    "would need to modify",
    "implementation plan",
    "pull request",
    "commit to the repo",
    "cannot make changes",
    "requires a developer",
]


def gemini_generate(prompt: str, model: str | None = None, max_tokens: int = 4096) -> str:
    """Low-level Gemini API call. Raises on failure."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    model = model or os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL)
    url   = _GEMINI_API.format(model=model)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.7,
            "topP": 0.9,
        },
    }

    resp = httpx.post(
        url,
        json=payload,
        params={"key": api_key},
        timeout=60,
    )
    resp.raise_for_status()

    data       = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError(f"Gemini returned no candidates: {data}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text  = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise ValueError("Gemini returned an empty response.")
    return text


def needs_code_changes(text: str) -> bool:
    """Heuristic: does Gemini's response indicate repository changes are needed?"""
    lower = text.lower()
    return any(phrase in lower for phrase in _NEEDS_CODE_PHRASES)


@ToolRegistry.register("gemini_escalate")
class GeminiEscalateTool(BaseTool):
    """Escalate to Gemini Flash when stuck — second opinion before GitHub issue.

    Pass your full problem context, what you've tried, and what's blocking you.
    Gemini will either resolve the problem or produce a detailed implementation
    plan you can use as the body of a github_issue.
    """

    tool_id = "gemini_escalate"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="gemini_escalate",
            description=(
                "Escalate a problem to Google Gemini Flash for a second opinion. "
                "Use this BEFORE creating a GitHub issue — Gemini may be able to "
                "solve the problem directly or produce a detailed implementation plan. "
                "Include everything you've tried and what's blocking you. "
                "The response will tell you either the solution OR whether a code "
                "change is needed (in which case use github_issue with this output)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "problem": {
                        "type": "string",
                        "description": (
                            "The specific problem or question you need help with. "
                            "Be precise and include any error messages."
                        ),
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "Full context: what you've tried, relevant code, "
                            "file paths, and what blocked each attempt. "
                            "More context = better response from Gemini."
                        ),
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["solve", "plan"],
                        "description": (
                            "'solve' = try to answer/fix directly (default). "
                            "'plan' = produce an implementation plan for a code change."
                        ),
                        "default": "solve",
                    },
                },
                "required": ["problem"],
            },
            category="escalation",
            latency_estimate=15.0,
        )

    def execute(self, **params: Any) -> ToolResult:
        problem: str = params.get("problem", "").strip()
        context: str = params.get("context", "").strip()
        mode: str    = params.get("mode", "solve")

        if not problem:
            return ToolResult(
                tool_name="gemini_escalate",
                content="Error: problem is required.",
                success=False,
            )

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return ToolResult(
                tool_name="gemini_escalate",
                content=(
                    "GEMINI_API_KEY is not set — cannot escalate to Gemini. "
                    "Either set the key or use the github_issue tool to escalate directly."
                ),
                success=False,
            )

        model = os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL)

        if mode == "plan":
            system_block = (
                "You are a senior software engineer helping plan a code change "
                "for the JarvisPanda/OpenJarvis project. "
                "Produce a detailed, actionable implementation plan that another "
                "developer (Claude Code) can follow step by step. "
                "Include: files to modify, functions to add/change, test strategy, "
                "and potential pitfalls. Be specific."
            )
        else:
            system_block = (
                "You are a helpful AI assistant being used as a second opinion. "
                "Try to solve the problem directly if you can. "
                "If the problem requires changes to a specific code repository "
                "you don't have access to, say so clearly and produce a detailed "
                "implementation plan instead."
            )

        parts = [system_block, ""]
        if context:
            parts.append(f"## Context\n{context}\n")
        parts.append(f"## Problem\n{problem}")
        prompt = "\n".join(parts)

        try:
            response = gemini_generate(prompt, model=model)
        except httpx.HTTPStatusError as exc:
            log.warning("Gemini API HTTP error: %s", exc)
            return ToolResult(
                tool_name="gemini_escalate",
                content=f"Gemini API error ({exc.response.status_code}): {exc.response.text[:300]}",
                success=False,
            )
        except Exception as exc:
            log.warning("Gemini escalation failed: %s", exc)
            return ToolResult(
                tool_name="gemini_escalate",
                content=f"Gemini escalation failed: {exc}",
                success=False,
            )

        code_needed = needs_code_changes(response)
        header = (
            f"**Gemini Flash ({model}) Response**"
            + (" — ⚠️ code changes required" if code_needed else "")
            + "\n\n"
        )

        return ToolResult(
            tool_name="gemini_escalate",
            content=header + response,
            success=True,
            metadata={
                "model": model,
                "needs_code_changes": code_needed,
                "mode": mode,
            },
        )


__all__ = ["GeminiEscalateTool", "gemini_generate", "needs_code_changes"]
