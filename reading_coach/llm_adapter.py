"""
reading_coach/llm_adapter.py — Adapter that converts the project's existing
Ollama infrastructure into the ``(messages) -> str`` callable interface
expected by ``analyze_spanish_source()``.

No Streamlit, no ReadingCoachResult, no network calls at import time.
Designed to be easy to fake in tests via the ``_raw_client`` parameter.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import requests

from infrastructure import ollama_client
from reading_coach.errors import ReadingCoachLLMError, ReadingCoachTimeoutError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class OllamaResponseError(ReadingCoachLLMError, ValueError):
    """Raised when the raw Ollama response has an unrecognised shape.

    Subclasses :class:`ReadingCoachLLMError` (part of the typed coach error
    hierarchy) and :class:`ValueError` (backward compat for Slice 1 callers
    that catch the broader type).
    """


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _normalize_response(raw: Any) -> str:
    """Extract a plain ``str`` from whatever the raw Ollama function returned.

    Handled shapes
    --------------
    - ``str``  — returned as-is (``infrastructure.ollama_client.chat`` always
      returns a str, so this is the normal production path).
    - ``dict`` with ``message.content`` — standard non-streaming
      ``/api/chat`` response shape; content string extracted.
    - ``dict`` with top-level ``content`` key — alternative shape used by
      some Ollama versions; content string extracted.
    - Anything else — raises ``OllamaResponseError`` with a clear message
      describing the actual type received.
    """
    if isinstance(raw, str):
        return raw

    if isinstance(raw, dict):
        # Standard /api/chat shape: {"message": {"content": "..."}}
        msg = raw.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"]

        # Fallback: {"content": "..."}
        content = raw.get("content")
        if isinstance(content, str):
            return content

    raise OllamaResponseError(
        f"Unexpected Ollama response shape: {type(raw).__name__!r}. "
        "Expected a str, or a dict with 'message.content' or 'content'."
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def make_ollama_coach_client(
    host: str,
    model: str,
    timeout: float,
    *,
    options: dict | None = None,
    _raw_client: Callable[[str, dict, float], Any] | None = None,
) -> Callable[[list[dict[str, str]]], str]:
    """Return a ``(messages) -> str`` callable for use with ``analyze_spanish_source()``.

    Parameters
    ----------
    host:
        Ollama base URL, e.g. ``"http://ollama:11434"``.
    model:
        Model tag, e.g. ``"qwen2.5:7b"``.
    timeout:
        Request timeout in seconds forwarded to the raw client.
    options:
        Optional Ollama sampling options forwarded inside the payload, e.g.
        ``{"temperature": 0.1, "num_ctx": 8192, "repeat_penalty": 1.15}``.
        Omitted from the payload entirely when ``None`` or empty.
    _raw_client:
        Override for the underlying chat function.  Must have the signature
        ``(host: str, payload: dict, timeout: float) -> str | dict``.
        Defaults to ``infrastructure.ollama_client.chat``.
        Intended for testing — do not pass this in production code.
    """
    raw = _raw_client if _raw_client is not None else ollama_client.chat

    def _client(messages: list[dict[str, str]]) -> str:
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",
        }
        if options:
            payload["options"] = options

        try:
            response = raw(host, payload, timeout)
        except requests.exceptions.Timeout as exc:
            raise ReadingCoachTimeoutError(
                f"LLM request timed out after {timeout}s. "
                "Try a shorter passage or increase the timeout."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ReadingCoachLLMError(
                f"LLM request failed: {exc}"
            ) from exc
        return _normalize_response(response)

    return _client
