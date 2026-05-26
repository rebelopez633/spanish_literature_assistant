"""reading_coach/errors.py — Typed exception hierarchy for the Reading Coach.

All error types in this module are pure Python — no Streamlit, no Ollama, no
network access — so they can be imported at any layer without side-effects.

Hierarchy
---------
ReadingCoachError                         # base: any coach-level failure
├── ReadingCoachLLMError                  # the LLM call itself failed
│   └── ReadingCoachTimeoutError          # LLM call timed out
└── ReadingCoachParseError                # JSON extraction / schema validation failed
    └── ReadingCoachValidationError       # Pydantic schema validation failure

Each class exposes a ``user_message`` property — a short, plain-English string
suitable for display in a UI (e.g. ``st.error(exc.user_message)``) without any
stack-trace details.
"""
from __future__ import annotations


class ReadingCoachError(Exception):
    """Base class for all Reading Coach errors.

    ``str(exc)`` returns the full diagnostic message (suitable for logging).
    ``exc.user_message`` returns the same string trimmed, falling back to a
    safe default when the message is empty.
    """

    _default_user_message: str = "An unexpected Reading Coach error occurred."

    @property
    def user_message(self) -> str:
        """Short user-facing description, safe to show in the UI."""
        return str(self).strip() or self._default_user_message


class ReadingCoachLLMError(ReadingCoachError):
    """Raised when the LLM call itself fails (network error, bad response shape, etc.)."""

    _default_user_message = (
        "The model request failed. "
        "Check that Ollama is running and the model is loaded."
    )


class ReadingCoachTimeoutError(ReadingCoachLLMError):
    """Raised when the LLM call exceeds the configured timeout."""

    _default_user_message = (
        "The model did not respond in time. "
        "Try a shorter passage or a faster model."
    )


class ReadingCoachParseError(ReadingCoachError):
    """Raised when the LLM response cannot be parsed into a ReadingCoachResult."""

    _default_user_message = (
        "The model response could not be parsed. Try the analysis again."
    )


class ReadingCoachValidationError(ReadingCoachParseError):
    """Raised when the JSON was found but failed Pydantic schema validation.

    Subclasses :class:`ReadingCoachParseError` so existing ``except
    ReadingCoachParseError`` clauses continue to catch schema failures.
    """

    _default_user_message = (
        "The model response did not match the expected structure. "
        "Try the analysis again."
    )
