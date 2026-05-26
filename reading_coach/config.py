"""
reading_coach/config.py — Environment-variable defaults for the Reading Coach mode.

No Streamlit, no Ollama, no I/O.  Pure dataclass + factory function.

Environment variables (all optional — built-in defaults used when unset or blank):

    READING_COACH_DEFAULT_LEVEL         CEFR level string (A1–C2).  Default: B1
    READING_COACH_MAX_ANNOTATIONS       Integer ≥ 0.                 Default: 7
    READING_COACH_INCLUDE_ENGLISH_GLOSS true/false.                  Default: true
    READING_COACH_INCLUDE_MODERN_SPANISH true/false.                  Default: true
    READING_COACH_TIMEOUT_SECONDS       Float > 0.                   Default: 120.0
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from reading_coach.schemas import DEFAULT_COACH_LEVEL, VALID_COACH_LEVELS

# ---------------------------------------------------------------------------
# Built-in defaults (used when env vars are absent or invalid)
# ---------------------------------------------------------------------------

_DEFAULT_LEVEL: str = DEFAULT_COACH_LEVEL   # "B1"
_DEFAULT_MAX_ANNOTATIONS: int = 7
_DEFAULT_INCLUDE_ENGLISH_GLOSS: bool = True
_DEFAULT_INCLUDE_MODERN_SPANISH: bool = True
_DEFAULT_TIMEOUT_SECONDS: float = 120.0


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoachSettings:
    """Reading Coach configuration parsed from environment variables.

    Instances are immutable (``frozen=True``) so they can be passed around
    freely without risk of accidental mutation.
    """

    default_level: str
    """Default CEFR reader level for new sessions (e.g. ``"B1"``)."""

    max_annotations: int
    """Upper limit on the number of difficult-phrase annotations the checker
    will accept before issuing a warning.  Passed to ``CoachCheckerConfig``."""

    include_english_gloss: bool
    """Whether the English gloss section is requested by default."""

    include_modern_spanish: bool
    """Whether the Modern Spanish paraphrase is requested by default."""

    timeout_seconds: float
    """LLM request timeout in seconds (``READING_COACH_TIMEOUT_SECONDS``)."""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_bool(key: str, default: bool) -> bool:
    """Return True/False from an env var; treat absent/blank as *default*."""
    raw = os.getenv(key, "").strip().lower()
    if not raw:
        return default
    return raw not in ("false", "0", "no", "off")


def _parse_level(key: str, default: str) -> str:
    """Return a valid CEFR level from an env var; fall back to *default*."""
    raw = os.getenv(key, "").strip().upper()
    if raw in VALID_COACH_LEVELS:
        return raw
    return default


def _parse_int(key: str, default: int) -> int:
    """Return an integer from an env var; fall back to *default* on error."""
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_positive_float(key: str, default: float) -> float:
    """Return a positive float from an env var; fall back to *default* on error.

    Zero and negative values are treated as invalid (a non-positive timeout
    has no sensible meaning) and also fall back to *default*.
    """
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
        return value if value > 0 else default
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_coach_settings() -> CoachSettings:
    """Build a :class:`CoachSettings` from the current process environment.

    All variables are optional.  Unset, blank, or invalid values fall back to
    the built-in defaults defined at the top of this module.

    This function is side-effect–free and can be called multiple times safely.
    """
    return CoachSettings(
        default_level=_parse_level(
            "READING_COACH_DEFAULT_LEVEL", _DEFAULT_LEVEL
        ),
        max_annotations=_parse_int(
            "READING_COACH_MAX_ANNOTATIONS", _DEFAULT_MAX_ANNOTATIONS
        ),
        include_english_gloss=_parse_bool(
            "READING_COACH_INCLUDE_ENGLISH_GLOSS", _DEFAULT_INCLUDE_ENGLISH_GLOSS
        ),
        include_modern_spanish=_parse_bool(
            "READING_COACH_INCLUDE_MODERN_SPANISH", _DEFAULT_INCLUDE_MODERN_SPANISH
        ),
        timeout_seconds=_parse_positive_float(
            "READING_COACH_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT_SECONDS
        ),
    )
