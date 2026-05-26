"""
core/app_modes.py — App mode constants and validation helpers.

No Streamlit, Ollama, or external-service imports — safe to test in isolation.
"""
from __future__ import annotations

import os
from typing import Any

# ---------------------------------------------------------------------------
# Mode label constants
# ---------------------------------------------------------------------------

PARALLEL_READER_MODE: str = "English → Spanish Parallel Reader"
READING_COACH_MODE: str = "Spanish Source Reading Coach"

# Ordered list — the first entry is the default.
APP_MODES: list[str] = [PARALLEL_READER_MODE, READING_COACH_MODE]

DEFAULT_MODE: str = APP_MODES[0]

# Fast membership set used by resolve_mode.
_VALID_MODES: frozenset[str] = frozenset(APP_MODES)


# ---------------------------------------------------------------------------
# resolve_mode — normalize / validate a raw mode value
# ---------------------------------------------------------------------------

def resolve_mode(value: Any) -> str:
    """Return a valid mode string, falling back to DEFAULT_MODE on bad input.

    Accepts:
    - A valid mode string — returned unchanged.
    - A dict (e.g. st.session_state proxy) — reads the ``"app_mode"`` key.
    - Anything else (None, int, unknown string, whitespace) — returns DEFAULT_MODE.
    """
    if isinstance(value, dict):
        value = value.get("app_mode")

    if isinstance(value, str) and value.strip() in _VALID_MODES:
        return value.strip()

    return DEFAULT_MODE


# ---------------------------------------------------------------------------
# get_default_mode — env-aware startup default
# ---------------------------------------------------------------------------

def get_default_mode() -> str:
    """Return the app's startup default mode, respecting ``APP_DEFAULT_MODE``.

    When ``APP_DEFAULT_MODE`` is set to a valid mode string in the process
    environment, that mode is returned.  Any unrecognised or blank value falls
    back to :data:`DEFAULT_MODE` (the parallel-reader, preserving legacy
    behaviour).
    """
    raw = os.getenv("APP_DEFAULT_MODE", "").strip()
    if not raw:
        return DEFAULT_MODE
    return resolve_mode(raw)
