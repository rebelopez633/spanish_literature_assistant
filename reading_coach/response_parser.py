"""reading_coach/response_parser.py — Robust JSON parser for ReadingCoachResult.

Converts raw LLM output into a validated ReadingCoachResult instance.  Handles
the common output variants that chat models produce:

* Clean JSON (most reliable models).
* JSON wrapped in markdown code fences (triple-backtick json blocks).
* JSON preceded or followed by commentary / preamble prose.
* ``<think>...</think>`` preambles from reasoning models.

All parsing failures surface as :class:`ReadingCoachParseError`.

No Streamlit, no Ollama, no network access.
"""
from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from reading_coach.errors import ReadingCoachParseError, ReadingCoachValidationError
from reading_coach.schemas import DEFAULT_COACH_LEVEL, ReadingCoachResult

logger = logging.getLogger(__name__)

# Regex patterns to strip Markdown code fences that some models emit.
_FENCE_START = re.compile(r"^```(?:json)?\s*", re.MULTILINE)
_FENCE_END = re.compile(r"```\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# ReadingCoachParseError is defined in reading_coach.errors and imported above.
# It remains importable from this module for backward compatibility.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Remove Markdown code fences and surrounding whitespace."""
    stripped = _FENCE_START.sub("", text.strip())
    stripped = _FENCE_END.sub("", stripped.strip()).strip()
    return stripped


def _validate_dict(obj: dict, fallback_level: str = DEFAULT_COACH_LEVEL) -> ReadingCoachResult:
    """Validate a parsed dict against ReadingCoachResult; wrap ValidationError.

    When *obj* has no ``overall_level`` key, *fallback_level* is injected so
    the result reflects the caller's context (e.g. the reader's selected level)
    rather than the schema's hardcoded constant.
    """
    if "overall_level" not in obj:
        logger.warning(
            "LLM response missing 'overall_level'; defaulting to %r.", fallback_level
        )
        obj = {**obj, "overall_level": fallback_level}
    try:
        return ReadingCoachResult.model_validate(obj)
    except ValidationError as exc:
        raise ReadingCoachValidationError(
            f"Failed to parse LLM response as ReadingCoachResult: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_reading_coach_response(
    raw_text: str,
    *,
    fallback_level: str = DEFAULT_COACH_LEVEL,
) -> ReadingCoachResult:
    """Parse *raw_text* into a :class:`ReadingCoachResult`.

    Parameters
    ----------
    raw_text:
        Verbatim string returned by the LLM (may contain fences, prose, etc.).
    fallback_level:
        CEFR level to use when the LLM omits ``overall_level`` entirely.
        Pass the reader's selected level so the fallback is meaningful rather
        than always defaulting to ``"B1"``.

    Returns
    -------
    ReadingCoachResult
        Fully validated result instance.

    Raises
    ------
    ReadingCoachParseError
        If no valid JSON object can be extracted, or if the parsed object does
        not satisfy the ReadingCoachResult schema.
    """
    if not raw_text or not raw_text.strip():
        raise ReadingCoachParseError(
            "Empty response; cannot parse ReadingCoachResult."
        )

    # Strategy A: strip markdown fences, try json.loads on the cleaned text.
    cleaned = _strip_fences(raw_text).strip()
    if cleaned:
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            obj = None

        if isinstance(obj, dict):
            return _validate_dict(obj, fallback_level)
        # obj is None (parse error) or a non-dict JSON value → fall through.

    # Strategy B: locate the first '{' and use json.JSONDecoder.raw_decode()
    # to extract the first complete JSON object, ignoring surrounding prose.
    first_brace = raw_text.find("{")
    if first_brace >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw_text, first_brace)
        except json.JSONDecodeError:
            obj = None

        if isinstance(obj, dict):
            return _validate_dict(obj, fallback_level)

    raise ReadingCoachParseError(
        f"Failed to parse LLM response as ReadingCoachResult: "
        f"no JSON object found. Raw (first 200 chars): {raw_text[:200]!r}"
    )
