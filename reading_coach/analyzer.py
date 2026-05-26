"""
reading_coach/analyzer.py — Orchestration layer for the Spanish Source Reading Coach.

analyze_spanish_source() wires together:
    source text → build_coach_prompt → llm_client → parse JSON → ReadingCoachResult
                                                                → check_coach_result

The LLM dependency is injected as a plain callable so callers can swap in a
real Ollama wrapper, a streaming variant, or a fake for unit tests without
touching this module.

No Streamlit, Ollama, or network imports here.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Callable, Optional

from pydantic import ValidationError

from reading_coach.checker import CoachCheckResult, CoachCheckerConfig, check_coach_result
from reading_coach.prompts import build_coach_prompt
from reading_coach.schemas import ReadingCoachResult

logger = logging.getLogger(__name__)

# Regex to strip Markdown code fences some models emit around JSON.
_FENCE_START = re.compile(r"^```(?:json)?\s*", re.MULTILINE)
_FENCE_END = re.compile(r"```\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class CoachAnalysisError(Exception):
    """Raised when the LLM response cannot be parsed into a ReadingCoachResult.

    The message always describes what went wrong so callers can surface it to
    the user without inspecting the cause chain.
    """


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """Bundled output of one analyze_spanish_source() call."""

    result: ReadingCoachResult
    check: CoachCheckResult
    raw_response: str  # verbatim string returned by llm_client


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Remove Markdown code fences and surrounding whitespace."""
    stripped = _FENCE_START.sub("", text.strip())
    stripped = _FENCE_END.sub("", stripped.strip()).strip()
    return stripped


def _parse_result(raw: str) -> ReadingCoachResult:
    """Attempt to parse *raw* into a ReadingCoachResult.

    Strategy (mirrors translate_chunk in app.py):
    1. Strip Markdown fences and whitespace.
    2. Try model_validate_json on the cleaned content directly.
    3. If that fails, search for the outermost {...} substring and retry.
    4. On any remaining failure, raise CoachAnalysisError with a clear message.
    """
    content = _strip_fences(raw)

    if not content:
        raise CoachAnalysisError(
            "LLM returned an empty response; cannot parse ReadingCoachResult."
        )

    # Attempt 1: direct parse.
    try:
        return ReadingCoachResult.model_validate_json(content)
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        first_exc = exc

    # Attempt 2: extract the outermost JSON object.
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return ReadingCoachResult.model_validate_json(content[start:end])
        except (ValidationError, ValueError, json.JSONDecodeError) as inner:
            raise CoachAnalysisError(
                f"Failed to parse LLM response as ReadingCoachResult: {inner}"
            ) from inner

    raise CoachAnalysisError(
        f"Failed to parse LLM response as ReadingCoachResult: {first_exc}"
    ) from first_exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_spanish_source(
    source_text: str,
    *,
    reader_level: str = "B1",
    annotation_density: str = "balanced",
    include_english_gloss: bool = True,
    include_modern_spanish: bool = True,
    llm_client: Callable[[list[dict[str, str]]], str],
    checker_config: Optional[CoachCheckerConfig] = None,
) -> AnalysisResult:
    """Orchestrate one full reading-coach analysis pass.

    Parameters
    ----------
    source_text:
        Original Spanish passage to annotate.
    reader_level:
        CEFR level of the target learner (``"B1"`` by default).
    annotation_density:
        ``"minimal"``, ``"balanced"``, or ``"detailed"``.
    include_english_gloss:
        Whether to request an English gloss of the full passage.
    include_modern_spanish:
        Whether to request a modern Spanish paraphrase of the full passage.
    llm_client:
        A callable that accepts a messages list (``list[dict[str, str]]``) and
        returns the model's raw string response.  Injected so tests can pass a
        fake without touching this module.
    checker_config:
        Optional ``CoachCheckerConfig`` forwarded to ``check_coach_result``.
        When ``None`` the checker's safe defaults are used.

    Returns
    -------
    AnalysisResult
        Bundles the parsed ``ReadingCoachResult``, the ``CoachCheckResult``
        from the deterministic checker, and the verbatim ``raw_response``.

    Raises
    ------
    CoachAnalysisError
        If the LLM response cannot be parsed into a valid ``ReadingCoachResult``.
    """
    messages = build_coach_prompt(
        source_text,
        reader_level=reader_level,
        annotation_density=annotation_density,
        include_english_gloss=include_english_gloss,
        include_modern_spanish=include_modern_spanish,
    )

    raw = llm_client(messages)

    logger.debug(
        "analyze_spanish_source: reader_level=%s density=%s raw_len=%d",
        reader_level,
        annotation_density,
        len(raw),
    )

    coach_result = _parse_result(raw)

    check = check_coach_result(source_text, coach_result, config=checker_config)

    return AnalysisResult(result=coach_result, check=check, raw_response=raw)
