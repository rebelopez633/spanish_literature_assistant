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

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from reading_coach.checker import CoachCheckResult, CoachCheckerConfig, check_coach_result
from reading_coach.errors import ReadingCoachError
from reading_coach.prompts import build_coach_prompt
from reading_coach.response_parser import ReadingCoachParseError, parse_reading_coach_response
from reading_coach.retry import RetryConfig, get_retry_config, with_retry
from reading_coach.schemas import ReadingCoachResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class CoachAnalysisError(ReadingCoachError):
    """Raised when the LLM response cannot be parsed into a ReadingCoachResult.

    Subclasses :class:`ReadingCoachError` so the Streamlit UI can catch either
    this specific type or the broader ``ReadingCoachError`` base.  The message
    always describes what went wrong so callers can surface it to the user
    without inspecting the cause chain.
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
    retry_config: RetryConfig | None = None,
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

    _config = retry_config if retry_config is not None else get_retry_config()

    def _attempt() -> tuple[str, ReadingCoachResult]:
        _raw = llm_client(messages)
        _result = parse_reading_coach_response(_raw)
        return _raw, _result

    try:
        raw, coach_result = with_retry(_attempt, config=_config)
    except ReadingCoachParseError as exc:
        raise CoachAnalysisError(str(exc)) from exc
    # ReadingCoachLLMError (after retries exhausted) propagates uncaught.

    logger.debug(
        "analyze_spanish_source: reader_level=%s density=%s raw_len=%d",
        reader_level,
        annotation_density,
        len(raw),
    )

    check = check_coach_result(source_text, coach_result, config=checker_config)

    return AnalysisResult(result=coach_result, check=check, raw_response=raw)
