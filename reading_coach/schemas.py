"""
reading_coach/schemas.py — Pydantic data models for the Spanish Source Reading Coach.

Completely independent of Streamlit, Ollama, and the English→Spanish translation
schemas in app.py. Safe to import and test without any external services.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CEFR level constants
# ---------------------------------------------------------------------------

VALID_COACH_LEVELS: tuple[str, ...] = ("A1", "A2", "B1", "B2", "C1", "C2")
DEFAULT_COACH_LEVEL: str = "B1"


def _coerce_level(value: object, field_name: str = "level") -> str:
    """Normalize a CEFR level string; fall back to DEFAULT_COACH_LEVEL.

    Accepts:
    - Exact match (any case): "b2" → "B2"
    - Substring prefix: "B2 (upper-intermediate)" → "B2", "C1 advanced" → "C1"
    - Anything unrecognizable → DEFAULT_COACH_LEVEL
    """
    if not isinstance(value, str) or not value.strip():
        logger.warning("Unrecognised %s value %r — defaulting to %r.", field_name, value, DEFAULT_COACH_LEVEL)
        return DEFAULT_COACH_LEVEL

    upper = value.strip().upper()

    # Exact match first.
    if upper in VALID_COACH_LEVELS:
        return upper

    # Substring scan: "B2 (upper-intermediate)" → "B2".
    for level in VALID_COACH_LEVELS:
        if upper.startswith(level):
            return level

    logger.warning("Unrecognised %s value %r — defaulting to %r.", field_name, value, DEFAULT_COACH_LEVEL)
    return DEFAULT_COACH_LEVEL


# ---------------------------------------------------------------------------
# DifficultPhrase
# ---------------------------------------------------------------------------

class DifficultPhrase(BaseModel):
    """A single phrase flagged as difficult, with coaching annotations."""

    phrase: str
    category: str
    difficulty_level: str
    why_difficult: str
    modern_spanish_equivalent: Optional[str] = None
    english_meaning: Optional[str] = None
    grammar_note: Optional[str] = None
    learner_tip: Optional[str] = None

    @field_validator("difficulty_level", mode="before")
    @classmethod
    def coerce_difficulty_level(cls, v: object) -> str:
        return _coerce_level(v, "difficulty_level")


# ---------------------------------------------------------------------------
# GrammarNote
# ---------------------------------------------------------------------------

class GrammarNote(BaseModel):
    """A grammar pattern observed in the source text."""

    topic: str
    explanation: str
    example_from_text: Optional[str] = None


# ---------------------------------------------------------------------------
# ComprehensionQuestion
# ---------------------------------------------------------------------------

class ComprehensionQuestion(BaseModel):
    """A reading-comprehension question about the source passage."""

    question: str
    answer_hint: Optional[str] = None


# ---------------------------------------------------------------------------
# ReadingCoachResult
# ---------------------------------------------------------------------------

class ReadingCoachResult(BaseModel):
    """Top-level result produced by the Spanish Source Reading Coach for one chunk."""

    original_spanish: str
    modern_spanish: Optional[str] = None
    english_gloss: Optional[str] = None
    overall_level: str = DEFAULT_COACH_LEVEL
    """CEFR level for the passage.  Defaults to :data:`DEFAULT_COACH_LEVEL` ("B1")
    when the LLM omits the field, consistent with the coerce-not-raise policy
    applied to unrecognised values."""
    difficult_phrases: List[DifficultPhrase] = Field(default_factory=list)
    grammar_notes: List[GrammarNote] = Field(default_factory=list)
    comprehension_question: Optional[ComprehensionQuestion] = None

    @field_validator("overall_level", mode="before")
    @classmethod
    def coerce_overall_level(cls, v: object) -> str:
        return _coerce_level(v, "overall_level")


# ---------------------------------------------------------------------------
# ChunkAnalysisResult
# ---------------------------------------------------------------------------

class ChunkAnalysisResult(BaseModel):
    """Per-chunk result bundled from one :func:`analyze_spanish_source` call.

    ``analysis`` holds the :class:`~reading_coach.analyzer.AnalysisResult`
    dataclass when the chunk succeeded, or ``None`` when it failed.  The
    ``error`` field carries the exception message on failure.

    ``analysis`` is typed as ``Optional[Any]`` to avoid a circular import
    (``analyzer.py`` imports from this module).  Callers that need the typed
    result can narrow with ``isinstance(chunk.analysis, AnalysisResult)``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk_index: int
    """Zero-based position of this chunk in the full passage."""

    total_chunks: int
    """Total number of chunks the passage was split into."""

    chunk_text: str
    """The original Spanish text for this chunk."""

    analysis: Optional[Any] = None
    """Populated on success; ``None`` on failure."""

    error: Optional[str] = None
    """Exception message when analysis failed; ``None`` on success."""

    # --- Added in slice 4 ---

    chunk_id: str = ""
    """Stable identifier in the form ``chunk_NNNN``; empty string when not set by the
    orchestrator (e.g. objects constructed directly in tests)."""

    checker_result: Optional[Any] = None
    """The :class:`~reading_coach.checker.CoachCheckResult` for this chunk, or
    ``None`` when the chunk failed.  Typed ``Optional[Any]`` to avoid a circular
    import (checker.py imports ReadingCoachResult from this module)."""

    status: str = ""
    """Checker status for successful chunks (``'passed'``, ``'warning'``,
    ``'failed'``), or ``'error'`` when the LLM/parse step raised.  Empty string
    when set by legacy code that does not populate it."""

    @property
    def succeeded(self) -> bool:
        """``True`` when :attr:`analysis` is populated (chunk analysed successfully)."""
        return self.analysis is not None

    @property
    def index(self) -> int:
        """Alias for :attr:`chunk_index` (added in slice 4)."""
        return self.chunk_index

    @property
    def source_text(self) -> str:
        """Alias for :attr:`chunk_text` (added in slice 4)."""
        return self.chunk_text

    @property
    def error_message(self) -> Optional[str]:
        """Alias for :attr:`error` (added in slice 4)."""
        return self.error


# ---------------------------------------------------------------------------
# MultiChunkAnalysisResult
# ---------------------------------------------------------------------------

class MultiChunkAnalysisResult(BaseModel):
    """Aggregate result for a full (potentially multi-chunk) analysis pass.

    Returned by :func:`reading_coach.multi_chunk_analyzer.analyze_spanish_source_chunks`.
    Holds the original Spanish verbatim, per-chunk results, and aggregate
    counts so callers need not iterate :attr:`chunks` for common queries.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    original_spanish: str
    """The verbatim source text passed to the analysis function."""

    chunks: List[ChunkAnalysisResult]
    """Per-chunk results in chunk-index order."""

    prompt_version: str
    """The prompt version that produced these results (traceable to source)."""

    total_chunks: int
    """Number of chunks the passage was split into."""

    successful_chunks: int
    """Number of chunks that were analysed without error."""

    failed_chunks: int
    """Number of chunks that could not be analysed (LLM or parse error)."""

    # --- Added in slice 4 ---

    checker_summary: Optional[str] = None
    """Human-readable summary of checker outcomes across all chunks, e.g.
    ``'1/1 passed'``.  Populated by the orchestrator; ``None`` for objects
    constructed without it."""

    metadata: Optional[Any] = None
    """Lightweight :class:`~reading_coach.metadata.AnalysisMetadata` for this
    analysis pass.  Typed ``Optional[Any]`` to avoid a circular import.
    Populated by the orchestrator; ``None`` for objects constructed without it."""

    @property
    def all_difficult_phrases(self) -> List[DifficultPhrase]:
        """Combined list of difficult phrases from all successful chunks, in order."""
        phrases: List[DifficultPhrase] = []
        for chunk in self.chunks:
            if chunk.analysis is not None:
                phrases.extend(chunk.analysis.result.difficult_phrases)
        return phrases

    @property
    def combined_difficult_phrases(self) -> List[DifficultPhrase]:
        """Alias for :attr:`all_difficult_phrases` (added in slice 4)."""
        return self.all_difficult_phrases
