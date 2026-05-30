"""reading_coach/session.py — Persistence-independent session models.

Two Pydantic models for the reading coach session layer:

  SavedPhrase   — a phrase the learner has saved from a coaching session.
  CoachSession  — a full reading session bundling source text, analysis
                  result, and references to saved phrases.

No MongoDB logic, no Streamlit imports, no Ollama calls.  Models serialize
cleanly to dict/JSON via :meth:`~pydantic.BaseModel.model_dump` and
:meth:`~pydantic.BaseModel.model_dump_json`.

ID generation
-------------
Both models generate UUID4 string IDs by default via :func:`_new_id`.
For tests (or any caller that needs deterministic IDs) simply pass
``session_id=`` / ``phrase_id=`` explicitly — the ``default_factory`` is
only invoked when the caller omits the field.

CEFR validation
---------------
:attr:`CoachSession.reader_level` and :attr:`SavedPhrase.difficulty_level`
raise ``pydantic.ValidationError`` for unrecognised values (unlike the
analyser layer which silently coerces to ``"B1"``).  Case is normalised to
upper-case.

Annotation density validation
------------------------------
:attr:`CoachSession.annotation_density` is validated against
:data:`VALID_ANNOTATION_DENSITIES`.  Case is normalised to lower-case.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from reading_coach.schemas import DEFAULT_COACH_LEVEL, VALID_COACH_LEVELS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ANNOTATION_DENSITIES: tuple[str, ...] = ("minimal", "balanced", "detailed")
"""Accepted values for :attr:`CoachSession.annotation_density`."""


# ---------------------------------------------------------------------------
# ID helper
# ---------------------------------------------------------------------------

def _new_id() -> str:
    """Return a new random UUID4 string.  Used as ``default_factory`` for ID fields."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# SavedPhrase
# ---------------------------------------------------------------------------

class SavedPhrase(BaseModel):
    """A phrase the learner has saved from a reading-coach session.

    ``phrase_id`` is auto-generated when omitted.  Pass an explicit value in
    tests or when reconstructing from storage.
    """

    phrase_id: str = Field(default_factory=_new_id)
    """Unique identifier for this saved phrase."""

    phrase: str
    """The exact phrase text as it appears in the source."""

    source_context: str
    """The surrounding sentence or passage from which the phrase was extracted."""

    category: str
    """Phrase category, e.g. ``'archaic vocabulary'``, ``'idiom'``, ``'grammar'``."""

    difficulty_level: str
    """CEFR level assigned to the phrase.  Must be one of :data:`VALID_COACH_LEVELS`."""

    english_meaning: Optional[str] = None
    """English translation or gloss."""

    modern_spanish_equivalent: Optional[str] = None
    """Modern Spanish paraphrase of the phrase."""

    grammar_note: Optional[str] = None
    """Relevant grammar observation."""

    learner_tip: Optional[str] = None
    """Practical tip for the learner."""

    session_id: Optional[str] = None
    """Back-reference to the :class:`CoachSession` this phrase came from."""

    chunk_id: Optional[str] = None
    """Back-reference to the chunk within a multi-chunk session, if applicable."""

    @field_validator("difficulty_level", mode="before")
    @classmethod
    def validate_difficulty_level(cls, v: object) -> str:
        """Accept any cased CEFR level; reject unrecognised values."""
        if isinstance(v, str) and v.strip().upper() in VALID_COACH_LEVELS:
            return v.strip().upper()
        raise ValueError(
            f"difficulty_level must be one of {VALID_COACH_LEVELS!r}, got {v!r}"
        )


# ---------------------------------------------------------------------------
# CoachSession
# ---------------------------------------------------------------------------

class CoachSession(BaseModel):
    """A complete reading-coach session.

    Bundles the source text, analysis result, and metadata about the reading
    session.  The ``result`` field is typed ``Optional[Any]`` to accommodate
    both :class:`~reading_coach.schemas.ReadingCoachResult` (single chunk) and
    :class:`~reading_coach.schemas.MultiChunkAnalysisResult` (multi-chunk)
    without creating a circular import.

    ``session_id`` is auto-generated when omitted.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str = Field(default_factory=_new_id)
    """Unique identifier for this session."""

    title: Optional[str] = None
    """Human-readable title for the session (e.g. Chapter 1)."""

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    """UTC timestamp when the session was created."""

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    """UTC timestamp when the session was last modified."""

    source_title: Optional[str] = None
    """Title of the source work (e.g. ``'Don Quijote'``)."""

    source_author: Optional[str] = None
    """Author of the source work (e.g. ``'Cervantes'``)."""

    original_spanish: str
    """The verbatim Spanish source text for this session."""

    reader_level: str = DEFAULT_COACH_LEVEL
    """CEFR level of the learner.  Must be one of :data:`VALID_COACH_LEVELS`."""

    annotation_density: str = "balanced"
    """Annotation density used for this session.
    Must be one of :data:`VALID_ANNOTATION_DENSITIES`."""

    prompt_version: str
    """The prompt version that produced :attr:`result`."""

    result: Optional[Any] = None
    """Analysis result for the session.  May be a
    :class:`~reading_coach.schemas.ReadingCoachResult` or a
    :class:`~reading_coach.schemas.MultiChunkAnalysisResult`, or ``None``
    when the session has not yet been analysed."""

    saved_phrase_ids: List[str] = Field(default_factory=list)
    """Ordered list of :attr:`SavedPhrase.phrase_id` values attached to this session."""

    @field_validator("reader_level", mode="before")
    @classmethod
    def validate_reader_level(cls, v: object) -> str:
        """Accept any cased CEFR level; reject unrecognised values."""
        if isinstance(v, str) and v.strip().upper() in VALID_COACH_LEVELS:
            return v.strip().upper()
        raise ValueError(
            f"reader_level must be one of {VALID_COACH_LEVELS!r}, got {v!r}"
        )

    @field_validator("annotation_density", mode="before")
    @classmethod
    def validate_annotation_density(cls, v: object) -> str:
        """Accept any cased density name; reject unrecognised values."""
        if isinstance(v, str) and v.strip().lower() in VALID_ANNOTATION_DENSITIES:
            return v.strip().lower()
        raise ValueError(
            f"annotation_density must be one of {VALID_ANNOTATION_DENSITIES!r}, got {v!r}"
        )
