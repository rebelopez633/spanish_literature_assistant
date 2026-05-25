"""
reading_coach/schemas.py — Pydantic data models for the Spanish Source Reading Coach.

Completely independent of Streamlit, Ollama, and the English→Spanish translation
schemas in app.py. Safe to import and test without any external services.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

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
    overall_level: str
    difficult_phrases: List[DifficultPhrase] = Field(default_factory=list)
    grammar_notes: List[GrammarNote] = Field(default_factory=list)
    comprehension_question: Optional[ComprehensionQuestion] = None

    @field_validator("overall_level", mode="before")
    @classmethod
    def coerce_overall_level(cls, v: object) -> str:
        return _coerce_level(v, "overall_level")
