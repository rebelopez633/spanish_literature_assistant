"""
Tests for reading_coach/session.py — CoachSession and SavedPhrase models.

Acceptance criteria from Phase 3 Slice 6:
  1. Create valid CoachSession.
  2. Create valid SavedPhrase.
  3. Invalid CEFR level rejected (ValidationError).
  4. Session serialization round-trips.
  5. Saved phrase serialization round-trips.
  6. original_spanish preserved exactly.

Additional:
  - Default IDs are non-empty strings unique across instances.
  - saved_phrase_ids defaults to empty list.
  - result can be None, ReadingCoachResult, or MultiChunkAnalysisResult.
  - Optional fields default to None.
  - annotation_density validated against known values.
  - created_at / updated_at are datetimes.
  - No Streamlit, no Ollama dependencies.

Pure unit tests — no network, no external services.
"""
from __future__ import annotations

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from reading_coach.prompts import SPANISH_SOURCE_PROMPT_VERSION
from reading_coach.schemas import (
    MultiChunkAnalysisResult,
    ReadingCoachResult,
    ChunkAnalysisResult,
)
from reading_coach.session import CoachSession, SavedPhrase

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

_SPANISH = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme."


def _make_phrase(**overrides) -> SavedPhrase:
    defaults = dict(
        phrase_id="phrase-001",
        phrase="no quiero acordarme",
        source_context=_SPANISH,
        category="idiom",
        difficulty_level="B2",
    )
    defaults.update(overrides)
    return SavedPhrase(**defaults)


def _make_session(**overrides) -> CoachSession:
    defaults = dict(
        session_id="session-001",
        original_spanish=_SPANISH,
        reader_level="B1",
        annotation_density="balanced",
        prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
    )
    defaults.update(overrides)
    return CoachSession(**defaults)


def _make_reading_coach_result(text: str = _SPANISH) -> ReadingCoachResult:
    return ReadingCoachResult(original_spanish=text, overall_level="B1")


def _make_multi_chunk_result(text: str = _SPANISH) -> MultiChunkAnalysisResult:
    chunk = ChunkAnalysisResult(
        chunk_index=0,
        total_chunks=1,
        chunk_text=text,
        chunk_id="chunk_0000",
        status="passed",
    )
    return MultiChunkAnalysisResult(
        original_spanish=text,
        chunks=[chunk],
        prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
        total_chunks=1,
        successful_chunks=0,
        failed_chunks=0,
    )


# ===========================================================================
# Criterion 2 — Create valid SavedPhrase
# ===========================================================================

class TestSavedPhraseModel:
    def test_basic_construction(self):
        phrase = _make_phrase()
        assert phrase.phrase == "no quiero acordarme"

    def test_phrase_id_field(self):
        phrase = _make_phrase(phrase_id="p-xyz")
        assert phrase.phrase_id == "p-xyz"

    def test_source_context_field(self):
        phrase = _make_phrase(source_context="some context")
        assert phrase.source_context == "some context"

    def test_category_field(self):
        phrase = _make_phrase(category="archaic vocabulary")
        assert phrase.category == "archaic vocabulary"

    def test_difficulty_level_field(self):
        phrase = _make_phrase(difficulty_level="C1")
        assert phrase.difficulty_level == "C1"

    def test_optional_fields_default_none(self):
        phrase = _make_phrase()
        assert phrase.english_meaning is None
        assert phrase.modern_spanish_equivalent is None
        assert phrase.grammar_note is None
        assert phrase.learner_tip is None
        assert phrase.session_id is None
        assert phrase.chunk_id is None

    def test_optional_fields_accept_values(self):
        phrase = _make_phrase(
            english_meaning="I do not want to remember",
            modern_spanish_equivalent="no quiero recordar",
            grammar_note="idiomatic use",
            learner_tip="common literary phrase",
            session_id="s-001",
            chunk_id="chunk_0000",
        )
        assert phrase.english_meaning == "I do not want to remember"
        assert phrase.session_id == "s-001"
        assert phrase.chunk_id == "chunk_0000"

    def test_default_phrase_id_generated(self):
        phrase = SavedPhrase(
            phrase="test",
            source_context="ctx",
            category="vocabulary",
            difficulty_level="B1",
        )
        assert phrase.phrase_id
        assert isinstance(phrase.phrase_id, str)

    def test_default_ids_are_unique(self):
        p1 = SavedPhrase(phrase="a", source_context="ctx", category="c", difficulty_level="A1")
        p2 = SavedPhrase(phrase="b", source_context="ctx", category="c", difficulty_level="A1")
        assert p1.phrase_id != p2.phrase_id


# ===========================================================================
# Criterion 3 — Invalid CEFR level rejected
# ===========================================================================

class TestSavedPhraseValidation:
    def test_valid_cefr_levels_accepted(self):
        for level in ("A1", "A2", "B1", "B2", "C1", "C2"):
            phrase = _make_phrase(difficulty_level=level)
            assert phrase.difficulty_level == level

    def test_lowercase_cefr_normalized(self):
        phrase = _make_phrase(difficulty_level="b2")
        assert phrase.difficulty_level == "B2"

    def test_invalid_cefr_level_raises(self):
        with pytest.raises(ValidationError, match="difficulty_level"):
            _make_phrase(difficulty_level="D1")

    def test_empty_level_raises(self):
        with pytest.raises(ValidationError):
            _make_phrase(difficulty_level="")

    def test_garbage_level_raises(self):
        with pytest.raises(ValidationError):
            _make_phrase(difficulty_level="fluent")


# ===========================================================================
# Criterion 1 — Create valid CoachSession
# ===========================================================================

class TestCoachSessionModel:
    def test_basic_construction(self):
        session = _make_session()
        assert session.original_spanish == _SPANISH

    def test_session_id_field(self):
        session = _make_session(session_id="s-xyz")
        assert session.session_id == "s-xyz"

    def test_reader_level_field(self):
        session = _make_session(reader_level="C1")
        assert session.reader_level == "C1"

    def test_annotation_density_field(self):
        session = _make_session(annotation_density="minimal")
        assert session.annotation_density == "minimal"

    def test_prompt_version_field(self):
        session = _make_session(prompt_version="v2")
        assert session.prompt_version == "v2"

    def test_optional_fields_default_none(self):
        session = _make_session()
        assert session.title is None
        assert session.source_title is None
        assert session.source_author is None
        assert session.result is None

    def test_optional_fields_accept_values(self):
        session = _make_session(
            title="My Session",
            source_title="Don Quijote",
            source_author="Cervantes",
        )
        assert session.title == "My Session"
        assert session.source_title == "Don Quijote"
        assert session.source_author == "Cervantes"

    def test_saved_phrase_ids_default_empty(self):
        session = _make_session()
        assert session.saved_phrase_ids == []

    def test_saved_phrase_ids_accepts_list(self):
        session = _make_session(saved_phrase_ids=["p-001", "p-002"])
        assert session.saved_phrase_ids == ["p-001", "p-002"]

    def test_created_at_is_datetime(self):
        session = _make_session()
        assert isinstance(session.created_at, datetime)

    def test_updated_at_is_datetime(self):
        session = _make_session()
        assert isinstance(session.updated_at, datetime)

    def test_default_session_id_generated(self):
        session = CoachSession(
            original_spanish="test",
            reader_level="B1",
            annotation_density="balanced",
            prompt_version="v1",
        )
        assert session.session_id
        assert isinstance(session.session_id, str)

    def test_default_session_ids_unique(self):
        s1 = CoachSession(
            original_spanish="text1",
            reader_level="B1",
            annotation_density="balanced",
            prompt_version="v1",
        )
        s2 = CoachSession(
            original_spanish="text2",
            reader_level="B1",
            annotation_density="balanced",
            prompt_version="v1",
        )
        assert s1.session_id != s2.session_id

    def test_result_accepts_reading_coach_result(self):
        session = _make_session(result=_make_reading_coach_result())
        assert session.result is not None

    def test_result_accepts_multi_chunk_result(self):
        session = _make_session(result=_make_multi_chunk_result())
        assert session.result is not None

    def test_result_accepts_none(self):
        session = _make_session(result=None)
        assert session.result is None


# ===========================================================================
# Criterion 3 — Invalid CEFR level rejected (CoachSession)
# ===========================================================================

class TestCoachSessionValidation:
    def test_valid_reader_levels_accepted(self):
        for level in ("A1", "A2", "B1", "B2", "C1", "C2"):
            session = _make_session(reader_level=level)
            assert session.reader_level == level

    def test_lowercase_reader_level_normalized(self):
        session = _make_session(reader_level="c2")
        assert session.reader_level == "C2"

    def test_invalid_reader_level_raises(self):
        with pytest.raises(ValidationError, match="reader_level"):
            _make_session(reader_level="D1")

    def test_empty_reader_level_raises(self):
        with pytest.raises(ValidationError):
            _make_session(reader_level="")

    def test_garbage_reader_level_raises(self):
        with pytest.raises(ValidationError):
            _make_session(reader_level="advanced")

    def test_valid_annotation_densities_accepted(self):
        for density in ("minimal", "balanced", "detailed"):
            session = _make_session(annotation_density=density)
            assert session.annotation_density == density

    def test_invalid_annotation_density_raises(self):
        with pytest.raises(ValidationError, match="annotation_density"):
            _make_session(annotation_density="verbose")


# ===========================================================================
# Criterion 4 — Session serialization round-trips
# ===========================================================================

class TestCoachSessionSerialization:
    def test_model_dump_returns_dict(self):
        session = _make_session()
        data = session.model_dump()
        assert isinstance(data, dict)

    def test_model_dump_has_required_keys(self):
        session = _make_session()
        data = session.model_dump()
        for key in ("session_id", "original_spanish", "reader_level",
                    "annotation_density", "prompt_version", "saved_phrase_ids"):
            assert key in data, f"Missing key: {key}"

    def test_model_dump_scalar_fields_preserved(self):
        session = _make_session()
        data = session.model_dump()
        assert data["session_id"] == "session-001"
        assert data["reader_level"] == "B1"
        assert data["annotation_density"] == "balanced"

    def test_round_trip_via_model_validate(self):
        session = _make_session()
        data = session.model_dump()
        session2 = CoachSession.model_validate(data)
        assert session2.session_id == session.session_id
        assert session2.original_spanish == session.original_spanish
        assert session2.reader_level == session.reader_level

    def test_model_dump_json_does_not_crash(self):
        session = _make_session()
        json_str = session.model_dump_json()
        assert isinstance(json_str, str)

    def test_json_round_trip_scalar_fields(self):
        session = _make_session()
        json_str = session.model_dump_json()
        data = json.loads(json_str)
        assert data["session_id"] == "session-001"
        assert data["reader_level"] == "B1"

    def test_round_trip_with_optional_fields(self):
        session = _make_session(
            title="Mi sesión",
            source_title="Don Quijote",
            source_author="Cervantes",
            saved_phrase_ids=["p-001"],
        )
        data = session.model_dump()
        session2 = CoachSession.model_validate(data)
        assert session2.title == "Mi sesión"
        assert session2.source_title == "Don Quijote"
        assert session2.saved_phrase_ids == ["p-001"]


# ===========================================================================
# Criterion 5 — Saved phrase serialization round-trips
# ===========================================================================

class TestSavedPhraseSerialization:
    def test_model_dump_returns_dict(self):
        phrase = _make_phrase()
        assert isinstance(phrase.model_dump(), dict)

    def test_model_dump_has_required_keys(self):
        phrase = _make_phrase()
        data = phrase.model_dump()
        for key in ("phrase_id", "phrase", "source_context", "category", "difficulty_level"):
            assert key in data, f"Missing key: {key}"

    def test_round_trip_via_model_validate(self):
        phrase = _make_phrase(english_meaning="I do not remember")
        data = phrase.model_dump()
        phrase2 = SavedPhrase.model_validate(data)
        assert phrase2.phrase_id == phrase.phrase_id
        assert phrase2.phrase == phrase.phrase
        assert phrase2.english_meaning == phrase.english_meaning

    def test_model_dump_json_does_not_crash(self):
        phrase = _make_phrase()
        json_str = phrase.model_dump_json()
        assert isinstance(json_str, str)

    def test_json_round_trip(self):
        phrase = _make_phrase()
        data = json.loads(phrase.model_dump_json())
        assert data["phrase"] == "no quiero acordarme"
        assert data["difficulty_level"] == "B2"

    def test_optional_fields_none_in_dump(self):
        phrase = _make_phrase()
        data = phrase.model_dump()
        assert data["english_meaning"] is None
        assert data["session_id"] is None
        assert data["chunk_id"] is None


# ===========================================================================
# Criterion 6 — original_spanish preserved exactly
# ===========================================================================

class TestOriginalSpanishPreserved:
    def test_exact_string_preserved(self):
        session = _make_session(original_spanish=_SPANISH)
        assert session.original_spanish == _SPANISH

    def test_unicode_preserved(self):
        text = "¿Quién es más grande? ¡Nadie lo sabe!"
        session = _make_session(original_spanish=text)
        assert session.original_spanish == text

    def test_leading_trailing_whitespace_preserved(self):
        text = "   texto con espacios   "
        session = _make_session(original_spanish=text)
        assert session.original_spanish == text

    def test_multiline_text_preserved(self):
        text = "Primera línea.\nSegunda línea.\n\nTercera línea."
        session = _make_session(original_spanish=text)
        assert session.original_spanish == text

    def test_original_spanish_in_dump(self):
        session = _make_session(original_spanish=_SPANISH)
        data = session.model_dump()
        assert data["original_spanish"] == _SPANISH

    def test_original_spanish_survives_round_trip(self):
        session = _make_session(original_spanish=_SPANISH)
        session2 = CoachSession.model_validate(session.model_dump())
        assert session2.original_spanish == _SPANISH
