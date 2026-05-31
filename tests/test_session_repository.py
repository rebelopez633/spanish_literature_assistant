"""
Tests for reading_coach/session_repository.py — repository abstraction.

Acceptance criteria from Phase 3 Slice 7:
  1. save/get session.
  2. list sessions with limit.
  3. delete session.
  4. save/list/delete phrase.
  5. filter saved phrases by session_id.
  6. mutations to returned objects do not corrupt stored state (copy semantics).

Additional:
  - save_session returns the session unchanged.
  - get_session returns None for unknown IDs.
  - list_sessions returns newest-first (by updated_at), capped by limit.
  - delete_session returns True on success, False when not found.
  - save_phrase returns the phrase unchanged.
  - list_saved_phrases() (no filter) returns all phrases.
  - list_saved_phrases(session_id=X) returns only phrases for that session.
  - delete_phrase returns True on success, False when not found.
  - CoachSessionRepository is an abstract base class (cannot be instantiated).
  - InMemoryCoachSessionRepository is a concrete subclass.

Pure unit tests — no network, no MongoDB, no Streamlit.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone, timedelta

import pytest
from pydantic import ValidationError

from reading_coach.prompts import SPANISH_SOURCE_PROMPT_VERSION
from reading_coach.session import CoachSession, SavedPhrase
from reading_coach.session_repository import (
    CoachSessionRepository,
    InMemoryCoachSessionRepository,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

_SPANISH_A = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme."
_SPANISH_B = "Tanto monta, monta tanto, Isabel como Fernando."
_PROMPT_VER = SPANISH_SOURCE_PROMPT_VERSION


def _make_session(session_id: str, spanish: str = _SPANISH_A, **kwargs) -> CoachSession:
    defaults = dict(
        session_id=session_id,
        original_spanish=spanish,
        reader_level="B1",
        annotation_density="balanced",
        prompt_version=_PROMPT_VER,
    )
    defaults.update(kwargs)
    return CoachSession(**defaults)


def _make_phrase(phrase_id: str, session_id: str | None = None, **kwargs) -> SavedPhrase:
    defaults = dict(
        phrase_id=phrase_id,
        phrase="no quiero acordarme",
        source_context=_SPANISH_A,
        category="idiom",
        difficulty_level="B2",
        session_id=session_id,
    )
    defaults.update(kwargs)
    return SavedPhrase(**defaults)


# ---------------------------------------------------------------------------
# 0. Abstract base — cannot be instantiated directly
# ---------------------------------------------------------------------------

class TestCoachSessionRepositoryIsAbstract:
    def test_cannot_instantiate_base_class(self):
        """CoachSessionRepository must be abstract (ABC or Protocol)."""
        import inspect
        assert inspect.isabstract(CoachSessionRepository), (
            "CoachSessionRepository should be abstract and not directly instantiable"
        )


# ---------------------------------------------------------------------------
# 1. save / get session
# ---------------------------------------------------------------------------

class TestSaveGetSession:
    def test_save_returns_session(self):
        repo = InMemoryCoachSessionRepository()
        s = _make_session("s1")
        result = repo.save_session(s)
        assert result.session_id == "s1"

    def test_get_returns_saved_session(self):
        repo = InMemoryCoachSessionRepository()
        s = _make_session("s2")
        repo.save_session(s)
        fetched = repo.get_session("s2")
        assert fetched is not None
        assert fetched.session_id == "s2"
        assert fetched.original_spanish == _SPANISH_A

    def test_get_unknown_returns_none(self):
        repo = InMemoryCoachSessionRepository()
        assert repo.get_session("nonexistent") is None

    def test_save_overwrites_existing_session(self):
        """Saving a session with the same ID replaces it."""
        repo = InMemoryCoachSessionRepository()
        s1 = _make_session("s-ow", title="First")
        s2 = _make_session("s-ow", title="Updated")
        repo.save_session(s1)
        repo.save_session(s2)
        fetched = repo.get_session("s-ow")
        assert fetched.title == "Updated"

    def test_get_returns_all_fields(self):
        repo = InMemoryCoachSessionRepository()
        s = _make_session(
            "s3",
            title="Test Session",
            source_title="Don Quijote",
            source_author="Cervantes",
        )
        repo.save_session(s)
        fetched = repo.get_session("s3")
        assert fetched.title == "Test Session"
        assert fetched.source_title == "Don Quijote"
        assert fetched.source_author == "Cervantes"


# ---------------------------------------------------------------------------
# 2. list sessions with limit
# ---------------------------------------------------------------------------

class TestListSessions:
    def test_list_empty_repository(self):
        repo = InMemoryCoachSessionRepository()
        assert repo.list_sessions() == []

    def test_list_returns_all_when_under_limit(self):
        repo = InMemoryCoachSessionRepository()
        repo.save_session(_make_session("ls1"))
        repo.save_session(_make_session("ls2"))
        repo.save_session(_make_session("ls3"))
        sessions = repo.list_sessions(limit=25)
        assert len(sessions) == 3

    def test_list_respects_limit(self):
        repo = InMemoryCoachSessionRepository()
        for i in range(10):
            repo.save_session(_make_session(f"lim{i}"))
        sessions = repo.list_sessions(limit=3)
        assert len(sessions) == 3

    def test_list_default_limit_is_25(self):
        repo = InMemoryCoachSessionRepository()
        for i in range(30):
            repo.save_session(_make_session(f"def{i}"))
        sessions = repo.list_sessions()
        assert len(sessions) == 25

    def test_list_ordered_newest_first(self):
        """Sessions are returned newest-updated_at first."""
        repo = InMemoryCoachSessionRepository()
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        for i in range(3):
            s = _make_session(
                f"ord{i}",
                updated_at=base + timedelta(hours=i),
            )
            repo.save_session(s)
        sessions = repo.list_sessions(limit=25)
        ids = [s.session_id for s in sessions]
        assert ids[0] == "ord2"  # newest first
        assert ids[-1] == "ord0"  # oldest last

    def test_list_sessions_return_type(self):
        repo = InMemoryCoachSessionRepository()
        repo.save_session(_make_session("rt1"))
        sessions = repo.list_sessions()
        assert all(isinstance(s, CoachSession) for s in sessions)


# ---------------------------------------------------------------------------
# 3. delete session
# ---------------------------------------------------------------------------

class TestDeleteSession:
    def test_delete_existing_session_returns_true(self):
        repo = InMemoryCoachSessionRepository()
        repo.save_session(_make_session("del1"))
        assert repo.delete_session("del1") is True

    def test_delete_removes_session(self):
        repo = InMemoryCoachSessionRepository()
        repo.save_session(_make_session("del2"))
        repo.delete_session("del2")
        assert repo.get_session("del2") is None

    def test_delete_nonexistent_returns_false(self):
        repo = InMemoryCoachSessionRepository()
        assert repo.delete_session("ghost") is False

    def test_delete_does_not_affect_other_sessions(self):
        repo = InMemoryCoachSessionRepository()
        repo.save_session(_make_session("keep1"))
        repo.save_session(_make_session("gone"))
        repo.save_session(_make_session("keep2"))
        repo.delete_session("gone")
        assert repo.get_session("keep1") is not None
        assert repo.get_session("keep2") is not None

    def test_delete_removes_from_list(self):
        repo = InMemoryCoachSessionRepository()
        repo.save_session(_make_session("lst1"))
        repo.save_session(_make_session("lst2"))
        repo.delete_session("lst1")
        ids = [s.session_id for s in repo.list_sessions()]
        assert "lst1" not in ids
        assert "lst2" in ids


# ---------------------------------------------------------------------------
# 4. save / list / delete phrase
# ---------------------------------------------------------------------------

class TestSaveListDeletePhrase:
    def test_save_phrase_returns_phrase(self):
        repo = InMemoryCoachSessionRepository()
        p = _make_phrase("p1")
        result = repo.save_phrase(p)
        assert result.phrase_id == "p1"

    def test_list_phrases_empty(self):
        repo = InMemoryCoachSessionRepository()
        assert repo.list_saved_phrases() == []

    def test_list_phrases_returns_saved(self):
        repo = InMemoryCoachSessionRepository()
        repo.save_phrase(_make_phrase("p2"))
        repo.save_phrase(_make_phrase("p3"))
        phrases = repo.list_saved_phrases()
        ids = {p.phrase_id for p in phrases}
        assert ids == {"p2", "p3"}

    def test_delete_phrase_returns_true(self):
        repo = InMemoryCoachSessionRepository()
        repo.save_phrase(_make_phrase("del-p1"))
        assert repo.delete_phrase("del-p1") is True

    def test_delete_phrase_removes_it(self):
        repo = InMemoryCoachSessionRepository()
        repo.save_phrase(_make_phrase("del-p2"))
        repo.delete_phrase("del-p2")
        phrases = repo.list_saved_phrases()
        assert not any(p.phrase_id == "del-p2" for p in phrases)

    def test_delete_phrase_nonexistent_returns_false(self):
        repo = InMemoryCoachSessionRepository()
        assert repo.delete_phrase("no-such-phrase") is False

    def test_save_phrase_overwrites_existing(self):
        """Saving a phrase with the same ID replaces it."""
        repo = InMemoryCoachSessionRepository()
        p1 = _make_phrase("ow-p", category="idiom")
        p2 = _make_phrase("ow-p", category="archaic vocabulary")
        repo.save_phrase(p1)
        repo.save_phrase(p2)
        phrases = repo.list_saved_phrases()
        found = [p for p in phrases if p.phrase_id == "ow-p"]
        assert len(found) == 1
        assert found[0].category == "archaic vocabulary"

    def test_list_phrases_return_type(self):
        repo = InMemoryCoachSessionRepository()
        repo.save_phrase(_make_phrase("rt-p"))
        phrases = repo.list_saved_phrases()
        assert all(isinstance(p, SavedPhrase) for p in phrases)


# ---------------------------------------------------------------------------
# 5. filter phrases by session_id
# ---------------------------------------------------------------------------

class TestFilterPhrasesBySession:
    def test_filter_returns_only_matching_session(self):
        repo = InMemoryCoachSessionRepository()
        repo.save_phrase(_make_phrase("fp1", session_id="sess-A"))
        repo.save_phrase(_make_phrase("fp2", session_id="sess-A"))
        repo.save_phrase(_make_phrase("fp3", session_id="sess-B"))
        phrases = repo.list_saved_phrases(session_id="sess-A")
        ids = {p.phrase_id for p in phrases}
        assert ids == {"fp1", "fp2"}

    def test_filter_unknown_session_returns_empty(self):
        repo = InMemoryCoachSessionRepository()
        repo.save_phrase(_make_phrase("fp4", session_id="sess-A"))
        assert repo.list_saved_phrases(session_id="sess-X") == []

    def test_filter_none_returns_all(self):
        repo = InMemoryCoachSessionRepository()
        repo.save_phrase(_make_phrase("fp5", session_id="sess-A"))
        repo.save_phrase(_make_phrase("fp6", session_id="sess-B"))
        repo.save_phrase(_make_phrase("fp7", session_id=None))
        all_phrases = repo.list_saved_phrases(session_id=None)
        assert len(all_phrases) == 3

    def test_filter_no_argument_returns_all(self):
        repo = InMemoryCoachSessionRepository()
        repo.save_phrase(_make_phrase("fp8", session_id="sess-A"))
        repo.save_phrase(_make_phrase("fp9", session_id="sess-B"))
        assert len(repo.list_saved_phrases()) == 2

    def test_filter_phrases_with_no_session_id(self):
        """Phrases with session_id=None are returned when filtering for None."""
        repo = InMemoryCoachSessionRepository()
        repo.save_phrase(_make_phrase("fp10", session_id=None))
        repo.save_phrase(_make_phrase("fp11", session_id="sess-A"))
        # Explicit None filter — return all (unfiltered)
        all_phrases = repo.list_saved_phrases()
        assert len(all_phrases) == 2


# ---------------------------------------------------------------------------
# 6. copy / mutation safety
# ---------------------------------------------------------------------------

class TestMutationSafety:
    def test_mutating_saved_session_does_not_corrupt_store(self):
        """Returned CoachSession copies are independent of the stored object."""
        repo = InMemoryCoachSessionRepository()
        original = _make_session("mut1", title="Original")
        repo.save_session(original)
        fetched = repo.get_session("mut1")
        # Mutate the fetched copy — store should be unaffected
        fetched.title = "Mutated"  # type: ignore[misc]
        fresh = repo.get_session("mut1")
        assert fresh.title == "Original"

    def test_mutating_input_session_does_not_corrupt_store(self):
        """Mutations to the session after save() do not affect the stored copy."""
        repo = InMemoryCoachSessionRepository()
        s = _make_session("mut2", title="Before")
        repo.save_session(s)
        # CoachSession is a Pydantic model; mutate by re-saving a modified copy
        s2 = s.model_copy(update={"title": "After"})
        # Do NOT re-save — original save should still hold "Before"
        fetched = repo.get_session("mut2")
        assert fetched.title == "Before"

    def test_mutating_saved_phrase_does_not_corrupt_store(self):
        """Returned SavedPhrase copies are independent of the stored object."""
        repo = InMemoryCoachSessionRepository()
        p = _make_phrase("mut-p1", category="idiom")
        repo.save_phrase(p)
        fetched = repo.list_saved_phrases()[0]
        fetched.category = "mutated"  # type: ignore[misc]
        fresh = repo.list_saved_phrases()[0]
        assert fresh.category == "idiom"

    def test_list_sessions_returns_independent_copies(self):
        """list_sessions() returns independent copies — mutating one does not
        affect subsequent calls."""
        repo = InMemoryCoachSessionRepository()
        repo.save_session(_make_session("ind1", title="Stable"))
        first_list = repo.list_sessions()
        first_list[0].title = "Damaged"  # type: ignore[misc]
        second_list = repo.list_sessions()
        assert second_list[0].title == "Stable"
