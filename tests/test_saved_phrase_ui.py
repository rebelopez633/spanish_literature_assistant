"""tests/test_saved_phrase_ui.py — UI tests for Phase 3 Slice 11.

Saved phrase UI: save, list, delete, export.

Strategy
--------
All tests use the conftest Streamlit stub — no Streamlit server, no Ollama.
pre-populate ``st.session_state`` as needed and call ``render_coach_mode``
with ``button_clicked=False`` (Analyze never fires).

Button keys
-----------
* Save phrase    : ``coach_save_phrase_{index}``   (index = 0-based position in
                   the current result's difficult_phrases list)
* Delete phrase  : ``coach_delete_phrase_{phrase_id}``

Acceptance criteria (Slice 11)
-------------------------------
1. Save phrase action stores a SavedPhrase in the repo.
2. Saved phrases render in the UI (subheader / expander / write calls).
3. Delete phrase removes it from the repo.
4. Export download contains saved phrase text.
5. No saved phrases → empty-state message is shown.
"""
from __future__ import annotations

import unittest.mock as mock

import pytest

pytestmark = pytest.mark.ui

from reading_coach.analyzer import AnalysisResult
from reading_coach.checker import CoachCheckResult, STATUS_PASSED
from reading_coach.prompts import SPANISH_SOURCE_PROMPT_VERSION
from reading_coach.schemas import DifficultPhrase, ReadingCoachResult
from reading_coach.session import SavedPhrase
from reading_coach.session_repository import InMemoryCoachSessionRepository

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SOURCE = "Había una vez un hidalgo de los de lanza en astillero."
_HOST = "http://localhost:11434"
_MODEL = "qwen2.5:7b"
_PHRASE_TEXT = "lanza en astillero"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dp(
    phrase: str = _PHRASE_TEXT,
    english_meaning: str | None = "lance in a rack",
    modern_spanish_equivalent: str | None = "lanza en un soporte",
    grammar_note: str | None = "Locative noun phrase.",
    learner_tip: str | None = "Common in Golden Age texts.",
) -> DifficultPhrase:
    return DifficultPhrase(
        phrase=phrase,
        category="archaic",
        difficulty_level="B2",
        why_difficult="Old military term.",
        english_meaning=english_meaning,
        modern_spanish_equivalent=modern_spanish_equivalent,
        grammar_note=grammar_note,
        learner_tip=learner_tip,
    )


def _make_result(dps: list[DifficultPhrase] | None = None) -> ReadingCoachResult:
    return ReadingCoachResult(
        original_spanish=_SOURCE,
        overall_level="B2",
        difficult_phrases=dps if dps is not None else [_make_dp()],
    )


def _make_analysis(dps: list[DifficultPhrase] | None = None) -> AnalysisResult:
    result = _make_result(dps)
    check = CoachCheckResult(status=STATUS_PASSED, summary="ok", issues=[])
    return AnalysisResult(result=result, check=check, raw_response="{}")


def _make_saved_phrase(phrase: str = _PHRASE_TEXT, phrase_id: str = "fixed-id-1") -> SavedPhrase:
    return SavedPhrase(
        phrase_id=phrase_id,
        phrase=phrase,
        source_context=_SOURCE,
        category="archaic",
        difficulty_level="B2",
        english_meaning="lance in a rack",
        modern_spanish_equivalent="lanza en un soporte",
        grammar_note="Locative noun phrase.",
        learner_tip="Common in Golden Age texts.",
    )


def _make_button_side_effect(clicking_key: str | None = None):
    def _side_effect(*args, **kwargs):
        return kwargs.get("key") == clicking_key
    return _side_effect


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _PhraseTestBase:
    _HOST = _HOST
    _MODEL = _MODEL

    def setup_method(self):
        import streamlit as st
        st.session_state._data.clear()
        for name in (
            "text_area", "button", "selectbox", "checkbox", "error",
            "spinner", "caption", "metric", "subheader", "download_button",
            "text_input", "success", "divider", "expander", "rerun", "write",
        ):
            m = getattr(st, name, None)
            if m is not None and hasattr(m, "reset_mock"):
                m.reset_mock(return_value=True, side_effect=True)
        st.text_area.return_value = _SOURCE
        st.text_input.return_value = ""
        st.button.return_value = False
        st.selectbox.side_effect = ["B1", "balanced"]
        st.checkbox.side_effect = [True, True]

    def _call_render(self) -> None:
        from ui.spanish_source_mode import render_coach_mode
        render_coach_mode(ollama_host=self._HOST, ollama_model=self._MODEL)

    def _get_repo(self) -> InMemoryCoachSessionRepository:
        import streamlit as st
        return st.session_state.coach_repo


# ===========================================================================
# 1. Save phrase action
# ===========================================================================

class TestSavePhrase(_PhraseTestBase):
    """Acceptance criterion 1: save phrase button stores a SavedPhrase in repo."""

    def test_save_phrase_button_stores_phrase_in_repo(self):
        import streamlit as st
        st.session_state.coach_analysis = _make_analysis()
        st.button.side_effect = _make_button_side_effect("coach_save_phrase_0")
        st.selectbox.side_effect = ["B1", "balanced"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        phrases = self._get_repo().list_saved_phrases()
        assert len(phrases) == 1

    def test_save_phrase_stores_correct_phrase_text(self):
        import streamlit as st
        st.session_state.coach_analysis = _make_analysis([_make_dp(_PHRASE_TEXT)])
        st.button.side_effect = _make_button_side_effect("coach_save_phrase_0")
        st.selectbox.side_effect = ["B1", "balanced"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        phrases = self._get_repo().list_saved_phrases()
        assert phrases[0].phrase == _PHRASE_TEXT

    def test_save_phrase_stores_source_context(self):
        import streamlit as st
        st.text_area.return_value = _SOURCE
        st.session_state.coach_analysis = _make_analysis()
        st.button.side_effect = _make_button_side_effect("coach_save_phrase_0")
        st.selectbox.side_effect = ["B1", "balanced"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        phrases = self._get_repo().list_saved_phrases()
        assert phrases[0].source_context == _SOURCE

    def test_save_second_phrase_separately(self):
        import streamlit as st
        dp1 = _make_dp("lanza en astillero")
        dp2 = _make_dp("hidalgo")
        st.session_state.coach_analysis = _make_analysis([dp1, dp2])
        st.button.side_effect = _make_button_side_effect("coach_save_phrase_1")
        st.selectbox.side_effect = ["B1", "balanced"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        phrases = self._get_repo().list_saved_phrases()
        assert len(phrases) == 1
        assert phrases[0].phrase == "hidalgo"

    def test_no_save_button_without_analysis(self):
        """No analysis in state → no save-phrase buttons rendered."""
        import streamlit as st
        self._call_render()
        save_calls = [
            c for c in st.button.call_args_list
            if str(c.kwargs.get("key", "")).startswith("coach_save_phrase_")
        ]
        assert len(save_calls) == 0

    def test_save_phrase_english_meaning_stored(self):
        import streamlit as st
        dp = _make_dp(english_meaning="test gloss")
        st.session_state.coach_analysis = _make_analysis([dp])
        st.button.side_effect = _make_button_side_effect("coach_save_phrase_0")
        st.selectbox.side_effect = ["B1", "balanced"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        phrases = self._get_repo().list_saved_phrases()
        assert phrases[0].english_meaning == "test gloss"


# ===========================================================================
# 2. Saved phrases render in UI
# ===========================================================================

class TestSavedPhrasesRender(_PhraseTestBase):
    """Acceptance criterion 2: saved phrases appear in the UI."""

    def _seed_repo(self, *phrases: SavedPhrase) -> InMemoryCoachSessionRepository:
        import streamlit as st
        repo = InMemoryCoachSessionRepository()
        for p in phrases:
            repo.save_phrase(p)
        st.session_state.coach_repo = repo
        return repo

    def test_empty_state_message_shown_when_no_saved_phrases(self):
        """Criterion 5: empty-state caption when no saved phrases."""
        import streamlit as st
        self._call_render()
        captions = [str(c.args[0]) for c in st.caption.call_args_list if c.args]
        assert any("no saved phrase" in cap.lower() for cap in captions)

    def test_saved_phrases_subheader_shown_when_phrases_present(self):
        import streamlit as st
        self._seed_repo(_make_saved_phrase())
        self._call_render()
        subheader_calls = [str(c.args[0]) for c in st.subheader.call_args_list if c.args]
        assert any("saved phrase" in s.lower() for s in subheader_calls)

    def test_phrase_text_appears_in_expander_label(self):
        import streamlit as st
        self._seed_repo(_make_saved_phrase(_PHRASE_TEXT))
        self._call_render()
        labels = [str(c.args[0]) for c in st.expander.call_args_list if c.args]
        assert any(_PHRASE_TEXT in lbl for lbl in labels)

    def test_two_phrases_render_at_least_two_expanders(self):
        import streamlit as st
        self._seed_repo(
            _make_saved_phrase("lanza en astillero", "id-1"),
            _make_saved_phrase("hidalgo", "id-2"),
        )
        self._call_render()
        phrase_expanders = [
            c for c in st.expander.call_args_list
            if c.args and ("lanza" in str(c.args[0]) or "hidalgo" in str(c.args[0]))
        ]
        assert len(phrase_expanders) >= 2

    def test_delete_button_rendered_per_saved_phrase(self):
        import streamlit as st
        sp = _make_saved_phrase(phrase_id="pid-abc")
        self._seed_repo(sp)
        self._call_render()
        delete_calls = [
            c for c in st.button.call_args_list
            if c.kwargs.get("key") == "coach_delete_phrase_pid-abc"
        ]
        assert len(delete_calls) == 1

    def test_english_meaning_written_for_phrase(self):
        import streamlit as st
        sp = _make_saved_phrase()
        self._seed_repo(sp)
        self._call_render()
        write_calls = [str(c.args[0]) for c in st.write.call_args_list if c.args]
        assert any("lance in a rack" in w for w in write_calls)

    def test_modern_spanish_written_for_phrase(self):
        import streamlit as st
        sp = _make_saved_phrase()
        self._seed_repo(sp)
        self._call_render()
        write_calls = [str(c.args[0]) for c in st.write.call_args_list if c.args]
        assert any("lanza en un soporte" in w for w in write_calls)

    def test_grammar_note_written_for_phrase(self):
        import streamlit as st
        sp = _make_saved_phrase()
        self._seed_repo(sp)
        self._call_render()
        write_calls = [str(c.args[0]) for c in st.write.call_args_list if c.args]
        assert any("Locative noun phrase" in w for w in write_calls)

    def test_learner_tip_written_for_phrase(self):
        import streamlit as st
        sp = _make_saved_phrase()
        self._seed_repo(sp)
        self._call_render()
        write_calls = [str(c.args[0]) for c in st.write.call_args_list if c.args]
        assert any("Golden Age" in w for w in write_calls)

    def test_source_context_written_for_phrase(self):
        import streamlit as st
        sp = _make_saved_phrase()
        self._seed_repo(sp)
        self._call_render()
        write_calls = [str(c.args[0]) for c in st.write.call_args_list if c.args]
        assert any(_SOURCE in w for w in write_calls)


# ===========================================================================
# 3. Delete phrase path
# ===========================================================================

class TestDeletePhrase(_PhraseTestBase):
    """Acceptance criterion 3: delete removes phrase from repo."""

    def _seed_repo(self, *phrases: SavedPhrase) -> InMemoryCoachSessionRepository:
        import streamlit as st
        repo = InMemoryCoachSessionRepository()
        for p in phrases:
            repo.save_phrase(p)
        st.session_state.coach_repo = repo
        return repo

    def test_delete_removes_phrase_from_repo(self):
        import streamlit as st
        sp = _make_saved_phrase(phrase_id="del-1")
        repo = self._seed_repo(sp)
        st.button.side_effect = _make_button_side_effect("coach_delete_phrase_del-1")
        st.selectbox.side_effect = ["B1", "balanced"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        assert repo.list_saved_phrases() == []

    def test_delete_calls_rerun(self):
        import streamlit as st
        sp = _make_saved_phrase(phrase_id="del-2")
        self._seed_repo(sp)
        st.button.side_effect = _make_button_side_effect("coach_delete_phrase_del-2")
        st.selectbox.side_effect = ["B1", "balanced"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        st.rerun.assert_called()

    def test_delete_one_leaves_other_intact(self):
        import streamlit as st
        sp1 = _make_saved_phrase("keep this", "id-keep")
        sp2 = _make_saved_phrase("delete this", "id-del")
        repo = self._seed_repo(sp1, sp2)
        st.button.side_effect = _make_button_side_effect("coach_delete_phrase_id-del")
        st.selectbox.side_effect = ["B1", "balanced"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        remaining = repo.list_saved_phrases()
        assert len(remaining) == 1
        assert remaining[0].phrase_id == "id-keep"


# ===========================================================================
# 4. Export download contains phrase text
# ===========================================================================

class TestSavedPhrasesExport(_PhraseTestBase):
    """Acceptance criterion 4: export download contains saved phrase text."""

    def _seed_repo(self, *phrases: SavedPhrase) -> InMemoryCoachSessionRepository:
        import streamlit as st
        repo = InMemoryCoachSessionRepository()
        for p in phrases:
            repo.save_phrase(p)
        st.session_state.coach_repo = repo
        return repo

    def test_export_download_button_rendered_when_phrases_present(self):
        import streamlit as st
        self._seed_repo(_make_saved_phrase())
        self._call_render()
        export_calls = [
            c for c in st.download_button.call_args_list
            if "phrase" in str(c.kwargs.get("key", "")).lower()
            or "phrase" in str(c.kwargs.get("file_name", "")).lower()
        ]
        assert len(export_calls) >= 1

    def test_export_content_contains_phrase_text(self):
        import streamlit as st
        self._seed_repo(_make_saved_phrase(_PHRASE_TEXT))
        self._call_render()
        export_calls = [
            c for c in st.download_button.call_args_list
            if "phrase" in str(c.kwargs.get("key", "")).lower()
            or "phrase" in str(c.kwargs.get("file_name", "")).lower()
        ]
        assert export_calls, "No phrase export download button rendered"
        data = export_calls[0].kwargs.get("data", "")
        assert _PHRASE_TEXT in data

    def test_export_content_contains_english_meaning(self):
        import streamlit as st
        sp = _make_saved_phrase()
        self._seed_repo(sp)
        self._call_render()
        export_calls = [
            c for c in st.download_button.call_args_list
            if "phrase" in str(c.kwargs.get("key", "")).lower()
            or "phrase" in str(c.kwargs.get("file_name", "")).lower()
        ]
        data = export_calls[0].kwargs.get("data", "")
        assert "lance in a rack" in data

    def test_no_export_button_when_no_saved_phrases(self):
        """Empty phrase list → no export button rendered for phrases."""
        import streamlit as st
        self._call_render()
        export_calls = [
            c for c in st.download_button.call_args_list
            if "phrase" in str(c.kwargs.get("key", "")).lower()
            or "phrase" in str(c.kwargs.get("file_name", "")).lower()
        ]
        assert len(export_calls) == 0

    def test_export_mime_type_is_markdown(self):
        import streamlit as st
        self._seed_repo(_make_saved_phrase())
        self._call_render()
        export_calls = [
            c for c in st.download_button.call_args_list
            if "phrase" in str(c.kwargs.get("key", "")).lower()
            or "phrase" in str(c.kwargs.get("file_name", "")).lower()
        ]
        mime = export_calls[0].kwargs.get("mime", "")
        assert mime == "text/markdown"
