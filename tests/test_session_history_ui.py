"""tests/test_session_history_ui.py — UI tests for Phase 3 Slice 10.

Session history UI: save, list, reload, delete.

Strategy
--------
All tests use the conftest Streamlit stub — no real Streamlit server, no Ollama.
``analyze_spanish_source`` is never called; pre-populate ``st.session_state``.

Buttons are identified by their ``key=`` kwarg.  ``st.button.side_effect`` is set
to a function that returns ``True`` only for the key under test; all other
button calls return ``False``.  This lets a single ``render_coach_mode`` call
exercise exactly one button's code path.

Acceptance criteria (Slice 10)
-------------------------------
1. Save button path — clicking "Save session" stores a CoachSession in the repo.
2. List sessions path — sessions in the repo appear in the sidebar.
3. Reload session path — clicking "Reload" restores the session into session_state.
4. Delete session path — clicking "Delete" removes the session from the repo.
"""
from __future__ import annotations

import unittest.mock as mock

import pytest

pytestmark = pytest.mark.ui

from reading_coach.analyzer import AnalysisResult
from reading_coach.checker import CoachCheckResult, STATUS_PASSED
from reading_coach.prompts import SPANISH_SOURCE_PROMPT_VERSION
from reading_coach.schemas import ReadingCoachResult
from reading_coach.session import CoachSession
from reading_coach.session_repository import InMemoryCoachSessionRepository

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SOURCE = "Había una vez un hidalgo de los de lanza en astillero."
_HOST = "http://localhost:11434"
_MODEL = "qwen2.5:7b"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analysis(source: str = _SOURCE) -> AnalysisResult:
    result = ReadingCoachResult(original_spanish=source, overall_level="B1")
    check = CoachCheckResult(status=STATUS_PASSED, summary="All checks passed.", issues=[])
    return AnalysisResult(result=result, check=check, raw_response="{}")


def _make_session(
    source: str = _SOURCE,
    title: str | None = None,
    reader_level: str = "B1",
    session_id: str | None = None,
) -> CoachSession:
    kwargs: dict = dict(
        original_spanish=source,
        reader_level=reader_level,
        annotation_density="balanced",
        prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
        result=_make_analysis(source),
        title=title,
    )
    if session_id is not None:
        kwargs["session_id"] = session_id
    return CoachSession(**kwargs)


def _make_button_side_effect(clicking_key: str | None = None):
    """Return a side_effect that returns ``True`` only for the given button key."""
    def _side_effect(*args, **kwargs):
        return kwargs.get("key") == clicking_key
    return _side_effect


# ---------------------------------------------------------------------------
# Base class — widget setup mirrors _DownloadTestBase in test_study_notes_download.py
# ---------------------------------------------------------------------------

class _HistoryTestBase:
    _HOST = _HOST
    _MODEL = _MODEL

    def setup_method(self):
        import streamlit as st
        st.session_state._data.clear()
        for name in (
            "text_area", "button", "selectbox", "checkbox", "error",
            "spinner", "caption", "metric", "subheader", "download_button",
            "text_input", "success", "divider", "expander", "rerun",
        ):
            m = getattr(st, name, None)
            if m is not None and hasattr(m, "reset_mock"):
                m.reset_mock(return_value=True, side_effect=True)
        # Sensible defaults for all tests.
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
# 1. Save button path
# ===========================================================================

class TestSaveSession(_HistoryTestBase):
    """Acceptance criterion 1: clicking Save stores a CoachSession in the repo."""

    def test_save_button_stores_session_in_repo(self):
        import streamlit as st
        st.session_state.coach_analysis = _make_analysis()
        st.button.side_effect = _make_button_side_effect("coach_save_session_btn")
        st.selectbox.side_effect = ["B1", "balanced"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        sessions = self._get_repo().list_sessions()
        assert len(sessions) == 1

    def test_save_stores_correct_source_text(self):
        import streamlit as st
        st.text_area.return_value = _SOURCE
        st.session_state.coach_analysis = _make_analysis(_SOURCE)
        st.button.side_effect = _make_button_side_effect("coach_save_session_btn")
        st.selectbox.side_effect = ["B1", "balanced"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        sessions = self._get_repo().list_sessions()
        assert sessions[0].original_spanish == _SOURCE

    def test_save_stores_reader_level(self):
        import streamlit as st
        st.session_state.coach_analysis = _make_analysis()
        st.button.side_effect = _make_button_side_effect("coach_save_session_btn")
        st.selectbox.side_effect = ["C1", "balanced"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        sessions = self._get_repo().list_sessions()
        assert sessions[0].reader_level == "C1"

    def test_save_stores_annotation_density(self):
        import streamlit as st
        st.session_state.coach_analysis = _make_analysis()
        st.button.side_effect = _make_button_side_effect("coach_save_session_btn")
        st.selectbox.side_effect = ["B1", "detailed"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        sessions = self._get_repo().list_sessions()
        assert sessions[0].annotation_density == "detailed"

    def test_save_preserves_prompt_version(self):
        import streamlit as st
        st.session_state.coach_analysis = _make_analysis()
        st.button.side_effect = _make_button_side_effect("coach_save_session_btn")
        st.selectbox.side_effect = ["B1", "balanced"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        sessions = self._get_repo().list_sessions()
        assert sessions[0].prompt_version == SPANISH_SOURCE_PROMPT_VERSION

    def test_save_sets_saved_session_id_in_state(self):
        import streamlit as st
        st.session_state.coach_analysis = _make_analysis()
        st.button.side_effect = _make_button_side_effect("coach_save_session_btn")
        st.selectbox.side_effect = ["B1", "balanced"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        assert "coach_saved_session_id" in st.session_state

    def test_save_button_not_rendered_without_analysis(self):
        """No coach_analysis in state → save button must not appear."""
        import streamlit as st
        self._call_render()
        save_calls = [
            c for c in st.button.call_args_list
            if c.kwargs.get("key") == "coach_save_session_btn"
        ]
        assert len(save_calls) == 0

    def test_save_stores_title_when_provided(self):
        import streamlit as st
        st.text_input.return_value = "Mi primera sesión"
        st.session_state.coach_analysis = _make_analysis()
        st.button.side_effect = _make_button_side_effect("coach_save_session_btn")
        st.selectbox.side_effect = ["B1", "balanced"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        sessions = self._get_repo().list_sessions()
        assert sessions[0].title == "Mi primera sesión"


# ===========================================================================
# 2. List sessions path
# ===========================================================================

class TestListSessions(_HistoryTestBase):
    """Acceptance criterion 2: sessions in the repo appear as sidebar expanders."""

    def test_empty_repo_shows_no_saved_sessions_caption(self):
        import streamlit as st
        self._call_render()
        captions = [str(c.args[0]) for c in st.caption.call_args_list if c.args]
        assert any("no saved" in cap.lower() for cap in captions)

    def test_one_session_renders_expander(self):
        import streamlit as st
        repo = InMemoryCoachSessionRepository()
        repo.save_session(_make_session())
        st.session_state.coach_repo = repo
        self._call_render()
        assert st.expander.call_count >= 1

    def test_session_title_used_as_expander_label(self):
        import streamlit as st
        repo = InMemoryCoachSessionRepository()
        repo.save_session(_make_session(title="Mi primera sesión"))
        st.session_state.coach_repo = repo
        self._call_render()
        labels = [str(c.args[0]) for c in st.expander.call_args_list if c.args]
        assert any("Mi primera sesión" in lbl for lbl in labels)

    def test_source_text_used_as_label_when_no_title(self):
        import streamlit as st
        repo = InMemoryCoachSessionRepository()
        repo.save_session(_make_session(title=None))
        st.session_state.coach_repo = repo
        self._call_render()
        labels = [str(c.args[0]) for c in st.expander.call_args_list if c.args]
        assert any(_SOURCE[:20] in lbl for lbl in labels)

    def test_two_sessions_render_at_least_two_expanders(self):
        import streamlit as st
        repo = InMemoryCoachSessionRepository()
        repo.save_session(_make_session(title="Session A"))
        repo.save_session(_make_session(title="Session B"))
        st.session_state.coach_repo = repo
        self._call_render()
        assert st.expander.call_count >= 2

    def test_reload_button_rendered_per_session(self):
        import streamlit as st
        session = _make_session()
        repo = InMemoryCoachSessionRepository()
        repo.save_session(session)
        st.session_state.coach_repo = repo
        self._call_render()
        reload_calls = [
            c for c in st.button.call_args_list
            if c.kwargs.get("key") == f"coach_reload_{session.session_id}"
        ]
        assert len(reload_calls) == 1

    def test_delete_button_rendered_per_session(self):
        import streamlit as st
        session = _make_session()
        repo = InMemoryCoachSessionRepository()
        repo.save_session(session)
        st.session_state.coach_repo = repo
        self._call_render()
        delete_calls = [
            c for c in st.button.call_args_list
            if c.kwargs.get("key") == f"coach_delete_{session.session_id}"
        ]
        assert len(delete_calls) == 1


# ===========================================================================
# 3. Reload session path
# ===========================================================================

class TestReloadSession(_HistoryTestBase):
    """Acceptance criterion 3: clicking Reload restores the session."""

    def _setup_repo_with_session(self, session: CoachSession) -> InMemoryCoachSessionRepository:
        import streamlit as st
        repo = InMemoryCoachSessionRepository()
        repo.save_session(session)
        st.session_state.coach_repo = repo
        return repo

    def test_reload_sets_coach_analysis_in_state(self):
        import streamlit as st
        session = _make_session()
        self._setup_repo_with_session(session)
        st.button.side_effect = _make_button_side_effect(
            f"coach_reload_{session.session_id}"
        )
        st.selectbox.side_effect = ["B1", "balanced"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        assert "coach_analysis" in st.session_state

    def test_reload_restores_correct_original_spanish(self):
        import streamlit as st
        custom = "Texto de prueba para recargar."
        session = _make_session(source=custom)
        self._setup_repo_with_session(session)
        st.button.side_effect = _make_button_side_effect(
            f"coach_reload_{session.session_id}"
        )
        st.selectbox.side_effect = ["B1", "balanced"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        restored: AnalysisResult = st.session_state.coach_analysis
        assert restored.result.original_spanish == custom

    def test_reload_sets_source_text_in_state(self):
        import streamlit as st
        session = _make_session(source="El texto original.")
        self._setup_repo_with_session(session)
        st.button.side_effect = _make_button_side_effect(
            f"coach_reload_{session.session_id}"
        )
        st.selectbox.side_effect = ["B1", "balanced"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        assert st.session_state.get("coach_source_text") == "El texto original."

    def test_reload_sets_reader_level_in_state(self):
        import streamlit as st
        session = _make_session(reader_level="C2")
        self._setup_repo_with_session(session)
        st.button.side_effect = _make_button_side_effect(
            f"coach_reload_{session.session_id}"
        )
        st.selectbox.side_effect = ["B1", "balanced"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        assert st.session_state.get("coach_reader_level") == "C2"

    def test_reload_calls_rerun(self):
        import streamlit as st
        session = _make_session()
        self._setup_repo_with_session(session)
        st.button.side_effect = _make_button_side_effect(
            f"coach_reload_{session.session_id}"
        )
        st.selectbox.side_effect = ["B1", "balanced"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        st.rerun.assert_called()


# ===========================================================================
# 4. Delete session path
# ===========================================================================

class TestDeleteSession(_HistoryTestBase):
    """Acceptance criterion 4: clicking Delete removes the session from the repo."""

    def _setup_repo_with_session(self, session: CoachSession) -> InMemoryCoachSessionRepository:
        import streamlit as st
        repo = InMemoryCoachSessionRepository()
        repo.save_session(session)
        st.session_state.coach_repo = repo
        return repo

    def test_delete_removes_session_from_repo(self):
        import streamlit as st
        session = _make_session()
        repo = self._setup_repo_with_session(session)
        st.button.side_effect = _make_button_side_effect(
            f"coach_delete_{session.session_id}"
        )
        st.selectbox.side_effect = ["B1", "balanced"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        assert repo.get_session(session.session_id) is None

    def test_delete_calls_rerun(self):
        import streamlit as st
        session = _make_session()
        self._setup_repo_with_session(session)
        st.button.side_effect = _make_button_side_effect(
            f"coach_delete_{session.session_id}"
        )
        st.selectbox.side_effect = ["B1", "balanced"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        st.rerun.assert_called()

    def test_delete_of_one_session_leaves_others_intact(self):
        import streamlit as st
        repo = InMemoryCoachSessionRepository()
        s1 = _make_session(title="Keep this")
        s2 = _make_session(title="Delete this")
        repo.save_session(s1)
        repo.save_session(s2)
        st.session_state.coach_repo = repo
        st.button.side_effect = _make_button_side_effect(
            f"coach_delete_{s2.session_id}"
        )
        st.selectbox.side_effect = ["B1", "balanced"]
        st.checkbox.side_effect = [True, True]
        self._call_render()
        assert repo.get_session(s1.session_id) is not None
        assert repo.get_session(s2.session_id) is None
