"""Tests for render_coach_mode integration — mocked analyzer, no real Ollama.

The conftest Streamlit stub lets us import and call render_coach_mode
without a running Streamlit server.  Streamlit widget calls are already
MagicMock instances from the stub; we set return_value/side_effect to
simulate user input and button clicks.

analyze_spanish_source is patched at the module boundary so no LLM or
network I/O occurs.  make_ollama_client is patched where needed to
verify timeout/host wiring without touching the real adapter.
"""
from __future__ import annotations

import unittest.mock as mock

import pytest

from reading_coach.analyzer import AnalysisResult, CoachAnalysisError
from reading_coach.checker import CoachCheckResult, STATUS_PASSED, STATUS_WARNING
from reading_coach.errors import ReadingCoachLLMError, ReadingCoachTimeoutError
from reading_coach.schemas import ReadingCoachResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SOURCE = "Había una vez un hidalgo de los de lanza en astillero."


def _make_result(source: str = _SOURCE) -> AnalysisResult:
    """Minimal passing AnalysisResult suitable for _display_coach_result."""
    result = ReadingCoachResult(original_spanish=source, overall_level="B1")
    check = CoachCheckResult(status=STATUS_PASSED, summary="All checks passed.", issues=[])
    return AnalysisResult(result=result, check=check, raw_response="{}")


def _make_warning_result(source: str = _SOURCE) -> AnalysisResult:
    result = ReadingCoachResult(original_spanish=source, overall_level="B1")
    check = CoachCheckResult(
        status=STATUS_WARNING,
        summary="1 warning found.",
        issues=["Annotation count 3 exceeds limit of 2."],
    )
    return AnalysisResult(result=result, check=check, raw_response="{}")


# ---------------------------------------------------------------------------
# Shared setup mixin
# ---------------------------------------------------------------------------

class _RenderCoachBase:
    """Shared setup/helpers for tests that call render_coach_mode."""

    _HOST = "http://localhost:11434"
    _MODEL = "qwen2.5:7b"

    def setup_method(self):
        import streamlit as st
        # Clear session state between tests.
        st.session_state._data.clear()
        # Reset widget mocks so stale side_effects don't bleed across tests.
        for name in ("text_area", "button", "selectbox", "checkbox", "error",
                     "spinner", "caption", "metric", "subheader"):
            m = getattr(st, name, None)
            if m is not None and hasattr(m, "reset_mock"):
                m.reset_mock(return_value=True, side_effect=True)

    def _configure_widgets(
        self,
        source: str = _SOURCE,
        reader_level: str = "B1",
        density: str = "balanced",
        include_modern_spanish: bool = True,
        include_english_gloss: bool = True,
        button_clicked: bool = True,
    ) -> None:
        """Pre-configure the Streamlit stub widgets to simulate user input."""
        import streamlit as st
        st.text_area.return_value = source
        # selectbox is called twice: reader_level then annotation_density
        st.selectbox.side_effect = [reader_level, density]
        # checkbox is called twice: include_modern_spanish then include_english_gloss
        st.checkbox.side_effect = [include_modern_spanish, include_english_gloss]
        st.button.return_value = button_clicked

    def _call_render(self, **kwargs) -> None:
        from ui.spanish_source_mode import render_coach_mode
        render_coach_mode(ollama_host=self._HOST, ollama_model=self._MODEL)

    def _patch_analyzer(self, return_value=None, side_effect=None):
        """Return a context manager that patches analyze_spanish_source."""
        p = mock.patch("ui.spanish_source_mode.analyze_spanish_source")
        m = p.start()
        if side_effect is not None:
            m.side_effect = side_effect
        else:
            m.return_value = return_value if return_value is not None else _make_result()
        return p, m


# ===========================================================================
# TestRenderCoachModeSuccessPath
# ===========================================================================

class TestRenderCoachModeSuccessPath(_RenderCoachBase):
    """Button clicked + text present → analyzer runs and result is stored."""

    def test_analyzer_called_once_on_click(self):
        self._configure_widgets()
        with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
            mock_fn.return_value = _make_result()
            self._call_render()
        mock_fn.assert_called_once()

    def test_analysis_stored_in_session_state(self):
        import streamlit as st
        analysis = _make_result()
        self._configure_widgets()
        with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
            mock_fn.return_value = analysis
            self._call_render()
        assert st.session_state.get("coach_analysis") is analysis

    def test_st_error_not_called_on_success(self):
        import streamlit as st
        self._configure_widgets()
        with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
            mock_fn.return_value = _make_result()
            self._call_render()
        st.error.assert_not_called()

    def test_analyzer_called_with_stripped_source_text(self):
        padded = "  " + _SOURCE + "  "
        self._configure_widgets(source=padded)
        with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
            mock_fn.return_value = _make_result()
            self._call_render()
        call_args = mock_fn.call_args
        assert call_args[0][0] == _SOURCE  # positional first arg is stripped text

    def test_checker_warning_does_not_call_st_error(self):
        """A checker warning is part of a valid result; it must not trigger st.error."""
        import streamlit as st
        self._configure_widgets()
        with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
            mock_fn.return_value = _make_warning_result()
            self._call_render()
        st.error.assert_not_called()


# ===========================================================================
# TestRenderCoachModeAnalyzerArgs — level, density, flags forwarded correctly
# ===========================================================================

class TestRenderCoachModeAnalyzerArgs(_RenderCoachBase):
    """Verify that UI selection values are forwarded to analyze_spanish_source."""

    def _call_and_get_kwargs(self, **widget_kwargs) -> dict:
        self._configure_widgets(**widget_kwargs)
        with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
            mock_fn.return_value = _make_result()
            self._call_render()
        return mock_fn.call_args.kwargs

    def test_reader_level_b1_forwarded(self):
        kw = self._call_and_get_kwargs(reader_level="B1")
        assert kw["reader_level"] == "B1"

    def test_reader_level_c2_forwarded(self):
        kw = self._call_and_get_kwargs(reader_level="C2")
        assert kw["reader_level"] == "C2"

    def test_annotation_density_minimal_forwarded(self):
        kw = self._call_and_get_kwargs(density="minimal")
        assert kw["annotation_density"] == "minimal"

    def test_annotation_density_detailed_forwarded(self):
        kw = self._call_and_get_kwargs(density="detailed")
        assert kw["annotation_density"] == "detailed"

    def test_include_english_gloss_true_forwarded(self):
        kw = self._call_and_get_kwargs(include_english_gloss=True)
        assert kw["include_english_gloss"] is True

    def test_include_english_gloss_false_forwarded(self):
        kw = self._call_and_get_kwargs(include_english_gloss=False)
        assert kw["include_english_gloss"] is False

    def test_include_modern_spanish_true_forwarded(self):
        kw = self._call_and_get_kwargs(include_modern_spanish=True)
        assert kw["include_modern_spanish"] is True

    def test_include_modern_spanish_false_forwarded(self):
        kw = self._call_and_get_kwargs(include_modern_spanish=False)
        assert kw["include_modern_spanish"] is False

    def test_llm_client_passed_to_analyzer(self):
        """analyze_spanish_source must receive an llm_client keyword arg."""
        kw = self._call_and_get_kwargs()
        assert "llm_client" in kw
        assert callable(kw["llm_client"])

    def test_checker_config_passed_to_analyzer(self):
        kw = self._call_and_get_kwargs()
        assert "checker_config" in kw
        assert kw["checker_config"] is not None


# ===========================================================================
# TestRenderCoachModeCheckerConfig — checker config picks up settings
# ===========================================================================

class TestRenderCoachModeCheckerConfig(_RenderCoachBase):
    def _get_checker_config(self, monkeypatch, **settings_env):
        for k, v in settings_env.items():
            monkeypatch.setenv(k, str(v))
        self._configure_widgets()
        with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
            mock_fn.return_value = _make_result()
            self._call_render()
        return mock_fn.call_args.kwargs["checker_config"]

    def test_max_annotations_from_settings(self, monkeypatch):
        cfg = self._get_checker_config(monkeypatch, READING_COACH_MAX_ANNOTATIONS="4")
        assert cfg.max_annotations == 4

    def test_annotation_limit_policy_from_settings(self, monkeypatch):
        cfg = self._get_checker_config(monkeypatch, READING_COACH_ANNOTATION_LIMIT_POLICY="fail")
        assert cfg.annotation_limit_policy == "fail"

    def test_annotation_limit_policy_default_warn(self, monkeypatch):
        monkeypatch.delenv("READING_COACH_ANNOTATION_LIMIT_POLICY", raising=False)
        cfg = self._get_checker_config(monkeypatch)
        assert cfg.annotation_limit_policy == "warn"

    def test_include_english_gloss_mirrors_ui_checkbox(self):
        """checker_config.include_english_gloss must match the UI checkbox value."""
        self._configure_widgets(include_english_gloss=False)
        with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
            mock_fn.return_value = _make_result()
            self._call_render()
        cfg = mock_fn.call_args.kwargs["checker_config"]
        assert cfg.include_english_gloss is False


# ===========================================================================
# TestRenderCoachModeSettingsTimeout — client uses settings.timeout_seconds
# ===========================================================================

class TestRenderCoachModeSettingsTimeout(_RenderCoachBase):
    def test_client_uses_settings_timeout(self, monkeypatch):
        monkeypatch.setenv("READING_COACH_TIMEOUT_SECONDS", "77.0")
        self._configure_widgets()
        with mock.patch("ui.spanish_source_mode.make_ollama_client") as mock_factory:
            mock_factory.return_value = mock.MagicMock()
            with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
                mock_fn.return_value = _make_result()
                self._call_render()
        # Third positional arg to make_ollama_client must be the settings timeout
        assert mock_factory.call_args[0][2] == 77.0

    def test_client_uses_default_timeout_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("READING_COACH_TIMEOUT_SECONDS", raising=False)
        self._configure_widgets()
        with mock.patch("ui.spanish_source_mode.make_ollama_client") as mock_factory:
            mock_factory.return_value = mock.MagicMock()
            with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
                mock_fn.return_value = _make_result()
                self._call_render()
        timeout = mock_factory.call_args[0][2]
        assert timeout == 120.0

    def test_host_forwarded_to_client(self):
        self._configure_widgets()
        with mock.patch("ui.spanish_source_mode.make_ollama_client") as mock_factory:
            mock_factory.return_value = mock.MagicMock()
            with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
                mock_fn.return_value = _make_result()
                self._call_render()
        assert mock_factory.call_args[0][0] == self._HOST

    def test_model_forwarded_to_client(self):
        self._configure_widgets()
        with mock.patch("ui.spanish_source_mode.make_ollama_client") as mock_factory:
            mock_factory.return_value = mock.MagicMock()
            with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
                mock_fn.return_value = _make_result()
                self._call_render()
        assert mock_factory.call_args[0][1] == self._MODEL


# ===========================================================================
# TestRenderCoachModeTimeoutError
# ===========================================================================

class TestRenderCoachModeTimeoutError(_RenderCoachBase):
    """ReadingCoachTimeoutError → user-friendly message, session_state cleared."""

    def _call_with_timeout_error(self):
        import streamlit as st
        self._configure_widgets()
        timeout_exc = ReadingCoachTimeoutError("The model did not respond in time.")
        with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
            mock_fn.side_effect = timeout_exc
            self._call_render()

    def test_st_error_called_on_timeout(self):
        import streamlit as st
        self._call_with_timeout_error()
        st.error.assert_called_once()

    def test_timeout_message_is_user_friendly(self):
        import streamlit as st
        exc = ReadingCoachTimeoutError("The model did not respond in time.")
        self._configure_widgets()
        with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
            mock_fn.side_effect = exc
            self._call_render()
        error_msg = st.error.call_args[0][0]
        assert "requests.exceptions" not in error_msg
        assert "Traceback" not in error_msg

    def test_timeout_message_contains_user_message(self):
        import streamlit as st
        exc = ReadingCoachTimeoutError("The model did not respond in time.")
        self._configure_widgets()
        with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
            mock_fn.side_effect = exc
            self._call_render()
        error_msg = st.error.call_args[0][0]
        # user_message should contain something time/respond related
        assert any(
            w in error_msg.lower()
            for w in ("time", "respond", "timeout", "slow", "wait")
        )

    def test_session_state_cleared_on_timeout(self):
        import streamlit as st
        # Pre-load stale analysis result
        st.session_state.coach_analysis = _make_result()
        self._call_with_timeout_error()
        assert st.session_state.get("coach_analysis") is None

    def test_timeout_error_not_shown_as_generic_exception(self):
        """The specific timeout branch must fire, not the generic except Exception."""
        import streamlit as st
        exc = ReadingCoachTimeoutError()
        self._configure_widgets()
        with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
            mock_fn.side_effect = exc
            self._call_render()
        # The user_message doesn't contain "Analysis failed:" (the generic prefix)
        error_msg = st.error.call_args[0][0]
        assert "Analysis failed" not in error_msg


# ===========================================================================
# TestRenderCoachModeParseError
# ===========================================================================

class TestRenderCoachModeParseError(_RenderCoachBase):
    """CoachAnalysisError → specific parse-failure message, session_state cleared."""

    def _call_with_parse_error(self, msg="JSON was malformed"):
        self._configure_widgets()
        exc = CoachAnalysisError(msg)
        with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
            mock_fn.side_effect = exc
            self._call_render()

    def test_st_error_called_on_parse_failure(self):
        import streamlit as st
        self._call_with_parse_error()
        st.error.assert_called_once()

    def test_parse_error_message_mentions_parse_or_model(self):
        import streamlit as st
        self._call_with_parse_error()
        error_msg = st.error.call_args[0][0].lower()
        assert any(w in error_msg for w in ("parse", "model", "response"))

    def test_session_state_cleared_on_parse_error(self):
        import streamlit as st
        st.session_state.coach_analysis = _make_result()
        self._call_with_parse_error()
        assert st.session_state.get("coach_analysis") is None

    def test_parse_error_different_from_timeout_message(self):
        """CoachAnalysisError and ReadingCoachTimeoutError produce different UX messages."""
        import streamlit as st
        timeout_exc = ReadingCoachTimeoutError()
        parse_exc = CoachAnalysisError("bad JSON")

        self._configure_widgets()
        with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
            mock_fn.side_effect = timeout_exc
            self._call_render()
        timeout_msg = st.error.call_args[0][0]

        st.error.reset_mock()
        self.setup_method()
        self._configure_widgets()
        with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
            mock_fn.side_effect = parse_exc
            self._call_render()
        parse_msg = st.error.call_args[0][0]

        assert timeout_msg != parse_msg


# ===========================================================================
# TestRenderCoachModeGenericError
# ===========================================================================

class TestRenderCoachModeGenericError(_RenderCoachBase):
    """Arbitrary Exception → generic error message, session_state cleared."""

    def _call_with_generic_error(self, msg="connection refused"):
        self._configure_widgets()
        with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
            mock_fn.side_effect = RuntimeError(msg)
            self._call_render()

    def test_st_error_called_on_generic_exception(self):
        import streamlit as st
        self._call_with_generic_error()
        st.error.assert_called_once()

    def test_generic_error_message_actionable(self):
        """User must be told to check Ollama, not shown a raw stack trace."""
        import streamlit as st
        self._call_with_generic_error()
        error_msg = st.error.call_args[0][0].lower()
        assert "ollama" in error_msg or "model" in error_msg or "check" in error_msg

    def test_session_state_cleared_on_generic_error(self):
        import streamlit as st
        st.session_state.coach_analysis = _make_result()
        self._call_with_generic_error()
        assert st.session_state.get("coach_analysis") is None

    def test_no_traceback_in_generic_error_message(self):
        import streamlit as st
        self._call_with_generic_error()
        error_msg = st.error.call_args[0][0]
        assert "Traceback" not in error_msg
        assert "File " not in error_msg


# ===========================================================================
# TestRenderCoachModeNoAction — analyzer must NOT be called
# ===========================================================================

class TestRenderCoachModeNoAction(_RenderCoachBase):
    """Guard rails: analyzer must not fire when there's no text or no button click."""

    def test_button_not_clicked_analyzer_not_called(self):
        self._configure_widgets(button_clicked=False)
        with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
            self._call_render()
        mock_fn.assert_not_called()

    def test_empty_source_analyzer_not_called(self):
        self._configure_widgets(source="")
        with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
            self._call_render()
        mock_fn.assert_not_called()

    def test_whitespace_source_analyzer_not_called(self):
        self._configure_widgets(source="   \n  ")
        with mock.patch("ui.spanish_source_mode.analyze_spanish_source") as mock_fn:
            self._call_render()
        mock_fn.assert_not_called()

    def test_no_action_session_state_unchanged(self):
        import streamlit as st
        prior = _make_result()
        st.session_state.coach_analysis = prior
        self._configure_widgets(button_clicked=False)
        with mock.patch("ui.spanish_source_mode.analyze_spanish_source"):
            self._call_render()
        assert st.session_state.get("coach_analysis") is prior


# ===========================================================================
# TestRenderCoachModeSignature — function accepts required args without timeout
# ===========================================================================

class TestRenderCoachModeSignature:
    def test_render_coach_mode_accepts_host_and_model(self):
        """render_coach_mode(ollama_host=..., ollama_model=...) must work
        without a timeout argument; settings drive the timeout internally."""
        import inspect
        from ui.spanish_source_mode import render_coach_mode
        sig = inspect.signature(render_coach_mode)
        assert "ollama_host" in sig.parameters
        assert "ollama_model" in sig.parameters

    def test_render_coach_mode_has_no_timeout_param(self):
        """Timeout must come from CoachSettings, not from the call site."""
        import inspect
        from ui.spanish_source_mode import render_coach_mode
        sig = inspect.signature(render_coach_mode)
        assert "timeout" not in sig.parameters
