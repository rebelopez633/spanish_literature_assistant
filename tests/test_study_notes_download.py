"""tests/test_study_notes_download.py — Tests for Markdown download button in the
Spanish Source Reading Coach UI.

Phase 3 Slice 9.

Strategy
--------
All tests use the conftest Streamlit stub — no real Streamlit server.
`analyze_spanish_source` is NOT called in most tests; instead we pre-populate
`st.session_state.coach_analysis` (single-chunk) or
`st.session_state.coach_multi_analysis` (multi-chunk) and call `render_coach_mode`
with `button_clicked=False` to trigger the display-only code path.

`reading_coach_result_to_markdown` / `multi_chunk_result_to_markdown` are patched
in content tests so the assertions are independent of the study-notes rendering
implementation.

Acceptance criteria
-------------------
1. Mocked successful result → download_button rendered.
2. Download content includes original Spanish.
3. Download content includes difficult phrases.
4. Failed analysis (no session state) → no download button.
5. Filename is safe.
"""
from __future__ import annotations

import re
import unittest.mock as mock

import pytest

pytestmark = pytest.mark.ui

from reading_coach.analyzer import AnalysisResult
from reading_coach.checker import CoachCheckResult, STATUS_PASSED
from reading_coach.schemas import (
    ChunkAnalysisResult,
    DifficultPhrase,
    MultiChunkAnalysisResult,
    ReadingCoachResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SOURCE = "Había una vez un hidalgo de los de lanza en astillero."


def _make_result(source: str = _SOURCE) -> AnalysisResult:
    result = ReadingCoachResult(original_spanish=source, overall_level="B1")
    check = CoachCheckResult(status=STATUS_PASSED, summary="All checks passed.", issues=[])
    return AnalysisResult(result=result, check=check, raw_response="{}")


def _make_result_with_phrase(phrase_text: str, source: str = _SOURCE) -> AnalysisResult:
    dp = DifficultPhrase(
        phrase=phrase_text,
        category="archaic",
        difficulty_level="B2",
        why_difficult="Old usage.",
    )
    result = ReadingCoachResult(
        original_spanish=source,
        overall_level="B2",
        difficult_phrases=[dp],
    )
    check = CoachCheckResult(status=STATUS_PASSED, summary="ok", issues=[])
    return AnalysisResult(result=result, check=check, raw_response="{}")


def _make_fake_analysis_result(rcr: ReadingCoachResult):
    """Minimal AnalysisResult stand-in for multi-chunk helpers."""

    class _Fake:
        def __init__(self, r: ReadingCoachResult) -> None:
            self.result = r

    return _Fake(rcr)


def _make_multi(n_phrases: int = 1) -> MultiChunkAnalysisResult:
    dp = DifficultPhrase(
        phrase="lanza en astillero",
        category="archaic",
        difficulty_level="B2",
        why_difficult="Old term.",
    )
    rcr = ReadingCoachResult(
        original_spanish=_SOURCE,
        overall_level="B1",
        difficult_phrases=[dp] * n_phrases,
    )
    chunk = ChunkAnalysisResult(
        chunk_index=0,
        total_chunks=1,
        chunk_text=_SOURCE,
        chunk_id="chunk_0000",
        analysis=_make_fake_analysis_result(rcr),
        status="passed",
    )
    return MultiChunkAnalysisResult(
        original_spanish=_SOURCE,
        chunks=[chunk],
        prompt_version="spanish_source_v2",
        total_chunks=1,
        successful_chunks=1,
        failed_chunks=0,
    )


# ---------------------------------------------------------------------------
# Base mixin — mirrors the pattern in test_render_coach_mode.py
# ---------------------------------------------------------------------------

class _DownloadTestBase:
    _HOST = "http://localhost:11434"
    _MODEL = "qwen2.5:7b"

    def setup_method(self):
        import streamlit as st
        st.session_state._data.clear()
        for name in ("text_area", "button", "selectbox", "checkbox", "error",
                     "spinner", "caption", "metric", "subheader",
                     "download_button", "text_input"):
            m = getattr(st, name, None)
            if m is not None and hasattr(m, "reset_mock"):
                m.reset_mock(return_value=True, side_effect=True)
        # text_input must return a plain string so _safe_filename works.
        st.text_input.return_value = ""

    def _configure_widgets(
        self,
        source: str = _SOURCE,
        reader_level: str = "B1",
        density: str = "balanced",
        include_modern_spanish: bool = True,
        include_english_gloss: bool = True,
        button_clicked: bool = False,
        title: str = "",
    ) -> None:
        import streamlit as st
        st.text_area.return_value = source
        st.selectbox.side_effect = [reader_level, density]
        st.checkbox.side_effect = [include_modern_spanish, include_english_gloss]
        st.button.return_value = button_clicked
        st.text_input.return_value = title

    def _call_render(self) -> None:
        from ui.spanish_source_mode import render_coach_mode
        render_coach_mode(ollama_host=self._HOST, ollama_model=self._MODEL)

    def _get_download_call_kwargs(self) -> dict:
        import streamlit as st
        ca = st.download_button.call_args
        # Support both positional and keyword call styles.
        kw = dict(ca.kwargs) if ca.kwargs else {}
        if ca.args:
            for i, key in enumerate(("label", "data", "file_name", "mime")):
                if i < len(ca.args):
                    kw.setdefault(key, ca.args[i])
        return kw


# ===========================================================================
# 1 + 4: Download button rendered on success; absent on failure
# ===========================================================================

class TestDownloadButtonVisibility(_DownloadTestBase):
    """Acceptance criteria 1 & 4."""

    def test_download_button_called_when_single_chunk_result_present(self):
        import streamlit as st
        st.session_state.coach_analysis = _make_result()
        self._configure_widgets()
        self._call_render()
        st.download_button.assert_called()

    def test_download_button_called_when_multi_chunk_result_present(self):
        import streamlit as st
        st.session_state.coach_multi_analysis = _make_multi()
        self._configure_widgets()
        self._call_render()
        st.download_button.assert_called()

    def test_no_download_button_when_no_session_state(self):
        """Empty session state → no result → no download button."""
        import streamlit as st
        self._configure_widgets()
        self._call_render()
        st.download_button.assert_not_called()

    def test_no_download_button_after_analysis_error(self):
        """Failed analysis removes coach_analysis from session state."""
        import streamlit as st
        # Simulate a previous error having cleared coach_analysis
        # (session state has no coach_analysis key)
        self._configure_widgets()
        self._call_render()
        st.download_button.assert_not_called()

    def test_download_button_called_once_for_single_result(self):
        import streamlit as st
        st.session_state.coach_analysis = _make_result()
        self._configure_widgets()
        self._call_render()
        assert st.download_button.call_count == 1

    def test_download_button_called_once_for_multi_chunk(self):
        import streamlit as st
        st.session_state.coach_multi_analysis = _make_multi()
        self._configure_widgets()
        self._call_render()
        assert st.download_button.call_count == 1


# ===========================================================================
# 2: Download content includes original Spanish
# ===========================================================================

class TestDownloadContentOriginalSpanish(_DownloadTestBase):
    """Acceptance criterion 2."""

    def test_content_includes_original_spanish_single_chunk(self):
        import streamlit as st
        custom_source = "En un lugar de la Mancha."
        st.session_state.coach_analysis = _make_result(custom_source)
        self._configure_widgets(source=custom_source)
        with mock.patch("ui.spanish_source_mode.reading_coach_result_to_markdown") as mock_md:
            mock_md.return_value = f"# Notes\n\n{custom_source}\n"
            self._call_render()
        data = self._get_download_call_kwargs().get("data", "")
        assert custom_source in data

    def test_content_includes_original_spanish_multi_chunk(self):
        import streamlit as st
        st.session_state.coach_multi_analysis = _make_multi()
        self._configure_widgets()
        with mock.patch("ui.spanish_source_mode.multi_chunk_result_to_markdown") as mock_md:
            mock_md.return_value = f"# Notes\n\n{_SOURCE}\n"
            self._call_render()
        data = self._get_download_call_kwargs().get("data", "")
        assert _SOURCE in data

    def test_markdown_function_called_with_correct_result_single(self):
        import streamlit as st
        analysis = _make_result()
        st.session_state.coach_analysis = analysis
        self._configure_widgets()
        with mock.patch("ui.spanish_source_mode.reading_coach_result_to_markdown") as mock_md:
            mock_md.return_value = "content"
            self._call_render()
        mock_md.assert_called_once()
        call_result = mock_md.call_args[0][0] if mock_md.call_args.args else mock_md.call_args.kwargs.get("result")
        assert call_result is analysis.result

    def test_markdown_function_called_with_correct_result_multi(self):
        import streamlit as st
        multi = _make_multi()
        st.session_state.coach_multi_analysis = multi
        self._configure_widgets()
        with mock.patch("ui.spanish_source_mode.multi_chunk_result_to_markdown") as mock_md:
            mock_md.return_value = "content"
            self._call_render()
        mock_md.assert_called_once()
        call_result = mock_md.call_args[0][0] if mock_md.call_args.args else mock_md.call_args.kwargs.get("result")
        assert call_result is multi


# ===========================================================================
# 3: Download content includes difficult phrases
# ===========================================================================

class TestDownloadContentDifficultPhrases(_DownloadTestBase):
    """Acceptance criterion 3."""

    _PHRASE = "lanza en astillero"

    def test_difficult_phrase_appears_in_download_data(self):
        import streamlit as st
        analysis = _make_result_with_phrase(self._PHRASE)
        st.session_state.coach_analysis = analysis
        self._configure_widgets()
        self._call_render()
        data = self._get_download_call_kwargs().get("data", "")
        assert self._PHRASE in data

    def test_difficult_phrase_in_multi_chunk_download(self):
        import streamlit as st
        multi = _make_multi(n_phrases=1)
        st.session_state.coach_multi_analysis = multi
        self._configure_widgets()
        self._call_render()
        data = self._get_download_call_kwargs().get("data", "")
        assert "lanza en astillero" in data

    def test_result_without_phrases_does_not_crash(self):
        import streamlit as st
        analysis = _make_result()  # no difficult_phrases
        st.session_state.coach_analysis = analysis
        self._configure_widgets()
        self._call_render()  # must not raise
        import streamlit as st
        st.download_button.assert_called()


# ===========================================================================
# 5: Filename is safe
# ===========================================================================

class TestSafeFilename:
    """Unit tests for _safe_filename — no Streamlit required."""

    @staticmethod
    def _fn(title):
        from ui.spanish_source_mode import _safe_filename
        return _safe_filename(title)

    def test_none_returns_default(self):
        assert self._fn(None) == "spanish-reading-notes.md"

    def test_empty_string_returns_default(self):
        assert self._fn("") == "spanish-reading-notes.md"

    def test_whitespace_only_returns_default(self):
        assert self._fn("   ") == "spanish-reading-notes.md"

    def test_simple_title_lowercased_and_hyphenated(self):
        assert self._fn("Don Quijote") == "don-quijote.md"

    def test_special_chars_removed(self):
        result = self._fn("Chapter 1: The Beginning!")
        assert ":" not in result
        assert "!" not in result
        assert result.endswith(".md")

    def test_result_always_ends_in_md(self):
        for title in ("Notes", "My Study Session", "  test  "):
            assert self._fn(title).endswith(".md"), f"Failed for {title!r}"

    def test_long_title_truncated(self):
        long = "A" * 80
        result = self._fn(long)
        # filename (without .md) should not exceed 60 chars
        stem = result[:-3]
        assert len(stem) <= 60

    def test_no_double_hyphens(self):
        result = self._fn("Hello  World  Test")
        assert "--" not in result

    def test_no_leading_or_trailing_hyphens_before_extension(self):
        stem = self._fn("  !!! hello !!!  ")[:-3]
        assert not stem.startswith("-")
        assert not stem.endswith("-")

    def test_accented_chars_allowed(self):
        # Unicode word chars should not be stripped
        result = self._fn("Español")
        assert result.endswith(".md")
        assert "español" in result or "espa" in result

    def test_numbers_preserved(self):
        assert "chapter-1" in self._fn("Chapter 1")


# ===========================================================================
# MIME type and label sanity
# ===========================================================================

class TestDownloadButtonMetadata(_DownloadTestBase):
    """Verify MIME type and label are set correctly."""

    def test_mime_type_is_text_markdown_single_chunk(self):
        import streamlit as st
        st.session_state.coach_analysis = _make_result()
        self._configure_widgets()
        self._call_render()
        mime = self._get_download_call_kwargs().get("mime", "")
        assert mime == "text/markdown"

    def test_mime_type_is_text_markdown_multi_chunk(self):
        import streamlit as st
        st.session_state.coach_multi_analysis = _make_multi()
        self._configure_widgets()
        self._call_render()
        mime = self._get_download_call_kwargs().get("mime", "")
        assert mime == "text/markdown"

    def test_default_filename_when_no_title_single(self):
        import streamlit as st
        st.session_state.coach_analysis = _make_result()
        self._configure_widgets(title="")
        self._call_render()
        file_name = self._get_download_call_kwargs().get("file_name", "")
        assert file_name == "spanish-reading-notes.md"

    def test_default_filename_when_no_title_multi(self):
        import streamlit as st
        st.session_state.coach_multi_analysis = _make_multi()
        self._configure_widgets(title="")
        self._call_render()
        file_name = self._get_download_call_kwargs().get("file_name", "")
        assert file_name == "spanish-reading-notes.md"

    def test_title_derived_filename_single(self):
        import streamlit as st
        st.session_state.coach_analysis = _make_result()
        self._configure_widgets(title="Don Quijote Chapter 1")
        self._call_render()
        file_name = self._get_download_call_kwargs().get("file_name", "")
        assert file_name == "don-quijote-chapter-1.md"

    def test_title_derived_filename_multi(self):
        import streamlit as st
        st.session_state.coach_multi_analysis = _make_multi()
        self._configure_widgets(title="La Celestina Acto I")
        self._call_render()
        file_name = self._get_download_call_kwargs().get("file_name", "")
        assert file_name == "la-celestina-acto-i.md"

    def test_label_is_a_nonempty_string(self):
        import streamlit as st
        st.session_state.coach_analysis = _make_result()
        self._configure_widgets()
        self._call_render()
        label = self._get_download_call_kwargs().get("label", "")
        assert isinstance(label, str) and len(label) > 0
