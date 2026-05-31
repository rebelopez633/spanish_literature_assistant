"""tests/test_analysis_metadata.py — Phase 3 Slice 12: observability metadata.

Captures lightweight, non-sensitive diagnostic information about each coach
analysis pass to help debug local-model quality and performance.

Test categories
---------------
1. TestSingleChunkMetadata  — metadata populated on AnalysisResult
2. TestMultiChunkMetadata   — metadata populated on MultiChunkAnalysisResult
3. TestDiagnosticsUI        — "Diagnostics" expander rendered in the UI
4. TestMetadataExport       — metadata included/omitted in Markdown export
5. TestNoExternalCalls      — metadata creation never touches the network

Design invariants (enforced by these tests)
-------------------------------------------
- AnalysisResult.metadata is populated by analyze_spanish_source().
- MultiChunkAnalysisResult.metadata is populated by analyze_spanish_source_chunks().
- build_metadata_from_single() and build_metadata_from_multi() are pure
  functions: no I/O, no network, no Streamlit.
- reading_coach_result_to_markdown() accepts include_metadata=True to append
  a full ## Diagnostics section; by default the section is omitted but a
  brief non-sensitive one-liner summary is included.
- render_coach_mode() renders a "Diagnostics" expander when metadata is present.
"""
from __future__ import annotations

import json
import unittest.mock as mock

import pytest

pytestmark = pytest.mark.unit

from reading_coach.analyzer import AnalysisResult, analyze_spanish_source
from reading_coach.checker import CoachCheckResult, CoachCheckerConfig, STATUS_PASSED
from reading_coach.metadata import (
    AnalysisMetadata,
    build_metadata_from_multi,
    build_metadata_from_single,
)
from reading_coach.multi_chunk_analyzer import analyze_spanish_source_chunks
from reading_coach.prompts import SPANISH_SOURCE_PROMPT_VERSION
from reading_coach.retry import RetryConfig
from reading_coach.schemas import ReadingCoachResult
from reading_coach.study_notes import reading_coach_result_to_markdown

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_SOURCE = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme."
_NO_RETRY = RetryConfig(max_retries=0, backoff_base=0.0, sleep_fn=lambda _: None)


def _payload(source: str = _SOURCE) -> dict:
    return {
        "original_spanish": source,
        "overall_level": "B2",
        "modern_spanish": None,
        "english_gloss": None,
        "difficult_phrases": [],
        "grammar_notes": [],
        "comprehension_question": None,
    }


def _fake_ok(messages: list[dict]) -> str:
    return json.dumps(_payload())


def _fake_ok_for_source(source: str):
    def _client(messages: list[dict]) -> str:  # noqa: ARG001
        return json.dumps(_payload(source))
    return _client


def _make_result() -> ReadingCoachResult:
    return ReadingCoachResult(
        original_spanish=_SOURCE,
        overall_level="B2",
    )


def _make_check() -> CoachCheckResult:
    return CoachCheckResult(status=STATUS_PASSED, summary="ok", issues=[])


def _make_analysis_result_with_metadata() -> AnalysisResult:
    """Return an AnalysisResult that carries pre-built AnalysisMetadata."""
    meta = build_metadata_from_single(
        prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
        reader_level="B1",
        annotation_density="balanced",
        checker_status=STATUS_PASSED,
    )
    return AnalysisResult(
        result=_make_result(),
        check=_make_check(),
        raw_response="{}",
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# 1. Single-chunk metadata
# ---------------------------------------------------------------------------

class TestSingleChunkMetadata:
    """analyze_spanish_source() populates AnalysisResult.metadata."""

    def test_metadata_is_not_none_after_single_chunk_analysis(self):
        ar = analyze_spanish_source(
            _SOURCE,
            reader_level="B1",
            annotation_density="balanced",
            llm_client=_fake_ok_for_source(_SOURCE),
            retry_config=_NO_RETRY,
        )
        assert ar.metadata is not None

    def test_metadata_is_analysis_metadata_type(self):
        ar = analyze_spanish_source(
            _SOURCE,
            reader_level="B2",
            annotation_density="detailed",
            llm_client=_fake_ok_for_source(_SOURCE),
            retry_config=_NO_RETRY,
        )
        assert isinstance(ar.metadata, AnalysisMetadata)

    def test_metadata_reader_level(self):
        ar = analyze_spanish_source(
            _SOURCE,
            reader_level="C1",
            annotation_density="balanced",
            llm_client=_fake_ok_for_source(_SOURCE),
            retry_config=_NO_RETRY,
        )
        assert ar.metadata.reader_level == "C1"

    def test_metadata_annotation_density(self):
        ar = analyze_spanish_source(
            _SOURCE,
            reader_level="B1",
            annotation_density="minimal",
            llm_client=_fake_ok_for_source(_SOURCE),
            retry_config=_NO_RETRY,
        )
        assert ar.metadata.annotation_density == "minimal"

    def test_metadata_prompt_version(self):
        ar = analyze_spanish_source(
            _SOURCE,
            reader_level="B1",
            annotation_density="balanced",
            llm_client=_fake_ok_for_source(_SOURCE),
            retry_config=_NO_RETRY,
        )
        assert ar.metadata.prompt_version == SPANISH_SOURCE_PROMPT_VERSION

    def test_metadata_chunk_counts_for_single(self):
        ar = analyze_spanish_source(
            _SOURCE,
            reader_level="B1",
            annotation_density="balanced",
            llm_client=_fake_ok_for_source(_SOURCE),
            retry_config=_NO_RETRY,
        )
        assert ar.metadata.chunk_count == 1
        assert ar.metadata.successful_chunk_count == 1
        assert ar.metadata.failed_chunk_count == 0

    def test_metadata_checker_status(self):
        ar = analyze_spanish_source(
            _SOURCE,
            reader_level="B1",
            annotation_density="balanced",
            llm_client=_fake_ok_for_source(_SOURCE),
            retry_config=_NO_RETRY,
        )
        assert ar.metadata.checker_status == ar.check.status

    def test_metadata_annotation_limit_policy_from_config(self):
        config = CoachCheckerConfig(annotation_limit_policy="truncate", max_annotations=5)
        ar = analyze_spanish_source(
            _SOURCE,
            reader_level="B1",
            annotation_density="balanced",
            llm_client=_fake_ok_for_source(_SOURCE),
            checker_config=config,
            retry_config=_NO_RETRY,
        )
        assert ar.metadata.annotation_limit_policy == "truncate"
        assert ar.metadata.max_annotations == 5

    def test_metadata_model_name_when_provided(self):
        ar = analyze_spanish_source(
            _SOURCE,
            reader_level="B1",
            annotation_density="balanced",
            llm_client=_fake_ok_for_source(_SOURCE),
            retry_config=_NO_RETRY,
            model_name="qwen2.5:7b",
        )
        assert ar.metadata.model_name == "qwen2.5:7b"

    def test_metadata_model_name_none_by_default(self):
        ar = analyze_spanish_source(
            _SOURCE,
            reader_level="B1",
            annotation_density="balanced",
            llm_client=_fake_ok_for_source(_SOURCE),
            retry_config=_NO_RETRY,
        )
        assert ar.metadata.model_name is None


# ---------------------------------------------------------------------------
# 2. Multi-chunk metadata
# ---------------------------------------------------------------------------

class TestMultiChunkMetadata:
    """analyze_spanish_source_chunks() populates MultiChunkAnalysisResult.metadata."""

    def test_metadata_is_not_none_after_multi_chunk_analysis(self):
        result = analyze_spanish_source_chunks(
            _SOURCE,
            llm_client=_fake_ok,
            retry_config=_NO_RETRY,
        )
        assert result.metadata is not None

    def test_metadata_is_analysis_metadata_type(self):
        result = analyze_spanish_source_chunks(
            _SOURCE,
            llm_client=_fake_ok,
            retry_config=_NO_RETRY,
        )
        assert isinstance(result.metadata, AnalysisMetadata)

    def test_metadata_chunk_counts_all_successful(self):
        result = analyze_spanish_source_chunks(
            _SOURCE,
            llm_client=_fake_ok,
            retry_config=_NO_RETRY,
        )
        meta = result.metadata
        assert meta.chunk_count == result.total_chunks
        assert meta.successful_chunk_count == result.successful_chunks
        assert meta.failed_chunk_count == result.failed_chunks

    def test_metadata_failed_chunk_count_when_partial_failure(self):
        """Metadata correctly records failed chunks when some chunks error."""
        calls = [0]

        def _sometimes_fail(messages):
            calls[0] += 1
            if calls[0] % 2 == 0:
                raise RuntimeError("fake LLM error")
            return json.dumps(_payload())

        # Use a short max_chars to force multiple chunks
        result = analyze_spanish_source_chunks(
            _SOURCE * 10,  # repeat to force multiple chunks
            max_chars_per_chunk=50,
            llm_client=_sometimes_fail,
            retry_config=_NO_RETRY,
        )
        assert result.metadata.failed_chunk_count == result.failed_chunks
        assert result.metadata.successful_chunk_count == result.successful_chunks

    def test_metadata_reader_level_multi(self):
        result = analyze_spanish_source_chunks(
            _SOURCE,
            reader_level="A2",
            annotation_density="minimal",
            llm_client=_fake_ok,
            retry_config=_NO_RETRY,
        )
        assert result.metadata.reader_level == "A2"
        assert result.metadata.annotation_density == "minimal"

    def test_metadata_model_name_multi(self):
        result = analyze_spanish_source_chunks(
            _SOURCE,
            llm_client=_fake_ok,
            retry_config=_NO_RETRY,
            model_name="gemma3:12b",
        )
        assert result.metadata.model_name == "gemma3:12b"


# ---------------------------------------------------------------------------
# 3. Diagnostics UI
# ---------------------------------------------------------------------------

class _UIBase:
    """Shared setup for UI tests."""

    _HOST = "http://localhost:11434"
    _MODEL = "qwen2.5:7b"

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

    def _call_render(self):
        from ui.spanish_source_mode import render_coach_mode
        render_coach_mode(ollama_host=self._HOST, ollama_model=self._MODEL)


class TestDiagnosticsUI(_UIBase):
    """Diagnostics expander is rendered when metadata is available."""

    def test_diagnostics_expander_rendered_when_metadata_present(self):
        import streamlit as st
        st.session_state.coach_analysis = _make_analysis_result_with_metadata()
        self._call_render()
        expander_labels = [str(c.args[0]) for c in st.expander.call_args_list if c.args]
        assert any("diagnostic" in lbl.lower() for lbl in expander_labels)

    def test_diagnostics_reader_level_written(self):
        import streamlit as st
        meta = build_metadata_from_single(
            prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
            reader_level="C2",
            annotation_density="detailed",
            checker_status=STATUS_PASSED,
        )
        analysis = AnalysisResult(
            result=_make_result(),
            check=_make_check(),
            raw_response="{}",
            metadata=meta,
        )
        st.session_state.coach_analysis = analysis
        self._call_render()
        write_calls = [str(c.args[0]) for c in st.write.call_args_list if c.args]
        assert any("C2" in w for w in write_calls)

    def test_diagnostics_expander_not_rendered_without_metadata(self):
        """When metadata is None (e.g. legacy AnalysisResult), no Diagnostics expander."""
        import streamlit as st
        st.session_state.coach_analysis = AnalysisResult(
            result=_make_result(),
            check=_make_check(),
            raw_response="{}",
            # no metadata
        )
        self._call_render()
        expander_labels = [str(c.args[0]) for c in st.expander.call_args_list if c.args]
        assert not any("diagnostic" in lbl.lower() for lbl in expander_labels)

    def test_diagnostics_checker_status_written(self):
        import streamlit as st
        meta = build_metadata_from_single(
            prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
            reader_level="B1",
            annotation_density="balanced",
            checker_status="warning",
        )
        analysis = AnalysisResult(
            result=_make_result(),
            check=_make_check(),
            raw_response="{}",
            metadata=meta,
        )
        st.session_state.coach_analysis = analysis
        self._call_render()
        write_calls = [str(c.args[0]) for c in st.write.call_args_list if c.args]
        assert any("warning" in w.lower() for w in write_calls)

    def test_diagnostics_prompt_version_written(self):
        import streamlit as st
        st.session_state.coach_analysis = _make_analysis_result_with_metadata()
        self._call_render()
        write_calls = [str(c.args[0]) for c in st.write.call_args_list if c.args]
        assert any(SPANISH_SOURCE_PROMPT_VERSION in w for w in write_calls)


# ---------------------------------------------------------------------------
# 4. Metadata in Markdown export
# ---------------------------------------------------------------------------

class TestMetadataExport:
    """reading_coach_result_to_markdown includes/omits diagnostics per flag."""

    def _make_meta(self) -> AnalysisMetadata:
        return build_metadata_from_single(
            prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
            reader_level="B2",
            annotation_density="balanced",
            checker_status=STATUS_PASSED,
            model_name="qwen2.5:7b",
        )

    def test_diagnostics_section_included_when_flag_true(self):
        md = reading_coach_result_to_markdown(
            _make_result(),
            metadata=self._make_meta(),
            include_metadata=True,
        )
        assert "Diagnostics" in md

    def test_diagnostics_section_omitted_when_flag_false(self):
        """Default: no ## Diagnostics block."""
        md = reading_coach_result_to_markdown(
            _make_result(),
            metadata=self._make_meta(),
            include_metadata=False,
        )
        assert "## Diagnostics" not in md

    def test_short_summary_present_by_default(self):
        """A brief non-sensitive one-liner is always included when metadata present."""
        md = reading_coach_result_to_markdown(
            _make_result(),
            metadata=self._make_meta(),
        )
        assert SPANISH_SOURCE_PROMPT_VERSION in md

    def test_diagnostics_contains_reader_level(self):
        md = reading_coach_result_to_markdown(
            _make_result(),
            metadata=self._make_meta(),
            include_metadata=True,
        )
        assert "B2" in md

    def test_diagnostics_contains_annotation_density(self):
        md = reading_coach_result_to_markdown(
            _make_result(),
            metadata=self._make_meta(),
            include_metadata=True,
        )
        assert "balanced" in md

    def test_diagnostics_contains_checker_status(self):
        md = reading_coach_result_to_markdown(
            _make_result(),
            metadata=self._make_meta(),
            include_metadata=True,
        )
        assert STATUS_PASSED in md

    def test_export_unchanged_when_metadata_is_none(self):
        """When no metadata is passed the output is identical to the old behaviour."""
        old = reading_coach_result_to_markdown(_make_result())
        new = reading_coach_result_to_markdown(_make_result(), metadata=None)
        assert old == new


# ---------------------------------------------------------------------------
# 5. No external calls
# ---------------------------------------------------------------------------

class TestNoExternalCalls:
    """Metadata creation must never trigger network I/O."""

    def test_build_metadata_from_single_makes_no_network_calls(self):
        with mock.patch("socket.getaddrinfo") as mock_dns, \
             mock.patch("socket.create_connection") as mock_conn:
            build_metadata_from_single(
                prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
                reader_level="B1",
                annotation_density="balanced",
                checker_status=STATUS_PASSED,
                model_name="qwen2.5:7b",
            )
        mock_dns.assert_not_called()
        mock_conn.assert_not_called()

    def test_build_metadata_from_multi_makes_no_network_calls(self):
        with mock.patch("socket.getaddrinfo") as mock_dns, \
             mock.patch("socket.create_connection") as mock_conn:
            build_metadata_from_multi(
                prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
                reader_level="B1",
                annotation_density="balanced",
                total_chunks=3,
                successful_chunks=2,
                failed_chunks=1,
                checker_summary="2/3 passed",
            )
        mock_dns.assert_not_called()
        mock_conn.assert_not_called()

    def test_analyze_single_with_fake_client_no_requests_calls(self):
        """Using a fake LLM client means no real HTTP requests are ever made."""
        with mock.patch("requests.post") as mock_post, \
             mock.patch("requests.get") as mock_get:
            analyze_spanish_source(
                _SOURCE,
                reader_level="B1",
                annotation_density="balanced",
                llm_client=_fake_ok_for_source(_SOURCE),
                retry_config=_NO_RETRY,
            )
        mock_post.assert_not_called()
        mock_get.assert_not_called()
