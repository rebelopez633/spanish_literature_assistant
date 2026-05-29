"""
Phase 3 Slice 4 — tests for multi-chunk orchestration enhancements.

Tests the 7 acceptance criteria from the slice spec:
  1. Short input delegates to one chunk.
  2. Multi-chunk input returns ordered chunk results.
  3. combined source equals original source.
  4. LLM failure captured when continue_on_error=True.
  5. LLM failure raises/stops when continue_on_error=False.
  6. prompt_version propagated.
  7. checker_results preserved per chunk.

Plus new-field coverage: chunk_id, status, index, source_text, error_message,
combined_difficult_phrases, checker_summary.

All tests use fake LLM clients. No Ollama, no network.
"""
from __future__ import annotations

import json

import pytest

from reading_coach.multi_chunk_analyzer import analyze_spanish_source_chunks
from reading_coach.prompts import SPANISH_SOURCE_PROMPT_VERSION
from reading_coach.retry import RetryConfig
from reading_coach.schemas import ChunkAnalysisResult, MultiChunkAnalysisResult

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers shared across all test classes
# ---------------------------------------------------------------------------

SHORT_SOURCE = "En un lugar de la Mancha."

_NO_RETRY = RetryConfig(max_retries=0, backoff_base=0.0, sleep_fn=lambda _: None)

# A multi-chunk source: two sentences long enough to force a split at small max_chars.
LONG_SOURCE = (
    "Esta es la primera oración del pasaje largo. "
    "Y esta es la segunda oración del pasaje largo."
)


def _payload(source: str = SHORT_SOURCE) -> dict:
    return {
        "original_spanish": source,
        "overall_level": "B1",
        "modern_spanish": None,
        "english_gloss": None,
        "difficult_phrases": [],
        "grammar_notes": [],
        "comprehension_question": None,
    }


def _fake_ok(messages: list[dict]) -> str:  # noqa: ARG001
    return json.dumps(_payload())


def _make_failing_client(exc: Exception | None = None):
    """Return a callable that always raises *exc* (default RuntimeError)."""
    error = exc or RuntimeError("LLM unavailable")

    def _fail(messages: list[dict]) -> str:  # noqa: ARG001
        raise error

    return _fail


# ===========================================================================
# Criterion 1 — Short input delegates to one chunk
# ===========================================================================

class TestShortInputOneChunk:
    def test_one_chunk_for_short_source(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        assert result.total_chunks == 1

    def test_single_chunk_has_chunk_id_0000(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        assert result.chunks[0].chunk_id == "chunk_0000"

    def test_single_chunk_index_is_zero(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        assert result.chunks[0].index == 0

    def test_single_chunk_source_text_is_short_source(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        assert result.chunks[0].source_text.strip() == SHORT_SOURCE.strip()


# ===========================================================================
# Criterion 2 — Multi-chunk returns ordered results
# ===========================================================================

class TestMultiChunkOrdering:
    def test_multi_chunk_produces_multiple_results(self):
        result = analyze_spanish_source_chunks(
            LONG_SOURCE,
            llm_client=_fake_ok,
            max_chars_per_chunk=50,
            retry_config=_NO_RETRY,
        )
        assert len(result.chunks) >= 2

    def test_chunk_ids_are_sequential(self):
        result = analyze_spanish_source_chunks(
            LONG_SOURCE,
            llm_client=_fake_ok,
            max_chars_per_chunk=50,
            retry_config=_NO_RETRY,
        )
        for i, chunk in enumerate(result.chunks):
            assert chunk.chunk_id == f"chunk_{i:04d}"

    def test_chunk_indices_are_sequential(self):
        result = analyze_spanish_source_chunks(
            LONG_SOURCE,
            llm_client=_fake_ok,
            max_chars_per_chunk=50,
            retry_config=_NO_RETRY,
        )
        for i, chunk in enumerate(result.chunks):
            assert chunk.index == i

    def test_chunk_id_equals_chunk_index_format(self):
        """chunk_id and index must always agree."""
        result = analyze_spanish_source_chunks(
            LONG_SOURCE,
            llm_client=_fake_ok,
            max_chars_per_chunk=50,
            retry_config=_NO_RETRY,
        )
        for chunk in result.chunks:
            assert chunk.chunk_id == f"chunk_{chunk.index:04d}"

    def test_chunk_ids_are_unique(self):
        result = analyze_spanish_source_chunks(
            LONG_SOURCE,
            llm_client=_fake_ok,
            max_chars_per_chunk=50,
            retry_config=_NO_RETRY,
        )
        ids = [c.chunk_id for c in result.chunks]
        assert len(ids) == len(set(ids))


# ===========================================================================
# Criterion 3 — Combined source equals original source
# ===========================================================================

class TestOriginalSourcePreserved:
    def test_original_spanish_equals_source(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        assert result.original_spanish == SHORT_SOURCE

    def test_original_spanish_preserved_verbatim_for_long_source(self):
        result = analyze_spanish_source_chunks(
            LONG_SOURCE,
            llm_client=_fake_ok,
            max_chars_per_chunk=50,
            retry_config=_NO_RETRY,
        )
        assert result.original_spanish == LONG_SOURCE

    def test_combined_difficult_phrases_is_list(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        assert isinstance(result.combined_difficult_phrases, list)

    def test_combined_difficult_phrases_same_as_all_difficult_phrases(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        assert result.combined_difficult_phrases == result.all_difficult_phrases


# ===========================================================================
# Criterion 4 — LLM failure captured when continue_on_error=True
# ===========================================================================

class TestContinueOnErrorTrue:
    def test_default_continue_on_error_captures_failure(self):
        """Default is continue_on_error=True — errors should not propagate."""
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE,
            llm_client=_make_failing_client(),
            retry_config=_NO_RETRY,
        )
        assert isinstance(result, MultiChunkAnalysisResult)

    def test_explicit_continue_on_error_true_captures_failure(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE,
            llm_client=_make_failing_client(),
            retry_config=_NO_RETRY,
            continue_on_error=True,
        )
        assert result.failed_chunks > 0

    def test_failed_chunk_error_message_populated(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE,
            llm_client=_make_failing_client(),
            retry_config=_NO_RETRY,
            continue_on_error=True,
        )
        for chunk in result.chunks:
            assert chunk.error_message is not None

    def test_failed_chunk_status_is_error(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE,
            llm_client=_make_failing_client(),
            retry_config=_NO_RETRY,
            continue_on_error=True,
        )
        for chunk in result.chunks:
            assert chunk.status == "error"

    def test_partial_failure_successful_chunks_unaffected(self):
        """Third chunk fails; chunks 1 and 2 must still succeed."""
        call_count = [0]

        def fail_on_third(messages: list[dict]) -> str:  # noqa: ARG001
            call_count[0] += 1
            if call_count[0] == 3:
                raise RuntimeError("third chunk failed")
            return json.dumps(_payload())

        text = "Frase. " * 12
        result = analyze_spanish_source_chunks(
            text,
            llm_client=fail_on_third,
            max_chars_per_chunk=30,
            retry_config=_NO_RETRY,
            continue_on_error=True,
        )
        assert result.successful_chunks >= 2
        assert result.failed_chunks >= 1

    def test_all_chunks_present_even_with_failures(self):
        """total_chunks must equal len(chunks) even when some fail."""
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE,
            llm_client=_make_failing_client(),
            retry_config=_NO_RETRY,
            continue_on_error=True,
        )
        assert len(result.chunks) == result.total_chunks


# ===========================================================================
# Criterion 5 — LLM failure raises when continue_on_error=False
# ===========================================================================

class TestContinueOnErrorFalse:
    def test_raises_on_llm_failure(self):
        with pytest.raises(Exception):
            analyze_spanish_source_chunks(
                SHORT_SOURCE,
                llm_client=_make_failing_client(),
                retry_config=_NO_RETRY,
                continue_on_error=False,
            )

    def test_raises_runtime_error_type(self):
        with pytest.raises(RuntimeError, match="LLM unavailable"):
            analyze_spanish_source_chunks(
                SHORT_SOURCE,
                llm_client=_make_failing_client(RuntimeError("LLM unavailable")),
                retry_config=_NO_RETRY,
                continue_on_error=False,
            )

    def test_stops_at_first_failed_chunk(self):
        """When continue_on_error=False, chunks after the failure are not processed."""
        call_count = [0]

        def fail_on_first(messages: list[dict]) -> str:  # noqa: ARG001
            call_count[0] += 1
            raise RuntimeError("first chunk failed")

        text = "Frase. " * 12  # multiple chunks

        with pytest.raises(RuntimeError):
            analyze_spanish_source_chunks(
                text,
                llm_client=fail_on_first,
                max_chars_per_chunk=30,
                retry_config=_NO_RETRY,
                continue_on_error=False,
            )
        # Should have stopped after the first call
        assert call_count[0] == 1

    def test_succeeds_when_no_errors(self):
        """continue_on_error=False must still work normally when LLM always succeeds."""
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE,
            llm_client=_fake_ok,
            retry_config=_NO_RETRY,
            continue_on_error=False,
        )
        assert isinstance(result, MultiChunkAnalysisResult)
        assert result.failed_chunks == 0


# ===========================================================================
# Criterion 6 — prompt_version propagated
# ===========================================================================

class TestPromptVersionPropagated:
    def test_prompt_version_equals_constant(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        assert result.prompt_version == SPANISH_SOURCE_PROMPT_VERSION

    def test_prompt_version_is_string(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        assert isinstance(result.prompt_version, str)
        assert result.prompt_version  # non-empty


# ===========================================================================
# Criterion 7 — checker_results preserved per chunk
# ===========================================================================

class TestCheckerResultsPerChunk:
    def test_successful_chunk_has_checker_result(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        for chunk in result.chunks:
            if chunk.succeeded:
                assert chunk.checker_result is not None

    def test_checker_result_has_status_attribute(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        for chunk in result.chunks:
            if chunk.succeeded:
                assert hasattr(chunk.checker_result, "status")

    def test_checker_result_status_is_valid(self):
        valid_statuses = {"passed", "warning", "failed"}
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        for chunk in result.chunks:
            if chunk.succeeded:
                assert chunk.checker_result.status in valid_statuses

    def test_failed_chunk_has_no_checker_result(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE,
            llm_client=_make_failing_client(),
            retry_config=_NO_RETRY,
            continue_on_error=True,
        )
        for chunk in result.chunks:
            if not chunk.succeeded:
                assert chunk.checker_result is None

    def test_chunk_status_matches_checker_status_on_success(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        for chunk in result.chunks:
            if chunk.succeeded:
                assert chunk.status == chunk.checker_result.status


# ===========================================================================
# New-field coverage (chunk_id, status, index, source_text, error_message)
# ===========================================================================

class TestNewChunkFields:
    def test_chunk_id_format(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        assert result.chunks[0].chunk_id == "chunk_0000"

    def test_index_property_equals_chunk_index(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        for chunk in result.chunks:
            assert chunk.index == chunk.chunk_index

    def test_source_text_property_equals_chunk_text(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        for chunk in result.chunks:
            assert chunk.source_text == chunk.chunk_text

    def test_error_message_none_on_success(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        for chunk in result.chunks:
            if chunk.succeeded:
                assert chunk.error_message is None

    def test_error_message_populated_on_failure(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE,
            llm_client=_make_failing_client(),
            retry_config=_NO_RETRY,
            continue_on_error=True,
        )
        for chunk in result.chunks:
            if not chunk.succeeded:
                assert chunk.error_message is not None
                assert "LLM unavailable" in chunk.error_message

    def test_error_message_equals_error_field(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE,
            llm_client=_make_failing_client(),
            retry_config=_NO_RETRY,
            continue_on_error=True,
        )
        for chunk in result.chunks:
            assert chunk.error_message == chunk.error

    def test_status_passed_on_clean_result(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        for chunk in result.chunks:
            assert chunk.status in {"passed", "warning", "failed"}

    def test_status_error_on_failed_chunk(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE,
            llm_client=_make_failing_client(),
            retry_config=_NO_RETRY,
            continue_on_error=True,
        )
        for chunk in result.chunks:
            assert chunk.status == "error"


# ===========================================================================
# MultiChunkAnalysisResult new fields: checker_summary, combined_difficult_phrases
# ===========================================================================

class TestMultiChunkNewFields:
    def test_checker_summary_is_string(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        assert isinstance(result.checker_summary, str)

    def test_checker_summary_non_empty(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        assert result.checker_summary

    def test_checker_summary_reflects_all_passed(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        # All chunks passed → summary should mention success
        assert "passed" in result.checker_summary or "1/1" in result.checker_summary

    def test_checker_summary_reflects_failure(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE,
            llm_client=_make_failing_client(),
            retry_config=_NO_RETRY,
            continue_on_error=True,
        )
        # At least one failure → summary should reflect it
        assert "error" in result.checker_summary or "0/" in result.checker_summary

    def test_combined_difficult_phrases_is_list(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        assert isinstance(result.combined_difficult_phrases, list)

    def test_combined_equals_all_difficult_phrases(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        assert result.combined_difficult_phrases == result.all_difficult_phrases
