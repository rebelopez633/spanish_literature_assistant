"""
Tests for reading_coach/multi_chunk_analyzer.py — multi-chunk orchestration.

All tests use fake LLM clients.  No Ollama, no network, no external services.
"""
from __future__ import annotations

import json

import pytest

from reading_coach.multi_chunk_analyzer import analyze_spanish_source_chunks
from reading_coach.prompts import SPANISH_SOURCE_PROMPT_VERSION
from reading_coach.retry import RetryConfig
from reading_coach.schemas import (
    ChunkAnalysisResult,
    MultiChunkAnalysisResult,
    ReadingCoachResult,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SHORT_SOURCE = "En un lugar de la Mancha."

_NO_RETRY = RetryConfig(max_retries=0, backoff_base=0.0, sleep_fn=lambda _: None)


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


class CapturingClient:
    """Records every call and returns a canned response."""

    def __init__(self, response_fn=None):
        self.calls: list[list[dict]] = []
        self._fn = response_fn or (lambda _: json.dumps(_payload()))

    def __call__(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self._fn(messages)


# ===========================================================================
# TestReturnType
# ===========================================================================

class TestReturnType:
    def test_returns_multi_chunk_analysis_result(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        assert isinstance(result, MultiChunkAnalysisResult)

    def test_has_original_spanish(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        assert result.original_spanish == SHORT_SOURCE

    def test_original_spanish_verbatim(self):
        """original_spanish must equal the source text byte-for-byte."""
        source = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme."
        result = analyze_spanish_source_chunks(
            source,
            llm_client=lambda _: json.dumps(_payload(source)),
            retry_config=_NO_RETRY,
        )
        assert result.original_spanish == source

    def test_chunks_is_list(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        assert isinstance(result.chunks, list)

    def test_at_least_one_chunk_for_non_empty_source(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        assert len(result.chunks) >= 1

    def test_total_chunks_equals_len_chunks(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        assert result.total_chunks == len(result.chunks)

    def test_prompt_version_populated(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        assert result.prompt_version == SPANISH_SOURCE_PROMPT_VERSION

    def test_successful_chunks_matches_succeeded_count(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        expected = sum(1 for c in result.chunks if c.succeeded)
        assert result.successful_chunks == expected

    def test_failed_chunks_matches_failed_count(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        expected = sum(1 for c in result.chunks if not c.succeeded)
        assert result.failed_chunks == expected

    def test_chunks_are_chunk_analysis_result_instances(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        for chunk in result.chunks:
            assert isinstance(chunk, ChunkAnalysisResult)

    def test_successful_plus_failed_equals_total(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        assert result.successful_chunks + result.failed_chunks == result.total_chunks


# ===========================================================================
# TestFailureHandling
# ===========================================================================

class TestFailureHandling:
    def test_failed_chunk_recorded_on_llm_error(self):
        def always_fail(messages):  # noqa: ARG001
            raise RuntimeError("LLM unavailable")

        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=always_fail, retry_config=_NO_RETRY
        )
        assert result.failed_chunks > 0

    def test_failed_chunk_analysis_is_none(self):
        def always_fail(messages):  # noqa: ARG001
            raise RuntimeError("LLM unavailable")

        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=always_fail, retry_config=_NO_RETRY
        )
        for chunk in result.chunks:
            assert chunk.analysis is None

    def test_failed_chunk_error_message_populated(self):
        def always_fail(messages):  # noqa: ARG001
            raise RuntimeError("LLM unavailable")

        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=always_fail, retry_config=_NO_RETRY
        )
        for chunk in result.chunks:
            assert chunk.error, "Error message should be non-empty"

    def test_total_failure_gives_zero_successful(self):
        def always_fail(messages):  # noqa: ARG001
            raise RuntimeError("LLM unavailable")

        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=always_fail, retry_config=_NO_RETRY
        )
        assert result.successful_chunks == 0

    def test_partial_failure_mixed_results(self):
        """Second chunk fails; first and third succeed."""
        call_count = [0]

        def fail_on_second(messages):  # noqa: ARG001
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("second chunk failed")
            return json.dumps(_payload())

        # Long enough to produce ≥3 chunks at max_chars=50
        text = "Esta es la oración uno. " * 4 + "Esta es la oración dos. " * 4
        result = analyze_spanish_source_chunks(
            text,
            llm_client=fail_on_second,
            max_chars_per_chunk=60,
            retry_config=_NO_RETRY,
        )
        assert result.successful_chunks >= 1
        assert result.failed_chunks >= 1

    def test_failed_chunk_succeeded_property_false(self):
        def always_fail(messages):  # noqa: ARG001
            raise RuntimeError("LLM unavailable")

        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=always_fail, retry_config=_NO_RETRY
        )
        assert all(not c.succeeded for c in result.chunks)


# ===========================================================================
# TestMultiChunkBehaviour
# ===========================================================================

class TestMultiChunkBehaviour:
    def test_multi_chunk_text_calls_llm_per_chunk(self):
        client = CapturingClient()
        text = "Frase corta. " * 8  # ~104 chars at max_chars=40 → multiple chunks
        result = analyze_spanish_source_chunks(
            text,
            llm_client=client,
            max_chars_per_chunk=40,
            retry_config=_NO_RETRY,
        )
        assert len(result.chunks) >= 2
        assert len(client.calls) == len(result.chunks)

    def test_chunk_indices_are_sequential(self):
        text = "palabra " * 40  # forces multiple chunks at small max_chars
        result = analyze_spanish_source_chunks(
            text,
            llm_client=_fake_ok,
            max_chars_per_chunk=80,
            retry_config=_NO_RETRY,
        )
        for i, chunk in enumerate(result.chunks):
            assert chunk.chunk_index == i

    def test_chunk_total_chunks_consistent_with_result(self):
        text = "palabra " * 40
        result = analyze_spanish_source_chunks(
            text,
            llm_client=_fake_ok,
            max_chars_per_chunk=80,
            retry_config=_NO_RETRY,
        )
        for chunk in result.chunks:
            assert chunk.total_chunks == result.total_chunks

    def test_single_chunk_text_has_total_one(self):
        result = analyze_spanish_source_chunks(
            SHORT_SOURCE, llm_client=_fake_ok, retry_config=_NO_RETRY
        )
        assert result.total_chunks == 1
        assert result.chunks[0].total_chunks == 1

    def test_chunk_text_present_in_original(self):
        """Each chunk's text must be a substring (or whitespace-stripped sub) of source."""
        source = "Primera frase. Segunda frase. Tercera frase. Cuarta frase."
        result = analyze_spanish_source_chunks(
            source,
            llm_client=_fake_ok,
            max_chars_per_chunk=30,
            retry_config=_NO_RETRY,
        )
        for chunk in result.chunks:
            assert chunk.chunk_text.strip() in source or any(
                word in source for word in chunk.chunk_text.split()
            ), f"Chunk text {chunk.chunk_text!r} has no words in source"
