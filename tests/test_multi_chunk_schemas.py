"""
Tests for ChunkAnalysisResult and MultiChunkAnalysisResult Pydantic models.

Pure unit tests — no Ollama, no network, no external services.
"""
from __future__ import annotations

import pytest

from reading_coach.analyzer import AnalysisResult
from reading_coach.checker import CoachCheckResult, STATUS_PASSED
from reading_coach.prompts import SPANISH_SOURCE_PROMPT_VERSION
from reading_coach.schemas import (
    ChunkAnalysisResult,
    DifficultPhrase,
    MultiChunkAnalysisResult,
    ReadingCoachResult,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

CHUNK_TEXT = "En un lugar de la Mancha."


def _make_analysis_result(text: str = CHUNK_TEXT) -> AnalysisResult:
    result = ReadingCoachResult(original_spanish=text, overall_level="B1")
    check = CoachCheckResult(status=STATUS_PASSED, summary="ok", issues=[])
    return AnalysisResult(result=result, check=check, raw_response="{}")


def _make_chunk(
    chunk_index: int = 0,
    total_chunks: int = 1,
    succeeded: bool = True,
    chunk_text: str = CHUNK_TEXT,
) -> ChunkAnalysisResult:
    return ChunkAnalysisResult(
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        chunk_text=chunk_text,
        analysis=_make_analysis_result(chunk_text) if succeeded else None,
        error=None if succeeded else "parse error",
    )


def _make_multi(
    n_chunks: int = 2,
    n_failed: int = 0,
    original_spanish: str = "Full text.",
) -> MultiChunkAnalysisResult:
    chunks = [
        _make_chunk(i, n_chunks, succeeded=(i >= n_failed))
        for i in range(n_chunks)
    ]
    return MultiChunkAnalysisResult(
        original_spanish=original_spanish,
        chunks=chunks,
        prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
        total_chunks=n_chunks,
        successful_chunks=n_chunks - n_failed,
        failed_chunks=n_failed,
    )


# ===========================================================================
# TestChunkAnalysisResult
# ===========================================================================

class TestChunkAnalysisResult:
    def test_has_chunk_index(self):
        c = _make_chunk(chunk_index=2)
        assert c.chunk_index == 2

    def test_has_total_chunks(self):
        c = _make_chunk(total_chunks=5)
        assert c.total_chunks == 5

    def test_has_chunk_text(self):
        c = _make_chunk(chunk_text="Hola mundo.")
        assert c.chunk_text == "Hola mundo."

    def test_succeeded_true_when_analysis_present(self):
        c = _make_chunk(succeeded=True)
        assert c.succeeded is True

    def test_succeeded_false_when_analysis_none(self):
        c = _make_chunk(succeeded=False)
        assert c.succeeded is False

    def test_error_none_on_success(self):
        c = _make_chunk(succeeded=True)
        assert c.error is None

    def test_error_populated_on_failure(self):
        c = _make_chunk(succeeded=False)
        assert c.error is not None

    def test_analysis_none_on_failure(self):
        c = _make_chunk(succeeded=False)
        assert c.analysis is None

    def test_analysis_present_on_success(self):
        c = _make_chunk(succeeded=True)
        assert c.analysis is not None

    def test_accepts_analysis_result_dataclass(self):
        ar = _make_analysis_result()
        c = ChunkAnalysisResult(
            chunk_index=0, total_chunks=1, chunk_text=CHUNK_TEXT, analysis=ar
        )
        assert c.analysis is ar

    def test_chunk_index_zero_based(self):
        c = _make_chunk(chunk_index=0, total_chunks=3)
        assert c.chunk_index == 0

    def test_total_chunks_reflects_full_count(self):
        c = _make_chunk(chunk_index=2, total_chunks=3)
        assert c.total_chunks == 3


# ===========================================================================
# TestMultiChunkAnalysisResult
# ===========================================================================

class TestMultiChunkAnalysisResult:
    def test_has_original_spanish(self):
        m = _make_multi(original_spanish="Texto completo.")
        assert m.original_spanish == "Texto completo."

    def test_original_spanish_preserved_exactly(self):
        text = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme."
        m = _make_multi(original_spanish=text)
        assert m.original_spanish == text

    def test_has_chunks_list(self):
        m = _make_multi(n_chunks=3)
        assert isinstance(m.chunks, list)
        assert len(m.chunks) == 3

    def test_has_prompt_version(self):
        m = _make_multi()
        assert m.prompt_version == SPANISH_SOURCE_PROMPT_VERSION

    def test_total_chunks_count(self):
        m = _make_multi(n_chunks=4)
        assert m.total_chunks == 4

    def test_successful_chunks_count(self):
        m = _make_multi(n_chunks=4, n_failed=1)
        assert m.successful_chunks == 3

    def test_failed_chunks_count(self):
        m = _make_multi(n_chunks=4, n_failed=2)
        assert m.failed_chunks == 2

    def test_all_successful_zero_failed(self):
        m = _make_multi(n_chunks=3, n_failed=0)
        assert m.failed_chunks == 0
        assert m.successful_chunks == 3

    def test_all_failed_zero_successful(self):
        m = _make_multi(n_chunks=2, n_failed=2)
        assert m.successful_chunks == 0
        assert m.failed_chunks == 2

    def test_all_difficult_phrases_empty_when_none(self):
        m = _make_multi(n_chunks=2, n_failed=0)
        # _make_analysis_result uses no difficult_phrases
        assert m.all_difficult_phrases == []

    def test_all_difficult_phrases_combined_across_chunks(self):
        phrase = DifficultPhrase(
            phrase="lugar", category="noun", difficulty_level="A1",
            why_difficult="test phrase",
        )
        result = ReadingCoachResult(
            original_spanish=CHUNK_TEXT,
            overall_level="B1",
            difficult_phrases=[phrase],
        )
        check = CoachCheckResult(status=STATUS_PASSED, summary="ok", issues=[])
        ar = AnalysisResult(result=result, check=check, raw_response="{}")
        chunk = ChunkAnalysisResult(
            chunk_index=0, total_chunks=1, chunk_text=CHUNK_TEXT, analysis=ar
        )
        m = MultiChunkAnalysisResult(
            original_spanish=CHUNK_TEXT,
            chunks=[chunk],
            prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
            total_chunks=1,
            successful_chunks=1,
            failed_chunks=0,
        )
        phrases = m.all_difficult_phrases
        assert len(phrases) == 1
        assert phrases[0].phrase == "lugar"

    def test_all_difficult_phrases_skips_failed_chunks(self):
        failed = _make_chunk(succeeded=False)
        m = MultiChunkAnalysisResult(
            original_spanish=CHUNK_TEXT,
            chunks=[failed],
            prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
            total_chunks=1,
            successful_chunks=0,
            failed_chunks=1,
        )
        assert m.all_difficult_phrases == []

    def test_all_difficult_phrases_from_two_chunks(self):
        def _phrase(text: str) -> DifficultPhrase:
            return DifficultPhrase(
                phrase=text, category="idiom", difficulty_level="B1",
                why_difficult="test",
            )

        def _chunk_with_phrase(i: int, phrase_text: str) -> ChunkAnalysisResult:
            result = ReadingCoachResult(
                original_spanish=CHUNK_TEXT,
                overall_level="B1",
                difficult_phrases=[_phrase(phrase_text)],
            )
            check = CoachCheckResult(status=STATUS_PASSED, summary="ok", issues=[])
            ar = AnalysisResult(result=result, check=check, raw_response="{}")
            return ChunkAnalysisResult(
                chunk_index=i, total_chunks=2, chunk_text=CHUNK_TEXT, analysis=ar
            )

        m = MultiChunkAnalysisResult(
            original_spanish=CHUNK_TEXT,
            chunks=[_chunk_with_phrase(0, "lugar"), _chunk_with_phrase(1, "Mancha")],
            prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
            total_chunks=2,
            successful_chunks=2,
            failed_chunks=0,
        )
        phrases = m.all_difficult_phrases
        assert len(phrases) == 2
        assert phrases[0].phrase == "lugar"
        assert phrases[1].phrase == "Mancha"

    def test_empty_chunks_list(self):
        m = MultiChunkAnalysisResult(
            original_spanish="",
            chunks=[],
            prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
            total_chunks=0,
            successful_chunks=0,
            failed_chunks=0,
        )
        assert m.all_difficult_phrases == []
        assert m.total_chunks == 0
