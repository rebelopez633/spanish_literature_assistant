"""
reading_coach/multi_chunk_analyzer.py — Multi-chunk orchestration layer.

The single public function is:

    analyze_spanish_source_chunks(
        source_text: str,
        *,
        max_chars_per_chunk: int = 2200,
        reader_level: str = "B1",
        annotation_density: str = "balanced",
        include_english_gloss: bool = True,
        include_modern_spanish: bool = True,
        llm_client: Callable[[list[dict[str, str]]], str],
        checker_config: CoachCheckerConfig | None = None,
        retry_config: RetryConfig | None = None,
    ) -> MultiChunkAnalysisResult

It splits *source_text* into chunks via :mod:`reading_coach.chunker`, calls
:func:`~reading_coach.analyzer.analyze_spanish_source` for each chunk, and
bundles the per-chunk results into a :class:`~reading_coach.schemas.MultiChunkAnalysisResult`.

Chunk failures are caught per-chunk and recorded in
:attr:`~reading_coach.schemas.ChunkAnalysisResult.error`; they never abort
the remaining chunks.

No Streamlit, no Ollama, no network access.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from reading_coach.analyzer import analyze_spanish_source
from reading_coach.checker import CoachCheckerConfig
from reading_coach.chunker import split_into_chunks
from reading_coach.prompts import SPANISH_SOURCE_PROMPT_VERSION
from reading_coach.retry import RetryConfig
from reading_coach.schemas import ChunkAnalysisResult, MultiChunkAnalysisResult

logger = logging.getLogger(__name__)


def analyze_spanish_source_chunks(
    source_text: str,
    *,
    max_chars_per_chunk: int = 2200,
    reader_level: str = "B1",
    annotation_density: str = "balanced",
    include_english_gloss: bool = True,
    include_modern_spanish: bool = True,
    llm_client: Callable[[list[dict[str, str]]], str],
    checker_config: Optional[CoachCheckerConfig] = None,
    retry_config: RetryConfig | None = None,
) -> MultiChunkAnalysisResult:
    """Analyse a (potentially long) Spanish passage by splitting into chunks.

    Parameters
    ----------
    source_text:
        The full original Spanish passage.  Preserved verbatim in
        :attr:`~reading_coach.schemas.MultiChunkAnalysisResult.original_spanish`.
    max_chars_per_chunk:
        Maximum characters per chunk.  Passed to
        :func:`~reading_coach.chunker.split_into_chunks`.
    reader_level, annotation_density, include_english_gloss, include_modern_spanish:
        Forwarded unchanged to :func:`~reading_coach.analyzer.analyze_spanish_source`
        for every chunk.
    llm_client:
        Callable injected into each per-chunk :func:`analyze_spanish_source` call.
    checker_config:
        Optional checker configuration forwarded to each chunk analysis.
    retry_config:
        Optional retry policy forwarded to each chunk analysis.

    Returns
    -------
    MultiChunkAnalysisResult
        Contains the verbatim source, per-chunk results, and aggregate counts.
        Chunks that fail are recorded with :attr:`~ChunkAnalysisResult.error`
        populated and :attr:`~ChunkAnalysisResult.analysis` as ``None``.
    """
    chunks_text = split_into_chunks(source_text, max_chars_per_chunk)
    total = len(chunks_text)
    chunk_results: list[ChunkAnalysisResult] = []

    for i, chunk_text in enumerate(chunks_text):
        try:
            ar = analyze_spanish_source(
                chunk_text,
                reader_level=reader_level,
                annotation_density=annotation_density,
                include_english_gloss=include_english_gloss,
                include_modern_spanish=include_modern_spanish,
                llm_client=llm_client,
                checker_config=checker_config,
                retry_config=retry_config,
            )
            chunk_results.append(
                ChunkAnalysisResult(
                    chunk_index=i,
                    total_chunks=total,
                    chunk_text=chunk_text,
                    analysis=ar,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Chunk %d/%d failed (%s: %s)",
                i + 1,
                total,
                type(exc).__name__,
                exc,
            )
            chunk_results.append(
                ChunkAnalysisResult(
                    chunk_index=i,
                    total_chunks=total,
                    chunk_text=chunk_text,
                    analysis=None,
                    error=str(exc),
                )
            )

    successful = sum(1 for c in chunk_results if c.succeeded)
    failed = sum(1 for c in chunk_results if not c.succeeded)

    return MultiChunkAnalysisResult(
        original_spanish=source_text,
        chunks=chunk_results,
        prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
        total_chunks=total,
        successful_chunks=successful,
        failed_chunks=failed,
    )
