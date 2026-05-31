"""reading_coach/metadata.py — Lightweight observability metadata for coach analyses.

``AnalysisMetadata`` captures non-sensitive diagnostic information about a
single analysis pass (single-chunk or multi-chunk) to help debug local model
quality and performance.

Design constraints
------------------
* No network I/O.
* No Streamlit imports.
* No system-level data (no hostname, username, process ID, etc.).
* Suitable for local logging and Markdown export only — never sent externally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AnalysisMetadata:
    """Non-sensitive diagnostic metadata for one reading-coach analysis pass.

    Attributes
    ----------
    prompt_version:
        The prompt version string (e.g. ``"spanish_source_v2"``).
    reader_level:
        CEFR level supplied by the user (e.g. ``"B1"``).
    annotation_density:
        Density setting used (``"minimal"``, ``"balanced"``, or ``"detailed"``).
    chunk_count:
        Total number of chunks the passage was split into (``1`` for a
        single-chunk analysis).
    successful_chunk_count:
        Number of chunks that were analysed without error.
    failed_chunk_count:
        Number of chunks that raised a parse or LLM error.
    checker_status:
        Aggregated checker outcome (``"passed"``, ``"warning"``, ``"failed"``,
        or a human-readable summary string for multi-chunk analyses).
    annotation_limit_policy:
        The annotation-limit policy in effect (``"warn"``, ``"truncate"``, or
        ``"fail"``).
    max_annotations:
        Maximum annotations allowed per chunk.
    model_name:
        Model tag (e.g. ``"qwen2.5:7b"``).  ``None`` when not supplied by the
        caller (the metadata layer has no direct access to the Ollama adapter).
    parser_strategy:
        Which JSON-extraction strategy the response parser used (``"fences"``,
        ``"raw_decode"``, etc.).  ``None`` until the parser exposes this.
    retry_attempts:
        Number of retry attempts that were made.  ``None`` until the retry
        wrapper exposes this count.
    """

    prompt_version: str
    reader_level: str
    annotation_density: str
    chunk_count: int
    successful_chunk_count: int
    failed_chunk_count: int
    checker_status: str
    annotation_limit_policy: str
    max_annotations: int
    model_name: Optional[str] = field(default=None)
    parser_strategy: Optional[str] = field(default=None)
    retry_attempts: Optional[int] = field(default=None)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_metadata_from_single(
    *,
    prompt_version: str,
    reader_level: str,
    annotation_density: str,
    checker_status: str,
    checker_config=None,
    model_name: Optional[str] = None,
) -> AnalysisMetadata:
    """Build :class:`AnalysisMetadata` for a single-chunk analysis pass.

    Parameters
    ----------
    prompt_version, reader_level, annotation_density:
        Forwarded directly from the call-site.
    checker_status:
        The ``CoachCheckResult.status`` value for this pass.
    checker_config:
        Optional ``CoachCheckerConfig``; defaults are used when ``None``.
    model_name:
        Model tag to record, or ``None``.
    """
    # Lazy import to avoid a circular dependency.
    from reading_coach.checker import CoachCheckerConfig

    cfg: CoachCheckerConfig = checker_config if checker_config is not None else CoachCheckerConfig()
    return AnalysisMetadata(
        prompt_version=prompt_version,
        reader_level=reader_level,
        annotation_density=annotation_density,
        chunk_count=1,
        successful_chunk_count=1,
        failed_chunk_count=0,
        checker_status=checker_status,
        annotation_limit_policy=cfg.annotation_limit_policy,
        max_annotations=cfg.max_annotations,
        model_name=model_name,
    )


def build_metadata_from_multi(
    *,
    prompt_version: str,
    reader_level: str,
    annotation_density: str,
    total_chunks: int,
    successful_chunks: int,
    failed_chunks: int,
    checker_summary: Optional[str] = None,
    checker_config=None,
    model_name: Optional[str] = None,
) -> AnalysisMetadata:
    """Build :class:`AnalysisMetadata` for a multi-chunk analysis pass.

    Parameters
    ----------
    total_chunks, successful_chunks, failed_chunks:
        Aggregate counts from :class:`~reading_coach.schemas.MultiChunkAnalysisResult`.
    checker_summary:
        Human-readable checker summary (e.g. ``"3/4 passed"``); used as the
        ``checker_status`` field when provided.
    checker_config:
        Optional ``CoachCheckerConfig``; defaults are used when ``None``.
    model_name:
        Model tag to record, or ``None``.
    """
    from reading_coach.checker import CoachCheckerConfig

    cfg: CoachCheckerConfig = checker_config if checker_config is not None else CoachCheckerConfig()
    _status = checker_summary or ("passed" if failed_chunks == 0 else "partial")
    return AnalysisMetadata(
        prompt_version=prompt_version,
        reader_level=reader_level,
        annotation_density=annotation_density,
        chunk_count=total_chunks,
        successful_chunk_count=successful_chunks,
        failed_chunk_count=failed_chunks,
        checker_status=_status,
        annotation_limit_policy=cfg.annotation_limit_policy,
        max_annotations=cfg.max_annotations,
        model_name=model_name,
    )
