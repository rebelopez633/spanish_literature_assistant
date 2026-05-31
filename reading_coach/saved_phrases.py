"""reading_coach/saved_phrases.py — Helpers to convert DifficultPhrase objects
into SavedPhrase records.

Three public functions
----------------------
:func:`saved_phrase_from_difficult_phrase`
    Convert a single :class:`~reading_coach.schemas.DifficultPhrase` into a
    :class:`~reading_coach.session.SavedPhrase`.

:func:`saved_phrases_from_result`
    Convert all difficult phrases in a
    :class:`~reading_coach.schemas.ReadingCoachResult` into a list of
    :class:`~reading_coach.session.SavedPhrase` records.

:func:`saved_phrases_from_multi_chunk_result`
    Convert all difficult phrases across every successful chunk in a
    :class:`~reading_coach.schemas.MultiChunkAnalysisResult`.  Each phrase
    receives the ``chunk_id`` from its source chunk.

Design notes
------------
* **Injectable ID factory** — every function accepts an optional ``id_factory``
  callable (``() -> str``).  When omitted, the default UUID4 factory from
  :mod:`reading_coach.session` is used.  Passing a deterministic factory in
  tests makes phrase IDs stable without monkey-patching.
* **source_context** — callers supply the surrounding text explicitly; the
  multi-chunk helper uses each chunk's ``chunk_text`` automatically.
* **Deduplication** — pass ``deduplicate=True`` to drop subsequent phrases that
  share the same ``(phrase, source_context)`` pair.  The first occurrence is
  kept; order is preserved.
* **No Streamlit, no MongoDB, no Ollama.**
"""
from __future__ import annotations

from typing import Callable, List, Optional

from reading_coach.schemas import DifficultPhrase, MultiChunkAnalysisResult, ReadingCoachResult
from reading_coach.session import SavedPhrase, _new_id

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def saved_phrase_from_difficult_phrase(
    phrase: DifficultPhrase,
    *,
    source_context: str,
    session_id: Optional[str] = None,
    chunk_id: Optional[str] = None,
    id_factory: Callable[[], str] = _new_id,
) -> SavedPhrase:
    """Convert a single :class:`~reading_coach.schemas.DifficultPhrase` into a
    :class:`~reading_coach.session.SavedPhrase`.

    Parameters
    ----------
    phrase:
        The difficult phrase to convert.
    source_context:
        The surrounding text from which *phrase* was extracted.  Stored
        verbatim — no whitespace stripping or normalisation is applied.
    session_id:
        Optional back-reference to the originating
        :class:`~reading_coach.session.CoachSession`.
    chunk_id:
        Optional back-reference to the originating chunk (for multi-chunk
        sessions).
    id_factory:
        Zero-argument callable that returns a unique string ID for the new
        :class:`~reading_coach.session.SavedPhrase`.  Defaults to UUID4.
        Supply a deterministic factory in tests.
    """
    return SavedPhrase(
        phrase_id=id_factory(),
        phrase=phrase.phrase,
        source_context=source_context,
        category=phrase.category,
        difficulty_level=phrase.difficulty_level,
        english_meaning=phrase.english_meaning,
        modern_spanish_equivalent=phrase.modern_spanish_equivalent,
        grammar_note=phrase.grammar_note,
        learner_tip=phrase.learner_tip,
        session_id=session_id,
        chunk_id=chunk_id,
    )


def saved_phrases_from_result(
    result: ReadingCoachResult,
    *,
    source_context: Optional[str] = None,
    session_id: Optional[str] = None,
    id_factory: Callable[[], str] = _new_id,
    deduplicate: bool = False,
) -> List[SavedPhrase]:
    """Convert all difficult phrases in a
    :class:`~reading_coach.schemas.ReadingCoachResult` into
    :class:`~reading_coach.session.SavedPhrase` records.

    Parameters
    ----------
    result:
        The single-chunk analysis result whose ``difficult_phrases`` to convert.
    source_context:
        The surrounding text to attach to every saved phrase.  When omitted,
        falls back to ``result.original_spanish``.
    session_id:
        Optional back-reference to the originating session.
    id_factory:
        Zero-argument callable used to generate ``phrase_id`` values.
    deduplicate:
        When ``True``, drop subsequent entries that share the same
        ``(phrase, source_context)`` pair.  Insertion order is preserved and
        the first occurrence is kept.
    """
    ctx = source_context if source_context is not None else result.original_spanish
    seen: set[tuple[str, str]] = set()
    out: List[SavedPhrase] = []

    for dp in result.difficult_phrases:
        key = (dp.phrase, ctx)
        if deduplicate and key in seen:
            continue
        seen.add(key)
        out.append(
            saved_phrase_from_difficult_phrase(
                dp,
                source_context=ctx,
                session_id=session_id,
                id_factory=id_factory,
            )
        )

    return out


def saved_phrases_from_multi_chunk_result(
    result: MultiChunkAnalysisResult,
    *,
    session_id: Optional[str] = None,
    id_factory: Callable[[], str] = _new_id,
    deduplicate: bool = False,
) -> List[SavedPhrase]:
    """Convert all difficult phrases across every successful chunk in a
    :class:`~reading_coach.schemas.MultiChunkAnalysisResult`.

    Failed chunks (``chunk.analysis is None``) are silently skipped.  Each
    phrase receives the ``chunk_id`` from its source chunk and uses the
    chunk's ``chunk_text`` as ``source_context``.

    Parameters
    ----------
    result:
        The multi-chunk analysis result to convert.
    session_id:
        Optional back-reference propagated to every saved phrase.
    id_factory:
        Zero-argument callable used to generate ``phrase_id`` values.
    deduplicate:
        When ``True``, drop subsequent entries that share the same
        ``(phrase, source_context)`` pair across all chunks.
    """
    seen: set[tuple[str, str]] = set()
    out: List[SavedPhrase] = []

    for chunk in result.chunks:
        if chunk.analysis is None:
            continue  # skip failed chunks

        ctx = chunk.chunk_text
        for dp in chunk.analysis.result.difficult_phrases:
            key = (dp.phrase, ctx)
            if deduplicate and key in seen:
                continue
            seen.add(key)
            out.append(
                saved_phrase_from_difficult_phrase(
                    dp,
                    source_context=ctx,
                    session_id=session_id,
                    chunk_id=chunk.chunk_id or None,
                    id_factory=id_factory,
                )
            )

    return out
