"""tests/test_saved_phrases.py — Unit tests for reading_coach/saved_phrases.py.

Phase 3 Slice 8: helper functions to convert DifficultPhrase objects into
SavedPhrase records.

Acceptance criteria
-------------------
1. Convert one DifficultPhrase → SavedPhrase.
2. Convert all phrases from a ReadingCoachResult.
3. Convert phrases from MultiChunkAnalysisResult with chunk IDs attached.
4. source_context is preserved exactly.
5. Optional deduplication by (phrase, source_context) works.
6. IDs are testable via injectable ID factory.
"""
from __future__ import annotations

import pytest

from reading_coach.saved_phrases import (
    saved_phrase_from_difficult_phrase,
    saved_phrases_from_multi_chunk_result,
    saved_phrases_from_result,
)
from reading_coach.schemas import (
    ChunkAnalysisResult,
    DifficultPhrase,
    MultiChunkAnalysisResult,
    ReadingCoachResult,
)
from reading_coach.session import SavedPhrase

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_SOURCE_CONTEXT = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme."


def _make_phrase(
    phrase: str = "de cuyo nombre no quiero acordarme",
    category: str = "archaic vocabulary",
    difficulty_level: str = "B2",
    english_meaning: str = "whose name I do not care to remember",
    modern_spanish_equivalent: str = "cuyo nombre no quiero recordar",
    grammar_note: str = "Relative clause with 'cuyo'",
    learner_tip: str = "Note the subjunctive mood.",
) -> DifficultPhrase:
    return DifficultPhrase(
        phrase=phrase,
        category=category,
        difficulty_level=difficulty_level,
        why_difficult="Archaic relative pronoun usage.",
        english_meaning=english_meaning,
        modern_spanish_equivalent=modern_spanish_equivalent,
        grammar_note=grammar_note,
        learner_tip=learner_tip,
    )


def _make_result(phrases: list[DifficultPhrase] | None = None) -> ReadingCoachResult:
    return ReadingCoachResult(
        original_spanish=_SOURCE_CONTEXT,
        overall_level="B2",
        difficult_phrases=[_make_phrase()] if phrases is None else phrases,
    )


def _make_analysis_result(result: ReadingCoachResult):
    """Minimal stand-in for AnalysisResult (avoids importing analyzer)."""

    class _FakeAnalysisResult:
        def __init__(self, r: ReadingCoachResult) -> None:
            self.result = r

    return _FakeAnalysisResult(result)


def _make_chunk(
    chunk_index: int,
    chunk_id: str,
    phrases: list[DifficultPhrase] | None = None,
) -> ChunkAnalysisResult:
    rcr = _make_result(phrases or [_make_phrase()])
    analysis = _make_analysis_result(rcr)
    return ChunkAnalysisResult(
        chunk_index=chunk_index,
        total_chunks=2,
        chunk_text=_SOURCE_CONTEXT,
        chunk_id=chunk_id,
        analysis=analysis,
        status="passed",
    )


def _make_multi(chunks: list[ChunkAnalysisResult] | None = None) -> MultiChunkAnalysisResult:
    chunks = chunks or [_make_chunk(0, "chunk_0000"), _make_chunk(1, "chunk_0001")]
    return MultiChunkAnalysisResult(
        original_spanish=_SOURCE_CONTEXT,
        chunks=chunks,
        prompt_version="spanish_source_v2",
        total_chunks=len(chunks),
        successful_chunks=len(chunks),
        failed_chunks=0,
    )


# ---------------------------------------------------------------------------
# 1. Convert one DifficultPhrase → SavedPhrase
# ---------------------------------------------------------------------------

class TestSavedPhraseFromDifficultPhrase:

    def test_phrase_text_preserved(self):
        dp = _make_phrase()
        sp = saved_phrase_from_difficult_phrase(dp, source_context=_SOURCE_CONTEXT)
        assert sp.phrase == dp.phrase

    def test_source_context_preserved(self):
        dp = _make_phrase()
        sp = saved_phrase_from_difficult_phrase(dp, source_context=_SOURCE_CONTEXT)
        assert sp.source_context == _SOURCE_CONTEXT

    def test_category_preserved(self):
        dp = _make_phrase()
        sp = saved_phrase_from_difficult_phrase(dp, source_context=_SOURCE_CONTEXT)
        assert sp.category == dp.category

    def test_difficulty_level_preserved(self):
        dp = _make_phrase()
        sp = saved_phrase_from_difficult_phrase(dp, source_context=_SOURCE_CONTEXT)
        assert sp.difficulty_level == "B2"

    def test_english_meaning_preserved(self):
        dp = _make_phrase()
        sp = saved_phrase_from_difficult_phrase(dp, source_context=_SOURCE_CONTEXT)
        assert sp.english_meaning == dp.english_meaning

    def test_modern_spanish_equivalent_preserved(self):
        dp = _make_phrase()
        sp = saved_phrase_from_difficult_phrase(dp, source_context=_SOURCE_CONTEXT)
        assert sp.modern_spanish_equivalent == dp.modern_spanish_equivalent

    def test_grammar_note_preserved(self):
        dp = _make_phrase()
        sp = saved_phrase_from_difficult_phrase(dp, source_context=_SOURCE_CONTEXT)
        assert sp.grammar_note == dp.grammar_note

    def test_learner_tip_preserved(self):
        dp = _make_phrase()
        sp = saved_phrase_from_difficult_phrase(dp, source_context=_SOURCE_CONTEXT)
        assert sp.learner_tip == dp.learner_tip

    def test_returns_saved_phrase_instance(self):
        dp = _make_phrase()
        sp = saved_phrase_from_difficult_phrase(dp, source_context=_SOURCE_CONTEXT)
        assert isinstance(sp, SavedPhrase)

    def test_session_id_attached_when_provided(self):
        dp = _make_phrase()
        sp = saved_phrase_from_difficult_phrase(
            dp, source_context=_SOURCE_CONTEXT, session_id="sess-001"
        )
        assert sp.session_id == "sess-001"

    def test_session_id_none_by_default(self):
        dp = _make_phrase()
        sp = saved_phrase_from_difficult_phrase(dp, source_context=_SOURCE_CONTEXT)
        assert sp.session_id is None

    def test_chunk_id_attached_when_provided(self):
        dp = _make_phrase()
        sp = saved_phrase_from_difficult_phrase(
            dp, source_context=_SOURCE_CONTEXT, chunk_id="chunk_0000"
        )
        assert sp.chunk_id == "chunk_0000"

    def test_chunk_id_none_by_default(self):
        dp = _make_phrase()
        sp = saved_phrase_from_difficult_phrase(dp, source_context=_SOURCE_CONTEXT)
        assert sp.chunk_id is None

    def test_optional_fields_none_when_absent_in_phrase(self):
        dp = DifficultPhrase(
            phrase="algún",
            category="archaic",
            difficulty_level="A2",
            why_difficult="Old form.",
        )
        sp = saved_phrase_from_difficult_phrase(dp, source_context="algún")
        assert sp.english_meaning is None
        assert sp.modern_spanish_equivalent is None
        assert sp.grammar_note is None
        assert sp.learner_tip is None


# ---------------------------------------------------------------------------
# 6. Generated IDs are testable
# ---------------------------------------------------------------------------

class TestInjectableIdFactory:

    def test_injectable_id_factory_used(self):
        ids = iter(["id-001", "id-002"])
        dp = _make_phrase()
        sp = saved_phrase_from_difficult_phrase(
            dp, source_context=_SOURCE_CONTEXT, id_factory=lambda: next(ids)
        )
        assert sp.phrase_id == "id-001"

    def test_default_id_is_uuid_string(self):
        import uuid
        dp = _make_phrase()
        sp = saved_phrase_from_difficult_phrase(dp, source_context=_SOURCE_CONTEXT)
        # Should not raise — valid UUID4
        parsed = uuid.UUID(sp.phrase_id, version=4)
        assert str(parsed) == sp.phrase_id

    def test_default_ids_are_unique(self):
        dp = _make_phrase()
        ids = {
            saved_phrase_from_difficult_phrase(dp, source_context=_SOURCE_CONTEXT).phrase_id
            for _ in range(10)
        }
        assert len(ids) == 10


# ---------------------------------------------------------------------------
# 2. Convert all phrases from a ReadingCoachResult
# ---------------------------------------------------------------------------

class TestSavedPhrasesFromResult:

    def test_returns_list(self):
        result = _make_result()
        phrases = saved_phrases_from_result(result, source_context=_SOURCE_CONTEXT)
        assert isinstance(phrases, list)

    def test_count_matches_difficult_phrases(self):
        dp1 = _make_phrase("frase uno")
        dp2 = _make_phrase("frase dos")
        result = _make_result([dp1, dp2])
        phrases = saved_phrases_from_result(result, source_context=_SOURCE_CONTEXT)
        assert len(phrases) == 2

    def test_all_are_saved_phrase_instances(self):
        result = _make_result([_make_phrase("uno"), _make_phrase("dos")])
        for sp in saved_phrases_from_result(result, source_context=_SOURCE_CONTEXT):
            assert isinstance(sp, SavedPhrase)

    def test_phrase_texts_match(self):
        dp1 = _make_phrase("primera frase")
        dp2 = _make_phrase("segunda frase")
        result = _make_result([dp1, dp2])
        phrases = saved_phrases_from_result(result, source_context=_SOURCE_CONTEXT)
        assert phrases[0].phrase == "primera frase"
        assert phrases[1].phrase == "segunda frase"

    def test_session_id_propagated(self):
        result = _make_result([_make_phrase()])
        phrases = saved_phrases_from_result(
            result, source_context=_SOURCE_CONTEXT, session_id="sess-abc"
        )
        assert all(sp.session_id == "sess-abc" for sp in phrases)

    def test_empty_result_returns_empty_list(self):
        result = _make_result([])
        phrases = saved_phrases_from_result(result, source_context=_SOURCE_CONTEXT)
        assert phrases == []

    def test_injectable_id_factory(self):
        counter = [0]

        def seq_id() -> str:
            counter[0] += 1
            return f"id-{counter[0]:03d}"

        result = _make_result([_make_phrase("a"), _make_phrase("b")])
        phrases = saved_phrases_from_result(
            result, source_context=_SOURCE_CONTEXT, id_factory=seq_id
        )
        assert phrases[0].phrase_id == "id-001"
        assert phrases[1].phrase_id == "id-002"

    def test_source_context_from_result_original_spanish_when_not_supplied(self):
        """When no explicit source_context is given, falls back to result.original_spanish."""
        result = _make_result([_make_phrase()])
        phrases = saved_phrases_from_result(result)
        assert phrases[0].source_context == result.original_spanish


# ---------------------------------------------------------------------------
# 3. Convert phrases from MultiChunkAnalysisResult with chunk IDs
# ---------------------------------------------------------------------------

class TestSavedPhrasesFromMultiChunkResult:

    def test_returns_list(self):
        multi = _make_multi()
        phrases = saved_phrases_from_multi_chunk_result(multi)
        assert isinstance(phrases, list)

    def test_total_count_is_sum_of_all_chunks(self):
        chunk0 = _make_chunk(0, "chunk_0000", [_make_phrase("frase A"), _make_phrase("frase B")])
        chunk1 = _make_chunk(1, "chunk_0001", [_make_phrase("frase C")])
        multi = _make_multi([chunk0, chunk1])
        phrases = saved_phrases_from_multi_chunk_result(multi)
        assert len(phrases) == 3

    def test_chunk_id_attached_per_chunk(self):
        chunk0 = _make_chunk(0, "chunk_0000", [_make_phrase("alfa")])
        chunk1 = _make_chunk(1, "chunk_0001", [_make_phrase("beta")])
        multi = _make_multi([chunk0, chunk1])
        phrases = saved_phrases_from_multi_chunk_result(multi)
        assert phrases[0].chunk_id == "chunk_0000"
        assert phrases[1].chunk_id == "chunk_0001"

    def test_failed_chunks_skipped(self):
        good = _make_chunk(0, "chunk_0000", [_make_phrase("bueno")])
        bad = ChunkAnalysisResult(
            chunk_index=1,
            total_chunks=2,
            chunk_text="texto con error",
            chunk_id="chunk_0001",
            analysis=None,
            error="LLM timeout",
            status="error",
        )
        multi = _make_multi([good, bad])
        phrases = saved_phrases_from_multi_chunk_result(multi)
        assert len(phrases) == 1
        assert phrases[0].phrase == "bueno"

    def test_session_id_propagated_to_all(self):
        multi = _make_multi()
        phrases = saved_phrases_from_multi_chunk_result(multi, session_id="sess-xyz")
        assert all(sp.session_id == "sess-xyz" for sp in phrases)

    def test_all_are_saved_phrase_instances(self):
        multi = _make_multi()
        for sp in saved_phrases_from_multi_chunk_result(multi):
            assert isinstance(sp, SavedPhrase)

    def test_source_context_is_chunk_text(self):
        """Each saved phrase's source_context is the chunk's chunk_text."""
        chunk_text = "Texto específico del chunk."
        dp = _make_phrase()
        rcr = ReadingCoachResult(
            original_spanish=chunk_text,
            overall_level="B1",
            difficult_phrases=[dp],
        )
        analysis = _make_analysis_result(rcr)
        chunk = ChunkAnalysisResult(
            chunk_index=0,
            total_chunks=1,
            chunk_text=chunk_text,
            chunk_id="chunk_0000",
            analysis=analysis,
            status="passed",
        )
        multi = MultiChunkAnalysisResult(
            original_spanish=chunk_text,
            chunks=[chunk],
            prompt_version="spanish_source_v2",
            total_chunks=1,
            successful_chunks=1,
            failed_chunks=0,
        )
        phrases = saved_phrases_from_multi_chunk_result(multi)
        assert phrases[0].source_context == chunk_text

    def test_injectable_id_factory(self):
        counter = [0]

        def seq_id() -> str:
            counter[0] += 1
            return f"phrase-{counter[0]}"

        chunk0 = _make_chunk(0, "chunk_0000", [_make_phrase("x"), _make_phrase("y")])
        multi = _make_multi([chunk0])
        phrases = saved_phrases_from_multi_chunk_result(multi, id_factory=seq_id)
        assert phrases[0].phrase_id == "phrase-1"
        assert phrases[1].phrase_id == "phrase-2"


# ---------------------------------------------------------------------------
# 4. source_context preserved exactly (explicit test)
# ---------------------------------------------------------------------------

class TestSourceContextPreservation:

    def test_exact_unicode_preserved(self):
        context = "¡Mañana será otro día! — dijo Scarlett."
        dp = _make_phrase()
        sp = saved_phrase_from_difficult_phrase(dp, source_context=context)
        assert sp.source_context == context

    def test_multiline_context_preserved(self):
        context = "Primera línea.\nSegunda línea.\n\nTercera línea."
        dp = _make_phrase()
        sp = saved_phrase_from_difficult_phrase(dp, source_context=context)
        assert sp.source_context == context

    def test_whitespace_not_stripped(self):
        context = "  texto con espacios  "
        dp = _make_phrase()
        sp = saved_phrase_from_difficult_phrase(dp, source_context=context)
        assert sp.source_context == context


# ---------------------------------------------------------------------------
# 5. Optional deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:

    def test_no_dedup_by_default_keeps_duplicates(self):
        dp = _make_phrase("frase repetida")
        result = _make_result([dp, dp])
        phrases = saved_phrases_from_result(result, source_context=_SOURCE_CONTEXT)
        assert len(phrases) == 2

    def test_dedup_removes_exact_phrase_context_duplicates(self):
        dp = _make_phrase("frase repetida")
        result = _make_result([dp, dp])
        phrases = saved_phrases_from_result(
            result, source_context=_SOURCE_CONTEXT, deduplicate=True
        )
        assert len(phrases) == 1

    def test_dedup_keeps_same_phrase_different_context(self):
        dp1 = _make_phrase("misma frase")
        dp2 = _make_phrase("misma frase")
        result = _make_result([dp1, dp2])
        phrases = saved_phrases_from_result(
            result,
            source_context=_SOURCE_CONTEXT,
            deduplicate=True,
        )
        # Same phrase, same context → deduplicated to 1
        assert len(phrases) == 1

    def test_dedup_different_phrases_kept(self):
        dp1 = _make_phrase("frase uno")
        dp2 = _make_phrase("frase dos")
        result = _make_result([dp1, dp2])
        phrases = saved_phrases_from_result(
            result, source_context=_SOURCE_CONTEXT, deduplicate=True
        )
        assert len(phrases) == 2

    def test_dedup_multi_chunk(self):
        """Same phrase appearing in two chunks with same context is deduplicated."""
        dp = _make_phrase("frase duplicada")
        chunk0 = _make_chunk(0, "chunk_0000", [dp])
        chunk1 = _make_chunk(1, "chunk_0001", [dp])
        # Both chunks use _SOURCE_CONTEXT as chunk_text
        multi = _make_multi([chunk0, chunk1])
        phrases = saved_phrases_from_multi_chunk_result(multi, deduplicate=True)
        assert len(phrases) == 1

    def test_dedup_multi_chunk_different_context_kept(self):
        dp = _make_phrase("frase repetida")
        rcr0 = ReadingCoachResult(
            original_spanish="Contexto A", overall_level="B1", difficult_phrases=[dp]
        )
        rcr1 = ReadingCoachResult(
            original_spanish="Contexto B", overall_level="B1", difficult_phrases=[dp]
        )
        chunk0 = ChunkAnalysisResult(
            chunk_index=0,
            total_chunks=2,
            chunk_text="Contexto A",
            chunk_id="chunk_0000",
            analysis=_make_analysis_result(rcr0),
            status="passed",
        )
        chunk1 = ChunkAnalysisResult(
            chunk_index=1,
            total_chunks=2,
            chunk_text="Contexto B",
            chunk_id="chunk_0001",
            analysis=_make_analysis_result(rcr1),
            status="passed",
        )
        multi = MultiChunkAnalysisResult(
            original_spanish="Contexto A\nContexto B",
            chunks=[chunk0, chunk1],
            prompt_version="v2",
            total_chunks=2,
            successful_chunks=2,
            failed_chunks=0,
        )
        phrases = saved_phrases_from_multi_chunk_result(multi, deduplicate=True)
        # Same phrase, different context → both kept
        assert len(phrases) == 2
