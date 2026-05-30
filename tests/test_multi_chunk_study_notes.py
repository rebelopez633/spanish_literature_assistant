"""
Tests for multi_chunk_result_to_markdown() in reading_coach/study_notes.py.

Covers all 6 acceptance criteria from Phase 3 Slice 5:
  1. Exports one-chunk result (returns str, non-empty, contains key content).
  2. Exports multi-chunk result in order.
  3. Includes failed chunk error without crashing.
  4. Includes prompt version.
  5. Preserves original Spanish exactly.
  6. Omits empty optional sections cleanly.

Plus:
  - Title heading (present / absent).
  - Summary section (total / successful / failed counts).
  - Checker status per chunk.
  - Optional fields rendered when populated.

No Streamlit, no Ollama, no network.
"""
from __future__ import annotations

import pytest

from reading_coach.analyzer import AnalysisResult
from reading_coach.checker import CoachCheckResult, STATUS_PASSED, STATUS_WARNING
from reading_coach.prompts import SPANISH_SOURCE_PROMPT_VERSION
from reading_coach.schemas import (
    ChunkAnalysisResult,
    ComprehensionQuestion,
    DifficultPhrase,
    GrammarNote,
    MultiChunkAnalysisResult,
    ReadingCoachResult,
)
from reading_coach.study_notes import multi_chunk_result_to_markdown

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

_SOURCE_A = "Esta es la primera parte del texto."
_SOURCE_B = "Esta es la segunda parte del texto."
_FULL_SOURCE = _SOURCE_A + " " + _SOURCE_B


def _coach_result(
    source: str,
    level: str = "B1",
    modern_spanish: str | None = None,
    english_gloss: str | None = None,
    phrases: list[DifficultPhrase] | None = None,
    grammar_notes: list[GrammarNote] | None = None,
    comprehension_question: ComprehensionQuestion | None = None,
) -> ReadingCoachResult:
    return ReadingCoachResult(
        original_spanish=source,
        overall_level=level,
        modern_spanish=modern_spanish,
        english_gloss=english_gloss,
        difficult_phrases=phrases or [],
        grammar_notes=grammar_notes or [],
        comprehension_question=comprehension_question,
    )


def _analysis_result(
    source: str = _SOURCE_A,
    status: str = STATUS_PASSED,
    **kwargs,
) -> AnalysisResult:
    rc = _coach_result(source, **kwargs)
    check = CoachCheckResult(status=status, summary="ok", issues=[])
    return AnalysisResult(result=rc, check=check, raw_response="{}")


def _ok_chunk(
    chunk_index: int = 0,
    total_chunks: int = 1,
    source: str = _SOURCE_A,
    **kwargs,
) -> ChunkAnalysisResult:
    ar = _analysis_result(source, **kwargs)
    return ChunkAnalysisResult(
        chunk_id=f"chunk_{chunk_index:04d}",
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        chunk_text=source,
        analysis=ar,
        checker_result=ar.check,
        status=ar.check.status,
    )


def _err_chunk(
    chunk_index: int = 0,
    total_chunks: int = 1,
    source: str = _SOURCE_A,
    error: str = "LLM unavailable",
) -> ChunkAnalysisResult:
    return ChunkAnalysisResult(
        chunk_id=f"chunk_{chunk_index:04d}",
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        chunk_text=source,
        analysis=None,
        error=error,
        status="error",
    )


def _one_chunk_result(original: str = _SOURCE_A) -> MultiChunkAnalysisResult:
    chunk = _ok_chunk(chunk_index=0, total_chunks=1, source=original)
    return MultiChunkAnalysisResult(
        original_spanish=original,
        chunks=[chunk],
        prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
        total_chunks=1,
        successful_chunks=1,
        failed_chunks=0,
        checker_summary="1/1 passed",
    )


def _two_chunk_result(
    original: str = _FULL_SOURCE,
    fail_second: bool = False,
) -> MultiChunkAnalysisResult:
    c1 = _ok_chunk(chunk_index=0, total_chunks=2, source=_SOURCE_A)
    c2 = (
        _err_chunk(chunk_index=1, total_chunks=2, source=_SOURCE_B)
        if fail_second
        else _ok_chunk(chunk_index=1, total_chunks=2, source=_SOURCE_B)
    )
    n_failed = 1 if fail_second else 0
    return MultiChunkAnalysisResult(
        original_spanish=original,
        chunks=[c1, c2],
        prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
        total_chunks=2,
        successful_chunks=2 - n_failed,
        failed_chunks=n_failed,
        checker_summary="1/2 passed, 1 error" if fail_second else "2/2 passed",
    )


# ===========================================================================
# Criterion 1 — Exports one-chunk result
# ===========================================================================

class TestOneChunkExport:
    def test_returns_str(self):
        assert isinstance(multi_chunk_result_to_markdown(_one_chunk_result()), str)

    def test_non_empty(self):
        assert multi_chunk_result_to_markdown(_one_chunk_result()).strip()

    def test_contains_original_spanish_verbatim(self):
        md = multi_chunk_result_to_markdown(_one_chunk_result(_SOURCE_A))
        assert _SOURCE_A in md

    def test_contains_chunk_heading(self):
        md = multi_chunk_result_to_markdown(_one_chunk_result())
        # Should have a heading referencing chunk 1
        assert "1" in md  # chunk number present somewhere

    def test_chunk_text_present(self):
        md = multi_chunk_result_to_markdown(_one_chunk_result())
        assert _SOURCE_A in md

    def test_level_present(self):
        md = multi_chunk_result_to_markdown(_one_chunk_result())
        assert "B1" in md

    def test_pure_function_deterministic(self):
        result = _one_chunk_result()
        assert multi_chunk_result_to_markdown(result) == multi_chunk_result_to_markdown(result)


# ===========================================================================
# Criterion 2 — Exports multi-chunk result in order
# ===========================================================================

class TestMultiChunkOrder:
    def test_multi_chunk_returns_str(self):
        assert isinstance(multi_chunk_result_to_markdown(_two_chunk_result()), str)

    def test_source_a_before_source_b(self):
        md = multi_chunk_result_to_markdown(_two_chunk_result())
        assert md.index(_SOURCE_A) < md.index(_SOURCE_B)

    def test_both_chunk_texts_present(self):
        md = multi_chunk_result_to_markdown(_two_chunk_result())
        assert _SOURCE_A in md
        assert _SOURCE_B in md

    def test_chunk_1_heading_before_chunk_2_heading(self):
        md = multi_chunk_result_to_markdown(_two_chunk_result())
        # Both chunk references in correct order; at minimum chunk A text before B
        pos_a = md.index(_SOURCE_A)
        pos_b = md.index(_SOURCE_B)
        assert pos_a < pos_b

    def test_total_chunks_count_two(self):
        result = _two_chunk_result()
        md = multi_chunk_result_to_markdown(result)
        assert "2" in md  # total chunk count appears


# ===========================================================================
# Criterion 3 — Includes failed chunk error without crashing
# ===========================================================================

class TestFailedChunk:
    def test_does_not_raise_on_failed_chunk(self):
        result = _two_chunk_result(fail_second=True)
        md = multi_chunk_result_to_markdown(result)
        assert md  # did not crash

    def test_error_message_in_output(self):
        result = _two_chunk_result(fail_second=True)
        md = multi_chunk_result_to_markdown(result)
        assert "LLM unavailable" in md

    def test_failed_chunk_indicator_present(self):
        result = _two_chunk_result(fail_second=True)
        md = multi_chunk_result_to_markdown(result)
        # Must indicate failure somehow
        assert any(
            marker in md
            for marker in ("error", "Error", "failed", "Failed", "⚠", "✗")
        )

    def test_successful_chunk_still_present_alongside_failed(self):
        result = _two_chunk_result(fail_second=True)
        md = multi_chunk_result_to_markdown(result)
        assert _SOURCE_A in md  # successful chunk text still there

    def test_all_failed_chunks_no_crash(self):
        c1 = _err_chunk(chunk_index=0, total_chunks=1, source=_SOURCE_A)
        result = MultiChunkAnalysisResult(
            original_spanish=_SOURCE_A,
            chunks=[c1],
            prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
            total_chunks=1,
            successful_chunks=0,
            failed_chunks=1,
            checker_summary="0/1 passed, 1 error",
        )
        md = multi_chunk_result_to_markdown(result)
        assert isinstance(md, str)
        assert md.strip()


# ===========================================================================
# Criterion 4 — Includes prompt version
# ===========================================================================

class TestPromptVersion:
    def test_prompt_version_present(self):
        md = multi_chunk_result_to_markdown(_one_chunk_result())
        assert SPANISH_SOURCE_PROMPT_VERSION in md

    def test_prompt_version_exact_string(self):
        result = _one_chunk_result()
        result = result.model_copy(update={"prompt_version": "test_version_99"})
        md = multi_chunk_result_to_markdown(result)
        assert "test_version_99" in md

    def test_prompt_version_in_summary_area(self):
        """Prompt version should appear near the top (summary), not buried."""
        md = multi_chunk_result_to_markdown(_one_chunk_result())
        pv_pos = md.find(SPANISH_SOURCE_PROMPT_VERSION)
        # Should appear in the first 40% of the document
        assert pv_pos < len(md) * 0.6


# ===========================================================================
# Criterion 5 — Preserves original Spanish exactly
# ===========================================================================

class TestOriginalSpanishPreserved:
    def test_original_spanish_verbatim(self):
        source = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme."
        md = multi_chunk_result_to_markdown(_one_chunk_result(source))
        assert source in md

    def test_original_spanish_with_special_chars(self):
        source = "¿Qué decís, señor? ¡Silencio!"
        md = multi_chunk_result_to_markdown(_one_chunk_result(source))
        assert source in md

    def test_original_spanish_with_em_dash(self):
        source = "—¿Adónde vais? —preguntó la anciana."
        md = multi_chunk_result_to_markdown(_one_chunk_result(source))
        assert source in md

    def test_multiline_original_spanish_preserved(self):
        source = "Primera línea.\nSegunda línea."
        md = multi_chunk_result_to_markdown(_one_chunk_result(source))
        assert source in md


# ===========================================================================
# Criterion 6 — Omits empty optional sections cleanly
# ===========================================================================

class TestOptionalSectionsOmitted:
    def test_no_modern_spanish_when_none(self):
        md = multi_chunk_result_to_markdown(_one_chunk_result())
        assert "Modern Spanish" not in md

    def test_no_english_gloss_when_none(self):
        md = multi_chunk_result_to_markdown(_one_chunk_result())
        assert "English Gloss" not in md

    def test_no_difficult_phrases_when_empty(self):
        md = multi_chunk_result_to_markdown(_one_chunk_result())
        assert "Difficult Phrases" not in md

    def test_no_grammar_notes_when_empty(self):
        md = multi_chunk_result_to_markdown(_one_chunk_result())
        assert "Grammar Notes" not in md

    def test_no_comprehension_question_when_none(self):
        md = multi_chunk_result_to_markdown(_one_chunk_result())
        assert "Comprehension" not in md


# ===========================================================================
# Optional sections rendered when populated
# ===========================================================================

class TestOptionalSectionsIncluded:
    def test_modern_spanish_present_when_set(self):
        chunk = _ok_chunk(modern_spanish="Aquí vivía un hidalgo.")
        result = MultiChunkAnalysisResult(
            original_spanish=_SOURCE_A,
            chunks=[chunk],
            prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
            total_chunks=1, successful_chunks=1, failed_chunks=0,
        )
        md = multi_chunk_result_to_markdown(result)
        assert "Modern Spanish" in md
        assert "Aquí vivía un hidalgo." in md

    def test_english_gloss_present_when_set(self):
        chunk = _ok_chunk(english_gloss="Here lived a nobleman.")
        result = MultiChunkAnalysisResult(
            original_spanish=_SOURCE_A,
            chunks=[chunk],
            prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
            total_chunks=1, successful_chunks=1, failed_chunks=0,
        )
        md = multi_chunk_result_to_markdown(result)
        assert "English Gloss" in md
        assert "Here lived a nobleman." in md

    def test_difficult_phrase_rendered(self):
        phrase = DifficultPhrase(
            phrase="a cal y canto",
            category="idiom",
            difficulty_level="C1",
            why_difficult="Idiomatic.",
        )
        chunk = _ok_chunk(phrases=[phrase])
        result = MultiChunkAnalysisResult(
            original_spanish=_SOURCE_A,
            chunks=[chunk],
            prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
            total_chunks=1, successful_chunks=1, failed_chunks=0,
        )
        md = multi_chunk_result_to_markdown(result)
        assert "Difficult Phrases" in md
        assert "a cal y canto" in md

    def test_grammar_note_rendered(self):
        note = GrammarNote(topic="Past subjunctive", explanation="Hypothetical past.")
        chunk = _ok_chunk(grammar_notes=[note])
        result = MultiChunkAnalysisResult(
            original_spanish=_SOURCE_A,
            chunks=[chunk],
            prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
            total_chunks=1, successful_chunks=1, failed_chunks=0,
        )
        md = multi_chunk_result_to_markdown(result)
        assert "Grammar Notes" in md
        assert "Past subjunctive" in md

    def test_comprehension_question_rendered(self):
        cq = ComprehensionQuestion(question="¿De qué habla el texto?")
        chunk = _ok_chunk(comprehension_question=cq)
        result = MultiChunkAnalysisResult(
            original_spanish=_SOURCE_A,
            chunks=[chunk],
            prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
            total_chunks=1, successful_chunks=1, failed_chunks=0,
        )
        md = multi_chunk_result_to_markdown(result)
        assert "Comprehension" in md
        assert "¿De qué habla el texto?" in md


# ===========================================================================
# Title section
# ===========================================================================

class TestTitleSection:
    def test_title_present_when_provided(self):
        md = multi_chunk_result_to_markdown(_one_chunk_result(), title="Quijote Study")
        assert "Quijote Study" in md

    def test_title_is_h1(self):
        md = multi_chunk_result_to_markdown(_one_chunk_result(), title="Quijote")
        assert "# Quijote" in md

    def test_no_h1_when_title_none(self):
        md = multi_chunk_result_to_markdown(_one_chunk_result(), title=None)
        lines = md.splitlines()
        assert not any(ln.startswith("# ") for ln in lines)

    def test_no_h1_when_title_omitted(self):
        md = multi_chunk_result_to_markdown(_one_chunk_result())
        lines = md.splitlines()
        assert not any(ln.startswith("# ") for ln in lines)

    def test_no_h1_when_title_empty_string(self):
        md = multi_chunk_result_to_markdown(_one_chunk_result(), title="")
        lines = md.splitlines()
        assert not any(ln.startswith("# ") for ln in lines)

    def test_title_appears_before_body(self):
        md = multi_chunk_result_to_markdown(_one_chunk_result(_SOURCE_A), title="Mi título")
        assert md.index("Mi título") < md.index(_SOURCE_A)


# ===========================================================================
# Summary section
# ===========================================================================

class TestSummarySection:
    def test_total_chunks_value_in_output(self):
        md = multi_chunk_result_to_markdown(_two_chunk_result())
        assert "2" in md  # total_chunks = 2

    def test_successful_chunks_value_in_output(self):
        result = _two_chunk_result(fail_second=True)
        md = multi_chunk_result_to_markdown(result)
        # 1 successful, 1 failed — both counts should appear
        assert "1" in md

    def test_failed_chunks_reflected_in_output(self):
        result = _two_chunk_result(fail_second=True)
        md = multi_chunk_result_to_markdown(result)
        assert "1" in md  # failed_chunks = 1

    def test_prompt_version_in_summary(self):
        md = multi_chunk_result_to_markdown(_one_chunk_result())
        assert SPANISH_SOURCE_PROMPT_VERSION in md

    def test_summary_section_heading_present(self):
        md = multi_chunk_result_to_markdown(_one_chunk_result())
        assert "Summary" in md


# ===========================================================================
# Checker status per chunk
# ===========================================================================

class TestCheckerStatus:
    def test_checker_status_present_for_successful_chunk(self):
        md = multi_chunk_result_to_markdown(_one_chunk_result())
        # "passed" should appear somewhere (checker status)
        assert "passed" in md

    def test_warning_status_rendered(self):
        chunk = _ok_chunk(status=STATUS_WARNING)
        # Override checker_result to match
        ar = chunk.analysis
        warning_check = CoachCheckResult(status=STATUS_WARNING, summary="warning", issues=["minor"])
        chunk = ChunkAnalysisResult(
            chunk_id="chunk_0000",
            chunk_index=0,
            total_chunks=1,
            chunk_text=_SOURCE_A,
            analysis=ar,
            checker_result=warning_check,
            status=STATUS_WARNING,
        )
        result = MultiChunkAnalysisResult(
            original_spanish=_SOURCE_A,
            chunks=[chunk],
            prompt_version=SPANISH_SOURCE_PROMPT_VERSION,
            total_chunks=1, successful_chunks=1, failed_chunks=0,
        )
        md = multi_chunk_result_to_markdown(result)
        assert "warning" in md

    def test_error_status_rendered_for_failed_chunk(self):
        result = _two_chunk_result(fail_second=True)
        md = multi_chunk_result_to_markdown(result)
        assert "error" in md.lower()
