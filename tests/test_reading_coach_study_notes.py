"""Tests for reading_coach/study_notes.py — pure Markdown export, no I/O.

Tests are written first (TDD red phase) and cover all acceptance criteria:
  - Output always a str.
  - Title heading present when provided; absent when not.
  - original_spanish appears verbatim.
  - Optional sections (modern_spanish, english_gloss, comprehension_question)
    are included only when the corresponding field is populated.
  - difficult_phrases section rendered with all sub-fields.
  - Grammar notes rendered with topic, explanation, and optional example.
  - Zero difficult phrases → phrase section cleanly absent.
  - Function is pure (deterministic, no side effects).
"""
from __future__ import annotations

import pytest

from reading_coach.study_notes import reading_coach_result_to_markdown
from reading_coach.schemas import (
    ComprehensionQuestion,
    DifficultPhrase,
    GrammarNote,
    ReadingCoachResult,
)


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

def _minimal_result(source: str = "Hola mundo.") -> ReadingCoachResult:
    """Smallest valid result — no optional fields populated."""
    return ReadingCoachResult(
        original_spanish=source,
        overall_level="B1",
    )


def _full_phrase() -> DifficultPhrase:
    return DifficultPhrase(
        phrase="a cal y canto",
        category="idiom",
        difficulty_level="C1",
        why_difficult="Idiomatic expression meaning 'firmly shut'.",
        modern_spanish_equivalent="completamente cerrado",
        english_meaning="firmly shut / sealed tight",
        grammar_note="Adverbial phrase derived from masonry vocabulary.",
        learner_tip="Memorise as a fixed chunk; do not translate literally.",
    )


def _sparse_phrase() -> DifficultPhrase:
    """Phrase with only the required fields."""
    return DifficultPhrase(
        phrase="fuese",
        category="verb form",
        difficulty_level="B2",
        why_difficult="Archaic past subjunctive of 'ser'.",
    )


def _grammar_note_with_example() -> GrammarNote:
    return GrammarNote(
        topic="Past subjunctive",
        explanation="Used to express hypothetical or contrary-to-fact past conditions.",
        example_from_text="mientras la honra no fuese satisfecha",
    )


def _grammar_note_no_example() -> GrammarNote:
    return GrammarNote(
        topic="Enclitic pronouns",
        explanation="Object pronouns appended directly to the verb in older Spanish.",
    )


def _comprehension_question_with_hint() -> ComprehensionQuestion:
    return ComprehensionQuestion(
        question="¿Por qué no habría paz entre los personajes?",
        answer_hint="La honra no había sido satisfecha.",
    )


def _comprehension_question_no_hint() -> ComprehensionQuestion:
    return ComprehensionQuestion(
        question="¿Qué hace el niño en el parque?",
    )


# ---------------------------------------------------------------------------
# TestReturnType
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_str(self):
        assert isinstance(reading_coach_result_to_markdown(_minimal_result()), str)

    def test_not_none(self):
        assert reading_coach_result_to_markdown(_minimal_result()) is not None

    def test_non_empty_for_minimal_result(self):
        assert reading_coach_result_to_markdown(_minimal_result()).strip()

    def test_returns_str_with_all_fields_populated(self):
        result = ReadingCoachResult(
            original_spanish="El caballero caminaba.",
            overall_level="B2",
            modern_spanish="El caballero andaba.",
            english_gloss="The knight was walking.",
            difficult_phrases=[_full_phrase()],
            grammar_notes=[_grammar_note_with_example()],
            comprehension_question=_comprehension_question_with_hint(),
        )
        md = reading_coach_result_to_markdown(result)
        assert isinstance(md, str)


# ---------------------------------------------------------------------------
# TestTitleSection
# ---------------------------------------------------------------------------

class TestTitleSection:
    def test_title_present_when_provided(self):
        md = reading_coach_result_to_markdown(_minimal_result(), title="My Study Notes")
        assert "My Study Notes" in md

    def test_title_is_h1_heading(self):
        md = reading_coach_result_to_markdown(_minimal_result(), title="Study Notes")
        assert "# Study Notes" in md

    def test_no_h1_heading_when_title_is_none(self):
        md = reading_coach_result_to_markdown(_minimal_result(), title=None)
        assert not md.startswith("# ")

    def test_no_h1_heading_when_title_omitted(self):
        md = reading_coach_result_to_markdown(_minimal_result())
        lines = md.splitlines()
        h1_lines = [ln for ln in lines if ln.startswith("# ")]
        assert h1_lines == []

    def test_title_appears_before_original_spanish(self):
        md = reading_coach_result_to_markdown(
            _minimal_result("Hola."), title="Notas"
        )
        title_pos = md.index("Notas")
        source_pos = md.index("Hola.")
        assert title_pos < source_pos

    def test_empty_string_title_treated_as_no_title(self):
        md = reading_coach_result_to_markdown(_minimal_result(), title="")
        lines = md.splitlines()
        h1_lines = [ln for ln in lines if ln.startswith("# ")]
        assert h1_lines == []


# ---------------------------------------------------------------------------
# TestOriginalSpanish
# ---------------------------------------------------------------------------

class TestOriginalSpanish:
    def test_original_spanish_present_verbatim(self):
        source = "En un lugar de la Mancha."
        md = reading_coach_result_to_markdown(_minimal_result(source))
        assert source in md

    def test_original_spanish_with_special_chars(self):
        source = "¿Qué decís, señor? ¡Silencio!"
        md = reading_coach_result_to_markdown(_minimal_result(source))
        assert source in md

    def test_original_spanish_with_em_dash(self):
        source = "—¿Adónde vais? —preguntó la anciana."
        md = reading_coach_result_to_markdown(_minimal_result(source))
        assert source in md

    def test_original_spanish_section_heading_present(self):
        md = reading_coach_result_to_markdown(_minimal_result())
        assert "Original Spanish" in md

    def test_overall_level_present(self):
        result = _minimal_result()
        md = reading_coach_result_to_markdown(result)
        assert result.overall_level in md


# ---------------------------------------------------------------------------
# TestOptionalSectionsOmitted
# ---------------------------------------------------------------------------

class TestOptionalSectionsOmitted:
    """Optional sections must not appear when the corresponding field is None."""

    def test_no_modern_spanish_section_when_none(self):
        md = reading_coach_result_to_markdown(_minimal_result())
        assert "Modern Spanish" not in md

    def test_no_english_gloss_section_when_none(self):
        md = reading_coach_result_to_markdown(_minimal_result())
        assert "English Gloss" not in md

    def test_no_comprehension_question_when_none(self):
        md = reading_coach_result_to_markdown(_minimal_result())
        assert "Comprehension" not in md

    def test_no_difficult_phrases_section_when_empty_list(self):
        md = reading_coach_result_to_markdown(_minimal_result())
        assert "Difficult Phrases" not in md

    def test_no_grammar_notes_section_when_empty_list(self):
        md = reading_coach_result_to_markdown(_minimal_result())
        assert "Grammar Notes" not in md


# ---------------------------------------------------------------------------
# TestOptionalSectionsIncluded
# ---------------------------------------------------------------------------

class TestOptionalSectionsIncluded:
    """Optional sections must appear when the corresponding field is populated."""

    def test_modern_spanish_section_present(self):
        result = ReadingCoachResult(
            original_spanish="Hola.",
            overall_level="A1",
            modern_spanish="Hola, ¿cómo estás?",
        )
        md = reading_coach_result_to_markdown(result)
        assert "Modern Spanish" in md

    def test_modern_spanish_content_present(self):
        result = ReadingCoachResult(
            original_spanish="Hola.",
            overall_level="A1",
            modern_spanish="Hola, ¿cómo estás?",
        )
        md = reading_coach_result_to_markdown(result)
        assert "Hola, ¿cómo estás?" in md

    def test_english_gloss_section_present(self):
        result = ReadingCoachResult(
            original_spanish="Hola.",
            overall_level="A1",
            english_gloss="Hello.",
        )
        md = reading_coach_result_to_markdown(result)
        assert "English Gloss" in md

    def test_english_gloss_content_present(self):
        result = ReadingCoachResult(
            original_spanish="Hola.",
            overall_level="A1",
            english_gloss="Hello.",
        )
        md = reading_coach_result_to_markdown(result)
        assert "Hello." in md

    def test_comprehension_question_section_present(self):
        result = ReadingCoachResult(
            original_spanish="Hola.",
            overall_level="A1",
            comprehension_question=_comprehension_question_with_hint(),
        )
        md = reading_coach_result_to_markdown(result)
        assert "Comprehension" in md

    def test_comprehension_question_text_present(self):
        q = _comprehension_question_with_hint()
        result = ReadingCoachResult(
            original_spanish="Hola.",
            overall_level="A1",
            comprehension_question=q,
        )
        md = reading_coach_result_to_markdown(result)
        assert q.question in md


# ---------------------------------------------------------------------------
# TestDifficultPhrases
# ---------------------------------------------------------------------------

class TestDifficultPhrases:
    def _result_with_full_phrase(self) -> ReadingCoachResult:
        return ReadingCoachResult(
            original_spanish="Hallaron las puertas cerradas a cal y canto.",
            overall_level="C1",
            difficult_phrases=[_full_phrase()],
        )

    def test_difficult_phrases_section_heading(self):
        md = reading_coach_result_to_markdown(self._result_with_full_phrase())
        assert "Difficult Phrases" in md

    def test_phrase_text_present(self):
        md = reading_coach_result_to_markdown(self._result_with_full_phrase())
        assert "a cal y canto" in md

    def test_phrase_category_present(self):
        md = reading_coach_result_to_markdown(self._result_with_full_phrase())
        assert "idiom" in md

    def test_phrase_difficulty_level_present(self):
        md = reading_coach_result_to_markdown(self._result_with_full_phrase())
        assert "C1" in md

    def test_phrase_why_difficult_present(self):
        md = reading_coach_result_to_markdown(self._result_with_full_phrase())
        assert "Idiomatic expression meaning" in md

    def test_phrase_modern_equivalent_present(self):
        md = reading_coach_result_to_markdown(self._result_with_full_phrase())
        assert "completamente cerrado" in md

    def test_phrase_english_meaning_present(self):
        md = reading_coach_result_to_markdown(self._result_with_full_phrase())
        assert "firmly shut" in md

    def test_phrase_grammar_note_present(self):
        md = reading_coach_result_to_markdown(self._result_with_full_phrase())
        assert "masonry" in md

    def test_phrase_learner_tip_present(self):
        md = reading_coach_result_to_markdown(self._result_with_full_phrase())
        assert "fixed chunk" in md

    def test_sparse_phrase_renders_without_error(self):
        result = ReadingCoachResult(
            original_spanish="Díjole que fuese.",
            overall_level="B2",
            difficult_phrases=[_sparse_phrase()],
        )
        md = reading_coach_result_to_markdown(result)
        assert "fuese" in md

    def test_sparse_phrase_omits_none_sub_fields(self):
        """Optional sub-fields that are None must not produce stray 'None' text."""
        result = ReadingCoachResult(
            original_spanish="Díjole que fuese.",
            overall_level="B2",
            difficult_phrases=[_sparse_phrase()],
        )
        md = reading_coach_result_to_markdown(result)
        assert "None" not in md

    def test_multiple_phrases_all_present(self):
        result = ReadingCoachResult(
            original_spanish="Díjole que fuese. Las puertas cerradas a cal y canto.",
            overall_level="C1",
            difficult_phrases=[_sparse_phrase(), _full_phrase()],
        )
        md = reading_coach_result_to_markdown(result)
        assert "fuese" in md
        assert "a cal y canto" in md

    def test_zero_phrases_no_difficult_phrases_section(self):
        md = reading_coach_result_to_markdown(_minimal_result())
        assert "Difficult Phrases" not in md

    def test_zero_phrases_no_none_in_output(self):
        md = reading_coach_result_to_markdown(_minimal_result())
        assert "None" not in md


# ---------------------------------------------------------------------------
# TestGrammarNotes
# ---------------------------------------------------------------------------

class TestGrammarNotes:
    def _result_with_note(self, note: GrammarNote) -> ReadingCoachResult:
        return ReadingCoachResult(
            original_spanish="Hola.",
            overall_level="B2",
            grammar_notes=[note],
        )

    def test_grammar_notes_section_heading(self):
        md = reading_coach_result_to_markdown(
            self._result_with_note(_grammar_note_with_example())
        )
        assert "Grammar Notes" in md

    def test_grammar_note_topic_present(self):
        md = reading_coach_result_to_markdown(
            self._result_with_note(_grammar_note_with_example())
        )
        assert "Past subjunctive" in md

    def test_grammar_note_explanation_present(self):
        md = reading_coach_result_to_markdown(
            self._result_with_note(_grammar_note_with_example())
        )
        assert "hypothetical" in md

    def test_grammar_note_example_present_when_provided(self):
        md = reading_coach_result_to_markdown(
            self._result_with_note(_grammar_note_with_example())
        )
        assert "mientras la honra no fuese satisfecha" in md

    def test_grammar_note_no_example_renders_cleanly(self):
        md = reading_coach_result_to_markdown(
            self._result_with_note(_grammar_note_no_example())
        )
        assert "Enclitic pronouns" in md
        assert "None" not in md

    def test_multiple_grammar_notes_all_present(self):
        result = ReadingCoachResult(
            original_spanish="Hola.",
            overall_level="B2",
            grammar_notes=[_grammar_note_with_example(), _grammar_note_no_example()],
        )
        md = reading_coach_result_to_markdown(result)
        assert "Past subjunctive" in md
        assert "Enclitic pronouns" in md


# ---------------------------------------------------------------------------
# TestComprehensionQuestion
# ---------------------------------------------------------------------------

class TestComprehensionQuestion:
    def test_question_text_present(self):
        result = ReadingCoachResult(
            original_spanish="Hola.",
            overall_level="A1",
            comprehension_question=_comprehension_question_with_hint(),
        )
        md = reading_coach_result_to_markdown(result)
        assert "¿Por qué no habría paz entre los personajes?" in md

    def test_answer_hint_present_when_provided(self):
        result = ReadingCoachResult(
            original_spanish="Hola.",
            overall_level="A1",
            comprehension_question=_comprehension_question_with_hint(),
        )
        md = reading_coach_result_to_markdown(result)
        assert "La honra no había sido satisfecha." in md

    def test_no_hint_renders_cleanly(self):
        result = ReadingCoachResult(
            original_spanish="Hola.",
            overall_level="A1",
            comprehension_question=_comprehension_question_no_hint(),
        )
        md = reading_coach_result_to_markdown(result)
        assert "¿Qué hace el niño" in md
        assert "None" not in md


# ---------------------------------------------------------------------------
# TestPurity
# ---------------------------------------------------------------------------

class TestPurity:
    def test_same_result_same_output(self):
        result = ReadingCoachResult(
            original_spanish="Hola.",
            overall_level="A1",
            modern_spanish="¡Hola!",
            difficult_phrases=[_full_phrase()],
        )
        assert reading_coach_result_to_markdown(result) == reading_coach_result_to_markdown(result)

    def test_does_not_mutate_result(self):
        result = ReadingCoachResult(
            original_spanish="Hola.",
            overall_level="A1",
            difficult_phrases=[_full_phrase()],
        )
        original_phrases_count = len(result.difficult_phrases)
        reading_coach_result_to_markdown(result)
        assert len(result.difficult_phrases) == original_phrases_count

    def test_title_does_not_bleed_into_second_call(self):
        result = _minimal_result()
        md_with = reading_coach_result_to_markdown(result, title="Title A")
        md_without = reading_coach_result_to_markdown(result)
        assert "Title A" not in md_without

    def test_no_streamlit_import_needed(self):
        """Importing study_notes must not require streamlit to be installed."""
        import importlib
        import sys
        # Verify the module is already importable (it is, or we'd have failed above)
        mod = sys.modules.get("reading_coach.study_notes")
        assert mod is not None

    def test_no_ollama_import_needed(self):
        """study_notes must not import infrastructure.ollama_client."""
        import sys
        mod = sys.modules.get("reading_coach.study_notes")
        assert mod is not None
        source = getattr(mod, "__file__", "") or ""
        # The module itself must not have ollama as a direct dependency
        if source:
            with open(source, encoding="utf-8") as fh:
                text = fh.read()
            assert "ollama" not in text
