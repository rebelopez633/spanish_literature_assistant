"""
Tests: ReadingCoachResult and related schemas.

Pure Pydantic — no Streamlit, no Ollama, no external services.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from reading_coach.schemas import (
    ComprehensionQuestion,
    DifficultPhrase,
    GrammarNote,
    ReadingCoachResult,
    VALID_COACH_LEVELS,
    DEFAULT_COACH_LEVEL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_result(**overrides) -> dict:
    """Minimal valid dict for ReadingCoachResult construction."""
    base = {
        "original_spanish": "Que no me ciegue el amor.",
        "overall_level": "B2",
    }
    base.update(overrides)
    return base


def _minimal_phrase(**overrides) -> dict:
    base = {
        "phrase": "non plus ultra",
        "category": "archaic_expression",
        "difficulty_level": "C1",
        "why_difficult": "Latin loanphrase rarely used in modern Spanish",
    }
    base.update(overrides)
    return base


# ===========================================================================
# VALID_COACH_LEVELS constant
# ===========================================================================

class TestValidCoachLevels:
    def test_contains_standard_cefr(self):
        for level in ("A1", "A2", "B1", "B2", "C1", "C2"):
            assert level in VALID_COACH_LEVELS

    def test_all_uppercase(self):
        for level in VALID_COACH_LEVELS:
            assert level == level.upper()

    def test_is_non_empty(self):
        assert len(VALID_COACH_LEVELS) > 0

    def test_default_level_is_in_valid_levels(self):
        assert DEFAULT_COACH_LEVEL in VALID_COACH_LEVELS


# ===========================================================================
# DifficultPhrase
# ===========================================================================

class TestDifficultPhrase:

    # --- required fields ---

    def test_minimal_construction(self):
        p = DifficultPhrase(**_minimal_phrase())
        assert p.phrase == "non plus ultra"
        assert p.category == "archaic_expression"
        assert p.difficulty_level == "C1"
        assert p.why_difficult == "Latin loanphrase rarely used in modern Spanish"

    def test_missing_phrase_raises(self):
        with pytest.raises(ValidationError):
            DifficultPhrase(**{k: v for k, v in _minimal_phrase().items() if k != "phrase"})

    def test_missing_category_raises(self):
        with pytest.raises(ValidationError):
            DifficultPhrase(**{k: v for k, v in _minimal_phrase().items() if k != "category"})

    def test_missing_difficulty_level_raises(self):
        with pytest.raises(ValidationError):
            DifficultPhrase(**{k: v for k, v in _minimal_phrase().items() if k != "difficulty_level"})

    def test_missing_why_difficult_raises(self):
        with pytest.raises(ValidationError):
            DifficultPhrase(**{k: v for k, v in _minimal_phrase().items() if k != "why_difficult"})

    # --- optional fields default to None ---

    def test_optional_fields_default_to_none(self):
        p = DifficultPhrase(**_minimal_phrase())
        assert p.modern_spanish_equivalent is None
        assert p.english_meaning is None
        assert p.grammar_note is None
        assert p.learner_tip is None

    def test_optional_fields_accept_values(self):
        p = DifficultPhrase(
            **_minimal_phrase(
                modern_spanish_equivalent="lo mejor",
                english_meaning="the very best",
                grammar_note="Latin origin, treated as noun phrase",
                learner_tip="Mostly used in formal or literary contexts",
            )
        )
        assert p.modern_spanish_equivalent == "lo mejor"
        assert p.english_meaning == "the very best"
        assert p.grammar_note == "Latin origin, treated as noun phrase"
        assert p.learner_tip == "Mostly used in formal or literary contexts"

    # --- difficulty_level normalization ---

    def test_lowercase_level_normalized(self):
        p = DifficultPhrase(**_minimal_phrase(difficulty_level="b1"))
        assert p.difficulty_level == "B1"

    def test_mixed_case_level_normalized(self):
        p = DifficultPhrase(**_minimal_phrase(difficulty_level="C1"))
        assert p.difficulty_level == "C1"

    def test_level_with_prose_suffix_normalized(self):
        p = DifficultPhrase(**_minimal_phrase(difficulty_level="B2 (upper-intermediate)"))
        assert p.difficulty_level == "B2"

    def test_invalid_level_falls_back_to_default(self):
        p = DifficultPhrase(**_minimal_phrase(difficulty_level="impossible_level"))
        assert p.difficulty_level == DEFAULT_COACH_LEVEL

    def test_numeric_level_string_falls_back(self):
        p = DifficultPhrase(**_minimal_phrase(difficulty_level="5"))
        assert p.difficulty_level == DEFAULT_COACH_LEVEL

    # --- JSON roundtrip ---

    def test_json_roundtrip(self):
        original = DifficultPhrase(**_minimal_phrase(english_meaning="the peak"))
        restored = DifficultPhrase.model_validate_json(original.model_dump_json())
        assert restored == original


# ===========================================================================
# GrammarNote
# ===========================================================================

class TestGrammarNote:

    def test_minimal_construction(self):
        g = GrammarNote(topic="Subjuntivo", explanation="Used to express doubt or wishes.")
        assert g.topic == "Subjuntivo"
        assert g.explanation == "Used to express doubt or wishes."
        assert g.example_from_text is None

    def test_with_example(self):
        g = GrammarNote(
            topic="Subjuntivo",
            explanation="Used after 'que' clauses expressing emotion.",
            example_from_text="Que no me ciegue el amor.",
        )
        assert g.example_from_text == "Que no me ciegue el amor."

    def test_missing_topic_raises(self):
        with pytest.raises(ValidationError):
            GrammarNote(explanation="Some explanation.")

    def test_missing_explanation_raises(self):
        with pytest.raises(ValidationError):
            GrammarNote(topic="Subjuntivo")

    def test_json_roundtrip(self):
        g = GrammarNote(topic="Ser vs Estar", explanation="Permanent vs temporary.", example_from_text="Ella está cansada.")
        assert GrammarNote.model_validate_json(g.model_dump_json()) == g


# ===========================================================================
# ComprehensionQuestion
# ===========================================================================

class TestComprehensionQuestion:

    def test_minimal_construction(self):
        q = ComprehensionQuestion(question="¿De qué trata el texto?")
        assert q.question == "¿De qué trata el texto?"
        assert q.answer_hint is None

    def test_with_answer_hint(self):
        q = ComprehensionQuestion(
            question="¿Por qué el poeta teme al amor?",
            answer_hint="El poema sugiere que el amor puede nublar la razón.",
        )
        assert q.answer_hint == "El poema sugiere que el amor puede nublar la razón."

    def test_missing_question_raises(self):
        with pytest.raises(ValidationError):
            ComprehensionQuestion(answer_hint="some hint")

    def test_json_roundtrip(self):
        q = ComprehensionQuestion(question="¿Qué significa esta metáfora?", answer_hint="Hint here.")
        assert ComprehensionQuestion.model_validate_json(q.model_dump_json()) == q


# ===========================================================================
# ReadingCoachResult
# ===========================================================================

class TestReadingCoachResultConstruction:

    def test_minimal_construction(self):
        r = ReadingCoachResult(**_minimal_result())
        assert r.original_spanish == "Que no me ciegue el amor."
        assert r.overall_level == "B2"

    def test_optional_fields_default_correctly(self):
        r = ReadingCoachResult(**_minimal_result())
        assert r.modern_spanish is None
        assert r.english_gloss is None
        assert r.difficult_phrases == []
        assert r.grammar_notes == []
        assert r.comprehension_question is None

    def test_full_construction(self):
        r = ReadingCoachResult(
            original_spanish="No hay mal que por bien no venga.",
            modern_spanish="No hay mal que por bien no venga.",
            english_gloss="Every cloud has a silver lining.",
            overall_level="B1",
            difficult_phrases=[
                DifficultPhrase(
                    phrase="por bien no venga",
                    category="proverb_clause",
                    difficulty_level="B2",
                    why_difficult="Idiomatic; word order reversed from standard.",
                    english_meaning="that doesn't come with some good",
                )
            ],
            grammar_notes=[
                GrammarNote(
                    topic="Relative clause with subjunctive",
                    explanation="'que por bien no venga' uses subjunctive after negation.",
                    example_from_text="No hay mal que por bien no venga.",
                )
            ],
            comprehension_question=ComprehensionQuestion(
                question="¿Qué enseña este refrán?",
                answer_hint="Habla sobre encontrar algo positivo en lo negativo.",
            ),
        )
        assert len(r.difficult_phrases) == 1
        assert len(r.grammar_notes) == 1
        assert r.comprehension_question is not None

    def test_missing_original_spanish_raises(self):
        with pytest.raises(ValidationError):
            ReadingCoachResult(overall_level="B1")

    def test_missing_overall_level_raises(self):
        with pytest.raises(ValidationError):
            ReadingCoachResult(original_spanish="Hola mundo.")


class TestReadingCoachResultOriginalSpanish:
    """original_spanish must be preserved byte-for-byte — no mutation."""

    def test_plain_text_preserved(self):
        text = "Vivir sin filosofía es vivir a ciegas."
        r = ReadingCoachResult(**_minimal_result(original_spanish=text))
        assert r.original_spanish == text

    def test_whitespace_preserved(self):
        text = "  Línea uno.\n\nLínea dos.  "
        r = ReadingCoachResult(**_minimal_result(original_spanish=text))
        assert r.original_spanish == text

    def test_unicode_accents_preserved(self):
        text = "¡Qué difícil es la poesía española del Siglo de Oro!"
        r = ReadingCoachResult(**_minimal_result(original_spanish=text))
        assert r.original_spanish == text

    def test_empty_string_preserved(self):
        r = ReadingCoachResult(**_minimal_result(original_spanish=""))
        assert r.original_spanish == ""


class TestReadingCoachResultLevelNormalization:
    """overall_level follows the same normalize-then-fallback rule."""

    def test_valid_uppercase_accepted(self):
        for level in ("A1", "A2", "B1", "B2", "C1", "C2"):
            r = ReadingCoachResult(**_minimal_result(overall_level=level))
            assert r.overall_level == level

    def test_lowercase_normalized(self):
        r = ReadingCoachResult(**_minimal_result(overall_level="b2"))
        assert r.overall_level == "B2"

    def test_level_with_prose_label_normalized(self):
        r = ReadingCoachResult(**_minimal_result(overall_level="C1 advanced"))
        assert r.overall_level == "C1"

    def test_invalid_level_falls_back_to_default(self):
        r = ReadingCoachResult(**_minimal_result(overall_level="native"))
        assert r.overall_level == DEFAULT_COACH_LEVEL

    def test_empty_level_falls_back_to_default(self):
        r = ReadingCoachResult(**_minimal_result(overall_level=""))
        assert r.overall_level == DEFAULT_COACH_LEVEL

    def test_none_level_falls_back_to_default(self):
        r = ReadingCoachResult(**_minimal_result(overall_level=None))
        assert r.overall_level == DEFAULT_COACH_LEVEL


class TestReadingCoachResultNestedObjects:

    def test_difficult_phrases_from_dicts(self):
        r = ReadingCoachResult(
            **_minimal_result(
                difficult_phrases=[
                    {
                        "phrase": "a ciegas",
                        "category": "adverbial_phrase",
                        "difficulty_level": "B1",
                        "why_difficult": "Figurative meaning differs from literal.",
                    }
                ]
            )
        )
        assert isinstance(r.difficult_phrases[0], DifficultPhrase)
        assert r.difficult_phrases[0].phrase == "a ciegas"

    def test_grammar_notes_from_dicts(self):
        r = ReadingCoachResult(
            **_minimal_result(
                grammar_notes=[
                    {"topic": "Gerund", "explanation": "Used to express ongoing action."}
                ]
            )
        )
        assert isinstance(r.grammar_notes[0], GrammarNote)

    def test_comprehension_question_from_dict(self):
        r = ReadingCoachResult(
            **_minimal_result(
                comprehension_question={
                    "question": "¿Qué significa 'a ciegas'?",
                    "answer_hint": "Sin ver, sin guía.",
                }
            )
        )
        assert isinstance(r.comprehension_question, ComprehensionQuestion)
        assert r.comprehension_question.question == "¿Qué significa 'a ciegas'?"

    def test_empty_difficult_phrases_allowed(self):
        r = ReadingCoachResult(**_minimal_result(difficult_phrases=[]))
        assert r.difficult_phrases == []

    def test_empty_grammar_notes_allowed(self):
        r = ReadingCoachResult(**_minimal_result(grammar_notes=[]))
        assert r.grammar_notes == []

    def test_none_comprehension_question_allowed(self):
        r = ReadingCoachResult(**_minimal_result(comprehension_question=None))
        assert r.comprehension_question is None


class TestReadingCoachResultJsonRoundtrip:

    def test_minimal_roundtrip(self):
        r = ReadingCoachResult(**_minimal_result())
        restored = ReadingCoachResult.model_validate_json(r.model_dump_json())
        assert restored.original_spanish == r.original_spanish
        assert restored.overall_level == r.overall_level

    def test_full_roundtrip(self):
        r = ReadingCoachResult(
            original_spanish="Con la Iglesia hemos topado, Sancho.",
            overall_level="C1",
            english_gloss="We have run up against the Church, Sancho.",
            difficult_phrases=[
                DifficultPhrase(
                    phrase="hemos topado",
                    category="verb_phrase",
                    difficulty_level="B2",
                    why_difficult="'Topar con' is colloquial; archaic first-person plural form.",
                    english_meaning="we have bumped into / run up against",
                )
            ],
            grammar_notes=[
                GrammarNote(
                    topic="Compound perfect",
                    explanation="'Hemos topado' uses the present perfect to describe a recent event.",
                    example_from_text="hemos topado",
                )
            ],
            comprehension_question=ComprehensionQuestion(
                question="¿Con quién tocan en este fragmento?",
                answer_hint="Con la Iglesia.",
            ),
        )
        payload = r.model_dump_json()
        restored = ReadingCoachResult.model_validate_json(payload)
        assert restored == r

    def test_dict_roundtrip_via_model_validate(self):
        r = ReadingCoachResult(**_minimal_result())
        restored = ReadingCoachResult.model_validate(r.model_dump())
        assert restored == r
