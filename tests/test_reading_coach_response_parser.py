"""Tests for reading_coach/response_parser.py.

No Ollama, no network, no Streamlit required.

Covers:
  - Clean JSON parses correctly.
  - JSON wrapped in ```json ... ``` fences parses correctly.
  - JSON wrapped in plain ``` fences parses correctly.
  - Prose before / after JSON is ignored; the JSON object is extracted.
  - Empty / whitespace-only input raises ReadingCoachParseError.
  - Completely invalid (non-JSON) input raises ReadingCoachParseError.
  - JSON that has no recognisable {} raises ReadingCoachParseError.
  - Schema validation failure (missing required field) raises ReadingCoachParseError.
  - Invalid CEFR level is coerced to the schema default (schema behaviour, not parse error).
  - original_spanish is preserved byte-for-byte from the JSON.
  - parse_reading_coach_response is importable from the module.
"""
from __future__ import annotations

import json

import pytest

from reading_coach.response_parser import ReadingCoachParseError, parse_reading_coach_response
from reading_coach.schemas import DEFAULT_COACH_LEVEL, ReadingCoachResult


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

SOURCE = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme."

_VALID_PAYLOAD: dict = {
    "original_spanish": SOURCE,
    "overall_level": "B2",
    "modern_spanish": "En un lugar de La Mancha, cuyo nombre prefiero no recordar.",
    "english_gloss": "In a place in La Mancha, whose name I do not wish to recall.",
    "difficult_phrases": [],
    "grammar_notes": [],
    "comprehension_question": None,
}

VALID_JSON = json.dumps(_VALID_PAYLOAD, ensure_ascii=False)


# ---------------------------------------------------------------------------
# TestPublicInterface
# ---------------------------------------------------------------------------

class TestPublicInterface:
    """Module exports and return type."""

    def test_function_is_importable(self):
        from reading_coach.response_parser import parse_reading_coach_response  # noqa: F401

    def test_error_class_is_importable(self):
        from reading_coach.response_parser import ReadingCoachParseError  # noqa: F401

    def test_error_is_exception_subclass(self):
        assert issubclass(ReadingCoachParseError, Exception)

    def test_returns_reading_coach_result(self):
        result = parse_reading_coach_response(VALID_JSON)
        assert isinstance(result, ReadingCoachResult)


# ---------------------------------------------------------------------------
# TestCleanJSON
# ---------------------------------------------------------------------------

class TestCleanJSON:
    """Scenario 1: clean, direct JSON input."""

    def test_clean_json_parses(self):
        result = parse_reading_coach_response(VALID_JSON)
        assert result.original_spanish == SOURCE

    def test_overall_level_preserved(self):
        result = parse_reading_coach_response(VALID_JSON)
        assert result.overall_level == "B2"

    def test_modern_spanish_preserved(self):
        result = parse_reading_coach_response(VALID_JSON)
        assert result.modern_spanish == _VALID_PAYLOAD["modern_spanish"]

    def test_english_gloss_preserved(self):
        result = parse_reading_coach_response(VALID_JSON)
        assert result.english_gloss == _VALID_PAYLOAD["english_gloss"]

    def test_empty_lists_preserved(self):
        result = parse_reading_coach_response(VALID_JSON)
        assert result.difficult_phrases == []
        assert result.grammar_notes == []

    def test_leading_trailing_whitespace_ignored(self):
        result = parse_reading_coach_response("  \n" + VALID_JSON + "\n  ")
        assert result.original_spanish == SOURCE


# ---------------------------------------------------------------------------
# TestFencedJSON
# ---------------------------------------------------------------------------

class TestFencedJSON:
    """Scenario 2: JSON wrapped inside markdown code fences."""

    def test_json_fence_with_language_tag(self):
        fenced = f"```json\n{VALID_JSON}\n```"
        result = parse_reading_coach_response(fenced)
        assert result.original_spanish == SOURCE

    def test_json_fence_without_language_tag(self):
        fenced = f"```\n{VALID_JSON}\n```"
        result = parse_reading_coach_response(fenced)
        assert result.original_spanish == SOURCE

    def test_json_fence_with_extra_whitespace(self):
        fenced = f"```json  \n  {VALID_JSON}  \n```  "
        result = parse_reading_coach_response(fenced)
        assert result.overall_level == "B2"

    def test_fence_only_around_json_not_prose(self):
        # Fence contains only the JSON block — no surrounding prose.
        fenced = "```json\n" + VALID_JSON + "\n```"
        result = parse_reading_coach_response(fenced)
        assert isinstance(result, ReadingCoachResult)


# ---------------------------------------------------------------------------
# TestProseWrapping
# ---------------------------------------------------------------------------

class TestProseWrapping:
    """Scenario 3: prose / commentary before or after the JSON object."""

    def test_prose_before_json(self):
        raw = "Here is the analysis:\n\n" + VALID_JSON
        result = parse_reading_coach_response(raw)
        assert result.original_spanish == SOURCE

    def test_prose_after_json(self):
        raw = VALID_JSON + "\n\nI hope this helps!"
        result = parse_reading_coach_response(raw)
        assert result.original_spanish == SOURCE

    def test_prose_before_and_after_json(self):
        raw = "Sure, here you go:\n" + VALID_JSON + "\nLet me know if you need more."
        result = parse_reading_coach_response(raw)
        assert result.original_spanish == SOURCE

    def test_thinking_preamble_before_json(self):
        # Simulates a <think>...</think> preamble that some reasoning models emit.
        raw = "<think>Let me analyse this carefully.</think>\n\n" + VALID_JSON
        result = parse_reading_coach_response(raw)
        assert result.original_spanish == SOURCE

    def test_multiple_newlines_before_json(self):
        raw = "\n\n\n" + VALID_JSON + "\n\n\n"
        result = parse_reading_coach_response(raw)
        assert result.overall_level == "B2"


# ---------------------------------------------------------------------------
# TestInvalidInput
# ---------------------------------------------------------------------------

class TestInvalidInput:
    """Scenario 4: unrecoverable input raises ReadingCoachParseError."""

    def test_empty_string_raises(self):
        with pytest.raises(ReadingCoachParseError):
            parse_reading_coach_response("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ReadingCoachParseError):
            parse_reading_coach_response("   \n\t  ")

    def test_plain_text_raises(self):
        with pytest.raises(ReadingCoachParseError):
            parse_reading_coach_response("This is not JSON at all.")

    def test_truncated_json_raises(self):
        truncated = VALID_JSON[:40]  # cut off mid-object
        with pytest.raises(ReadingCoachParseError):
            parse_reading_coach_response(truncated)

    def test_array_root_raises(self):
        # JSON array instead of object — valid JSON but wrong shape.
        with pytest.raises(ReadingCoachParseError):
            parse_reading_coach_response("[1, 2, 3]")

    def test_no_curly_brace_json_raises(self):
        # A string token — valid JSON scalar, but no {} to extract.
        with pytest.raises(ReadingCoachParseError):
            parse_reading_coach_response('"just a string"')

    def test_error_message_is_non_empty(self):
        with pytest.raises(ReadingCoachParseError) as exc_info:
            parse_reading_coach_response("not json")
        assert str(exc_info.value)


# ---------------------------------------------------------------------------
# TestSchemaValidation
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    """Scenario 5 & 6: schema-level failures are surfaced as ReadingCoachParseError."""

    def test_missing_original_spanish_raises(self):
        payload = dict(_VALID_PAYLOAD)
        del payload["original_spanish"]
        with pytest.raises(ReadingCoachParseError):
            parse_reading_coach_response(json.dumps(payload))

    def test_missing_overall_level_raises(self):
        payload = dict(_VALID_PAYLOAD)
        del payload["overall_level"]
        with pytest.raises(ReadingCoachParseError):
            parse_reading_coach_response(json.dumps(payload))

    def test_wrong_type_for_original_spanish_raises(self):
        payload = dict(_VALID_PAYLOAD)
        payload["original_spanish"] = 42  # must be str
        with pytest.raises(ReadingCoachParseError):
            parse_reading_coach_response(json.dumps(payload))

    def test_invalid_cefr_level_coerced_not_error(self):
        # The schema coerces unrecognised levels to DEFAULT_COACH_LEVEL —
        # this is schema behaviour, not a parse error.
        payload = dict(_VALID_PAYLOAD)
        payload["overall_level"] = "ZZZZ"
        result = parse_reading_coach_response(json.dumps(payload))
        assert result.overall_level == DEFAULT_COACH_LEVEL

    def test_difficult_phrases_invalid_item_raises(self):
        payload = dict(_VALID_PAYLOAD)
        # phrase is a required field inside DifficultPhrase.
        payload["difficult_phrases"] = [{"category": "vocab"}]  # missing phrase etc.
        with pytest.raises(ReadingCoachParseError):
            parse_reading_coach_response(json.dumps(payload))


# ---------------------------------------------------------------------------
# TestOriginalSpanishPreservation
# ---------------------------------------------------------------------------

class TestOriginalSpanishPreservation:
    """Scenario 5: original_spanish is returned byte-for-byte from the JSON."""

    def test_exact_unicode_preservation(self):
        spanish = "¡Hola! ¿Cómo estás? Señorita."
        payload = dict(_VALID_PAYLOAD)
        payload["original_spanish"] = spanish
        result = parse_reading_coach_response(json.dumps(payload, ensure_ascii=False))
        assert result.original_spanish == spanish

    def test_newlines_preserved(self):
        spanish = "Primera línea.\nSegunda línea."
        payload = dict(_VALID_PAYLOAD)
        payload["original_spanish"] = spanish
        result = parse_reading_coach_response(json.dumps(payload, ensure_ascii=False))
        assert result.original_spanish == spanish

    def test_original_spanish_not_stripped(self):
        # Leading/trailing spaces in the field value should survive JSON round-trip.
        spanish = "  Texto con espacios  "
        payload = dict(_VALID_PAYLOAD)
        payload["original_spanish"] = spanish
        result = parse_reading_coach_response(json.dumps(payload))
        assert result.original_spanish == spanish


# ---------------------------------------------------------------------------
# TestWithRichPayload
# ---------------------------------------------------------------------------

class TestWithRichPayload:
    """Parser handles nested structures (DifficultPhrase, GrammarNote, etc.)."""

    _RICH: dict = {
        "original_spanish": SOURCE,
        "overall_level": "C1",
        "modern_spanish": "Paráfrasis moderna.",
        "english_gloss": "Modern paraphrase.",
        "difficult_phrases": [
            {
                "phrase": "de cuyo nombre",
                "category": "archaic syntax",
                "difficulty_level": "C1",
                "why_difficult": "Relative clause with cuyo.",
                "modern_spanish_equivalent": "cuyo nombre",
                "english_meaning": "whose name",
                "grammar_note": "Cuyo agrees with the noun it modifies.",
                "learner_tip": "Remember cuyo as a possessive relative.",
            }
        ],
        "grammar_notes": [
            {
                "topic": "Subjunctive mood",
                "explanation": "Used after querer + que.",
                "example_from_text": "no quiero acordarme",
            }
        ],
        "comprehension_question": {
            "question": "¿Por qué el narrador no quiere recordar el nombre del lugar?",
            "answer_hint": "Es una decisión narrativa para mantener el misterio.",
        },
    }

    def test_rich_payload_parses(self):
        result = parse_reading_coach_response(json.dumps(self._RICH, ensure_ascii=False))
        assert isinstance(result, ReadingCoachResult)

    def test_difficult_phrase_count(self):
        result = parse_reading_coach_response(json.dumps(self._RICH, ensure_ascii=False))
        assert len(result.difficult_phrases) == 1

    def test_difficult_phrase_text(self):
        result = parse_reading_coach_response(json.dumps(self._RICH, ensure_ascii=False))
        assert result.difficult_phrases[0].phrase == "de cuyo nombre"

    def test_grammar_note_count(self):
        result = parse_reading_coach_response(json.dumps(self._RICH, ensure_ascii=False))
        assert len(result.grammar_notes) == 1

    def test_comprehension_question_present(self):
        result = parse_reading_coach_response(json.dumps(self._RICH, ensure_ascii=False))
        assert result.comprehension_question is not None

    def test_comprehension_question_text(self):
        result = parse_reading_coach_response(json.dumps(self._RICH, ensure_ascii=False))
        assert "narrador" in result.comprehension_question.question

    def test_rich_fenced_parses(self):
        fenced = "```json\n" + json.dumps(self._RICH, ensure_ascii=False) + "\n```"
        result = parse_reading_coach_response(fenced)
        assert result.overall_level == "C1"
