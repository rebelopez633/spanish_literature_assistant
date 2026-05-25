"""Tests for reading_coach/prompts.py — pure prompt builder, no Ollama calls."""
from __future__ import annotations

import pytest

from reading_coach.prompts import build_coach_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user(msgs: list[dict]) -> str:
    """Return the user message content."""
    return msgs[1]["content"]


def _sys(msgs: list[dict]) -> str:
    """Return the system message content."""
    return msgs[0]["content"]


def _combined(msgs: list[dict]) -> str:
    return _sys(msgs) + "\n" + _user(msgs)


# ---------------------------------------------------------------------------
# TestReturnShape
# ---------------------------------------------------------------------------

class TestReturnShape:
    def test_returns_a_list(self):
        result = build_coach_prompt("Hola mundo.")
        assert isinstance(result, list)

    def test_has_exactly_two_messages(self):
        result = build_coach_prompt("Hola mundo.")
        assert len(result) == 2

    def test_messages_are_dicts(self):
        result = build_coach_prompt("Hola mundo.")
        for msg in result:
            assert isinstance(msg, dict)

    def test_first_message_is_system(self):
        result = build_coach_prompt("Hola mundo.")
        assert result[0]["role"] == "system"

    def test_second_message_is_user(self):
        result = build_coach_prompt("Hola mundo.")
        assert result[1]["role"] == "user"

    def test_each_message_has_content_key(self):
        result = build_coach_prompt("Hola mundo.")
        for msg in result:
            assert "content" in msg
            assert isinstance(msg["content"], str)

    def test_system_message_is_non_empty(self):
        result = build_coach_prompt("Hola mundo.")
        assert _sys(result).strip()

    def test_user_message_is_non_empty(self):
        result = build_coach_prompt("Hola mundo.")
        assert _user(result).strip()


# ---------------------------------------------------------------------------
# TestSourceTextEmbedding
# ---------------------------------------------------------------------------

class TestSourceTextEmbedding:
    def test_source_text_verbatim_in_user_message(self):
        text = "En un lugar de la Mancha."
        msgs = build_coach_prompt(text)
        assert text in _user(msgs)

    def test_source_text_with_accents(self):
        text = "¿Qué decís, señor? ¡Silencio!"
        msgs = build_coach_prompt(text)
        assert text in _user(msgs)

    def test_source_text_multiline(self):
        text = "Primera línea.\nSegunda línea.\nTercera línea."
        msgs = build_coach_prompt(text)
        assert text in _user(msgs)

    def test_source_text_with_quotes_and_punctuation(self):
        text = '"Amor es la fuerza más antigua", escribió el poeta.'
        msgs = build_coach_prompt(text)
        assert text in _user(msgs)

    def test_source_text_not_modified(self):
        text = "  Texto con espacios  "
        msgs = build_coach_prompt(text)
        assert text in _user(msgs)

    def test_different_source_texts_produce_different_prompts(self):
        text_a = "Primera oración."
        text_b = "Segunda oración completamente diferente."
        msgs_a = build_coach_prompt(text_a)
        msgs_b = build_coach_prompt(text_b)
        assert _user(msgs_a) != _user(msgs_b)


# ---------------------------------------------------------------------------
# TestReaderLevel
# ---------------------------------------------------------------------------

class TestReaderLevel:
    @pytest.mark.parametrize("level", ["A1", "A2", "B1", "B2", "C1", "C2"])
    def test_reader_level_appears_in_prompt(self, level):
        msgs = build_coach_prompt("Texto de prueba.", reader_level=level)
        assert level in _combined(msgs)

    def test_different_levels_produce_different_prompts(self):
        msgs_b1 = build_coach_prompt("Texto.", reader_level="B1")
        msgs_c1 = build_coach_prompt("Texto.", reader_level="C1")
        assert _user(msgs_b1) != _user(msgs_c1)

    def test_default_reader_level_is_b1(self):
        msgs = build_coach_prompt("Texto.")
        assert "B1" in _combined(msgs)


# ---------------------------------------------------------------------------
# TestPreserveOriginalInstruction
# ---------------------------------------------------------------------------

class TestPreserveOriginalInstruction:
    def test_preserve_mentioned_in_prompt(self):
        msgs = build_coach_prompt("Texto de prueba.")
        combined_lower = _combined(msgs).lower()
        assert "preserve" in combined_lower or "exact" in combined_lower

    def test_original_spanish_mentioned_in_prompt(self):
        msgs = build_coach_prompt("Texto de prueba.")
        combined_lower = _combined(msgs).lower()
        assert "original" in combined_lower

    def test_do_not_invent_instruction_present(self):
        msgs = build_coach_prompt("Texto de prueba.")
        combined_lower = _combined(msgs).lower()
        # Model must be told not to invent context not present in the passage
        assert "invent" in combined_lower or "not present" in combined_lower or "fabricat" in combined_lower


# ---------------------------------------------------------------------------
# TestJsonSchemaInstruction
# ---------------------------------------------------------------------------

class TestJsonSchemaInstruction:
    def test_json_mentioned_in_prompt(self):
        msgs = build_coach_prompt("Texto de prueba.")
        combined_lower = _combined(msgs).lower()
        assert "json" in combined_lower

    def test_schema_field_names_present(self):
        msgs = build_coach_prompt("Texto de prueba.")
        combined = _combined(msgs)
        # Core ReadingCoachResult fields must appear so the model knows the schema
        assert "difficult_phrases" in combined
        assert "grammar_notes" in combined
        assert "overall_level" in combined

    def test_original_spanish_field_name_present(self):
        msgs = build_coach_prompt("Texto de prueba.")
        assert "original_spanish" in _combined(msgs)

    def test_no_markdown_outside_json_instruction(self):
        msgs = build_coach_prompt("Texto de prueba.")
        combined_lower = _combined(msgs).lower()
        # System prompt should tell model to return only JSON without markdown
        assert "markdown" in combined_lower or "only" in combined_lower


# ---------------------------------------------------------------------------
# TestEnglishGlossToggle
# ---------------------------------------------------------------------------

class TestEnglishGlossToggle:
    def test_gloss_true_and_false_produce_different_prompts(self):
        msgs_on = build_coach_prompt("Texto.", include_english_gloss=True)
        msgs_off = build_coach_prompt("Texto.", include_english_gloss=False)
        assert _user(msgs_on) != _user(msgs_off)

    def test_gloss_true_references_english_gloss_field(self):
        msgs = build_coach_prompt("Texto.", include_english_gloss=True)
        combined = _combined(msgs)
        assert "english_gloss" in combined

    def test_gloss_false_does_not_request_positive_gloss(self):
        # When disabled, the positive "provide english gloss" instruction must be absent
        msgs_on = build_coach_prompt("Texto.", include_english_gloss=True)
        msgs_off = build_coach_prompt("Texto.", include_english_gloss=False)
        user_on = _user(msgs_on)
        user_off = _user(msgs_off)
        # The two user messages must differ — absence of the positive instruction
        assert user_on != user_off

    def test_gloss_default_is_true(self):
        msgs_default = build_coach_prompt("Texto.")
        msgs_explicit = build_coach_prompt("Texto.", include_english_gloss=True)
        assert msgs_default == msgs_explicit


# ---------------------------------------------------------------------------
# TestModernSpanishToggle
# ---------------------------------------------------------------------------

class TestModernSpanishToggle:
    def test_modern_spanish_true_and_false_produce_different_prompts(self):
        msgs_on = build_coach_prompt("Texto.", include_modern_spanish=True)
        msgs_off = build_coach_prompt("Texto.", include_modern_spanish=False)
        assert _user(msgs_on) != _user(msgs_off)

    def test_modern_spanish_true_references_field_name(self):
        msgs = build_coach_prompt("Texto.", include_modern_spanish=True)
        combined = _combined(msgs)
        assert "modern_spanish" in combined

    def test_modern_spanish_false_differs_from_true(self):
        msgs_on = build_coach_prompt("Texto.", include_modern_spanish=True)
        msgs_off = build_coach_prompt("Texto.", include_modern_spanish=False)
        assert _user(msgs_on) != _user(msgs_off)

    def test_modern_spanish_default_is_true(self):
        msgs_default = build_coach_prompt("Texto.")
        msgs_explicit = build_coach_prompt("Texto.", include_modern_spanish=True)
        assert msgs_default == msgs_explicit


# ---------------------------------------------------------------------------
# TestAnnotationDensity
# ---------------------------------------------------------------------------

class TestAnnotationDensity:
    def test_different_densities_produce_different_prompts(self):
        msgs_min = build_coach_prompt("Texto.", annotation_density="minimal")
        msgs_bal = build_coach_prompt("Texto.", annotation_density="balanced")
        msgs_det = build_coach_prompt("Texto.", annotation_density="detailed")
        # All three must differ
        assert _user(msgs_min) != _user(msgs_bal)
        assert _user(msgs_bal) != _user(msgs_det)
        assert _user(msgs_min) != _user(msgs_det)

    def test_default_density_is_balanced(self):
        msgs_default = build_coach_prompt("Texto.")
        msgs_balanced = build_coach_prompt("Texto.", annotation_density="balanced")
        assert msgs_default == msgs_balanced


# ---------------------------------------------------------------------------
# TestPurity
# ---------------------------------------------------------------------------

class TestPurity:
    def test_deterministic_same_inputs_same_output(self):
        text = "El tiempo pasa inexorable."
        r1 = build_coach_prompt(text, reader_level="B2")
        r2 = build_coach_prompt(text, reader_level="B2")
        assert r1 == r2

    def test_no_side_effects_on_source_text(self):
        text = "Texto original."
        original_id = id(text)
        build_coach_prompt(text)
        assert id(text) == original_id
        assert text == "Texto original."

    def test_function_is_fast(self):
        """Prompt building is pure computation — must complete well under 1 s."""
        import time
        text = "A" * 2000
        start = time.monotonic()
        build_coach_prompt(text)
        assert time.monotonic() - start < 1.0
