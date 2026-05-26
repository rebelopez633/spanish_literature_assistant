"""Tests for slice 7: Spanish passage fixtures and their use across the pipeline.

Four acceptance criteria:
  1. original_spanish is preserved exactly in the LLM result.
  2. Prompt builder embeds the exact source text in the user message.
  3. Analyzer with a fake LLM can fully process a literary-style sentence.
  4. Checker flags phrases that are absent from the source text.

No external files or network access are needed — all LLM calls use
plain callables that return canned JSON.
"""
from __future__ import annotations

import json

import pytest

from tests.fixtures.spanish_passages import PASSAGES, SpanishPassage, get_passage
from reading_coach.prompts import build_coach_prompt
from reading_coach.analyzer import analyze_spanish_source, AnalysisResult
from reading_coach.checker import (
    check_coach_result,
    CoachCheckerConfig,
    STATUS_PASSED,
    STATUS_WARNING,
)
from reading_coach.schemas import DifficultPhrase, ReadingCoachResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_llm(source_text: str, *, level: str = "B2"):
    """Return a fake LLM callable whose response preserves source_text exactly."""
    payload = {
        "original_spanish": source_text,
        "overall_level": level,
        "modern_spanish": source_text + " [modernised]",
        "english_gloss": "English translation stub.",
        "difficult_phrases": [],
        "grammar_notes": [],
        "comprehension_question": None,
    }
    response = json.dumps(payload)

    def _client(messages: list[dict]) -> str:  # noqa: ARG001
        return response

    return _client


def _result_with_phrase(source_text: str, phrase: str) -> ReadingCoachResult:
    """Build a ReadingCoachResult containing one DifficultPhrase."""
    return ReadingCoachResult(
        original_spanish=source_text,
        overall_level="B2",
        difficult_phrases=[
            DifficultPhrase(
                phrase=phrase,
                category="vocabulary",
                difficulty_level="B2",
                why_difficult="Test fixture phrase.",
            )
        ],
    )


# ---------------------------------------------------------------------------
# TestFixtureCollection
# ---------------------------------------------------------------------------

class TestFixtureCollection:
    """Structural integrity of the fixture module itself."""

    def test_has_five_passages(self):
        assert len(PASSAGES) == 5

    def test_all_are_spanish_passage_instances(self):
        for p in PASSAGES:
            assert isinstance(p, SpanishPassage)

    def test_all_texts_are_non_empty_strings(self):
        for p in PASSAGES:
            assert isinstance(p.text, str)
            assert p.text.strip()

    def test_keys_are_unique(self):
        keys = [p.key for p in PASSAGES]
        assert len(keys) == len(set(keys))

    def test_registers_cover_all_required_categories(self):
        registers = {p.register for p in PASSAGES}
        for required in ("modern", "archaic", "mystical", "complex", "dialogue"):
            assert required in registers

    def test_all_notes_are_non_empty(self):
        for p in PASSAGES:
            assert p.note.strip()

    def test_all_passages_are_short(self):
        for p in PASSAGES:
            assert len(p.text) < 400, (
                f"Passage '{p.key}' is too long ({len(p.text)} chars); fixtures must be < 400 chars."
            )

    def test_get_passage_returns_correct_key(self):
        p = get_passage("simple_modern")
        assert p.key == "simple_modern"

    def test_get_passage_raises_for_unknown_key(self):
        with pytest.raises(KeyError):
            get_passage("does_not_exist")

    def test_simple_modern_key_present(self):
        get_passage("simple_modern")  # must not raise

    def test_archaic_literary_key_present(self):
        get_passage("archaic_literary")

    def test_mystical_religious_key_present(self):
        get_passage("mystical_religious")

    def test_complex_subordinate_key_present(self):
        get_passage("complex_subordinate")

    def test_dialogue_key_present(self):
        get_passage("dialogue")


# ---------------------------------------------------------------------------
# TestOriginalSpanishPreservation  (acceptance criterion 1)
# ---------------------------------------------------------------------------

class TestOriginalSpanishPreservation:
    """original_spanish in the LLM result must be byte-identical to the source."""

    def test_simple_modern_preserved(self):
        p = get_passage("simple_modern")
        ar = analyze_spanish_source(p.text, llm_client=_fake_llm(p.text))
        assert ar.result.original_spanish == p.text

    def test_archaic_preserved(self):
        p = get_passage("archaic_literary")
        ar = analyze_spanish_source(p.text, llm_client=_fake_llm(p.text))
        assert ar.result.original_spanish == p.text

    def test_mystical_preserved(self):
        p = get_passage("mystical_religious")
        ar = analyze_spanish_source(p.text, llm_client=_fake_llm(p.text))
        assert ar.result.original_spanish == p.text

    def test_complex_preserved(self):
        p = get_passage("complex_subordinate")
        ar = analyze_spanish_source(p.text, llm_client=_fake_llm(p.text))
        assert ar.result.original_spanish == p.text

    def test_dialogue_preserved(self):
        p = get_passage("dialogue")
        ar = analyze_spanish_source(p.text, llm_client=_fake_llm(p.text))
        assert ar.result.original_spanish == p.text


# ---------------------------------------------------------------------------
# TestPromptBuilderIncludesSourceText  (acceptance criterion 2)
# ---------------------------------------------------------------------------

class TestPromptBuilderIncludesSourceText:
    """build_coach_prompt must embed each fixture verbatim in the user message."""

    def test_simple_modern_in_user_message(self):
        p = get_passage("simple_modern")
        msgs = build_coach_prompt(p.text)
        assert p.text in msgs[1]["content"]

    def test_archaic_in_user_message(self):
        p = get_passage("archaic_literary")
        msgs = build_coach_prompt(p.text)
        assert p.text in msgs[1]["content"]

    def test_mystical_in_user_message(self):
        p = get_passage("mystical_religious")
        msgs = build_coach_prompt(p.text)
        assert p.text in msgs[1]["content"]

    def test_complex_in_user_message(self):
        p = get_passage("complex_subordinate")
        msgs = build_coach_prompt(p.text)
        assert p.text in msgs[1]["content"]

    def test_dialogue_in_user_message(self):
        p = get_passage("dialogue")
        msgs = build_coach_prompt(p.text)
        assert p.text in msgs[1]["content"]

    def test_archaic_source_not_in_system_message(self):
        """Source text should appear in the user turn, not the system prompt."""
        p = get_passage("archaic_literary")
        msgs = build_coach_prompt(p.text)
        assert p.text not in msgs[0]["content"]


# ---------------------------------------------------------------------------
# TestAnalyzerWithLiteraryFixture  (acceptance criterion 3)
# ---------------------------------------------------------------------------

class TestAnalyzerWithLiteraryFixture:
    """Analyzer completes the full pipeline with literary / non-trivial passages."""

    def test_archaic_returns_analysis_result(self):
        p = get_passage("archaic_literary")
        ar = analyze_spanish_source(p.text, llm_client=_fake_llm(p.text))
        assert isinstance(ar, AnalysisResult)

    def test_archaic_result_is_reading_coach_result(self):
        p = get_passage("archaic_literary")
        ar = analyze_spanish_source(p.text, llm_client=_fake_llm(p.text))
        assert isinstance(ar.result, ReadingCoachResult)

    def test_archaic_raw_response_is_non_empty(self):
        p = get_passage("archaic_literary")
        ar = analyze_spanish_source(p.text, llm_client=_fake_llm(p.text))
        assert isinstance(ar.raw_response, str)
        assert ar.raw_response.strip()

    def test_archaic_with_b1_reader_level(self):
        p = get_passage("archaic_literary")
        ar = analyze_spanish_source(
            p.text, reader_level="B1", llm_client=_fake_llm(p.text, level="B1")
        )
        assert ar.result.overall_level == "B1"

    def test_mystical_passage_processes_ok(self):
        p = get_passage("mystical_religious")
        ar = analyze_spanish_source(p.text, llm_client=_fake_llm(p.text))
        assert ar.result.original_spanish == p.text

    def test_complex_passage_processes_ok(self):
        p = get_passage("complex_subordinate")
        ar = analyze_spanish_source(p.text, llm_client=_fake_llm(p.text))
        assert isinstance(ar, AnalysisResult)

    def test_dialogue_passage_processes_ok(self):
        p = get_passage("dialogue")
        ar = analyze_spanish_source(p.text, llm_client=_fake_llm(p.text))
        assert ar.result.original_spanish == p.text

    def test_archaic_check_result_attached(self):
        p = get_passage("archaic_literary")
        ar = analyze_spanish_source(p.text, llm_client=_fake_llm(p.text))
        assert hasattr(ar, "check")
        assert ar.check.status in (STATUS_PASSED, STATUS_WARNING)


# ---------------------------------------------------------------------------
# TestCheckerWithFixtures  (acceptance criterion 4)
# ---------------------------------------------------------------------------

class TestCheckerWithFixtures:
    """Checker must flag phrases absent from the source text."""

    def test_phantom_phrase_on_modern_passage_is_warning(self):
        p = get_passage("simple_modern")
        result = _result_with_phrase(p.text, "absolutamente inexistente")
        check = check_coach_result(p.text, result)
        assert check.status == STATUS_WARNING

    def test_phantom_phrase_text_appears_in_issues(self):
        p = get_passage("simple_modern")
        phantom = "frase fantasma"
        result = _result_with_phrase(p.text, phantom)
        check = check_coach_result(p.text, result)
        combined_issues = " ".join(check.issues)
        assert phantom in combined_issues

    def test_real_phrase_from_archaic_does_not_warn(self):
        p = get_passage("archaic_literary")
        # "caballero" is present in the archaic passage
        result = _result_with_phrase(p.text, "caballero")
        check = check_coach_result(p.text, result)
        phrase_warnings = [issue for issue in check.issues if "caballero" in issue]
        assert phrase_warnings == []

    def test_phantom_phrase_on_mystical_passage_is_warning(self):
        p = get_passage("mystical_religious")
        result = _result_with_phrase(p.text, "concepto inventado")
        check = check_coach_result(p.text, result)
        assert check.status == STATUS_WARNING

    def test_phantom_phrase_on_dialogue_passage_is_warning(self):
        p = get_passage("dialogue")
        result = _result_with_phrase(p.text, "inexistente completamente")
        check = check_coach_result(p.text, result)
        assert check.status == STATUS_WARNING

    def test_empty_difficult_phrases_passes_checker(self):
        p = get_passage("complex_subordinate")
        result = ReadingCoachResult(
            original_spanish=p.text,
            overall_level="B2",
            difficult_phrases=[],
        )
        check = check_coach_result(p.text, result)
        assert check.status == STATUS_PASSED

    def test_phrase_present_in_complex_passage_no_warning(self):
        p = get_passage("complex_subordinate")
        # "viajeros" is in the complex passage
        result = _result_with_phrase(p.text, "viajeros")
        check = check_coach_result(p.text, result)
        phrase_warnings = [issue for issue in check.issues if "viajeros" in issue]
        assert phrase_warnings == []
