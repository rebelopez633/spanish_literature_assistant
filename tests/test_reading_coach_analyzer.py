"""Tests for reading_coach/analyzer.py — orchestration layer.

The fake LLM client is a plain callable that returns canned JSON, so no
Ollama process is needed.  All tests are pure unit tests.
"""
from __future__ import annotations

import json

import pytest

from reading_coach.analyzer import (
    AnalysisResult,
    CoachAnalysisError,
    analyze_spanish_source,
)
from reading_coach.checker import CoachCheckResult, CoachCheckerConfig, STATUS_PASSED, STATUS_WARNING
from reading_coach.schemas import ReadingCoachResult


# ---------------------------------------------------------------------------
# Shared fixtures
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

VALID_JSON = json.dumps(_VALID_PAYLOAD)


def fake_ok(messages: list[dict]) -> str:  # noqa: ARG001
    """Fake LLM client that always returns a valid ReadingCoachResult JSON."""
    return VALID_JSON


class CapturingClient:
    """Fake LLM client that records the messages it received."""

    def __init__(self, response: str = VALID_JSON) -> None:
        self.received: list[dict] | None = None
        self._response = response

    def __call__(self, messages: list[dict]) -> str:
        self.received = messages
        return self._response


# ---------------------------------------------------------------------------
# TestAnalysisResultShape
# ---------------------------------------------------------------------------

class TestAnalysisResultShape:
    def test_returns_analysis_result(self):
        result = analyze_spanish_source(SOURCE, llm_client=fake_ok)
        assert isinstance(result, AnalysisResult)

    def test_has_result_attribute(self):
        ar = analyze_spanish_source(SOURCE, llm_client=fake_ok)
        assert hasattr(ar, "result")

    def test_has_check_attribute(self):
        ar = analyze_spanish_source(SOURCE, llm_client=fake_ok)
        assert hasattr(ar, "check")

    def test_has_raw_response_attribute(self):
        ar = analyze_spanish_source(SOURCE, llm_client=fake_ok)
        assert hasattr(ar, "raw_response")

    def test_result_is_reading_coach_result(self):
        ar = analyze_spanish_source(SOURCE, llm_client=fake_ok)
        assert isinstance(ar.result, ReadingCoachResult)

    def test_check_is_coach_check_result(self):
        ar = analyze_spanish_source(SOURCE, llm_client=fake_ok)
        assert isinstance(ar.check, CoachCheckResult)

    def test_raw_response_is_str(self):
        ar = analyze_spanish_source(SOURCE, llm_client=fake_ok)
        assert isinstance(ar.raw_response, str)


# ---------------------------------------------------------------------------
# TestReadingCoachResultContents
# ---------------------------------------------------------------------------

class TestReadingCoachResultContents:
    def test_original_spanish_preserved(self):
        ar = analyze_spanish_source(SOURCE, llm_client=fake_ok)
        assert ar.result.original_spanish == SOURCE

    def test_overall_level_parsed(self):
        ar = analyze_spanish_source(SOURCE, llm_client=fake_ok)
        assert ar.result.overall_level  # non-empty

    def test_raw_response_matches_fake_output(self):
        ar = analyze_spanish_source(SOURCE, llm_client=fake_ok)
        assert ar.raw_response == VALID_JSON

    def test_modern_spanish_populated(self):
        ar = analyze_spanish_source(SOURCE, llm_client=fake_ok)
        assert ar.result.modern_spanish == _VALID_PAYLOAD["modern_spanish"]

    def test_english_gloss_populated(self):
        ar = analyze_spanish_source(SOURCE, llm_client=fake_ok)
        assert ar.result.english_gloss == _VALID_PAYLOAD["english_gloss"]

    def test_difficult_phrases_is_list(self):
        ar = analyze_spanish_source(SOURCE, llm_client=fake_ok)
        assert isinstance(ar.result.difficult_phrases, list)


# ---------------------------------------------------------------------------
# TestCheckerIntegration
# ---------------------------------------------------------------------------

class TestCheckerIntegration:
    def test_passed_for_good_result(self):
        ar = analyze_spanish_source(SOURCE, llm_client=fake_ok)
        assert ar.check.status == STATUS_PASSED

    def test_check_has_summary(self):
        ar = analyze_spanish_source(SOURCE, llm_client=fake_ok)
        assert ar.check.summary.strip()

    def test_check_issues_is_list(self):
        ar = analyze_spanish_source(SOURCE, llm_client=fake_ok)
        assert isinstance(ar.check.issues, list)

    def test_warning_for_phrase_not_in_source(self):
        """When the model returns a phrase that doesn't appear in the source,
        the checker must flag it as a warning."""
        bad_payload = dict(_VALID_PAYLOAD)
        bad_payload["difficult_phrases"] = [
            {
                "phrase": "XYZ_palabra_que_no_existe",
                "category": "archaic_vocab",
                "difficulty_level": "B2",
                "why_difficult": "test phrase absent from source",
            }
        ]
        client = CapturingClient(json.dumps(bad_payload))
        ar = analyze_spanish_source(SOURCE, llm_client=client)
        assert ar.check.status == STATUS_WARNING

    def test_warning_includes_phrase_name(self):
        bad_payload = dict(_VALID_PAYLOAD)
        bad_payload["difficult_phrases"] = [
            {
                "phrase": "ABSENT_PHRASE",
                "category": "idiom",
                "difficulty_level": "B1",
                "why_difficult": "not in source",
            }
        ]
        client = CapturingClient(json.dumps(bad_payload))
        ar = analyze_spanish_source(SOURCE, llm_client=client)
        assert any("ABSENT_PHRASE" in issue for issue in ar.check.issues)

    def test_checker_config_gloss_warning(self):
        """Passing a config with include_english_gloss=True on a result that has
        no english_gloss must trigger a warning."""
        no_gloss = dict(_VALID_PAYLOAD)
        no_gloss["english_gloss"] = None
        client = CapturingClient(json.dumps(no_gloss))
        config = CoachCheckerConfig(include_english_gloss=True)
        ar = analyze_spanish_source(SOURCE, llm_client=client, checker_config=config)
        assert ar.check.status == STATUS_WARNING

    def test_checker_config_no_gloss_warning_by_default(self):
        """Default config does NOT require english_gloss, so a null gloss is fine."""
        no_gloss = dict(_VALID_PAYLOAD)
        no_gloss["english_gloss"] = None
        client = CapturingClient(json.dumps(no_gloss))
        ar = analyze_spanish_source(SOURCE, llm_client=client)
        assert ar.check.status == STATUS_PASSED

    def test_phrase_in_source_passes(self):
        """A phrase that IS a substring of the source must not trigger a warning."""
        phrase_payload = dict(_VALID_PAYLOAD)
        phrase_payload["difficult_phrases"] = [
            {
                "phrase": "la Mancha",
                "category": "proper_noun",
                "difficulty_level": "B1",
                "why_difficult": "regional name",
            }
        ]
        client = CapturingClient(json.dumps(phrase_payload))
        ar = analyze_spanish_source(SOURCE, llm_client=client)
        assert ar.check.status == STATUS_PASSED


# ---------------------------------------------------------------------------
# TestLLMClientProtocol
# ---------------------------------------------------------------------------

class TestLLMClientProtocol:
    def test_client_is_called(self):
        client = CapturingClient()
        analyze_spanish_source(SOURCE, llm_client=client)
        assert client.received is not None

    def test_client_receives_messages_list(self):
        client = CapturingClient()
        analyze_spanish_source(SOURCE, llm_client=client)
        assert isinstance(client.received, list)

    def test_client_receives_two_messages(self):
        client = CapturingClient()
        analyze_spanish_source(SOURCE, llm_client=client)
        assert len(client.received) == 2

    def test_messages_have_role_keys(self):
        client = CapturingClient()
        analyze_spanish_source(SOURCE, llm_client=client)
        for msg in client.received:
            assert "role" in msg
            assert "content" in msg

    def test_first_message_is_system(self):
        client = CapturingClient()
        analyze_spanish_source(SOURCE, llm_client=client)
        assert client.received[0]["role"] == "system"

    def test_second_message_is_user(self):
        client = CapturingClient()
        analyze_spanish_source(SOURCE, llm_client=client)
        assert client.received[1]["role"] == "user"

    def test_source_text_in_user_message(self):
        client = CapturingClient()
        analyze_spanish_source(SOURCE, llm_client=client)
        assert SOURCE in client.received[1]["content"]

    def test_reader_level_in_prompt(self):
        client = CapturingClient()
        analyze_spanish_source(SOURCE, reader_level="C1", llm_client=client)
        combined = client.received[0]["content"] + client.received[1]["content"]
        assert "C1" in combined

    def test_default_reader_level_is_b1(self):
        client = CapturingClient()
        analyze_spanish_source(SOURCE, llm_client=client)
        combined = client.received[0]["content"] + client.received[1]["content"]
        assert "B1" in combined

    def test_annotation_density_affects_prompt(self):
        client_min = CapturingClient()
        client_det = CapturingClient()
        analyze_spanish_source(SOURCE, annotation_density="minimal", llm_client=client_min)
        analyze_spanish_source(SOURCE, annotation_density="detailed", llm_client=client_det)
        assert client_min.received[1]["content"] != client_det.received[1]["content"]

    def test_include_english_gloss_toggle_affects_prompt(self):
        client_on = CapturingClient()
        client_off = CapturingClient()
        analyze_spanish_source(SOURCE, include_english_gloss=True, llm_client=client_on)
        analyze_spanish_source(SOURCE, include_english_gloss=False, llm_client=client_off)
        assert client_on.received[1]["content"] != client_off.received[1]["content"]

    def test_include_modern_spanish_toggle_affects_prompt(self):
        client_on = CapturingClient()
        client_off = CapturingClient()
        analyze_spanish_source(SOURCE, include_modern_spanish=True, llm_client=client_on)
        analyze_spanish_source(SOURCE, include_modern_spanish=False, llm_client=client_off)
        assert client_on.received[1]["content"] != client_off.received[1]["content"]


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_invalid_json_raises_coach_analysis_error(self):
        def bad_client(messages):  # noqa: ARG001
            return "this is not json at all"

        with pytest.raises(CoachAnalysisError):
            analyze_spanish_source(SOURCE, llm_client=bad_client)

    def test_error_message_mentions_parsing(self):
        def bad_client(messages):  # noqa: ARG001
            return "not json"

        with pytest.raises(CoachAnalysisError, match=r"(?i)(parse|ReadingCoachResult|invalid)"):
            analyze_spanish_source(SOURCE, llm_client=bad_client)

    def test_empty_response_raises(self):
        def empty_client(messages):  # noqa: ARG001
            return ""

        with pytest.raises(CoachAnalysisError):
            analyze_spanish_source(SOURCE, llm_client=empty_client)

    def test_missing_required_field_raises(self):
        """JSON that omits the required 'original_spanish' field must raise."""
        incomplete = json.dumps({"overall_level": "B2"})

        def incomplete_client(messages):  # noqa: ARG001
            return incomplete

        with pytest.raises(CoachAnalysisError):
            analyze_spanish_source(SOURCE, llm_client=incomplete_client)

    def test_missing_overall_level_raises(self):
        """JSON that omits the required 'overall_level' field must raise."""
        incomplete = json.dumps({"original_spanish": SOURCE})

        def incomplete_client(messages):  # noqa: ARG001
            return incomplete

        with pytest.raises(CoachAnalysisError):
            analyze_spanish_source(SOURCE, llm_client=incomplete_client)

    def test_json_array_raises(self):
        """A JSON array (not an object) must raise, not silently succeed."""
        def array_client(messages):  # noqa: ARG001
            return json.dumps([1, 2, 3])

        with pytest.raises(CoachAnalysisError):
            analyze_spanish_source(SOURCE, llm_client=array_client)


# ---------------------------------------------------------------------------
# TestMarkdownFenceStripping
# ---------------------------------------------------------------------------

class TestMarkdownFenceStripping:
    def test_json_fenced_with_backticks(self):
        fenced = f"```json\n{VALID_JSON}\n```"
        client = CapturingClient(fenced)
        ar = analyze_spanish_source(SOURCE, llm_client=client)
        assert ar.result.original_spanish == SOURCE

    def test_json_fenced_without_language_tag(self):
        fenced = f"```\n{VALID_JSON}\n```"
        client = CapturingClient(fenced)
        ar = analyze_spanish_source(SOURCE, llm_client=client)
        assert isinstance(ar.result, ReadingCoachResult)

    def test_raw_response_is_original_unfenced(self):
        """raw_response must preserve the LLM's raw output, fences and all."""
        fenced = f"```json\n{VALID_JSON}\n```"
        client = CapturingClient(fenced)
        ar = analyze_spanish_source(SOURCE, llm_client=client)
        assert ar.raw_response == fenced

    def test_leading_and_trailing_whitespace_stripped(self):
        padded = f"   \n{VALID_JSON}\n   "
        client = CapturingClient(padded)
        ar = analyze_spanish_source(SOURCE, llm_client=client)
        assert isinstance(ar.result, ReadingCoachResult)


# ---------------------------------------------------------------------------
# TestPurity
# ---------------------------------------------------------------------------

class TestPurity:
    def test_does_not_mutate_source_text(self):
        original = SOURCE[:]
        analyze_spanish_source(SOURCE, llm_client=fake_ok)
        assert SOURCE == original

    def test_two_calls_same_input_same_result(self):
        ar1 = analyze_spanish_source(SOURCE, llm_client=fake_ok)
        ar2 = analyze_spanish_source(SOURCE, llm_client=fake_ok)
        assert ar1.result.original_spanish == ar2.result.original_spanish
        assert ar1.check.status == ar2.check.status
