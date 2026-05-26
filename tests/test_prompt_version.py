"""
Prompt versioning and regression fixtures — reading_coach/prompts.py.

Two concerns covered here:

1. Version constant stability
   SPANISH_SOURCE_PROMPT_VERSION is a code-level string that identifies which
   generation of the prompt builder is in use.  Tests assert its value is
   stable so any bump is an explicit, reviewable change.

2. Prompt regression invariants
   Rather than snapshotting the entire prompt (brittle), these tests assert
   that specific *behavioural instructions* are present.  If a prompt edit
   removes or contradicts one of these invariants the corresponding test goes
   red, signalling that the change was significant and the developer must also
   bump SPANISH_SOURCE_PROMPT_VERSION.

   The invariants cover:
     - preserve original Spanish exactly (verbatim copy rule)
     - phrase-level explanations preferred over dictionary glosses
     - anti-hallucination / no fabricated context
     - return valid JSON, no Markdown
     - reader level embedded in analysis rules
     - annotation density keyword + count guidance per tier
     - modern Spanish instruction toggled correctly
     - English gloss instruction toggled correctly

3. AnalysisResult carries prompt_version metadata
   Results are traceable to the prompt that produced them.
"""
from __future__ import annotations

import json
import unittest.mock as mock

import pytest

from reading_coach.prompts import SPANISH_SOURCE_PROMPT_VERSION, build_coach_prompt


# ---------------------------------------------------------------------------
# Helpers (mirrors test_reading_coach_prompts.py conventions)
# ---------------------------------------------------------------------------

def _sys(msgs: list[dict]) -> str:
    return msgs[0]["content"]


def _user(msgs: list[dict]) -> str:
    return msgs[1]["content"]


# ===========================================================================
# TestPromptVersionConstant
# ===========================================================================

class TestPromptVersionConstant:
    """SPANISH_SOURCE_PROMPT_VERSION must be a stable, well-formed string."""

    def test_constant_is_a_string(self):
        assert isinstance(SPANISH_SOURCE_PROMPT_VERSION, str)

    def test_constant_is_non_empty(self):
        assert SPANISH_SOURCE_PROMPT_VERSION.strip()

    def test_constant_stable_value(self):
        """Fail here if the version string is changed without a deliberate update.

        Bumping the version is intentional: change the expected value below AND
        update reading_coach/prompts.py AND adjust any invariants that changed.
        """
        assert SPANISH_SOURCE_PROMPT_VERSION == "spanish_source_v2"

    def test_constant_naming_convention(self):
        """Version identifiers follow the 'spanish_source_' prefix convention."""
        assert SPANISH_SOURCE_PROMPT_VERSION.startswith("spanish_source_")


# ===========================================================================
# TestPromptVersionInAnalysisResult
# ===========================================================================

class TestPromptVersionInAnalysisResult:
    """AnalysisResult must carry prompt_version so results are traceable."""

    def _make_minimal_result(self):
        from reading_coach.analyzer import AnalysisResult
        from reading_coach.checker import CoachCheckResult, STATUS_PASSED
        from reading_coach.schemas import ReadingCoachResult
        check = CoachCheckResult(status=STATUS_PASSED, summary="ok", issues=[])
        return AnalysisResult(
            result=ReadingCoachResult(original_spanish="x", overall_level="B1"),
            check=check,
            raw_response="{}",
        )

    def test_analysis_result_has_prompt_version_field(self):
        ar = self._make_minimal_result()
        assert hasattr(ar, "prompt_version")

    def test_analysis_result_prompt_version_is_string(self):
        ar = self._make_minimal_result()
        assert isinstance(ar.prompt_version, str)

    def test_analysis_result_default_version_matches_constant(self):
        ar = self._make_minimal_result()
        assert ar.prompt_version == SPANISH_SOURCE_PROMPT_VERSION

    def test_analyze_spanish_source_populates_prompt_version(self):
        """End-to-end: the live analyzer must stamp prompt_version onto its result."""
        from reading_coach.analyzer import analyze_spanish_source
        fake_response = json.dumps({
            "original_spanish": "Texto de prueba.",
            "overall_level": "B1",
            "difficult_phrases": [],
            "grammar_notes": [],
        })
        fake_client = mock.MagicMock(return_value=fake_response)
        ar = analyze_spanish_source(
            "Texto de prueba.",
            reader_level="B1",
            annotation_density="balanced",
            include_english_gloss=False,
            include_modern_spanish=False,
            llm_client=fake_client,
        )
        assert ar.prompt_version == SPANISH_SOURCE_PROMPT_VERSION


# ===========================================================================
# TestPromptInvariantPreserveOriginal
# Regression guards for the "copy source verbatim" rules.
# ===========================================================================

class TestPromptInvariantPreserveOriginal:
    """The prompt must unambiguously instruct the model to copy the source verbatim."""

    def test_exact_copy_instruction_in_user(self):
        """User message must use 'exact' to describe the copy requirement."""
        msgs = build_coach_prompt("Texto.")
        assert "exact" in _user(msgs).lower()

    def test_never_alter_instruction_present(self):
        """Model must be told explicitly not to alter the original."""
        msgs = build_coach_prompt("Texto.")
        combined = (_sys(msgs) + _user(msgs)).lower()
        assert "never alter" in combined or "do not alter" in combined

    def test_original_spanish_field_name_in_user_rule(self):
        """The copy rule explicitly references the 'original_spanish' field."""
        msgs = build_coach_prompt("Texto.")
        assert "original_spanish" in _user(msgs)

    def test_source_text_section_label_present(self):
        """The user message delimits the passage with a 'SOURCE TEXT' heading."""
        msgs = build_coach_prompt("Texto.")
        assert "SOURCE TEXT" in _user(msgs)

    def test_source_text_embedded_verbatim(self):
        """The passage itself must appear unchanged in the user message."""
        passage = "No ha mucho tiempo que vivía un hidalgo."
        msgs = build_coach_prompt(passage)
        assert passage in _user(msgs)


# ===========================================================================
# TestPromptInvariantPhraseLevelExplanations
# Regression guards for annotation-quality instructions.
# ===========================================================================

class TestPromptInvariantPhraseLevelExplanations:
    """Phrase-level explanations must be preferred over isolated dictionary glosses."""

    def test_phrase_level_instruction_present(self):
        msgs = build_coach_prompt("Texto.")
        assert "phrase-level" in _user(msgs).lower()

    def test_difficult_phrases_rule_references_learner_level(self):
        """The difficult_phrases rule must mention the reader's specific level."""
        msgs = build_coach_prompt("Texto.", reader_level="C1")
        assert "C1" in _user(msgs)

    def test_exact_substring_instruction_present(self):
        """Phrases must be exact substrings of the source text."""
        msgs = build_coach_prompt("Texto.")
        user_lower = _user(msgs).lower()
        assert "exact substring" in user_lower or ("exact" in user_lower and "substring" in user_lower)

    def test_avoid_over_annotation_instruction_present(self):
        """Model must be told not to over-annotate obvious or common words."""
        msgs = build_coach_prompt("Texto.")
        user_lower = _user(msgs).lower()
        assert "over-annotat" in user_lower or "avoid" in user_lower


# ===========================================================================
# TestPromptInvariantAntiHallucination
# Regression guards for anti-fabrication instructions.
# ===========================================================================

class TestPromptInvariantAntiHallucination:
    """Explicit instructions to avoid hallucination must be in the system message."""

    def test_do_not_invent_instruction_in_system(self):
        msgs = build_coach_prompt("Texto.")
        sys_lower = _sys(msgs).lower()
        assert "do not invent" in sys_lower or "not invent" in sys_lower

    def test_do_not_fabricate_instruction_in_system(self):
        msgs = build_coach_prompt("Texto.")
        assert "fabricat" in _sys(msgs).lower()

    def test_restriction_to_passage_content_present(self):
        """Model must be restricted to context present in the passage."""
        msgs = build_coach_prompt("Texto.")
        sys_lower = _sys(msgs).lower()
        assert "not present" in sys_lower or "in the passage" in sys_lower or "source text" in sys_lower


# ===========================================================================
# TestPromptInvariantJsonOutput
# Regression guards for JSON-only output instructions.
# ===========================================================================

class TestPromptInvariantJsonOutput:
    """The prompt must instruct the model to return only valid JSON."""

    def test_valid_json_instruction_in_system(self):
        msgs = build_coach_prompt("Texto.")
        assert "valid json" in _sys(msgs).lower()

    def test_no_markdown_instruction_in_system(self):
        """System message must tell the model not to wrap output in Markdown."""
        msgs = build_coach_prompt("Texto.")
        assert "markdown" in _sys(msgs).lower()

    def test_json_reinforced_in_user_message(self):
        """User message must also reference JSON output."""
        msgs = build_coach_prompt("Texto.")
        assert "json" in _user(msgs).lower()

    def test_schema_example_embedded_in_user(self):
        """A concrete JSON schema example must be embedded in the user message."""
        msgs = build_coach_prompt("Texto.")
        user = _user(msgs)
        assert '"difficult_phrases"' in user or "difficult_phrases" in user

    def test_cefr_values_enumerated_in_user(self):
        """Model must be told to use only valid CEFR codes for level fields."""
        msgs = build_coach_prompt("Texto.")
        user = _user(msgs)
        assert "A1" in user and "C2" in user


# ===========================================================================
# TestPromptInvariantAnnotationDensity
# Regression guards for density tier instructions.
# ===========================================================================

class TestPromptInvariantAnnotationDensity:
    """Each density tier must include its keyword and count guidance."""

    def test_minimal_keyword_uppercased_in_user(self):
        msgs = build_coach_prompt("Texto.", annotation_density="minimal")
        assert "MINIMAL" in _user(msgs)

    def test_balanced_keyword_uppercased_in_user(self):
        msgs = build_coach_prompt("Texto.", annotation_density="balanced")
        assert "BALANCED" in _user(msgs)

    def test_detailed_keyword_uppercased_in_user(self):
        msgs = build_coach_prompt("Texto.", annotation_density="detailed")
        assert "DETAILED" in _user(msgs)

    def test_minimal_includes_low_count_guidance(self):
        """Minimal density must specify a small annotation count target."""
        msgs = build_coach_prompt("Texto.", annotation_density="minimal")
        user = _user(msgs).lower()
        # "three to five" or numeric equivalents
        has_guidance = (
            ("three" in user or "3" in user)
            and ("five" in user or "5" in user)
        )
        assert has_guidance

    def test_detailed_includes_high_count_guidance(self):
        """Detailed density must specify ten-or-more annotations."""
        msgs = build_coach_prompt("Texto.", annotation_density="detailed")
        user = _user(msgs).lower()
        assert "ten" in user or "10" in user


# ===========================================================================
# TestPromptInvariantModernSpanishInstruction
# ===========================================================================

class TestPromptInvariantModernSpanishInstruction:
    """Modern Spanish paraphrase instruction must match the toggle state."""

    def test_enabled_requests_paraphrase(self):
        msgs = build_coach_prompt("Texto.", include_modern_spanish=True)
        user_lower = _user(msgs).lower()
        assert "modern spanish paraphrase" in user_lower or "modern_spanish" in user_lower

    def test_enabled_does_not_say_set_to_null(self):
        """When enabled the prompt must not tell the model to set the field null."""
        msgs = build_coach_prompt("Texto.", include_modern_spanish=True)
        assert "do not provide a modern spanish" not in _user(msgs).lower()

    def test_disabled_instructs_null(self):
        msgs = build_coach_prompt("Texto.", include_modern_spanish=False)
        user_lower = _user(msgs).lower()
        assert "null" in user_lower
        assert "do not provide" in user_lower

    def test_disabled_omits_positive_paraphrase_instruction(self):
        msgs_off = build_coach_prompt("Texto.", include_modern_spanish=False)
        assert "provide a clear modern spanish paraphrase" not in _user(msgs_off).lower()

    def test_enabled_requests_per_phrase_equivalent(self):
        """When modern Spanish is on, the prompt must ask for per-phrase equivalents."""
        msgs = build_coach_prompt("Texto.", include_modern_spanish=True)
        user_lower = _user(msgs).lower()
        assert "modern_spanish_equivalent" in user_lower
        assert "set to null" not in user_lower.split("modern_spanish_equivalent")[1].split("\n")[0]

    def test_disabled_suppresses_per_phrase_equivalent(self):
        """When modern Spanish is off, per-phrase equivalents must be suppressed (null)."""
        msgs = build_coach_prompt("Texto.", include_modern_spanish=False)
        user_lower = _user(msgs).lower()
        assert "modern_spanish_equivalent" in user_lower
        # The rule appears before the schema example — use find() to hit the rule line.
        idx = user_lower.find("modern_spanish_equivalent")
        surrounding = user_lower[idx:idx + 120]
        assert "null" in surrounding


# ===========================================================================
# TestPromptInvariantEnglishGlossInstruction
# ===========================================================================

class TestPromptInvariantEnglishGlossInstruction:
    """English gloss instruction must match the toggle state."""

    def test_enabled_requests_translation(self):
        msgs = build_coach_prompt("Texto.", include_english_gloss=True)
        user_lower = _user(msgs).lower()
        assert "english translation" in user_lower or "flowing english" in user_lower

    def test_enabled_does_not_say_set_to_null(self):
        msgs = build_coach_prompt("Texto.", include_english_gloss=True)
        assert "do not provide an english translation" not in _user(msgs).lower()

    def test_disabled_instructs_null(self):
        msgs = build_coach_prompt("Texto.", include_english_gloss=False)
        user_lower = _user(msgs).lower()
        assert "null" in user_lower
        assert "do not provide" in user_lower

    def test_disabled_omits_positive_translation_instruction(self):
        msgs_off = build_coach_prompt("Texto.", include_english_gloss=False)
        assert "provide a flowing english translation" not in _user(msgs_off).lower()
