"""
Tests: deterministic checker for ReadingCoachResult objects.

Pure Python — no Streamlit, no Ollama, no external services.
"""
from __future__ import annotations

import pytest

from reading_coach.schemas import (
    ComprehensionQuestion,
    DifficultPhrase,
    GrammarNote,
    ReadingCoachResult,
)
from reading_coach.checker import (
    STATUS_PASSED,
    STATUS_WARNING,
    STATUS_FAILED,
    CoachCheckResult,
    CoachCheckerConfig,
    check_coach_result,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SOURCE = (
    "No hay mal que por bien no venga. "
    "Hemos topado con la Iglesia, Sancho. "
    "A ciegas caminamos por este mundo."
)


def _phrase(text: str, **kw) -> DifficultPhrase:
    return DifficultPhrase(
        phrase=text,
        category="expression",
        difficulty_level="B2",
        why_difficult="Idiomatic.",
        **kw,
    )


def _result(**overrides) -> ReadingCoachResult:
    """Build a minimal passing ReadingCoachResult against SOURCE."""
    base: dict = {
        "original_spanish": SOURCE,
        "overall_level": "B2",
    }
    base.update(overrides)
    return ReadingCoachResult(**base)


def _default_cfg(**kw) -> CoachCheckerConfig:
    return CoachCheckerConfig(**kw)


# ===========================================================================
# Status constants
# ===========================================================================

class TestStatusConstants:
    def test_passed_value(self):
        assert STATUS_PASSED == "passed"

    def test_warning_value(self):
        assert STATUS_WARNING == "warning"

    def test_failed_value(self):
        assert STATUS_FAILED == "failed"

    def test_all_distinct(self):
        assert len({STATUS_PASSED, STATUS_WARNING, STATUS_FAILED}) == 3


# ===========================================================================
# CoachCheckResult structure
# ===========================================================================

class TestCoachCheckResult:
    def test_passed_construction(self):
        r = CoachCheckResult(status=STATUS_PASSED, summary="All checks passed.", issues=[])
        assert r.status == STATUS_PASSED
        assert r.issues == []

    def test_issues_list_preserved(self):
        r = CoachCheckResult(status=STATUS_WARNING, summary="One issue.", issues=["Issue A"])
        assert r.issues == ["Issue A"]

    def test_status_must_be_valid(self):
        # Any of the three valid statuses should be accepted.
        for s in (STATUS_PASSED, STATUS_WARNING, STATUS_FAILED):
            CoachCheckResult(status=s, summary="ok", issues=[])

    def test_summary_is_str(self):
        r = CoachCheckResult(status=STATUS_PASSED, summary="ok", issues=[])
        assert isinstance(r.summary, str)


# ===========================================================================
# CoachCheckerConfig defaults
# ===========================================================================

class TestCoachCheckerConfig:
    def test_default_include_english_gloss_is_false(self):
        cfg = CoachCheckerConfig()
        assert cfg.include_english_gloss is False

    def test_default_max_annotations_is_positive(self):
        cfg = CoachCheckerConfig()
        assert cfg.max_annotations > 0

    def test_default_trivial_length_threshold_is_positive(self):
        cfg = CoachCheckerConfig()
        assert cfg.trivial_length_threshold > 0

    def test_custom_values_accepted(self):
        cfg = CoachCheckerConfig(include_english_gloss=True, max_annotations=5, trivial_length_threshold=50)
        assert cfg.include_english_gloss is True
        assert cfg.max_annotations == 5
        assert cfg.trivial_length_threshold == 50


# ===========================================================================
# Check 1: original_spanish exact match
# ===========================================================================

class TestOriginalSpanishMatch:
    def test_exact_match_passes(self):
        result = check_coach_result(SOURCE, _result())
        assert result.status == STATUS_PASSED

    def test_truncated_original_fails(self):
        result = check_coach_result(SOURCE, _result(original_spanish=SOURCE[:-1]))
        assert result.status == STATUS_FAILED

    def test_prepended_text_fails(self):
        result = check_coach_result(SOURCE, _result(original_spanish="Extra. " + SOURCE))
        assert result.status == STATUS_FAILED

    def test_different_text_fails(self):
        result = check_coach_result(SOURCE, _result(original_spanish="Completamente diferente."))
        assert result.status == STATUS_FAILED

    def test_mismatch_issue_is_reported(self):
        result = check_coach_result(SOURCE, _result(original_spanish="Modified."))
        assert any("original_spanish" in issue.lower() or "source" in issue.lower()
                   for issue in result.issues)

    def test_empty_source_and_empty_original_passes(self):
        r = ReadingCoachResult(original_spanish="", overall_level="B1")
        result = check_coach_result("", r)
        assert result.status == STATUS_PASSED

    def test_whitespace_difference_fails(self):
        # Adding a trailing space counts as a mismatch.
        result = check_coach_result(SOURCE, _result(original_spanish=SOURCE + " "))
        assert result.status == STATUS_FAILED


# ===========================================================================
# Check 2: difficult phrases appear in source_text (case-insensitive)
# ===========================================================================

class TestDifficultPhrasePresence:
    def test_phrase_in_source_no_issue(self):
        r = _result(difficult_phrases=[_phrase("a ciegas")])
        result = check_coach_result(SOURCE, r)
        assert result.status == STATUS_PASSED

    def test_phrase_not_in_source_is_warning(self):
        r = _result(difficult_phrases=[_phrase("frase inventada xyz")])
        result = check_coach_result(SOURCE, r)
        assert result.status == STATUS_WARNING

    def test_missing_phrase_reported_in_issues(self):
        r = _result(difficult_phrases=[_phrase("frase inventada xyz")])
        result = check_coach_result(SOURCE, r)
        assert any("frase inventada xyz" in issue for issue in result.issues)

    def test_multiple_phrases_one_missing_is_warning(self):
        r = _result(difficult_phrases=[
            _phrase("a ciegas"),          # present
            _phrase("palabra_inexistente"),  # absent
        ])
        result = check_coach_result(SOURCE, r)
        assert result.status == STATUS_WARNING

    def test_all_phrases_present_passes(self):
        r = _result(difficult_phrases=[
            _phrase("por bien no venga"),
            _phrase("hemos topado"),
            _phrase("a ciegas"),
        ])
        result = check_coach_result(SOURCE, r)
        assert result.status == STATUS_PASSED

    def test_phrase_match_is_case_insensitive(self):
        # "No hay mal" is in SOURCE; test uppercase lookup succeeds.
        r = _result(difficult_phrases=[_phrase("NO HAY MAL")])
        result = check_coach_result(SOURCE, r)
        assert result.status == STATUS_PASSED

    def test_empty_difficult_phrases_always_passes_this_check(self):
        r = _result(difficult_phrases=[])
        result = check_coach_result(SOURCE, r)
        assert result.status == STATUS_PASSED


# ===========================================================================
# Check 3: modern_spanish should differ from original_spanish when there
#          are difficult phrases
# ===========================================================================

class TestModernSpanishIdentity:
    def test_modern_none_no_warning(self):
        r = _result(
            difficult_phrases=[_phrase("por bien no venga")],
            modern_spanish=None,
        )
        result = check_coach_result(SOURCE, r)
        assert result.status == STATUS_PASSED

    def test_modern_differs_no_warning(self):
        r = _result(
            difficult_phrases=[_phrase("por bien no venga")],
            modern_spanish="No hay problema que no traiga algo bueno.",
        )
        result = check_coach_result(SOURCE, r)
        assert result.status == STATUS_PASSED

    def test_modern_identical_to_original_with_phrases_is_warning(self):
        r = _result(
            difficult_phrases=[_phrase("por bien no venga")],
            modern_spanish=SOURCE,
        )
        result = check_coach_result(SOURCE, r)
        assert result.status == STATUS_WARNING

    def test_modern_identical_no_phrases_no_warning(self):
        # No difficult phrases → modernisation wasn't attempted; no warning.
        r = _result(difficult_phrases=[], modern_spanish=SOURCE)
        result = check_coach_result(SOURCE, r)
        assert result.status == STATUS_PASSED

    def test_identity_issue_reported(self):
        r = _result(
            difficult_phrases=[_phrase("por bien no venga")],
            modern_spanish=SOURCE,
        )
        result = check_coach_result(SOURCE, r)
        assert any("modern" in i.lower() or "identical" in i.lower() for i in result.issues)


# ===========================================================================
# Check 4: english_gloss non-empty when include_english_gloss=True
# ===========================================================================

class TestEnglishGloss:
    def test_gloss_present_no_warning(self):
        r = _result(english_gloss="Every cloud has a silver lining.")
        cfg = _default_cfg(include_english_gloss=True)
        result = check_coach_result(SOURCE, r, config=cfg)
        assert result.status == STATUS_PASSED

    def test_gloss_empty_with_flag_is_warning(self):
        r = _result(english_gloss=None)
        cfg = _default_cfg(include_english_gloss=True)
        result = check_coach_result(SOURCE, r, config=cfg)
        assert result.status == STATUS_WARNING

    def test_gloss_empty_string_with_flag_is_warning(self):
        r = _result(english_gloss="")
        cfg = _default_cfg(include_english_gloss=True)
        result = check_coach_result(SOURCE, r, config=cfg)
        assert result.status == STATUS_WARNING

    def test_gloss_absent_flag_false_no_warning(self):
        r = _result(english_gloss=None)
        cfg = _default_cfg(include_english_gloss=False)
        result = check_coach_result(SOURCE, r, config=cfg)
        assert result.status == STATUS_PASSED

    def test_gloss_issue_reported_in_issues(self):
        r = _result(english_gloss=None)
        cfg = _default_cfg(include_english_gloss=True)
        result = check_coach_result(SOURCE, r, config=cfg)
        assert any("gloss" in i.lower() or "english" in i.lower() for i in result.issues)


# ===========================================================================
# Check 5: annotation count ≤ max_annotations
# ===========================================================================

class TestAnnotationCount:
    def _make_phrases(self, n: int) -> list[DifficultPhrase]:
        # All use a phrase that IS in SOURCE so phrase-presence check passes.
        return [_phrase("a ciegas") for _ in range(n)]

    def test_within_limit_passes(self):
        cfg = _default_cfg(max_annotations=5)
        r = _result(difficult_phrases=self._make_phrases(5))
        result = check_coach_result(SOURCE, r, config=cfg)
        assert result.status == STATUS_PASSED

    def test_exactly_at_limit_passes(self):
        cfg = _default_cfg(max_annotations=3)
        r = _result(difficult_phrases=self._make_phrases(3))
        result = check_coach_result(SOURCE, r, config=cfg)
        assert result.status == STATUS_PASSED

    def test_one_over_limit_is_warning(self):
        cfg = _default_cfg(max_annotations=3)
        r = _result(difficult_phrases=self._make_phrases(4))
        result = check_coach_result(SOURCE, r, config=cfg)
        assert result.status == STATUS_WARNING

    def test_far_over_limit_is_warning(self):
        cfg = _default_cfg(max_annotations=2)
        r = _result(difficult_phrases=self._make_phrases(10))
        result = check_coach_result(SOURCE, r, config=cfg)
        assert result.status == STATUS_WARNING

    def test_count_issue_reported(self):
        cfg = _default_cfg(max_annotations=2)
        r = _result(difficult_phrases=self._make_phrases(5))
        result = check_coach_result(SOURCE, r, config=cfg)
        assert any("annotation" in i.lower() or "phrase" in i.lower() for i in result.issues)


# ===========================================================================
# Check 6: beginner level (A1/A2) + non-trivial length + empty phrases → warning
# ===========================================================================

LONG_SOURCE = "Esta es una frase muy larga. " * 10  # > any reasonable threshold


class TestBeginnerLevelEmptyAnnotations:
    def test_a1_long_source_no_phrases_is_warning(self):
        r = ReadingCoachResult(
            original_spanish=LONG_SOURCE,
            overall_level="A1",
            difficult_phrases=[],
        )
        cfg = _default_cfg(trivial_length_threshold=50)
        result = check_coach_result(LONG_SOURCE, r, config=cfg)
        assert result.status == STATUS_WARNING

    def test_a2_long_source_no_phrases_is_warning(self):
        r = ReadingCoachResult(
            original_spanish=LONG_SOURCE,
            overall_level="A2",
            difficult_phrases=[],
        )
        cfg = _default_cfg(trivial_length_threshold=50)
        result = check_coach_result(LONG_SOURCE, r, config=cfg)
        assert result.status == STATUS_WARNING

    def test_a1_short_source_no_phrases_no_warning(self):
        short = "Hola."
        r = ReadingCoachResult(original_spanish=short, overall_level="A1", difficult_phrases=[])
        cfg = _default_cfg(trivial_length_threshold=50)
        result = check_coach_result(short, r, config=cfg)
        assert result.status == STATUS_PASSED

    def test_b1_long_source_no_phrases_no_warning(self):
        r = ReadingCoachResult(
            original_spanish=LONG_SOURCE,
            overall_level="B1",
            difficult_phrases=[],
        )
        cfg = _default_cfg(trivial_length_threshold=50)
        result = check_coach_result(LONG_SOURCE, r, config=cfg)
        assert result.status == STATUS_PASSED

    def test_a1_long_source_with_phrases_no_warning(self):
        r = ReadingCoachResult(
            original_spanish=LONG_SOURCE,
            overall_level="A1",
            difficult_phrases=[_phrase("Esta es una frase")],
        )
        cfg = _default_cfg(trivial_length_threshold=50)
        result = check_coach_result(LONG_SOURCE, r, config=cfg)
        assert result.status == STATUS_PASSED

    def test_beginner_issue_reported(self):
        r = ReadingCoachResult(
            original_spanish=LONG_SOURCE,
            overall_level="A1",
            difficult_phrases=[],
        )
        cfg = _default_cfg(trivial_length_threshold=50)
        result = check_coach_result(LONG_SOURCE, r, config=cfg)
        assert any(
            "a1" in i.lower() or "a2" in i.lower() or "beginner" in i.lower() or "phrase" in i.lower()
            for i in result.issues
        )


# ===========================================================================
# Severity ordering — failed beats warning
# ===========================================================================

class TestSeverityOrdering:
    def test_failed_beats_warning(self):
        # Trigger failed (wrong original_spanish) AND warning (phrase not in source)
        r = ReadingCoachResult(
            original_spanish="Wrong text.",
            overall_level="B1",
            difficult_phrases=[_phrase("palabra_inexistente_xyz")],
        )
        result = check_coach_result(SOURCE, r)
        assert result.status == STATUS_FAILED

    def test_multiple_warnings_stay_warning(self):
        # Trigger two warning conditions but no failed.
        r = _result(
            difficult_phrases=[_phrase("frase_no_existe")],
            modern_spanish=SOURCE,  # identical to original + has phrases
        )
        result = check_coach_result(SOURCE, r)
        assert result.status == STATUS_WARNING

    def test_all_issues_collected_even_when_failed(self):
        # Both failed AND warning checks fire — all issues should be listed.
        r = ReadingCoachResult(
            original_spanish="Wrong.",
            overall_level="B1",
            difficult_phrases=[_phrase("frase_inventada_xyz")],
        )
        result = check_coach_result(SOURCE, r)
        assert len(result.issues) >= 2


# ===========================================================================
# summary is always a non-empty string
# ===========================================================================

class TestSummaryAlwaysSet:
    def test_passing_result_has_summary(self):
        result = check_coach_result(SOURCE, _result())
        assert isinstance(result.summary, str)
        assert result.summary.strip() != ""

    def test_failed_result_has_summary(self):
        r = _result(original_spanish="Different.")
        result = check_coach_result(SOURCE, r)
        assert result.summary.strip() != ""

    def test_warning_result_has_summary(self):
        r = _result(difficult_phrases=[_phrase("no_existe")])
        result = check_coach_result(SOURCE, r)
        assert result.summary.strip() != ""


# ===========================================================================
# No Ollama calls — check_coach_result is synchronous and pure
# ===========================================================================

class TestNoPollution:
    def test_returns_immediately_no_io(self):
        """check_coach_result must return synchronously with no side effects."""
        import time
        start = time.monotonic()
        check_coach_result(SOURCE, _result())
        elapsed = time.monotonic() - start
        # Any result taking > 1 s would imply a network call.
        assert elapsed < 1.0
