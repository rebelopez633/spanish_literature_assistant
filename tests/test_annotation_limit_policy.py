"""Tests for Phase 2 Slice 6 — annotation limit policy.

Covers:
  1. Policy constants (POLICY_WARN, POLICY_TRUNCATE, POLICY_FAIL).
  2. apply_annotation_limit() pure function for each policy.
  3. Invalid policy falls back to "warn".
  4. Original ReadingCoachResult is never mutated (truncate creates a new object).
  5. check_coach_result uses the policy from CoachCheckerConfig.
  6. READING_COACH_ANNOTATION_LIMIT_POLICY env var drives CoachSettings.

No Ollama calls, no Streamlit, no I/O.
"""
from __future__ import annotations

import pytest

from reading_coach.schemas import DifficultPhrase, ReadingCoachResult
from reading_coach.checker import (
    STATUS_PASSED,
    STATUS_WARNING,
    STATUS_FAILED,
    CoachCheckerConfig,
    check_coach_result,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SOURCE = (
    "No hay mal que por bien no venga. "
    "Hemos topado con la Iglesia, Sancho."
)


def _phrase(text: str = "no hay mal") -> DifficultPhrase:
    return DifficultPhrase(
        phrase=text,
        category="expression",
        difficulty_level="B2",
        why_difficult="Idiomatic.",
    )


def _result(n_phrases: int = 0, phrases=None) -> ReadingCoachResult:
    """Build a ReadingCoachResult against SOURCE with *n_phrases* identical phrases."""
    dp = phrases if phrases is not None else [_phrase() for _ in range(n_phrases)]
    return ReadingCoachResult(original_spanish=SOURCE, overall_level="B2", difficult_phrases=dp)


# ===========================================================================
# TestPolicyConstants — module exports POLICY_* constants
# ===========================================================================

class TestPolicyConstants:
    def test_policy_warn_exists(self):
        from reading_coach.checker import POLICY_WARN
        assert isinstance(POLICY_WARN, str)

    def test_policy_truncate_exists(self):
        from reading_coach.checker import POLICY_TRUNCATE
        assert isinstance(POLICY_TRUNCATE, str)

    def test_policy_fail_exists(self):
        from reading_coach.checker import POLICY_FAIL
        assert isinstance(POLICY_FAIL, str)

    def test_policy_values_are_distinct(self):
        from reading_coach.checker import POLICY_WARN, POLICY_TRUNCATE, POLICY_FAIL
        assert len({POLICY_WARN, POLICY_TRUNCATE, POLICY_FAIL}) == 3

    def test_policy_warn_value(self):
        from reading_coach.checker import POLICY_WARN
        assert POLICY_WARN == "warn"

    def test_policy_truncate_value(self):
        from reading_coach.checker import POLICY_TRUNCATE
        assert POLICY_TRUNCATE == "truncate"

    def test_policy_fail_value(self):
        from reading_coach.checker import POLICY_FAIL
        assert POLICY_FAIL == "fail"


# ===========================================================================
# TestApplyAnnotationLimitUnderLimit — no issue when count ≤ max
# ===========================================================================

class TestApplyAnnotationLimitUnderLimit:
    def _call(self, n, max_a, policy="warn"):
        from reading_coach.checker import apply_annotation_limit
        result = _result(n_phrases=n)
        return apply_annotation_limit(result, max_a, policy)

    def test_returns_two_tuple(self):
        out = self._call(3, 5)
        assert isinstance(out, tuple)
        assert len(out) == 2

    def test_no_issue_when_under_limit(self):
        _, issue = self._call(3, 5)
        assert issue is None

    def test_no_issue_when_exactly_at_limit(self):
        _, issue = self._call(5, 5)
        assert issue is None

    def test_original_result_returned_unchanged(self):
        r = _result(n_phrases=3)
        from reading_coach.checker import apply_annotation_limit
        out_r, _ = apply_annotation_limit(r, 5, "warn")
        assert out_r is r

    def test_zero_phrases_no_issue(self):
        _, issue = self._call(0, 5)
        assert issue is None

    def test_all_policies_silent_when_under_limit(self):
        from reading_coach.checker import apply_annotation_limit
        for policy in ("warn", "truncate", "fail"):
            _, issue = apply_annotation_limit(_result(2), 5, policy)
            assert issue is None, f"policy={policy!r} emitted unexpected issue"


# ===========================================================================
# TestApplyAnnotationLimitWarnPolicy
# ===========================================================================

class TestApplyAnnotationLimitWarnPolicy:
    def _call(self, n=5, max_a=3):
        from reading_coach.checker import apply_annotation_limit
        result = _result(n_phrases=n)
        return result, apply_annotation_limit(result, max_a, "warn")

    def test_returns_original_result(self):
        r, (out_r, _) = self._call()
        assert out_r is r

    def test_phrases_not_truncated(self):
        r, (out_r, _) = self._call(n=5, max_a=3)
        assert len(out_r.difficult_phrases) == 5

    def test_issue_is_not_none(self):
        _, (_, issue) = self._call()
        assert issue is not None

    def test_issue_severity_is_warning(self):
        _, (_, issue) = self._call()
        severity, _ = issue
        assert severity == STATUS_WARNING

    def test_issue_message_mentions_count(self):
        _, (_, issue) = self._call(n=5, max_a=3)
        _, message = issue
        assert "5" in message or "annotation" in message.lower()

    def test_issue_message_mentions_limit(self):
        _, (_, issue) = self._call(n=5, max_a=3)
        _, message = issue
        assert "3" in message

    def test_original_spanish_preserved(self):
        r, (out_r, _) = self._call()
        assert out_r.original_spanish == SOURCE


# ===========================================================================
# TestApplyAnnotationLimitTruncatePolicy
# ===========================================================================

class TestApplyAnnotationLimitTruncatePolicy:
    def _call(self, n=6, max_a=4):
        from reading_coach.checker import apply_annotation_limit
        result = _result(n_phrases=n)
        return result, apply_annotation_limit(result, max_a, "truncate")

    def test_returns_new_object(self):
        """Truncate must produce a new result, not mutate the original."""
        r, (out_r, _) = self._call()
        assert out_r is not r

    def test_difficult_phrases_truncated_to_max(self):
        _, (out_r, _) = self._call(n=6, max_a=4)
        assert len(out_r.difficult_phrases) == 4

    def test_first_n_phrases_kept(self):
        """The first max_annotations phrases are retained in order."""
        phrases = [_phrase(f"phrase {i}") for i in range(6)]
        from reading_coach.checker import apply_annotation_limit
        r = _result(phrases=phrases)
        out_r, _ = apply_annotation_limit(r, 3, "truncate")
        for i in range(3):
            assert out_r.difficult_phrases[i].phrase == f"phrase {i}"

    def test_original_phrases_not_mutated(self):
        r, (out_r, _) = self._call(n=6, max_a=4)
        assert len(r.difficult_phrases) == 6  # unchanged

    def test_issue_is_not_none(self):
        _, (_, issue) = self._call()
        assert issue is not None

    def test_issue_severity_is_warning(self):
        _, (_, issue) = self._call()
        severity, _ = issue
        assert severity == STATUS_WARNING

    def test_original_spanish_preserved_in_new_result(self):
        _, (out_r, _) = self._call()
        assert out_r.original_spanish == SOURCE

    def test_other_fields_preserved(self):
        """overall_level and other metadata carry over to the truncated result."""
        _, (out_r, _) = self._call()
        assert out_r.overall_level == "B2"

    def test_truncate_to_zero_is_empty(self):
        from reading_coach.checker import apply_annotation_limit
        r = _result(n_phrases=3)
        out_r, issue = apply_annotation_limit(r, 0, "truncate")
        # max_annotations=0 means no phrases allowed; issue raised, phrases truncated
        assert len(out_r.difficult_phrases) == 0
        assert issue is not None


# ===========================================================================
# TestApplyAnnotationLimitFailPolicy
# ===========================================================================

class TestApplyAnnotationLimitFailPolicy:
    def _call(self, n=5, max_a=3):
        from reading_coach.checker import apply_annotation_limit
        result = _result(n_phrases=n)
        return result, apply_annotation_limit(result, max_a, "fail")

    def test_returns_original_result(self):
        r, (out_r, _) = self._call()
        assert out_r is r

    def test_phrases_not_truncated(self):
        r, (out_r, _) = self._call(n=5, max_a=3)
        assert len(out_r.difficult_phrases) == 5

    def test_issue_is_not_none(self):
        _, (_, issue) = self._call()
        assert issue is not None

    def test_issue_severity_is_failed(self):
        _, (_, issue) = self._call()
        severity, _ = issue
        assert severity == STATUS_FAILED

    def test_fail_severity_differs_from_warn(self):
        from reading_coach.checker import apply_annotation_limit
        r = _result(n_phrases=5)
        _, (_, warn_issue) = _result(5), apply_annotation_limit(r, 3, "warn")
        _, (_, fail_issue) = _result(5), apply_annotation_limit(r, 3, "fail")
        assert warn_issue[0] == STATUS_WARNING
        assert fail_issue[0] == STATUS_FAILED

    def test_issue_message_mentions_limit(self):
        _, (_, issue) = self._call(n=5, max_a=3)
        _, message = issue
        assert "3" in message

    def test_original_spanish_preserved(self):
        r, (out_r, _) = self._call()
        assert out_r.original_spanish == SOURCE


# ===========================================================================
# TestApplyAnnotationLimitInvalidPolicy — fallback to warn
# ===========================================================================

class TestApplyAnnotationLimitInvalidPolicy:
    def _call(self, policy):
        from reading_coach.checker import apply_annotation_limit
        result = _result(n_phrases=5)
        return result, apply_annotation_limit(result, 3, policy)

    def test_empty_string_falls_back_to_warn(self):
        r, (out_r, issue) = self._call("")
        assert issue is not None
        severity, _ = issue
        assert severity == STATUS_WARNING

    def test_unknown_string_falls_back_to_warn(self):
        r, (out_r, issue) = self._call("strict")
        assert issue is not None
        severity, _ = issue
        assert severity == STATUS_WARNING

    def test_uppercase_known_value_falls_back_to_warn(self):
        """Policy matching is case-sensitive; 'WARN' != 'warn'."""
        r, (out_r, issue) = self._call("WARN")
        assert issue is not None
        severity, _ = issue
        assert severity == STATUS_WARNING

    def test_original_result_returned_on_fallback(self):
        r, (out_r, _) = self._call("nonsense")
        assert out_r is r

    def test_phrases_not_truncated_on_fallback(self):
        r, (out_r, _) = self._call("nonsense")
        assert len(out_r.difficult_phrases) == 5


# ===========================================================================
# TestCheckCoachResultWithPolicy — integration through check_coach_result
# ===========================================================================

class TestCheckCoachResultWithPolicy:
    def _cfg(self, policy, max_a=3):
        return CoachCheckerConfig(max_annotations=max_a, annotation_limit_policy=policy)

    def _r(self, n):
        return _result(n_phrases=n)

    # warn policy --------------------------------------------------------

    def test_warn_policy_over_limit_is_warning(self):
        result = check_coach_result(SOURCE, self._r(5), config=self._cfg("warn"))
        assert result.status == STATUS_WARNING

    def test_warn_policy_under_limit_passes(self):
        result = check_coach_result(SOURCE, self._r(2), config=self._cfg("warn"))
        assert result.status == STATUS_PASSED

    # truncate policy ----------------------------------------------------

    def test_truncate_policy_over_limit_is_warning(self):
        result = check_coach_result(SOURCE, self._r(5), config=self._cfg("truncate"))
        assert result.status == STATUS_WARNING

    def test_truncate_policy_under_limit_passes(self):
        result = check_coach_result(SOURCE, self._r(2), config=self._cfg("truncate"))
        assert result.status == STATUS_PASSED

    # fail policy --------------------------------------------------------

    def test_fail_policy_over_limit_is_failed(self):
        result = check_coach_result(SOURCE, self._r(5), config=self._cfg("fail"))
        assert result.status == STATUS_FAILED

    def test_fail_policy_under_limit_passes(self):
        result = check_coach_result(SOURCE, self._r(2), config=self._cfg("fail"))
        assert result.status == STATUS_PASSED

    def test_fail_policy_issue_reported(self):
        result = check_coach_result(SOURCE, self._r(5), config=self._cfg("fail"))
        assert any("annotation" in i.lower() or "phrase" in i.lower() for i in result.issues)

    # default policy (warn) ----------------------------------------------

    def test_default_policy_is_warn(self):
        """CoachCheckerConfig() with no explicit policy defaults to warn behaviour."""
        r = self._r(5)
        result = check_coach_result(SOURCE, r, config=CoachCheckerConfig(max_annotations=3))
        assert result.status == STATUS_WARNING

    def test_fail_overrides_annotation_warning(self):
        """fail policy must elevate annotation-limit issue to STATUS_FAILED."""
        result = check_coach_result(SOURCE, self._r(10), config=self._cfg("fail", max_a=2))
        assert result.status == STATUS_FAILED


# ===========================================================================
# TestCoachCheckerConfigPolicy — policy field on CoachCheckerConfig
# ===========================================================================

class TestCoachCheckerConfigPolicy:
    def test_default_policy_is_warn(self):
        cfg = CoachCheckerConfig()
        assert cfg.annotation_limit_policy == "warn"

    def test_explicit_truncate_accepted(self):
        cfg = CoachCheckerConfig(annotation_limit_policy="truncate")
        assert cfg.annotation_limit_policy == "truncate"

    def test_explicit_fail_accepted(self):
        cfg = CoachCheckerConfig(annotation_limit_policy="fail")
        assert cfg.annotation_limit_policy == "fail"

    def test_explicit_warn_accepted(self):
        cfg = CoachCheckerConfig(annotation_limit_policy="warn")
        assert cfg.annotation_limit_policy == "warn"


# ===========================================================================
# TestAnnotationLimitPolicyConfig — READING_COACH_ANNOTATION_LIMIT_POLICY
# ===========================================================================

class TestAnnotationLimitPolicyConfig:
    def test_coach_settings_has_annotation_limit_policy(self):
        from reading_coach.config import CoachSettings
        assert "annotation_limit_policy" in CoachSettings.__dataclass_fields__

    def test_default_policy_is_warn(self, monkeypatch):
        monkeypatch.delenv("READING_COACH_ANNOTATION_LIMIT_POLICY", raising=False)
        from reading_coach.config import get_coach_settings
        assert get_coach_settings().annotation_limit_policy == "warn"

    def test_reads_truncate_from_env(self, monkeypatch):
        monkeypatch.setenv("READING_COACH_ANNOTATION_LIMIT_POLICY", "truncate")
        from reading_coach.config import get_coach_settings
        assert get_coach_settings().annotation_limit_policy == "truncate"

    def test_reads_fail_from_env(self, monkeypatch):
        monkeypatch.setenv("READING_COACH_ANNOTATION_LIMIT_POLICY", "fail")
        from reading_coach.config import get_coach_settings
        assert get_coach_settings().annotation_limit_policy == "fail"

    def test_reads_warn_from_env(self, monkeypatch):
        monkeypatch.setenv("READING_COACH_ANNOTATION_LIMIT_POLICY", "warn")
        from reading_coach.config import get_coach_settings
        assert get_coach_settings().annotation_limit_policy == "warn"

    def test_invalid_env_falls_back_to_warn(self, monkeypatch):
        monkeypatch.setenv("READING_COACH_ANNOTATION_LIMIT_POLICY", "strict")
        from reading_coach.config import get_coach_settings
        assert get_coach_settings().annotation_limit_policy == "warn"

    def test_empty_env_falls_back_to_warn(self, monkeypatch):
        monkeypatch.setenv("READING_COACH_ANNOTATION_LIMIT_POLICY", "")
        from reading_coach.config import get_coach_settings
        assert get_coach_settings().annotation_limit_policy == "warn"

    def test_uppercase_env_falls_back_to_warn(self, monkeypatch):
        """Policy parsing is case-sensitive; 'TRUNCATE' is not valid."""
        monkeypatch.setenv("READING_COACH_ANNOTATION_LIMIT_POLICY", "TRUNCATE")
        from reading_coach.config import get_coach_settings
        assert get_coach_settings().annotation_limit_policy == "warn"
