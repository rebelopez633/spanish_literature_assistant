"""Tests for reading_coach/retry.py — configurable retry policy.

All tests are pure unit tests with no real Ollama calls.
Sleep is replaced by a no-op lambda to keep the test suite fast.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from reading_coach.errors import (
    ReadingCoachLLMError,
    ReadingCoachParseError,
    ReadingCoachTimeoutError,
    ReadingCoachValidationError,
)

# ---------------------------------------------------------------------------
# Convenience: a sleep that does nothing (no real delay in tests)
# ---------------------------------------------------------------------------

_NO_SLEEP = lambda _: None  # noqa: E731

# ---------------------------------------------------------------------------
# Shared valid JSON payload for analyzer integration tests
# ---------------------------------------------------------------------------

_SOURCE = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme."
_VALID_PAYLOAD = {
    "original_spanish": _SOURCE,
    "overall_level": "B2",
    "modern_spanish": None,
    "english_gloss": None,
    "difficult_phrases": [],
    "grammar_notes": [],
    "comprehension_question": None,
}
_VALID_JSON = json.dumps(_VALID_PAYLOAD)


# ===========================================================================
# TestRetryConfigDefaults — default values are sane and conservative
# ===========================================================================

class TestRetryConfigDefaults:
    def test_importable(self):
        from reading_coach.retry import RetryConfig  # noqa: F401

    def test_default_max_retries_is_one(self):
        from reading_coach.retry import RetryConfig
        config = RetryConfig()
        assert config.max_retries == 1

    def test_default_max_retries_is_conservative(self):
        """Default must be ≤ 2 (conservative — do not hammer the model)."""
        from reading_coach.retry import RetryConfig
        assert RetryConfig().max_retries <= 2

    def test_default_retryable_contains_llm_error(self):
        from reading_coach.retry import RetryConfig
        config = RetryConfig()
        assert ReadingCoachLLMError in config.retryable

    def test_default_retryable_does_not_contain_parse_error(self):
        from reading_coach.retry import RetryConfig
        config = RetryConfig()
        assert ReadingCoachParseError not in config.retryable

    def test_default_retryable_does_not_contain_validation_error(self):
        from reading_coach.retry import RetryConfig
        config = RetryConfig()
        assert ReadingCoachValidationError not in config.retryable

    def test_default_backoff_base_positive(self):
        from reading_coach.retry import RetryConfig
        config = RetryConfig()
        assert config.backoff_base > 0

    def test_default_sleep_fn_is_none(self):
        """None means 'use time.sleep in production'."""
        from reading_coach.retry import RetryConfig
        config = RetryConfig()
        assert config.sleep_fn is None

    def test_timeout_error_matches_default_retryable(self):
        """ReadingCoachTimeoutError is a ReadingCoachLLMError — retried by default."""
        from reading_coach.retry import RetryConfig
        config = RetryConfig()
        exc = ReadingCoachTimeoutError("timed out")
        assert isinstance(exc, tuple(config.retryable))

    def test_custom_max_retries(self):
        from reading_coach.retry import RetryConfig
        config = RetryConfig(max_retries=3)
        assert config.max_retries == 3

    def test_zero_max_retries(self):
        from reading_coach.retry import RetryConfig
        config = RetryConfig(max_retries=0)
        assert config.max_retries == 0

    def test_custom_retryable_includes_parse_error(self):
        from reading_coach.retry import RetryConfig
        config = RetryConfig(retryable=(ReadingCoachLLMError, ReadingCoachParseError))
        assert ReadingCoachParseError in config.retryable

    def test_custom_sleep_fn_stored(self):
        from reading_coach.retry import RetryConfig
        fn = lambda _: None  # noqa: E731
        config = RetryConfig(sleep_fn=fn)
        assert config.sleep_fn is fn


# ===========================================================================
# TestGetRetryConfig — env-var driven factory
# ===========================================================================

class TestGetRetryConfig:
    def test_importable(self):
        from reading_coach.retry import get_retry_config  # noqa: F401

    def test_returns_retry_config_instance(self, monkeypatch):
        monkeypatch.delenv("READING_COACH_MAX_RETRIES", raising=False)
        from reading_coach.retry import RetryConfig, get_retry_config
        assert isinstance(get_retry_config(), RetryConfig)

    def test_default_when_env_absent(self, monkeypatch):
        monkeypatch.delenv("READING_COACH_MAX_RETRIES", raising=False)
        from reading_coach.retry import get_retry_config
        config = get_retry_config()
        assert config.max_retries == 1

    def test_reads_max_retries_from_env(self, monkeypatch):
        monkeypatch.setenv("READING_COACH_MAX_RETRIES", "3")
        from reading_coach.retry import get_retry_config
        assert get_retry_config().max_retries == 3

    def test_env_zero_disables_retry(self, monkeypatch):
        monkeypatch.setenv("READING_COACH_MAX_RETRIES", "0")
        from reading_coach.retry import get_retry_config
        assert get_retry_config().max_retries == 0

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("READING_COACH_MAX_RETRIES", "not-a-number")
        from reading_coach.retry import get_retry_config
        assert get_retry_config().max_retries == 1

    def test_negative_env_clamped_to_zero(self, monkeypatch):
        monkeypatch.setenv("READING_COACH_MAX_RETRIES", "-5")
        from reading_coach.retry import get_retry_config
        assert get_retry_config().max_retries == 0

    def test_whitespace_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("READING_COACH_MAX_RETRIES", "   ")
        from reading_coach.retry import get_retry_config
        assert get_retry_config().max_retries == 1


# ===========================================================================
# TestWithRetryImport
# ===========================================================================

class TestWithRetryImport:
    def test_importable(self):
        from reading_coach.retry import with_retry  # noqa: F401


# ===========================================================================
# TestWithRetrySuccess — callable succeeds (eventually)
# ===========================================================================

class TestWithRetrySuccess:
    def test_returns_value_on_first_attempt(self):
        from reading_coach.retry import RetryConfig, with_retry
        func = MagicMock(return_value=42)
        config = RetryConfig(max_retries=2, sleep_fn=_NO_SLEEP)
        result = with_retry(func, config=config)
        assert result == 42
        assert func.call_count == 1

    def test_returns_value_on_second_attempt(self):
        from reading_coach.retry import RetryConfig, with_retry
        func = MagicMock(side_effect=[ReadingCoachLLMError("fail"), "success"])
        config = RetryConfig(max_retries=1, sleep_fn=_NO_SLEEP)
        result = with_retry(func, config=config)
        assert result == "success"
        assert func.call_count == 2

    def test_returns_value_on_third_attempt(self):
        from reading_coach.retry import RetryConfig, with_retry
        func = MagicMock(side_effect=[
            ReadingCoachLLMError("fail1"),
            ReadingCoachLLMError("fail2"),
            "final success",
        ])
        config = RetryConfig(max_retries=2, sleep_fn=_NO_SLEEP)
        result = with_retry(func, config=config)
        assert result == "final success"
        assert func.call_count == 3

    def test_timeout_subclass_is_retried(self):
        from reading_coach.retry import RetryConfig, with_retry
        func = MagicMock(side_effect=[ReadingCoachTimeoutError("slow"), "ok"])
        config = RetryConfig(max_retries=1, sleep_fn=_NO_SLEEP)
        result = with_retry(func, config=config)
        assert result == "ok"


# ===========================================================================
# TestWithRetryExhausted — callable always fails
# ===========================================================================

class TestWithRetryExhausted:
    def test_raises_after_exhausting_retries(self):
        from reading_coach.retry import RetryConfig, with_retry
        func = MagicMock(side_effect=ReadingCoachLLMError("always fails"))
        config = RetryConfig(max_retries=2, sleep_fn=_NO_SLEEP)
        with pytest.raises(ReadingCoachLLMError):
            with_retry(func, config=config)

    def test_func_called_max_retries_plus_one_times(self):
        from reading_coach.retry import RetryConfig, with_retry
        func = MagicMock(side_effect=ReadingCoachLLMError("always fails"))
        config = RetryConfig(max_retries=2, sleep_fn=_NO_SLEEP)
        with pytest.raises(ReadingCoachLLMError):
            with_retry(func, config=config)
        assert func.call_count == 3  # 1 initial + 2 retries

    def test_zero_retries_calls_func_exactly_once(self):
        from reading_coach.retry import RetryConfig, with_retry
        func = MagicMock(side_effect=ReadingCoachLLMError("fail"))
        config = RetryConfig(max_retries=0, sleep_fn=_NO_SLEEP)
        with pytest.raises(ReadingCoachLLMError):
            with_retry(func, config=config)
        assert func.call_count == 1

    def test_raises_last_exception_instance(self):
        """The re-raised exception is the last one, preserving its message."""
        from reading_coach.retry import RetryConfig, with_retry
        func = MagicMock(side_effect=[
            ReadingCoachLLMError("first fail"),
            ReadingCoachLLMError("second fail"),
        ])
        config = RetryConfig(max_retries=1, sleep_fn=_NO_SLEEP)
        with pytest.raises(ReadingCoachLLMError, match="second fail"):
            with_retry(func, config=config)


# ===========================================================================
# TestWithRetryNonRetryable — non-retryable exceptions propagate immediately
# ===========================================================================

class TestWithRetryNonRetryable:
    def test_parse_error_not_retried_by_default(self):
        from reading_coach.retry import RetryConfig, with_retry
        func = MagicMock(side_effect=ReadingCoachParseError("bad json"))
        config = RetryConfig(max_retries=5, sleep_fn=_NO_SLEEP)
        with pytest.raises(ReadingCoachParseError):
            with_retry(func, config=config)
        assert func.call_count == 1  # immediate re-raise

    def test_validation_error_not_retried_by_default(self):
        from reading_coach.retry import RetryConfig, with_retry
        func = MagicMock(side_effect=ReadingCoachValidationError("bad schema"))
        config = RetryConfig(max_retries=5, sleep_fn=_NO_SLEEP)
        with pytest.raises(ReadingCoachValidationError):
            with_retry(func, config=config)
        assert func.call_count == 1

    def test_arbitrary_exception_not_retried(self):
        from reading_coach.retry import RetryConfig, with_retry
        func = MagicMock(side_effect=RuntimeError("unexpected"))
        config = RetryConfig(max_retries=3, sleep_fn=_NO_SLEEP)
        with pytest.raises(RuntimeError):
            with_retry(func, config=config)
        assert func.call_count == 1

    def test_parse_error_retried_when_in_retryable(self):
        from reading_coach.retry import RetryConfig, with_retry
        func = MagicMock(side_effect=ReadingCoachParseError("bad json"))
        config = RetryConfig(
            max_retries=2,
            retryable=(ReadingCoachLLMError, ReadingCoachParseError),
            sleep_fn=_NO_SLEEP,
        )
        with pytest.raises(ReadingCoachParseError):
            with_retry(func, config=config)
        assert func.call_count == 3  # retried when explicitly configured


# ===========================================================================
# TestWithRetrySleepBehavior — sleep injection and backoff
# ===========================================================================

class TestWithRetrySleepBehavior:
    def test_sleep_called_between_retries(self):
        from reading_coach.retry import RetryConfig, with_retry
        sleep_calls: list[float] = []
        func = MagicMock(side_effect=[ReadingCoachLLMError("fail"), "ok"])
        config = RetryConfig(max_retries=1, sleep_fn=lambda d: sleep_calls.append(d))
        with_retry(func, config=config)
        assert len(sleep_calls) == 1

    def test_sleep_not_called_on_first_attempt_success(self):
        from reading_coach.retry import RetryConfig, with_retry
        sleep_calls: list[float] = []
        func = MagicMock(return_value="ok")
        config = RetryConfig(max_retries=2, sleep_fn=lambda d: sleep_calls.append(d))
        with_retry(func, config=config)
        assert sleep_calls == []

    def test_sleep_not_called_after_last_failed_attempt(self):
        """No sleep after the final exhausted attempt (only sleep *between* attempts)."""
        from reading_coach.retry import RetryConfig, with_retry
        sleep_calls: list[float] = []
        func = MagicMock(side_effect=ReadingCoachLLMError("always fails"))
        config = RetryConfig(max_retries=2, sleep_fn=lambda d: sleep_calls.append(d))
        with pytest.raises(ReadingCoachLLMError):
            with_retry(func, config=config)
        # max_retries=2 → 3 total attempts → sleep after attempt 0 and 1 (not 2)
        assert len(sleep_calls) == 2

    def test_sleep_uses_exponential_backoff(self):
        from reading_coach.retry import RetryConfig, with_retry
        sleep_calls: list[float] = []
        func = MagicMock(side_effect=ReadingCoachLLMError("always fails"))
        config = RetryConfig(
            max_retries=3,
            backoff_base=1.0,
            sleep_fn=lambda d: sleep_calls.append(d),
        )
        with pytest.raises(ReadingCoachLLMError):
            with_retry(func, config=config)
        # delays: 1.0*2^0=1.0, 1.0*2^1=2.0, 1.0*2^2=4.0 (not after final attempt)
        assert sleep_calls == [1.0, 2.0, 4.0]

    def test_backoff_base_scales_delay(self):
        from reading_coach.retry import RetryConfig, with_retry
        sleep_calls: list[float] = []
        func = MagicMock(side_effect=ReadingCoachLLMError("fail"))
        config = RetryConfig(max_retries=1, backoff_base=2.0, sleep_fn=lambda d: sleep_calls.append(d))
        with pytest.raises(ReadingCoachLLMError):
            with_retry(func, config=config)
        assert sleep_calls == [2.0]  # 2.0 * 2^0 = 2.0

    def test_no_real_delay_with_injected_noop(self):
        """Verifies the no-op sleep keeps the test fast (< 0.5s)."""
        import time
        from reading_coach.retry import RetryConfig, with_retry
        func = MagicMock(side_effect=[ReadingCoachLLMError("fail"), "ok"])
        config = RetryConfig(max_retries=1, backoff_base=60.0, sleep_fn=_NO_SLEEP)
        t0 = time.monotonic()
        with_retry(func, config=config)
        assert time.monotonic() - t0 < 0.5


# ===========================================================================
# TestAnalyzerRetryIntegration — analyze_spanish_source uses retry_config
# ===========================================================================

class TestAnalyzerRetryIntegration:
    """Integration tests: analyze_spanish_source honours retry_config."""

    def test_retry_config_param_accepted(self):
        """analyze_spanish_source accepts a retry_config keyword argument."""
        import inspect
        from reading_coach.analyzer import analyze_spanish_source
        sig = inspect.signature(analyze_spanish_source)
        assert "retry_config" in sig.parameters

    def test_success_with_no_retry(self):
        from reading_coach.analyzer import AnalysisResult, analyze_spanish_source
        from reading_coach.retry import RetryConfig
        client = MagicMock(return_value=_VALID_JSON)
        config = RetryConfig(max_retries=0, sleep_fn=_NO_SLEEP)
        ar = analyze_spanish_source(_SOURCE, llm_client=client, retry_config=config)
        assert isinstance(ar, AnalysisResult)
        assert client.call_count == 1

    def test_llm_error_retried_and_succeeds(self):
        """Callable fails once with ReadingCoachLLMError, then succeeds."""
        from reading_coach.analyzer import AnalysisResult, analyze_spanish_source
        from reading_coach.retry import RetryConfig
        client = MagicMock(side_effect=[ReadingCoachLLMError("timeout"), _VALID_JSON])
        config = RetryConfig(max_retries=1, sleep_fn=_NO_SLEEP)
        ar = analyze_spanish_source(_SOURCE, llm_client=client, retry_config=config)
        assert isinstance(ar, AnalysisResult)
        assert client.call_count == 2

    def test_timeout_error_retried_and_succeeds(self):
        """ReadingCoachTimeoutError (subclass of LLMError) is also retried."""
        from reading_coach.analyzer import AnalysisResult, analyze_spanish_source
        from reading_coach.retry import RetryConfig
        client = MagicMock(side_effect=[ReadingCoachTimeoutError("slow"), _VALID_JSON])
        config = RetryConfig(max_retries=1, sleep_fn=_NO_SLEEP)
        ar = analyze_spanish_source(_SOURCE, llm_client=client, retry_config=config)
        assert isinstance(ar, AnalysisResult)
        assert client.call_count == 2

    def test_llm_always_fails_raises_llm_error(self):
        """After all retries exhausted, ReadingCoachLLMError propagates (not wrapped)."""
        from reading_coach.analyzer import analyze_spanish_source
        from reading_coach.retry import RetryConfig
        client = MagicMock(side_effect=ReadingCoachLLMError("connection refused"))
        config = RetryConfig(max_retries=2, sleep_fn=_NO_SLEEP)
        with pytest.raises(ReadingCoachLLMError):
            analyze_spanish_source(_SOURCE, llm_client=client, retry_config=config)
        assert client.call_count == 3  # initial + 2 retries

    def test_llm_error_not_raised_as_coach_analysis_error(self):
        """ReadingCoachLLMError must propagate as-is, not wrapped in CoachAnalysisError."""
        from reading_coach.analyzer import CoachAnalysisError, analyze_spanish_source
        from reading_coach.retry import RetryConfig
        client = MagicMock(side_effect=ReadingCoachLLMError("fail"))
        config = RetryConfig(max_retries=0, sleep_fn=_NO_SLEEP)
        with pytest.raises(ReadingCoachLLMError):
            analyze_spanish_source(_SOURCE, llm_client=client, retry_config=config)
        # Verify it is NOT a CoachAnalysisError
        try:
            analyze_spanish_source(_SOURCE, llm_client=client, retry_config=config)
        except ReadingCoachLLMError as exc:
            assert not isinstance(exc, CoachAnalysisError)
        except Exception:
            pass

    def test_parse_failure_not_retried_by_default(self):
        """Bad JSON is a parse failure — not retried with default retryable=(LLMError,)."""
        from reading_coach.analyzer import CoachAnalysisError, analyze_spanish_source
        from reading_coach.retry import RetryConfig
        client = MagicMock(return_value="not json at all")
        config = RetryConfig(max_retries=2, sleep_fn=_NO_SLEEP)
        with pytest.raises(CoachAnalysisError):
            analyze_spanish_source(_SOURCE, llm_client=client, retry_config=config)
        assert client.call_count == 1  # not retried

    def test_parse_failure_retried_when_configured(self):
        """Parse failure IS retried when ReadingCoachParseError is in retryable."""
        from reading_coach.analyzer import CoachAnalysisError, analyze_spanish_source
        from reading_coach.retry import RetryConfig
        client = MagicMock(return_value="not json at all")
        config = RetryConfig(
            max_retries=2,
            retryable=(ReadingCoachLLMError, ReadingCoachParseError),
            sleep_fn=_NO_SLEEP,
        )
        with pytest.raises(CoachAnalysisError):
            analyze_spanish_source(_SOURCE, llm_client=client, retry_config=config)
        assert client.call_count == 3  # initial + 2 retries (parse error retried)

    def test_validation_failure_not_retried_by_default(self):
        """Missing required field → ReadingCoachValidationError — not retried by default."""
        from reading_coach.analyzer import CoachAnalysisError, analyze_spanish_source
        from reading_coach.retry import RetryConfig
        bad_payload = {"overall_level": "B2"}  # missing required original_spanish
        client = MagicMock(return_value=json.dumps(bad_payload))
        config = RetryConfig(max_retries=2, sleep_fn=_NO_SLEEP)
        with pytest.raises(CoachAnalysisError):
            analyze_spanish_source(_SOURCE, llm_client=client, retry_config=config)
        assert client.call_count == 1

    def test_no_real_sleep_in_analyzer_retry(self):
        """sleep_fn=_NO_SLEEP eliminates real delays even with large backoff_base."""
        import time
        from reading_coach.analyzer import analyze_spanish_source
        from reading_coach.retry import RetryConfig
        client = MagicMock(side_effect=[ReadingCoachLLMError("fail"), _VALID_JSON])
        config = RetryConfig(max_retries=1, backoff_base=60.0, sleep_fn=_NO_SLEEP)
        t0 = time.monotonic()
        analyze_spanish_source(_SOURCE, llm_client=client, retry_config=config)
        assert time.monotonic() - t0 < 0.5

    def test_retry_config_none_uses_env_default(self, monkeypatch):
        """retry_config=None delegates to get_retry_config() (env-configured defaults)."""
        monkeypatch.setenv("READING_COACH_MAX_RETRIES", "0")
        from reading_coach.analyzer import analyze_spanish_source
        client = MagicMock(side_effect=ReadingCoachLLMError("fail"))
        with pytest.raises(ReadingCoachLLMError):
            analyze_spanish_source(_SOURCE, llm_client=client)
        # With max_retries=0, only one attempt is made
        assert client.call_count == 1

    def test_existing_tests_unaffected_bad_json(self):
        """Existing behavior: bad JSON → CoachAnalysisError (no retry, parse error)."""
        from reading_coach.analyzer import CoachAnalysisError, analyze_spanish_source
        from reading_coach.retry import RetryConfig
        config = RetryConfig(max_retries=0, sleep_fn=_NO_SLEEP)

        def bad_client(messages):  # noqa: ARG001
            return "not json"

        with pytest.raises(CoachAnalysisError, match=r"(?i)(parse|ReadingCoachResult|invalid)"):
            analyze_spanish_source(_SOURCE, llm_client=bad_client, retry_config=config)
