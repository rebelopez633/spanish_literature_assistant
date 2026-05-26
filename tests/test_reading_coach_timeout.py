"""Tests for Phase 2 Slice 5 — timeout handling for coach-mode LLM calls.

Covers:
  1. READING_COACH_TIMEOUT_SECONDS drives CoachSettings.timeout_seconds.
  2. The UI make_ollama_client() routes through the typed adapter so
     requests.exceptions.Timeout becomes ReadingCoachTimeoutError.
  3. ReadingCoachTimeoutError carries a user-friendly message suitable
     for display in st.error().
  4. The Streamlit render function can distinguish timeout failures from
     parse failures (different except branches).

No real Ollama calls.  No long waits.  Existing translation workflow untouched.
"""
from __future__ import annotations

import unittest.mock as mock

import pytest
import requests

from reading_coach.errors import (
    ReadingCoachLLMError,
    ReadingCoachTimeoutError,
)


# ===========================================================================
# TestCoachTimeoutConfig — CoachSettings exposes READING_COACH_TIMEOUT_SECONDS
# ===========================================================================

class TestCoachTimeoutConfig:
    def test_coach_settings_has_timeout_seconds(self):
        from reading_coach.config import CoachSettings
        assert hasattr(CoachSettings, "__dataclass_fields__")
        assert "timeout_seconds" in CoachSettings.__dataclass_fields__

    def test_default_timeout_is_120(self, monkeypatch):
        monkeypatch.delenv("READING_COACH_TIMEOUT_SECONDS", raising=False)
        from reading_coach.config import get_coach_settings
        settings = get_coach_settings()
        assert settings.timeout_seconds == 120.0

    def test_reads_timeout_from_env(self, monkeypatch):
        monkeypatch.setenv("READING_COACH_TIMEOUT_SECONDS", "60")
        from reading_coach.config import get_coach_settings
        settings = get_coach_settings()
        assert settings.timeout_seconds == 60.0

    def test_fractional_timeout_accepted(self, monkeypatch):
        monkeypatch.setenv("READING_COACH_TIMEOUT_SECONDS", "45.5")
        from reading_coach.config import get_coach_settings
        settings = get_coach_settings()
        assert settings.timeout_seconds == 45.5

    def test_zero_timeout_falls_back_to_default(self, monkeypatch):
        """Zero is not a valid timeout — must fall back to the default."""
        monkeypatch.setenv("READING_COACH_TIMEOUT_SECONDS", "0")
        from reading_coach.config import get_coach_settings
        settings = get_coach_settings()
        assert settings.timeout_seconds == 120.0

    def test_negative_timeout_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("READING_COACH_TIMEOUT_SECONDS", "-30")
        from reading_coach.config import get_coach_settings
        settings = get_coach_settings()
        assert settings.timeout_seconds == 120.0

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("READING_COACH_TIMEOUT_SECONDS", "not-a-number")
        from reading_coach.config import get_coach_settings
        settings = get_coach_settings()
        assert settings.timeout_seconds == 120.0

    def test_whitespace_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("READING_COACH_TIMEOUT_SECONDS", "   ")
        from reading_coach.config import get_coach_settings
        settings = get_coach_settings()
        assert settings.timeout_seconds == 120.0

    def test_timeout_seconds_is_positive(self, monkeypatch):
        monkeypatch.delenv("READING_COACH_TIMEOUT_SECONDS", raising=False)
        from reading_coach.config import get_coach_settings
        assert get_coach_settings().timeout_seconds > 0

    def test_large_timeout_accepted(self, monkeypatch):
        """Very large timeouts (e.g., 3600s for slow hardware) should be accepted."""
        monkeypatch.setenv("READING_COACH_TIMEOUT_SECONDS", "3600")
        from reading_coach.config import get_coach_settings
        assert get_coach_settings().timeout_seconds == 3600.0

    def test_returns_coach_settings_instance(self, monkeypatch):
        monkeypatch.delenv("READING_COACH_TIMEOUT_SECONDS", raising=False)
        from reading_coach.config import CoachSettings, get_coach_settings
        assert isinstance(get_coach_settings(), CoachSettings)


# ===========================================================================
# TestMakeOllamaClientTimeoutWiring — UI helper routes through typed adapter
# ===========================================================================

class TestMakeOllamaClientTimeoutWiring:
    """make_ollama_client() in spanish_source_mode now uses the typed adapter,
    so network exceptions are mapped to ReadingCoachError subclasses."""

    _MESSAGES = [{"role": "user", "content": "Analiza esto."}]

    def _make_client_with_side_effect(self, side_effect, timeout=30.0):
        """Build a client whose underlying chat raises *side_effect*."""
        from ui.spanish_source_mode import make_ollama_client
        with mock.patch("reading_coach.llm_adapter.ollama_client.chat") as patched:
            patched.side_effect = side_effect
            client = make_ollama_client("http://h", "model", timeout)
        # patch is no longer active here but 'raw' was captured inside
        # make_ollama_coach_client at factory time — we must stay inside
        # the context; return a thin helper instead
        return client, patched

    def test_requests_timeout_raises_reading_coach_timeout_error(self):
        from ui.spanish_source_mode import make_ollama_client
        with mock.patch("reading_coach.llm_adapter.ollama_client.chat") as patched:
            patched.side_effect = requests.exceptions.Timeout("timed out")
            client = make_ollama_client("http://h", "model", 30.0)
            with pytest.raises(ReadingCoachTimeoutError):
                client(self._MESSAGES)

    def test_timeout_error_is_reading_coach_llm_error(self):
        """ReadingCoachTimeoutError is a ReadingCoachLLMError — same error hierarchy."""
        from ui.spanish_source_mode import make_ollama_client
        with mock.patch("reading_coach.llm_adapter.ollama_client.chat") as patched:
            patched.side_effect = requests.exceptions.Timeout("timed out")
            client = make_ollama_client("http://h", "model", 30.0)
            with pytest.raises(ReadingCoachLLMError):
                client(self._MESSAGES)

    def test_connection_error_raises_reading_coach_llm_error(self):
        from ui.spanish_source_mode import make_ollama_client
        with mock.patch("reading_coach.llm_adapter.ollama_client.chat") as patched:
            patched.side_effect = requests.exceptions.ConnectionError("refused")
            client = make_ollama_client("http://h", "model", 30.0)
            with pytest.raises(ReadingCoachLLMError):
                client(self._MESSAGES)

    def test_connection_error_is_not_timeout_error(self):
        """A connection error must NOT be confused with a timeout."""
        from ui.spanish_source_mode import make_ollama_client
        with mock.patch("reading_coach.llm_adapter.ollama_client.chat") as patched:
            patched.side_effect = requests.exceptions.ConnectionError("refused")
            client = make_ollama_client("http://h", "model", 30.0)
            with pytest.raises(ReadingCoachLLMError) as exc_info:
                client(self._MESSAGES)
            assert not isinstance(exc_info.value, ReadingCoachTimeoutError)

    def test_bare_requests_timeout_not_exposed(self):
        """The raw requests.Timeout must never propagate — it must be wrapped."""
        from ui.spanish_source_mode import make_ollama_client
        with mock.patch("reading_coach.llm_adapter.ollama_client.chat") as patched:
            patched.side_effect = requests.exceptions.Timeout("raw timeout")
            client = make_ollama_client("http://h", "model", 30.0)
            try:
                client(self._MESSAGES)
            except ReadingCoachTimeoutError:
                pass  # expected
            except requests.exceptions.Timeout:
                pytest.fail("Raw requests.Timeout leaked out — should be wrapped")

    def test_http_error_raises_reading_coach_llm_error(self):
        from ui.spanish_source_mode import make_ollama_client
        with mock.patch("reading_coach.llm_adapter.ollama_client.chat") as patched:
            patched.side_effect = requests.exceptions.HTTPError("500 Server Error")
            client = make_ollama_client("http://h", "model", 30.0)
            with pytest.raises(ReadingCoachLLMError):
                client(self._MESSAGES)

    def test_timeout_forwarded_to_raw_client(self):
        """The timeout value is forwarded to the underlying ollama_client.chat call."""
        from ui.spanish_source_mode import make_ollama_client
        with mock.patch("reading_coach.llm_adapter.ollama_client.chat") as patched:
            patched.return_value = '{"original_spanish":"x","overall_level":"B1"}'
            client = make_ollama_client("http://h", "model", 77.0)
            client(self._MESSAGES)
        assert patched.call_args[0][2] == 77.0

    def test_original_exception_chained(self):
        """The underlying requests error is preserved as __cause__."""
        from ui.spanish_source_mode import make_ollama_client
        cause = requests.exceptions.Timeout("connection timed out")
        with mock.patch("reading_coach.llm_adapter.ollama_client.chat") as patched:
            patched.side_effect = cause
            client = make_ollama_client("http://h", "model", 30.0)
            with pytest.raises(ReadingCoachTimeoutError) as exc_info:
                client(self._MESSAGES)
        assert exc_info.value.__cause__ is cause


# ===========================================================================
# TestTimeoutErrorUserMessage — user-friendly message for Streamlit display
# ===========================================================================

class TestTimeoutErrorUserMessage:
    def test_user_message_is_non_empty_string(self):
        exc = ReadingCoachTimeoutError("timed out after 120s")
        msg = exc.user_message
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_default_user_message_non_empty(self):
        exc = ReadingCoachTimeoutError()
        assert exc.user_message

    def test_user_message_mentions_time_concept(self):
        """The message must give the user actionable context about timeout."""
        exc = ReadingCoachTimeoutError()
        msg = exc.user_message.lower()
        assert any(word in msg for word in ("time", "timeout", "respond", "slow", "wait"))

    def test_explicit_message_surfaces_as_user_message(self):
        """A specific message supplied at raise-time is accessible via user_message."""
        exc = ReadingCoachTimeoutError("The model did not respond in 120 seconds.")
        assert "120" in exc.user_message

    def test_adapter_produced_timeout_has_user_message(self):
        """Timeout raised by the adapter carries a user_message (not just str)."""
        from ui.spanish_source_mode import make_ollama_client
        with mock.patch("reading_coach.llm_adapter.ollama_client.chat") as patched:
            patched.side_effect = requests.exceptions.Timeout("timed out")
            client = make_ollama_client("http://h", "model", 120.0)
            with pytest.raises(ReadingCoachTimeoutError) as exc_info:
                client([{"role": "user", "content": "x"}])
        assert exc_info.value.user_message

    def test_adapter_timeout_message_mentions_seconds(self):
        """The adapter bakes the timeout value into its error message."""
        from ui.spanish_source_mode import make_ollama_client
        with mock.patch("reading_coach.llm_adapter.ollama_client.chat") as patched:
            patched.side_effect = requests.exceptions.Timeout("timed out")
            client = make_ollama_client("http://h", "model", 120.0)
            with pytest.raises(ReadingCoachTimeoutError) as exc_info:
                client([{"role": "user", "content": "x"}])
        # The adapter message must reference the configured timeout value
        assert "120" in exc_info.value.user_message


# ===========================================================================
# TestStreamlitHandlerCompatibility — render_coach_mode can handle timeouts
# ===========================================================================

class TestStreamlitHandlerCompatibility:
    """Structural tests verifying the Streamlit handler can give clear timeout UX."""

    def test_timeout_error_not_a_coach_analysis_error(self):
        """ReadingCoachTimeoutError and CoachAnalysisError are disjoint — they
        require separate except branches in render_coach_mode."""
        from reading_coach.analyzer import CoachAnalysisError
        assert not issubclass(ReadingCoachTimeoutError, CoachAnalysisError)
        assert not issubclass(CoachAnalysisError, ReadingCoachTimeoutError)

    def test_timeout_error_catchable_by_specific_type(self):
        """Verify an except ReadingCoachTimeoutError branch works as expected."""
        caught = False
        try:
            raise ReadingCoachTimeoutError("took too long")
        except ReadingCoachTimeoutError:
            caught = True
        assert caught

    def test_timeout_error_catchable_as_llm_error(self):
        """render_coach_mode could also catch it via ReadingCoachLLMError if desired."""
        caught_as_llm = False
        try:
            raise ReadingCoachTimeoutError("took too long")
        except ReadingCoachLLMError:
            caught_as_llm = True
        assert caught_as_llm

    def test_render_coach_mode_imports_timeout_error(self):
        """render_coach_mode module must import ReadingCoachTimeoutError for the
        specific except clause to work at runtime."""
        import ui.spanish_source_mode as mod
        assert hasattr(mod, "ReadingCoachTimeoutError"), (
            "ReadingCoachTimeoutError must be imported in spanish_source_mode.py "
            "so render_coach_mode can catch it with a specific except branch."
        )

    def test_timeout_user_message_suitable_for_st_error(self):
        """The user_message text must be non-empty and devoid of raw exception noise."""
        exc = ReadingCoachTimeoutError()
        msg = exc.user_message
        # Should not expose raw Python exception repr to the user
        assert "requests.exceptions" not in msg
        assert len(msg.strip()) > 0

    def test_config_timeout_passable_to_make_ollama_client(self):
        """CoachSettings.timeout_seconds can be fed directly to make_ollama_client."""
        from reading_coach.config import get_coach_settings
        from ui.spanish_source_mode import make_ollama_client
        import inspect
        sig = inspect.signature(make_ollama_client)
        # Function accepts (host, model, timeout) — verify the call is accepted
        settings_timeout = get_coach_settings().timeout_seconds
        client = make_ollama_client("http://h", "model", settings_timeout)
        assert callable(client)
