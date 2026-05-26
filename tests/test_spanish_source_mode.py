"""Tests for ui/spanish_source_mode.py — pure helper functions only.

All Streamlit rendering (render_coach_mode, _display_coach_result) is
exercised manually; only the unit-testable make_ollama_client() factory
and module-level constants are verified here.

The Streamlit stub installed by conftest.py allows importing the module
without a running Streamlit server.
"""
from __future__ import annotations

import unittest.mock as mock

import pytest

from ui.spanish_source_mode import DENSITY_LABELS, make_ollama_client
from reading_coach.schemas import VALID_COACH_LEVELS


# ---------------------------------------------------------------------------
# TestDensityLabels
# ---------------------------------------------------------------------------

class TestDensityLabels:
    def test_has_three_labels(self):
        assert len(DENSITY_LABELS) == 3

    def test_contains_minimal(self):
        assert "minimal" in DENSITY_LABELS

    def test_contains_balanced(self):
        assert "balanced" in DENSITY_LABELS

    def test_contains_detailed(self):
        assert "detailed" in DENSITY_LABELS

    def test_default_balanced_is_second(self):
        assert DENSITY_LABELS[1] == "balanced"

    def test_labels_are_strings(self):
        for label in DENSITY_LABELS:
            assert isinstance(label, str)


# ---------------------------------------------------------------------------
# TestValidCoachLevelsCompatibility
# ---------------------------------------------------------------------------

class TestValidCoachLevelsCompatibility:
    """Verify the levels the UI offers match the schema constants."""

    def test_b1_is_available(self):
        assert "B1" in VALID_COACH_LEVELS

    def test_all_cefr_levels_present(self):
        for level in ("A1", "A2", "B1", "B2", "C1", "C2"):
            assert level in VALID_COACH_LEVELS

    def test_b1_has_valid_index(self):
        levels = list(VALID_COACH_LEVELS)
        assert levels.index("B1") >= 0


# ---------------------------------------------------------------------------
# TestMakeOllamaClient
# ---------------------------------------------------------------------------

class TestMakeOllamaClientReturnType:
    def test_returns_callable(self):
        client = make_ollama_client("http://localhost:11434", "qwen2.5:7b", 30.0)
        assert callable(client)

    def test_distinct_factories_return_distinct_callables(self):
        c1 = make_ollama_client("http://host-a:11434", "model-a", 10.0)
        c2 = make_ollama_client("http://host-b:11434", "model-b", 20.0)
        assert c1 is not c2


class TestMakeOllamaClientPayload:
    """Verify the factory builds the correct Ollama /api/chat payload."""

    _MESSAGES = [
        {"role": "system", "content": "You are a coach."},
        {"role": "user", "content": "Analyse this."},
    ]

    def _call_with_mock(self, host, model, timeout, messages=None):
        """Call the factory, invoke the client, capture the payload sent to chat."""
        if messages is None:
            messages = self._MESSAGES
        with mock.patch("ui.spanish_source_mode.ollama_client.chat") as patched:
            patched.return_value = '{"original_spanish":"x","overall_level":"B1"}'
            client = make_ollama_client(host, model, timeout)
            client(messages)
            return patched

    def test_chat_is_called_once(self):
        patched = self._call_with_mock("http://localhost:11434", "qwen2.5:7b", 30.0)
        patched.assert_called_once()

    def test_host_passed_to_chat(self):
        host = "http://myhost:11434"
        patched = self._call_with_mock(host, "qwen2.5:7b", 30.0)
        call_host = patched.call_args[0][0]
        assert call_host == host

    def test_payload_has_model_key(self):
        patched = self._call_with_mock("http://localhost:11434", "qwen2.5:7b", 30.0)
        payload = patched.call_args[0][1]
        assert "model" in payload

    def test_payload_model_matches_factory_arg(self):
        model = "qwen2.5:14b"
        patched = self._call_with_mock("http://localhost:11434", model, 30.0)
        payload = patched.call_args[0][1]
        assert payload["model"] == model

    def test_payload_messages_match_input(self):
        patched = self._call_with_mock("http://localhost:11434", "qwen2.5:7b", 30.0)
        payload = patched.call_args[0][1]
        assert payload["messages"] == self._MESSAGES

    def test_payload_format_is_json(self):
        patched = self._call_with_mock("http://localhost:11434", "qwen2.5:7b", 30.0)
        payload = patched.call_args[0][1]
        assert payload.get("format") == "json"

    def test_payload_stream_is_false(self):
        patched = self._call_with_mock("http://localhost:11434", "qwen2.5:7b", 30.0)
        payload = patched.call_args[0][1]
        assert payload.get("stream") is False

    def test_timeout_passed_to_chat(self):
        timeout = 42.0
        patched = self._call_with_mock("http://localhost:11434", "qwen2.5:7b", timeout)
        call_timeout = patched.call_args[0][2]
        assert call_timeout == timeout

    def test_client_returns_chat_result(self):
        expected = '{"original_spanish":"test","overall_level":"B2"}'
        with mock.patch("ui.spanish_source_mode.ollama_client.chat") as patched:
            patched.return_value = expected
            client = make_ollama_client("http://localhost:11434", "qwen2.5:7b", 30.0)
            result = client(self._MESSAGES)
        assert result == expected

    def test_different_messages_produce_different_payloads(self):
        msgs_a = [{"role": "user", "content": "Text A"}]
        msgs_b = [{"role": "user", "content": "Text B"}]
        with mock.patch("ui.spanish_source_mode.ollama_client.chat") as patched:
            patched.return_value = "{}"
            client = make_ollama_client("http://localhost:11434", "qwen2.5:7b", 30.0)
            client(msgs_a)
            payload_a = patched.call_args[0][1]
            client(msgs_b)
            payload_b = patched.call_args[0][1]
        assert payload_a["messages"] != payload_b["messages"]
