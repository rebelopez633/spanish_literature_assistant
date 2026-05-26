"""Tests for reading_coach/llm_adapter.py.

All underlying transport is replaced with simple lambdas or unittest.mock —
no real Ollama process is required.

Coverage:
  - OllamaResponseError is raised for unrecognised response shapes.
  - Plain-string responses are passed through unchanged.
  - Dict responses with message/content shape are normalised to str.
  - Dict responses with top-level content key are normalised to str.
  - Messages list is forwarded to the raw client unchanged.
  - Payload includes expected keys (model, stream, format).
  - Optional ``options`` dict is included / excluded correctly.
  - host and timeout are forwarded to the raw client.
  - Factory returns a callable each time.
  - Default raw client falls back to infrastructure.ollama_client.chat.
"""
from __future__ import annotations

import inspect
import unittest.mock as mock

import pytest

from reading_coach.llm_adapter import OllamaResponseError, make_ollama_coach_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_raw(return_value):
    """Return a raw-client stub that ignores all args and returns *return_value*."""

    def _raw(host, payload, timeout):
        return return_value

    return _raw


def _capture_call(model="qwen2.5:7b", host="http://localhost:11434", timeout=30.0,
                  options=None, messages=None):
    """Call the factory, invoke the returned client, and return a dict of what
    the raw client received plus what the adapter returned."""
    if messages is None:
        messages = [{"role": "user", "content": "Analiza."}]

    captured: dict = {}

    def _raw(h, payload, t):
        captured["host"] = h
        captured["payload"] = payload
        captured["timeout"] = t
        return "response_string"

    client = make_ollama_coach_client(host, model, timeout, options=options, _raw_client=_raw)
    captured["result"] = client(messages)
    return captured


# ---------------------------------------------------------------------------
# TestOllamaResponseError
# ---------------------------------------------------------------------------

class TestOllamaResponseError:
    """OllamaResponseError is a ValueError subclass with a descriptive message."""

    def test_is_value_error_subclass(self):
        assert issubclass(OllamaResponseError, ValueError)

    def test_can_be_raised_with_message(self):
        with pytest.raises(OllamaResponseError, match="bad shape"):
            raise OllamaResponseError("bad shape")


# ---------------------------------------------------------------------------
# TestResponseNormalisation
# ---------------------------------------------------------------------------

class TestResponseNormalisation:
    """Adapter normalises whatever the raw client returns into a plain str."""

    # --- Test 1: plain string passthrough -----------------------------------

    def test_plain_string_returned_unchanged(self):
        expected = '{"original_spanish": "Hola mundo"}'
        client = make_ollama_coach_client(
            "http://h", "m", 10.0,
            _raw_client=_fake_raw(expected),
        )
        assert client([{"role": "user", "content": "x"}]) == expected

    def test_empty_string_returned_unchanged(self):
        client = make_ollama_coach_client("http://h", "m", 10.0, _raw_client=_fake_raw(""))
        assert client([{"role": "user", "content": "x"}]) == ""

    # --- Test 2: dict with message/content shape ----------------------------

    def test_dict_message_content_extracted(self):
        payload = {"message": {"content": '{"original_spanish":"Cervantes"}'}}
        client = make_ollama_coach_client("http://h", "m", 10.0, _raw_client=_fake_raw(payload))
        result = client([{"role": "user", "content": "x"}])
        assert result == '{"original_spanish":"Cervantes"}'

    def test_dict_message_content_empty_string_extracted(self):
        payload = {"message": {"content": ""}}
        client = make_ollama_coach_client("http://h", "m", 10.0, _raw_client=_fake_raw(payload))
        assert client([{"role": "user", "content": "x"}]) == ""

    # --- Test 2b: dict with top-level content key ---------------------------

    def test_dict_top_level_content_extracted(self):
        payload = {"content": '{"original_spanish":"Quijote"}'}
        client = make_ollama_coach_client("http://h", "m", 10.0, _raw_client=_fake_raw(payload))
        assert client([{"role": "user", "content": "x"}]) == '{"original_spanish":"Quijote"}'

    # --- Test 3: unexpected shapes raise OllamaResponseError ---------------

    def test_integer_response_raises_error(self):
        client = make_ollama_coach_client("http://h", "m", 10.0, _raw_client=_fake_raw(42))
        with pytest.raises(OllamaResponseError):
            client([{"role": "user", "content": "x"}])

    def test_none_response_raises_error(self):
        client = make_ollama_coach_client("http://h", "m", 10.0, _raw_client=_fake_raw(None))
        with pytest.raises(OllamaResponseError):
            client([{"role": "user", "content": "x"}])

    def test_list_response_raises_error(self):
        client = make_ollama_coach_client(
            "http://h", "m", 10.0,
            _raw_client=_fake_raw([{"message": "oops"}]),
        )
        with pytest.raises(OllamaResponseError):
            client([{"role": "user", "content": "x"}])

    def test_dict_without_recognised_keys_raises_error(self):
        client = make_ollama_coach_client(
            "http://h", "m", 10.0,
            _raw_client=_fake_raw({"unexpected_key": "some_value"}),
        )
        with pytest.raises(OllamaResponseError):
            client([{"role": "user", "content": "x"}])

    def test_error_message_includes_type_name(self):
        client = make_ollama_coach_client("http://h", "m", 10.0, _raw_client=_fake_raw(99))
        with pytest.raises(OllamaResponseError, match="int"):
            client([{"role": "user", "content": "x"}])

    def test_error_message_for_unrecognised_dict(self):
        client = make_ollama_coach_client(
            "http://h", "m", 10.0,
            _raw_client=_fake_raw({"foo": "bar"}),
        )
        with pytest.raises(OllamaResponseError, match="dict"):
            client([{"role": "user", "content": "x"}])


# ---------------------------------------------------------------------------
# TestMessageForwarding
# ---------------------------------------------------------------------------

class TestMessageForwarding:
    """Adapter forwards messages list unchanged to the raw client."""

    # --- Test 4: messages forwarded unchanged -------------------------------

    def test_messages_in_payload_match_input(self):
        msgs = [
            {"role": "system", "content": "You are a coach."},
            {"role": "user", "content": "Analiza este texto."},
        ]
        c = _capture_call(messages=msgs)
        assert c["payload"]["messages"] == msgs

    def test_empty_messages_list_forwarded(self):
        c = _capture_call(messages=[])
        assert c["payload"]["messages"] == []

    def test_single_message_forwarded(self):
        msgs = [{"role": "user", "content": "Hola"}]
        c = _capture_call(messages=msgs)
        assert c["payload"]["messages"] == msgs

    def test_messages_not_mutated_by_adapter(self):
        msgs = [{"role": "user", "content": "original"}]
        original_copy = [dict(m) for m in msgs]
        _capture_call(messages=msgs)
        assert msgs == original_copy

    def test_multiple_calls_use_respective_messages(self):
        captured_payloads: list[dict] = []

        def _raw(host, payload, timeout):
            captured_payloads.append(payload.copy())
            return "ok"

        client = make_ollama_coach_client("http://h", "model", 10.0, _raw_client=_raw)
        msgs_a = [{"role": "user", "content": "A"}]
        msgs_b = [{"role": "user", "content": "B"}]
        client(msgs_a)
        client(msgs_b)
        assert captured_payloads[0]["messages"] == msgs_a
        assert captured_payloads[1]["messages"] == msgs_b


# ---------------------------------------------------------------------------
# TestPayloadShape
# ---------------------------------------------------------------------------

class TestPayloadShape:
    """Adapter builds a well-formed Ollama /api/chat payload."""

    def test_payload_contains_model_key(self):
        c = _capture_call(model="qwen2.5:7b")
        assert "model" in c["payload"]

    def test_payload_model_matches_factory_arg(self):
        c = _capture_call(model="qwen2.5:14b")
        assert c["payload"]["model"] == "qwen2.5:14b"

    def test_payload_stream_is_false(self):
        c = _capture_call()
        assert c["payload"]["stream"] is False

    def test_payload_format_is_json(self):
        c = _capture_call()
        assert c["payload"]["format"] == "json"

    def test_host_forwarded_to_raw_client(self):
        c = _capture_call(host="http://myhost:11434")
        assert c["host"] == "http://myhost:11434"

    def test_timeout_forwarded_to_raw_client(self):
        c = _capture_call(timeout=99.5)
        assert c["timeout"] == 99.5

    # --- Test 5: optional options -------------------------------------------

    def test_no_options_key_absent_when_none(self):
        c = _capture_call(options=None)
        assert "options" not in c["payload"]

    def test_options_included_when_provided(self):
        opts = {"temperature": 0.1, "num_ctx": 8192}
        c = _capture_call(options=opts)
        assert c["payload"]["options"] == opts

    def test_empty_options_dict_not_included(self):
        # Empty dict is falsy — should behave like None
        c = _capture_call(options={})
        assert "options" not in c["payload"]


# ---------------------------------------------------------------------------
# TestClientIsCallable
# ---------------------------------------------------------------------------

class TestClientIsCallable:
    """Factory always returns a callable suitable for injection."""

    def test_returns_callable(self):
        client = make_ollama_coach_client(
            "http://h", "m", 10.0,
            _raw_client=_fake_raw("ok"),
        )
        assert callable(client)

    def test_distinct_factories_are_distinct_callables(self):
        raw = _fake_raw("ok")
        c1 = make_ollama_coach_client("http://a", "model-a", 10.0, _raw_client=raw)
        c2 = make_ollama_coach_client("http://b", "model-b", 20.0, _raw_client=raw)
        assert c1 is not c2

    def test_return_value_is_str(self):
        client = make_ollama_coach_client(
            "http://h", "m", 10.0,
            _raw_client=_fake_raw("some json string"),
        )
        result = client([{"role": "user", "content": "test"}])
        assert isinstance(result, str)

    def test_llm_client_parameter_exists_on_analyzer(self):
        """Sanity-check: analyze_spanish_source still accepts llm_client."""
        from reading_coach.analyzer import analyze_spanish_source
        sig = inspect.signature(analyze_spanish_source)
        assert "llm_client" in sig.parameters


# ---------------------------------------------------------------------------
# TestDefaultRawClient
# ---------------------------------------------------------------------------

class TestDefaultRawClient:
    """When _raw_client is omitted, falls back to infrastructure.ollama_client.chat."""

    def test_uses_infrastructure_chat_by_default(self):
        with mock.patch("reading_coach.llm_adapter.ollama_client") as patched_oc:
            patched_oc.chat.return_value = "response"
            client = make_ollama_coach_client("http://h", "m", 10.0)
            client([{"role": "user", "content": "x"}])
            patched_oc.chat.assert_called_once()

    def test_default_client_passes_correct_host(self):
        with mock.patch("reading_coach.llm_adapter.ollama_client") as patched_oc:
            patched_oc.chat.return_value = "response"
            client = make_ollama_coach_client("http://myhost:11434", "m", 10.0)
            client([{"role": "user", "content": "x"}])
            call_host = patched_oc.chat.call_args[0][0]
        assert call_host == "http://myhost:11434"

    def test_default_client_passes_timeout(self):
        with mock.patch("reading_coach.llm_adapter.ollama_client") as patched_oc:
            patched_oc.chat.return_value = "response"
            client = make_ollama_coach_client("http://h", "m", 77.0)
            client([{"role": "user", "content": "x"}])
            call_timeout = patched_oc.chat.call_args[0][2]
        assert call_timeout == 77.0
