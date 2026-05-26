"""tests/test_reading_coach_errors.py — Phase 2 Slice 3 TDD tests.

Verifies the typed exception hierarchy introduced in reading_coach/errors.py:

    ReadingCoachError
    ├── ReadingCoachLLMError
    │   └── ReadingCoachTimeoutError
    └── ReadingCoachParseError
        └── ReadingCoachValidationError

Also verifies integration points:
  - OllamaResponseError (in llm_adapter) is now a ReadingCoachLLMError.
  - Network errors from the raw Ollama client are wrapped in typed errors.
  - ReadingCoachParseError remains importable from reading_coach.response_parser.
  - ReadingCoachValidationError is raised for schema validation failures.
  - CoachAnalysisError (in analyzer) is now a ReadingCoachError subclass.
"""
from __future__ import annotations

import inspect
import json
from unittest.mock import MagicMock

import pytest
import requests

from reading_coach.errors import (
    ReadingCoachError,
    ReadingCoachLLMError,
    ReadingCoachParseError,
    ReadingCoachTimeoutError,
    ReadingCoachValidationError,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Minimal valid payload for parse_reading_coach_response
_VALID_PAYLOAD: dict = {
    "original_spanish": "En un lugar de la Mancha.",
    "overall_level": "B2",
    "modern_spanish": None,
    "english_gloss": None,
    "difficult_phrases": [],
    "grammar_notes": [],
    "comprehension_question": None,
}


# ---------------------------------------------------------------------------
# TestErrorModuleImports
# ---------------------------------------------------------------------------

class TestErrorModuleImports:
    """All five public error types are importable from reading_coach.errors."""

    def test_import_base(self):
        from reading_coach.errors import ReadingCoachError  # noqa: F401
        assert ReadingCoachError is not None

    def test_import_llm_error(self):
        from reading_coach.errors import ReadingCoachLLMError  # noqa: F401
        assert ReadingCoachLLMError is not None

    def test_import_timeout_error(self):
        from reading_coach.errors import ReadingCoachTimeoutError  # noqa: F401
        assert ReadingCoachTimeoutError is not None

    def test_import_parse_error(self):
        from reading_coach.errors import ReadingCoachParseError  # noqa: F401
        assert ReadingCoachParseError is not None

    def test_import_validation_error(self):
        from reading_coach.errors import ReadingCoachValidationError  # noqa: F401
        assert ReadingCoachValidationError is not None

    def test_errors_module_has_no_streamlit_import(self):
        """errors.py must not import Streamlit (keeps it usable at all layers)."""
        import reading_coach.errors as err_mod
        src = inspect.getsource(err_mod)
        assert "import streamlit" not in src, "errors.py must not import streamlit"

    def test_errors_module_has_no_ollama_import(self):
        """errors.py must not import Ollama infrastructure."""
        import reading_coach.errors as err_mod
        src = inspect.getsource(err_mod)
        assert "import ollama" not in src, "errors.py must not import ollama"
        assert "ollama_client" not in src, "errors.py must not reference ollama_client"


# ---------------------------------------------------------------------------
# TestHierarchy
# ---------------------------------------------------------------------------

class TestHierarchy:
    """Exception subclassing relationships match the documented tree."""

    def test_base_is_exception(self):
        assert issubclass(ReadingCoachError, Exception)

    def test_llm_error_is_base(self):
        assert issubclass(ReadingCoachLLMError, ReadingCoachError)

    def test_timeout_is_llm_error(self):
        assert issubclass(ReadingCoachTimeoutError, ReadingCoachLLMError)

    def test_timeout_is_base(self):
        assert issubclass(ReadingCoachTimeoutError, ReadingCoachError)

    def test_parse_error_is_base(self):
        assert issubclass(ReadingCoachParseError, ReadingCoachError)

    def test_validation_error_is_parse_error(self):
        assert issubclass(ReadingCoachValidationError, ReadingCoachParseError)

    def test_validation_error_is_base(self):
        assert issubclass(ReadingCoachValidationError, ReadingCoachError)

    def test_timeout_is_not_parse_error(self):
        assert not issubclass(ReadingCoachTimeoutError, ReadingCoachParseError)

    def test_llm_error_is_not_parse_error(self):
        assert not issubclass(ReadingCoachLLMError, ReadingCoachParseError)

    def test_parse_error_is_not_llm_error(self):
        assert not issubclass(ReadingCoachParseError, ReadingCoachLLMError)


# ---------------------------------------------------------------------------
# TestUserMessage
# ---------------------------------------------------------------------------

class TestUserMessage:
    """Each error type exposes a non-empty user_message property."""

    def test_base_user_message_from_constructor(self):
        exc = ReadingCoachError("something broke")
        assert exc.user_message == "something broke"

    def test_base_user_message_default_non_empty(self):
        exc = ReadingCoachError()
        assert exc.user_message  # must be truthy

    def test_llm_user_message_from_constructor(self):
        exc = ReadingCoachLLMError("Ollama unreachable")
        assert "Ollama unreachable" in exc.user_message

    def test_llm_user_message_default_non_empty(self):
        exc = ReadingCoachLLMError()
        assert exc.user_message

    def test_timeout_user_message_from_constructor(self):
        exc = ReadingCoachTimeoutError("timed out after 60s")
        assert exc.user_message

    def test_timeout_user_message_default_non_empty(self):
        exc = ReadingCoachTimeoutError()
        assert exc.user_message

    def test_parse_user_message_from_constructor(self):
        exc = ReadingCoachParseError("no JSON found")
        assert exc.user_message == "no JSON found"

    def test_parse_user_message_default_non_empty(self):
        exc = ReadingCoachParseError()
        assert exc.user_message

    def test_validation_user_message_from_constructor(self):
        exc = ReadingCoachValidationError("missing field: original_spanish")
        assert "missing field" in exc.user_message

    def test_validation_user_message_default_non_empty(self):
        exc = ReadingCoachValidationError()
        assert exc.user_message

    def test_user_message_has_no_traceback(self):
        """user_message must be a single short string, not a stack trace."""
        exc = ReadingCoachParseError("bad json")
        msg = exc.user_message
        assert "Traceback" not in msg
        assert "File " not in msg


# ---------------------------------------------------------------------------
# TestBackwardCompatParseError
# ---------------------------------------------------------------------------

class TestBackwardCompatParseError:
    """ReadingCoachParseError stays importable from reading_coach.response_parser."""

    def test_importable_from_response_parser(self):
        from reading_coach.response_parser import ReadingCoachParseError as ParserErr  # noqa: F401
        assert ParserErr is ReadingCoachParseError  # same class object

    def test_is_still_exception_subclass(self):
        """Slice 2 tests assert issubclass(ReadingCoachParseError, Exception)."""
        from reading_coach.response_parser import ReadingCoachParseError as ParserErr
        assert issubclass(ParserErr, Exception)

    def test_validation_error_caught_as_parse_error(self):
        """ReadingCoachValidationError must be catchable as ReadingCoachParseError."""
        with pytest.raises(ReadingCoachParseError):
            raise ReadingCoachValidationError("bad schema")

    def test_direct_parse_error_still_works(self):
        with pytest.raises(ReadingCoachParseError):
            raise ReadingCoachParseError("no JSON found")


# ---------------------------------------------------------------------------
# TestResponseParserRaisesValidationError
# ---------------------------------------------------------------------------

class TestResponseParserRaisesValidationError:
    """Schema validation failures in the parser now raise ReadingCoachValidationError."""

    def test_missing_required_field_raises_validation_error(self):
        from reading_coach.response_parser import parse_reading_coach_response
        payload = dict(_VALID_PAYLOAD)
        del payload["original_spanish"]
        with pytest.raises(ReadingCoachValidationError):
            parse_reading_coach_response(json.dumps(payload))

    def test_missing_overall_level_raises_validation_error(self):
        from reading_coach.response_parser import parse_reading_coach_response
        payload = dict(_VALID_PAYLOAD)
        del payload["overall_level"]
        with pytest.raises(ReadingCoachValidationError):
            parse_reading_coach_response(json.dumps(payload))

    def test_validation_error_is_parse_error(self):
        """Schema failures are still catchable as ReadingCoachParseError (backward compat)."""
        from reading_coach.response_parser import parse_reading_coach_response
        payload = dict(_VALID_PAYLOAD)
        del payload["original_spanish"]
        with pytest.raises(ReadingCoachParseError):
            parse_reading_coach_response(json.dumps(payload))

    def test_json_decode_failure_raises_plain_parse_error(self):
        """JSON decode failures raise ReadingCoachParseError (not the subclass)."""
        from reading_coach.response_parser import parse_reading_coach_response
        with pytest.raises(ReadingCoachParseError) as exc_info:
            parse_reading_coach_response("not json at all !@#")
        # Must not be the more-specific ValidationError — that's for schema failures
        assert not isinstance(exc_info.value, ReadingCoachValidationError)

    def test_no_json_object_raises_plain_parse_error(self):
        """Input with no '{' raises ReadingCoachParseError (not ReadingCoachValidationError)."""
        from reading_coach.response_parser import parse_reading_coach_response
        with pytest.raises(ReadingCoachParseError) as exc_info:
            parse_reading_coach_response("just a plain sentence")
        assert not isinstance(exc_info.value, ReadingCoachValidationError)

    def test_validation_error_user_message_non_empty(self):
        """The ReadingCoachValidationError raised by the parser has a user_message."""
        from reading_coach.response_parser import parse_reading_coach_response
        payload = dict(_VALID_PAYLOAD)
        del payload["original_spanish"]
        with pytest.raises(ReadingCoachValidationError) as exc_info:
            parse_reading_coach_response(json.dumps(payload))
        assert exc_info.value.user_message


# ---------------------------------------------------------------------------
# TestOllamaResponseError
# ---------------------------------------------------------------------------

class TestOllamaResponseError:
    """OllamaResponseError is a ReadingCoachLLMError AND still a ValueError."""

    def test_is_reading_coach_llm_error(self):
        from reading_coach.llm_adapter import OllamaResponseError
        assert issubclass(OllamaResponseError, ReadingCoachLLMError)

    def test_is_reading_coach_error(self):
        from reading_coach.llm_adapter import OllamaResponseError
        assert issubclass(OllamaResponseError, ReadingCoachError)

    def test_still_is_value_error(self):
        """Backward compat: Slice 1 tests assert issubclass(OllamaResponseError, ValueError)."""
        from reading_coach.llm_adapter import OllamaResponseError
        assert issubclass(OllamaResponseError, ValueError)

    def test_caught_as_llm_error(self):
        from reading_coach.llm_adapter import OllamaResponseError
        with pytest.raises(ReadingCoachLLMError):
            raise OllamaResponseError("bad shape")

    def test_caught_as_value_error(self):
        from reading_coach.llm_adapter import OllamaResponseError
        with pytest.raises(ValueError):
            raise OllamaResponseError("bad shape")

    def test_has_user_message(self):
        from reading_coach.llm_adapter import OllamaResponseError
        exc = OllamaResponseError("unexpected shape: int")
        assert exc.user_message


# ---------------------------------------------------------------------------
# TestAdapterErrorWrapping
# ---------------------------------------------------------------------------

class TestAdapterErrorWrapping:
    """The llm_adapter wraps network errors from the raw client into typed errors."""

    def _make_client(self, side_effect):
        """Build a coach client whose raw underlying call raises *side_effect*."""
        from reading_coach.llm_adapter import make_ollama_coach_client
        mock = MagicMock(side_effect=side_effect)
        return make_ollama_coach_client(
            "http://localhost:11434",
            "test-model",
            timeout=10.0,
            _raw_client=mock,
        )

    # --- Timeout ---

    def test_requests_timeout_raises_reading_coach_timeout_error(self):
        client = self._make_client(requests.exceptions.Timeout("timed out"))
        with pytest.raises(ReadingCoachTimeoutError):
            client([{"role": "user", "content": "hello"}])

    def test_reading_coach_timeout_is_llm_error(self):
        """ReadingCoachTimeoutError is also a ReadingCoachLLMError."""
        client = self._make_client(requests.exceptions.Timeout("timed out"))
        with pytest.raises(ReadingCoachLLMError):
            client([{"role": "user", "content": "hello"}])

    def test_timeout_error_has_user_message(self):
        client = self._make_client(requests.exceptions.Timeout("timed out"))
        with pytest.raises(ReadingCoachTimeoutError) as exc_info:
            client([{"role": "user", "content": "hello"}])
        assert exc_info.value.user_message

    # --- Connection error ---

    def test_connection_error_raises_reading_coach_llm_error(self):
        client = self._make_client(requests.exceptions.ConnectionError("no route"))
        with pytest.raises(ReadingCoachLLMError):
            client([{"role": "user", "content": "hello"}])

    def test_connection_error_not_timeout(self):
        """A ConnectionError is NOT a timeout."""
        client = self._make_client(requests.exceptions.ConnectionError("no route"))
        with pytest.raises(ReadingCoachLLMError) as exc_info:
            client([{"role": "user", "content": "hello"}])
        assert not isinstance(exc_info.value, ReadingCoachTimeoutError)

    def test_connection_error_user_message_non_empty(self):
        client = self._make_client(requests.exceptions.ConnectionError("no route"))
        with pytest.raises(ReadingCoachLLMError) as exc_info:
            client([{"role": "user", "content": "hello"}])
        assert exc_info.value.user_message

    # --- HTTP error ---

    def test_http_error_raises_reading_coach_llm_error(self):
        client = self._make_client(requests.exceptions.HTTPError("500 Server Error"))
        with pytest.raises(ReadingCoachLLMError):
            client([{"role": "user", "content": "hello"}])

    # --- Chained cause ---

    def test_original_exception_is_chained(self):
        """The wrapped error must chain the original via __cause__."""
        cause = requests.exceptions.ConnectionError("no route to host")
        client = self._make_client(cause)
        with pytest.raises(ReadingCoachLLMError) as exc_info:
            client([{"role": "user", "content": "hello"}])
        assert exc_info.value.__cause__ is cause


# ---------------------------------------------------------------------------
# TestCoachAnalysisErrorIntegration
# ---------------------------------------------------------------------------

class TestCoachAnalysisErrorIntegration:
    """CoachAnalysisError is a ReadingCoachError; the analyzer surfaces typed errors."""

    def test_coach_analysis_error_is_reading_coach_error(self):
        from reading_coach.analyzer import CoachAnalysisError
        assert issubclass(CoachAnalysisError, ReadingCoachError)

    def test_coach_analysis_error_is_exception(self):
        from reading_coach.analyzer import CoachAnalysisError
        assert issubclass(CoachAnalysisError, Exception)

    def test_analyze_raises_coach_analysis_error_on_bad_json(self):
        from reading_coach.analyzer import CoachAnalysisError, analyze_spanish_source
        bad_client = MagicMock(return_value="not json at all")
        with pytest.raises(CoachAnalysisError):
            analyze_spanish_source("Test passage.", llm_client=bad_client)

    def test_analyze_raises_reading_coach_error_on_bad_json(self):
        """CoachAnalysisError is catchable as ReadingCoachError."""
        from reading_coach.analyzer import analyze_spanish_source
        bad_client = MagicMock(return_value="not json at all")
        with pytest.raises(ReadingCoachError):
            analyze_spanish_source("Test passage.", llm_client=bad_client)

    def test_coach_analysis_error_has_user_message(self):
        from reading_coach.analyzer import CoachAnalysisError
        exc = CoachAnalysisError("something went wrong")
        assert exc.user_message == "something went wrong"

    def test_coach_analysis_error_user_message_default_non_empty(self):
        from reading_coach.analyzer import CoachAnalysisError
        exc = CoachAnalysisError()
        assert exc.user_message

    def test_propagated_error_has_user_message(self):
        """An error that bubbles through analyze_spanish_source has user_message."""
        from reading_coach.analyzer import analyze_spanish_source
        bad_client = MagicMock(return_value="bad json")
        with pytest.raises(ReadingCoachError) as exc_info:
            analyze_spanish_source("Test.", llm_client=bad_client)
        assert exc_info.value.user_message

    def test_error_message_contains_parse_keywords(self):
        """Match constraint: analyzer tests check for 'parse'/'ReadingCoachResult'/'invalid'."""
        from reading_coach.analyzer import analyze_spanish_source
        bad_client = MagicMock(return_value="not json")
        with pytest.raises(ReadingCoachError, match=r"(?i)(parse|ReadingCoachResult|invalid)"):
            analyze_spanish_source("Test.", llm_client=bad_client)
