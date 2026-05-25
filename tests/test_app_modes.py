"""
Tests: app mode constants and validation logic.

All tests are pure-Python — no Streamlit runtime, no Ollama, no Docker.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------

from core.app_modes import (
    PARALLEL_READER_MODE,
    READING_COACH_MODE,
    APP_MODES,
    DEFAULT_MODE,
    resolve_mode,
)


# ---------------------------------------------------------------------------
# Mode label constants
# ---------------------------------------------------------------------------

class TestModeLabels:
    def test_parallel_reader_label(self):
        assert PARALLEL_READER_MODE == "English → Spanish Parallel Reader"

    def test_reading_coach_label(self):
        assert READING_COACH_MODE == "Spanish Source Reading Coach"

    def test_app_modes_contains_both(self):
        assert PARALLEL_READER_MODE in APP_MODES
        assert READING_COACH_MODE in APP_MODES

    def test_app_modes_has_exactly_two_entries(self):
        assert len(APP_MODES) == 2

    def test_app_modes_are_non_empty_strings(self):
        for mode in APP_MODES:
            assert isinstance(mode, str)
            assert mode.strip() != ""

    def test_app_modes_are_distinct(self):
        assert len(set(APP_MODES)) == len(APP_MODES)


# ---------------------------------------------------------------------------
# Default mode
# ---------------------------------------------------------------------------

class TestDefaultMode:
    def test_default_is_parallel_reader(self):
        assert DEFAULT_MODE == PARALLEL_READER_MODE

    def test_default_is_first_in_app_modes(self):
        assert APP_MODES[0] == DEFAULT_MODE


# ---------------------------------------------------------------------------
# resolve_mode — normalization / fallback
# ---------------------------------------------------------------------------

class TestResolveMode:
    def test_parallel_reader_roundtrips(self):
        assert resolve_mode(PARALLEL_READER_MODE) == PARALLEL_READER_MODE

    def test_reading_coach_roundtrips(self):
        assert resolve_mode(READING_COACH_MODE) == READING_COACH_MODE

    def test_none_falls_back_to_default(self):
        assert resolve_mode(None) == DEFAULT_MODE

    def test_empty_string_falls_back_to_default(self):
        assert resolve_mode("") == DEFAULT_MODE

    def test_unknown_string_falls_back_to_default(self):
        assert resolve_mode("some-future-mode") == DEFAULT_MODE

    def test_whitespace_only_falls_back_to_default(self):
        assert resolve_mode("   ") == DEFAULT_MODE

    def test_non_string_falls_back_to_default(self):
        assert resolve_mode(42) == DEFAULT_MODE  # type: ignore[arg-type]

    def test_case_sensitive_partial_match_falls_back(self):
        # "reading coach" without full label should NOT match
        assert resolve_mode("reading coach") == DEFAULT_MODE

    def test_return_type_is_always_str(self):
        for value in [None, "", "garbage", PARALLEL_READER_MODE, READING_COACH_MODE, 0]:
            result = resolve_mode(value)  # type: ignore[arg-type]
            assert isinstance(result, str)


# ---------------------------------------------------------------------------
# resolve_mode — session-state dict helper (optional convenience overload)
# ---------------------------------------------------------------------------

class TestResolveModeFromDict:
    """resolve_mode should also accept a dict (session-state proxy)."""

    def test_dict_with_app_mode_key(self):
        state = {"app_mode": READING_COACH_MODE}
        assert resolve_mode(state) == READING_COACH_MODE

    def test_dict_missing_app_mode_key_falls_back(self):
        state: dict = {}
        assert resolve_mode(state) == DEFAULT_MODE

    def test_dict_with_invalid_mode_falls_back(self):
        state = {"app_mode": "nonsense"}
        assert resolve_mode(state) == DEFAULT_MODE

    def test_dict_with_none_value_falls_back(self):
        state = {"app_mode": None}
        assert resolve_mode(state) == DEFAULT_MODE
