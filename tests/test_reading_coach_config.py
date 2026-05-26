"""Tests for reading_coach/config.py — environment-variable config parsing.

Also tests:
  - get_default_mode() in core/app_modes.py (APP_DEFAULT_MODE env var)
  - .env.example documents all five new variables

All env-var tests use patch.dict so they do not pollute the real
process environment and remain order-independent.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from reading_coach.config import CoachSettings, get_coach_settings
from reading_coach.schemas import DEFAULT_COACH_LEVEL, VALID_COACH_LEVELS
from core.app_modes import (
    get_default_mode,
    DEFAULT_MODE,
    PARALLEL_READER_MODE,
    READING_COACH_MODE,
)

ROOT = Path(__file__).parent.parent

# Keys we control in these tests — scrubbed before each env-override test.
_COACH_KEYS = {
    "READING_COACH_DEFAULT_LEVEL",
    "READING_COACH_MAX_ANNOTATIONS",
    "READING_COACH_INCLUDE_ENGLISH_GLOSS",
    "READING_COACH_INCLUDE_MODERN_SPANISH",
}


def _clean_env(**overrides: str) -> dict:
    """Return an env dict with all coach keys removed, then overrides applied."""
    base = {k: v for k, v in os.environ.items() if k not in _COACH_KEYS}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# TestCoachSettingsType
# ---------------------------------------------------------------------------

class TestCoachSettingsType:
    def test_returns_coach_settings_instance(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            assert isinstance(get_coach_settings(), CoachSettings)

    def test_is_frozen_dataclass(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            s = get_coach_settings()
            with pytest.raises((AttributeError, TypeError)):
                s.default_level = "C2"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestCoachSettingsDefaults
# ---------------------------------------------------------------------------

class TestCoachSettingsDefaults:
    """When no env vars are set the built-in defaults are returned."""

    def _settings(self) -> CoachSettings:
        with patch.dict(os.environ, _clean_env(), clear=True):
            return get_coach_settings()

    def test_default_level_is_b1(self):
        assert self._settings().default_level == "B1"

    def test_default_level_matches_schema_default(self):
        assert self._settings().default_level == DEFAULT_COACH_LEVEL

    def test_default_max_annotations_is_7(self):
        assert self._settings().max_annotations == 7

    def test_default_include_english_gloss_is_true(self):
        assert self._settings().include_english_gloss is True

    def test_default_include_modern_spanish_is_true(self):
        assert self._settings().include_modern_spanish is True


# ---------------------------------------------------------------------------
# TestCoachSettingsFromEnv
# ---------------------------------------------------------------------------

class TestCoachSettingsFromEnv:
    """Env vars are honoured when set to valid values."""

    def test_custom_level_b2(self):
        with patch.dict(os.environ, _clean_env(READING_COACH_DEFAULT_LEVEL="B2"), clear=True):
            assert get_coach_settings().default_level == "B2"

    def test_custom_level_c1(self):
        with patch.dict(os.environ, _clean_env(READING_COACH_DEFAULT_LEVEL="C1"), clear=True):
            assert get_coach_settings().default_level == "C1"

    def test_level_lowercase_normalised(self):
        with patch.dict(os.environ, _clean_env(READING_COACH_DEFAULT_LEVEL="b2"), clear=True):
            assert get_coach_settings().default_level == "B2"

    def test_custom_max_annotations(self):
        with patch.dict(os.environ, _clean_env(READING_COACH_MAX_ANNOTATIONS="12"), clear=True):
            assert get_coach_settings().max_annotations == 12

    def test_max_annotations_zero_allowed(self):
        with patch.dict(os.environ, _clean_env(READING_COACH_MAX_ANNOTATIONS="0"), clear=True):
            assert get_coach_settings().max_annotations == 0

    def test_english_gloss_disabled(self):
        with patch.dict(os.environ, _clean_env(READING_COACH_INCLUDE_ENGLISH_GLOSS="false"), clear=True):
            assert get_coach_settings().include_english_gloss is False

    def test_english_gloss_disabled_zero(self):
        with patch.dict(os.environ, _clean_env(READING_COACH_INCLUDE_ENGLISH_GLOSS="0"), clear=True):
            assert get_coach_settings().include_english_gloss is False

    def test_english_gloss_enabled_explicit(self):
        with patch.dict(os.environ, _clean_env(READING_COACH_INCLUDE_ENGLISH_GLOSS="true"), clear=True):
            assert get_coach_settings().include_english_gloss is True

    def test_modern_spanish_disabled(self):
        with patch.dict(os.environ, _clean_env(READING_COACH_INCLUDE_MODERN_SPANISH="false"), clear=True):
            assert get_coach_settings().include_modern_spanish is False

    def test_modern_spanish_disabled_off(self):
        with patch.dict(os.environ, _clean_env(READING_COACH_INCLUDE_MODERN_SPANISH="off"), clear=True):
            assert get_coach_settings().include_modern_spanish is False

    def test_modern_spanish_enabled_explicit(self):
        with patch.dict(os.environ, _clean_env(READING_COACH_INCLUDE_MODERN_SPANISH="true"), clear=True):
            assert get_coach_settings().include_modern_spanish is True


# ---------------------------------------------------------------------------
# TestCoachSettingsEdgeCases
# ---------------------------------------------------------------------------

class TestCoachSettingsEdgeCases:
    """Invalid or blank values fall back to built-in defaults."""

    def test_blank_level_uses_default(self):
        with patch.dict(os.environ, _clean_env(READING_COACH_DEFAULT_LEVEL=""), clear=True):
            assert get_coach_settings().default_level == "B1"

    def test_invalid_level_uses_default(self):
        with patch.dict(os.environ, _clean_env(READING_COACH_DEFAULT_LEVEL="X9"), clear=True):
            assert get_coach_settings().default_level == "B1"

    def test_non_integer_max_annotations_uses_default(self):
        with patch.dict(os.environ, _clean_env(READING_COACH_MAX_ANNOTATIONS="seven"), clear=True):
            assert get_coach_settings().max_annotations == 7

    def test_blank_max_annotations_uses_default(self):
        with patch.dict(os.environ, _clean_env(READING_COACH_MAX_ANNOTATIONS=""), clear=True):
            assert get_coach_settings().max_annotations == 7

    def test_all_valid_levels_accepted(self):
        for level in VALID_COACH_LEVELS:
            with patch.dict(os.environ, _clean_env(READING_COACH_DEFAULT_LEVEL=level), clear=True):
                assert get_coach_settings().default_level == level


# ---------------------------------------------------------------------------
# TestGetDefaultMode
# ---------------------------------------------------------------------------

class TestGetDefaultMode:
    """get_default_mode() reads APP_DEFAULT_MODE and falls back to DEFAULT_MODE."""

    def test_returns_string(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("APP_DEFAULT_MODE", None)
            assert isinstance(get_default_mode(), str)

    def test_default_is_parallel_reader_when_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "APP_DEFAULT_MODE"}
        with patch.dict(os.environ, env, clear=True):
            assert get_default_mode() == PARALLEL_READER_MODE

    def test_default_matches_default_mode_constant(self):
        env = {k: v for k, v in os.environ.items() if k != "APP_DEFAULT_MODE"}
        with patch.dict(os.environ, env, clear=True):
            assert get_default_mode() == DEFAULT_MODE

    def test_reading_coach_mode_honoured(self):
        with patch.dict(os.environ, {"APP_DEFAULT_MODE": READING_COACH_MODE}):
            assert get_default_mode() == READING_COACH_MODE

    def test_parallel_reader_mode_honoured(self):
        with patch.dict(os.environ, {"APP_DEFAULT_MODE": PARALLEL_READER_MODE}):
            assert get_default_mode() == PARALLEL_READER_MODE

    def test_invalid_value_falls_back_to_default(self):
        with patch.dict(os.environ, {"APP_DEFAULT_MODE": "completely_unknown_mode"}):
            assert get_default_mode() == DEFAULT_MODE

    def test_blank_value_falls_back_to_default(self):
        with patch.dict(os.environ, {"APP_DEFAULT_MODE": ""}):
            assert get_default_mode() == DEFAULT_MODE

    def test_whitespace_only_falls_back_to_default(self):
        with patch.dict(os.environ, {"APP_DEFAULT_MODE": "   "}):
            assert get_default_mode() == DEFAULT_MODE


# ---------------------------------------------------------------------------
# TestEnvExample
# ---------------------------------------------------------------------------

class TestEnvExample:
    """Static test: .env.example must document all five new variables."""

    def _content(self) -> str:
        return (ROOT / ".env.example").read_text(encoding="utf-8")

    def test_app_default_mode_documented(self):
        assert "APP_DEFAULT_MODE" in self._content()

    def test_reading_coach_default_level_documented(self):
        assert "READING_COACH_DEFAULT_LEVEL" in self._content()

    def test_reading_coach_max_annotations_documented(self):
        assert "READING_COACH_MAX_ANNOTATIONS" in self._content()

    def test_reading_coach_include_english_gloss_documented(self):
        assert "READING_COACH_INCLUDE_ENGLISH_GLOSS" in self._content()

    def test_reading_coach_include_modern_spanish_documented(self):
        assert "READING_COACH_INCLUDE_MODERN_SPANISH" in self._content()

    def test_app_default_mode_default_value_is_parallel_reader(self):
        """The uncommented default must set the legacy parallel-reader mode."""
        content = self._content()
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith("APP_DEFAULT_MODE="):
                value = stripped.split("=", 1)[1]
                assert PARALLEL_READER_MODE in value, (
                    f"APP_DEFAULT_MODE default must preserve legacy mode, got: {value!r}"
                )
                return
        # Key not present as uncommented line — that's also acceptable (opt-in)

    def test_reading_coach_section_present(self):
        assert "Reading Coach" in self._content() or "reading_coach" in self._content().lower()
