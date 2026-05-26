"""
Static checks that .env.example documents every env var consumed by the
reading_coach package.

These tests read .env.example as a text file and assert that each variable
name appears as a token.  They are purely static — no env vars are set and
no imports of the reading_coach package are needed.

Add an assertion here whenever a new env var is added to the codebase so
the documentation gap is caught automatically in CI.
"""
from __future__ import annotations

import pathlib

import pytest

# Resolve .env.example relative to this file's parent (the project root).
_ENV_EXAMPLE = pathlib.Path(__file__).parent.parent / ".env.example"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _env_lines() -> list[str]:
    return _ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()


def _contains_var(name: str) -> bool:
    """Return True if *name* appears as a token on any non-comment line,
    or in a comment that clearly documents it (e.g. '# SOME_VAR description')."""
    return any(name in line for line in _env_lines())


# ===========================================================================
# TestEnvExampleExists
# ===========================================================================

class TestEnvExampleExists:
    def test_file_exists(self):
        assert _ENV_EXAMPLE.exists(), f".env.example not found at {_ENV_EXAMPLE}"

    def test_file_non_empty(self):
        assert _ENV_EXAMPLE.stat().st_size > 0


# ===========================================================================
# TestReadingCoachVarsDocumented
# All env vars consumed by the reading_coach package must appear in .env.example.
# ===========================================================================

class TestReadingCoachVarsDocumented:
    """Every READING_COACH_* env var must be documented in .env.example."""

    def test_default_level_documented(self):
        assert _contains_var("READING_COACH_DEFAULT_LEVEL")

    def test_max_annotations_documented(self):
        assert _contains_var("READING_COACH_MAX_ANNOTATIONS")

    def test_include_english_gloss_documented(self):
        assert _contains_var("READING_COACH_INCLUDE_ENGLISH_GLOSS")

    def test_include_modern_spanish_documented(self):
        assert _contains_var("READING_COACH_INCLUDE_MODERN_SPANISH")

    def test_timeout_seconds_documented(self):
        assert _contains_var("READING_COACH_TIMEOUT_SECONDS")

    def test_annotation_limit_policy_documented(self):
        assert _contains_var("READING_COACH_ANNOTATION_LIMIT_POLICY")

    def test_max_retries_documented(self):
        """READING_COACH_MAX_RETRIES (reading_coach/retry.py) must be documented."""
        assert _contains_var("READING_COACH_MAX_RETRIES")
