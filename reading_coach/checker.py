"""
reading_coach/checker.py — Deterministic checker for ReadingCoachResult objects.

Three-step severity ladder: passed → warning → failed.
No LLM calls, no Streamlit imports, no external services.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel

from reading_coach.schemas import ReadingCoachResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

STATUS_PASSED: str = "passed"
STATUS_WARNING: str = "warning"
STATUS_FAILED: str = "failed"

_SEVERITY_RANK: dict[str, int] = {STATUS_PASSED: 0, STATUS_WARNING: 1, STATUS_FAILED: 2}
_BEGINNER_LEVELS: frozenset[str] = frozenset({"A1", "A2"})


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

class CoachCheckResult(BaseModel):
    """Result of deterministically checking one ReadingCoachResult."""

    status: str          # STATUS_PASSED | STATUS_WARNING | STATUS_FAILED
    summary: str
    issues: List[str]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CoachCheckerConfig:
    """Knobs for the deterministic checker.

    All have safe defaults — calling ``check_coach_result`` without a config
    uses these values.
    """
    include_english_gloss: bool = False
    max_annotations: int = 20
    trivial_length_threshold: int = 100


# ---------------------------------------------------------------------------
# Internal accumulator
# ---------------------------------------------------------------------------

class _Accumulator:
    """Collects issues and tracks the worst severity seen so far."""

    def __init__(self) -> None:
        self._status = STATUS_PASSED
        self.issues: List[str] = []

    def add(self, severity: str, message: str) -> None:
        self.issues.append(message)
        if _SEVERITY_RANK.get(severity, 0) > _SEVERITY_RANK.get(self._status, 0):
            self._status = severity

    @property
    def status(self) -> str:
        return self._status


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_original_spanish(
    source_text: str,
    result: ReadingCoachResult,
    acc: _Accumulator,
) -> None:
    """Check 1: original_spanish must be byte-identical to source_text."""
    if result.original_spanish != source_text:
        acc.add(
            STATUS_FAILED,
            f"original_spanish does not match source text "
            f"(got {len(result.original_spanish)} chars, expected {len(source_text)} chars).",
        )


def _check_phrase_presence(
    source_text: str,
    result: ReadingCoachResult,
    acc: _Accumulator,
) -> None:
    """Check 2: each difficult phrase must appear in source_text (case-insensitive)."""
    source_lower = source_text.lower()
    for dp in result.difficult_phrases:
        if dp.phrase.lower() not in source_lower:
            acc.add(
                STATUS_WARNING,
                f"Difficult phrase not found in source text: \"{dp.phrase}\".",
            )


def _check_modern_spanish_identity(
    result: ReadingCoachResult,
    acc: _Accumulator,
) -> None:
    """Check 3: modern_spanish should differ from original_spanish when there
    are difficult phrases (identical suggests no actual modernisation happened).
    """
    if (
        result.modern_spanish is not None
        and result.modern_spanish == result.original_spanish
        and result.difficult_phrases
    ):
        acc.add(
            STATUS_WARNING,
            "modern_spanish is identical to original_spanish despite difficult phrases being present; "
            "modernisation may not have been applied.",
        )


def _check_english_gloss(
    result: ReadingCoachResult,
    config: CoachCheckerConfig,
    acc: _Accumulator,
) -> None:
    """Check 4: english_gloss must be non-empty when include_english_gloss is True."""
    if config.include_english_gloss and not (result.english_gloss or "").strip():
        acc.add(
            STATUS_WARNING,
            "english_gloss is empty but include_english_gloss is enabled.",
        )


def _check_annotation_count(
    result: ReadingCoachResult,
    config: CoachCheckerConfig,
    acc: _Accumulator,
) -> None:
    """Check 5: annotation count must not exceed max_annotations."""
    count = len(result.difficult_phrases)
    if count > config.max_annotations:
        acc.add(
            STATUS_WARNING,
            f"Annotation count {count} exceeds max_annotations limit of {config.max_annotations}. "
            "Consider reducing the number of flagged phrases.",
        )


def _check_beginner_empty_annotations(
    source_text: str,
    result: ReadingCoachResult,
    config: CoachCheckerConfig,
    acc: _Accumulator,
) -> None:
    """Check 6: A1/A2 + source longer than threshold + no difficult_phrases → warning.

    A beginner-labelled non-trivial text with zero annotations is suspicious —
    the model likely failed to annotate.
    """
    if (
        result.overall_level in _BEGINNER_LEVELS
        and len(source_text) > config.trivial_length_threshold
        and not result.difficult_phrases
    ):
        acc.add(
            STATUS_WARNING,
            f"Result is labelled {result.overall_level} (beginner) on a non-trivial source "
            f"({len(source_text)} chars) but contains no difficult phrase annotations.",
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def check_coach_result(
    source_text: str,
    result: ReadingCoachResult,
    *,
    config: Optional[CoachCheckerConfig] = None,
) -> CoachCheckResult:
    """Run all deterministic checks and return a CoachCheckResult.

    Always returns synchronously with no I/O.  Never raises — any unexpected
    exception inside a check is caught and recorded as a warning.
    """
    if config is None:
        config = CoachCheckerConfig()

    acc = _Accumulator()

    _checks = [
        lambda: _check_original_spanish(source_text, result, acc),
        lambda: _check_phrase_presence(source_text, result, acc),
        lambda: _check_modern_spanish_identity(result, acc),
        lambda: _check_english_gloss(result, config, acc),
        lambda: _check_annotation_count(result, config, acc),
        lambda: _check_beginner_empty_annotations(source_text, result, config, acc),
    ]

    for check in _checks:
        try:
            check()
        except Exception as exc:  # pragma: no cover
            logger.warning("Unexpected error in coach checker: %s", exc)
            acc.add(STATUS_WARNING, f"Checker error: {exc}")

    if acc.status == STATUS_PASSED:
        summary = "All checks passed."
    elif acc.status == STATUS_WARNING:
        n = len(acc.issues)
        summary = f"{n} warning{'s' if n != 1 else ''} found."
    else:
        n = len(acc.issues)
        summary = f"Check failed with {n} issue{'s' if n != 1 else ''}."

    return CoachCheckResult(status=acc.status, summary=summary, issues=acc.issues)
