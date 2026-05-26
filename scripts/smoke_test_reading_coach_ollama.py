"""
scripts/smoke_test_reading_coach_ollama.py
==========================================

Manual smoke test for the Spanish Source Reading Coach against a *real* local
Ollama instance.

REQUIREMENTS
------------
- Ollama must be running locally (default: http://localhost:11434).
- The model must already be pulled (``ollama pull <model>``).
- No MongoDB or Streamlit is required — this script is pure Python.

USAGE
-----
Run from the repository root:

    python scripts/smoke_test_reading_coach_ollama.py

Optional environment overrides (same vars used by the Streamlit app):

    OLLAMA_HOST=http://localhost:11434
    OLLAMA_MODEL=qwen2.5:7b
    READING_COACH_TIMEOUT_SECONDS=120
    READING_COACH_MAX_ANNOTATIONS=7
    READING_COACH_ANNOTATION_LIMIT_POLICY=warn

EXIT CODES
----------
0 — analysis parsed and validated successfully.
1 — LLM timeout, connection error, parse failure, or validation error.

PYTEST
------
This file is intentionally NOT under the ``tests/`` directory, so it is
never collected by ``pytest`` during normal CI runs.  pytest.ini sets
``testpaths = tests``, which excludes scripts/.  To run this script
explicitly pass its path directly to Python, not to pytest.
"""
from __future__ import annotations

import os
import sys
import textwrap

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so ``reading_coach`` et al. import
# cleanly when the script is invoked from any working directory.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reading_coach.analyzer import AnalysisResult, CoachAnalysisError, analyze_spanish_source
from reading_coach.checker import CoachCheckerConfig, STATUS_PASSED, STATUS_WARNING, STATUS_FAILED
from reading_coach.config import get_coach_settings
from reading_coach.errors import ReadingCoachLLMError, ReadingCoachTimeoutError
from reading_coach.llm_adapter import make_ollama_coach_client

# ---------------------------------------------------------------------------
# Config — reads the same env vars as the Streamlit app, with local defaults
# ---------------------------------------------------------------------------

_OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
_OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# ---------------------------------------------------------------------------
# Synthetic test passage — short, Golden Age Spanish, multiple difficulty
# levels present so the model has useful material to annotate.
# ---------------------------------------------------------------------------

_PASSAGE = (
    "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, "
    "no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, "
    "adarga antigua, rocín flaco y galgo corredor."
)

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_SEP = "─" * 60


def _section(title: str) -> None:
    print(f"\n{_SEP}\n  {title}\n{_SEP}")


def _wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=76, initial_indent=prefix, subsequent_indent=prefix)


def _print_result(ar: AnalysisResult) -> None:
    r = ar.result
    check = ar.check

    _section("Original Spanish")
    print(_wrap(r.original_spanish))

    _section("Overall Level")
    print(f"    {r.overall_level}")

    _section("Modern Spanish Paraphrase")
    if r.modern_spanish:
        print(_wrap(r.modern_spanish))
    else:
        print("    (not provided)")

    _section("English Gloss")
    if r.english_gloss:
        print(_wrap(r.english_gloss))
    else:
        print("    (not provided)")

    _section(f"Difficult Phrases  ({len(r.difficult_phrases)} total)")
    if r.difficult_phrases:
        for i, dp in enumerate(r.difficult_phrases, 1):
            print(f"    [{i}] \"{dp.phrase}\"  ({dp.category}, {dp.difficulty_level})")
            print(f"        Why difficult : {dp.why_difficult}")
            if dp.modern_spanish_equivalent:
                print(f"        Modern Spanish: {dp.modern_spanish_equivalent}")
            if dp.english_meaning:
                print(f"        English       : {dp.english_meaning}")
            if dp.grammar_note:
                print(f"        Grammar note  : {dp.grammar_note}")
    else:
        print("    (none)")

    _section("Checker Result")
    _STATUS_ICON = {STATUS_PASSED: "✓", STATUS_WARNING: "⚠", STATUS_FAILED: "✗"}
    icon = _STATUS_ICON.get(check.status, "?")
    print(f"    {icon}  {check.status.upper()} — {check.summary}")
    for issue in check.issues:
        print(f"        • {issue}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    settings = get_coach_settings()

    print(f"\nSpanish Source Reading Coach — Smoke Test")
    print(f"  Host    : {_OLLAMA_HOST}")
    print(f"  Model   : {_OLLAMA_MODEL}")
    print(f"  Timeout : {settings.timeout_seconds}s")
    print(f"  Policy  : {settings.annotation_limit_policy}")
    print(f"  Passage : {_PASSAGE[:60]}…")

    llm_client = make_ollama_coach_client(
        host=_OLLAMA_HOST,
        model=_OLLAMA_MODEL,
        timeout=settings.timeout_seconds,
    )

    checker_config = CoachCheckerConfig(
        include_english_gloss=True,
        max_annotations=settings.max_annotations,
        annotation_limit_policy=settings.annotation_limit_policy,
    )

    print("\nCalling analyze_spanish_source() …  (this may take a moment)")

    try:
        result = analyze_spanish_source(
            _PASSAGE,
            reader_level=settings.default_level,
            annotation_density="balanced",
            include_english_gloss=True,
            include_modern_spanish=True,
            llm_client=llm_client,
            checker_config=checker_config,
        )
    except ReadingCoachTimeoutError as exc:
        print(f"\n[TIMEOUT] {exc.user_message}", file=sys.stderr)
        print("         Increase READING_COACH_TIMEOUT_SECONDS or choose a smaller/faster model.", file=sys.stderr)
        return 1
    except ReadingCoachLLMError as exc:
        print(f"\n[LLM ERROR] {exc}", file=sys.stderr)
        print("            Check that Ollama is running and the model is loaded.", file=sys.stderr)
        return 1
    except CoachAnalysisError as exc:
        print(f"\n[PARSE ERROR] {exc}", file=sys.stderr)
        return 1

    _print_result(result)

    print(f"\n{_SEP}")
    final_status = result.check.status
    if final_status == STATUS_FAILED:
        print("  SMOKE TEST RESULT: FAILED (checker reported failure)")
        return 1

    print(f"  SMOKE TEST RESULT: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
