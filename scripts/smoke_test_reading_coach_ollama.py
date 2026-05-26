"""
scripts/smoke_test_reading_coach_ollama.py
==========================================

Manual smoke test for the Spanish Source Reading Coach against a *real* local
Ollama instance.

REQUIREMENTS
------------
- Ollama must be running locally (default: http://localhost:11434).
- The model must already be pulled (``ollama pull <model>``).
- No MongoDB or Streamlit is required.

USAGE
-----
Run from the repository root:

    # Default: run the golden-age passage
    python scripts/smoke_test_reading_coach_ollama.py

    # Run a specific named passage
    python scripts/smoke_test_reading_coach_ollama.py --passage modern-short

    # Run multiple passages
    python scripts/smoke_test_reading_coach_ollama.py --passage golden-age mystical-religious

    # Run all passages
    python scripts/smoke_test_reading_coach_ollama.py --all

    # List available passages and exit (no Ollama call)
    python scripts/smoke_test_reading_coach_ollama.py --list

Optional environment overrides (same vars used by the Streamlit app):

    OLLAMA_HOST=http://localhost:11434
    OLLAMA_MODEL=qwen2.5:7b
    READING_COACH_TIMEOUT_SECONDS=120
    READING_COACH_MAX_ANNOTATIONS=7
    READING_COACH_ANNOTATION_LIMIT_POLICY=warn

EXIT CODES
----------
0 - all selected passages parsed and validated successfully.
1 - one or more passages failed (timeout, parse error, checker FAILED).

PYTEST
------
This file is intentionally NOT under the ``tests/`` directory, so it is never
collected by pytest.  pytest.ini sets ``testpaths = tests``.  Run this script
directly with Python, not with pytest.  The passage registry and pure helpers
live in reading_coach/smoke_passages.py and ARE covered by unit tests.
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so reading_coach et al. import cleanly
# when the script is invoked from any working directory.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reading_coach.analyzer import CoachAnalysisError, analyze_spanish_source
from reading_coach.checker import CoachCheckerConfig, STATUS_FAILED
from reading_coach.config import get_coach_settings
from reading_coach.errors import ReadingCoachLLMError, ReadingCoachTimeoutError
from reading_coach.llm_adapter import make_ollama_coach_client
from reading_coach.prompts import SPANISH_SOURCE_PROMPT_VERSION
from reading_coach.smoke_passages import (
    PASSAGES,
    SmokePassage,
    format_passage_header,
    format_phrase_summary,
    list_passage_names,
    parse_cli_args,
    resolve_passages,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
_OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

_SEP = "-" * 64
_STATUS_ICON: dict[str, str] = {"passed": "OK", "warning": "WARN", "failed": "FAIL"}

# ---------------------------------------------------------------------------
# Per-passage runner
# ---------------------------------------------------------------------------


def run_passage(passage: SmokePassage, llm_client, settings) -> int:
    """Run one smoke passage.  Returns 0 on success, 1 on failure."""
    print(f"\n{_SEP}")
    print(format_passage_header(passage, _OLLAMA_MODEL, SPANISH_SOURCE_PROMPT_VERSION))
    print(f"  {passage.description}")
    print(f"  reader_level={passage.reader_level}  density={passage.annotation_density}")
    preview = passage.text[:80] + ("..." if len(passage.text) > 80 else "")
    print(f"  text: {preview}")
    print()

    checker_config = CoachCheckerConfig(
        include_english_gloss=passage.include_english_gloss,
        max_annotations=settings.max_annotations,
        annotation_limit_policy=settings.annotation_limit_policy,
    )

    try:
        ar = analyze_spanish_source(
            passage.text,
            reader_level=passage.reader_level,
            annotation_density=passage.annotation_density,
            include_english_gloss=passage.include_english_gloss,
            include_modern_spanish=passage.include_modern_spanish,
            llm_client=llm_client,
            checker_config=checker_config,
        )
    except ReadingCoachTimeoutError as exc:
        print(f"  [TIMEOUT] {exc.user_message}", file=sys.stderr)
        print("  Increase READING_COACH_TIMEOUT_SECONDS or use a smaller model.", file=sys.stderr)
        return 1
    except ReadingCoachLLMError as exc:
        print(f"  [LLM ERROR] {exc}", file=sys.stderr)
        print("  Check that Ollama is running and the model is loaded.", file=sys.stderr)
        return 1
    except CoachAnalysisError as exc:
        print(f"  [PARSE ERROR] {exc}", file=sys.stderr)
        return 1

    r = ar.result
    check = ar.check
    icon = _STATUS_ICON.get(check.status, "?")

    print(f"  overall level : {r.overall_level}")
    print(f"  checker       : [{icon}] {check.status.upper()} -- {check.summary}")
    print(f"  phrases found : {len(r.difficult_phrases)}")
    print(format_phrase_summary(r.difficult_phrases, max_phrases=3))
    if check.issues:
        for issue in check.issues:
            print(f"  ! {issue}")

    return 1 if check.status == STATUS_FAILED else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_cli_args(argv)

    if args.list_only:
        print("Available passages:")
        for name in list_passage_names():
            p = PASSAGES[name]
            print(f"  {name:<28} {p.category}")
        return 0

    passages = resolve_passages(args)
    settings = get_coach_settings()

    print("\nSpanish Source Reading Coach -- Smoke Test")
    print(f"  Host    : {_OLLAMA_HOST}")
    print(f"  Model   : {_OLLAMA_MODEL}")
    print(f"  Timeout : {settings.timeout_seconds}s")
    print(f"  Policy  : {settings.annotation_limit_policy}")
    print(f"  Passages: {len(passages)} -- {', '.join(p.name for p in passages)}")

    llm_client = make_ollama_coach_client(
        host=_OLLAMA_HOST,
        model=_OLLAMA_MODEL,
        timeout=settings.timeout_seconds,
    )

    failed: list[str] = []
    for passage in passages:
        rc = run_passage(passage, llm_client, settings)
        if rc != 0:
            failed.append(passage.name)

    total = len(passages)
    print(f"\n{_SEP}")
    if failed:
        print(
            f"  SMOKE TEST RESULT: FAILED -- "
            f"{len(failed)}/{total} passage(s) failed: {', '.join(failed)}"
        )
        return 1

    print(f"  SMOKE TEST RESULT: PASSED -- {total}/{total} passage(s) passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
