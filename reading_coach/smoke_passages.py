"""
reading_coach/smoke_passages.py — Passage registry and CLI helpers for the
Reading Coach smoke runner.

Pure functions and data only.  No Ollama, Streamlit, or network access.
Importable by scripts/smoke_test_reading_coach_ollama.py and by unit tests
without any external services.

Add a new entry to PASSAGES whenever a new passage category is needed and
add a corresponding assertion in tests/test_smoke_passages.py so the registry
stays consistent.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# SmokePassage
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SmokePassage:
    """One named test passage for the smoke runner.

    All fields except *reader_level*, *annotation_density*,
    *include_modern_spanish*, and *include_english_gloss* are required.
    """

    name: str
    """Short identifier used on the command line (e.g. ``golden-age``)."""

    category: str
    """Human-readable category label shown in the header line."""

    text: str
    """The Spanish passage text passed to ``analyze_spanish_source``."""

    description: str
    """One-line description of what this passage tests."""

    reader_level: str = "B1"
    """CEFR level forwarded to ``analyze_spanish_source``."""

    annotation_density: str = "balanced"
    """Density tier forwarded to ``analyze_spanish_source``."""

    include_modern_spanish: bool = True
    """Whether to request a modern Spanish paraphrase."""

    include_english_gloss: bool = True
    """Whether to request an English gloss."""


# ---------------------------------------------------------------------------
# Passage registry
# ---------------------------------------------------------------------------

PASSAGES: dict[str, SmokePassage] = {
    "modern-short": SmokePassage(
        name="modern-short",
        category="modern Spanish (very short)",
        text=(
            "Hoy hace mucho calor en Madrid. "
            "Salí temprano a comprar el periódico y me tomé un café en la terraza."
        ),
        description=(
            "Very short, entirely modern Spanish — verifies the model handles "
            "trivial text without hallucinating difficulty."
        ),
        reader_level="A2",
        annotation_density="minimal",
    ),

    "golden-age": SmokePassage(
        name="golden-age",
        category="Golden Age literary prose",
        text=(
            "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, "
            "no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, "
            "adarga antigua, rocín flaco y galgo corredor."
        ),
        description=(
            "Opening of Don Quijote (Cervantes, 1605) — canonical Golden Age prose "
            "with archaic vocabulary and elliptical noun phrases."
        ),
        reader_level="B1",
        annotation_density="balanced",
    ),

    "mystical-religious": SmokePassage(
        name="mystical-religious",
        category="mystical/religious register",
        text=(
            "Noche oscura del alma, en ansias en amores inflamada, "
            "¡oh dichosa ventura! salí sin ser notada, "
            "estando ya mi casa sosegada."
        ),
        description=(
            "First stanza of San Juan de la Cruz's 'Noche oscura del alma' — "
            "dense mystical register, archaic orthography, and figurative language."
        ),
        reader_level="C1",
        annotation_density="detailed",
    ),

    "zero-phrases": SmokePassage(
        name="zero-phrases",
        category="simple modern text (expected: few or no annotations)",
        text=(
            "El niño fue a la escuela. Su madre le dio el desayuno antes de salir. "
            "El profesor explicó la lección y todos los alumnos escucharon con atención."
        ),
        description=(
            "Deliberately A1-level text — verifies the model does not over-annotate "
            "simple, common vocabulary."
        ),
        reader_level="A1",
        annotation_density="minimal",
    ),

    "subordinate-clauses": SmokePassage(
        name="subordinate-clauses",
        category="long sentence with heavy subordination",
        text=(
            "Aunque los viajeros que llegaron al puerto antes del amanecer "
            "esperaban encontrar los barcos listos para hacerse a la vela, "
            "resultó que el capitán, a quien todos temían por su severidad, "
            "había ordenado que nadie se acercara a los muelles hasta que "
            "se despejara la niebla que cubría la bahía desde la noche anterior."
        ),
        description=(
            "Single long sentence with multiple subordinate clauses — tests "
            "annotation of syntactic complexity and subjunctive constructions."
        ),
        reader_level="B2",
        annotation_density="balanced",
    ),
}

# Canonical ordering for --list and --all
_PASSAGE_ORDER: tuple[str, ...] = (
    "modern-short",
    "golden-age",
    "mystical-religious",
    "zero-phrases",
    "subordinate-clauses",
)

_DEFAULT_PASSAGE: str = "golden-age"


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def list_passage_names() -> list[str]:
    """Return passage names in canonical registration order."""
    return [name for name in _PASSAGE_ORDER if name in PASSAGES]


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    epilog_lines = ["Available passages:"]
    for name in list_passage_names():
        p = PASSAGES[name]
        epilog_lines.append(f"  {name:<28} {p.category}")

    parser = argparse.ArgumentParser(
        prog="smoke_test_reading_coach_ollama",
        description="Manual smoke test for the Spanish Source Reading Coach.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(epilog_lines),
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--passage",
        metavar="NAME",
        dest="selected_passages",
        nargs="+",
        choices=list(PASSAGES.keys()),
        default=None,
        help="Run one or more named passages (space-separated).",
    )
    group.add_argument(
        "--all",
        dest="all_passages",
        action="store_true",
        default=False,
        help="Run every passage in the registry.",
    )
    parser.add_argument(
        "--list",
        dest="list_only",
        action="store_true",
        default=False,
        help="Print available passage names and exit without calling Ollama.",
    )
    return parser


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the smoke runner.

    Parameters
    ----------
    argv:
        Argument list (default: ``sys.argv[1:]``).  Pass an explicit list in
        tests to avoid touching the real ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Attributes: ``selected_passages`` (list or None), ``all_passages``
        (bool), ``list_only`` (bool).
    """
    return _build_parser().parse_args(argv)


def resolve_passages(args: argparse.Namespace) -> list[SmokePassage]:
    """Return the ordered list of passages to run based on parsed CLI args.

    Default (no flags): the ``golden-age`` passage, preserving the original
    single-passage behaviour.
    """
    if args.all_passages:
        return [PASSAGES[name] for name in list_passage_names()]
    if args.selected_passages:
        return [PASSAGES[n] for n in args.selected_passages]
    return [PASSAGES[_DEFAULT_PASSAGE]]


# ---------------------------------------------------------------------------
# Result formatting helpers
# ---------------------------------------------------------------------------

def format_passage_header(passage: SmokePassage, model: str, prompt_version: str) -> str:
    """Return a compact header line for one passage run."""
    return (
        f"passage={passage.name}  model={model}  "
        f"version={prompt_version}  category={passage.category}"
    )


def format_phrase_summary(result_phrases: list[Any], max_phrases: int = 3) -> str:
    """Return a compact multi-line summary of the first *max_phrases* phrases.

    Accepts any objects with ``.phrase``, ``.category``, and
    ``.difficulty_level`` attributes (e.g. ``DifficultPhrase`` instances or
    test stubs using ``types.SimpleNamespace``).
    """
    if not result_phrases:
        return "  (no difficult phrases)"
    lines: list[str] = []
    for dp in result_phrases[:max_phrases]:
        lines.append(
            f"  \u2022 \"{dp.phrase}\"  [{dp.category}, {dp.difficulty_level}]"
        )
    remainder = len(result_phrases) - max_phrases
    if remainder > 0:
        lines.append(f"  \u2026 and {remainder} more")
    return "\n".join(lines)
