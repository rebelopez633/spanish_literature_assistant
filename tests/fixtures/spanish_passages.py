"""
tests/fixtures/spanish_passages.py — Short synthetic Spanish passages for pipeline tests.

All five passages are either in the public domain (classic first lines) or
entirely synthetic.  None require external files or network access.

Registers covered:
  modern     — contemporary everyday prose
  archaic    — literary/classical word order and vocabulary
  mystical   — religious-mystical exclamatory verse (synthetic)
  complex    — long sentence with multiple subordinate clauses
  dialogue   — archaic second-person plural, direct speech
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SpanishPassage:
    """A short, self-contained Spanish text used as a test fixture."""

    key: str        # unique identifier used in tests to look up a passage
    text: str       # the actual Spanish text — verbatim, no surrounding whitespace
    register: str   # "modern" | "archaic" | "mystical" | "complex" | "dialogue"
    note: str       # brief plain-English description of linguistic features


PASSAGES: Tuple[SpanishPassage, ...] = (
    SpanishPassage(
        key="simple_modern",
        text="El niño juega en el parque con su perro.",
        register="modern",
        note="Simple present tense; everyday A1–A2 vocabulary; no ambiguity.",
    ),
    SpanishPassage(
        key="archaic_literary",
        text=(
            "Díjole el caballero que no habría paz entre ellos"
            " mientras la honra no fuese satisfecha."
        ),
        register="archaic",
        note=(
            "Archaic enclitic pronoun (díjole), verb-initial word order, "
            "past subjunctive (fuese), honour-culture theme."
        ),
    ),
    SpanishPassage(
        key="mystical_religious",
        text=(
            "¡Oh llama de amor viva,"
            " que tiernamente hieres de mi alma en el más profundo centro!"
        ),
        register="mystical",
        note=(
            "Synthetic exclamatory verse in the religious-mystical register; "
            "inverted syntax; archaic second-person singular implied subject."
        ),
    ),
    SpanishPassage(
        key="complex_subordinate",
        text=(
            "Cuando los viajeros llegaron al pueblo,"
            " que había sido fundado por sus abuelos en tiempos de grandes penurias,"
            " hallaron las calles vacías y las puertas cerradas a cal y canto."
        ),
        register="complex",
        note=(
            "Temporal clause (cuando), embedded relative clause, "
            "pluperfect (había sido fundado), literary synonym (hallaron/encontraron), "
            "fixed expression (a cal y canto)."
        ),
    ),
    SpanishPassage(
        key="dialogue",
        text=(
            "—¿Adónde vais con tanta prisa?"
            " —preguntó la anciana desde el umbral de su casa."
        ),
        register="dialogue",
        note=(
            "Em-dash dialogue convention, archaic vosotros form (vais), "
            "interrogative with accent, direct-speech attribution."
        ),
    ),
)


def get_passage(key: str) -> SpanishPassage:
    """Return the passage with the given key, raising KeyError if not found."""
    for passage in PASSAGES:
        if passage.key == key:
            return passage
    raise KeyError(f"No passage with key {key!r}. Available: {[p.key for p in PASSAGES]}")
