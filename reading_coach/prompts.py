"""
Pure prompt builder for the Spanish Source Reading Coach.

build_coach_prompt() constructs the Ollama chat messages list used to ask a
local LLM to annotate a passage of original Spanish for English-speaking
learners.  It performs no I/O and makes no Ollama calls.

To evolve the prompt, bump SPANISH_SOURCE_PROMPT_VERSION and update the
regression invariant tests in tests/test_prompt_version.py so the change is
explicit and reviewable.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Prompt version
# Bump this string whenever the prompt instructions change in a way that
# affects model behaviour.  The new value propagates automatically to every
# AnalysisResult so results are traceable to the prompt that produced them.
# ---------------------------------------------------------------------------

SPANISH_SOURCE_PROMPT_VERSION: str = "spanish_source_v2"

# ---------------------------------------------------------------------------
# Annotation-density descriptions injected into the user prompt
# ---------------------------------------------------------------------------

_DENSITY_INSTRUCTIONS: dict[str, str] = {
    "minimal": (
        "Annotation density: MINIMAL. "
        "Identify only the most essential phrases — archaic vocabulary, opaque "
        "idioms, or grammar structures that would block comprehension. "
        "Aim for three to five entries in difficult_phrases at most. "
        "Do not annotate common modern Spanish words."
    ),
    "balanced": (
        "Annotation density: BALANCED. "
        "Identify phrases that would genuinely challenge a learner at the stated "
        "level — unusual vocabulary, literary constructions, archaic forms, and "
        "non-obvious idioms. Aim for five to ten entries in difficult_phrases. "
        "Skip words a learner at this level is very likely to know."
    ),
    "detailed": (
        "Annotation density: DETAILED. "
        "Provide comprehensive annotation: unusual vocabulary, archaic forms, "
        "idiomatic expressions, rhetorical devices, syntactic inversions, and "
        "cultural references. Aim for ten or more entries in difficult_phrases "
        "where the passage warrants it. Still avoid annotating obvious words."
    ),
}

_DEFAULT_DENSITY = "balanced"

# ---------------------------------------------------------------------------
# JSON schema example embedded in the prompt so the model knows the shape
# ---------------------------------------------------------------------------

_SCHEMA_EXAMPLE = """\
{
  "original_spanish": "<exact copy of the source text — never alter>",
  "overall_level": "B2",
  "modern_spanish": "<modern Spanish paraphrase of the full passage, or null>",
  "english_gloss": "<flowing English translation of the full passage, or null>",
  "difficult_phrases": [
    {
      "phrase": "<exact substring from source text>",
      "category": "<e.g. archaic_vocab | idiom | literary | religious | rhetorical | syntactic>",
      "difficulty_level": "B2",
      "why_difficult": "<explanation of why this phrase is hard at the learner level>",
      "modern_spanish_equivalent": "<optional modern Spanish replacement>",
      "english_meaning": "<optional English gloss for this phrase>",
      "grammar_note": "<optional grammar observation>",
      "learner_tip": "<optional mnemonic or study tip>"
    }
  ],
  "grammar_notes": [
    {
      "topic": "<grammar topic name>",
      "explanation": "<explanation of the pattern>",
      "example_from_text": "<optional verbatim example from the passage>"
    }
  ],
  "comprehension_question": {
    "question": "<a question about the passage>",
    "answer_hint": "<optional brief hint>"
  }
}"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_coach_prompt(
    source_text: str,
    *,
    reader_level: str = "B1",
    annotation_density: str = "balanced",
    include_english_gloss: bool = True,
    include_modern_spanish: bool = True,
) -> list[dict[str, str]]:
    """Return a two-message list suitable for Ollama's /api/chat endpoint.

    The messages instruct the model to act as a Spanish reading coach and
    return a JSON object conforming to ReadingCoachResult.  This function
    is pure: no I/O, no Ollama calls, no side effects.

    Parameters
    ----------
    source_text:
        Original Spanish passage to annotate.  Embedded verbatim.
    reader_level:
        CEFR level of the target learner, e.g. ``"B1"``.
    annotation_density:
        One of ``"minimal"``, ``"balanced"``, or ``"detailed"``.
        Controls how many phrases the model is asked to annotate.
    include_english_gloss:
        When ``True``, instruct the model to populate the ``english_gloss``
        field with a flowing English translation.  When ``False``, the
        field must be set to ``null``.
    include_modern_spanish:
        When ``True``, instruct the model to populate the ``modern_spanish``
        field.  When ``False``, the field must be set to ``null``.
    """
    density_instruction = _DENSITY_INSTRUCTIONS.get(
        annotation_density, _DENSITY_INSTRUCTIONS[_DEFAULT_DENSITY]
    )

    if include_modern_spanish:
        modern_spanish_rule = (
            "- modern_spanish: Provide a clear modern Spanish paraphrase of the "
            "entire passage that replaces archaic or literary forms with "
            "contemporary equivalents while preserving meaning."
        )
    else:
        modern_spanish_rule = (
            "- modern_spanish: Set to null. Do not provide a modern Spanish paraphrase."
        )

    if include_modern_spanish:
        modern_spanish_equivalent_rule = (
            "- modern_spanish_equivalent (within each difficult_phrase): Where a modern "
            "Spanish replacement exists, populate this field with a concise contemporary "
            "equivalent word or phrase."
        )
    else:
        modern_spanish_equivalent_rule = (
            "- modern_spanish_equivalent (within each difficult_phrase): Set to null for "
            "all phrases. Do not provide modern Spanish equivalents for individual phrases."
        )

    if include_english_gloss:
        english_gloss_rule = (
            "- english_gloss: Provide a flowing English translation of the entire "
            "passage. Prioritise clarity and natural English over word-for-word "
            "fidelity. Populate the english_gloss field."
        )
    else:
        english_gloss_rule = (
            "- english_gloss: Set to null. Do not provide an English translation."
        )

    system = (
        "You are an expert Spanish reading coach for English-speaking learners. "
        "You have deep knowledge of classical, medieval, Golden Age, and modern "
        "Spanish literature, grammar, and rhetoric. "
        "Return only valid JSON matching the provided schema. "
        "Do not include Markdown, XML, chain-of-thought, or commentary outside JSON. "
        "Do not invent historical, plot, theological, or biographical context not "
        "present in the passage. "
        "Do not fabricate information about the author or period unless it is "
        "explicitly stated in the source text."
    )

    user = f"""Analyse the Spanish passage below for an English-speaking learner at CEFR level {reader_level}.

{density_instruction}

Rules:
- original_spanish MUST be an exact, character-for-character copy of the SOURCE TEXT.
  Never alter, correct, or normalise the original in any way.
- overall_level: Estimate the CEFR level of the passage (A1–C2).
- difficult_phrases: Each phrase MUST be an exact substring of the source text (case-sensitive).
  Only annotate phrases that are genuinely likely to challenge a learner at level {reader_level}.
  Avoid over-annotating obvious or common words.
  Prefer phrase-level explanations over isolated dictionary glosses.
  Explain archaic, literary, religious, rhetorical, or syntactic difficulty when relevant.
- grammar_notes: Include notes on non-obvious grammar patterns in the passage.
- comprehension_question: Provide one question that checks understanding of the passage.
{modern_spanish_rule}
{modern_spanish_equivalent_rule}
{english_gloss_rule}
- Use only CEFR values for all level fields: A1, A2, B1, B2, C1, C2.
- Output ONLY a single valid JSON object that follows this schema exactly.
  Replace every placeholder with real content for the passage.
  Use null (not an empty string) for absent optional fields.

JSON SCHEMA EXAMPLE:
{_SCHEMA_EXAMPLE}

SOURCE TEXT:
{source_text}"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
