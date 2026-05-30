"""
reading_coach/study_notes.py — Pure Markdown export for reading-coach results.

No Streamlit, no Ollama, no I/O.  Public functions:

    reading_coach_result_to_markdown(
        result: ReadingCoachResult,
        title: str | None = None,
    ) -> str

    multi_chunk_result_to_markdown(
        result: MultiChunkAnalysisResult,
        title: str | None = None,
    ) -> str

Both return self-contained Markdown documents suitable for saving as study
notes, pasting into a notes app, or rendering in the UI.
"""
from __future__ import annotations

from typing import Optional

from reading_coach.schemas import (
    ChunkAnalysisResult,
    ComprehensionQuestion,
    DifficultPhrase,
    GrammarNote,
    MultiChunkAnalysisResult,
    ReadingCoachResult,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _section(heading: str, level: int, body: str) -> str:
    """Return a Markdown section string (heading + blank line + body)."""
    prefix = "#" * level
    return f"{prefix} {heading}\n\n{body}\n"


def _phrase_block(phrase: DifficultPhrase) -> str:
    """Render one DifficultPhrase as a Markdown sub-section."""
    lines: list[str] = [f"### {phrase.phrase}\n"]
    lines.append(f"- **Category:** {phrase.category}")
    lines.append(f"- **Difficulty:** {phrase.difficulty_level}")
    lines.append(f"- **Why difficult:** {phrase.why_difficult}")
    if phrase.modern_spanish_equivalent is not None:
        lines.append(f"- **Modern equivalent:** {phrase.modern_spanish_equivalent}")
    if phrase.english_meaning is not None:
        lines.append(f"- **English meaning:** {phrase.english_meaning}")
    if phrase.grammar_note is not None:
        lines.append(f"- **Grammar note:** {phrase.grammar_note}")
    if phrase.learner_tip is not None:
        lines.append(f"- **Learner tip:** {phrase.learner_tip}")
    return "\n".join(lines)


def _grammar_note_block(note: GrammarNote) -> str:
    """Render one GrammarNote as a Markdown sub-section."""
    lines: list[str] = [f"### {note.topic}\n"]
    lines.append(note.explanation)
    if note.example_from_text is not None:
        lines.append(f"\n> *Example from text:* «{note.example_from_text}»")
    return "\n".join(lines)


def _comprehension_block(q: ComprehensionQuestion) -> str:
    """Render a ComprehensionQuestion as Markdown."""
    lines: list[str] = [f"**{q.question}**"]
    if q.answer_hint is not None:
        lines.append(f"\n*Hint:* {q.answer_hint}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reading_coach_result_to_markdown(
    result: ReadingCoachResult,
    title: Optional[str] = None,
) -> str:
    """Convert a ReadingCoachResult into a Markdown study-notes document.

    Parameters
    ----------
    result:
        The analysis result to render.
    title:
        Optional H1 heading placed at the top of the document.
        A ``None`` or empty string value suppresses the heading entirely.

    Returns
    -------
    str
        A UTF-8–safe Markdown string.  The function is pure: calling it
        multiple times with the same arguments always yields the same output,
        and the ``result`` object is never mutated.
    """
    parts: list[str] = []

    # --- Optional title ---
    if title:
        parts.append(f"# {title}\n")

    # --- Original Spanish + level ---
    parts.append(_section(
        "Original Spanish",
        level=2,
        body=f"{result.original_spanish}\n\n*Level: {result.overall_level}*",
    ))

    # --- Modern Spanish (optional) ---
    if result.modern_spanish is not None:
        parts.append(_section("Modern Spanish", level=2, body=result.modern_spanish))

    # --- English Gloss (optional) ---
    if result.english_gloss is not None:
        parts.append(_section("English Gloss", level=2, body=result.english_gloss))

    # --- Difficult Phrases (optional) ---
    if result.difficult_phrases:
        phrase_blocks = "\n\n".join(
            _phrase_block(p) for p in result.difficult_phrases
        )
        parts.append(_section("Difficult Phrases", level=2, body=phrase_blocks))

    # --- Grammar Notes (optional) ---
    if result.grammar_notes:
        note_blocks = "\n\n".join(
            _grammar_note_block(n) for n in result.grammar_notes
        )
        parts.append(_section("Grammar Notes", level=2, body=note_blocks))

    # --- Comprehension Question (optional) ---
    if result.comprehension_question is not None:
        parts.append(_section(
            "Comprehension Question",
            level=2,
            body=_comprehension_block(result.comprehension_question),
        ))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Multi-chunk export
# ---------------------------------------------------------------------------

def multi_chunk_result_to_markdown(
    result: MultiChunkAnalysisResult,
    title: Optional[str] = None,
) -> str:
    """Convert a MultiChunkAnalysisResult into a Markdown study-notes document.

    Parameters
    ----------
    result:
        The aggregate multi-chunk analysis result to render.
    title:
        Optional H1 heading placed at the top of the document.
        ``None`` or empty string suppresses the heading.

    Returns
    -------
    str
        A UTF-8–safe Markdown string.  Pure function — no side effects.
    """
    parts: list[str] = []

    # --- Optional title ---
    if title:
        parts.append(f"# {title}\n")

    # --- Analysis Summary ---
    summary_lines = [
        f"- **Total chunks:** {result.total_chunks}",
        f"- **Successful:** {result.successful_chunks}",
        f"- **Failed:** {result.failed_chunks}",
        f"- **Prompt version:** `{result.prompt_version}`",
    ]
    if result.checker_summary:
        summary_lines.append(f"- **Checker summary:** {result.checker_summary}")
    parts.append(_section("Analysis Summary", level=2, body="\n".join(summary_lines)))

    # --- Full original Spanish ---
    parts.append(_section("Original Spanish", level=2, body=result.original_spanish))

    # --- Per-chunk sections ---
    for chunk in result.chunks:
        parts.append("---\n")
        heading = f"Chunk {chunk.chunk_index + 1} of {result.total_chunks}"
        chunk_parts = _render_chunk(chunk)
        parts.append(_section(heading, level=2, body="\n".join(chunk_parts)))

    return "\n".join(parts)


def _render_chunk(chunk: ChunkAnalysisResult) -> list[str]:
    """Return a list of Markdown paragraph strings for one chunk."""
    parts: list[str] = []

    if chunk.succeeded and chunk.analysis is not None:
        rc: ReadingCoachResult = chunk.analysis.result

        # Source text + level
        parts.append(f"{chunk.chunk_text}\n\n*Level: {rc.overall_level}*")

        # Modern Spanish (optional)
        if rc.modern_spanish is not None:
            parts.append(_section("Modern Spanish", level=3, body=rc.modern_spanish))

        # English Gloss (optional)
        if rc.english_gloss is not None:
            parts.append(_section("English Gloss", level=3, body=rc.english_gloss))

        # Difficult Phrases (optional)
        if rc.difficult_phrases:
            phrase_blocks = "\n\n".join(_phrase_block(p) for p in rc.difficult_phrases)
            parts.append(_section("Difficult Phrases", level=3, body=phrase_blocks))

        # Grammar Notes (optional)
        if rc.grammar_notes:
            note_blocks = "\n\n".join(_grammar_note_block(n) for n in rc.grammar_notes)
            parts.append(_section("Grammar Notes", level=3, body=note_blocks))

        # Comprehension Question (optional)
        if rc.comprehension_question is not None:
            parts.append(_section(
                "Comprehension Question",
                level=3,
                body=_comprehension_block(rc.comprehension_question),
            ))

        # Checker status
        status = chunk.status or (chunk.analysis.check.status if hasattr(chunk.analysis, "check") else "")
        if status:
            parts.append(f"> **Checker:** {status}\n")

    else:
        # Failed chunk — show source text and error
        parts.append(f"{chunk.chunk_text}\n")
        parts.append("> ⚠️ *This chunk failed to analyse.*\n")
        if chunk.error:
            parts.append(f"**Error:** {chunk.error}\n")

    return parts
