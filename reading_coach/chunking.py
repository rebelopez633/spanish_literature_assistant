"""reading_coach/chunking.py — Chunk planner for Spanish source text.

Returns SpanishChunk objects with exact character offsets.

Guarantees:
  - Reconstruction: "".join(c.text for c in chunks) == source_text (non-blank input)
  - Offset:         source_text[c.start_char:c.end_char] == c.text  (every chunk)
  - Validation:     max_chars < 1 raises ValueError
  - Empty behaviour: empty or whitespace-only source_text returns []
"""
from __future__ import annotations

from pydantic import BaseModel


class SpanishChunk(BaseModel):
    """A contiguous slice of Spanish source text with position metadata."""

    chunk_id: str    # "chunk_0000", "chunk_0001", …
    index: int       # 0-based sequential position
    text: str        # source_text[start_char:end_char]
    start_char: int  # inclusive offset into original source
    end_char: int    # exclusive offset into original source


def plan_spanish_chunks(
    source_text: str,
    max_chars: int,
    preserve_paragraphs: bool = True,
) -> list[SpanishChunk]:
    """Split *source_text* into SpanishChunk objects of at most *max_chars* each.

    Args:
        source_text: The Spanish text to split.
        max_chars: Maximum number of characters per chunk (must be >= 1).
        preserve_paragraphs: When True, prefer splitting on ``\\n\\n`` boundaries
            before sentence or word boundaries.

    Returns:
        A list of SpanishChunk objects.  Empty list if *source_text* is blank.

    Raises:
        ValueError: If *max_chars* is less than 1.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be at least 1, got {max_chars!r}")

    if not source_text.strip():
        return []

    if len(source_text) <= max_chars:
        return [
            SpanishChunk(
                chunk_id="chunk_0000",
                index=0,
                text=source_text,
                start_char=0,
                end_char=len(source_text),
            )
        ]

    chunks: list[SpanishChunk] = []
    start = 0
    n = len(source_text)

    while start < n:
        remaining = n - start
        if remaining <= max_chars:
            end = n
        else:
            end = _find_best_end(source_text, start, start + max_chars, preserve_paragraphs)
            if end <= start:
                end = start + max_chars  # hard fallback — always advances

        idx = len(chunks)
        chunks.append(
            SpanishChunk(
                chunk_id=f"chunk_{idx:04d}",
                index=idx,
                text=source_text[start:end],
                start_char=start,
                end_char=end,
            )
        )
        start = end

    return chunks


def _find_best_end(
    text: str,
    start: int,
    max_end: int,
    preserve_paragraphs: bool,
) -> int:
    """Return the best split position within ``text[start:max_end]``.

    Priority order:
    1. Paragraph boundary (``\\n\\n``) — when *preserve_paragraphs* is True.
    2. Sentence boundary (``[.?!]`` followed by space or newline).
    3. Word / space boundary.
    4. Hard split at *max_end* (fallback, returned as 0 to let caller apply it).

    Returns the exclusive end index into *text*.  Returns 0 if no boundary was
    found (signals the caller to use the hard-split fallback).
    """
    window = text[start:max_end]
    wlen = len(window)

    # 1. Paragraph boundary: include the \n\n in the current chunk.
    if preserve_paragraphs:
        idx = window.rfind("\n\n")
        if idx > 0:
            return start + idx + 2  # include both newlines

    # 2. Sentence boundary: '.'/'?'/'!' followed by space or newline.
    for i in range(wlen - 2, 0, -1):
        if window[i] in ".?!" and window[i + 1] in " \n":
            return start + i + 2  # include terminator + separator

    # 3. Word boundary: any space or newline.
    for i in range(wlen - 1, 0, -1):
        if window[i] in " \n":
            return start + i + 1  # include the space/newline

    # 4. No boundary found — signal hard split.
    return 0
