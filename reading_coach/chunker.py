"""
reading_coach/chunker.py — Pure text splitter for multi-chunk analysis.

Splits a Spanish passage into chunks of at most *max_chars* characters.
Split priority (highest to lowest):

1. Paragraph boundaries (``\\n\\n``)
2. Sentence boundaries (period / question / exclamation followed by whitespace)
3. Hard character split (last resort; preserves all content)

No Streamlit, no Ollama, no external services.  Fully pure and testable.
"""
from __future__ import annotations

import re


def split_into_chunks(text: str, max_chars: int = 2200) -> list[str]:
    """Split *text* into chunks of at most *max_chars* characters.

    Parameters
    ----------
    text:
        The Spanish passage to split.  May contain paragraphs (``\\n\\n``)
        and/or sentence-ending punctuation (``.``, ``?``, ``!``).
    max_chars:
        Maximum characters per chunk.  Defaults to ``2200`` — the value
        of ``MAX_CHARS_PER_CHUNK`` used by the Streamlit app.

    Returns
    -------
    list[str]
        Ordered list of non-empty chunks.  Returns ``[]`` for blank input.
        Each chunk satisfies ``len(chunk) <= max_chars`` except in the
        degenerate case where a single token (e.g. a very long URL) exceeds
        *max_chars*; in that case the token is hard-split at the boundary.
    """
    if not text.strip():
        return []

    # Fast path: entire text fits in one chunk.
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > max_chars:
            # Paragraph is too large → flush current and split by sentences.
            if current:
                chunks.append(current)
                current = ""
            for sent in _split_sentences(para):
                if len(sent) > max_chars:
                    # Sentence is too large → flush current and hard-split.
                    if current:
                        chunks.append(current)
                        current = ""
                    for i in range(0, len(sent), max_chars):
                        chunks.append(sent[i : i + max_chars])
                elif current and len(current) + 1 + len(sent) > max_chars:
                    chunks.append(current)
                    current = sent
                else:
                    current = (current + " " + sent) if current else sent
        elif current and len(current) + 2 + len(para) > max_chars:
            chunks.append(current)
            current = para
        else:
            current = (current + "\n\n" + para) if current else para

    if current.strip():
        chunks.append(current)

    return [c for c in chunks if c.strip()]


def _split_sentences(text: str) -> list[str]:
    """Split *text* at sentence boundaries (``[.?!]`` followed by whitespace)."""
    parts = re.split(r"(?<=[.?!])\s+", text)
    return [p.strip() for p in parts if p.strip()]
