"""
Tests for reading_coach/chunker.py — pure text splitter.

Pure unit tests — no Ollama, no network, no external services.
"""
from __future__ import annotations

import pytest

from reading_coach.chunker import split_into_chunks

pytestmark = pytest.mark.unit


class TestSplitIntoChunks:
    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_string_returns_empty_list(self):
        assert split_into_chunks("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert split_into_chunks("   \n\n  ") == []

    def test_single_newline_only_returns_empty_list(self):
        assert split_into_chunks("\n") == []

    # ------------------------------------------------------------------
    # Short text (fits in one chunk)
    # ------------------------------------------------------------------

    def test_short_text_returns_single_chunk(self):
        text = "Hola mundo."
        chunks = split_into_chunks(text, max_chars=100)
        assert chunks == [text]

    def test_text_exactly_max_chars_returns_single_chunk(self):
        text = "a" * 100
        assert split_into_chunks(text, max_chars=100) == [text]

    def test_text_one_under_max_returns_single_chunk(self):
        text = "a" * 99
        assert len(split_into_chunks(text, max_chars=100)) == 1

    # ------------------------------------------------------------------
    # Multi-chunk splitting
    # ------------------------------------------------------------------

    def test_long_text_produces_multiple_chunks(self):
        # Many short sentences separated by \n\n — easily splits
        sentences = [f"Oración {i}." for i in range(20)]
        text = "\n\n".join(sentences)
        chunks = split_into_chunks(text, max_chars=40)
        assert len(chunks) > 1

    def test_each_chunk_within_max_chars(self):
        text = "Esta es una oración larga. " * 50
        max_chars = 200
        chunks = split_into_chunks(text, max_chars=max_chars)
        for i, chunk in enumerate(chunks):
            assert len(chunk) <= max_chars, (
                f"Chunk {i} has {len(chunk)} chars, exceeds max {max_chars}"
            )

    def test_no_empty_chunks(self):
        text = "Hola mundo.\n\nAdios.\n\nBuenas noches."
        chunks = split_into_chunks(text, max_chars=20)
        for chunk in chunks:
            assert chunk.strip(), "Empty chunk found"

    def test_chunks_are_strings(self):
        chunks = split_into_chunks("Hola.", max_chars=100)
        for c in chunks:
            assert isinstance(c, str)

    # ------------------------------------------------------------------
    # Content preservation
    # ------------------------------------------------------------------

    def test_content_preserved_paragraph_split(self):
        """All paragraph sentences survive chunking when split on \\n\\n."""
        sentences = [f"Esta es la oración número {i}." for i in range(15)]
        text = "\n\n".join(sentences)
        chunks = split_into_chunks(text, max_chars=60)
        for sentence in sentences:
            assert any(sentence in chunk for chunk in chunks), (
                f"Sentence lost: {sentence!r}"
            )

    def test_content_preserved_sentence_split(self):
        """Words survive when splitting on sentence boundaries."""
        words = [f"palabra{i}" for i in range(10)]
        text = ". ".join(words) + "."
        chunks = split_into_chunks(text, max_chars=50)
        joined = "".join(chunks)
        for word in words:
            assert word in joined, f"Word {word!r} lost during chunking"

    def test_hard_split_preserves_content(self):
        """A single word longer than max_chars must not be dropped."""
        long_word = "x" * 300
        chunks = split_into_chunks(long_word, max_chars=100)
        assert "".join(chunks) == long_word

    # ------------------------------------------------------------------
    # Boundary and special cases
    # ------------------------------------------------------------------

    def test_paragraph_split_respects_double_newlines(self):
        """Two paragraphs each short enough → can end up in one or two chunks."""
        para1 = "Primera parte del texto."
        para2 = "Segunda parte del texto."
        text = para1 + "\n\n" + para2
        chunks = split_into_chunks(text, max_chars=200)
        # Both paragraphs must be present somewhere in the chunks
        combined = "".join(chunks)
        assert para1 in combined
        assert para2 in combined

    def test_sentence_boundary_split_on_period(self):
        """Sentences separated by period-space are split correctly."""
        text = "Primera oración. Segunda oración. Tercera oración."
        chunks = split_into_chunks(text, max_chars=30)
        # Max sentence length is ~19 chars ("Segunda oración.")
        for chunk in chunks:
            assert len(chunk) <= 30

    def test_single_paragraph_long_splits_into_multiple(self):
        """A single long sentence with no split points gets hard-split."""
        text = "a" * 250
        chunks = split_into_chunks(text, max_chars=100)
        assert len(chunks) >= 2
        assert all(len(c) <= 100 for c in chunks)

    def test_default_max_chars_accepts_typical_passage(self):
        """A typical 200-char passage fits in one chunk at the 2200-char default."""
        text = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo."
        chunks = split_into_chunks(text)  # default max_chars=2200
        assert len(chunks) == 1
