"""
Tests for reading_coach/chunking.py — SpanishChunk planner.

Acceptance criteria from Phase 3 slice 3:
  1. Short text produces one chunk.
  2. Paragraph text chunks on paragraph boundaries.
  3. Long paragraph falls back without dropping characters.
  4. Concatenating chunk.text reconstructs original source exactly.
  5. start_char/end_char match original slices.
  6. Empty/blank input behaviour is explicit and tested.
  7. max_chars validation is tested.

Additional:
  - chunk_id format is stable and unique.
  - index is sequential from 0.
  - preserve_paragraphs=False bypasses \\n\\n boundary preference.
  - Hard split (no natural boundaries) produces correct results.
  - One-character max_chars edge case.

Pure unit tests — no Ollama, no network.
"""
from __future__ import annotations

import pytest

from reading_coach.chunking import SpanishChunk, plan_spanish_chunks

pytestmark = pytest.mark.unit


# ===========================================================================
# Helpers
# ===========================================================================

def _reconstruct(chunks: list[SpanishChunk]) -> str:
    return "".join(c.text for c in chunks)


# ===========================================================================
# TestSpanishChunkModel
# ===========================================================================

class TestSpanishChunkModel:
    def test_has_chunk_id(self):
        c = SpanishChunk(chunk_id="chunk_0000", index=0, text="Hola.", start_char=0, end_char=5)
        assert c.chunk_id == "chunk_0000"

    def test_has_index(self):
        c = SpanishChunk(chunk_id="chunk_0000", index=0, text="Hola.", start_char=0, end_char=5)
        assert c.index == 0

    def test_has_text(self):
        c = SpanishChunk(chunk_id="chunk_0000", index=0, text="Hola.", start_char=0, end_char=5)
        assert c.text == "Hola."

    def test_has_start_char(self):
        c = SpanishChunk(chunk_id="chunk_0000", index=0, text="Hola.", start_char=3, end_char=8)
        assert c.start_char == 3

    def test_has_end_char(self):
        c = SpanishChunk(chunk_id="chunk_0000", index=0, text="Hola.", start_char=0, end_char=5)
        assert c.end_char == 5

    def test_chunk_id_is_str(self):
        c = SpanishChunk(chunk_id="x", index=0, text="t", start_char=0, end_char=1)
        assert isinstance(c.chunk_id, str)


# ===========================================================================
# TestInputValidation (acceptance criterion 6 + 7)
# ===========================================================================

class TestInputValidation:
    def test_empty_string_returns_empty_list(self):
        assert plan_spanish_chunks("", max_chars=100) == []

    def test_whitespace_only_returns_empty_list(self):
        assert plan_spanish_chunks("   ", max_chars=100) == []

    def test_newlines_only_returns_empty_list(self):
        assert plan_spanish_chunks("\n\n\n", max_chars=100) == []

    def test_mixed_whitespace_returns_empty_list(self):
        assert plan_spanish_chunks("   \n\n  \t  ", max_chars=100) == []

    def test_max_chars_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="max_chars"):
            plan_spanish_chunks("texto", max_chars=0)

    def test_max_chars_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="max_chars"):
            plan_spanish_chunks("texto", max_chars=-10)

    def test_max_chars_one_is_valid(self):
        """max_chars=1 is the minimum valid value; should not raise."""
        chunks = plan_spanish_chunks("AB", max_chars=1)
        assert _reconstruct(chunks) == "AB"

    def test_return_type_is_list(self):
        chunks = plan_spanish_chunks("Hola.", max_chars=100)
        assert isinstance(chunks, list)

    def test_elements_are_spanish_chunk(self):
        chunks = plan_spanish_chunks("Hola.", max_chars=100)
        for c in chunks:
            assert isinstance(c, SpanishChunk)


# ===========================================================================
# TestShortText (acceptance criterion 1)
# ===========================================================================

class TestShortText:
    def test_short_text_produces_one_chunk(self):
        text = "En un lugar de la Mancha."
        chunks = plan_spanish_chunks(text, max_chars=200)
        assert len(chunks) == 1

    def test_short_text_chunk_text_equals_source(self):
        text = "En un lugar de la Mancha."
        chunks = plan_spanish_chunks(text, max_chars=200)
        assert chunks[0].text == text

    def test_short_text_start_char_is_zero(self):
        text = "Hola mundo."
        chunks = plan_spanish_chunks(text, max_chars=100)
        assert chunks[0].start_char == 0

    def test_short_text_end_char_is_len(self):
        text = "Hola mundo."
        chunks = plan_spanish_chunks(text, max_chars=100)
        assert chunks[0].end_char == len(text)

    def test_short_text_chunk_id_is_chunk_0000(self):
        text = "Hola."
        chunks = plan_spanish_chunks(text, max_chars=100)
        assert chunks[0].chunk_id == "chunk_0000"

    def test_short_text_index_is_zero(self):
        text = "Hola."
        chunks = plan_spanish_chunks(text, max_chars=100)
        assert chunks[0].index == 0

    def test_text_exactly_max_chars_is_one_chunk(self):
        text = "a" * 50
        chunks = plan_spanish_chunks(text, max_chars=50)
        assert len(chunks) == 1
        assert chunks[0].text == text


# ===========================================================================
# TestParagraphSplit (acceptance criterion 2)
# ===========================================================================

class TestParagraphSplit:
    def test_paragraph_text_splits_into_multiple_chunks(self):
        # "ABC.\n\n" = 6 chars, "DEF." = 4 chars; max_chars=7 forces a split
        text = "ABC.\n\nDEF."
        chunks = plan_spanish_chunks(text, max_chars=7)
        assert len(chunks) > 1

    def test_paragraph_split_at_double_newline(self):
        # window[0:7] = "ABC.\n\nD" → rfind('\n\n') at 4 → end=6
        text = "ABC.\n\nDEF."
        chunks = plan_spanish_chunks(text, max_chars=7)
        assert chunks[0].text == "ABC.\n\n"
        assert chunks[1].text == "DEF."

    def test_paragraph_boundary_not_dropped(self):
        text = "Para uno.\n\nPara dos.\n\nPara tres."
        chunks = plan_spanish_chunks(text, max_chars=15)
        assert _reconstruct(chunks) == text

    def test_multiple_paragraphs_all_present(self):
        text = "Para uno.\n\nPara dos.\n\nPara tres."
        chunks = plan_spanish_chunks(text, max_chars=15)
        assert any("Para uno." in c.text for c in chunks)
        assert any("Para dos." in c.text for c in chunks)
        assert any("Para tres." in c.text for c in chunks)

    def test_paragraph_newlines_preserved_in_chunk_text(self):
        text = "Primero.\n\nSegundo."
        chunks = plan_spanish_chunks(text, max_chars=12)
        # \n\n must appear in exactly one chunk (not dropped)
        combined = _reconstruct(chunks)
        assert "\n\n" in combined
        assert combined.count("\n\n") == text.count("\n\n")

    def test_preserve_paragraphs_false_does_not_split_at_double_newline(self):
        # With preserve_paragraphs=False, \n\n is not a preferred split point.
        # "ABC.\n\nD" window → sentence boundary at '.' + '\n' → end=5 ("ABC.\n")
        text = "ABC.\n\nDEF."
        chunks = plan_spanish_chunks(text, max_chars=7, preserve_paragraphs=False)
        # First chunk must NOT be "ABC.\n\n" (that would be the paragraph split)
        assert chunks[0].text != "ABC.\n\n"
        # But reconstruction must still be exact
        assert _reconstruct(chunks) == text

    def test_short_paragraph_text_fits_without_splitting(self):
        text = "Short.\n\nAlso short."
        chunks = plan_spanish_chunks(text, max_chars=200)
        assert len(chunks) == 1
        assert chunks[0].text == text


# ===========================================================================
# TestSentenceFallback (acceptance criterion 3)
# ===========================================================================

class TestSentenceFallback:
    def test_long_paragraph_produces_multiple_chunks(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = plan_spanish_chunks(text, max_chars=20)
        assert len(chunks) > 1

    def test_sentence_fallback_preserves_all_text(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = plan_spanish_chunks(text, max_chars=20)
        assert _reconstruct(chunks) == text

    def test_sentence_split_includes_terminator(self):
        # Sentence terminator '.' and following space must appear in output
        text = "Hello. World."
        chunks = plan_spanish_chunks(text, max_chars=8)
        # "Hello. " is 7 chars ≤ 8; window[0:8] = "Hello. W"
        # sentence boundary at i=5 ('.')+i+1 (' ') → return 0+5+2=7
        assert chunks[0].text == "Hello. "
        assert chunks[1].text == "World."

    def test_question_mark_is_sentence_boundary(self):
        text = "¿Hola? Mundo."
        chunks = plan_spanish_chunks(text, max_chars=9)
        assert _reconstruct(chunks) == text

    def test_exclamation_mark_is_sentence_boundary(self):
        text = "Hola! Mundo."
        chunks = plan_spanish_chunks(text, max_chars=8)
        assert _reconstruct(chunks) == text

    def test_long_sentence_no_natural_boundary_hard_splits(self):
        """A single very long sentence with no punctuation falls back to hard split."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 26 chars, no spaces or punctuation
        chunks = plan_spanish_chunks(text, max_chars=10)
        assert len(chunks) > 1
        assert _reconstruct(chunks) == text
        for c in chunks:
            assert len(c.text) <= 10


# ===========================================================================
# TestReconstructionInvariant (acceptance criterion 4)
# ===========================================================================

class TestReconstructionInvariant:
    def _assert_reconstructs(self, text: str, max_chars: int, **kwargs) -> None:
        chunks = plan_spanish_chunks(text, max_chars=max_chars, **kwargs)
        assert _reconstruct(chunks) == text, (
            f"Reconstruction failed for max_chars={max_chars!r}"
        )

    def test_simple_prose(self):
        self._assert_reconstructs(
            "En un lugar de la Mancha, de cuyo nombre no quiero acordarme.",
            max_chars=25,
        )

    def test_multi_paragraph(self):
        self._assert_reconstructs(
            "Párrafo uno.\n\nPárrafo dos.\n\nPárrafo tres.",
            max_chars=15,
        )

    def test_no_punctuation(self):
        self._assert_reconstructs("abcde fghij klmno pqrst", max_chars=8)

    def test_no_spaces_or_punctuation(self):
        self._assert_reconstructs("abcdefghijklmnopqrstuvwxyz", max_chars=6)

    def test_repeated_newlines(self):
        self._assert_reconstructs("A.\n\n\nB.\n\nC.", max_chars=5)

    def test_trailing_whitespace_preserved(self):
        text = "Sentence one.   "
        chunks = plan_spanish_chunks(text, max_chars=100)
        assert _reconstruct(chunks) == text

    def test_leading_whitespace_preserved_in_chunk(self):
        """Whitespace at the start of a chunk (after a split) is not stripped."""
        text = "Hello. World."
        chunks = plan_spanish_chunks(text, max_chars=8)
        # Reconstruction must include every character
        assert _reconstruct(chunks) == text

    def test_preserve_paragraphs_false(self):
        self._assert_reconstructs(
            "Párrafo uno.\n\nPárrafo dos.", max_chars=14, preserve_paragraphs=False
        )

    def test_max_chars_one(self):
        text = "ABC"
        self._assert_reconstructs(text, max_chars=1)


# ===========================================================================
# TestCharOffsets (acceptance criterion 5)
# ===========================================================================

class TestCharOffsets:
    def test_start_end_slice_matches_chunk_text(self):
        text = "ABCD\n\nEFGH\n\nIJKL"
        chunks = plan_spanish_chunks(text, max_chars=8)
        for chunk in chunks:
            assert text[chunk.start_char:chunk.end_char] == chunk.text

    def test_first_chunk_start_char_is_zero(self):
        text = "Some text here."
        chunks = plan_spanish_chunks(text, max_chars=8)
        assert chunks[0].start_char == 0

    def test_last_chunk_end_char_is_len_source(self):
        text = "Some text here and beyond."
        chunks = plan_spanish_chunks(text, max_chars=8)
        assert chunks[-1].end_char == len(text)

    def test_chunks_are_contiguous(self):
        """end_char of chunk N equals start_char of chunk N+1."""
        text = "Primera frase. Segunda frase. Tercera frase."
        chunks = plan_spanish_chunks(text, max_chars=16)
        for a, b in zip(chunks, chunks[1:]):
            assert a.end_char == b.start_char

    def test_no_gap_between_chunks(self):
        text = "ABCDEFGHIJKLMNOP"
        chunks = plan_spanish_chunks(text, max_chars=5)
        positions = [(c.start_char, c.end_char) for c in chunks]
        # Each pair must be contiguous
        for i in range(len(positions) - 1):
            assert positions[i][1] == positions[i + 1][0]

    def test_single_chunk_covers_full_text(self):
        text = "Short text."
        chunks = plan_spanish_chunks(text, max_chars=100)
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == len(text)

    def test_end_char_minus_start_char_equals_len_text(self):
        text = "Hola mundo."
        chunks = plan_spanish_chunks(text, max_chars=5)
        for chunk in chunks:
            assert chunk.end_char - chunk.start_char == len(chunk.text)


# ===========================================================================
# TestChunkIds
# ===========================================================================

class TestChunkIds:
    def test_chunk_ids_are_unique(self):
        text = "A. B. C. D. E. F. G. H."
        chunks = plan_spanish_chunks(text, max_chars=6)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_id_format_is_chunk_nnnn(self):
        text = "A. B. C. D."
        chunks = plan_spanish_chunks(text, max_chars=4)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"chunk_{i:04d}"

    def test_chunk_ids_stable_across_calls(self):
        text = "A. B. C. D."
        chunks1 = plan_spanish_chunks(text, max_chars=4)
        chunks2 = plan_spanish_chunks(text, max_chars=4)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_chunk_index_is_sequential(self):
        text = "A. B. C. D. E."
        chunks = plan_spanish_chunks(text, max_chars=5)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_single_chunk_id_is_chunk_0000(self):
        chunks = plan_spanish_chunks("Short.", max_chars=100)
        assert chunks[0].chunk_id == "chunk_0000"
