"""
Tests for reading_coach/smoke_passages.py — passage registry, CLI parsing,
and formatting helpers.

Pure unit tests: no Ollama, no network, no external services.
"""
from __future__ import annotations

import types

import pytest

from reading_coach.smoke_passages import (
    PASSAGES,
    SmokePassage,
    format_passage_header,
    format_phrase_summary,
    list_passage_names,
    parse_cli_args,
    resolve_passages,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_DENSITIES = frozenset({"minimal", "balanced", "detailed"})
_VALID_LEVELS = frozenset({"A1", "A2", "B1", "B2", "C1", "C2"})


def _stub_phrase(phrase="test", category="idiom", difficulty_level="B1"):
    return types.SimpleNamespace(
        phrase=phrase,
        category=category,
        difficulty_level=difficulty_level,
    )


def _default_args(**kwargs):
    """Return a Namespace with all parse_cli_args defaults, optionally overridden."""
    import argparse
    ns = argparse.Namespace(
        selected_passages=None,
        all_passages=False,
        list_only=False,
    )
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


# ===========================================================================
# TestSmokePassageDataclass
# ===========================================================================

class TestSmokePassageDataclass:
    def test_frozen_cannot_assign(self):
        p = SmokePassage(name="x", category="c", text="t", description="d")
        with pytest.raises((TypeError, AttributeError)):
            p.name = "y"  # type: ignore[misc]

    def test_default_reader_level(self):
        p = SmokePassage(name="x", category="c", text="t", description="d")
        assert p.reader_level == "B1"

    def test_default_annotation_density(self):
        p = SmokePassage(name="x", category="c", text="t", description="d")
        assert p.annotation_density == "balanced"

    def test_default_include_modern_spanish(self):
        p = SmokePassage(name="x", category="c", text="t", description="d")
        assert p.include_modern_spanish is True

    def test_default_include_english_gloss(self):
        p = SmokePassage(name="x", category="c", text="t", description="d")
        assert p.include_english_gloss is True

    def test_custom_fields_preserved(self):
        p = SmokePassage(
            name="n", category="cat", text="txt", description="desc",
            reader_level="C1", annotation_density="detailed",
            include_modern_spanish=False, include_english_gloss=False,
        )
        assert p.reader_level == "C1"
        assert p.annotation_density == "detailed"
        assert p.include_modern_spanish is False
        assert p.include_english_gloss is False


# ===========================================================================
# TestPassageRegistry
# ===========================================================================

class TestPassageRegistry:
    def test_five_required_passages_present(self):
        required = {
            "modern-short",
            "golden-age",
            "mystical-religious",
            "zero-phrases",
            "subordinate-clauses",
        }
        assert required <= set(PASSAGES.keys())

    def test_all_names_match_dict_keys(self):
        for key, passage in PASSAGES.items():
            assert passage.name == key

    def test_all_texts_non_empty(self):
        for name, p in PASSAGES.items():
            assert p.text.strip(), f"{name}: text is empty"

    def test_all_descriptions_non_empty(self):
        for name, p in PASSAGES.items():
            assert p.description.strip(), f"{name}: description is empty"

    def test_all_reader_levels_valid_cefr(self):
        for name, p in PASSAGES.items():
            assert p.reader_level in _VALID_LEVELS, (
                f"{name}: invalid reader_level {p.reader_level!r}"
            )

    def test_all_densities_valid(self):
        for name, p in PASSAGES.items():
            assert p.annotation_density in _VALID_DENSITIES, (
                f"{name}: invalid density {p.annotation_density!r}"
            )

    def test_modern_short_is_short(self):
        """The modern-short passage should be meaningfully shorter than 200 chars."""
        p = PASSAGES["modern-short"]
        assert len(p.text) < 200

    def test_modern_short_level_is_beginner(self):
        assert PASSAGES["modern-short"].reader_level in {"A1", "A2"}

    def test_mystical_religious_level_is_advanced(self):
        assert PASSAGES["mystical-religious"].reader_level in {"C1", "C2"}

    def test_zero_phrases_level_is_beginner(self):
        assert PASSAGES["zero-phrases"].reader_level in {"A1", "A2"}

    def test_zero_phrases_density_is_minimal(self):
        assert PASSAGES["zero-phrases"].annotation_density == "minimal"

    def test_mystical_religious_density_is_detailed(self):
        assert PASSAGES["mystical-religious"].annotation_density == "detailed"

    def test_golden_age_contains_cervantes_text(self):
        p = PASSAGES["golden-age"]
        assert "Mancha" in p.text

    def test_subordinate_clauses_text_is_long(self):
        """The subordinate-clauses passage should be substantially longer than 100 chars."""
        assert len(PASSAGES["subordinate-clauses"].text) > 100

    def test_list_passage_names_returns_all_five(self):
        names = list_passage_names()
        assert len(names) == len(PASSAGES)

    def test_list_passage_names_are_strings(self):
        for name in list_passage_names():
            assert isinstance(name, str)

    def test_list_passage_names_stable_across_calls(self):
        assert list_passage_names() == list_passage_names()

    def test_list_passage_names_all_in_passages(self):
        for name in list_passage_names():
            assert name in PASSAGES


# ===========================================================================
# TestParseCLIArgs
# ===========================================================================

class TestParseCLIArgs:
    def test_no_args_has_no_selected_passages(self):
        args = parse_cli_args([])
        assert args.selected_passages is None

    def test_no_args_all_passages_false(self):
        args = parse_cli_args([])
        assert args.all_passages is False

    def test_no_args_list_only_false(self):
        args = parse_cli_args([])
        assert args.list_only is False

    def test_passage_flag_single(self):
        args = parse_cli_args(["--passage", "golden-age"])
        assert args.selected_passages == ["golden-age"]

    def test_passage_flag_multiple(self):
        args = parse_cli_args(["--passage", "golden-age", "modern-short"])
        assert args.selected_passages == ["golden-age", "modern-short"]

    def test_all_flag_sets_all_passages(self):
        args = parse_cli_args(["--all"])
        assert args.all_passages is True

    def test_list_flag_sets_list_only(self):
        args = parse_cli_args(["--list"])
        assert args.list_only is True

    def test_unknown_passage_name_raises_system_exit(self):
        with pytest.raises(SystemExit) as exc_info:
            parse_cli_args(["--passage", "not-a-real-passage"])
        assert exc_info.value.code != 0

    def test_passage_and_all_are_mutually_exclusive(self):
        with pytest.raises(SystemExit) as exc_info:
            parse_cli_args(["--passage", "golden-age", "--all"])
        assert exc_info.value.code != 0

    def test_list_can_combine_with_nothing(self):
        """--list on its own must not raise."""
        args = parse_cli_args(["--list"])
        assert args.list_only is True


# ===========================================================================
# TestResolvePassages
# ===========================================================================

class TestResolvePassages:
    def test_no_selection_returns_golden_age(self):
        args = _default_args()
        passages = resolve_passages(args)
        assert len(passages) == 1
        assert passages[0].name == "golden-age"

    def test_all_flag_returns_all_passages(self):
        args = _default_args(all_passages=True)
        passages = resolve_passages(args)
        assert len(passages) == len(PASSAGES)

    def test_all_flag_returns_passages_in_canonical_order(self):
        args = _default_args(all_passages=True)
        passages = resolve_passages(args)
        assert [p.name for p in passages] == list_passage_names()

    def test_specific_passage_selected(self):
        args = _default_args(selected_passages=["mystical-religious"])
        passages = resolve_passages(args)
        assert len(passages) == 1
        assert passages[0].name == "mystical-religious"

    def test_multiple_passages_selected(self):
        args = _default_args(selected_passages=["modern-short", "zero-phrases"])
        passages = resolve_passages(args)
        assert [p.name for p in passages] == ["modern-short", "zero-phrases"]

    def test_returns_smoke_passage_instances(self):
        args = _default_args()
        passages = resolve_passages(args)
        for p in passages:
            assert isinstance(p, SmokePassage)


# ===========================================================================
# TestFormatPassageHeader
# ===========================================================================

class TestFormatPassageHeader:
    def _header(self, name="golden-age", model="qwen2.5:7b", version="spanish_source_v2"):
        return format_passage_header(PASSAGES[name], model, version)

    def test_contains_passage_name(self):
        assert "golden-age" in self._header()

    def test_contains_model(self):
        assert "qwen2.5:7b" in self._header()

    def test_contains_prompt_version(self):
        assert "spanish_source_v2" in self._header()

    def test_contains_category(self):
        header = self._header("mystical-religious")
        assert "mystical" in header.lower()

    def test_returns_single_line(self):
        assert "\n" not in self._header()

    def test_different_models_reflected(self):
        h1 = self._header(model="qwen2.5:7b")
        h2 = self._header(model="llama3:8b")
        assert h1 != h2


# ===========================================================================
# TestFormatPhraseSummary
# ===========================================================================

class TestFormatPhraseSummary:
    def test_empty_list_returns_no_phrases_string(self):
        result = format_phrase_summary([])
        assert "no difficult phrases" in result.lower()

    def test_single_phrase_shown(self):
        phrase = _stub_phrase("vivía", "archaic_vocab", "B1")
        result = format_phrase_summary([phrase])
        assert "vivía" in result

    def test_phrase_includes_category(self):
        phrase = _stub_phrase("vivía", "archaic_vocab", "B1")
        result = format_phrase_summary([phrase])
        assert "archaic_vocab" in result

    def test_phrase_includes_difficulty_level(self):
        phrase = _stub_phrase("vivía", "archaic_vocab", "B2")
        result = format_phrase_summary([phrase])
        assert "B2" in result

    def test_max_phrases_respected(self):
        phrases = [_stub_phrase(f"phrase{i}", "idiom", "B1") for i in range(6)]
        result = format_phrase_summary(phrases, max_phrases=3)
        assert "phrase0" in result
        assert "phrase1" in result
        assert "phrase2" in result
        assert "phrase3" not in result

    def test_remainder_count_shown(self):
        phrases = [_stub_phrase(f"p{i}", "idiom", "B1") for i in range(5)]
        result = format_phrase_summary(phrases, max_phrases=2)
        assert "3 more" in result

    def test_no_remainder_line_when_all_shown(self):
        phrases = [_stub_phrase(f"p{i}", "idiom", "B1") for i in range(2)]
        result = format_phrase_summary(phrases, max_phrases=3)
        assert "more" not in result

    def test_exactly_max_shows_no_remainder(self):
        phrases = [_stub_phrase(f"p{i}", "idiom", "B1") for i in range(3)]
        result = format_phrase_summary(phrases, max_phrases=3)
        assert "more" not in result

    def test_returns_string(self):
        assert isinstance(format_phrase_summary([]), str)
