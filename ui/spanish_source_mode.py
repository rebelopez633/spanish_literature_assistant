"""
ui/spanish_source_mode.py — Streamlit UI for the Spanish Source Reading Coach mode.

render_coach_mode() is called by app.py when the user selects
"Spanish Source Reading Coach" from the mode selector.  All LLM
communication goes through the injected llm_client so no Ollama
process is needed in unit tests.

The only unit-testable public helper here is make_ollama_client().
Everything that renders Streamlit widgets is tested manually / via
integration tests only.
"""
from __future__ import annotations

import streamlit as st

from reading_coach.analyzer import AnalysisResult, CoachAnalysisError, analyze_spanish_source
from reading_coach.checker import CoachCheckerConfig
from reading_coach.config import get_coach_settings
from reading_coach.errors import ReadingCoachTimeoutError
from reading_coach.llm_adapter import make_ollama_coach_client
from reading_coach.schemas import VALID_COACH_LEVELS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DENSITY_LABELS: list[str] = ["minimal", "balanced", "detailed"]

_DENSITY_HELP: dict[str, str] = {
    "minimal":  "Only the most essential phrases — 3-5 annotations.",
    "balanced": "Moderate annotation for typical study use — 5-10 annotations.",
    "detailed": "Comprehensive coverage for close reading — 10+ annotations.",
}


# ---------------------------------------------------------------------------
# Ollama client factory (unit-testable)
# ---------------------------------------------------------------------------

def make_ollama_client(host: str, model: str, timeout: float):
    """Return a ``(messages) -> str`` callable backed by the typed coach adapter.

    Delegates to :func:`reading_coach.llm_adapter.make_ollama_coach_client` so
    that network errors are mapped to typed :class:`ReadingCoachError` subclasses
    instead of leaking raw ``requests`` exceptions into the Streamlit layer:

    - ``requests.exceptions.Timeout``          → :class:`ReadingCoachTimeoutError`
    - ``requests.exceptions.RequestException`` → :class:`ReadingCoachLLMError`

    Parameters
    ----------
    host:
        Ollama base URL, e.g. ``"http://ollama:11434"``.
    model:
        Model tag, e.g. ``"qwen2.5:7b"``.
    timeout:
        Request timeout in seconds.  Use ``get_coach_settings().timeout_seconds``
        to source this from the ``READING_COACH_TIMEOUT_SECONDS`` env var.
    """
    return make_ollama_coach_client(host, model, timeout)


# ---------------------------------------------------------------------------
# Result display (pure Streamlit — no unit tests)
# ---------------------------------------------------------------------------

def _display_coach_result(analysis: AnalysisResult) -> None:
    """Render a ReadingCoachResult and its check badge in the main content area."""
    result = analysis.result
    check = analysis.check

    st.divider()

    # Quality-check badge
    _badge = {
        "passed":  "✅ Passed",
        "warning": "⚠️ Warning",
        "failed":  "❌ Failed",
    }
    st.caption(f"Quality check: {_badge.get(check.status, check.status)} — {check.summary}")
    if check.issues:
        with st.expander("Check issues"):
            for issue in check.issues:
                st.write(f"- {issue}")

    # Top-level stats
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Overall level", result.overall_level)
    with col_b:
        st.metric("Difficult phrases", len(result.difficult_phrases))

    # Optional full-text fields
    if result.modern_spanish:
        with st.expander("Modern Spanish paraphrase", expanded=False):
            st.write(result.modern_spanish)

    if result.english_gloss:
        with st.expander("English gloss", expanded=False):
            st.write(result.english_gloss)

    # Difficult phrases
    if result.difficult_phrases:
        st.subheader("Difficult phrases")
        for i, dp in enumerate(result.difficult_phrases, 1):
            with st.expander(f"{i}. {dp.phrase}  ({dp.category})", expanded=False):
                st.write(f"**Level:** {dp.difficulty_level}")
                st.write(f"**Why difficult:** {dp.why_difficult}")
                if dp.modern_spanish_equivalent:
                    st.write(f"**Modern Spanish:** {dp.modern_spanish_equivalent}")
                if dp.english_meaning:
                    st.write(f"**English:** {dp.english_meaning}")
                if dp.grammar_note:
                    st.write(f"**Grammar:** {dp.grammar_note}")
                if dp.learner_tip:
                    st.write(f"**Tip:** {dp.learner_tip}")

    # Grammar notes
    if result.grammar_notes:
        st.subheader("Grammar notes")
        for gn in result.grammar_notes:
            with st.expander(gn.topic, expanded=False):
                st.write(gn.explanation)
                if gn.example_from_text:
                    st.code(gn.example_from_text, language=None)

    # Comprehension question
    if result.comprehension_question:
        st.subheader("Comprehension question")
        st.write(result.comprehension_question.question)
        if result.comprehension_question.answer_hint:
            with st.expander("Answer hint", expanded=False):
                st.write(result.comprehension_question.answer_hint)


# ---------------------------------------------------------------------------
# Main render function (called from app.py)
# ---------------------------------------------------------------------------

def render_coach_mode(
    *,
    ollama_host: str,
    ollama_model: str,
    timeout: float,
) -> None:
    """Render the complete Spanish Source Reading Coach UI panel.

    Called from app.py immediately before ``st.stop()`` when the user has
    selected the Reading Coach app mode.  All Streamlit output rendered here
    replaces the parallel-reader main content area.
    """
    st.subheader("Spanish Source Reading Coach")
    st.caption("Paste a Spanish passage and let the coach annotate it for your level.")

    # --- Input controls ---------------------------------------------------
    source_text: str = st.text_area(
        "Spanish source text",
        height=280,
        placeholder="Paste original Spanish here — novel excerpt, poem, article…",
        key="coach_source_text",
    )

    ctrl_cols = st.columns([1, 1, 1, 1])
    with ctrl_cols[0]:
        reader_level: str = st.selectbox(
            "Reader level",
            list(VALID_COACH_LEVELS),
            index=list(VALID_COACH_LEVELS).index("B1"),
            key="coach_reader_level",
        )
    with ctrl_cols[1]:
        annotation_density: str = st.selectbox(
            "Annotation density",
            DENSITY_LABELS,
            index=DENSITY_LABELS.index("balanced"),
            key="coach_annotation_density",
        )
    with ctrl_cols[2]:
        include_modern_spanish: bool = st.checkbox(
            "Modern Spanish",
            value=True,
            key="coach_include_modern_spanish",
        )
    with ctrl_cols[3]:
        include_english_gloss: bool = st.checkbox(
            "English gloss",
            value=True,
            key="coach_include_english_gloss",
        )

    analyse_clicked: bool = st.button(
        "Analyze",
        type="primary",
        disabled=not (source_text or "").strip(),
        key="coach_analyze_btn",
    )

    # --- Analysis on button click -----------------------------------------
    if analyse_clicked and (source_text or "").strip():
        with st.spinner("Analyzing passage…"):
            try:
                client = make_ollama_client(ollama_host, ollama_model, timeout)
                _settings = get_coach_settings()
                analysis = analyze_spanish_source(
                    source_text.strip(),
                    reader_level=reader_level,
                    annotation_density=annotation_density,
                    include_english_gloss=include_english_gloss,
                    include_modern_spanish=include_modern_spanish,
                    llm_client=client,
                    checker_config=CoachCheckerConfig(
                        include_english_gloss=include_english_gloss,
                        max_annotations=_settings.max_annotations,
                    ),
                )
                st.session_state.coach_analysis = analysis
            except ReadingCoachTimeoutError as exc:
                st.error(exc.user_message)
                st.session_state.pop("coach_analysis", None)
            except CoachAnalysisError as exc:
                st.error(f"Could not parse model response: {exc}")
                st.session_state.pop("coach_analysis", None)
            except Exception as exc:  # noqa: BLE001 — surface connection errors to user
                st.error(
                    f"Analysis failed: {exc}\n\n"
                    "Check that Ollama is running and the model is loaded."
                )
                st.session_state.pop("coach_analysis", None)

    # --- Display last result (persists across reruns) ----------------------
    analysis: AnalysisResult | None = st.session_state.get("coach_analysis")
    if analysis is not None:
        _display_coach_result(analysis)
