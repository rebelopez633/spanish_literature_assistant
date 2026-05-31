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

import re

import streamlit as st

from reading_coach.analyzer import AnalysisResult, CoachAnalysisError, analyze_spanish_source
from reading_coach.checker import CoachCheckerConfig
from reading_coach.config import get_coach_settings
from reading_coach.errors import ReadingCoachTimeoutError
from reading_coach.llm_adapter import make_ollama_coach_client
from reading_coach.saved_phrases import saved_phrase_from_difficult_phrase
from reading_coach.schemas import VALID_COACH_LEVELS, MultiChunkAnalysisResult
from reading_coach.session import CoachSession
from reading_coach.session_repository import InMemoryCoachSessionRepository
from reading_coach.study_notes import (
    multi_chunk_result_to_markdown,
    reading_coach_result_to_markdown,
)

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
# Filename helper
# ---------------------------------------------------------------------------


def _safe_filename(title: str | None) -> str:
    """Return a safe ``.md`` filename for study notes.

    Falls back to ``'spanish-reading-notes.md'`` when *title* is empty or
    ``None``.  Otherwise: lower-case, non-alphanumeric chars → hyphens,
    consecutive hyphens collapsed, edges stripped, stem truncated at 60 chars.
    """
    if not title or not title.strip():
        return "spanish-reading-notes.md"
    slug = title.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)   # drop punctuation
    slug = re.sub(r"[\s_]+", "-", slug)     # spaces/underscores → hyphens
    slug = re.sub(r"-+", "-", slug)          # collapse runs
    slug = slug.strip("-")[:60].rstrip("-")  # trim and truncate
    return f"{slug or 'spanish-reading-notes'}.md"


# ---------------------------------------------------------------------------
# Session history helpers
# ---------------------------------------------------------------------------


def _get_or_create_repo() -> InMemoryCoachSessionRepository:
    """Return the shared in-memory repository from session_state, creating it when absent.

    The repository is keyed as ``st.session_state.coach_repo`` and persists for
    the lifetime of the Streamlit app process.  It is intentionally *not* reset
    between reruns.
    """
    if "coach_repo" not in st.session_state:
        st.session_state.coach_repo = InMemoryCoachSessionRepository()
    return st.session_state.coach_repo


def _build_coach_session(
    analysis: AnalysisResult,
    *,
    source_text: str,
    reader_level: str,
    annotation_density: str,
    title: str,
) -> CoachSession:
    """Build a :class:`CoachSession` from a completed analysis.

    The full :class:`AnalysisResult` is stored in :attr:`CoachSession.result`
    so that reloading restores both the parsed result and the check outcome.
    """
    return CoachSession(
        original_spanish=source_text,
        reader_level=reader_level,
        annotation_density=annotation_density,
        prompt_version=analysis.prompt_version,
        result=analysis,
        title=title or None,
    )


def _render_save_button(
    analysis: AnalysisResult,
    *,
    source_text: str,
    reader_level: str,
    annotation_density: str,
    title: str,
    repo: InMemoryCoachSessionRepository,
) -> None:
    """Render a \"Save session\" button below the analysis result.

    On click: builds a :class:`CoachSession`, persists it to *repo*, and stores
    the saved ``session_id`` in ``st.session_state.coach_saved_session_id``.
    """
    if st.button("\U0001f4be Save session", key="coach_save_session_btn"):
        session = _build_coach_session(
            analysis,
            source_text=source_text,
            reader_level=reader_level,
            annotation_density=annotation_density,
            title=title,
        )
        saved = repo.save_session(session)
        st.session_state.coach_saved_session_id = saved.session_id
        st.success("Session saved!")


def _apply_session_reload(session: CoachSession) -> None:
    """Restore a saved :class:`CoachSession` into Streamlit session state.

    Sets ``coach_analysis`` and all relevant widget-keyed entries so that the
    next rerun re-populates the UI with the session's data.
    """
    if session.result is not None:
        st.session_state.coach_analysis = session.result
    st.session_state.coach_source_text = session.original_spanish
    st.session_state.coach_reader_level = session.reader_level
    st.session_state.coach_annotation_density = session.annotation_density
    if session.title:
        st.session_state.coach_session_title = session.title
    st.rerun()


def _render_session_history(repo: InMemoryCoachSessionRepository) -> None:
    """Render the session history panel in the sidebar.

    Shows a collapsible entry per session (newest first) with Reload and
    Delete buttons.  When the repo is empty a \"No saved sessions yet.\" hint
    is displayed instead.
    """
    sessions = repo.list_sessions()
    with st.sidebar:
        st.subheader("Session history")
        if not sessions:
            st.caption("No saved sessions yet.")
            return
        for session in sessions:
            label = (session.title or session.original_spanish[:50]).strip()
            with st.expander(label, expanded=False):
                st.caption(
                    f"{session.reader_level} \u00b7 {session.annotation_density} \u00b7 "
                    f"{session.created_at.strftime('%Y-%m-%d %H:%M')}"
                )
                col_r, col_d = st.columns(2)
                with col_r:
                    if st.button(
                        "\u21a9 Reload",
                        key=f"coach_reload_{session.session_id}",
                    ):
                        _apply_session_reload(session)
                with col_d:
                    if st.button(
                        "\U0001f5d1 Delete",
                        key=f"coach_delete_{session.session_id}",
                    ):
                        repo.delete_session(session.session_id)
                        st.rerun()


# ---------------------------------------------------------------------------
# Saved-phrase helpers
# ---------------------------------------------------------------------------


def _saved_phrases_to_markdown(phrases) -> str:
    """Render a list of :class:`~reading_coach.session.SavedPhrase` as Markdown.

    Produces a ``## Saved Phrases`` heading followed by one ``###`` subsection
    per phrase, showing all populated optional fields.
    """
    lines = ["## Saved Phrases", ""]
    for sp in phrases:
        lines.append(f"### {sp.phrase}")
        lines.append(f"**Category:** {sp.category}  ")
        lines.append(f"**Level:** {sp.difficulty_level}  ")
        if sp.english_meaning:
            lines.append(f"**English:** {sp.english_meaning}  ")
        if sp.modern_spanish_equivalent:
            lines.append(f"**Modern Spanish:** {sp.modern_spanish_equivalent}  ")
        if sp.grammar_note:
            lines.append(f"**Grammar:** {sp.grammar_note}  ")
        if sp.learner_tip:
            lines.append(f"**Tip:** {sp.learner_tip}  ")
        lines.append(f"**Source:** *{sp.source_context}*  ")
        lines.append("")
    return "\n".join(lines)


def _render_saved_phrases_panel(repo: InMemoryCoachSessionRepository) -> None:
    """Render the Saved Phrases section below the result area.

    When no phrases have been saved a short empty-state hint is shown.
    When phrases are present each is shown in a collapsible expander with all
    populated fields and a Delete button.  An export download button appears
    at the top of the list.
    """
    phrases = repo.list_saved_phrases()
    st.subheader("Saved phrases")
    if not phrases:
        st.caption("No saved phrases yet. Click \"Save phrase\" below any difficult phrase.")
        return

    # Export button (top of list)
    _md = _saved_phrases_to_markdown(phrases)
    st.download_button(
        label="\U0001f4e5 Export saved phrases (Markdown)",
        data=_md,
        file_name="saved-phrases.md",
        mime="text/markdown",
        key="coach_export_phrases_btn",
    )

    for sp in phrases:
        with st.expander(sp.phrase, expanded=False):
            st.caption(f"{sp.category} \u00b7 {sp.difficulty_level}")
            if sp.english_meaning:
                st.write(f"**English:** {sp.english_meaning}")
            if sp.modern_spanish_equivalent:
                st.write(f"**Modern Spanish:** {sp.modern_spanish_equivalent}")
            if sp.grammar_note:
                st.write(f"**Grammar:** {sp.grammar_note}")
            if sp.learner_tip:
                st.write(f"**Tip:** {sp.learner_tip}")
            st.write(f"**Source:** *{sp.source_context}*")
            if st.button(
                "\U0001f5d1 Delete phrase",
                key=f"coach_delete_phrase_{sp.phrase_id}",
            ):
                repo.delete_phrase(sp.phrase_id)
                st.rerun()


def _render_phrase_save_buttons(
    result,
    *,
    source_context: str,
    session_id: str | None,
    repo: InMemoryCoachSessionRepository,
) -> None:
    """Render a \"Save phrase\" button for each difficult phrase in *result*.

    Buttons are rendered immediately below the phrase expanders so users can
    save individual phrases without leaving the analysis view.  Each button
    uses a zero-based index key ``coach_save_phrase_{i}`` so the key is stable
    within a single render cycle even when phrase text contains special chars.
    """
    for i, dp in enumerate(result.difficult_phrases):
        if st.button(
            f"\U0001f4cc Save phrase: \u2018{dp.phrase}\u2019",
            key=f"coach_save_phrase_{i}",
        ):
            sp = saved_phrase_from_difficult_phrase(
                dp,
                source_context=source_context,
                session_id=session_id,
            )
            repo.save_phrase(sp)
            st.success(f"Phrase \u2018{dp.phrase}\u2019 saved!")


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

def _display_coach_result(
    analysis: AnalysisResult,
    title: str = "",
    *,
    repo: InMemoryCoachSessionRepository | None = None,
    source_text: str = "",
    session_id: str | None = None,
) -> None:
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

    # Diagnostics expander (only when metadata available)
    if analysis.metadata is not None:
        meta = analysis.metadata
        with st.expander("Diagnostics", expanded=False):
            st.write(f"**Prompt version:** `{meta.prompt_version}`")
            st.write(f"**Reader level:** {meta.reader_level}")
            st.write(f"**Annotation density:** {meta.annotation_density}")
            st.write(f"**Checker status:** {meta.checker_status}")
            st.write(f"**Annotation limit policy:** {meta.annotation_limit_policy}")
            st.write(f"**Max annotations:** {meta.max_annotations}")
            if meta.model_name:
                st.write(f"**Model:** {meta.model_name}")
            if meta.parser_strategy:
                st.write(f"**Parser strategy:** {meta.parser_strategy}")
            if meta.retry_attempts is not None:
                st.write(f"**Retry attempts:** {meta.retry_attempts}")

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

        if repo is not None:
            _render_phrase_save_buttons(
                result,
                source_context=source_text or result.original_spanish,
                session_id=session_id,
                repo=repo,
            )

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

    # --- Markdown download ---
    _content = reading_coach_result_to_markdown(
        result,
        title=title or None,
        metadata=analysis.metadata,
    )
    st.download_button(
        label="📥 Download study notes (Markdown)",
        data=_content,
        file_name=_safe_filename(title),
        mime="text/markdown",
    )


def _display_multi_chunk_coach_result(
    multi: MultiChunkAnalysisResult, title: str = ""
) -> None:
    """Render a summary and download button for a multi-chunk analysis result."""
    st.divider()
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Total chunks", multi.total_chunks)
    with col_b:
        st.metric("Successful", multi.successful_chunks)
    with col_c:
        st.metric("Failed", multi.failed_chunks)

    if multi.checker_summary:
        st.caption(f"Checker: {multi.checker_summary}")

    phrase_count = len(multi.all_difficult_phrases)
    if phrase_count:
        st.metric("Difficult phrases (total)", phrase_count)

    # Diagnostics expander (only when metadata available)
    if multi.metadata is not None:
        meta = multi.metadata
        with st.expander("Diagnostics", expanded=False):
            st.write(f"**Prompt version:** `{meta.prompt_version}`")
            st.write(f"**Reader level:** {meta.reader_level}")
            st.write(f"**Annotation density:** {meta.annotation_density}")
            st.write(f"**Chunks:** {meta.chunk_count} total, {meta.successful_chunk_count} successful, {meta.failed_chunk_count} failed")
            st.write(f"**Checker status:** {meta.checker_status}")
            st.write(f"**Annotation limit policy:** {meta.annotation_limit_policy}")
            st.write(f"**Max annotations:** {meta.max_annotations}")
            if meta.model_name:
                st.write(f"**Model:** {meta.model_name}")

    # --- Markdown download ---
    _content = multi_chunk_result_to_markdown(multi, title=title or None)
    st.download_button(
        label="📥 Download study notes (Markdown)",
        data=_content,
        file_name=_safe_filename(title),
        mime="text/markdown",
    )


# ---------------------------------------------------------------------------
# Main render function (called from app.py)
# ---------------------------------------------------------------------------

def render_coach_mode(
    *,
    ollama_host: str,
    ollama_model: str,
) -> None:
    """Render the complete Spanish Source Reading Coach UI panel.

    Called from app.py immediately before ``st.stop()`` when the user has
    selected the Reading Coach app mode.  All Streamlit output rendered here
    replaces the parallel-reader main content area.

    The LLM timeout and checker policy are read from :func:`get_coach_settings`
    so they can be controlled via environment variables without touching this
    function's call site.
    """
    repo = _get_or_create_repo()
    _render_session_history(repo)

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

    _raw_title = st.text_input(
        "Session title",
        value="",
        placeholder="Optional — used in the Markdown heading and filename",
        key="coach_session_title",
    )
    title: str = _raw_title if isinstance(_raw_title, str) else ""

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
                _settings = get_coach_settings()
                client = make_ollama_client(ollama_host, ollama_model, _settings.timeout_seconds)
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
                        annotation_limit_policy=_settings.annotation_limit_policy,
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
    _saved_session_id: str | None = st.session_state.get("coach_saved_session_id")
    if analysis is not None:
        _display_coach_result(
            analysis,
            title=title,
            repo=repo,
            source_text=source_text,
            session_id=_saved_session_id,
        )
        _render_save_button(
            analysis,
            source_text=source_text,
            reader_level=reader_level,
            annotation_density=annotation_density,
            title=title,
            repo=repo,
        )

    _render_saved_phrases_panel(repo)

    multi: MultiChunkAnalysisResult | None = st.session_state.get("coach_multi_analysis")
    if multi is not None:
        _display_multi_chunk_coach_result(multi, title=title)
