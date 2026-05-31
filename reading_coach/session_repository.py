"""reading_coach/session_repository.py — Repository abstraction for coach sessions.

Defines two things:

1. :class:`CoachSessionRepository` — an abstract base class that declares the
   persistence interface.  It contains no storage logic and cannot be
   instantiated directly.  Swap in a MongoDB implementation later by subclassing
   and overriding every abstract method.

2. :class:`InMemoryCoachSessionRepository` — a concrete, in-process
   implementation backed by plain dictionaries.  Useful for unit tests, local
   demos, and any context where MongoDB is not available.

Design notes
------------
* **Copy semantics** — every ``save_*`` call stores a deep copy of the caller's
  object; every ``get_*`` / ``list_*`` call returns a fresh deep copy.  This
  prevents accidental mutation of stored state through returned references.
* **Ordering** — ``list_sessions`` returns sessions sorted by
  ``updated_at`` descending (newest first), matching typical UI expectations.
* **No external dependencies** — no MongoDB driver, no Streamlit, no Ollama.
  Standard library + Pydantic only.
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from reading_coach.session import CoachSession, SavedPhrase

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class CoachSessionRepository(ABC):
    """Abstract repository interface for :class:`~reading_coach.session.CoachSession`
    and :class:`~reading_coach.session.SavedPhrase` objects.

    All concrete implementations must override every abstract method.  The
    interface is intentionally narrow so that a MongoDB (or SQLite, or REST)
    backend can be dropped in without changing any call-sites.
    """

    # --- sessions -----------------------------------------------------------

    @abstractmethod
    def save_session(self, session: CoachSession) -> CoachSession:
        """Persist *session*, creating or replacing any existing record with the
        same ``session_id``.  Returns the stored session."""

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[CoachSession]:
        """Return the session with *session_id*, or ``None`` if it does not exist."""

    @abstractmethod
    def list_sessions(self, limit: int = 25) -> List[CoachSession]:
        """Return up to *limit* sessions, newest-updated first."""

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Remove the session with *session_id*.
        Returns ``True`` if it existed, ``False`` otherwise."""

    # --- phrases ------------------------------------------------------------

    @abstractmethod
    def save_phrase(self, phrase: SavedPhrase) -> SavedPhrase:
        """Persist *phrase*, creating or replacing any existing record with the
        same ``phrase_id``.  Returns the stored phrase."""

    @abstractmethod
    def list_saved_phrases(
        self, session_id: Optional[str] = None
    ) -> List[SavedPhrase]:
        """Return saved phrases.

        Parameters
        ----------
        session_id:
            When provided, return only phrases whose ``session_id`` matches.
            When ``None`` (default), return all phrases.
        """

    @abstractmethod
    def delete_phrase(self, phrase_id: str) -> bool:
        """Remove the phrase with *phrase_id*.
        Returns ``True`` if it existed, ``False`` otherwise."""


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------


class InMemoryCoachSessionRepository(CoachSessionRepository):
    """In-process repository backed by plain Python dicts.

    Thread-safety is explicitly *not* guaranteed — this implementation is
    designed for single-threaded use in tests and local demos.

    Insertion/update order within each dict is preserved by CPython 3.7+, but
    ``list_sessions`` sorts by ``updated_at`` descending so the logical order
    is always newest-first regardless of insertion order.
    """

    def __init__(self) -> None:
        # phrase_id → deep copy of SavedPhrase
        self._phrases: Dict[str, SavedPhrase] = {}
        # session_id → deep copy of CoachSession; preserves insertion order
        self._sessions: Dict[str, CoachSession] = {}

    # --- sessions -----------------------------------------------------------

    def save_session(self, session: CoachSession) -> CoachSession:
        stored = copy.deepcopy(session)
        self._sessions[session.session_id] = stored
        return copy.deepcopy(stored)

    def get_session(self, session_id: str) -> Optional[CoachSession]:
        stored = self._sessions.get(session_id)
        if stored is None:
            return None
        return copy.deepcopy(stored)

    def list_sessions(self, limit: int = 25) -> List[CoachSession]:
        sorted_sessions = sorted(
            self._sessions.values(),
            key=lambda s: s.updated_at,
            reverse=True,
        )
        return [copy.deepcopy(s) for s in sorted_sessions[:limit]]

    def delete_session(self, session_id: str) -> bool:
        if session_id not in self._sessions:
            return False
        del self._sessions[session_id]
        return True

    # --- phrases ------------------------------------------------------------

    def save_phrase(self, phrase: SavedPhrase) -> SavedPhrase:
        stored = copy.deepcopy(phrase)
        self._phrases[phrase.phrase_id] = stored
        return copy.deepcopy(stored)

    def list_saved_phrases(
        self, session_id: Optional[str] = None
    ) -> List[SavedPhrase]:
        phrases = self._phrases.values()
        if session_id is not None:
            phrases = (p for p in phrases if p.session_id == session_id)  # type: ignore[assignment]
        return [copy.deepcopy(p) for p in phrases]

    def delete_phrase(self, phrase_id: str) -> bool:
        if phrase_id not in self._phrases:
            return False
        del self._phrases[phrase_id]
        return True
