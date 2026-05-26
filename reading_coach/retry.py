"""reading_coach/retry.py — Configurable retry policy for transient LLM failures.

Provides a simple retry wrapper for callable operations that may fail due to
transient network or LLM issues.

Default policy
--------------
* ``max_retries = 1``      — one additional attempt after the first (two total).
* ``retryable = (ReadingCoachLLMError,)`` — only network/LLM errors are retried.
* Parse failures and schema validation failures are **not** retried by default;
  they represent deterministic model output problems, not transient network faults.
* Exponential back-off starting at ``backoff_base`` seconds.

Environment variables
---------------------
READING_COACH_MAX_RETRIES
    Integer ≥ 0.  Invalid or absent values fall back to the built-in default (1).

No Streamlit, no Ollama, no network access at import time.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

from reading_coach.errors import ReadingCoachLLMError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_RETRIES: int = 1
_DEFAULT_RETRYABLE: tuple[type[BaseException], ...] = (ReadingCoachLLMError,)
_DEFAULT_BACKOFF_BASE: float = 1.0


# ---------------------------------------------------------------------------
# RetryConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class RetryConfig:
    """Configuration for the retry policy around transient LLM failures.

    Attributes
    ----------
    max_retries:
        Number of *additional* attempts after the first call (``0`` = no retry,
        so the callable is invoked exactly once).
    retryable:
        Tuple of exception types that trigger a retry.  Exception types not in
        this tuple cause an immediate re-raise without further attempts.
    backoff_base:
        Base delay in seconds.  Actual delay before attempt *n* (0-indexed) is
        ``backoff_base * 2 ** n`` (exponential back-off).  Set to ``0.0`` to
        use zero delay (still calls ``sleep_fn``).
    sleep_fn:
        Injectable sleep callable.  ``None`` means use ``time.sleep`` at
        runtime.  Pass ``lambda _: None`` in unit tests to eliminate real
        delays without monkey-patching the standard library.
    """

    max_retries: int = _DEFAULT_MAX_RETRIES
    retryable: tuple[type[BaseException], ...] = field(default=_DEFAULT_RETRYABLE)
    backoff_base: float = _DEFAULT_BACKOFF_BASE
    sleep_fn: Callable[[float], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def _get_sleep(self) -> Callable[[float], None]:
        """Return the effective sleep callable (``time.sleep`` when unset)."""
        return self.sleep_fn if self.sleep_fn is not None else time.sleep


# ---------------------------------------------------------------------------
# Env-var factory
# ---------------------------------------------------------------------------

def get_retry_config() -> RetryConfig:
    """Build a :class:`RetryConfig` from the current process environment.

    Reads ``READING_COACH_MAX_RETRIES`` (integer ≥ 0).  Absent, blank, or
    invalid values fall back to :data:`_DEFAULT_MAX_RETRIES`.  Negative
    integers are clamped to ``0`` (no retry).
    """
    raw = os.getenv("READING_COACH_MAX_RETRIES", "").strip()
    try:
        max_retries = max(0, int(raw)) if raw else _DEFAULT_MAX_RETRIES
    except ValueError:
        logger.warning(
            "Invalid READING_COACH_MAX_RETRIES=%r; using default %d",
            raw,
            _DEFAULT_MAX_RETRIES,
        )
        max_retries = _DEFAULT_MAX_RETRIES
    return RetryConfig(max_retries=max_retries)


# ---------------------------------------------------------------------------
# Core retry function
# ---------------------------------------------------------------------------

def with_retry(
    func: Callable[[], T],
    *,
    config: RetryConfig,
) -> T:
    """Call *func* with retry on configured transient exceptions.

    Parameters
    ----------
    func:
        Zero-argument callable to invoke.  May raise on failure.
    config:
        Retry policy — controls the maximum number of attempts, which
        exception types trigger a retry, the back-off timing, and the sleep
        callable (injectable for tests).

    Returns
    -------
    T
        The return value of the first successful invocation of *func*.

    Raises
    ------
    Exception
        * The **last** retryable exception if all attempts are exhausted.
        * The **first** non-retryable exception encountered (propagated
          immediately without consuming any retry budget).
    """
    _sleep = config._get_sleep()
    total_attempts = config.max_retries + 1
    last_exc: BaseException | None = None

    for attempt in range(total_attempts):
        try:
            return func()
        except BaseException as exc:
            if not isinstance(exc, config.retryable):
                raise  # non-retryable: propagate immediately
            last_exc = exc
            if attempt < config.max_retries:
                delay = config.backoff_base * (2.0 ** attempt)
                logger.warning(
                    "Attempt %d/%d failed (%s: %s) — retrying in %.2fs",
                    attempt + 1,
                    total_attempts,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                _sleep(delay)
            else:
                logger.warning(
                    "Attempt %d/%d failed (%s: %s) — no more retries",
                    attempt + 1,
                    total_attempts,
                    type(exc).__name__,
                    exc,
                )

    # All attempts exhausted — re-raise the last retryable exception.
    assert last_exc is not None  # loop ran at least once (max_retries >= 0)
    raise last_exc
