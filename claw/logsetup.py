"""Log-level application + runtime verbose toggle.

Split out of ``main`` so ``agent`` can flip verbosity at runtime (the
``%verbose`` command) without importing ``main`` (which imports ``Agent`` —
that would be a circular import).

``cfg.verbose`` is only read once, at boot. The live state of verbose
logging *after* boot is the effective logger level, which is what
``ollama`` checks via ``log.isEnabledFor(logging.DEBUG)`` for its verbose
tool-call suffix. So a runtime toggle just re-applies levels here; nothing
needs to mutate the frozen ``Config``.
"""

from __future__ import annotations

import logging

_state = {"verbose": False}


def apply_log_levels(verbose: bool) -> None:
    """Set claw's root level and tame third-party library noise.

    Idempotent and safe to call at runtime: it only adjusts levels, never
    touches handlers/formatters (those are installed once by
    ``main._configure_logging`` via ``logging.basicConfig``).
    """
    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.INFO)
    # nio (room state, crypto, join callbacks) and apscheduler (per-job add
    # lines) at INFO drown out claw's own lines; surface them only at WARNING
    # unless verbose flips the firehose back on.
    library_level = logging.INFO if verbose else logging.WARNING
    logging.getLogger("nio").setLevel(library_level)
    logging.getLogger("apscheduler").setLevel(library_level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _state["verbose"] = verbose


def set_verbose(enabled: bool) -> None:
    """Runtime toggle entry point (used by the %verbose command)."""
    apply_log_levels(enabled)


def verbose_enabled() -> bool:
    """Current verbose state (boot value, then whatever set_verbose set)."""
    return _state["verbose"]
