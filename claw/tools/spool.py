"""Bounded tool results: cap oversized output, spool the remainder to disk.

``bash`` and ``web_search`` return whatever they happen to produce. The agent
asked a *question* ("did this command work?", "what's on this topic?") and the
volume comes along for the ride — so a single verbose build log or a chatty
search can dump tens of thousands of characters into a transcript that then
carries them for the rest of the session. Nothing in the tool layer prevented
that before this module.

``read_file`` is deliberately NOT bounded here: there the content *is* the
request, and it's also the documented way a parent retrieves a subagent's
spooled result (see ``subagent._RESULT_PREVIEW_CHARS``). Silently truncating it
would sever that path and leave the agent reasoning over a partial artifact it
believes is whole.

Spool files land in the same ``<workspace>/.tool-results/<sid>/`` scratch tree
as subagent results, so ``Agent.sweep_spool_tree`` reaps them on the same >24h
grace — no separate lifecycle.

**Bound at write time, never retroactively.** Rewriting a tool result that is
already in the transcript changes the prompt prefix, which invalidates the KV
cache for every token after it — the same failure mode that motivated moving
``<retrieved_memory>`` out of the system prompt.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from claw.runctx import current_sid

log = logging.getLogger("claw.tools.spool")

# Results at or under this many chars are passed through untouched. Sized
# against observed traffic (bash mean ~1k / max ~8k chars, web_search mean
# ~1.5k / max ~3.3k): ordinary output is unaffected, outliers get bounded.
MAX_INLINE_CHARS = 6000

# When a result is bounded, keep this much of the head and this much of the
# tail. Both ends matter: ``_run_bash`` appends ``[exit_code] N`` *last*, so a
# head-only truncation would discard whether the command actually succeeded.
HEAD_CHARS = 3000
TAIL_CHARS = 1500


def _spool(text: str, workspace_dir: Path, tool: str) -> str | None:
    """Write ``text`` under ``<workspace>/.tool-results/<sid>/`` and return
    its workspace-relative path, or None if the write failed.

    The filename is content-addressed (tool name + short digest) so identical
    output spools once and the path is stable across retries.
    """
    try:
        sid = current_sid.get() or "_"
        spool_dir = workspace_dir / ".tool-results" / sid
        spool_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:12]
        path = spool_dir / f"{tool}-{digest}.txt"
        if not path.exists():
            path.write_text(text)
        return str(path.relative_to(workspace_dir))
    except OSError:
        log.exception("failed to spool %s result", tool)
        return None


def bound_result(
    text: str,
    *,
    workspace_dir: Path,
    tool: str,
    cap: int = MAX_INLINE_CHARS,
) -> str:
    """Return ``text`` unchanged when it fits, else a head+tail excerpt with
    an explicit note naming the full size and the spooled path.

    The note is deliberately loud: the agent must be able to tell that it is
    looking at an excerpt, and must have a way to get the rest. If spooling
    fails the excerpt is still returned (bounded beats unbounded) but the note
    says the remainder was dropped rather than pointing at a path that isn't
    there.
    """
    if len(text) <= cap:
        return text

    path = _spool(text, workspace_dir, tool)
    head = text[:HEAD_CHARS]
    tail = text[-TAIL_CHARS:]
    omitted = len(text) - HEAD_CHARS - TAIL_CHARS

    if path:
        note = (
            f"[truncated: {len(text)} chars total, {omitted} omitted from the "
            f"middle. Full output saved to: {path} — read_file it if you need "
            f"the rest.]"
        )
    else:
        note = (
            f"[truncated: {len(text)} chars total, {omitted} omitted from the "
            f"middle. Spooling the full output failed, so the omitted portion "
            f"is unavailable — re-run with a narrower command if you need it.]"
        )
    return f"{head}\n\n{note}\n\n{tail}"
