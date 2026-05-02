"""Transcript compaction — idle recap (boot) + mid-session (background).

Two distinct triggers, two distinct headings:

* ``## Last Session Recap`` — eager during agent init when the prior session's
  last turn is older than ``compaction.idle_recap_seconds``. Archives the old
  JSONL and prepends the recap as a system block on the next prompt. Runs
  synchronously since it's pre-live.
* ``## Pre-compaction Recap`` — fires when estimated transcript tokens exceed
  ``compaction.mid_session_token_threshold``. **Runs as a background task off
  the user-reply critical path**: the slow summarize() call happens against a
  snapshot, then the swap into the on-disk transcript happens under the
  per-session lock so any user turns that arrived during summarize are
  preserved.

Recap blocks are stable across turns so prompt caches stay warm.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from claw.config import Config
from claw.ollama import OllamaClient
from claw.transcript import (
    TranscriptStore,
    estimate_tokens,
    parse_iso,
)

log = logging.getLogger("claw.compaction")

_IDLE_INSTRUCTION = (
    "You are summarizing a prior conversation between a user and an AI agent. "
    "Produce a concise recap (≤200 words) that preserves: key decisions, "
    "outstanding questions, agreed-on facts, and anything the agent committed "
    "to do. Use bullet points. End with a one-line 'Open:' list of unresolved "
    "items, or 'Open: none.' if there are none."
)

_MID_SESSION_INSTRUCTION = (
    "You are summarizing the earlier portion of an in-progress conversation "
    "to free up context. Produce a concise recap (≤250 words) that preserves: "
    "key decisions, outstanding questions, agreed-on facts, tool results that "
    "remain relevant, and anything the agent committed to do. Use bullet "
    "points. End with a one-line 'Open:' list of unresolved items, or "
    "'Open: none.'"
)


def _human_delta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"about {minutes} min ago"
    hours = minutes / 60
    if hours < 24:
        return f"about {hours:.1f} h ago"
    days = hours / 24
    return f"about {days:.1f} days ago"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows_to_text(rows: list[dict[str, Any]]) -> str:
    """Render OpenAI-flat rows as a labeled transcript for the summarizer."""
    lines: list[str] = []
    for r in rows:
        role = r.get("role", "")
        content = r.get("content") or ""
        if role == "tool":
            txt = content[:400] if isinstance(content, str) else ""
            if txt:
                lines.append(f"[tool result] {txt}")
            continue
        for tc in r.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments")
            args_str = args if isinstance(args, str) else json.dumps(args, separators=(",", ":"))
            lines.append(f"[{role} → tool {fn.get('name')}] input={args_str}")
        if isinstance(content, str) and content.strip():
            lines.append(f"[{role}] {content.strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Idle recap (boot-time, eager, synchronous — runs pre-live)
# ---------------------------------------------------------------------------


async def maybe_idle_recap(
    *,
    cfg: Config,
    ollama: OllamaClient,
    transcripts: TranscriptStore,
    sid: str,
    compaction_model: str,
) -> str | None:
    """If the most-recent turn is older than ``idle_recap_seconds``, summarize
    the prior session, archive its JSONL, and return the system-block
    markdown for prepending to the next prompt. Otherwise return ``None``.
    """
    last_ts = transcripts.last_ts(sid)
    if last_ts is None:
        return None
    last_unix = parse_iso(last_ts)
    age = time.time() - last_unix
    if age < cfg.compaction.idle_recap_seconds:
        return None

    rows = transcripts.load(sid)
    if not rows:
        return None

    log.info(
        "idle recap: session %s last activity %s, summarizing %d turns",
        sid, _human_delta(age), len(rows),
    )

    text = _rows_to_text(rows)
    try:
        summary = await ollama.summarize(
            model=compaction_model,
            instruction=_IDLE_INSTRUCTION,
            transcript_text=text,
        )
    except Exception:
        log.exception("idle recap summarize failed; skipping recap")
        return None

    now_iso = _now().isoformat()
    archived = transcripts.archive(sid, f"recap-{now_iso.replace(':', '-')}")
    if archived:
        log.info("archived prior session to %s", archived)

    return (
        f"## Last Session Recap\n"
        f"Last session ended {last_ts} ({_human_delta(age)}).\n"
        f"Current time: {now_iso}.\n\n"
        f"{summary}"
    )


# ---------------------------------------------------------------------------
# Mid-session compaction — async, snapshot + atomic-swap pattern
# ---------------------------------------------------------------------------


def _split_for_compaction(
    rows: list[dict[str, Any]],
    reserve_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into (older, newer), respecting tool atomicity.

    Walk rows newest-first, accumulating estimated tokens, and place the cut
    once we've collected at least ``reserve_tokens`` worth — then advance
    *forward* (toward newer) until we land on a fresh ``user`` row, so the
    older slice never ends mid tool-call sequence (assistant tool_calls →
    tool results → assistant final). This keeps the newest ``reserve_tokens``
    of context verbatim and summarizes the rest.

    OpenAI-flat tool calls run as: ``assistant(tool_calls=[...]) -> tool
    result -> tool result -> ... -> assistant(text)``. The cut must land at a
    fresh user turn so we don't slice between an assistant's tool_calls and
    the matching tool results.

    Falls back to "everything older, nothing newer" when the transcript is
    too small to compact (caller's threshold predicate already decided
    compaction is worth doing).
    """
    n = len(rows)
    if n < 4:
        return rows, []

    # Walk from the end, accumulating tokens, until we cover reserve_tokens.
    # ``cut`` is the index of the FIRST row in the "newer" slice.
    acc = 0
    cut = n
    for i in range(n - 1, -1, -1):
        acc += estimate_tokens([rows[i]])
        cut = i
        if acc >= reserve_tokens:
            break

    # Advance cut forward to the next ``user`` boundary so the older slice
    # ends cleanly before a fresh user turn.
    while cut < n and rows[cut].get("role") != "user":
        cut += 1

    if cut <= 0 or cut >= n:
        # Either everything fits in reserve (nothing to compact), or we
        # couldn't find a user boundary in the newer region.
        return ([], rows) if cut <= 0 else (rows, [])

    return rows[:cut], rows[cut:]


def will_mid_session_compact(cfg: Config, rows: list[dict[str, Any]]) -> bool:
    """Predicate-only check (no I/O). The agent uses this to decide whether
    to spawn a background compaction task.
    """
    return estimate_tokens(rows) > cfg.compaction.mid_session_token_threshold


async def run_mid_session_compact_async(
    *,
    cfg: Config,
    ollama: OllamaClient,
    transcripts: TranscriptStore,
    session_lock: asyncio.Lock,
    sid: str,
    rows_snapshot: list[dict[str, Any]],
    compaction_model: str,
) -> bool:
    """Run compaction off the critical path, then atomically swap.

    Computes the older/newer split + summarize() against ``rows_snapshot``
    (which was captured by the caller at the moment compaction was decided
    to fire). The snapshot's older portion becomes a single recap turn;
    everything from the older boundary onward is preserved by re-reading
    the on-disk transcript at swap time — so any user turns that arrived
    during the summarize call survive.

    Returns True if a swap happened, False if the predicate failed or the
    swap was skipped (transcript shrank, race with another compactor, etc.).

    The caller is responsible for the predicate gate. We re-check here for
    safety but expect the caller to have already decided.
    """
    if not will_mid_session_compact(cfg, rows_snapshot):
        return False

    older, newer = _split_for_compaction(rows_snapshot, cfg.compaction.reserve_tokens)
    if not older:
        return False
    older_count = len(older)

    log.info(
        "[%s] background compaction: summarizing %d older turns (snapshot=%d rows)",
        sid, older_count, len(rows_snapshot),
    )

    text = _rows_to_text(older)
    try:
        summary = await ollama.summarize(
            model=compaction_model,
            instruction=_MID_SESSION_INSTRUCTION,
            transcript_text=text,
        )
    except Exception:
        log.exception("[%s] background compaction summarize failed", sid)
        return False

    covers_from = older[0].get("ts", "?")
    covers_to = older[-1].get("ts", "?")
    compacted_at = _now().isoformat()
    block = (
        f"## Pre-compaction Recap\n"
        f"Covers turns from {covers_from} through {covers_to}.\n"
        f"Compacted at: {compacted_at}.\n\n"
        f"{summary}"
    )
    recap_row = {"role": "user", "content": block, "ts": compacted_at}

    # Atomic swap: take the per-session lock, re-read the live transcript
    # (it may have grown during summarize), and slice from older_count
    # onward — preserving the snapshot's newer portion AND any rows the
    # agent appended while we were summarizing.
    async with session_lock:
        current = transcripts.load(sid)
        if len(current) < older_count:
            log.warning(
                "[%s] background compaction skipped — transcript has %d rows "
                "but snapshot's older count was %d (another compactor raced?)",
                sid, len(current), older_count,
            )
            return False
        preserved = current[older_count:]
        new_rows = [recap_row] + preserved
        transcripts.replace(sid, new_rows)

    log.info(
        "[%s] background compaction swapped: %d rows -> %d rows",
        sid, len(current), len(new_rows),
    )
    return True
