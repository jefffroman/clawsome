"""Transcript compaction — idle recap (boot) + mid-session (background).

Two distinct triggers, two distinct headings:

* ``## Last Session Recap`` — eager during agent init when the prior session's
  last turn is older than ``compaction.idle_recap_seconds``. Archives the old
  JSONL and prepends the recap as a system block on the next prompt. Runs
  synchronously since it's pre-live.
* ``## Pre-compaction Recap`` — fires when the estimated *real prompt*
  (transcript tokens + system/memory ``overhead_tokens``) exceeds
  ``compaction.mid_session_token_threshold``. **Runs as a background task,
  spawned at turn-end** (after the reply is sent, by ``Agent._process_batch``)
  so its summarize() never contends for GPU with the reply's own generation:
  the slow summarize() happens against a snapshot, then the swap into the
  on-disk transcript happens under the per-session lock so any user turns that
  arrived during summarize are preserved.

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
from claw.ollama import TOOL_RESULT_THRESHOLD_CHARS, OllamaClient
from claw.transcript import (
    TranscriptStore,
    estimate_tokens,
    parse_iso,
)

log = logging.getLogger("claw.compaction")

# Per-row clip applied to tool results before they reach the summarizer.
#
# This is the summarizer's INPUT budget — _rows_to_text feeds summarize()
# only; it never touches what the agent sees. Tool results reaching the agent
# are bounded separately (tools/spool.py at write time, _spool_and_truncate
# for the persisted row).
#
# It was 400, which double-truncated: rows in the transcript are ALREADY
# capped at TOOL_RESULT_THRESHOLD_CHARS, so a second, far tighter clip only
# discarded evidence the summarizer needed. Measured on a real 166-row
# session: 78 tool rows, 95,478 chars, of which 400-clipping kept 17,754 —
# 19%. Matching the upstream cap makes this a no-op for anything that came
# through the transcript, while still bounding a hypothetical caller that
# hands us unbounded rows.
_TOOL_CLIP_CHARS = TOOL_RESULT_THRESHOLD_CHARS

# Recap length, scaled to what is being compressed rather than fixed.
#
# The recap lands in the system prompt and is re-injected every turn for the
# session's life, so it IS permanent overhead and does need a bound — but it
# also sits in the stable KV prefix, so it is prefilled once and cached
# thereafter. The old flat ≤200 words was the same budget for a 4-turn
# session and a 166-turn one; the latter measured ~140 words out, i.e. a 99%+
# compression of the afternoon. At the 96,000-token compaction trigger, even
# the max here costs ~1.1% of budget.
_RECAP_WORDS_MIN = 200
_RECAP_WORDS_MAX = 800
_RECAP_WORDS_PER_TOKEN = 40  # 1 word of recap per 40 tokens summarized

# Mid-session gets a more generous ceiling than the idle recap, because it is
# compressing *live working context* — the agent is mid-task and still needs
# the detail, where an idle recap summarizes a conversation that already
# ended. It can afford to: the recap lands as a transcript row alongside the
# reserve_tokens slice (48,000) inside the mid_session_token_threshold
# (96,000), so post-compaction headroom is ~47,000 tokens and even this
# ceiling (~2,050 tokens) spends ~4% of it. The floor is higher for the same
# reason — an in-progress task summarized to 200 words is not resumable.
_MID_RECAP_WORDS_MIN = 300
_MID_RECAP_WORDS_MAX = 1500

# Floor on the *older* slice — the part actually summarized — as opposed to
# the reserve, which is a floor on the whole transcript. Derived rather than
# configured, so it stays correct if the word bounds move: compacting is only
# worth it when the rows being discarded are several times larger than the
# recap replacing them. Below that the swap frees little or nothing while
# permanently destroying verbatim context, which is the wrong trade at any
# setting.
_TOKENS_PER_WORD = 1.33
_MIN_COMPACTION_RATIO = 4
_MIN_OLDER_TOKENS = int(
    _MID_RECAP_WORDS_MIN * _TOKENS_PER_WORD * _MIN_COMPACTION_RATIO
)  # ~1596


def _recap_word_budget(
    transcript_tokens: int,
    *,
    lo: int = _RECAP_WORDS_MIN,
    hi: int = _RECAP_WORDS_MAX,
) -> int:
    """Words to allow a recap summarizing ``transcript_tokens``, clamped."""
    scaled = transcript_tokens // _RECAP_WORDS_PER_TOKEN
    return max(lo, min(hi, scaled))


def _idle_instruction(word_budget: int) -> str:
    return (
        "You are summarizing a prior conversation between a user and an AI "
        f"agent. Produce a recap (≤{word_budget} words) that preserves: key "
        "decisions, outstanding questions, agreed-on facts, and anything the "
        "agent committed to do. Prefer concrete detail — file paths, "
        "commands, names, numbers — over general description; the underlying "
        "transcript is being discarded, so anything you omit is lost. Use "
        "bullet points. End with a one-line 'Open:' list of unresolved items, "
        "or 'Open: none.' if there are none. The 'Open:' line is required."
    )

def _mid_session_instruction(word_budget: int) -> str:
    """Instruction for compacting the earlier part of a LIVE conversation.

    Distinct from the idle version in what it optimizes for: the agent is
    mid-task and will keep working directly from this text, so resumability
    matters more than narrative. It asks for in-flight state explicitly —
    what is done, what is half-done, which files and commands are in play —
    because that is what the discarded rows were carrying.
    """
    return (
        "You are summarizing the earlier portion of an in-progress "
        "conversation to free up context. The agent will continue working "
        f"from your summary alone. Produce a recap (≤{word_budget} words) "
        "that preserves: key decisions, outstanding questions, agreed-on "
        "facts, tool results that remain relevant, and anything the agent "
        "committed to do. State the current in-flight state explicitly — "
        "what is finished, what is half-finished, and what was about to "
        "happen next. Prefer concrete detail — file paths, commands, names, "
        "numbers — over general description; the underlying turns are being "
        "discarded, so anything you omit is lost to the rest of this "
        "session. Use bullet points. End with a one-line 'Open:' list of "
        "unresolved items, or 'Open: none.' if there are none. The 'Open:' "
        "line is required."
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
            txt = content[:_TOOL_CLIP_CHARS] if isinstance(content, str) else ""
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

    # Size floor. Recapping is only worth it when there is something to
    # compress: below this the summary is bigger than, or comparable to, the
    # rows it replaces, and it is strictly lossy — the verbatim turns are
    # better context than a paraphrase of them. Observed live: a 2-turn
    # session was recapped at boot, spending a summarize call to replace 193
    # tokens of transcript. Returning None leaves the rows in place; the
    # timestamps agents rely on ride in the message text, not the recap.
    transcript_tokens = estimate_tokens(rows)
    if transcript_tokens < cfg.compaction.idle_recap_min_tokens:
        log.info(
            "idle recap: session %s idle %s but only ~%d tokens "
            "(floor %d) — keeping %d rows verbatim",
            sid, _human_delta(age), transcript_tokens,
            cfg.compaction.idle_recap_min_tokens, len(rows),
        )
        return None

    word_budget = _recap_word_budget(transcript_tokens)
    log.info(
        "idle recap: session %s last activity %s, summarizing %d turns "
        "(~%d tokens) into <=%d words",
        sid, _human_delta(age), len(rows), transcript_tokens, word_budget,
    )

    text = _rows_to_text(rows)
    try:
        summary = await ollama.summarize(
            model=compaction_model,
            instruction=_idle_instruction(word_budget),
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


def _safe_cut_points(rows: list[dict[str, Any]]) -> list[int]:
    """Indices where ``rows`` can be split without orphaning a tool result
    from the assistant turn that requested it.

    OpenAI-flat tool calls run as ``assistant(tool_calls=[...]) -> tool
    result -> tool result -> ... -> assistant(text)``. A ``tool`` row is only
    meaningful next to the ``assistant`` row carrying its ``tool_call_id``, so
    a cut is illegal exactly while some tool_call is still awaiting its
    result. Tracking that outstanding count is the real invariant; requiring a
    ``user`` row (the rule until 2026-08-08) is merely a sufficient condition
    for it, and a very expensive one — on a real tool-heavy transcript it
    admitted 13 cut points where this admits 113, cutting the worst-case
    placement error from ±10,540 tokens to ±2,291.
    """
    pending = 0
    safe: list[int] = []
    for i, r in enumerate(rows):
        if pending == 0:
            safe.append(i)
        role = r.get("role")
        if role == "assistant":
            pending += len(r.get("tool_calls") or [])
        elif role == "tool":
            # max(0, ...) so a malformed transcript (a tool result with no
            # matching call) can't drive the counter negative and mark the
            # rest of the transcript uncuttable.
            pending = max(0, pending - 1)
    return safe


def _split_for_compaction(
    rows: list[dict[str, Any]],
    reserve_tokens: int,
    *,
    min_older_tokens: int = _MIN_OLDER_TOKENS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into (older, newer), keeping the newest ~``reserve_tokens``
    verbatim and handing the rest to the summarizer.

    The cut goes to whichever legal point leaves the newer slice **nearest**
    ``reserve_tokens``. Legal means ``_safe_cut_points``: anywhere no tool_call
    is still awaiting its result.

    Two things about that, both learned the hard way on 2026-08-08:

    *Which points are legal.* This used to require a ``user`` row. That's
    sufficient for tool atomicity but far stronger than necessary, and on
    tool-heavy transcripts user rows are sparse (measured: 1 per 18 rows, gaps
    of 186 … 21,080 tokens). The reserve was therefore snapped to one of ~13
    possible outcomes, so distinct settings produced identical splits — 32000
    and 16000 both kept 15,983. Cutting between completed tool rounds instead
    gives ~113 points on the same transcript.

    *Which direction to round.* Only ever advancing forward made the reserve a
    systematically-undershot ceiling (48000 kept 39,662). Only ever retreating
    has the mirror flaw (48000 would keep 60,160) and frees less per
    compaction, so compaction fires more often — costly, since every
    compaction rewrites the head of the transcript and invalidates the whole
    KV prefix cache. Nearest bounds the error in both directions instead. Ties
    go to the earlier cut, i.e. toward preserving more: over-preserving costs
    one later compaction, under-preserving silently drops context.

    Note the newer slice may now begin with an ``assistant`` row, so after the
    swap the model can resume mid-task — its context starts with the recap
    (a ``user`` row) followed immediately by its own unfinished turn. That is
    valid OpenAI-flat and valid ChatML; it is simply a shape the user-row rule
    never produced.

    Returns ``([], rows)`` — nothing to compact — when the transcript fits
    inside the reserve, offers no legal interior cut, or would yield an
    ``older`` slice too small to be worth summarizing (``_MIN_OLDER_TOKENS``;
    clearing the reserve is not the same as having something worth
    compacting, since the cut lands *nearest* it).
    """
    n = len(rows)
    # No small-n special case. There was one until 2026-08-08 —
    # ``if n < 4: return rows, []`` — which inverted this function's contract
    # for exactly the inputs it claimed to protect: it compacted EVERYTHING
    # and kept nothing, while the identical content one row longer returned
    # ([], rows) and compacted nothing. It was reachable, because %compact
    # passes force=True and so bypasses the token predicate entirely: a
    # 1-3 row session could be replaced wholesale by its own summary.
    #
    # Removing it is strictly better than making it return ([], rows): the
    # two guards below already cover small n correctly (n=1 offers no
    # interior cut; anything fitting the reserve exits early), AND a short
    # transcript that genuinely exceeds the reserve now gets a real split
    # instead of being all-or-nothing.

    # suffix[i] = estimated tokens in rows[i:]. One pass, so scoring every
    # candidate below stays linear rather than quadratic.
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + estimate_tokens([rows[i]])

    if suffix[0] < reserve_tokens:
        # The whole transcript fits inside the reserve — nothing is old enough
        # to summarize. Checked explicitly rather than left to fall out of the
        # boundary search, which depended on rows[0] being a user row and
        # silently compacted a prefix when it wasn't.
        return [], rows

    # Interior cuts only: 0 would compact nothing, n would keep nothing.
    usable = [c for c in _safe_cut_points(rows) if 0 < c < n]
    if not usable:
        return [], rows

    cut = min(usable, key=lambda c: abs(suffix[c] - reserve_tokens))
    older_tokens = suffix[0] - suffix[cut]
    if older_tokens < min_older_tokens:
        # Clearing the reserve is not the same as having something worth
        # compacting. The cut lands NEAREST the reserve, so a transcript only
        # slightly over it yields a tiny `older` — measured: a 48,200-token
        # transcript against a 48,000 reserve gives older=200, which the
        # 300-word floor would "compress" into ~399 tokens. That grows the
        # transcript and permanently discards the rows to do it.
        #
        # Unreachable on the automatic path, where the 96,000 trigger
        # guarantees older ≈ threshold - overhead - reserve. Reachable via
        # %compact, which passes force=True and so skips that trigger.
        #
        # Overridable so placement tests can drive the cut logic at synthetic
        # scale — this floor is a policy about whether compacting is WORTH it,
        # separate from where the cut goes. Production callers take the
        # default; nothing else passes it.
        return [], rows
    return rows[:cut], rows[cut:]


def will_mid_session_compact(
    cfg: Config,
    rows: list[dict[str, Any]],
    *,
    overhead_tokens: int = 0,
) -> bool:
    """Predicate-only check (no I/O). The agent uses this to decide whether
    to spawn a background compaction task.

    ``overhead_tokens`` accounts for the non-transcript portion of the real
    prompt — the system block (workspace files + skill catalog) and the
    retrieved-memory injection — which ``estimate_tokens(rows)`` alone omits.
    Passing it makes the trigger fire at the *actual* prompt size rather than
    transcript-only, which otherwise undercounts by tens of thousands of
    tokens (2026-07-23: a "96k of transcript" trigger was a ~129k real prompt).
    """
    return (
        estimate_tokens(rows) + overhead_tokens
        > cfg.compaction.mid_session_token_threshold
    )


def will_compact(
    cfg: Config,
    rows: list[dict[str, Any]],
    *,
    force: bool = False,
    overhead_tokens: int = 0,
) -> bool:
    """Predicate-only (no I/O): would ``run_mid_session_compact_async``
    with the same ``force`` actually swap? Mirrors its two gates exactly —
    the token-threshold predicate (bypassed by ``force``) AND a non-empty
    older split given the reserve/keep window. ``%compact`` uses this to
    skip the (expensive) pre-compact flush entirely when there is nothing
    to compact, instead of flushing and then no-op'ing.
    """
    if not force and not will_mid_session_compact(
        cfg, rows, overhead_tokens=overhead_tokens
    ):
        return False
    older, _ = _split_for_compaction(rows, cfg.compaction.reserve_tokens)
    return bool(older)


def compaction_preview(
    cfg: Config, rows: list[dict[str, Any]],
) -> tuple[int, int]:
    """``(older_tokens, newer_tokens)`` for the split a forced compaction
    would perform on ``rows`` right now. ``older_tokens == 0`` means nothing
    would be compacted.

    Exists so ``%context`` can report the operation that will actually happen
    instead of leaving the operator to infer it from ``reserve_tokens``. The
    configured number is a *budget*, not the outcome: the cut can only land on
    a user boundary, and those are coarse (see ``_split_for_compaction``), so
    the kept slice routinely differs from the reserve by thousands of tokens
    in either direction.
    """
    older, newer = _split_for_compaction(rows, cfg.compaction.reserve_tokens)
    return estimate_tokens(older), estimate_tokens(newer)


async def run_mid_session_compact_async(
    *,
    cfg: Config,
    ollama: OllamaClient,
    transcripts: TranscriptStore,
    session_lock: asyncio.Lock,
    sid: str,
    rows_snapshot: list[dict[str, Any]],
    compaction_model: str,
    force: bool = False,
    overhead_tokens: int = 0,
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
    safety but expect the caller to have already decided. ``force=True`` (a
    manual ``%compact``) bypasses *only* the token-threshold predicate — the
    "nothing compactable" early return below is still honored, so a forced
    compact on a too-short transcript is a clean no-op (returns False).
    """
    if not force and not will_mid_session_compact(
        cfg, rows_snapshot, overhead_tokens=overhead_tokens
    ):
        return False

    older, newer = _split_for_compaction(rows_snapshot, cfg.compaction.reserve_tokens)
    if not older:
        return False
    older_count = len(older)
    older_tokens = estimate_tokens(older)
    word_budget = _recap_word_budget(
        older_tokens, lo=_MID_RECAP_WORDS_MIN, hi=_MID_RECAP_WORDS_MAX,
    )

    log.info(
        "[%s] background compaction: summarizing %d older turns (~%d tokens) "
        "into <=%d words (snapshot=%d rows)",
        sid, older_count, older_tokens, word_budget, len(rows_snapshot),
    )

    text = _rows_to_text(older)
    try:
        summary = await ollama.summarize(
            model=compaction_model,
            instruction=_mid_session_instruction(word_budget),
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
