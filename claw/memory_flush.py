"""Memory flush — durable-knowledge capture during long sessions.

Flush runs the agent's primary model (with its full tool registry) against
a synthetic prompt asking it to capture durable knowledge into
``memory/YYYY-MM-DD.md``. The flush turn is NOT persisted to the
user-visible transcript — the side effect we want is files on disk that
the next reindex picks up.

Two trigger paths, both async (off the user-reply critical path):

1. **Pre-compact** — fired from ``Agent._spawn_bg_compaction_if_needed``
   when ``will_mid_session_compact`` trips (transcript tokens >
   ``compaction.mid_session_token_threshold``). The bg task runs flush
   then compaction in sequence so durable items are captured before
   older turns get summarized away.
2. **Periodic** — fired from the maintenance loop (paired with reindex)
   when a session has grown by ``periodic_growth_threshold`` tokens since
   its last flush. Lets long-running sessions persist learning regularly
   instead of waiting for compaction to be imminent.

Both paths reach the same agent turn; only the trigger differs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from claw.config import Config
from claw.ollama import OllamaClient
from claw.tools.base import Tool
from claw.transcript import TranscriptStore, as_message, estimate_tokens

log = logging.getLogger("claw.memory_flush")


def today_iso_date(tz_name: str | None = None) -> str:
    """The date used for daily-note filenames.

    ``tz_name`` is an IANA zone (e.g. ``"America/New_York"``); ``None``
    falls back to UTC. Aligning the day boundary to the operator's locale
    keeps cron / calendar / log timing consistent — UTC and local can
    otherwise differ by hours.
    """
    tz = ZoneInfo(tz_name) if tz_name else timezone.utc
    return datetime.now(tz).strftime("%Y-%m-%d")


def _flush_system_prompt(today: str) -> str:
    """Thorough self-contained system prompt for the flush turn.

    Replaces the per-agent ``extra_paths`` workspace block — flush does
    not inherit IDENTITY/USER/SOUL/AGENTS/TOOLS priming, because it has
    a single fixed job and the broader role context tends to nudge the
    model toward bash/read_file chains it shouldn't be running.
    """
    return (
        "You are a memory-flushing turn. Capture durable knowledge from "
        "the conversation above and append it to today's memory file. "
        "You have one tool: `append_file`.\n\n"
        "## Durable signal — keep\n\n"
        "- Decisions and the reasoning behind them\n"
        "- Facts confirmed or invalidated (root causes, version "
        "pinnings, environment quirks)\n"
        "- Commitments, deadlines, scheduled work\n"
        "- Surprises and gotchas — things that bit you that will bite again\n"
        "- Concrete outcomes: versions chosen, workarounds proven, "
        "libraries rejected\n\n"
        "## Ephemera — skip\n\n"
        "- In-progress chatter without a resolution\n"
        "- Raw tool output (bash stdout, grep dumps, file listings, "
        "search results)\n"
        "- Exploratory commands and their negative results\n"
        "- Step-by-step debugging narrative — keep the conclusion, "
        "drop the journey\n"
        "- Pleasantries, status pings, acknowledgements\n\n"
        "## Tool output\n\n"
        "Tool results above are evidence, not content. Read them for "
        "context, write the finding. NEVER copy raw tool output into "
        "the memory file. If a 200-line bash result confirmed a single "
        "root cause, the memory entry is that one sentence.\n\n"
        "## Format\n\n"
        "Markdown headings + bullets, one section per topic, tight "
        "prose over narrative.\n\n"
        "Begin every `##` section with a timestamp marker on its own line, "
        "exactly:\n"
        f"`<!-- mem ts={today} -->`\n"
        "Write only the `ts` field — nothing else. The nightly curator adds "
        "any `id`/`status` later; do not write those yourself.\n\n"
        "## Strict rules\n\n"
        "- Call `append_file` at most once. Bundle all observations "
        "into one content string.\n"
        f"- Target path: `memory/{today}.md`. Do not write elsewhere.\n"
        f"- Do not create dated variants like `memory/{today}-topic.md` — "
        "those are journal-only and not indexed.\n"
        "- If nothing meets the bar, don't call any tool. Silence is fine.\n\n"
        "Reply with one short sentence when done."
    )


_FLUSH_USER_TRIGGER = (
    "Memory flush. Append durable notes from the conversation above "
    "and reply when done."
)


async def run_memory_flush(
    *,
    agent_id: str,
    peer_label: str,
    ollama: OllamaClient,
    sid: str,
    workspace_dir: Path,
    rows: list[dict[str, Any]],
    primary_model: str,
    tools: dict[str, Tool],
    reason: str,
    tz_name: str | None = None,
) -> bool:
    """Run one flush turn against ``rows`` (typically a snapshot). The agent
    appends durable memory via ``append_file``. Flush turn output is
    discarded — the side effect on disk is what we want.

    ``reason`` is a short label ("pre-compact" / "periodic-growth")
    logged when the flush starts. ``agent_id`` and ``peer_label`` shape the
    run_turn label as ``<agent_id>:flush:<reason>:<peer_label>`` so log
    lines self-identify whose conversation is being flushed. ``sid`` is
    forwarded to ``ollama.run_turn`` for tool-result spooling and as the
    verbose-only correlation handle in the log prefix. ``tz_name`` (IANA
    zone) controls which day's `memory/YYYY-MM-DD.md` file the agent is
    asked to append to; ``None`` falls back to UTC. Caller typically
    passes ``cfg.tz``.

    Runs with ``think=False``: the flush turn's prose is discarded, so a
    reasoning trace is pure waste here (7x measured — see the call site).

    The flush runs with a self-contained system prompt (no workspace
    extra_paths) and is expected to be passed a narrowed ``tools`` dict
    (just ``append_file``) so the model can't drift into bash/read_file
    chains it shouldn't be running.
    """
    log.info("[%s] memory flush starting (reason=%s, %d rows)", sid, reason, len(rows))
    today = today_iso_date(tz_name)
    history = [as_message(r) for r in rows]
    history.append({"role": "user", "content": _FLUSH_USER_TRIGGER})
    try:
        await ollama.run_turn(
            model=primary_model,
            history=history,
            system=_flush_system_prompt(today),
            tools=tools,
            sid=sid,
            workspace_dir=workspace_dir,
            label=f"{agent_id}:flush:{reason}:{peer_label}",
            verbose_suffix=sid,
            # Bound the turn. Both were already run_turn parameters; this call
            # simply never passed them, so a flush inherited the widest
            # possible settings: max_tool_turns fell back to the client-wide
            # value (50, for a turn with ONE tool), and num_predict was never
            # sent at all — run_turn only builds `options` when the parameter
            # is given, so Ollama applied its own unlimited default rather
            # than the configured 8192.
            #
            # num_predict is a CEILING, not an allocation — unused headroom
            # costs nothing — so the only job here is catching a runaway, and
            # being tight buys nothing while risking real harm. A flush's
            # output is almost entirely the append_file tool-call ARGUMENTS,
            # so hitting the cap truncates the memory content mid-JSON: a
            # malformed call and a wasted recovery retry, on the one turn
            # whose output IS the product. Matches SUMMARY_NUM_PREDICT, set
            # at 4096 on the same model for the same reason ("or the recap is
            # truncated mid-sentence right at the size where it matters
            # most"). Measured need is ~200 eval tokens, so this is ~20x —
            # deliberately, and free.
            #
            # 6 tool turns is 8x tighter than the client-wide 50 while still
            # allowing several appends plus a closing turn. Both are backstops
            # against a confused flush eating its whole turn_timeout_s, not
            # limits a healthy one should approach.
            num_predict=4096,
            max_tool_turns=6,
            # A flush discards the model's prose entirely — the append_file
            # side effect is the whole product — so a reasoning trace is
            # produced and thrown away, exactly as it was on summarize().
            # Measured 2026-08-24 on qwen3.6:35b-a3b with a real 25-row flush
            # prompt, n=3: median 19.0s / 1162 eval tokens with thinking on
            # against 2.7s / 181 with it off, a 7x difference, and the range
            # tightens from 10.7-31.8s to 2.5-3.1s — which matters as much on
            # a path carrying a turn_timeout_s deadline. The append_file call
            # was emitted in every run of both arms, so suppressing the trace
            # does not cost the tool call.
            #
            # NOT a blanket setting: think stays at the model default for
            # conversational turns (where %thinking surfaces it) and for the
            # curator (whose judgement calls need it).
            think=False,
        )
    except Exception:
        log.exception("[%s] memory flush turn failed (reason=%s)", sid, reason)
        return False
    log.info("[%s] memory flush done (reason=%s)", sid, reason)
    return True
