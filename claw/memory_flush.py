"""Pre-compaction memory flush — durable-knowledge capture before compaction.

Flush runs the agent's primary model (with its full tool registry) against
a synthetic prompt asking it to capture durable knowledge into
``memory/YYYY-MM-DD.md``. The flush turn is NOT persisted to the
user-visible transcript — the side effect we want is files on disk that
the next reindex picks up.

Two trigger paths, both async (off the user-reply critical path):

1. **Pre-compaction** — fired from ``Agent._spawn_bg_compaction_if_needed``
   when transcript is within ``soft_threshold_tokens`` of the compaction
   threshold (or transcript file > ``force_flush_transcript_bytes``).
   Captures durable info BEFORE older turns are summarized away.
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


def _flush_prompt(today: str) -> str:
    return (
        "Memory flush.\n\n"
        "Look at the conversation so far and capture any durable knowledge worth "
        "persisting across sessions. Use the `append_file` tool ONCE with "
        f"`path=\"memory/{today}.md\"` to add bullet-style notes at the end of "
        "today's memory file. Bundle all your observations into a single "
        "content string in that one tool call.\n\n"
        "Rules:\n"
        "- Treat top-level files (MEMORY.md, SOUL.md, AGENTS.md, IDENTITY.md, "
        "USER.md, etc.) as read-only. Do not touch them.\n"
        "- Do not create timestamped variants like `memory/YYYY-MM-DD-foo.md`. "
        "Those are journal-only and not indexed by the memory system.\n"
        "- Preserve only durable signal: decisions, facts confirmed, commitments, "
        "surprises, repeated patterns. Skip ephemera (in-progress chatter, search "
        "noise, raw tool outputs).\n"
        "- If nothing meets the bar, do nothing — silence is fine.\n\n"
        "Reply briefly when done."
    )


def will_pre_compact_flush(
    cfg: Config,
    transcripts: TranscriptStore,
    sid: str,
    rows: list[dict[str, Any]],
) -> bool:
    """Predicate for the pre-compaction trigger path. Used by Agent to decide
    whether to spawn a background flush+compact task on the trigger turn.
    """
    if not cfg.memory_flush.enabled:
        return False
    threshold = cfg.compaction.mid_session_token_threshold
    soft = cfg.memory_flush.soft_threshold_tokens
    tokens = estimate_tokens(rows)
    file_size = 0
    transcript_path = transcripts.dir / f"{sid}.jsonl"
    if transcript_path.exists():
        try:
            file_size = transcript_path.stat().st_size
        except OSError:
            pass
    token_trip = tokens >= max(0, threshold - soft)
    size_trip = file_size >= cfg.memory_flush.force_flush_transcript_bytes
    return token_trip or size_trip


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
    workspace_system_block: str,
    reason: str,
    tz_name: str | None = None,
) -> bool:
    """Run one flush turn against ``rows`` (typically a snapshot). The agent
    appends durable memory via ``append_file``. Flush turn output is
    discarded — the side effect on disk is what we want.

    ``reason`` is a short label ("pre-compact" / "periodic-growth") logged
    when the flush starts. ``agent_id`` and ``peer_label`` shape the
    run_turn label as ``<agent_id>:flush:<reason>:<peer_label>`` so log
    lines self-identify whose conversation is being flushed. ``sid`` is
    forwarded to ``ollama.run_turn`` for tool-result spooling and as the
    verbose-only correlation handle in the log prefix. ``tz_name`` (IANA
    zone) controls which day's `memory/YYYY-MM-DD.md` file the agent is
    asked to append to; ``None`` falls back to UTC. Caller typically
    passes ``cfg.tz``.
    """
    log.info("[%s] memory flush starting (reason=%s, %d rows)", sid, reason, len(rows))
    history = [as_message(r) for r in rows]
    history.append({"role": "user", "content": _flush_prompt(today_iso_date(tz_name))})
    try:
        await ollama.run_turn(
            model=primary_model,
            history=history,
            system=workspace_system_block,
            tools=tools,
            sid=sid,
            workspace_dir=workspace_dir,
            label=f"{agent_id}:flush:{reason}:{peer_label}",
            verbose_suffix=sid,
        )
    except Exception:
        log.exception("[%s] memory flush turn failed (reason=%s)", sid, reason)
        return False
    log.info("[%s] memory flush done (reason=%s)", sid, reason)
    return True
