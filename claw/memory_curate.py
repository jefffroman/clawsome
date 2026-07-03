"""Nightly memory curator — "forgetory".

A separate, heavier pass than the frequent collector (``memory_flush``). Once
per night a larger model grooms each agent's markdown memory:

- **dedup** — collapse near-identical memories (recurring-cron churn) to one
  canonical copy, archiving the rest;
- **supersession** — point an outdated long-term fact forward to its successor
  with ``supersededBy=<id>``; retrieval auto-follows old->new so the timeline
  stays visible (see ``MemoryIndex._resolve_supersession``);
- **archive** — move lapsed ephemera (past appointments, "today is X") out of
  the indexed daily notes into ``memory/archive.md`` (preserved, never
  retrieved);
- **uncertain** — flag genuine unresolved contradictions with an inline caveat.

Markdown is the source of truth: the curator edits the ``.md`` files directly
and the index is re-derived, so ``rm -rf .memory/`` rebuilds everything intact.
Because the edits are agentic (the model investigates and rewrites files
itself), the audit trail is ``/var/log/claw-curator.log``: a reasoned action
ledger the curator writes itself via the ``record_action`` tool — one line per
archive/supersede/dedup/uncertain decision with its justification — interleaved
with the harness's own progress lines (working-set size, per-file start,
timeouts). The mechanical per-tool-call spam is intentionally *not* logged here;
claw.log already records tool-call names per turn, and the ``reason`` field
captures the conclusion of any investigation. The markdown diff is the ground
truth for the edits themselves.

The curator is coupled to the collector: it only runs for agents when
``memory_flush`` is enabled (no memory collected -> nothing to curate).

This module exposes both the nightly entrypoint (``curate_all_agents``, called
from the gateway's nightly loop) and a one-off bootstrap CLI
(``python -m claw.memory_curate --config ... --agent ... [--dry-run]``) for the
first full-corpus pass.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from claw.config import Config, load
from claw.memory import ARCHIVE_NOTE, MemoryIndex
from claw.memory_flush import today_iso_date
from claw.ollama import OllamaClient
from claw.tools.base import Tool
from claw.tools.memory_search import build_curator_search_tool

log = logging.getLogger("claw.memory_curate")

# Workspace-relative path to the curator's non-indexed archive sink. Single
# sourced from memory.ARCHIVE_NOTE so the "not date-shaped -> not indexed"
# guarantee and this path can never drift apart.
ARCHIVE_PATH = f"memory/{ARCHIVE_NOTE}"

# Tools the curator gets — a narrowed, investigation-capable set pulled from the
# agent's full registry (no subagent_*/cron_*/web_search).
CURATOR_TOOL_NAMES = (
    "read_file", "list_dir", "memory_search",
    "write_file", "append_file", "bash",
)
# Mutating tools whose calls are recorded to the audit log.
_MUTATING_TOOLS = ("write_file", "append_file", "bash")

_CURATOR_LOG_PATH = "/var/log/claw-curator.log"


# --- audit logging ----------------------------------------------------------

def curator_logger() -> logging.Logger:
    """Dedicated logger writing to ``/var/log/claw-curator.log`` (the audit
    trail for the agentic edits). Falls back to the default handlers if the
    file isn't writable (e.g. a dev box running the bootstrap CLI)."""
    logger = logging.getLogger("claw.curator")
    if getattr(logger, "_curator_configured", False):
        return logger
    logger.setLevel(logging.INFO)
    try:
        h = logging.FileHandler(_CURATOR_LOG_PATH)
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(h)
    except OSError:
        logger.warning(
            "curator log %s not writable; using default handlers",
            _CURATOR_LOG_PATH,
        )
    logger._curator_configured = True  # type: ignore[attr-defined]
    return logger


# The decision types the curator records. Kept loose (validated as a set, not
# an enum in the schema) so an unforeseen-but-sensible label still logs rather
# than erroring mid-run.
_RECORD_ACTIONS = ("archive", "supersede", "dedup", "uncertain")


def _summarize_call(name: str, inp: dict[str, Any]) -> str:
    if name in ("write_file", "append_file"):
        return f"path={inp.get('path', '?')} ({len(inp.get('content', '') or '')} chars)"
    if name == "bash":
        return "$ " + (inp.get("command", "") or "").replace("\n", " ")[:160]
    return str(inp)[:160]


def build_record_action_tool(
    logger: logging.Logger, agent_id: str, dry_run: bool,
) -> Tool:
    """The curator's own decision ledger. Every archive/supersede/dedup/uncertain
    edit must be preceded by a ``record_action`` call, so the audit trail carries
    the model's *reasoning* — not just the mechanical tool call (which claw.log
    already records per turn). One reasoned line per decision lands in
    ``/var/log/claw-curator.log``."""
    async def _run(args: dict[str, Any]) -> str:
        action = (args.get("action") or "").strip().lower()
        target = (args.get("target") or "").strip()
        reason = (args.get("reason") or "").strip()
        if not action or not target or not reason:
            return "error: action, target, and reason are all required"
        if action not in _RECORD_ACTIONS:
            # Log it anyway (don't lose the record) but flag the odd label back
            # to the curator so it can self-correct.
            logger.info(
                "[%s] %sACTION %s | %s | %s",
                agent_id, "DRY-RUN " if dry_run else "",
                action.upper(), target, reason,
            )
            return (
                f"recorded, but '{action}' is not one of "
                f"{', '.join(_RECORD_ACTIONS)} — use one of those next time"
            )
        logger.info(
            "[%s] %sACTION %s | %s | %s",
            agent_id, "DRY-RUN " if dry_run else "",
            action.upper(), target, reason,
        )
        return "recorded"

    return Tool(
        name="record_action",
        description=(
            "Record a curation decision to the durable audit ledger. Call this "
            "IMMEDIATELY BEFORE every archive, supersession, dedup, or "
            "uncertain-flag edit — one call per decision. This is the only place "
            "your reasoning is preserved, so make `reason` a real justification "
            "(why this is stale/duplicate/superseded), not a restatement."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": (
                        "One of: archive, supersede, dedup, uncertain."
                    ),
                },
                "target": {
                    "type": "string",
                    "description": (
                        "What you're acting on: file, section header, and marker "
                        "id — e.g. `memory/2026-05-13.md#Daily Log (m-1a2b3c4d)`."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "One-sentence justification for the decision."
                    ),
                },
            },
            "required": ["action", "target", "reason"],
        },
        run=_run,
    )


def _dryrun_wrap(tool: Tool, logger: logging.Logger, agent_id: str) -> Tool:
    """Dry-run guard for a mutating tool: log the intended edit and return a
    stub WITHOUT executing, so a bootstrap ``--dry-run`` can be reviewed before
    it touches any file. Only applied when ``dry_run`` is set; a live nightly
    pass runs the tool unwrapped (its reasoning is captured by ``record_action``,
    and claw.log logs the tool-call name per turn)."""
    async def run(inp: dict[str, Any]) -> str:
        summary = _summarize_call(tool.name, inp)
        logger.info("[%s] DRY-RUN skip %s %s", agent_id, tool.name, summary)
        return f"[dry-run] {tool.name} not executed: {summary}"
    return Tool(tool.name, tool.description, tool.input_schema, run)


def _curator_tools(
    tools: dict[str, Tool], logger: logging.Logger, agent_id: str, dry_run: bool,
) -> dict[str, Tool]:
    out: dict[str, Tool] = {}
    for name in CURATOR_TOOL_NAMES:
        tool = tools.get(name)
        if tool is None:
            continue
        out[name] = (
            _dryrun_wrap(tool, logger, agent_id)
            if dry_run and name in _MUTATING_TOOLS else tool
        )
    out["record_action"] = build_record_action_tool(logger, agent_id, dry_run)
    return out


# --- working-set briefing ---------------------------------------------------

def _marker_display(meta: dict[str, Any]) -> str:
    parts = []
    if meta.get("ts"):
        parts.append(f"ts={meta['ts']}")
    if meta.get("mem_id"):
        parts.append(f"id={meta['mem_id']}")
    parts.append(f"status={meta.get('status', 'active')}")
    if meta.get("superseded_by"):
        parts.append(f"supersededBy={meta['superseded_by']}")
    return " ".join(parts)


def _recent_window_sources(
    memory: MemoryIndex, today: str, days: int,
) -> set[str]:
    """Source keys of daily notes whose filename-date is within ``days`` of
    today — the time-triggered rescan set for lapsed ephemera."""
    try:
        cutoff = date.fromisoformat(today) - timedelta(days=days)
    except ValueError:
        return set()
    out: set[str] = set()
    for p in memory._daily_notes():
        try:
            d = date.fromisoformat(p.name[:-3])
        except ValueError:
            continue
        if d >= cutoff:
            out.add(f"memory/{p.name}")
    return out


def _section_key(c: dict[str, Any]) -> str:
    return f"{c['metadata']['source']}#{c['metadata']['section']}"


def _select_candidate_files(
    memory: MemoryIndex,
    today: str,
    recent_window_days: int,
    full_corpus: bool,
    prev_hashes: dict[str, str],
) -> tuple[list[str], dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Group current sections by source file and choose which files get a
    curation turn this run. Returns ``(ordered_files, by_file, cur_hashes)``.

    The curator processes ONE file per turn (a whole-corpus briefing wouldn't
    fit context), so the working set is a list of *files*:
    - today's daily note is always excluded (the collector may be appending);
    - ``full_corpus`` (bootstrap): every non-today file whose sections aren't
      ALL already id-marked — so an interrupted bootstrap resumes (fully
      curated files are skipped);
    - nightly: files with a section changed since the watermark, plus
      recent-window files re-scanned for lapsed ephemera.
    """
    by_file: dict[str, list[dict[str, Any]]] = {}
    cur_hashes: dict[str, str] = {}
    for c in memory.collect_sections():
        cur_hashes[_section_key(c)] = hashlib.sha1(
            c["content"].encode("utf-8")
        ).hexdigest()
        by_file.setdefault(c["metadata"]["source"], []).append(c)

    today_src = f"memory/{today}.md"
    recent_srcs = _recent_window_sources(memory, today, recent_window_days)

    ordered: list[str] = []
    for src in sorted(by_file):  # MEMORY.md first, then daily notes chronological
        if src == today_src:
            continue
        secs = by_file[src]
        if full_corpus:
            take = not all(c["metadata"].get("mem_id") for c in secs)
        else:
            changed = any(
                prev_hashes.get(_section_key(c)) != cur_hashes[_section_key(c)]
                for c in secs
            )
            take = changed or src in recent_srcs
        if take:
            ordered.append(src)
    return ordered, by_file, cur_hashes


def _build_file_briefing(
    memory: MemoryIndex,
    src: str,
    file_sections: list[dict[str, Any]],
    today: str,
    near_neighbor_k: int,
) -> str:
    """One-file briefing: the file's memories, each with near-neighbours from
    elsewhere in memory so the curator can judge dedup/supersession. Bounded
    by a single file's section count, so it fits context."""
    lines: list[str] = [
        f"# Memory curation — {today}",
        f"\nYou are curating one file: `{src}` ({len(file_sections)} "
        "memories). Investigate (read_file / memory_search) before editing. "
        "Snippets are truncated — read the file for full content.",
    ]
    today_src = f"memory/{today}.md"
    for c in file_sections:
        lines.append(f"\n### {c['metadata']['section']}")
        lines.append(f"`{_marker_display(c['metadata'])}`")
        lines.append(c["content"][:1000])
        # Exclude today's note from neighbours: it's off-limits to the curator
        # and its memories have no ids yet, so they're not valid dedup/
        # supersession candidates (tomorrow's pass handles them).
        neighbors = [
            nb for nb in memory.search(c["content"], n=near_neighbor_k)
            if nb["text"][:80] != c["content"][:80]
            and nb["id"].rsplit(":", 1)[0] != today_src
        ]
        if neighbors:
            lines.append("\n**Near-neighbours elsewhere in memory:**")
            for nb in neighbors:
                nsrc = nb["id"].rsplit(":", 1)[0]
                nid = nb.get("mem_id") or "—"
                lines.append(
                    f"- [{nsrc}] **{nb['section']}** (id={nid}): {nb['text'][:200]}"
                )
    return "\n".join(lines)


def _build_supersession_briefing(
    memory: MemoryIndex, today: str, archive_days: int,
) -> str:
    """Brief for the supersession-review turn: EVERY memory currently carrying
    a ``supersededBy`` pointer, with how long it's been superseded — measured
    from the *superseding* memory's ``ts`` (the marker records when a memory
    was written, never when it was superseded). This is the complete set (any
    age, whole corpus, NOT the recent-file window), so long-buried supersessions
    still get reviewed. Returns ``''`` when there are none (skip the turn)."""
    sections = memory.collect_sections()
    by_id = {
        c["metadata"]["mem_id"]: c
        for c in sections if c["metadata"].get("mem_id")
    }
    try:
        today_d: date | None = date.fromisoformat(today)
    except ValueError:
        today_d = None

    rows: list[tuple[int, str]] = []  # (age_days, rendered) — sort oldest-first
    for c in sections:
        sup = c["metadata"].get("superseded_by")
        if not sup:
            continue
        m = c["metadata"]
        head = by_id.get(sup)
        head_ts = head["metadata"].get("ts") if head else None
        age = -1
        age_str = "unknown"
        if head_ts and today_d:
            try:
                age = (today_d - date.fromisoformat(head_ts)).days
                age_str = f"{age} days"
            except ValueError:
                pass
        if head is not None:
            head_desc = f"{head['metadata']['section']} (id={sup}, ts={head_ts or '?'})"
        else:
            head_desc = (
                f"id={sup} — **MISSING**: no memory has this id "
                "(dangling pointer → UNCERTAIN, do not archive)"
            )
        rows.append((
            age,
            f"\n### [{m['source']}] {m['section']}\n"
            f"`{_marker_display(m)}`\n"
            f"- superseded by: {head_desc}\n"
            f"- superseded for: **{age_str}**\n"
            f"- content: {c['content'][:300]}",
        ))
    if not rows:
        return ""
    rows.sort(key=lambda r: r[0], reverse=True)  # most-stale first
    header = (
        f"# Supersession review — {today}\n\n"
        "Below is the COMPLETE list of memories that currently carry a "
        "`supersededBy` pointer — any age, whole corpus, nothing hidden by the "
        "recent-file window. Each is kept searchable so its timeline stays "
        "visible, but once a replacement has settled the old copy becomes "
        "noise.\n\n"
        "Decide for EACH, case by case:\n"
        f"- replacement PRESENT and superseded ~{archive_days}+ days ago → "
        "usually ARCHIVE the old copy (keep it only if its history is genuinely "
        "still useful; archive a clearly-dead one sooner);\n"
        "- replacement MISSING (the supersededBy id matches no memory) → the "
        "supersession is unverifiable, so do NOT archive — flag the memory "
        "UNCERTAIN with an inline caveat and keep it searchable;\n"
        "- otherwise → keep. Be conservative when unsure.\n"
    )
    return header + "".join(r for _, r in rows)


# --- system prompt ----------------------------------------------------------

def _curation_system_prompt(today: str) -> str:
    return (
        "You are the nightly **memory curator** for a local AI agent. Keep its "
        "long-term memory truthful and tidy. You edit markdown files directly; "
        "markdown is the single source of truth (the search index is rebuilt "
        f"from it). Today is {today}.\n\n"
        "## The memory marker\n"
        "Every memory is a `##` section. Curated memories carry a metadata "
        "marker as the FIRST line of the section body:\n"
        "`<!-- mem ts=<date> id=<m-xxxxxxxx> status=<active|archived> "
        "supersededBy=<id> -->`\n"
        "- `ts` — when recorded (the collector writes this). Preserve it; if "
        f"absent, set it to the daily-note's filename date or {today}.\n"
        "- `id` — stable identity YOU assign: `m-` + 8 hex chars. Any section "
        "missing an `id` needs a fresh unique one (check existing ids, don't "
        "collide). NEVER reuse or change an existing id.\n"
        "- `status` — `active` (default) or `archived`.\n"
        "- `supersededBy` — optional; the id of the memory that replaces this "
        "one.\n"
        "\nThe marker is index-side metadata only — it is STRIPPED before the "
        "agent reads a memory at retrieval, so `ts` is provenance, not a date "
        "the agent can see. If a memory is time-sensitive (an appointment, "
        '"released on <date>", a version state), make sure the date appears in '
        "the memory TEXT — the section header or a bullet — not just the "
        "marker.\n\n"
        "## What to do (investigate before you edit)\n"
        "1. **Dedup.** When sections say the same thing (recurring reminders "
        "spawn many near-identical copies), keep ONE canonical copy (clearest, "
        "most recent) and ARCHIVE the rest. Do not count or weight repeats — "
        "they are noise.\n"
        "2. **Supersession.** When a newer memory updates/contradicts an older "
        "long-term fact, add `supersededBy=<new id>` to the OLD memory's "
        "marker and leave it in place — it stays searchable and the index "
        "surfaces the newer one alongside it (a visible timeline). Archive a "
        "superseded memory only if its history truly isn't worth keeping.\n"
        "3. **Archive (lapsed ephemera).** Some memories were only ever "
        "temporary — a past appointment, \"X happens Thursday\", \"today is "
        "someone's birthday\". Once the moment passes they are dead weight. "
        "Cron/reminder action reports are the most common case — \"reminder "
        "fired\", \"notification delivered to <user>\", \"cron ran as "
        "expected\", \"updated reminders-sent.json\". They are useful "
        "reassurance for about a day, then pure noise: archive them once they "
        "are more than a day old. Lift out any durable fact they carry first "
        "(a rescheduled appointment, a new standing arrangement) as its own "
        "memory — but the fired/delivered log itself is not worth keeping. "
        "Move the ENTIRE `##` section out of its daily note into "
        f"`{ARCHIVE_PATH}`, setting `status=archived` on its marker. Archived "
        "memories are preserved on disk but never retrieved.\n"
        "4. **Uncertain.** Reserve ONLY for a genuine conflict, contradiction, "
        "or glaring illogic between memories that you cannot resolve — NOT for "
        "facts you merely can't verify right now (absence of proof is not a "
        "contradiction). Flag with a short inline caveat in the memory text, "
        "e.g. `(uncertain: conflicts with <other> — unresolved)`. Keep it "
        "searchable; do not archive it.\n\n"
        "## How to edit (mechanics)\n"
        "- ARCHIVE a section: `read_file` the daily note, `write_file` it back "
        "WITHOUT that section, then `append_file` the removed section (marker "
        f"set to `status=archived`) to `{ARCHIVE_PATH}`.\n"
        "- Mark SUPERSESSION / add an id / add a caveat: `read_file` the file, "
        "edit the one section's marker or text, `write_file` the whole file "
        "back. Change ONLY the intended section.\n"
        "- Use `bash` (grep/cat) to investigate, but make all changes via "
        "read_file/write_file/append_file.\n\n"
        "## Record every decision (REQUIRED)\n"
        "IMMEDIATELY BEFORE each archive/supersede/dedup/uncertain edit, call "
        "`record_action(action, target, reason)` — one call per decision. This "
        "is the durable audit trail; the `reason` is the ONLY place your "
        "justification survives, so give a real one (why it is stale / a "
        "duplicate / superseded), not a restatement. Do not make one of these "
        "edits without first recording it. (Pure id-assignment or a ts backfill "
        "needs no record_action — only the four decision types do.)\n\n"
        "## Rules\n"
        "- Edit ONLY: `MEMORY.md`, `memory/YYYY-MM-DD.md` daily notes, and "
        f"`{ARCHIVE_PATH}`.\n"
        f"- NEVER edit today's note `memory/{today}.md` — a live session may be "
        "appending to it.\n"
        "- NEVER touch IDENTITY/USER/SOUL/AGENTS/TOOLS files, transcripts, or "
        "`.memory/`.\n"
        "- Be conservative: when unsure whether something is stale, leave it "
        "active. Keeping a memory beats wrongly burying it.\n\n"
        "When finished, reply with a one-line summary (or 'no changes needed')."
    )


def _supersession_review_prompt(today: str, archive_days: int) -> str:
    return (
        "You are the nightly **memory curator**, running the supersession-review "
        f"pass. Today is {today}. Markdown is the source of truth.\n\n"
        "The message lists the COMPLETE set of memories that currently carry a "
        "`supersededBy` pointer — any age, whole corpus (NOT limited to recent "
        "files). A superseded memory is kept searchable so the timeline stays "
        "visible, but once its replacement has settled the old copy is noise.\n\n"
        "For EACH listed memory decide:\n"
        f"- **Archive** it when its replacement is PRESENT and it has been "
        f"superseded long enough (rule of thumb ~{archive_days}+ days, measured "
        "from the replacement's date) that its history no longer earns its keep. "
        f"Move the ENTIRE `##` section out of its daily note into `{ARCHIVE_PATH}` "
        "with `status=archived`: `read_file` the note, `write_file` it back "
        "without that section, then `append_file` the section (marker flipped to "
        "`status=archived`, keep its id/ts/supersededBy) to the archive.\n"
        "- **Flag uncertain** when its replacement is MISSING (the supersededBy "
        "id matches no existing memory): the supersession can't be verified, so "
        "do NOT archive. Instead add a short inline caveat to the memory's text "
        "— e.g. `(uncertain: supersededBy points to a missing memory — "
        "unresolved)` — and keep it searchable (`read_file` the note, "
        "`write_file` it back with the caveat added to that one section).\n"
        "- **Keep** it (do nothing) when the timeline is genuinely still useful, "
        "or it was superseded only recently.\n\n"
        "Use `memory_search` (it returns ids) and `read_file` to confirm the "
        "replacement really is the current canonical version before archiving. "
        "Be conservative: when unsure, keep.\n\n"
        f"Edit ONLY daily notes and `{ARCHIVE_PATH}`. NEVER edit today's note "
        f"`memory/{today}.md`, and never touch IDENTITY/USER/SOUL/AGENTS/TOOLS, "
        "transcripts, or `.memory/`.\n\n"
        "IMMEDIATELY BEFORE each archive or uncertain-flag edit, call "
        "`record_action(action, target, reason)` (action `archive` or "
        "`uncertain`) — one call per decision, with a real justification. It is "
        "the durable audit trail; do not make the edit without first recording "
        "it.\n\n"
        "When finished, reply with a one-line summary (or 'no changes needed')."
    )


# --- core run ---------------------------------------------------------------

async def run_curation_turn(
    *,
    agent_id: str,
    ollama: OllamaClient,
    memory: MemoryIndex,
    tools: dict[str, Tool],
    cfg: Config,
    logger: logging.Logger,
    full_corpus: bool = False,
    dry_run: bool = False,
) -> bool:
    """Run a curation pass for an agent: ONE curator turn per candidate file
    (per-file batching — a whole-corpus briefing wouldn't fit context), THEN one
    whole-corpus supersession-review turn that GCs long-superseded memories.
    Each per-file turn investigates and edits that file itself; the watermark is
    advanced per file so an interrupted run resumes. The review turn runs every
    pass, independent of the working set. Returns True iff anything was curated."""
    today = today_iso_date(cfg.tz)
    cur = cfg.memory_curation
    prev_hashes = memory.load_curation_state().get("sectionHashes", {}) or {}

    loop = asyncio.get_running_loop()
    async with memory.lock:
        ordered, by_file, _ = await loop.run_in_executor(
            None, _select_candidate_files,
            memory, today, cur.recent_window_days, full_corpus, prev_hashes,
        )

    curator_tools = _curator_tools(tools, logger, agent_id, dry_run)
    # The curator's memory_search exposes marker ids / supersededBy (the
    # agent-facing tool hides them) so the curator can find and reference
    # supersession/dedup candidates anywhere in the corpus — any age, not just
    # the file in front of it (the recent-file window doesn't bound search).
    curator_tools["memory_search"] = build_curator_search_tool(memory, today)

    if cur.max_files_per_run and len(ordered) > cur.max_files_per_run:
        logger.info(
            "[%s] curation: %d candidate files, capping at %d this run",
            agent_id, len(ordered), cur.max_files_per_run,
        )
        ordered = ordered[:cur.max_files_per_run]

    logger.info(
        "[%s] curation start (today=%s full_corpus=%s dry_run=%s files=%d)",
        agent_id, today, full_corpus, dry_run, len(ordered),
    )

    seen = dict(prev_hashes)
    done = 0
    for i, src in enumerate(ordered, 1):
        async with memory.lock:
            briefing = await loop.run_in_executor(
                None, _build_file_briefing,
                memory, src, by_file[src], today, cur.near_neighbor_k,
            )
        logger.info("[%s] curating %s (%d/%d)", agent_id, src, i, len(ordered))
        try:
            await asyncio.wait_for(
                ollama.run_turn(
                    model=cur.model,
                    history=[{"role": "user", "content": briefing}],
                    system=_curation_system_prompt(today),
                    tools=curator_tools,
                    sid="curation",
                    workspace_dir=memory.workspace_dir,
                    label=f"{agent_id}:curate:{src}",
                    max_tool_turns=cur.max_tool_turns,
                    num_predict=cur.num_predict,
                ),
                timeout=cur.turn_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[%s] curation of %s timed out after %.0fs; moving on",
                agent_id, src, cur.turn_timeout_s,
            )
            continue
        except Exception:
            log.exception("[%s] curation of %s raised; moving on", agent_id, src)
            continue
        done += 1
        # Resumable watermark: refresh this file's section hashes from disk
        # (post-edit) so a re-run skips it. Section keys are prefixed by
        # source, so drop the file's old keys and overlay its current ones.
        if not dry_run:
            try:
                fresh = await loop.run_in_executor(None, memory.section_hashes)
                pfx = src + "#"
                seen = {k: v for k, v in seen.items() if not k.startswith(pfx)}
                seen.update({k: v for k, v in fresh.items() if k.startswith(pfx)})
                memory.write_curation_state(seen)
            except OSError:
                log.exception("[%s] failed to persist curation state", agent_id)

    # Final clean baseline over the whole corpus (captures cross-file edits,
    # e.g. supersededBy pointers added into MEMORY.md).
    if not dry_run and done:
        try:
            memory.write_curation_state(memory.section_hashes())
        except OSError:
            log.exception("[%s] failed to persist final curation state", agent_id)

    # Whole-corpus supersession review — runs every pass, independent of the
    # per-file working set, so memories superseded long ago still get GC'd on
    # nights with no file edits.
    reviewed = await _run_supersession_review(
        agent_id=agent_id, ollama=ollama, memory=memory,
        curator_tools=curator_tools, cur=cur, today=today,
        logger=logger, dry_run=dry_run, loop=loop,
    )

    logger.info(
        "[%s] curation done (%d file(s) curated, supersession_review=%s)",
        agent_id, done, reviewed,
    )
    return done > 0 or reviewed


async def _run_supersession_review(
    *,
    agent_id: str,
    ollama: OllamaClient,
    memory: MemoryIndex,
    curator_tools: dict[str, Tool],
    cur: Any,
    today: str,
    logger: logging.Logger,
    dry_run: bool,
    loop: Any,
) -> bool:
    """One whole-corpus turn: hand the curator the COMPLETE superseded list
    (with ages) and let it archive the stale ones case-by-case via its normal
    read/write/append tools. No deterministic rule — the model judges, but the
    list is complete so coverage doesn't depend on what it happens to search.
    Returns True iff the turn ran (briefing non-empty and didn't error)."""
    async with memory.lock:
        briefing = await loop.run_in_executor(
            None, _build_supersession_briefing,
            memory, today, cur.superseded_archive_days,
        )
    if not briefing:
        log.info("[%s] supersession review: no superseded memories", agent_id)
        return False
    logger.info("[%s] supersession review start", agent_id)
    try:
        await asyncio.wait_for(
            ollama.run_turn(
                model=cur.model,
                history=[{"role": "user", "content": briefing}],
                system=_supersession_review_prompt(today, cur.superseded_archive_days),
                tools=curator_tools,
                sid="curation",
                workspace_dir=memory.workspace_dir,
                label=f"{agent_id}:curate:supersessions",
                max_tool_turns=cur.max_tool_turns,
                num_predict=cur.num_predict,
            ),
            timeout=cur.turn_timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "[%s] supersession review timed out after %.0fs",
            agent_id, cur.turn_timeout_s,
        )
        return False
    except Exception:
        log.exception("[%s] supersession review raised", agent_id)
        return False
    if not dry_run:
        try:
            memory.write_curation_state(memory.section_hashes())
        except OSError:
            log.exception(
                "[%s] failed to persist curation state after review", agent_id,
            )
    return True


async def curate_agent(agent: Any, cfg: Config) -> None:
    """Nightly per-agent entry: run a curation pass then reindex so the edits
    land in retrieval immediately."""
    logger = curator_logger()
    try:
        ran = await run_curation_turn(
            agent_id=agent.id,
            ollama=agent.ollama,
            memory=agent.memory,
            tools=agent.tools,
            cfg=cfg,
            logger=logger,
        )
        if ran:
            await agent.memory.reindex_if_stale()
    except Exception:
        log.exception("[%s] curate_agent raised", agent.id)


async def curate_all_agents(agents: list[Any], cfg: Config) -> None:
    """Nightly entrypoint called from the gateway loop. Coupled to the
    collector: no-op when memory_flush is disabled."""
    if not cfg.memory_flush.enabled:
        log.info("curation skipped: memory_flush disabled (curator follows collector)")
        return
    for agent in agents:
        await curate_agent(agent, cfg)


# --- bootstrap CLI ----------------------------------------------------------

async def _bootstrap(config_path: str, agent_id: str, dry_run: bool) -> int:
    """One-off full-corpus curation over a single agent's existing memory.
    Run from the installed venv, ideally with the daemon stopped (or during
    the quiet hour) so the collector isn't appending to today's note."""
    from claw.skills import build_agent_registry
    from claw.tools.memory_search import build_memory_search_tool

    cfg = load(config_path)
    matches = [a for a in cfg.agents if a.id == agent_id]
    if not matches:
        log.error("unknown agent %r; known: %s",
                  agent_id, [a.id for a in cfg.agents])
        return 2
    ac = matches[0]

    ollama = OllamaClient(cfg.ollama)
    memory = MemoryIndex(ac.id, ac.workspace)
    await memory.warmup_async()
    await memory.reindex_if_stale()

    tools, _ = build_agent_registry(cfg, ac.workspace)
    tools["memory_search"] = build_memory_search_tool(memory)

    logger = curator_logger()
    log.info("bootstrap curation for %s (dry_run=%s)", agent_id, dry_run)
    try:
        ran = await run_curation_turn(
            agent_id=ac.id,
            ollama=ollama,
            memory=memory,
            tools=tools,
            cfg=cfg,
            logger=logger,
            full_corpus=True,
            dry_run=dry_run,
        )
        if ran and not dry_run:
            result = await memory.reindex_if_stale()
            log.info("post-bootstrap reindex: %s", result)
    finally:
        await ollama.aclose()
    return 0


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="claw-curate",
        description="One-off full-corpus memory curation (bootstrap pass).",
    )
    parser.add_argument("--config", required=True, help="path to claw.yaml")
    parser.add_argument("--agent", required=True, help="agent id to curate")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="log intended edits to the curator log without writing files",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    raise SystemExit(asyncio.run(_bootstrap(args.config, args.agent, args.dry_run)))


if __name__ == "__main__":
    cli()
