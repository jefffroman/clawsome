"""Per-agent orchestrator. Wires channel inbound -> ollama + tools + memory.

One ``Agent`` per ``AgentConfig`` block in claw.yaml; per-peer state lives
in the ``TranscriptStore`` keyed by ``session_id(channel, peer_id)``.

Each inbound message:

1. Run on-demand idle-recap (once per session per process) if not already done.
2. Memory retrieval.
3. Append user turn to transcript.
4. **If transcript tokens exceed ``compaction.mid_session_token_threshold``,
   spawn a single background task that runs flush then mid-session
   compaction off the critical path.** The user's reply is NOT delayed.
5. ``ollama.run_turn`` against the un-compacted history (the same-turn cost
   of bigger context is the explicit tradeoff for snappier UX).
6. Append result turns. Send final assistant text back to the channel.

Background compaction snapshots the rows-list at spawn time, runs
flush + summarize against that snapshot, then takes the per-session lock
and atomically swaps in a recap turn for the snapshot's older portion —
preserving any user/assistant turns that arrived during the work.

Both background gates — compaction and memory-flush growth — are evaluated
at ONE point, ``_spawn_bg_maintenance_if_needed``, at turn-end. Transcript
growth only ever comes from a turn appending rows, so a turn-boundary check
catches every growth event; a timer cannot see anything extra, and firing on
one risks starting GPU work alongside a live reply.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
import unicodedata
from datetime import datetime, timezone

from claw import logsetup
from claw.runctx import current_sid, current_turn_id
from claw.tools.builtin import kill_subagent_bash, kill_turn_bash
from claw.channel.base import Channel, InboundMessage
from claw.channel.envelope import format_inbound_envelope
from claw.commands import (
    KNOWN,
    ParsedCommand,
    command_usage,
    parse_command,
    usage,
)
from claw.compaction import (
    compaction_preview,
    maybe_idle_recap,
    run_mid_session_compact_async,
    will_compact,
    will_mid_session_compact,
)
from claw.config import AgentConfig, Config
from claw.memory import MemoryIndex
from claw.memory_flush import run_memory_flush
from claw.ollama import OllamaClient
from claw.sink import ChannelSink, Sink
from claw.tools.base import Tool
from claw.tools.subagent import (
    SubagentSpawner,
    build_subagent_list_tool,
    build_subagent_status_tool,
    build_subagent_stop_tool,
    build_subagent_spawn_tool,
)
from claw.transcript import (
    TranscriptStore,
    as_message,
    estimate_tokens,
    is_archived_transcript,
    session_id,
    sid_for_key,
)
from claw.tools.memory_search import build_memory_search_tool
from claw.triggers.initial_prompt import maybe_dispatch_initial_prompt
from claw.triggers.scheduler import (
    JobRunner,
    build_cron_add_tool,
    build_cron_list_tool,
    build_cron_remove_tool,
)
from claw.workspace_inject import render_extra_paths

log = logging.getLogger("claw.agent")


def _derive_peer_label(msg: InboundMessage) -> str:
    """Short human-meaningful identifier for ``msg``'s sender, used in
    run_turn labels. MXID localpart for matrix (``@user-1:example.org`` ->
    ``user-1``), sender_name for synthetic channels (cron / initial_prompt
    / subagent_completion), channel name as last resort.
    """
    sid = msg.sender_id
    if sid.startswith("@") and ":" in sid:
        return sid[1:].split(":", 1)[0]
    if sid:
        return sid
    if msg.sender_name:
        return msg.sender_name
    return msg.channel or "?"


def _has_visible_content(text: str) -> bool:
    """True if *text* has at least one character that actually renders.

    The reasoning-trace surface gates on this rather than ``str.strip()``.
    ``str.strip()`` only removes *whitespace*, so a trace whose entire body
    is zero-width / format / control characters — U+200B, U+FEFF, soft
    hyphen, etc., which qwen3 occasionally emits as its whole final-turn
    ``message.thinking`` — passes ``.strip()`` yet ``_as_thinking_blockquote``
    then renders a header-only block with an invisible body. Reject when
    every character is whitespace or in Unicode category C* (control /
    format / surrogate / private-use / unassigned). Shared by both the
    final-only (%thinking on) and every-step (%thinking full) paths so the
    two modes suppress body-less traces identically.
    """
    return any(
        not ch.isspace() and unicodedata.category(ch)[0] != "C"
        for ch in text
    )


def _quote_body(text: str) -> str:
    """Prefix every line with ``> `` (blank source lines become a bare
    ``>`` so the blockquote doesn't terminate mid-body). The matrix
    channel's markdown-it renderer turns this into a real
    ``<blockquote>``; literal HTML is escaped by its ``html=False``.
    """
    return "\n".join(
        f"> {line}" if line else ">" for line in text.splitlines()
    )


def _as_thinking_blockquote(text: str) -> str:
    """Render a model reasoning trace as a 🧠-headed blockquote."""
    return f"> 🧠 **reasoning**\n>\n{_quote_body(text)}"


def _as_system_blockquote(text: str) -> str:
    """Render a control-plane (admin command) reply as a 🖥️-headed
    blockquote, so it reads visually as a *system* message — distinct
    from the 🧠 reasoning surface and from ordinary agent prose.
    """
    return f"> 🖥️ **system**\n>\n{_quote_body(text)}"


# Separator prefixed to the answer when another bot block already went
# out this turn ahead of it — a %thinking reasoning block, or a 🖥️
# system block a command emitted while the turn was in flight. Those are
# consecutive same-sender Matrix events, which clients group tightly
# with no speaker-switch gap.
# A real blank line can't fix this: CommonMark discards body-edge
# whitespace, so "\n"/"\n\n" prepended to the answer (or appended to the
# block) renders identically to no separator — verified against the
# channel's exact markdown-it config. We need visible structure, so we
# prefix a zero-width-space (U+200B) then one newline: with the channel's
# breaks=True, "\u200b\n" -> <p>U+200B<br>...answer...</p> \u2014 a single
# hard line break above the answer (one-line gap). "\u200b\n\n" instead
# makes a whole empty <p>U+200B</p> paragraph (two-line gap \u2014 the
# original tuning, found heavier than needed). ZWSP rather than a bare
# " " (-> empty/stripped) so the spacer line survives. If the answer
# opens with a block construct (code fence, list, heading, blockquote)
# it interrupts the ZWSP paragraph and renders exactly as the two-line
# form did \u2014 a graceful content-preserving fallback.
# U+200B is the one character _has_visible_content() rejects, but that
# gate only ever inspects *reasoning traces*, never the answer, so
# reusing it here as a deliberate spacer cannot interfere with
# body-less-block suppression. Escape (not a literal ZWSP) so the source
# stays greppable.
_THINKING_ANSWER_SEP = "\u200b\n"


# Provenance marker prepended (as its own user-slot row) to a cron reply when
# it is mirrored into the human-facing session. Cron turns are stateless \u2014 they
# keep no session transcript of their own (see _process_batch), so the mirror is
# the *only* durable record of the interaction. It therefore carries both the
# trigger prompt AND the reply, so a follow-up from the human reads against full
# context (trigger \u2192 reply \u2192 their message) and the agent doesn't mistake the
# mirrored reply for an answer to some missing user turn.
#
# It lives in the *content* of a user-role row, not a novel role: qwen3's chat
# template silently drops messages whose role isn't system/user/assistant/tool
# (verified empirically), so a "cron" role would vanish. User slot + a \u2699\ufe0f
# self-label matches the existing persisted-annotation convention (the
# missed-review turn, the per-turn envelope) that the model already reads as
# meta without echoing. The row is transcript-only \u2014 never re-sent \u2014 so the
# user never sees it.
def _cron_mirror_note(trigger: str) -> str:
    return (
        "\u2699\ufe0f System note \u2014 not from the user. A scheduled (cron) trigger fired "
        "with this instruction to you:\n\n"
        f"{_quote_body(trigger)}\n\n"
        "The assistant message that follows is the response you generated for "
        "that trigger and delivered in this channel. Recorded for context in "
        "case the user replies."
    )


class Agent:
    def __init__(
        self,
        cfg: Config,
        agent_cfg: AgentConfig,
        ollama: OllamaClient,
        memory: MemoryIndex,
        tools: dict[str, Tool],
        transcripts: TranscriptStore,
        channel: Channel,
        skill_catalog: str = "",
        spawner: SubagentSpawner | None = None,
        depth: int = 0,
        job_runner: JobRunner | None = None,
        remaining_spawn_budget: int | None = None,
        allowed_spawn_personas: tuple[str, ...] | None = None,
        parent_id: str | None = None,
        spawn_task_id: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.agent_cfg = agent_cfg
        self.ollama = ollama
        self.memory = memory
        # Copy tools so we can mutate without mutating the caller's dict.
        self.tools = dict(tools)
        self.transcripts = transcripts
        self.channel = channel
        # Additional outbound channels keyed by inbound channel name (e.g.
        # "voice"). A turn's reply is delivered back to the channel the inbound
        # arrived on; self.channel (the agent's primary, matrix) is the default.
        self._channels: dict[str, Channel] = {}
        self.skill_catalog = skill_catalog
        self.spawner = spawner
        self.depth = depth
        self.job_runner = job_runner
        # Budget: how many more chain levels can spawn under me. For top-level
        # agents the runtime fills in agent_cfg.max_spawn_depth. For forks the
        # caller computes min(parent_remaining - 1, persona.max_spawn_depth).
        self.remaining_spawn_budget = (
            remaining_spawn_budget
            if remaining_spawn_budget is not None
            else agent_cfg.max_spawn_depth
        )
        # Persona allowlist. None = any persona allowed (subject to budget).
        # Tuple = restricted set. Caller passes the persona/agent's can_spawn
        # if any; otherwise we fall back to agent_cfg.can_spawn.
        self.allowed_spawn_personas = (
            allowed_spawn_personas
            if allowed_spawn_personas is not None
            else agent_cfg.can_spawn
        )
        # Set on subagent forks — used to label run_turn invocations as
        # ``<parent_id>:subagent:<spawn_task_id>`` so the log stream
        # self-identifies whose child this is and which spawn produced it.
        # None for top-level agents.
        self.parent_id = parent_id
        self.spawn_task_id = spawn_task_id

        # Per-session state.
        self._session_locks: dict[str, asyncio.Lock] = {}
        # Where this agent's replies go. None => a ChannelSink built per
        # turn from the inbound's channel + peer. A forked subagent sets a
        # ParentSink here instead, which is the only difference between
        # the two kinds of agent on the reply path.
        self._sink_override: Sink | None = None
        # Pending inbound messages per session. Messages that arrive while a
        # turn is in flight queue here and get coalesced into the next
        # batch (one combined user turn -> one run_turn -> one reply).
        self._pending_inbound: dict[str, list[InboundMessage]] = {}
        # Wake-event per session — handle_inbound sets it; drainer awaits it.
        self._has_pending: dict[str, asyncio.Event] = {}
        # Long-lived drainer task per session. Decouples matrix-nio's
        # sequential callback dispatch from agent processing — without this,
        # sync_forever blocks on each handle_inbound and msg N+1 sits on
        # Synapse until msg N's reply has been sent, defeating coalescing.
        self._drainer_tasks: dict[str, asyncio.Task] = {}
        # Sid -> turn_id of the turn the drainer is actively inside
        # _process_batch for (a real conversational turn, not idle-waiting
        # / flush / compaction). Presence = a turn is in flight; the value
        # is the turn id %stop cancels (and the key used to scope its
        # cascade-cancel + bash-kill). Set by the drainer around the
        # _process_batch call, popped in its finally.
        self._inflight_turn: dict[str, str] = {}
        # turn_ids that had a 🖥️ system (command) block emitted while
        # in-flight. Mirrors the thinking_emitted flag: such a block and
        # the turn's answer are consecutive same-sender events, so the
        # answer gets the same one-line _THINKING_ANSWER_SEP gap. Tagged
        # at the _cmd_reply chokepoint, read in _process_batch before the
        # answer send, discarded by the drainer's finally (so it never
        # leaks for answerless turns or post-answer system sends).
        self._system_emitted_turns: set[str] = set()
        # At most one background flush+compact task per session at a time.
        self._bg_compaction: dict[str, asyncio.Task] = {}
        # Token count at last periodic flush per session, for delta-trigger.
        self._last_periodic_flush_tokens: dict[str, int] = {}
        # Row count at last successful flush per session, for incremental
        # slicing. Lazy-loaded from the on-disk sidecar so the count survives
        # claw restarts (otherwise the first periodic tick after restart
        # would re-flush the entire transcript).
        self._last_flushed_row_count: dict[str, int] = {}
        # Sids with a flush turn currently in flight. Bg flushes (pre-compact,
        # periodic) skip if their sid is present; the synchronous pre-rotate
        # flush waits on _flush_lock(sid) instead.
        self._flush_in_flight: set[str] = set()
        # Per-session lock serializing flush turns. Bg flushes acquire-and-hold
        # for the duration; pre-rotate awaits to run after any bg flush.
        self._flush_locks: dict[str, asyncio.Lock] = {}
        # Per-session timestamp of the previous _process_batch invocation.
        # Powers the envelope's elapsed-time delta. Cleared across restarts —
        # the first inbound after a restart simply has no `+elapsed` suffix.
        self._last_inbound_at: dict[str, datetime] = {}
        # (channel, peer_id) of the inbound currently being processed —
        # set at the top of _process_batch, read by subagent_spawn so the
        # eventual completion message can be delivered to the same
        # session. None outside an active turn.
        self._active_inbound: tuple[str, str] | None = None
        # Per-session cache of the last-seen peer_label, populated in
        # _process_batch. Background flush sites (periodic, pre-rotate)
        # read this so their log labels match the conversation's peer.
        self._peer_label_by_sid: dict[str, str] = {}

        self._workspace_block: str | None = None
        # Boot-time recap state, keyed by session id.
        self._idle_recapped: set[str] = set()
        self._idle_recap_blocks: dict[str, str] = {}
        # Per-session reasoning-trace surfacing, set by %thinking. Maps
        # sid -> "final" (the final answer turn's reasoning, =#53) or
        # "full" (every loop iteration's reasoning, a superset — the
        # final-only send is suppressed for that sid so the last block
        # isn't duplicated). Absent => off. Ephemeral and in-memory:
        # never persisted, resets on daemon restart (same as #53), and a
        # default-empty map is zero behavior change until a conversation
        # explicitly opts in.
        self._thinking_mode: dict[str, str] = {}

        # Always expose memory_search — read-only retrieval over this agent's
        # MemoryIndex. Subagents inherit it via their parent's tool registry.
        self.tools["memory_search"] = build_memory_search_tool(memory)

        # Expose subagent_spawn only if I have remaining budget. Per-agent
        # gate; the global cfg.subagents has no max_spawn_depth field.
        # The subagent_{status,list,stop} family pairs with subagent_spawn —
        # the parent always wants to inspect/cancel the task_ids it was
        # just handed.
        if spawner is not None and self.remaining_spawn_budget > 0:
            self.tools["subagent_spawn"] = build_subagent_spawn_tool(self, spawner, depth)
            self.tools["subagent_status"] = build_subagent_status_tool(self, spawner)
            self.tools["subagent_list"] = build_subagent_list_tool(self, spawner)
            self.tools["subagent_stop"] = build_subagent_stop_tool(self, spawner)
        # cron_* exposure is gated by config (cron.exposed_to). Single
        # responsibility per deployment — typically one agent owns
        # scheduling — so we don't hardcode names. Subagents inherit
        # the parent's tools but the spawn step strips the cron_* family
        # so children can't escalate.
        if (
            job_runner is not None
            and depth == 0
            and cfg.cron.enabled
            and self.id in cfg.cron.exposed_to
        ):
            self.tools["cron_add"] = build_cron_add_tool(self.id, job_runner, cfg.cron.default_deliver_to, cfg.tz)
            self.tools["cron_list"] = build_cron_list_tool(job_runner)
            self.tools["cron_remove"] = build_cron_remove_tool(job_runner)

    @property
    def id(self) -> str:
        return self.agent_cfg.id

    @property
    def compaction_model(self) -> str:
        return self.agent_cfg.compaction_model or self.cfg.ollama.default_compaction_model

    @property
    def num_predict(self) -> int | None:
        """Effective per-call generation cap. Per-agent (or persona) override
        wins; otherwise global OllamaConfig.num_predict applies. None / -1
        disables the cap."""
        if self.agent_cfg.num_predict is not None:
            return self.agent_cfg.num_predict
        return self.cfg.ollama.num_predict

    @property
    def max_tool_turns(self) -> int:
        """Effective tool-loop ceiling for this agent's run_turn calls.
        Per-agent (or persona) override wins; otherwise global
        OllamaConfig.max_tool_turns applies."""
        if self.agent_cfg.max_tool_turns is not None:
            return self.agent_cfg.max_tool_turns
        return self.cfg.ollama.max_tool_turns

    @property
    def scratch_sid(self) -> str:
        """This subagent's session id. Keyed by task_id, not agent id: two
        concurrent spawns of the same persona from the same parent share an
        agent id and would otherwise collide in one transcript."""
        return f"subagent-{self.spawn_task_id or self.id}"

    def attach_sink(self, sink: Sink) -> None:
        """Make this agent a subagent: its replies go to ``sink`` instead of a
        channel, and it stops being eligible for a conversational drainer."""
        self._sink_override = sink

    @property
    def sink(self) -> Sink | None:
        """Where this agent's replies go, when it is not a participant."""
        return self._sink_override

    def _sink_for(self, channel: Channel, peer_id: str) -> Sink:
        return self._sink_override or ChannelSink(channel, peer_id)

    def _session_lock(self, sid: str) -> asyncio.Lock:
        return self._session_locks.setdefault(sid, asyncio.Lock())

    def _flush_lock(self, sid: str) -> asyncio.Lock:
        return self._flush_locks.setdefault(sid, asyncio.Lock())

    async def _run_flush_guarded(
        self,
        *,
        sid: str,
        full_rows: list[dict],
        reason: str,
        mode: str,
    ) -> bool:
        """Slice ``full_rows`` to just the rows added since the previous
        successful flush for ``sid``, gate against concurrent flushes, run
        the flush turn, and persist the new row count on success.

        ``mode="bg"`` (pre-compact, periodic): return immediately if another
        flush is in flight for this sid (skip-if-busy). ``mode="sync"``
        (pre-rotate, on session archival): await the per-sid flush lock so
        we run after any in-flight bg flush completes.

        If ``len(full_rows)`` is below the stored count, the transcript was
        replaced (mid-session compaction) — reset to 0 and re-flush from the
        start. If the slice is empty (no new rows), skip.

        Returns True iff a flush turn ran AND the agent's append_file
        succeeded (``run_memory_flush`` return).
        """
        if not self.cfg.memory_flush.enabled:
            return False
        if mode not in ("bg", "sync"):
            raise ValueError(f"unknown flush mode: {mode!r}")

        last_count = self._last_flushed_row_count.get(sid)
        if last_count is None:
            last_count = self.transcripts.read_flush_state(sid)
            self._last_flushed_row_count[sid] = last_count
        if last_count > len(full_rows):
            last_count = 0  # transcript shrank (replace()); start over

        new_rows = full_rows[last_count:]
        if not new_rows:
            log.debug(
                "[%s] no new rows since last flush, skipping (reason=%s, sid=%s)",
                self.id, reason, sid,
            )
            return False

        if mode == "bg":
            if sid in self._flush_in_flight:
                log.debug(
                    "[%s] flush already in flight, skipping (reason=%s, sid=%s)",
                    self.id, reason, sid,
                )
                return False
            self._flush_in_flight.add(sid)
        try:
            async with self._flush_lock(sid):
                try:
                    ok = await asyncio.wait_for(
                        run_memory_flush(
                            agent_id=self.id,
                            peer_label=self._peer_label_by_sid.get(sid, "?"),
                            ollama=self.ollama,
                            sid=sid,
                            workspace_dir=self.agent_cfg.workspace,
                            rows=new_rows,
                            primary_model=self.compaction_model,
                            tools={"append_file": self.tools["append_file"]},
                            reason=reason,
                            tz_name=self.cfg.tz,
                        ),
                        timeout=self.cfg.memory_flush.turn_timeout_s,
                    )
                except asyncio.TimeoutError:
                    log.warning(
                        "[%s] %s flush timed out after %.0fs for %s",
                        self.id, reason,
                        self.cfg.memory_flush.turn_timeout_s, sid,
                    )
                    return False
                except Exception:
                    log.exception("[%s] %s flush raised for %s", self.id, reason, sid)
                    return False
                if ok:
                    new_count = len(full_rows)
                    self._last_flushed_row_count[sid] = new_count
                    self._last_periodic_flush_tokens[sid] = estimate_tokens(full_rows)
                    try:
                        self.transcripts.write_flush_state(sid, new_count)
                    except OSError:
                        log.exception(
                            "[%s] failed to persist flush state for %s",
                            self.id, sid,
                        )
                return ok
        finally:
            if mode == "bg":
                self._flush_in_flight.discard(sid)

    def _workspace_system_block(self) -> str:
        if self._workspace_block is None:
            self._workspace_block = render_extra_paths(
                self.agent_cfg.workspace, self.agent_cfg.extra_paths,
            )
        return self._workspace_block

    def _build_system_prompt(
        self,
        recap_block: str | None = None,
        modality: str | None = None,
    ) -> str:
        """The stable half of the prompt: workspace files, skill catalog, and
        any steering hints.

        Everything here changes rarely — at most once per idle recap — which
        is what makes the KV prefix cache useful. Per-turn volatile content
        (retrieved memory) deliberately does NOT live here; see
        ``_retrieval_row``.
        """
        parts: list[str] = []
        ws = self._workspace_system_block()
        if ws:
            parts.append(ws)
        # Language steering for ALL modalities/channels: a bilingual model can
        # code-switch (e.g. qwen replying in Chinese), which is merely odd on text
        # but breaks voice — the TTS can't read non-Latin output. Gated on a
        # configured language so it's opt-in.
        if self.cfg.language:
            parts.append(f"Always reply in {self.cfg.language}.")
        if self.skill_catalog:
            parts.append(self.skill_catalog)
        # Voice-modality steering: when the turn's modality is "voice", prepend
        # a note so the agent keeps its reply brief and speakable and accounts
        # for STT homophones. Gated on modality (not channel) so a voice turn
        # keeps the hint even when it lives in a shared "home" session next to
        # non-voice (event/state) sources; gated on a non-empty hint so an
        # operator can disable it via config.
        voice_hint = self._voice_modality_hint(modality)
        if voice_hint:
            parts.append(voice_hint)
        if recap_block:
            parts.append(recap_block)
        return "\n\n".join(parts)

    @staticmethod
    def _retrieval_row(retrieval_block: str) -> dict | None:
        """The per-turn retrieved-memory block, as a history row rather than a
        system-prompt section. None when retrieval came back empty.

        This block is re-ranked against every user message, so its content
        changes whenever the topic shifts. Attention is causal, so the KV cache
        is a *prefix* cache: changing a token invalidates every token after it.
        While this sat at the end of the system prompt — i.e. immediately
        BEFORE the whole transcript — each change invalidated the entire
        conversation and forced a full re-prefill. Observed on 2026-08-08: the
        cache restore point collapsed from ~75k to 3551 and prompt eval went
        from 720 ms to 353 s on a 70k-token prompt.

        Emitting it near the tail instead confines that invalidation to the
        last turn's worth of tokens. It is NOT persisted to the transcript —
        it's ephemeral per-turn context, and persisting it would accumulate
        stale retrievals forever.
        """
        if not retrieval_block:
            return None
        return {
            "role": "user",
            "content": (
                f"<retrieved_memory>\n{retrieval_block}\n</retrieved_memory>"
            ),
        }

    @classmethod
    def _with_retrieval(cls, history: list[dict], retrieval_block: str) -> list[dict]:
        """``history`` with the retrieval row spliced in just before the final
        (current) user message, so the user's own text stays last.

        Returns the list unchanged when there's nothing to retrieve.
        """
        row = cls._retrieval_row(retrieval_block)
        if row is None:
            return history
        if not history:
            return [row]
        return [*history[:-1], row, history[-1]]

    def _voice_modality_hint(self, modality: str | None) -> str:
        """The configured voice-modality system note, or "" when the turn
        isn't a voice turn / voice isn't configured / the hint is empty."""
        if modality != "voice":
            return ""
        voice_cfg = self.cfg.voice
        if voice_cfg is None or not voice_cfg.modality_hint:
            return ""
        return voice_cfg.modality_hint

    # --- subagent fork --------------------------------------------------

    def fork(self, persona: str, task_id: str | None = None) -> "Agent":
        """Return a transient child agent for one-shot subagent execution.

        ``task_id`` is the spawner-assigned id for this specific spawn; the
        child stores it for use in run_turn labels so logs self-identify
        as ``<parent_id>:subagent:<task_id>``.
        """
        persona_cfg = self.cfg.subagents.personas[persona.lower()]
        child_id = f"{self.id}.{persona.lower()}"
        if self.depth > 0:
            child_id += f".d{self.depth + 1}"
        child_cfg = AgentConfig(
            id=child_id,
            workspace=self.agent_cfg.workspace,
            primary_model=persona_cfg.model,
            matrix=self.agent_cfg.matrix,
            compaction_model=self.agent_cfg.compaction_model,
            extra_paths=self.agent_cfg.extra_paths,
            # Persona-level num_predict flows through the same per-agent
            # resolution path as a top-level agent's override; None falls
            # back to the global OllamaConfig.num_predict.
            num_predict=persona_cfg.num_predict,
            # Same pattern for max_tool_turns: persona-level override flows
            # via child AgentConfig; None inherits the global.
            max_tool_turns=persona_cfg.max_tool_turns,
        )
        base_tools = {
            k: v for k, v in self.tools.items()
            if k != "subagent_spawn" and not k.startswith("cron_")
        }
        # Child's spawn budget = min(parent_remaining - 1, persona's own ceiling).
        # Both constraints must hold; whichever is tighter wins. Floored at 0.
        child_budget = max(
            0,
            min(self.remaining_spawn_budget - 1, persona_cfg.max_spawn_depth),
        )
        return Agent(
            cfg=self.cfg,
            agent_cfg=child_cfg,
            ollama=self.ollama,
            memory=self.memory,
            tools=base_tools,
            # Scratch store: <workspace>/transcripts/subagent/. A subagent's
            # context is real (it survives between its own turns, so a child
            # can wake it) but must never sit in the participant's transcript
            # dir — every walker there filters on *.jsonl, so a subdirectory
            # is skipped by rotation, boot recap and the flush pass alike.
            # A fork of a fork reuses the same store rather than nesting.
            transcripts=(
                self.transcripts if self.depth > 0
                else TranscriptStore(self.transcripts.dir / "subagent")
            ),
            channel=self.channel,
            skill_catalog=self.skill_catalog,
            spawner=self.spawner,
            depth=self.depth + 1,
            remaining_spawn_budget=child_budget,
            allowed_spawn_personas=persona_cfg.can_spawn,
            parent_id=self.id,
            spawn_task_id=task_id,
        )

    async def run_task(self, prompt: str | None = None) -> str:
        """One turn of a subagent, against its own scratch session.

        Replaces ``run_one_shot``. The difference that matters is that the turn
        is backed by a real transcript and an inbox, so this agent can be woken
        by its own children and resume with its context intact. ``run_one_shot``
        discarded ``new_messages``, which is why a subagent that spawned a child
        had nowhere for the result to land: it was a callable, and the delivery
        mechanism needs a participant.

        Called with ``prompt`` for the opening turn and without it for a resume,
        where the queued child report is the only new input.

        Still deliberately unlike ``_process_batch``: no idle recap, no
        compaction, no memory flush, no envelope wrap. Those are participant
        concerns, and the opening prompt must stay byte-identical to what
        subagents have always been given.
        """
        sid = self.scratch_sid
        # Ambient sid for this subagent's own work, so its bash/web_search
        # spools land under subagent-<task_id>/ (which sweep_spool_tree already
        # documents) instead of the human's session dir, and so any grandchild
        # it spawns keys its completion to THIS session rather than the
        # participant's.
        tok_sid = current_sid.set(sid)
        # subagent_spawn reads this to address its child's completion back here.
        self._active_inbound = ("subagent", self.spawn_task_id or self.id)
        try:
            if prompt is not None:
                self.transcripts.append(sid, {"role": "user", "content": prompt})
            # Reports that landed while this subagent was idle between turns.
            # (Ones arriving mid-turn are picked up by drain_inbox below.)
            for row in self._take_subagent_completions(sid):
                self.transcripts.append(sid, row)

            rows = self.transcripts.load(sid)
            try:
                retrieval_block = await self.memory.retrieve_markdown(
                    prompt or (rows[-1].get("content") if rows else "") or "",
                    top_n=self.cfg.memory_retrieval.top_n,
                    compact=self.cfg.memory_retrieval.compact,
                )
            except Exception:
                log.exception("[%s] subagent memory retrieve failed", self.id)
                retrieval_block = ""
            history = self._with_retrieval(
                [as_message(r) for r in rows], retrieval_block,
            )
            system = self._build_system_prompt()

            if self.parent_id and self.spawn_task_id:
                label = f"{self.parent_id}:subagent:{self.spawn_task_id}"
            else:
                label = f"subagent:{sid}"
            new_messages, final_text, _ = await self.ollama.run_turn(
                model=self.agent_cfg.primary_model,
                history=history,
                system=system,
                tools=self.tools,
                sid=sid,
                workspace_dir=self.agent_cfg.workspace,
                label=label,
                num_predict=self.num_predict,
                max_tool_turns=self.max_tool_turns,
                drain_inbox=lambda: self._take_subagent_completions(sid),
            )
            for m in new_messages:
                self.transcripts.append(sid, m)
            return final_text or ""
        except Exception:
            log.exception("[%s] subagent run_turn failed", self.id)
            return f"error: subagent {self.id} run_turn failed"
        finally:
            current_sid.reset(tok_sid)

    def discard_scratch_session(self) -> None:
        """Reap this subagent's session. Its handle is gone, so nothing can
        reach it again; a later spawn of the same persona must start clean
        rather than inherit this one's context."""
        sid = self.scratch_sid
        self._pending_inbound.pop(sid, None)
        self._has_pending.pop(sid, None)
        self._session_locks.pop(sid, None)
        try:
            (self.transcripts.dir / f"{sid}.jsonl").unlink(missing_ok=True)
        except OSError:
            log.exception("[%s] failed to reap scratch session %s", self.id, sid)

    async def wait_for_inbound(self, timeout: float) -> bool:
        """Block until something lands in this subagent's inbox. False on
        timeout."""
        ev = self._has_pending.setdefault(self.scratch_sid, asyncio.Event())
        try:
            await asyncio.wait_for(ev.wait(), timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return False
        ev.clear()
        return True

    # --- boot recap -----------------------------------------------------

    async def boot_recap_known_sessions(self) -> None:
        """Eagerly run idle compaction on every existing transcript before
        going live. Adds ~1-2 s per stale session to cold boot but means
        the first user turn after boot doesn't pay summarizer latency.
        """
        td = self.transcripts.dir
        if not td.is_dir():
            return
        for entry in sorted(os.listdir(td)):
            if not entry.endswith(".jsonl"):
                continue
            if is_archived_transcript(entry):
                continue
            sid = entry[: -len(".jsonl")]
            try:
                block = await maybe_idle_recap(
                    cfg=self.cfg,
                    ollama=self.ollama,
                    transcripts=self.transcripts,
                    sid=sid,
                    compaction_model=self.compaction_model,
                )
            except Exception:
                log.exception("[%s] boot recap failed for %s", self.id, sid)
                self._idle_recapped.add(sid)
                continue
            self._idle_recapped.add(sid)
            if block:
                self._idle_recap_blocks[sid] = block
                log.info("[%s] boot recap installed for session %s", self.id, sid)

    # --- inbound handling -----------------------------------------------

    def register_channel(self, name: str, channel: Channel) -> None:
        """Register an additional outbound channel. A turn whose inbound
        ``channel`` equals ``name`` delivers its reply here instead of the
        agent's primary channel. Used to wire the voice channel so a voice
        turn's reply (and TTS) goes back to the box, not to matrix."""
        self._channels[name] = channel

    def _channel_for(self, channel_name: str) -> Channel:
        """Outbound channel for an inbound from ``channel_name``; falls back to
        the agent's primary channel (matrix) for matrix + synthetic channels."""
        return self._channels.get(channel_name, self.channel)

    async def _mirror_synthetic_reply(
        self, channel: Channel, peer_id: str, own_sid: str,
        trigger: str, text: str,
    ) -> None:
        """Copy a cron turn's trigger + delivered reply into the human-facing
        session for ``peer_id`` so a later reply from that human has context.

        Cron turns are stateless (no session transcript of their own), and the
        text delivered to ``deliver_to`` never lands in the matrix DM/room
        transcript the human actually replies in. So this is the only durable
        record: a user-slot provenance note carrying the ``trigger`` prompt,
        then the assistant ``text``. The agent reads it as its own proactive
        message (with the trigger that caused it), not an answer to a missing
        user turn.

        No-ops unless the outbound channel can resolve ``peer_id`` to a
        *distinct* primary session (``primary_session_key``) — a normal matrix
        turn resolves back to its own sid (guarded), and channels without the
        capability are skipped. The two rows are appended under the target
        session's lock, in one hold, so a mid-session compaction swap on that
        session (which re-reads then rewrites the transcript) can neither split
        the note from its reply nor race the append. Lock order is always
        cron→matrix, never the reverse, so nesting it inside the cron turn's
        own session lock can't deadlock.
        """
        resolver = getattr(channel, "primary_session_key", None)
        if resolver is None:
            return
        try:
            key = resolver(peer_id)
        except Exception:
            log.exception("[%s] primary_session_key(%s) raised", self.id, peer_id)
            return
        if not key:
            return
        target_sid = sid_for_key(key)
        if target_sid == own_sid:
            return
        note = _cron_mirror_note(trigger)
        async with self._session_lock(target_sid):
            self.transcripts.append(target_sid, {"role": "user", "content": note})
            self.transcripts.append(target_sid, {"role": "assistant", "content": text})
        log.info(
            "[%s] mirrored cron reply into primary session %s (from %s)",
            self.id, target_sid, own_sid,
        )
        # Upholds the invariant on _spawn_bg_maintenance_if_needed: this is the
        # only path that appends to a transcript the running turn does not own,
        # so the mirrored-into session would otherwise never have its gates
        # evaluated. Safe here — a cron turn has already replied by this point,
        # so nothing is generating.
        self._spawn_bg_maintenance_if_needed(
            target_sid, self.transcripts.load(target_sid),
        )

    async def handle_inbound(self, msg: InboundMessage) -> None:
        """Enqueue the message and signal the per-session drainer. Returns
        in microseconds so matrix-nio's sync_forever can immediately fire
        the next callback (and itself keep polling Synapse for newer
        events). This is what makes coalescing actually work — without
        the decoupling, nio serializes callbacks and msg N+1 doesn't reach
        the queue until msg N's _process_batch has already been popped.

        An in-band admin command short-circuits here before enqueue, so the
        command text is never transcribed and never reaches the LLM (see the
        gate below).
        """
        sid = sid_for_key(msg.session_key)

        # Control plane: in-band admin commands. A message is a command only
        # when commands are enabled, it came from matrix (synthetic channels —
        # cron / subagent_completion / initial_prompt — bypass via this
        # check), the sender is on the command allowlist, and the body parses.
        # If any condition fails, control falls through to the normal enqueue
        # path and the text is processed as an ordinary turn — no command, no
        # reply, no indication the sigil meant anything (identical to an
        # unauthorized user typing it). Dispatched as a detached task so this
        # method keeps its microsecond return; the task acquires the session
        # lock itself inside clear_session / forced compaction and therefore
        # serializes behind any in-flight turn for this session.
        if (
            msg.channel == "matrix"
            and self.cfg.commands.enabled
            and self._command_authorized(msg.sender_id)
        ):
            cmd = parse_command(msg.text, self.cfg.commands.prefix)
            if cmd is not None:
                asyncio.create_task(
                    self._handle_command(sid, msg, cmd),
                    name=f"cmd-{self.id}-{sid}-{cmd.name or 'help'}",
                )
                return  # not enqueued, not transcribed, not sent to the LLM

        self._pending_inbound.setdefault(sid, []).append(msg)
        # A subagent drives its own turns from _run_subagent and has no
        # channel to reply on, so it must NEVER start a conversational
        # drainer. Without this guard a grandchild's completion would spin one
        # up on a fork: it would take the FORK's session lock (a different
        # object from the participant's, so no mutual exclusion), run a full
        # turn against the human's transcript, and post to the room. The queue
        # is still filled above — run_task drains it between and within turns.
        if self._sink_override is not None:
            self._has_pending.setdefault(sid, asyncio.Event()).set()
            return
        # Lazy-create the drainer for this session on first inbound.
        if sid not in self._drainer_tasks or self._drainer_tasks[sid].done():
            self._drainer_tasks[sid] = asyncio.create_task(
                self._drain(sid, msg.peer_id, msg.channel),
                name=f"drain-{self.id}-{sid}",
            )
        # Wake the drainer.
        self._has_pending.setdefault(sid, asyncio.Event()).set()

    async def _drain(self, sid: str, peer_id: str, channel_name: str) -> None:
        """Long-lived per-session drainer. Waits on the wake event,
        acquires the session lock, drains the pending queue (coalescing
        whatever's in it into a single combined turn), then sleeps again.

        Runs forever until the asyncio loop shuts down.

        ``channel_name`` is the inbound's channel (fixed per session, since the
        session key encodes it); it selects the outbound channel so a voice
        turn's typing + reply go back to the box, not to matrix.
        """
        channel = self._channel_for(channel_name)
        sink = self._sink_for(channel, peer_id)
        has_pending = self._has_pending.setdefault(sid, asyncio.Event())
        while True:
            try:
                await has_pending.wait()
                async with self._session_lock(sid):
                    async with sink.typing():
                        while True:
                            batch = self._pending_inbound.pop(sid, [])
                            if not batch:
                                # Queue empty — clear the wake-event and
                                # break out of the inner loop. Outer loop
                                # will await has_pending again.
                                has_pending.clear()
                                break
                            if len(batch) > 1:
                                log.info(
                                    "[%s] coalescing %d inbound messages into one turn",
                                    self.id, len(batch),
                                )
                            turn_id = f"{sid}#{secrets.token_hex(4)}"
                            self._inflight_turn[sid] = turn_id
                            try:
                                await self._process_batch(
                                    sid, batch, turn_id, channel, sink,
                                )
                            finally:
                                self._inflight_turn.pop(sid, None)
                                self._system_emitted_turns.discard(turn_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("[%s] drainer iteration raised; continuing", self.id)

    def _take_subagent_completions(self, sid: str) -> list[dict]:
        """Take any subagent completions queued for ``sid``, as user rows.

        Called once per tool turn (via ``run_turn``'s ``drain_inbox`` hook) so
        a child that finishes mid-turn reports into the turn that spawned it,
        rather than waiting for that turn to release the session lock. Before
        this existed, the tool loop was a closed system and a completion could
        only be seen by the NEXT conversational turn — on a long chain, an hour
        late and usually useless.

        ONLY subagent completions are taken. A message from a person stays
        queued for the drainer and becomes its own conversational turn:
        mid-turn injection is for work this agent itself started and may be
        waiting on, not for interrupting it with new instructions.

        Synchronous by contract — it reads and mutates ``_pending_inbound``
        with no await in between, so the drainer (parked on this session's lock
        for the duration of the turn) cannot interleave with it.
        """
        queued = self._pending_inbound.get(sid)
        if not queued:
            return []
        taken = [m for m in queued if m.is_subagent_completion]
        if not taken:
            return []
        remaining = [m for m in queued if not m.is_subagent_completion]
        if remaining:
            self._pending_inbound[sid] = remaining
        else:
            self._pending_inbound.pop(sid, None)
        rows: list[dict] = []
        for m in taken:
            now = datetime.now(timezone.utc)
            body = f"{m.sender_name}: {m.text}" if m.sender_name else m.text
            # Same envelope a post-turn delivery would have carried, so an
            # injected completion is indistinguishable in the transcript from
            # one that arrived at a turn boundary.
            rows.append({
                "role": "user",
                "content": format_inbound_envelope(
                    channel=m.channel,
                    sender=m.sender_name or None,
                    body=body,
                    ts=now,
                    prev_ts=self._last_inbound_at.get(sid),
                    tz_name=self.cfg.tz,
                ),
            })
            self._last_inbound_at[sid] = now
        log.info(
            "[%s] %d subagent completion(s) delivered mid-turn on %s",
            self.id, len(taken), sid,
        )
        return rows

    async def _process_batch(
        self, sid: str, msgs: list[InboundMessage], turn_id: str = "",
        channel: Channel | None = None, sink: Sink | None = None,
    ) -> None:
        """Process a batch of one or more inbound messages as a single turn.

        Combines all messages into one user-role transcript row (preserving
        per-message sender prefixes for group rooms) and runs one
        ``ollama.run_turn``. The agent sees a single longer user message
        rather than multiple back-to-back turns.
        """
        # Outbound channel for this turn's reply (threaded from the drainer;
        # resolve from the batch as a fallback for any direct call).
        if channel is None:
            channel = self._channel_for(msgs[0].channel)
        if sink is None:
            sink = self._sink_for(channel, msgs[0].peer_id)

        # Cron turns are stateless: each scheduled fire is an independent event,
        # so it loads no prior history and writes no transcript of its own.
        # Without this, every cron job delivering to a given target shared one
        # rolling session and each fire answered against the *previous* cron
        # job's turn — cross-event bleed. Continuity for cron comes from
        # retrieved memory (relevance-ranked) instead, and the only durable
        # record of the interaction is the mirror into the human-facing session
        # (see _mirror_synthetic_reply). The sid still exists as a lock/drainer
        # handle so cron turns stay concurrent with the human conversation.
        stateless = msgs[0].channel == "cron"

        # Idle recap — runs at most once per session per process, on the
        # first turn we see. boot_recap_known_sessions() may have already
        # run it; if not, we run it on demand. Skipped for stateless turns
        # (no transcript to recap).
        recap_block = "" if stateless else await self._idle_recap_for(sid)

        # Combine the batch into one user-text. Each message keeps its
        # sender prefix so group-room attribution survives.
        parts: list[str] = []
        for m in msgs:
            if m.sender_name:
                parts.append(f"{m.sender_name}: {m.text}")
            else:
                parts.append(m.text)
        body = "\n\n".join(parts)
        peer_id = msgs[0].peer_id  # all batched msgs share peer_id by sid keying
        # Make (channel, peer_id) visible to tools that fire later
        # (notably subagent_spawn, which needs to know where to deliver
        # the synthetic completion message).
        self._active_inbound = (msgs[0].channel, peer_id)

        # peer_label: short human-meaningful identifier for the conversation
        # partner who triggered this turn — used in the run_turn label so
        # log lines self-identify (``[agent-1:main:user-1]``). Prefer the MXID
        # localpart when sender_id is a Matrix MXID; fall back to
        # sender_name (synthetic channels like cron/initial_prompt set
        # sender_name to a meaningful tag); ultimate fallback is the
        # channel name so labels never collapse to bare ``[agent-1:main]``.
        peer_label = _derive_peer_label(msgs[0])
        self._peer_label_by_sid[sid] = peer_label
        # Anchor log line at the top of every turn — closes the visibility
        # gap for plain-text replies, which otherwise produce no claw.*
        # output (the ollama tool-turn lines fire only when tool_calls do).
        log.info(
            "[%s:main:%s] turn starting (%d inbound)",
            self.id, peer_label, len(msgs),
        )
        turn_t0 = time.monotonic()

        # Envelope wrap: gives the model a per-turn anchor for current date,
        # day-of-week, and elapsed time since the prior turn. Without this,
        # qwen3.6:27b hallucinates dates because it has no other ground-truth
        # source for "today." Cron and initial-prompt inbounds flow through
        # this same path, so they're stamped too.
        now = datetime.now(timezone.utc)
        prev = self._last_inbound_at.get(sid)
        envelope_sender = msgs[0].sender_name if msgs[0].sender_name else None
        user_text = format_inbound_envelope(
            channel=msgs[0].channel,
            sender=envelope_sender,
            body=body,
            ts=now,
            prev_ts=prev,
            tz_name=self.cfg.tz,
        )
        self._last_inbound_at[sid] = now

        # Memory retrieval — query against the combined text.
        try:
            retrieval_block = await self.memory.retrieve_markdown(
                user_text,
                top_n=self.cfg.memory_retrieval.top_n,
                compact=self.cfg.memory_retrieval.compact,
            )
        except Exception:
            log.exception("[%s] memory retrieve failed; continuing", self.id)
            retrieval_block = ""

        user_row = {"role": "user", "content": user_text}
        if stateless:
            # No persistence, no prior history: this turn sees only its own
            # (memory-augmented) prompt. Nothing to compact — the cron sid
            # never accumulates a transcript.
            history = [user_row]
        else:
            self.transcripts.append(sid, user_row)
            rows = self.transcripts.load(sid)
            history = [as_message(r) for r in rows]
            # NB: background flush+compaction is spawned at turn-END now (see
            # the tail of this method), not here. Spawning it before run_turn
            # made its compaction-model summarize contend for GPU bandwidth
            # with this turn's own reply generation on the (batch-1,
            # bandwidth-bound) box — for zero benefit, since the reply prompt
            # is already built and the swap serializes behind this turn's
            # session lock regardless. Deferring removes the contention.
        system = self._build_system_prompt(
            recap_block, modality=msgs[0].modality
        )
        # Retrieved memory rides at the tail of the history, not in `system` —
        # see _retrieval_row for why. Spliced into the in-memory prompt only;
        # the persisted transcript keeps the user's message verbatim.
        history = self._with_retrieval(history, retrieval_block)

        # Per-conversation reasoning-trace mode (set by %thinking).
        # "full" => stream every loop iteration's reasoning live via the
        # run_turn callback; "final" => surface only the final turn's,
        # sent after run_turn returns (below); None => off. The callback
        # only sends — it never touches new_messages/transcript — so the
        # ephemerality invariant holds even though it fires mid-loop
        # (before the transcript append) rather than after.
        thinking_mode = self._thinking_mode.get(sid)

        # True once a *visible* reasoning block has actually been surfaced
        # this turn — set by the full-mode callback below (after its
        # visible-content gate) or by the final-mode post-send. Gates the
        # answer's _THINKING_ANSWER_SEP prefix so the gap is added iff
        # there is *both* a reasoning block and an answer; no spurious
        # separator when thinking is off (the user->bot speaker switch
        # already separates) or when a body-less trace was suppressed.
        # This per-turn flag is the async-safe crux: in full mode whether
        # any block was emitted isn't known until run_turn returns.
        thinking_emitted = False

        async def _emit_thinking(trace: str) -> None:
            nonlocal thinking_emitted
            # ollama fires this on .strip() truthiness; apply the stronger
            # visible-content gate here so "full" suppresses body-less
            # (zero-width/format-only) traces exactly like "final" does.
            if not sink.shows_asides or not _has_visible_content(trace):
                return
            thinking_emitted = True
            await sink.aside(_as_thinking_blockquote(trace))

        # Make the session id AND this turn's id ambient for the whole
        # turn so the bash tool tags spawned process groups with both, and
        # so spawned subagents inherit the turn id (asyncio.create_task
        # copies the context) — that's what lets %stop cancel exactly this
        # turn's cascade and kill its bash. Covers nested subagent
        # run_turns too. Reset in a finally so it's cleared even on
        # cancellation / early return.
        tok_sid = current_sid.set(sid)
        tok_turn = current_turn_id.set(turn_id)
        try:
            try:
                new_messages, final_text, final_thinking = await self.ollama.run_turn(
                    model=self.agent_cfg.primary_model,
                    history=history,
                    system=system,
                    tools=self.tools,
                    sid=sid,
                    workspace_dir=self.agent_cfg.workspace,
                    label=f"{self.id}:main:{peer_label}",
                    verbose_suffix=sid,
                    num_predict=self.num_predict,
                    max_tool_turns=self.max_tool_turns,
                    on_thinking=_emit_thinking if thinking_mode == "full" else None,
                    drain_inbox=lambda: self._take_subagent_completions(sid),
                )
            except Exception:
                log.exception("[%s] ollama.run_turn failed", self.id)
                await sink.reply("Sorry — I hit an error. Could you try again?")
                return
        finally:
            current_turn_id.reset(tok_turn)
            current_sid.reset(tok_sid)

        if not stateless:
            for m in new_messages:
                self.transcripts.append(sid, m)

        # Final-only reasoning surface (%thinking on). Sent AFTER the
        # transcript append (final_thinking is never in new_messages —
        # ephemeral by construction) and BEFORE the answer, as its own
        # message so it and the answer occupy independent chunk streams.
        # Skipped in "full" mode: the run_turn callback already emitted
        # every iteration's reasoning (incl. this final turn's), so a
        # post-send here would duplicate the last block. None => no-op.
        # _has_visible_content (not .strip()) so a body-less trace never
        # renders as a header-only block.
        if (
            sink.shows_asides
            and thinking_mode == "final"
            and _has_visible_content(final_thinking)
        ):
            thinking_emitted = True
            await sink.aside(_as_thinking_blockquote(final_thinking.strip()))
        answer = final_text.strip()
        replied = bool(answer)
        if replied:
            # One-line gap above the answer iff some bot block already
            # went out this turn ahead of it (see _THINKING_ANSWER_SEP):
            # a %thinking reasoning block, OR a 🖥️ system block a command
            # emitted while this turn was in flight. Either way they're
            # consecutive same-sender events with no speaker-switch gap.
            # `in`-test (not discard) here — the drainer's finally is the
            # single owner of removal, so a system block that lands AFTER
            # the answer doesn't wrongly gap a later turn. The prefix goes
            # ONLY on the sent body — never on `answer`, which the
            # turn-complete log measures — so "reply N chars" stays the
            # true answer length and the separator never reaches the
            # transcript (final_text alone is what was appended above).
            pre_block = thinking_emitted or turn_id in self._system_emitted_turns
            send_body = _THINKING_ANSWER_SEP + answer if pre_block else answer
            await sink.reply(send_body)
        else:
            # Backstop. run_turn substitutes a sentinel for every
            # never-legitimate exit, so reaching this means a path returned
            # empty text that nothing else accounted for. Silence is the one
            # outcome the user cannot interpret — it reads as "ignored" —
            # so say something rather than only logging NO REPLY SENT below.
            log.warning(
                "[%s] empty reply reached the send site; run_turn should "
                "have substituted a fallback", self.id,
            )
            await sink.reply(
                "[claw: I finished this turn without producing a reply. "
                "Check /var/log/claw.log for this turn.]"
            )

        # A stateless cron turn leaves no transcript of its own, so mirror the
        # trigger + delivered reply into the peer's human-facing session — the
        # only durable record, and the context the human's follow-up needs.
        # `body` is the raw trigger prompt(s) (unclobbered — the send text is
        # the separately-named `send_body`); `answer` is the pristine reply.
        if replied and stateless:
            await self._mirror_synthetic_reply(channel, peer_id, sid, body, answer)

        # Close bracket, symmetric with "turn starting". INFO = metadata
        # only (visible under normal operation); the reply snippet is
        # appended only at DEBUG (%verbose on) — same noisy-detail-behind
        # -DEBUG convention as ollama's verbose_suffix. "NO REPLY SENT"
        # flags the empty-reply failure class explicitly.
        tool_turns = sum(
            1 for m in new_messages
            if m.get("role") == "assistant" and m.get("tool_calls")
        )
        reply_part = (
            f"reply {len(answer)} chars" if replied
            else "NO REPLY SENT"
        )
        tail = ""
        if replied and log.isEnabledFor(logging.DEBUG):
            snip = " ".join(final_text.split())
            if len(snip) > 100:
                snip = snip[:100] + "…"
            tail = f': "{snip}"'
        log.info(
            "[%s:main:%s] turn complete (%d rows, %d tool turn(s), %s, %.1fs)%s",
            self.id, peer_label, len(new_messages), tool_turns,
            reply_part, time.monotonic() - turn_t0, tail,
        )

        # Background flush + compaction — spawned HERE, after the reply is
        # sent, so its compaction-model summarize never contends for GPU with
        # the reply's own generation. The snapshot is the post-turn transcript
        # (includes this turn's rows), and the predicate is fed the system
        # prompt's token cost as overhead so it fires at the true prompt size
        # (workspace files + skill catalog + retrieved memory), not
        # transcript-only. When a compaction actually starts, drop a one-line
        # 🖥️ system notice into the room so the user knows the background work
        # is underway (matrix only — a voice channel would read it aloud).
        if not stateless:
            rows_now = self.transcripts.load(sid)
            # Non-transcript overhead: the system block PLUS the retrieved
            # memory row. Retrieval now rides in the history rather than in
            # `system` (see _retrieval_row), and it is never persisted — so it
            # appears in neither `system` nor `rows_now` and has to be added
            # back explicitly, or the trigger silently undercounts the real
            # prompt by the size of the retrieval block.
            overhead_rows = [{"content": system}]
            retrieval_row = self._retrieval_row(retrieval_block)
            if retrieval_row is not None:
                overhead_rows.append(retrieval_row)
            overhead_tokens = estimate_tokens(overhead_rows)
            started = self._spawn_bg_maintenance_if_needed(
                sid, rows_now, overhead_tokens=overhead_tokens
            )
            # Only compaction is announced. A growth flush is routine, costs
            # the user nothing, and does not shorten their context — a notice
            # for it would be noise.
            if started == "compact" and msgs[0].channel == "matrix":
                await sink.aside(
                    _as_system_blockquote(
                        "Auto-compaction in progress — condensing the earlier "
                        "part of this conversation to free up context. Running "
                        "in the background."
                    )
                )

    async def _idle_recap_for(self, sid: str) -> str | None:
        """Cached per-session idle recap. Computes once on first call,
        returns the cached block on subsequent calls.
        """
        if sid in self._idle_recapped:
            return self._idle_recap_blocks.get(sid)
        block: str | None = None
        try:
            block = await maybe_idle_recap(
                cfg=self.cfg,
                ollama=self.ollama,
                transcripts=self.transcripts,
                sid=sid,
                compaction_model=self.compaction_model,
            )
        except Exception:
            log.exception("[%s] on-demand idle recap failed for %s", self.id, sid)
        self._idle_recapped.add(sid)
        if block:
            self._idle_recap_blocks[sid] = block
        return block

    # --- background flush + compaction ---------------------------------

    def _spawn_bg_maintenance_if_needed(
        self,
        sid: str,
        rows_snapshot: list[dict],
        *,
        overhead_tokens: int = 0,
    ) -> str:
        """Evaluate BOTH background gates for this session and spawn at most
        one task. Returns "compact", "flush", or "" (nothing spawned).

        Two thresholds, one decision point:

        - compaction (~mid_session_token_threshold): the prompt is getting
          big; flush first, then summarise the older slice away.
        - growth (memory_flush.periodic_growth_threshold, ~24x smaller): the
          session has accumulated enough new material to be worth persisting,
          long before anything is at risk of being compacted away.

        These used to live in different places — compaction here, growth in a
        300s maintenance timer (``periodic_flush_pass``). That timer fired
        blind to turn state, which is how a flush came to run concurrently
        with a live reply and contend with it for the GPU: exactly the failure
        d352b0e moved THIS path to turn-end to prevent, reintroduced through
        the other door. It is safe to retire the timer because transcript
        growth only ever comes from a turn appending rows, so a check at
        turn-end catches every growth event by construction.

        INVARIANT: anything that appends to a transcript outside the turn that
        owns it must call this for the affected sid, at its own turn-end. The
        only such path today is ``_mirror_synthetic_reply``.

        ``overhead_tokens`` is the estimated size of the non-transcript part
        of the prompt (system block + retrieved memory); it's added to the
        transcript-token estimate so the compaction trigger reflects the real
        prompt. It does not apply to the growth gate, which measures how much
        new *transcript* there is to remember.
        """
        existing = self._bg_compaction.get(sid)
        if existing is not None and not existing.done():
            return ""

        if will_mid_session_compact(
            self.cfg, rows_snapshot, overhead_tokens=overhead_tokens
        ):
            task = asyncio.create_task(
                self._run_bg_compaction(
                    sid, rows_snapshot, overhead_tokens=overhead_tokens,
                ),
                name=f"bg-compact-{self.id}-{sid}",
            )
            self._bg_compaction[sid] = task
            return "compact"

        if not self.cfg.memory_flush.enabled or not rows_snapshot:
            return ""
        grown = (
            estimate_tokens(rows_snapshot)
            - self._last_periodic_flush_tokens.get(sid, 0)
        )
        if grown < self.cfg.memory_flush.periodic_growth_threshold:
            return ""
        task = asyncio.create_task(
            self._run_flush_guarded(
                sid=sid,
                full_rows=list(rows_snapshot),
                reason="periodic-growth",
                mode="bg",
            ),
            name=f"growth-flush-{self.id}-{sid}",
        )
        self._bg_compaction[sid] = task
        return "flush"

    async def _run_bg_compaction(
        self,
        sid: str,
        rows_snapshot: list[dict],
        *,
        overhead_tokens: int = 0,
    ) -> None:
        """Background body. Runs the pre-compact flush then mid-session
        compaction in sequence on the compaction model. Spawned at turn-end
        (after the reply is sent), so it no longer overlaps — and no longer
        starves — the reply generation. A flush failure does not block
        compaction and vice-versa.
        """
        try:
            await self._run_flush_guarded(
                sid=sid,
                full_rows=rows_snapshot,
                reason="pre-compact",
                mode="bg",
            )
            try:
                await run_mid_session_compact_async(
                    cfg=self.cfg,
                    ollama=self.ollama,
                    transcripts=self.transcripts,
                    session_lock=self._session_lock(sid),
                    sid=sid,
                    rows_snapshot=rows_snapshot,
                    compaction_model=self.compaction_model,
                    overhead_tokens=overhead_tokens,
                )
            except Exception:
                log.exception("[%s] background compaction failed for %s", self.id, sid)
        finally:
            # Drop our reference once done so the next trigger can spawn a
            # fresh task. Other code paths only check `.done()` so the task
            # remaining in the dict transiently isn't a correctness problem.
            self._bg_compaction.pop(sid, None)

    # --- in-band admin commands ----------------------------------------

    def _command_authorized(self, sender_id: str) -> bool:
        # === AUTH SOURCE — single switch point =========================
        # Dedicated fail-closed allowlist (empty tuple = nobody). To instead
        # reuse the per-agent matrix DM allowlist, replace the next line with:
        #     allowed = self.agent_cfg.matrix.allow_from
        allowed = self.cfg.commands.allow
        # ===============================================================
        return bool(sender_id) and sender_id in allowed

    async def _cmd_reply(self, peer_id: str, text: str) -> None:
        """Send a control-plane reply, visually marked as a system
        message (🖥️ blockquote). All command-handler output goes through
        here so it's uniformly distinguishable from agent prose.

        If a turn is in flight for this room's session, tag it so its
        answer gets the same one-line gap a %thinking block earns — the
        system block and the answer are otherwise consecutive same-sender
        events. The command path is matrix-only, so the session id is
        ``session_id("matrix", peer_id)`` (peer_id is the room id), the
        same key the drainer uses. Tag before the send so the intent is
        recorded regardless of send latency; the drainer's finally is the
        sole remover.
        """
        tid = self._inflight_turn.get(session_id("matrix", peer_id))
        if tid:
            self._system_emitted_turns.add(tid)
        await self.channel.send(peer_id, _as_system_blockquote(text))

    async def _handle_command(
        self, sid: str, msg: InboundMessage, cmd: ParsedCommand,
    ) -> None:
        """Detached task: run one admin command and reply.

        Authorization is already enforced at the handle_inbound gate, so
        there is no auth/reject branch here. Each session-mutating handler
        (clear / compact) calls into code that takes the per-session lock
        itself, so this serializes behind any in-flight turn for ``sid`` —
        and never reentrantly, because handle_inbound returned before the
        drainer acquired that lock. Unknown/bare commands echo usage (only
        authorized senders ever reach this method).
        """
        peer = msg.peer_id
        prefix = self.cfg.commands.prefix
        handlers = {
            "clear": self._cmd_clear,
            "compact": self._cmd_compact,
            "verbose": self._cmd_verbose,
            "context": self._cmd_context,
            "stop": self._cmd_stop,
            "subagents": self._cmd_subagents,
            "thinking": self._cmd_thinking,
        }
        handler = handlers.get(cmd.name)
        try:
            if handler is None:
                await self._cmd_reply(peer, usage(prefix))
            else:
                await handler(sid, msg, cmd)
            # A reply sent while a turn is in flight clears the bot's
            # typing client-side, but the turn's typing heartbeat only
            # ever re-asserts True — never a transition — so the server
            # never re-broadcasts m.typing and the indicator stays gone
            # for the rest of the turn. Drop server typing here so the
            # heartbeat's next (≤6 s) re-assert is a real false→true the
            # client renders. Skip %stop: it tears the turn down and its
            # own unwind handles typing.
            if cmd.name != "stop" and self._inflight_turn.get(sid):
                await self.channel.clear_typing(peer)
        except Exception:
            log.exception(
                "[%s] command handler raised (cmd=%r room=%s)",
                self.id, cmd.name, peer,
            )

    async def _reject_extra_args(
        self, name: str, msg: InboundMessage, cmd: ParsedCommand,
    ) -> bool:
        """For no-argument commands: if any token was passed, report it
        (specific per-command usage) and return True so the caller bails.
        Incorrect parameters are surfaced, never silently ignored.
        """
        if cmd.args.strip():
            await self._cmd_reply(
                msg.peer_id,
                command_usage(name, self.cfg.commands.prefix, cmd.args),
            )
            return True
        return False

    async def _cmd_clear(
        self, sid: str, msg: InboundMessage, cmd: ParsedCommand,
    ) -> None:
        if await self._reject_extra_args("clear", msg, cmd):
            return
        rows_before = len(self.transcripts.load(sid))
        if rows_before == 0:
            await self._cmd_reply(
                msg.peer_id, "Session already empty — nothing to clear."
            )
            return
        # Long op: a sync pre-rotate flush turn on the compaction model,
        # then archive. Immediate ack + keepalived typing indicator (same
        # mechanism normal turns use) so it's visibly running; the
        # completion message lands when done.
        await self._cmd_reply(
            msg.peer_id, "Flushing memory, then clearing the session…",
        )
        async with self.channel.typing(msg.peer_id):
            archived = await self.clear_session(sid, run_final_flush=True)
        if archived:
            await self._cmd_reply(
                msg.peer_id,
                f"Session cleared — {rows_before} rows archived, memory "
                f"flushed. Starting fresh.",
            )
        else:
            await self._cmd_reply(
                msg.peer_id, "Session already empty — nothing to clear."
            )

    async def _cmd_compact(
        self, sid: str, msg: InboundMessage, cmd: ParsedCommand,
    ) -> None:
        if await self._reject_extra_args("compact", msg, cmd):
            return
        rows = self.transcripts.load(sid)
        if not rows:
            await self._cmd_reply(
                msg.peer_id, "Nothing to compact — session is empty."
            )
            return
        # Gate the (expensive) pre-compact flush on the SAME predicate the
        # compaction itself uses: if a forced compaction wouldn't swap
        # (transcript fits within the reserve/keep window), skip flush AND
        # compact and just say so — don't burn a flush turn to then no-op.
        if not will_compact(self.cfg, rows, force=True):
            await self._cmd_reply(
                msg.peer_id,
                "Nothing to compact — transcript fits within the keep "
                "window. No flush run.",
            )
            return
        # Long op: a sync pre-compact flush turn on the compaction model,
        # then the summarize+swap. Immediate ack + keepalived typing
        # indicator so it's visibly running; completion lands when done.
        await self._cmd_reply(
            msg.peer_id, "Flushing memory, then compacting…",
        )
        async with self.channel.typing(msg.peer_id):
            # Flush durable knowledge BEFORE the lossy summarize, mirroring
            # the automatic flush+compact path (_run_bg_compaction) and
            # %clear's pre-wipe flush — otherwise %compact can summarize
            # away knowledge never written to memory/. mode="sync" (not the
            # bg path's skip-if-busy "bg"): a manual command waits out any
            # in-flight flush. _run_flush_guarded swallows its own errors
            # and returns a bool; guard anyway so a flush failure never
            # blocks the compaction.
            flushed = False
            try:
                flushed = await self._run_flush_guarded(
                    sid=sid,
                    full_rows=list(rows),
                    reason="pre-compact",
                    mode="sync",
                )
            except Exception:
                log.exception(
                    "[%s] pre-compact flush raised for %s", self.id, sid
                )
            swapped = await run_mid_session_compact_async(
                cfg=self.cfg,
                ollama=self.ollama,
                transcripts=self.transcripts,
                session_lock=self._session_lock(sid),
                sid=sid,
                rows_snapshot=list(rows),
                compaction_model=self.compaction_model,
                force=True,
            )
        if swapped:
            reply = (
                "Context compacted (memory flushed first)."
                if flushed else "Context compacted."
            )
        else:
            reply = "Nothing to compact (already below the compaction split)."
        await self._cmd_reply(msg.peer_id, reply)

    async def _cmd_verbose(
        self, sid: str, msg: InboundMessage, cmd: ParsedCommand,
    ) -> None:
        arg = cmd.args.strip().lower()
        prefix = self.cfg.commands.prefix
        if arg == "":
            # No bare toggle — bare command is a status *read*: report
            # current state + how to change it.
            cur = "ON" if logsetup.verbose_enabled() else "OFF"
            await self._cmd_reply(
                msg.peer_id,
                f"Verbose logging is {cur} (process-wide). "
                f"Set with {prefix}verbose <on|off>.",
            )
            return
        if arg == "on":
            target = True
        elif arg == "off":
            target = False
        else:
            # Unknown token: reported, not silently ignored.
            await self._cmd_reply(
                msg.peer_id, command_usage("verbose", prefix, cmd.args),
            )
            return
        logsetup.set_verbose(target)
        await self._cmd_reply(
            msg.peer_id,
            f"Verbose logging {'ON' if target else 'OFF'} "
            f"(process-wide — affects all agents).",
        )

    async def _cmd_thinking(
        self, sid: str, msg: InboundMessage, cmd: ParsedCommand,
    ) -> None:
        arg = cmd.args.strip().lower()
        prefix = self.cfg.commands.prefix
        # Per-session and explicit-only: exactly one of on|off|full to
        # *change* state (no bare toggle); bare command is a status read;
        # any other token is a reported error. on/full are mutually
        # exclusive for a sid (the map holds at most one mode).
        labels = {
            None: "OFF",
            "final": "ON (final answer's reasoning)",
            "full": "ON (full — every step's reasoning)",
        }
        if arg == "":
            cur = labels[self._thinking_mode.get(sid)]
            await self._cmd_reply(
                msg.peer_id,
                f"Reasoning trace is {cur} for this conversation. "
                f"Set with {prefix}thinking <on|off|full>.",
            )
            return
        if arg == "on":
            self._thinking_mode[sid] = "final"
        elif arg == "full":
            self._thinking_mode[sid] = "full"
        elif arg == "off":
            self._thinking_mode.pop(sid, None)
        else:
            await self._cmd_reply(
                msg.peer_id, command_usage("thinking", prefix, cmd.args),
            )
            return
        await self._cmd_reply(
            msg.peer_id,
            f"Reasoning trace {labels[self._thinking_mode.get(sid)]} "
            f"for this conversation (ephemeral — not transcribed or logged).",
        )

    async def _cmd_context(
        self, sid: str, msg: InboundMessage, cmd: ParsedCommand,
    ) -> None:
        if await self._reject_extra_args("context", msg, cmd):
            return
        rows = self.transcripts.load(sid)
        # The idle-recap summary in force for this session (installed at boot
        # or on the last turn, and re-injected into every turn's system prompt
        # until the session is cleared or rotated — it is read, never popped).
        # Read the cache directly — do NOT call _idle_recap_for, which could
        # trigger an LLM summarize. None when no recap is installed.
        recap = self._idle_recap_blocks.get(sid)
        # Reproduce exactly what the next turn's system prompt will be.
        # _build_system_prompt is sync + side-effect-free. The retrieval block
        # is no longer part of it at all (it rides in the history now), so this
        # is the whole system prompt rather than an approximation of it.
        system = self._build_system_prompt(recap)
        sys_toks = estimate_tokens([{"content": system}])
        hist_toks = estimate_tokens(rows)
        start_toks = sys_toks + hist_toks
        thr = self.cfg.compaction.mid_session_token_threshold
        # Auto-compaction counts the real prompt — non-transcript overhead
        # PLUS transcript rows — so report the % against that combined figure,
        # not transcript-only. (This estimate omits the per-turn retrieval
        # block, which is unknowable until the user types, so the live trigger
        # fires a touch earlier than shown.)
        pct = round(100 * start_toks / thr) if thr else 0
        recap_note = (
            f", incl. ~{estimate_tokens([{'content': recap}]):,} recap"
            if recap else ""
        )
        # Report the OPERATION, not just the budget. The trigger above counts
        # overhead and compares against mid_session_token_threshold; the split
        # counts transcript rows only and compares against reserve_tokens. Two
        # different quantities — so a prompt sitting well below the trigger can
        # still have nothing to compact, which reads as a silent no-op unless
        # the floor is stated outright.
        reserve = self.cfg.compaction.reserve_tokens
        would_older, would_newer = compaction_preview(self.cfg, rows)
        if would_older:
            compact_note = (
                f"%compact now would summarize ~{would_older:,} and keep "
                f"~{would_newer:,} verbatim (reserve {reserve:,})."
            )
        else:
            compact_note = (
                f"%compact now would do nothing — nothing older is worth "
                f"summarizing (transcript ~{hist_toks:,} against a "
                f"{reserve:,} reserve). Either it fits inside the reserve, "
                f"or the slice past it is too small to be worth replacing "
                f"with a recap."
            )
        await self._cmd_reply(
            msg.peer_id,
            f"Context: next turn starts at ~{start_toks:,} tokens — system "
            f"~{sys_toks:,}{recap_note} + {len(rows)} transcript rows "
            f"~{hist_toks:,}, before your message / memory retrieval / tool "
            f"schema. Auto-compaction triggers on the full prompt at "
            f"{thr:,} (~{pct}% there). {compact_note}",
        )

    async def _cmd_stop(
        self, sid: str, msg: InboundMessage, cmd: ParsedCommand,
    ) -> None:
        """The just-in-case button. Every action is bounded to a single
        turn (or one subagent subtree within it) — never session-wide.

        ``%stop`` — cancel this session's in-flight conversational turn:
        cancel the per-session drainer while it's inside _process_batch
        (CancelledError is a BaseException so _process_batch's
        `except Exception` can't swallow it; it unwinds the session-lock +
        typing CMs and ends the drainer; handle_inbound lazily recreates
        it next inbound). Also cancel the *entire spawn cascade* rooted at
        this turn (matched by turn_id, inherited transitively) and suppress
        those subagents' completion so a zombie can't resurrect the
        stopped session. By default SIGKILL the turn's bash process trees
        (main + cascade); ``--soft`` (also ``-s`` / ``--keep-bash``) skips
        only the bash kill — for the rare case where an in-flight,
        non-idempotent shell command (backup / migration / build) should
        be allowed to finish even though the agent is being stopped. A
        synthetic user-role marker is appended so the next turn knows the
        instruction was operator-cancelled and must not be auto-resumed.

        ``%stop <task_id>`` — cancel just that subagent and its descendant
        subtree; the parent turn and sibling subagents are untouched. The
        target keeps its normal (cancelled) completion delivery — the
        session is alive and should learn it died, same as the
        model-facing subagent_stop. Its bash subtree is SIGKILLed unless
        ``--soft``.

        Background flush/compaction and an in-progress %clear/%compact run
        off the drainer and are never affected.
        """
        peer = msg.peer_id
        prefix = self.cfg.commands.prefix
        # Strict: at most one positional (task_id) and only the soft-flag
        # spellings. An unknown flag or a second positional is reported,
        # not silently dropped (old next()/any() quietly ignored both).
        tokens = cmd.args.split()
        soft_flags = {"--soft", "-s", "--keep-bash"}
        flags = [t for t in tokens if t.startswith("-")]
        positionals = [t for t in tokens if not t.startswith("-")]
        if any(f not in soft_flags for f in flags) or len(positionals) > 1:
            await self._cmd_reply(peer, command_usage("stop", prefix, cmd.args))
            return
        soft = bool(flags)
        target = positionals[0] if positionals else None
        # Label for the log lines below, symmetric with the
        # "[id:main:peer] turn starting/complete" anchors — a %stop used to
        # produce no claw.log output at all (only the user-facing reply).
        label = self._peer_label_by_sid.get(sid) or _derive_peer_label(msg)

        # ---- targeted: %stop <task_id> ---------------------------------
        if target is not None:
            sp = self.spawner
            ct = sp.tasks.get(target) if sp is not None else None
            if (
                ct is None
                or ct.status != "running"
                or ct.origin_session_key != sid
            ):
                await self._cmd_reply(
                    peer,
                    f"No running subagent {target!r} in this conversation "
                    f"— {prefix}subagents to list them.",
                )
                return
            cancelled = sp.cancel_subtree(target)
            killed = 0 if soft else kill_subagent_bash(set(cancelled))
            extra = len(cancelled) - 1
            log.info(
                "[%s:main:%s] %sstop cancelled subagent %s (%d in subtree, %s)",
                self.id, label, prefix, target, len(cancelled),
                "bash kept (--soft)" if soft
                else f"{killed} bash group(s) killed",
            )
            who = (
                f"subagent {target}"
                + (f" + {extra} descendant(s)" if extra > 0 else "")
            )
            bash = (
                " Its bash was left running (--soft)." if soft
                else f" {killed} bash group(s) killed."
            )
            await self._cmd_reply(
                peer,
                f"Cancelled {who}.{bash} Parent turn untouched.",
            )
            return

        # ---- whole turn: %stop ----------------------------------------
        turn_id = self._inflight_turn.get(sid)
        task = self._drainer_tasks.get(sid)
        if not turn_id or task is None or task.done():
            log.info(
                "[%s:main:%s] %sstop: no in-flight turn to cancel",
                self.id, label, prefix,
            )
            await self._cmd_reply(
                peer,
                "Nothing running — no in-flight turn to stop. (Background "
                "flush/compaction is not affected. For subagents still "
                f"running from a prior turn: {prefix}subagents to list, "
                f"{prefix}stop <task_id> to cancel one.)",
            )
            return

        await self._cmd_reply(peer, "Stopping the current turn…")
        # Cancel the turn's whole cascade FIRST, with suppression, so even
        # a subagent that finishes in the cancel window can't deliver a
        # completion back into the session we're stopping.
        cancelled = (
            self.spawner.cancel_turn(turn_id, suppress=True)
            if self.spawner is not None else []
        )
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        # Clean stop: drop anything queued/coalesced so a backlog doesn't
        # fire on the next unrelated message, and clear the wake event.
        self._pending_inbound.pop(sid, None)
        ev = self._has_pending.get(sid)
        if ev is not None:
            ev.clear()
        # Default: SIGKILL this turn's bash trees (main + cascade — all
        # tagged with turn_id). Registry still has the pgids; dereg runs
        # in the worker thread, not the cancelled await. --soft skips it.
        killed = 0 if soft else kill_turn_bash(turn_id)
        # Anchor the cancellation in claw.log, symmetric with "turn complete"
        # — the operator-visible reply alone left no server-side trace.
        log.info(
            "[%s:main:%s] turn cancelled via %sstop (turn=%s, %d subagent(s), %s)",
            self.id, label, prefix, turn_id, len(cancelled),
            "bash kept (--soft)" if soft
            else f"{killed} bash group(s) killed",
        )

        # Record the cancellation IN THE TRANSCRIPT so the next turn knows
        # the prior instruction was deliberately killed by the operator —
        # not an error, not the agent's own choice — and must not be
        # auto-resumed (else the orphaned user message is re-attempted →
        # stop/start loop). User-role + bracket matches claw's injection
        # convention. Under the session lock (the cancelled drainer
        # released it) so it serialises with any bg compaction/clear swap.
        note = (
            "[SYSTEM (out-of-band notice — not a user message, not an "
            "error): the operator deliberately cancelled the previous "
            "turn via the %stop command before it finished. That "
            "instruction is abandoned — do NOT resume or retry it; treat "
            "it as cancelled-by-user and wait for a new instruction. Any "
            "side effects from it may be partial."
        )
        if cancelled:
            note += (
                f" {len(cancelled)} spawned subagent(s) were also cancelled."
            )
        note += (
            " Shell processes from it were left running (--soft)."
            if soft else
            f" {killed} shell process group(s) were killed."
        )
        note += "]"
        async with self._session_lock(sid):
            self.transcripts.append(sid, {"role": "user", "content": note})

        subs = f" {len(cancelled)} subagent(s) cancelled." if cancelled else ""
        bash = (
            " Bash left running (--soft)." if soft
            else f" {killed} bash group(s) killed."
        )
        await self._cmd_reply(
            peer,
            "Stopped — turn cancelled, queued messages dropped."
            + subs + bash,
        )

    async def _cmd_subagents(
        self, sid: str, msg: InboundMessage, cmd: ParsedCommand,
    ) -> None:
        """Read-only discovery: list this conversation's running subagents
        (session-scoped — every turn's children — since you need to see
        them all to pick one to %stop <task_id>). Listing, not acting."""
        if await self._reject_extra_args("subagents", msg, cmd):
            return
        if self.spawner is None:
            await self._cmd_reply(
                msg.peer_id, "Subagents are not enabled for this agent."
            )
            return
        cts = self.spawner.running_for_session(sid)
        await self._cmd_reply(msg.peer_id, self.spawner.format_running(cts))

    # --- session rotate (full-context clear) ---------------------------

    async def clear_session(self, sid: str, *, run_final_flush: bool = True) -> bool:
        """Wipe one session's transcript + per-sid in-memory state.

        Holds the per-session lock so any in-flight bg compaction's
        atomic-swap will queue and then no-op (the swap rechecks row count
        against its snapshot and bails when it finds an empty/shrunk file).

        ``run_final_flush=True`` does one synchronous memory_flush against
        the current rows under the lock so anything since the last periodic
        flush still lands in memory/YYYY-MM-DD.md before the wipe. Skipped
        when the transcript is empty.

        Returns True if a transcript was archived, False if nothing to do.
        """
        async with self._session_lock(sid):
            self._pending_inbound.pop(sid, None)
            has_pending = self._has_pending.get(sid)
            if has_pending is not None:
                has_pending.clear()

            rows = self.transcripts.load(sid)
            if not rows:
                self._idle_recapped.discard(sid)
                self._idle_recap_blocks.pop(sid, None)
                self._last_periodic_flush_tokens.pop(sid, None)
                self._last_flushed_row_count.pop(sid, None)
                self._flush_locks.pop(sid, None)
                return False

            if run_final_flush and self.cfg.memory_flush.enabled:
                await self._run_flush_guarded(
                    sid=sid,
                    full_rows=rows,
                    reason="pre-rotate",
                    mode="sync",
                )

            ts = datetime.now(timezone.utc).isoformat().replace(":", "-")
            archived = self.transcripts.archive(sid, f"reset-{ts}")
            self._idle_recapped.discard(sid)
            self._idle_recap_blocks.pop(sid, None)
            self._last_periodic_flush_tokens.pop(sid, None)
            self._last_flushed_row_count.pop(sid, None)
            self._flush_locks.pop(sid, None)
            log.info(
                "[%s] session %s rotated (%d rows -> %s)",
                self.id, sid, len(rows), archived.name if archived else "no-archive",
            )
            return archived is not None

    def sweep_spool_tree(self, *, grace_seconds: int = 86400) -> None:
        """Sweep <workspace>/.tool-results/*/ of files older than the grace
        window. Workspace-scoped, not session-scoped — spool files are
        scratch owned by the workspace, not by any one session, and the
        grace window makes timing independent of session lifecycle.

        Removes files older than the cutoff, then prunes any subdirs that
        end up empty. Subagent dirs (``subagent-<child_id>``) are walked the
        same as real-session dirs — they need no special-casing because
        their contents follow the same age semantics.
        """
        root = self.agent_cfg.workspace / ".tool-results"
        if not root.is_dir():
            return
        cutoff = time.time() - grace_seconds
        total_removed = 0
        try:
            for spool_dir in root.iterdir():
                if not spool_dir.is_dir():
                    continue
                try:
                    for f in spool_dir.iterdir():
                        if f.is_file() and f.stat().st_mtime < cutoff:
                            f.unlink()
                            total_removed += 1
                    if not any(spool_dir.iterdir()):
                        spool_dir.rmdir()
                except OSError:
                    log.exception("[%s] failed to sweep spool dir %s", self.id, spool_dir)
            if total_removed:
                log.info(
                    "[%s] swept %d stale spool file(s) from %s",
                    self.id, total_removed, root,
                )
        except OSError:
            log.exception("[%s] failed to walk spool tree %s", self.id, root)

    async def clear_all_sessions(
        self,
        *,
        run_final_flush: bool = True,
        dispatch_initial_prompt: bool = True,
    ) -> int:
        """Rotate every active (non-archived) session for this agent.

        After all sessions are wiped, optionally dispatches
        ``.initial_prompt.md`` if present, so an agent that wrote itself a
        kickoff note before rotating gets a fresh opening turn (same hook
        the boot path uses, but no gateway restart needed).

        Returns the number of sessions archived.
        """
        td = self.transcripts.dir
        if not td.is_dir():
            return 0
        rotated = 0
        for entry in sorted(os.listdir(td)):
            if not entry.endswith(".jsonl"):
                continue
            if is_archived_transcript(entry):
                continue
            sid = entry[: -len(".jsonl")]
            try:
                if await self.clear_session(sid, run_final_flush=run_final_flush):
                    rotated += 1
            except Exception:
                log.exception("[%s] clear_session raised for %s", self.id, sid)
        # Sweep workspace-wide spool scratch in one pass after all rotates.
        # 24 h grace covers in-flight tool calls; subagent-* dirs are picked
        # up alongside real-session dirs.
        try:
            self.sweep_spool_tree(grace_seconds=86400)
        except Exception:
            log.exception("[%s] sweep_spool_tree raised", self.id)
        if dispatch_initial_prompt:
            try:
                await maybe_dispatch_initial_prompt(self)
            except Exception:
                log.exception("[%s] post-rotate initial_prompt dispatch raised", self.id)
        return rotated
