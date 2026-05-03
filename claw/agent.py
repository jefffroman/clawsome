"""Per-agent orchestrator. Wires channel inbound -> ollama + tools + memory.

One ``Agent`` per ``AgentConfig`` block in claw.yaml; per-peer state lives
in the ``TranscriptStore`` keyed by ``session_id(channel, peer_id)``.

Each inbound message:

1. Run on-demand idle-recap (once per session per process) if not already done.
2. Memory retrieval.
3. Append user turn to transcript.
4. **If transcript triggers flush or compaction predicates, spawn a single
   background task that runs flush then mid-session compaction off the
   critical path.** The user's reply is NOT delayed.
5. ``ollama.run_turn`` against the un-compacted history (the same-turn cost
   of bigger context is the explicit tradeoff for snappier UX).
6. Append result turns. Send final assistant text back to the channel.

Background compaction snapshots the rows-list at spawn time, runs
flush + summarize against that snapshot, then takes the per-session lock
and atomically swaps in a recap turn for the snapshot's older portion —
preserving any user/assistant turns that arrived during the work.

A separate ``periodic_flush_pass`` is invoked from the maintenance loop
(paired with reindex) to flush sessions that have grown by
``memory_flush.periodic_growth_threshold`` tokens since the last flush.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone

from claw.channel.base import Channel, InboundMessage
from claw.channel.envelope import format_inbound_envelope
from claw.compaction import (
    maybe_idle_recap,
    run_mid_session_compact_async,
    will_mid_session_compact,
)
from claw.config import AgentConfig, Config
from claw.memory import MemoryIndex
from claw.memory_flush import run_memory_flush, will_pre_compact_flush
from claw.ollama import OllamaClient
from claw.tools.base import Tool
from claw.tools.subagent import SubagentSpawner, build_subagent_tool
from claw.transcript import (
    TranscriptStore,
    as_message,
    estimate_tokens,
    is_archived_transcript,
    session_id,
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
    ) -> None:
        self.cfg = cfg
        self.agent_cfg = agent_cfg
        self.ollama = ollama
        self.memory = memory
        # Copy tools so we can mutate without mutating the caller's dict.
        self.tools = dict(tools)
        self.transcripts = transcripts
        self.channel = channel
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

        # Per-session state.
        self._session_locks: dict[str, asyncio.Lock] = {}
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
        # At most one background flush+compact task per session at a time.
        self._bg_compaction: dict[str, asyncio.Task] = {}
        # Token count at last periodic flush per session, for delta-trigger.
        self._last_periodic_flush_tokens: dict[str, int] = {}
        # Per-session timestamp of the previous _process_batch invocation.
        # Powers the envelope's elapsed-time delta. Cleared across restarts —
        # the first inbound after a restart simply has no `+elapsed` suffix.
        self._last_inbound_at: dict[str, datetime] = {}

        self._workspace_block: str | None = None
        # Boot-time recap state, keyed by session id.
        self._idle_recapped: set[str] = set()
        self._idle_recap_blocks: dict[str, str] = {}

        # Always expose memory_search — read-only retrieval over this agent's
        # MemoryIndex. Subagents inherit it via their parent's tool registry.
        self.tools["memory_search"] = build_memory_search_tool(memory)

        # Expose spawn_subagent only if I have remaining budget. Per-agent
        # gate; the global cfg.subagents has no max_spawn_depth field.
        if spawner is not None and self.remaining_spawn_budget > 0:
            self.tools["spawn_subagent"] = build_subagent_tool(self, spawner, depth)
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
            self.tools["cron_add"] = build_cron_add_tool(self.id, job_runner, cfg.cron.default_deliver_to)
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

    def _session_lock(self, sid: str) -> asyncio.Lock:
        return self._session_locks.setdefault(sid, asyncio.Lock())

    def _workspace_system_block(self) -> str:
        if self._workspace_block is None:
            self._workspace_block = render_extra_paths(
                self.agent_cfg.workspace, self.agent_cfg.extra_paths,
            )
        return self._workspace_block

    def _build_system_prompt(
        self,
        retrieval_block: str,
        recap_block: str | None = None,
    ) -> str:
        parts: list[str] = []
        ws = self._workspace_system_block()
        if ws:
            parts.append(ws)
        if self.skill_catalog:
            parts.append(self.skill_catalog)
        if recap_block:
            parts.append(recap_block)
        if retrieval_block:
            parts.append(f"<retrieved_memory>\n{retrieval_block}\n</retrieved_memory>")
        return "\n\n".join(parts)

    # --- subagent fork --------------------------------------------------

    def fork(self, persona: str) -> "Agent":
        """Return a transient child agent for one-shot subagent execution."""
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
        )
        base_tools = {
            k: v for k, v in self.tools.items()
            if k != "spawn_subagent" and not k.startswith("cron_")
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
            transcripts=self.transcripts,
            channel=self.channel,
            skill_catalog=self.skill_catalog,
            spawner=self.spawner,
            depth=self.depth + 1,
            remaining_spawn_budget=child_budget,
            allowed_spawn_personas=persona_cfg.can_spawn,
        )

    async def run_one_shot(self, prompt: str) -> str:
        """Single-turn run for subagents. No persistence, no compaction."""
        try:
            retrieval_block = await self.memory.retrieve_markdown(
                prompt,
                top_n=self.cfg.memory_retrieval.top_n,
                compact=self.cfg.memory_retrieval.compact,
            )
        except Exception:
            log.exception("[%s] subagent memory retrieve failed", self.id)
            retrieval_block = ""

        history = [{"role": "user", "content": prompt}]
        system = self._build_system_prompt(retrieval_block)
        try:
            _new_messages, final_text = await self.ollama.run_turn(
                model=self.agent_cfg.primary_model,
                history=history,
                system=system,
                tools=self.tools,
                # Subagent one-shots discard new_messages, but the model
                # still sees full tool results in-loop and the spool side
                # effect is bounded (one shot, <max_tool_turns calls).
                sid=f"subagent-{self.id}",
                workspace_dir=self.agent_cfg.workspace,
                num_predict=self.num_predict,
            )
        except Exception:
            log.exception("[%s] subagent run_turn failed", self.id)
            return f"error: subagent {self.id} run_turn failed"
        return final_text or ""

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

    async def handle_inbound(self, msg: InboundMessage) -> None:
        """Enqueue the message and signal the per-session drainer. Returns
        in microseconds so matrix-nio's sync_forever can immediately fire
        the next callback (and itself keep polling Synapse for newer
        events). This is what makes coalescing actually work — without
        the decoupling, nio serializes callbacks and msg N+1 doesn't reach
        the queue until msg N's _process_batch has already been popped.
        """
        sid = session_id(msg.channel, msg.peer_id)
        self._pending_inbound.setdefault(sid, []).append(msg)
        # Lazy-create the drainer for this session on first inbound.
        if sid not in self._drainer_tasks or self._drainer_tasks[sid].done():
            self._drainer_tasks[sid] = asyncio.create_task(
                self._drain(sid, msg.peer_id),
                name=f"drain-{self.id}-{sid}",
            )
        # Wake the drainer.
        self._has_pending.setdefault(sid, asyncio.Event()).set()

    async def _drain(self, sid: str, peer_id: str) -> None:
        """Long-lived per-session drainer. Waits on the wake event,
        acquires the session lock, drains the pending queue (coalescing
        whatever's in it into a single combined turn), then sleeps again.

        Runs forever until the asyncio loop shuts down.
        """
        has_pending = self._has_pending.setdefault(sid, asyncio.Event())
        while True:
            try:
                await has_pending.wait()
                async with self._session_lock(sid):
                    async with self.channel.typing(peer_id):
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
                            await self._process_batch(sid, batch)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("[%s] drainer iteration raised; continuing", self.id)

    async def _process_batch(self, sid: str, msgs: list[InboundMessage]) -> None:
        """Process a batch of one or more inbound messages as a single turn.

        Combines all messages into one user-role transcript row (preserving
        per-message sender prefixes for group rooms) and runs one
        ``ollama.run_turn``. The agent sees a single longer user message
        rather than multiple back-to-back turns.
        """
        # Idle recap — runs at most once per session per process, on the
        # first turn we see. boot_recap_known_sessions() may have already
        # run it; if not, we run it on demand.
        recap_block = await self._idle_recap_for(sid)

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

        self.transcripts.append(sid, {"role": "user", "content": user_text})

        rows = self.transcripts.load(sid)

        # If predicates trip, spawn a background flush+compact task. Doesn't
        # block the user's reply — task runs concurrently with run_turn and
        # subsequent turns. The atomic-swap pattern preserves any rows the
        # agent appends during the work.
        self._spawn_bg_compaction_if_needed(sid, list(rows))

        history = [as_message(r) for r in rows]
        system = self._build_system_prompt(retrieval_block, recap_block)

        try:
            new_messages, final_text = await self.ollama.run_turn(
                model=self.agent_cfg.primary_model,
                history=history,
                system=system,
                tools=self.tools,
                sid=sid,
                workspace_dir=self.agent_cfg.workspace,
                num_predict=self.num_predict,
            )
        except Exception:
            log.exception("[%s] ollama.run_turn failed", self.id)
            await self.channel.send(peer_id, "Sorry — I hit an error. Could you try again?")
            return

        for m in new_messages:
            self.transcripts.append(sid, m)

        if final_text.strip():
            await self.channel.send(peer_id, final_text.strip())

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

    def _spawn_bg_compaction_if_needed(
        self,
        sid: str,
        rows_snapshot: list[dict],
    ) -> bool:
        """If a flush or compaction predicate trips and no task is currently
        in flight for this session, spawn one. Returns True if a task was
        spawned, False otherwise.
        """
        existing = self._bg_compaction.get(sid)
        if existing is not None and not existing.done():
            return False

        do_flush = will_pre_compact_flush(self.cfg, self.transcripts, sid, rows_snapshot)
        do_compact = will_mid_session_compact(self.cfg, rows_snapshot)
        if not (do_flush or do_compact):
            return False

        task = asyncio.create_task(
            self._run_bg_compaction(sid, rows_snapshot, do_flush, do_compact),
            name=f"bg-compact-{self.id}-{sid}",
        )
        self._bg_compaction[sid] = task
        return True

    async def _run_bg_compaction(
        self,
        sid: str,
        rows_snapshot: list[dict],
        do_flush: bool,
        do_compact: bool,
    ) -> None:
        """Background body. Runs flush (if needed) then compaction. Each
        phase is independent so a flush failure doesn't block compaction
        and vice-versa.
        """
        try:
            if do_flush:
                try:
                    await asyncio.wait_for(
                        run_memory_flush(
                            ollama=self.ollama,
                            sid=sid,
                            workspace_dir=self.agent_cfg.workspace,
                            rows=rows_snapshot,
                            primary_model=self.compaction_model,
                            tools=self.tools,
                            workspace_system_block=self._workspace_system_block(),
                            reason="pre-compact",
                            tz_name=self.cfg.tz,
                        ),
                        timeout=self.cfg.memory_flush.turn_timeout_s,
                    )
                    self._last_periodic_flush_tokens[sid] = estimate_tokens(rows_snapshot)
                except asyncio.TimeoutError:
                    log.warning(
                        "[%s] background flush timed out after %.0fs for %s — proceeding to compaction",
                        self.id, self.cfg.memory_flush.turn_timeout_s, sid,
                    )
                except Exception:
                    log.exception("[%s] background flush failed for %s", self.id, sid)

            if do_compact:
                try:
                    await run_mid_session_compact_async(
                        cfg=self.cfg,
                        ollama=self.ollama,
                        transcripts=self.transcripts,
                        session_lock=self._session_lock(sid),
                        sid=sid,
                        rows_snapshot=rows_snapshot,
                        compaction_model=self.compaction_model,
                    )
                except Exception:
                    log.exception("[%s] background compaction failed for %s", self.id, sid)
        finally:
            # Drop our reference once done so the next trigger can spawn a
            # fresh task. Other code paths only check `.done()` so the task
            # remaining in the dict transiently isn't a correctness problem.
            self._bg_compaction.pop(sid, None)

    # --- periodic flush (called from maintenance loop) ------------------

    def periodic_flush_pass(self) -> list[asyncio.Task]:
        """Walk active session transcripts; spawn a background flush per
        session whose token count has grown by
        ``memory_flush.periodic_growth_threshold`` since the last flush.

        Returns the list of spawned tasks so the caller (maintenance loop)
        can ``asyncio.gather`` them before reindex if it wants the fresh
        memory file content captured in the same tick.
        """
        tasks: list[asyncio.Task] = []
        if not self.cfg.memory_flush.enabled:
            return tasks
        td = self.transcripts.dir
        if not td.is_dir():
            return tasks
        delta_threshold = self.cfg.memory_flush.periodic_growth_threshold
        for entry in os.listdir(td):
            if not entry.endswith(".jsonl"):
                continue
            if is_archived_transcript(entry):
                continue
            sid = entry[: -len(".jsonl")]
            existing = self._bg_compaction.get(sid)
            if existing is not None and not existing.done():
                # A pre-compact flush is already running for this session;
                # the periodic flush would duplicate work.
                continue
            rows = self.transcripts.load(sid)
            if not rows:
                continue
            current_tokens = estimate_tokens(rows)
            last = self._last_periodic_flush_tokens.get(sid, 0)
            if current_tokens - last < delta_threshold:
                continue
            task = asyncio.create_task(
                self._run_periodic_flush(sid, list(rows), current_tokens),
                name=f"periodic-flush-{self.id}-{sid}",
            )
            tasks.append(task)
        return tasks

    async def _run_periodic_flush(
        self,
        sid: str,
        rows_snapshot: list[dict],
        observed_tokens: int,
    ) -> None:
        try:
            ok = await asyncio.wait_for(
                run_memory_flush(
                    ollama=self.ollama,
                    sid=sid,
                    workspace_dir=self.agent_cfg.workspace,
                    rows=rows_snapshot,
                    primary_model=self.compaction_model,
                    tools=self.tools,
                    workspace_system_block=self._workspace_system_block(),
                    reason="periodic-growth",
                    tz_name=self.cfg.tz,
                ),
                timeout=self.cfg.memory_flush.turn_timeout_s,
            )
            if ok:
                self._last_periodic_flush_tokens[sid] = observed_tokens
        except asyncio.TimeoutError:
            log.warning(
                "[%s] periodic flush timed out after %.0fs for %s",
                self.id, self.cfg.memory_flush.turn_timeout_s, sid,
            )
        except Exception:
            log.exception("[%s] periodic flush task raised for %s", self.id, sid)

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
                return False

            if run_final_flush and self.cfg.memory_flush.enabled:
                try:
                    await asyncio.wait_for(
                        run_memory_flush(
                            ollama=self.ollama,
                            sid=sid,
                            workspace_dir=self.agent_cfg.workspace,
                            rows=rows,
                            primary_model=self.compaction_model,
                            tools=self.tools,
                            workspace_system_block=self._workspace_system_block(),
                            reason="pre-rotate",
                            tz_name=self.cfg.tz,
                        ),
                        timeout=self.cfg.memory_flush.turn_timeout_s,
                    )
                except asyncio.TimeoutError:
                    log.warning(
                        "[%s] pre-rotate flush timed out after %.0fs for %s — rotating anyway",
                        self.id, self.cfg.memory_flush.turn_timeout_s, sid,
                    )
                except Exception:
                    log.exception("[%s] pre-rotate flush raised for %s — rotating anyway", self.id, sid)

            ts = datetime.now(timezone.utc).isoformat().replace(":", "-")
            archived = self.transcripts.archive(sid, f"reset-{ts}")
            self._idle_recapped.discard(sid)
            self._idle_recap_blocks.pop(sid, None)
            self._last_periodic_flush_tokens.pop(sid, None)
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
