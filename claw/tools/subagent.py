"""Subagent spawning — async persona-based agent forks.

Personas are configured in ``claw.yaml`` under ``subagents.personas:``
(case-insensitive lookup). Each spawn creates a transient ``Agent``
sharing the parent's workspace, tools (minus ``subagent_spawn`` and
``cron_*``), and memory. Transcripts are NOT persisted — a subagent's
job is one-shot.

**Async semantics.** ``subagent_spawn`` returns immediately with a
``task_id``. The child runs as a detached ``asyncio.Task`` held by
the ``SubagentSpawner`` registry. On completion, the spawner fires a
synthetic ``InboundMessage(channel=<origin>, peer_id=<origin>,
sender_name="subagent:<persona>:<task_id>")`` into the parent's
drainer — the parent's next turn carries the result in context.
``subagent_status(task_id)`` exists for explicit polling.

Spawn permission is per-agent: each agent carries a
``remaining_spawn_budget`` (top-level agents seed it from
``AgentConfig.max_spawn_depth``; forks compute
``min(parent_remaining - 1, persona.max_spawn_depth)``). A hardcoded
``ABSOLUTE_MAX_CHAIN_DEPTH`` is the runtime safety net.

Concurrency: the global ``max_concurrent`` semaphore is held over the
detached task's lifetime; per-parent ``children_by_parent`` is bumped
in ``spawn_async`` (so queued tasks count against the limit too) and
released in the task's ``finally``.

The gateway holds a single ``SubagentSpawner`` shared across all
agents. Its registry is in-memory only — gateway restart kills any
in-flight subagents and forgets completed results. v1 deliberately
does not persist this.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from claw.channel.base import InboundMessage
from claw.config import SubagentsConfig
from claw.runctx import current_sid, current_task_id, current_turn_id
from claw.tools.base import Tool

if TYPE_CHECKING:
    from claw.agent import Agent

log = logging.getLogger("claw.tools.subagent")

# Absolute chain-depth ceiling. Per-agent budgets handle the common case;
# this is a runtime safety so a buggy config can't cause runaway recursion.
ABSOLUTE_MAX_CHAIN_DEPTH = 5

# Soft cap on registry size. When exceeded, oldest completed/failed
# entries are evicted; running entries are never evicted.
_MAX_REGISTRY_SIZE = 100


@dataclass
class ChildTask:
    id: str
    parent_id: str
    persona: str
    prompt: str
    origin_channel: str
    origin_peer_id: str
    # The sid of the turn that spawned this child. Carried explicitly (not
    # rebuilt from channel+peer_id) so completions re-enter the right session
    # even when that session is shared, e.g. voice's "home".
    origin_session_key: str
    started_at: datetime
    status: str = "running"  # running | completed | failed | cancelled
    completed_at: datetime | None = None
    result: str | None = None
    aio_task: asyncio.Task | None = field(default=None, repr=False)
    # Turn that rooted this spawn (inherited transitively through the
    # cascade) — %stop cancels by this. Empty if spawned outside a turn.
    spawn_turn_id: str = ""
    # The subagent that spawned this one ("" if spawned by the top-level
    # agent) — lets a targeted %stop <task_id> cancel the whole subtree.
    parent_task_id: str = ""
    # Set by %stop on the whole-turn path: skip _deliver_completion so a
    # stopped session is never resurrected by its own zombie subagents.
    suppress_delivery: bool = False


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _allocate_task_id(persona: str, taken: set[str]) -> str:
    while True:
        tid = f"{persona}-{secrets.token_hex(4)}"
        if tid not in taken:
            return tid


def _format_elapsed(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    m = s // 60
    if m < 60:
        return f"{m}m{s % 60}s"
    h = m // 60
    return f"{h}h{m % 60}m"


class SubagentSpawner:
    def __init__(self, cfg: SubagentsConfig) -> None:
        self.cfg = cfg
        self.semaphore = asyncio.Semaphore(cfg.max_concurrent)
        # parent_id -> count of currently-pending-or-running children
        self.children_by_parent: dict[str, int] = {}
        # task_id -> ChildTask. Kept after completion so subagent_status
        # can still report results; trimmed by _gc_registry().
        self.tasks: dict[str, ChildTask] = {}

    # --- spawn -----------------------------------------------------------

    def spawn_async(
        self,
        parent: "Agent",
        persona: str,
        prompt: str,
        depth: int,
        origin_channel: str,
        origin_peer_id: str,
    ) -> str:
        """Validate, register, kick off a detached task, return a status
        string. Does NOT await the child run.
        """
        if depth > ABSOLUTE_MAX_CHAIN_DEPTH:
            return (
                f"error: subagent chain hit absolute safety ceiling "
                f"(depth={depth}, max={ABSOLUTE_MAX_CHAIN_DEPTH})"
            )

        persona_key = persona.lower()
        persona_cfg = self.cfg.personas.get(persona_key)
        if persona_cfg is None:
            available = ", ".join(sorted(self.cfg.personas.keys())) or "(none configured)"
            return f"error: unknown persona {persona!r}; available: {available}"

        # Per-parent allowlist — defense in depth alongside the tool's
        # persona schema, which already lists only allowed entries.
        allowed = parent.allowed_spawn_personas
        if allowed is not None and persona_key not in allowed:
            allowed_str = ", ".join(sorted(allowed)) or "(none)"
            return (
                f"error: parent {parent.id!r} is not allowed to spawn "
                f"persona {persona_key!r}; allowed: {allowed_str}"
            )

        count = self.children_by_parent.get(parent.id, 0)
        if count >= self.cfg.max_children_per_agent:
            return (
                f"error: parent {parent.id} has hit max_children_per_agent="
                f"{self.cfg.max_children_per_agent} pending+running children"
            )

        # Reserve the slot now so concurrent spawn calls can't race past
        # max_children_per_agent while their tasks queue at the semaphore.
        self.children_by_parent[parent.id] = count + 1

        task_id = _allocate_task_id(persona_key, set(self.tasks.keys()))
        ct = ChildTask(
            id=task_id,
            parent_id=parent.id,
            persona=persona_key,
            prompt=prompt,
            origin_channel=origin_channel,
            origin_peer_id=origin_peer_id,
            origin_session_key=current_sid.get(),
            started_at=_now(),
            spawn_turn_id=current_turn_id.get(),
            parent_task_id=current_task_id.get(),
        )
        self.tasks[task_id] = ct
        ct.aio_task = asyncio.create_task(
            self._run_subagent(parent, ct),
            name=f"subagent-{task_id}",
        )
        log.info(
            "[%s] spawning subagent task_id=%s persona=%s depth=%d model=%s",
            parent.id, task_id, persona_key, depth, persona_cfg.model,
        )
        # Soft hint about async semantics in the tool result so the model
        # sees it on the same turn that emitted the spawn. Earlier
        # iterations carried a strong "end your turn now" directive, but
        # that caused over-correction — empty-reply turns after unrelated
        # long tool sequences as the directive bled out of context. The
        # softer phrasing relies on the model's own judgement; the
        # ollama.py empty-reply recovery branch catches any stragglers.
        return (
            f"started: task_id={task_id} persona={persona_key} "
            "(running in background). This tool returned a task id "
            "immediately; you'll get a prompt when the task completes. "
            f"(subagent_status with task_id={task_id} is available for "
            "polling.)"
        )

    async def _run_subagent(self, parent: "Agent", ct: ChildTask) -> None:
        """Detached worker: holds the global semaphore, runs the child's
        single-turn inference, stores the result on the ChildTask, and
        fires the synthetic completion message.
        """
        # Tag this subagent's context so its bash (and any deeper spawn it
        # makes) is attributable to ct.id; a grandchild's own _run_subagent
        # overrides this for its subtree. Reset in the outer finally.
        tok_task = current_task_id.set(ct.id)
        try:
            try:
                async with self.semaphore:
                    child = parent.fork(ct.persona, task_id=ct.id)
                    result_text = await child.run_one_shot(ct.prompt)
                    ct.status = "completed"
                    ct.result = result_text
            except asyncio.CancelledError:
                log.info(
                    "[%s] subagent task_id=%s persona=%s cancelled",
                    parent.id, ct.id, ct.persona,
                )
                ct.status = "cancelled"
                ct.result = "(cancelled)"
            except Exception as e:
                log.exception(
                    "[%s] subagent task_id=%s persona=%s raised",
                    parent.id, ct.id, ct.persona,
                )
                ct.status = "failed"
                ct.result = f"error: subagent failed: {e}"
            ct.completed_at = _now()
            if ct.suppress_delivery:
                # Whole-turn %stop killed this cascade: do NOT inbound a
                # completion, or the stopped session would be resurrected
                # by its own zombie. Status is set; %subagents still shows.
                log.info(
                    "[%s] subagent task_id=%s completion suppressed "
                    "(session stopped)", parent.id, ct.id,
                )
            else:
                # Deliver outside the semaphore for all exit modes
                # (completed/failed/cancelled) — uniform notification
                # shape regardless of how the run ended.
                try:
                    await self._deliver_completion(parent, ct)
                except asyncio.CancelledError:
                    # Loop is shutting down (gateway restart). Status is
                    # already set on ct; subagent_status will still report.
                    pass
            self._gc_registry()
        finally:
            current_task_id.reset(tok_task)
            self.children_by_parent[parent.id] = max(
                0, self.children_by_parent.get(parent.id, 1) - 1,
            )

    async def _deliver_completion(self, parent: "Agent", ct: ChildTask) -> None:
        """Fire a synthetic InboundMessage on the origin session. Channel
        + peer_id match the original session so the parent's transcript
        carries the completion in the same conversational thread.
        """
        elapsed = ""
        if ct.completed_at is not None:
            elapsed = f" elapsed={_format_elapsed((ct.completed_at - ct.started_at).total_seconds())}"
        body_lines = [
            f"task_id={ct.id} persona={ct.persona} status={ct.status}{elapsed}",
            "",
            "Original prompt:",
            ct.prompt,
            "",
            "Result:",
            ct.result or "(no output)",
        ]
        msg = InboundMessage(
            peer_id=ct.origin_peer_id,
            sender_name=f"subagent:{ct.persona}:{ct.id}",
            text="\n".join(body_lines),
            channel=ct.origin_channel,
            # Re-enter the spawning turn's session explicitly — channel+peer_id
            # alone can't rebuild a shared key like voice's "home".
            session_key=ct.origin_session_key,
            # peer_label derivation in agent._derive_peer_label uses
            # sender_id when present; setting the task_id here gives the
            # parent's resulting run_turn label a useful tag like
            # ``[agent-1:main:persona-3-a1b2c3d4]`` instead of the
            # ``subagent:persona-3:persona-3-a1b2c3d4`` sender_name fallback.
            sender_id=ct.id,
        )
        try:
            await parent.handle_inbound(msg)
        except Exception:
            log.exception(
                "[%s] subagent completion delivery raised for task_id=%s",
                parent.id, ct.id,
            )

    def _gc_registry(self) -> None:
        """Evict oldest completed/failed entries when the registry grows
        past the soft cap. Running entries are never evicted.
        """
        if len(self.tasks) <= _MAX_REGISTRY_SIZE:
            return
        completed = [
            ct for ct in self.tasks.values()
            if ct.status != "running" and ct.completed_at is not None
        ]
        completed.sort(key=lambda c: c.completed_at)  # type: ignore[arg-type,return-value]
        excess = len(self.tasks) - _MAX_REGISTRY_SIZE
        for ct in completed[:excess]:
            self.tasks.pop(ct.id, None)

    # --- operator %stop support -----------------------------------------

    def running_for_session(self, sid: str) -> list["ChildTask"]:
        """Running subagents whose origin session is ``sid``, newest first.
        Session-scoped (every turn's children) — for %subagents discovery.
        """
        out = [
            ct for ct in self.tasks.values()
            if ct.status == "running"
            and ct.origin_session_key == sid
        ]
        out.sort(key=lambda c: c.started_at, reverse=True)
        return out

    def cancel_turn(self, turn_id: str, *, suppress: bool) -> list[str]:
        """Cancel every running subagent whose ``spawn_turn_id == turn_id``
        — i.e. the entire cascade rooted at one turn, at any depth (the
        turn id is inherited transitively). ``suppress`` sets
        ``suppress_delivery`` first so a stopped session is not resurrected
        by these children's completions. Returns the cancelled task ids.
        """
        if not turn_id:
            return []
        hit: list[str] = []
        for ct in list(self.tasks.values()):
            if ct.status != "running" or ct.spawn_turn_id != turn_id:
                continue
            ct.suppress_delivery = suppress
            if ct.aio_task is not None and not ct.aio_task.done():
                ct.aio_task.cancel()
            hit.append(ct.id)
        return hit

    def cancel_subtree(self, task_id: str) -> list[str]:
        """Cancel ``task_id`` and its transitive descendants (children via
        ``parent_task_id``). The target keeps normal completion delivery
        (the session is alive and should learn it was killed, same as the
        model-facing subagent_stop); collateral descendants are suppressed
        so they don't spam the session. Returns the cancelled task ids.
        """
        target = self.tasks.get(task_id)
        if target is None:
            return []
        # BFS the parent_task_id forest from the target.
        subtree = {task_id}
        frontier = [task_id]
        while frontier:
            parent = frontier.pop()
            for ct in self.tasks.values():
                if ct.parent_task_id == parent and ct.id not in subtree:
                    subtree.add(ct.id)
                    frontier.append(ct.id)
        hit: list[str] = []
        for tid in subtree:
            ct = self.tasks.get(tid)
            if ct is None or ct.status != "running":
                continue
            ct.suppress_delivery = tid != task_id  # deliver only the target
            if ct.aio_task is not None and not ct.aio_task.done():
                ct.aio_task.cancel()
            hit.append(tid)
        return hit

    def format_running(self, cts: list["ChildTask"]) -> str:
        if not cts:
            return "No subagents running for this session."
        lines = [f"{len(cts)} subagent(s) running:"]
        for ct in cts:
            elapsed = _format_elapsed((_now() - ct.started_at).total_seconds())
            lines.append(
                f"  {ct.id}  persona={ct.persona}  elapsed={elapsed}"
            )
        return "\n".join(lines)

    # --- status (read-only) ---------------------------------------------

    def format_status(self, ct: ChildTask) -> str:
        if ct.status == "running":
            elapsed_s = (_now() - ct.started_at).total_seconds()
            return (
                f"task_id={ct.id} persona={ct.persona} status=running "
                f"elapsed={_format_elapsed(elapsed_s)}"
            )
        elapsed = ""
        if ct.completed_at is not None:
            elapsed = f" elapsed={_format_elapsed((ct.completed_at - ct.started_at).total_seconds())}"
        return (
            f"task_id={ct.id} persona={ct.persona} status={ct.status}{elapsed}\n"
            f"\nResult:\n{ct.result or '(no output)'}"
        )


def build_subagent_spawn_tool(
    parent: "Agent",
    spawner: SubagentSpawner,
    depth: int,
) -> Tool:
    """Build a ``subagent_spawn`` Tool bound to a specific parent agent and
    spawn depth. The tool's closure captures both, so each fork gets its
    own correctly-scoped tool. The persona schema lists only the personas
    the parent is actually allowed to spawn.
    """
    all_keys = set(spawner.cfg.personas.keys())
    if parent.allowed_spawn_personas is None:
        persona_keys = sorted(all_keys)
    else:
        persona_keys = sorted(all_keys & set(parent.allowed_spawn_personas))
    persona_list = ", ".join(persona_keys) if persona_keys else "(none configured)"

    async def _run(args: dict[str, Any]) -> str:
        prompt = (args.get("prompt") or "").strip()
        persona = (args.get("persona") or "").strip()
        if not prompt:
            return "error: prompt is required"
        if not persona:
            return "error: persona is required"
        # Capture the parent's currently-active inbound so the eventual
        # completion message lands in the same session. _active_inbound
        # is set by _process_batch; outside a turn (e.g. run_one_shot in
        # a subagent) there's no active inbound — and in that case the
        # caller wouldn't have subagent_spawn in their tool registry
        # anyway (forks strip it). Defensive guard keeps the contract clean.
        origin = parent._active_inbound
        if origin is None:
            return "error: subagent_spawn called outside an active session"
        return spawner.spawn_async(
            parent, persona, prompt, depth + 1, origin[0], origin[1],
        )

    return Tool(
        name="subagent_spawn",
        description=(
            "Spawn a subagent persona for a one-shot task. The child runs "
            "asynchronously against its persona's configured model and "
            "shares your workspace, tools, and memory. Persona is "
            "case-insensitive. This tool returns a task id immediately; "
            "you'll get a prompt when the task completes. (subagent_status "
            "is available for polling.) Available personas: " + persona_list
        ),
        input_schema={
            "type": "object",
            "properties": {
                "persona": {
                    "type": "string",
                    "description": (
                        "Persona name (case-insensitive). One of: "
                        + (", ".join(persona_keys) if persona_keys else "(none)")
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "What you want the subagent to do. Be specific — "
                        "they have no other context beyond the workspace."
                    ),
                },
            },
            "required": ["persona", "prompt"],
        },
        run=_run,
    )


def build_subagent_status_tool(parent: "Agent", spawner: SubagentSpawner) -> Tool:
    """Read-only lookup by task_id, parent-scoped (you can only see your
    own children). Returns the status line and the result body if the
    task has completed. Available alongside subagent_spawn.
    """

    async def _run(args: dict[str, Any]) -> str:
        task_id = (args.get("task_id") or "").strip()
        if not task_id:
            return "error: task_id is required"
        ct = spawner.tasks.get(task_id)
        if ct is None:
            return (
                f"error: unknown task_id {task_id!r} "
                "(may have been evicted from the registry)"
            )
        if ct.parent_id != parent.id:
            return (
                f"error: task_id {task_id!r} belongs to {ct.parent_id!r}, "
                f"not you ({parent.id!r}); you can only inspect your own children"
            )
        return spawner.format_status(ct)

    return Tool(
        name="subagent_status",
        description=(
            "Look up the status of a previously-spawned subagent by "
            "task_id. Returns running + elapsed time, or completed/failed/"
            "cancelled status with the full result body. Use this to poll "
            "without waiting for the auto-prompt."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": (
                        "Task id returned by subagent_spawn (e.g. "
                        "'persona-3-a1b2c3d4')."
                    ),
                },
            },
            "required": ["task_id"],
        },
        run=_run,
    )


def build_subagent_list_tool(parent: "Agent", spawner: SubagentSpawner) -> Tool:
    """List the calling parent's subagent tasks, newest first."""

    async def _run(_args: dict[str, Any]) -> str:
        mine = [ct for ct in spawner.tasks.values() if ct.parent_id == parent.id]
        if not mine:
            return "(no subagent tasks for this agent)"
        mine.sort(key=lambda c: c.started_at, reverse=True)
        lines: list[str] = []
        for ct in mine:
            if ct.status == "running":
                elapsed = _format_elapsed((_now() - ct.started_at).total_seconds())
                lines.append(
                    f"- task_id={ct.id} persona={ct.persona} status=running "
                    f"elapsed={elapsed}"
                )
            else:
                elapsed = ""
                if ct.completed_at is not None:
                    elapsed = (
                        f" elapsed="
                        f"{_format_elapsed((ct.completed_at - ct.started_at).total_seconds())}"
                    )
                lines.append(
                    f"- task_id={ct.id} persona={ct.persona} "
                    f"status={ct.status}{elapsed}"
                )
        return "\n".join(lines)

    return Tool(
        name="subagent_list",
        description=(
            "List subagents you have spawned, newest first. Returns "
            "task_id, persona, status, and elapsed time per task. Includes "
            "running and completed entries (the registry retains roughly "
            "the last 100 across all parents, oldest completed evicted "
            "first). Use this to recover a task_id you may have lost or "
            "to survey what's outstanding."
        ),
        input_schema={"type": "object", "properties": {}},
        run=_run,
    )


def build_subagent_stop_tool(parent: "Agent", spawner: SubagentSpawner) -> Tool:
    """Cancel a running subagent owned by this parent."""

    async def _run(args: dict[str, Any]) -> str:
        task_id = (args.get("task_id") or "").strip()
        if not task_id:
            return "error: task_id is required"
        ct = spawner.tasks.get(task_id)
        if ct is None:
            return f"error: unknown task_id {task_id!r}"
        if ct.parent_id != parent.id:
            return (
                f"error: task_id {task_id!r} belongs to {ct.parent_id!r}, "
                f"not you ({parent.id!r}); you can only stop your own children"
            )
        if ct.status != "running":
            return f"task_id={task_id} already {ct.status}; nothing to stop"
        if ct.aio_task is None or ct.aio_task.done():
            ct.status = "cancelled"
            ct.completed_at = _now()
            ct.result = "(cancelled — task already exited)"
            return (
                f"task_id={task_id} was running but the asyncio task had "
                "already exited; marked cancelled."
            )
        ct.aio_task.cancel()
        return (
            f"cancellation requested for task_id={task_id}. The subagent "
            "stops as soon as its current tool call returns — in-flight "
            "bash subprocesses run to completion or hit their timeout "
            "before cancellation propagates. You'll get the standard "
            "completion notification (status=cancelled) when cleanup "
            "finishes."
        )

    return Tool(
        name="subagent_stop",
        description=(
            "Cancel a running subagent by task_id. Sends an asyncio "
            "cancellation; the cleanup notification is delivered via the "
            "same mechanism as natural completion (synthetic message in "
            "this session, status=cancelled). Cancellation propagates "
            "between tool calls — an in-flight bash subprocess will run "
            "to completion or hit its timeout first. You can only stop "
            "your own children."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Task id of the running subagent to cancel.",
                },
            },
            "required": ["task_id"],
        },
        run=_run,
    )
