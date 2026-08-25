"""Subagent spawning — async persona-based agent forks.

Personas are configured in ``claw.yaml`` under ``subagents.personas:``
(case-insensitive lookup). Each spawn creates a transient ``Agent``
sharing the parent's workspace, tools (minus ``subagent_spawn`` and
``cron_*``), and memory. Transcripts are NOT persisted — a subagent's
job is one-shot.

**Async semantics.** ``subagent_spawn`` returns immediately with a
``task_id``. The child runs as a detached ``asyncio.Task`` held by
the ``SubagentSpawner`` registry. Its report goes out through a
``claw.sink.ParentSink``, which spools the full text and delivers a
compact summary into the spawner's inbox as a synthetic
``InboundMessage(..., is_subagent_completion=True)``. That flag lets
the spawner pick it up mid-turn, between its own tool calls, rather
than only after its current turn ends.

The completion the parent sees is deliberately compact: the caller-
supplied ``task_name`` (so it knows which task finished) plus a short
response preview and the workspace path of the full record, which is
spooled to ``<workspace>/.tool-results/<sid>/``. The spool file holds the
original **prompt** *and* the full **result** — the prompt is preserved
there (its only on-disk home; the transcript stubs the spawn-call args and
the registry is in-memory) but is NOT echoed into the transcript, and the
full result is NOT inlined — that round-trip was the dominant
transcript-bloat source, since (unlike tool results) it bypassed the
persistence-side truncation. Have subagents pass/return file paths,
not contents.

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

from claw.config import SubagentsConfig
# Re-exported: the completion formatting + spool helpers moved to
# claw.sink (ParentSink owns them now). Kept importable here because
# this is their historical home and the spool contract is documented
# against it.
from claw.sink import (  # noqa: F401
    _RESULT_PREVIEW_CHARS,
    _format_elapsed,
    _result_preview,
    _spool_contents,
    _spool_result,
    _task_slug,
    ParentSink,
)
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
    # Every report this task has sent its spawner. A subagent woken by its
    # own child can report again, so one spawn is not necessarily one
    # result; ``result`` stays the latest so status/list keep their shape.
    emissions: list[str] = field(default_factory=list)
    # The report being assembled for delivery; separate from ``result``,
    # which the sink sets once the text is spooled.
    pending_emission: str = ""
    # Short caller-supplied label, echoed back in the completion so the parent
    # can match a result to its request without the prompt being echoed
    # verbatim. The subagent itself never sees it.
    task_name: str = ""
    # Workspace-relative path of the spooled full result, set once the child
    # finishes. None if there was nothing to spool or the write failed.
    result_path: str | None = None
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
        task_name: str,
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
            task_name=task_name,
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
            "immediately. You do not need to collect the result: it will be "
            "delivered to you automatically as a message in this "
            "conversation when the child finishes, including partway through "
            "this turn between your tool calls. Do not sleep, poll, or do "
            "the work yourself while waiting — carry on with other work, or "
            "end your turn if there is nothing else to do. "
            f"(subagent_status with task_id={task_id} only reports whether "
            "it is still alive; you do not need it to get the result.)"
        )

    def has_outstanding_children(self, task_id: str) -> bool:
        """Whether anything this task spawned is still running — i.e. whether
        it can still be woken by a report."""
        return any(
            t.parent_task_id == task_id and t.status == "running"
            for t in self.tasks.values()
        )

    async def _emit(self, child: "Agent", ct: ChildTask) -> None:
        """Send one report to the spawner.

        Terminal unless this task's own children are still running, in which
        case it may yet be woken and report again."""
        ct.status = (
            "reporting" if self.has_outstanding_children(ct.id) else "completed"
        )
        await child.sink.reply(ct.pending_emission)
        ct.emissions.append(ct.pending_emission)

    async def _drive(self, child: "Agent", ct: ChildTask) -> None:
        """Run the task, then keep the subagent alive for as long as its own
        children could still report to it.

        This loop is the reason a subagent needs a session at all. Its context
        persists between turns, so a child's report resumes the conversation
        that asked for it rather than arriving at an agent with no history.
        """
        ct.pending_emission = await child.run_task(ct.prompt)
        log.info(
            "[%s] subagent task_id=%s persona=%s completed in %s (%d chars)",
            child.parent_id, ct.id, ct.persona,
            _format_elapsed((_now() - ct.started_at).total_seconds()),
            len(ct.pending_emission or ""),
        )
        await self._emit(child, ct)
        while self.has_outstanding_children(ct.id):
            # Bounded by the outer task timeout, which covers the whole life.
            if not await child.wait_for_inbound(self.cfg.task_timeout_seconds):
                break
            ct.pending_emission = await child.run_task()
            log.info(
                "[%s] subagent task_id=%s resumed on a child report (%d chars)",
                child.parent_id, ct.id, len(ct.pending_emission or ""),
            )
            await self._emit(child, ct)

    async def _run_subagent(self, parent: "Agent", ct: ChildTask) -> None:
        """Detached worker: holds the global semaphore, runs the child to
        completion, and reaps its scratch session afterwards."""
        # Tag this subagent's context so its bash (and any deeper spawn it
        # makes) is attributable to ct.id; a grandchild's own _run_subagent
        # overrides this for its subtree. Reset in the outer finally.
        tok_task = current_task_id.set(ct.id)
        child: "Agent | None" = None
        try:
            try:
                async with self.semaphore:
                    child = parent.fork(ct.persona, task_id=ct.id)
                    child.attach_sink(ParentSink(parent, ct))
                    await asyncio.wait_for(
                        self._drive(child, ct),
                        timeout=self.cfg.task_timeout_seconds,
                    )
            except asyncio.CancelledError:
                log.info(
                    "[%s] subagent task_id=%s persona=%s cancelled",
                    parent.id, ct.id, ct.persona,
                )
                ct.status = "cancelled"
                await self._emit_terminal(parent, ct, "(cancelled)")
            except (asyncio.TimeoutError, TimeoutError):
                log.warning(
                    "[%s] subagent task_id=%s persona=%s hit "
                    "task_timeout_seconds=%d", parent.id, ct.id, ct.persona,
                    self.cfg.task_timeout_seconds,
                )
                ct.status = "failed"
                await self._emit_terminal(
                    parent, ct,
                    f"error: subagent timed out after "
                    f"{self.cfg.task_timeout_seconds}s",
                )
            except Exception as e:
                log.exception(
                    "[%s] subagent task_id=%s persona=%s raised",
                    parent.id, ct.id, ct.persona,
                )
                ct.status = "failed"
                await self._emit_terminal(
                    parent, ct, f"error: subagent failed: {e}",
                )
            self._gc_registry()
        finally:
            if child is not None:
                # The handle is dead, so nothing can reach this session again.
                # Reaping here is what makes a later spawn of the same persona
                # start clean instead of inheriting this one's context.
                child.discard_scratch_session()
            current_task_id.reset(tok_task)
            self.children_by_parent[parent.id] = max(
                0, self.children_by_parent.get(parent.id, 1) - 1,
            )

    async def _emit_terminal(
        self, parent: "Agent", ct: ChildTask, text: str,
    ) -> None:
        """Report an abnormal exit. Separate from _emit because the child may
        not exist yet (a fork that raised), so the sink is built from the
        spawner directly."""
        try:
            ct.pending_emission = text
            await ParentSink(parent, ct).reply(text)
            ct.emissions.append(text)
        except asyncio.CancelledError:
            # Loop is shutting down (gateway restart). Status is already set
            # on ct; subagent_status will still report.
            pass

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
        head = (
            f"task_id={ct.id} task_name={ct.task_name!r} persona={ct.persona} "
            f"status={ct.status}{elapsed}"
        )
        if ct.result_path is not None:
            return (
                f"{head}\n\nFull prompt + result saved to: {ct.result_path}\n\n"
                f"Preview (first {_RESULT_PREVIEW_CHARS} chars):\n"
                f"{_result_preview(ct.result or '(no output)')}"
            )
        return f"{head}\n\nResult:\n{ct.result or '(no output)'}"


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
        task_name = (args.get("task_name") or "").strip()
        if not prompt:
            return "error: prompt is required"
        if not persona:
            return "error: persona is required"
        if not task_name:
            return (
                "error: task_name is required — a short label for this task "
                "(e.g. 'convert homepage HTML'), echoed back to you when it "
                "finishes so you can match the result to the request"
            )
        # Capture the parent's currently-active inbound so the eventual
        # completion message lands in the same session. _active_inbound
        # is set by _process_batch for a participant and by run_task for a
        # subagent, so a spawn from either has one. Defensive guard keeps the
        # contract clean for any caller outside a turn entirely.
        origin = parent._active_inbound
        if origin is None:
            return "error: subagent_spawn called outside an active session"
        return spawner.spawn_async(
            parent, persona, prompt, task_name, depth + 1, origin[0], origin[1],
        )

    return Tool(
        name="subagent_spawn",
        description=(
            "Spawn a subagent persona for a one-shot task. The child runs "
            "asynchronously against its persona's configured model and "
            "shares your workspace, tools, and memory. Persona is "
            "case-insensitive. This tool returns a task id immediately.\n\n"
            "YOU DO NOT NEED TO DO ANYTHING TO COLLECT THE RESULT. It is "
            "delivered to you automatically, as a message in this "
            "conversation, as soon as the child finishes — including partway "
            "through the turn you are in, in between your own tool calls. "
            "Never sleep, poll, re-run the task yourself, or read files "
            "hoping the child has written them: just carry on with other "
            "work, or end your turn if there is nothing else to do.\n\n"
            "Keep both sides' context small: pass file PATHS in your prompt, "
            "not pasted file contents — the subagent shares your workspace "
            "and can read_file them itself. Its output comes back as a short "
            "preview plus a path to the full result on disk (not inlined), "
            "so prefer having it write large artifacts to a file too. "
            "Available personas: " + persona_list
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
                "task_name": {
                    "type": "string",
                    "description": (
                        "A short label for this task (a few words, e.g. "
                        "'convert homepage HTML'). Echoed back to you when the "
                        "subagent finishes so you can tell which result is "
                        "which; the subagent itself never sees it."
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "What you want the subagent to do. Be specific — they "
                        "have no other context beyond the workspace. Pass file "
                        "paths, not pasted file contents: the subagent can "
                        "read_file them from the shared workspace, and it "
                        "keeps both contexts small."
                    ),
                },
            },
            "required": ["persona", "task_name", "prompt"],
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
            "cancelled status with the full result body.\n\n"
            "You do NOT need this to receive a result — completions arrive on "
            "their own. Use it only to check whether a long-running child is "
            "still alive, never as a way to wait for one."
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
