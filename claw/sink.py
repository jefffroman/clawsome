"""Where an agent's replies go.

Claw has historically had two incompatible notions of "agent", and the split
was never designed — it is residue from the commit that made ``subagent_spawn``
asynchronous. Before that, spawn was ``await spawner.spawn(...)`` returning the
child's reply text, so a subagent was a plain function call: it needed no inbox
and no session, at any depth. Making it async meant the result could no longer
come back as a return value, so it needed somewhere to *arrive*. An inbox was
added — but only for top-level agents, because the parent of a top-level spawn
is always one.

A ``Sink`` is the seam that removes the split. Every agent runs the same turn
loop; the only thing that varies is where its replies land:

- ``ChannelSink`` — a participant. Writes to a transport (matrix, voice), which
  is exactly what ``Agent`` did inline before this existed.
- ``ParentSink`` — a subagent. Spools the full text to a workspace file and
  delivers a compact summary into its spawner's inbox.

**The compact-summary contract is load-bearing.** A subagent's full output must
never be inlined into its parent's persisted transcript — that round-trip was
the dominant transcript-bloat source, since (unlike tool results) it bypassed
the persistence-side truncation. What the parent sees is the caller-supplied
``task_name``, the status, the elapsed time, the path to the full record, and a
``_RESULT_PREVIEW_CHARS``-bounded preview. The spool file is also the only
on-disk home of the original prompt, since the parent's persisted tool-call args
are stubbed past 200 chars.

Asides (reasoning traces, operator notices) are participant-only. A subagent's
spawner wants its answer, not its thinking, so ``ParentSink`` drops them and
declares ``shows_asides = False`` for callers that need to know whether anything
went out ahead of the answer.

PUBLIC MIRROR: no estate specifics here.
"""

from __future__ import annotations

import contextlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, AsyncContextManager, Protocol

from claw.channel.base import Channel, InboundMessage

if TYPE_CHECKING:
    from claw.agent import Agent
    from claw.tools.subagent import ChildTask

log = logging.getLogger("claw.sink")

# A subagent's full result is spooled to a workspace file; only this many
# leading chars are inlined into the completion the parent receives. Keeps a
# large result (e.g. a whole HTML file) out of the parent's persisted
# transcript — the parent read_file's the spooled path for the full output.
_RESULT_PREVIEW_CHARS = 200


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _format_elapsed(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    m = s // 60
    if m < 60:
        return f"{m}m{s % 60}s"
    h = m // 60
    return f"{h}h{m % 60}m"


def _task_slug(name: str, maxlen: int = 40) -> str:
    """Filesystem-safe slug from a task_name, for the spool filename."""
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return slug[:maxlen].strip("-") or "task"


def _result_preview(result: str) -> str:
    if len(result) <= _RESULT_PREVIEW_CHARS:
        return result
    more = len(result) - _RESULT_PREVIEW_CHARS
    return f"{result[:_RESULT_PREVIEW_CHARS]}… (+{more:,} more chars in the file)"


def _spool_contents(ct: "ChildTask") -> str:
    """The full spool payload: a task header, the ORIGINAL PROMPT, then the
    RESULT.

    The prompt is included here because the spool is the **only** place it's
    kept on disk — the parent's persisted transcript stubs the spawn-call args
    once they pass 200 chars (``_truncate_persisted_tool_call_args``), the
    spawn log line records only metadata, and the in-memory registry is lost on
    restart / evicted after ~100 tasks. So the ref'd file is the sole durable
    (for the ~24h scratch lifetime) record of what the subagent was asked.
    The completion preview stays response-only — it reads ``ct.result``, not
    this payload.
    """
    return (
        f"=== SUBAGENT TASK {ct.task_name!r}  "
        f"(persona={ct.persona}, task_id={ct.id}, status={ct.status}) ===\n\n"
        f"--- PROMPT ---\n{ct.prompt}\n\n"
        f"--- RESULT ---\n{ct.result or ''}\n"
    )


def _spool_result(ct: "ChildTask", workspace_dir: Path) -> str | None:
    """Write the prompt + ``ct.result`` to a transient spool file under
    ``<workspace>/.tool-results/<origin_sid>/`` and return its
    workspace-relative path. Reuses the same scratch tree as large tool
    results, so ``Agent.sweep_spool_tree`` (run after each nightly rotate)
    reaps it on the same >24h grace. Returns None if there is nothing to
    spool or the write fails — the caller then falls back to a bounded
    inline result rather than losing it.
    """
    if ct.result is None:
        return None
    try:
        spool_dir = workspace_dir / ".tool-results" / ct.origin_session_key
        spool_dir.mkdir(parents=True, exist_ok=True)
        path = spool_dir / f"{_task_slug(ct.task_name)}-{ct.id}.txt"
        path.write_text(_spool_contents(ct))
        return str(path.relative_to(workspace_dir))
    except OSError:
        log.exception(
            "[%s] failed to spool subagent result for task_id=%s",
            ct.parent_id, ct.id,
        )
        return None


def completion_body(ct: "ChildTask") -> str:
    """The compact summary a spawner receives for one emission.

    Deliberately does NOT contain the full result. See the module docstring —
    inlining it here is the transcript-bloat regression this shape exists to
    prevent.
    """
    elapsed = ""
    if ct.completed_at is not None:
        elapsed = f" elapsed={_format_elapsed((ct.completed_at - ct.started_at).total_seconds())}"
    # A subagent whose own children are still running may report before it is
    # done, so the verb has to be honest about which of the two this is —
    # "finished ... status=reporting" would be a contradiction.
    verb = "reported (still working)" if ct.status == "reporting" else "finished"
    header = (
        f"Subagent task {ct.task_name!r} {verb}. "
        f"task_id={ct.id} persona={ct.persona} status={ct.status}{elapsed}"
    )
    preview = _result_preview(ct.result or "(no output)")
    if ct.result_path is not None:
        body_lines = [
            header,
            "",
            f"Full prompt + result saved to: {ct.result_path}",
            "(Transient scratch — reaped ~24h after the next session "
            "rotate. read_file / bash that path for the full prompt and "
            "output; copy it elsewhere to keep it.)",
            "",
            f"Preview (first {_RESULT_PREVIEW_CHARS} chars):",
            preview,
        ]
    else:
        # Spool unavailable — inline a bounded preview rather than the
        # full result, so a disk hiccup can't reintroduce the bloat.
        body_lines = [header, "", "Result:", preview]
    return "\n".join(body_lines)


class Sink(Protocol):
    """Where one agent's replies go. Implementations must be safe to call
    from inside a turn."""

    # Whether asides (reasoning traces, operator notices) actually reach a
    # reader. Callers use this to decide whether anything preceded the answer.
    shows_asides: bool

    async def reply(self, text: str) -> None:
        """Deliver the turn's answer."""
        ...

    async def aside(self, text: str) -> None:
        """Deliver an out-of-band block. May be dropped."""
        ...

    def typing(self) -> AsyncContextManager[None]:
        """Signal work-in-progress for the duration of a turn."""
        ...


class ChannelSink:
    """A participant: replies go out over a transport.

    A thin wrapper over what ``Agent`` previously called inline, so participant
    behaviour is unchanged by the existence of this seam.
    """

    shows_asides = True

    def __init__(self, channel: Channel, peer_id: str) -> None:
        self.channel = channel
        self.peer_id = peer_id

    async def reply(self, text: str) -> None:
        await self.channel.send(self.peer_id, text)

    async def aside(self, text: str) -> None:
        await self.channel.send(self.peer_id, text)

    def typing(self) -> AsyncContextManager[None]:
        return self.channel.typing(self.peer_id)


class ParentSink:
    """A subagent: replies are spooled, then summarised into the spawner's
    inbox as a synthetic inbound.

    ``is_subagent_completion`` marks the message so the spawner may pick it up
    mid-turn (``Agent._take_subagent_completions``) rather than only after its
    current turn releases the session lock.
    """

    shows_asides = False

    def __init__(self, parent: "Agent", ct: "ChildTask") -> None:
        self.parent = parent
        self.ct = ct

    async def reply(self, text: str) -> None:
        ct = self.ct
        ct.result = text
        ct.completed_at = _now()
        ct.result_path = _spool_result(ct, self.parent.agent_cfg.workspace)
        if ct.suppress_delivery:
            # Whole-turn %stop killed this cascade: do NOT inbound a
            # completion, or the stopped session would be resurrected
            # by its own zombie. Status is set; %subagents still shows.
            log.info(
                "[%s] subagent task_id=%s completion suppressed "
                "(session stopped)", self.parent.id, ct.id,
            )
            return
        msg = InboundMessage(
            peer_id=ct.origin_peer_id,
            sender_name=f"subagent:{ct.persona}:{ct.id}",
            text=completion_body(ct),
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
            # Lets the spawner pick this up mid-turn instead of only
            # after the current turn releases the session lock.
            is_subagent_completion=True,
        )
        try:
            await self.parent.handle_inbound(msg)
        except Exception:
            log.exception(
                "[%s] subagent completion delivery raised for task_id=%s",
                self.parent.id, ct.id,
            )

    async def aside(self, text: str) -> None:
        """Dropped. A spawner wants its subagent's answer, not its reasoning."""
        return None

    def typing(self) -> AsyncContextManager[None]:
        return contextlib.nullcontext()
