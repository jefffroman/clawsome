"""Ambient per-turn run context.

`current_sid` carries the session id of the in-flight turn down into tool
calls without threading it through the `Tool.run(args)` signature. It is
set by `OllamaClient.run_turn` and read by the bash tool so a spawned
subprocess can be registered under the session that owns it — which is
what lets `%stop --force` kill that session's orphaned process trees.

ContextVars are task/coroutine-local and copy into awaited coroutines, so
each concurrent session's turn sees its own sid; nested run_turns
(subagents) set/reset with the returned token.
"""

from __future__ import annotations

from contextvars import ContextVar

current_sid: ContextVar[str] = ContextVar("claw_current_sid", default="")

# Identifies the one conversational turn (_process_batch invocation) that
# owns this context. Inherited transitively into spawned subagent tasks
# (asyncio.create_task copies the context), so an entire spawn cascade —
# at any depth — shares the turn id of the turn that rooted it. %stop uses
# this to cancel exactly the stopped turn's cascade (and tag its bash).
current_turn_id: ContextVar[str] = ContextVar("claw_current_turn_id", default="")

# The subagent whose run owns this context (its ChildTask.id), or "" for
# the top-level agent. Set per-subagent in _run_subagent, so each level of
# a cascade tags its own bash with its own task id (a grandchild overrides
# its parent's value for its own subtree). Lets %stop <task_id> kill just
# that subagent's bash.
current_task_id: ContextVar[str] = ContextVar("claw_current_task_id", default="")
