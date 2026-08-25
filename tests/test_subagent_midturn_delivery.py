"""A subagent completion must reach the turn that spawned it.

Before this, ``run_turn``'s tool loop was a closed system: it could only learn
what was true when the turn started, plus its own tool results. A child that
finished mid-turn fired a synthetic InboundMessage at its spawner, which queued
behind the per-session lock that the very same turn was holding — so the report
was invisible until the turn ended. The docstring's promise that "the parent's
next turn carries the result" held only for the next *conversational* turn; the
tool turns in between could not see it.

Measured live 2026-08-24: child finished in 63s, spawner ran 38 more tool turns
and hit the ceiling, and the completion was delivered 56m32s later — 48ms after
the turn released the lock. The spawner had meanwhile redone the work itself.

Two layers are covered: ``run_turn``'s drain hook, and the agent-side selection
of what is eligible for it.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from claw.channel.base import InboundMessage
from claw.config import OllamaConfig
from claw.tools.base import Tool
from claw.ollama import OllamaClient


def _tool_reply(name: str = "bash") -> dict:
    return {
        "message": {"content": "", "tool_calls": [
            {"function": {"name": name, "arguments": {"command": "true"}}},
        ]},
        "done_reason": "stop",
    }


def _reply(content: str = "done") -> dict:
    return {"message": {"content": content}, "done_reason": "stop"}


def _client() -> OllamaClient:
    return OllamaClient(OllamaConfig(
        base_url="http://ollama.invalid",
        default_compaction_model="test-compact-model",
    ))


async def _echo(_input) -> str:  # noqa: ANN001
    return "tool-output"


def _tools() -> dict:
    return {"bash": Tool(
        name="bash", description="t",
        input_schema={"type": "object", "properties": {}}, run=_echo,
    )}


async def _run(scripted: list, drain_inbox=None) -> tuple:
    client = _client()
    n = {"i": 0}
    seen: list[list[dict]] = []

    async def fake_chat_once(**kw):
        # Snapshot what the model was shown on each call.
        seen.append([dict(m) for m in kw["messages"]])
        item = scripted[n["i"]]
        n["i"] += 1
        return item

    client.chat_once = fake_chat_once  # type: ignore[assignment]
    result = await client.run_turn(
        model="test-model",
        history=[{"role": "user", "content": "hi"}],
        system=None,
        tools=_tools(),
        sid="s",
        workspace_dir=Path("/tmp"),
        label="test",
        drain_inbox=drain_inbox,
    )
    return result, seen


# --- run_turn: the drain hook ------------------------------------------

async def test_completion_arriving_midturn_reaches_the_same_turn():
    """THE regression. The child reports during tool turn 0; the model must
    see it on the very next call, not after the turn ends."""
    inbox = [[{"role": "user", "content": "subagent-x: finished"}]]

    def drain():
        return inbox.pop(0) if inbox else []

    (new_messages, _, _), seen = await _run(
        [_tool_reply(), _reply("ok")], drain_inbox=drain,
    )
    # Turn 1's prompt carries the completion.
    turn1 = seen[1]
    assert any(m.get("content") == "subagent-x: finished" for m in turn1), \
        "the completion must be visible to the next model call"
    # And it is persisted, so the transcript records what the model saw.
    assert any(m.get("content") == "subagent-x: finished" for m in new_messages)


async def test_injection_never_splits_a_tool_call_from_its_result():
    """Tool atomicity: the injected row lands after every tool result for the
    preceding assistant, never between. Compaction's cut-point invariant and
    ChatML both depend on this."""
    def drain():
        return [{"role": "user", "content": "subagent-x: finished"}]

    (new_messages, _, _), _ = await _run(
        [_tool_reply(), _tool_reply(), _reply("ok")], drain_inbox=drain,
    )
    for i, m in enumerate(new_messages):
        if m["role"] != "user" or i == 0:
            continue
        prev = new_messages[i - 1]
        # The row an injection lands on must never be an assistant still
        # awaiting its tool results.
        assert not (prev["role"] == "assistant" and prev.get("tool_calls")), \
            f"injected row at {i} splits a tool_call from its result"
    # And every tool_call in the run still has its result immediately after.
    for i, m in enumerate(new_messages):
        for k, _tc in enumerate(m.get("tool_calls") or []):
            nxt = new_messages[i + 1 + k]
            assert nxt["role"] == "tool", \
                f"tool_call {k} of row {i} is not followed by its result"


async def test_drain_is_consulted_every_tool_turn():
    """Not once per turn, not only at the start — every iteration."""
    calls = {"n": 0}

    def drain():
        calls["n"] += 1
        return []

    await _run([_tool_reply(), _tool_reply(), _reply("ok")], drain_inbox=drain)
    assert calls["n"] == 3, f"expected one drain per tool turn, got {calls['n']}"


async def test_absent_hook_is_a_noop():
    """Callers that pass no hook (flush, curator) are unaffected."""
    (new_messages, final, _), _ = await _run([_tool_reply(), _reply("ok")])
    assert final == "ok"
    assert not [m for m in new_messages if m["role"] == "user"]


# --- agent: what is eligible -------------------------------------------

def _agent():
    """A bare Agent with only the attributes _take_subagent_completions uses."""
    from claw.agent import Agent
    a = Agent.__new__(Agent)
    a.agent_cfg = SimpleNamespace(id="agent-1")
    a._pending_inbound = {}
    a._last_inbound_at = {}
    a.cfg = type("C", (), {"tz": "UTC"})()
    return a


def _completion() -> InboundMessage:
    return InboundMessage(
        peer_id="!room", sender_name="subagent:p:p-1", text="Subagent task done",
        channel="matrix", sender_id="p-1", is_subagent_completion=True,
    )


def _human() -> InboundMessage:
    return InboundMessage(
        peer_id="!room", sender_name="user-1", text="hello",
        channel="matrix", sender_id="@user-1:example.org",
    )


def test_only_subagent_completions_are_taken():
    """A human message mid-turn stays queued: it becomes its own turn, and is
    NOT allowed to interrupt work already in flight."""
    a = _agent()
    sid = "matrix__room"
    a._pending_inbound[sid] = [_human(), _completion(), _human()]
    rows = a._take_subagent_completions(sid)
    assert len(rows) == 1
    assert "Subagent task done" in rows[0]["content"]
    left = a._pending_inbound[sid]
    assert len(left) == 2 and all(not m.is_subagent_completion for m in left)


def test_taking_is_idempotent():
    """A completion is delivered once; the next tool turn must not re-inject
    it (which would loop the same report into the prompt every iteration)."""
    a = _agent()
    sid = "matrix__room"
    a._pending_inbound[sid] = [_completion()]
    assert len(a._take_subagent_completions(sid)) == 1
    assert a._take_subagent_completions(sid) == []


def test_empty_queue_is_cheap_and_silent():
    a = _agent()
    assert a._take_subagent_completions("nothing-here") == []


# --- the delivery contract the model actually reads ---------------------
#
# Tool descriptions sit in the tool spec of EVERY request, so they are the
# durable contract — far more load-bearing than the one-off string returned
# at spawn time. Both used to undersell delivery ("you'll get a prompt")
# while subagent_status advertised itself as the way to "poll without waiting
# for the auto-prompt". Faced with that, the model reasonably concluded it had
# to fetch the result, and hand-rolled `sleep 20; cat <the child's file>`.

def _spawn_tool():
    from claw.config import SubagentsConfig
    from claw.tools.subagent import SubagentSpawner, build_subagent_spawn_tool
    spawner = SubagentSpawner(SubagentsConfig(
        max_concurrent=2, max_children_per_agent=2,
        default_model="m", personas={},
    ))
    parent = SimpleNamespace(
        id="parent-1", allowed_spawn_personas=None, _active_inbound=("matrix", "!r"),
    )
    return build_subagent_spawn_tool(parent, spawner, depth=0), spawner, parent


def test_spawn_states_delivery_is_automatic():
    """The contract must say the result ARRIVES, not merely that one exists."""
    tool, _, _ = _spawn_tool()
    d = tool.description.lower()
    assert "automatically" in d
    assert "do not need to do anything to collect" in d


def test_spawn_forbids_the_observed_workaround():
    tool, _, _ = _spawn_tool()
    d = tool.description.lower()
    for banned in ("sleep", "poll", "read files"):
        assert banned in d, f"the prohibition on {banned!r} must be explicit"
    assert "available for polling" not in d


def test_status_does_not_present_itself_as_the_way_to_get_a_result():
    """THE regression. This description shipped in every request telling the
    model to poll instead of waiting."""
    from claw.config import SubagentsConfig
    from claw.tools.subagent import SubagentSpawner, build_subagent_status_tool
    spawner = SubagentSpawner(SubagentsConfig(
        max_concurrent=2, max_children_per_agent=2,
        default_model="m", personas={},
    ))
    tool = build_subagent_status_tool(SimpleNamespace(id="parent-1"), spawner)
    d = tool.description.lower()
    assert "use this to poll" not in d
    assert "you do not need this to receive a result" in d


async def test_spawn_return_string_repeats_the_contract(tmp_path):
    """Said again at the moment of spawning, where the decision is made —
    a real spawn, not the unknown-persona error path."""
    from claw.config import PersonaConfig, SubagentsConfig
    from claw.tools.subagent import SubagentSpawner, build_subagent_spawn_tool

    spawner = SubagentSpawner(SubagentsConfig(
        max_concurrent=2, max_children_per_agent=2, default_model="m",
        personas={"grunt": PersonaConfig(model="m", role="grunt")},
    ))

    class _Parent:
        id = "parent-1"
        allowed_spawn_personas = None
        _active_inbound = ("matrix", "!room:x")
        agent_cfg = SimpleNamespace(workspace=tmp_path)

        def fork(self, _persona, task_id=""):  # noqa: ANN001
            # Minimal stand-in for the child agent _run_subagent drives.
            async def _run_task(_prompt=None):  # noqa: ANN001
                return "child result"

            child = SimpleNamespace(
                run_task=_run_task,
                parent_id="parent-1",
                discard_scratch_session=lambda: None,
            )
            child.attach_sink = lambda sink: setattr(child, "sink", sink)
            return child

        async def handle_inbound(self, msg):  # noqa: ANN001
            pass

    tool = build_subagent_spawn_tool(_Parent(), spawner, depth=0)
    out = (await tool.run({
        "persona": "grunt", "task_name": "t", "prompt": "p",
    })).lower()
    assert "automatically" in out
    assert "do not need to collect the result" in out
    assert "do not sleep, poll" in out
