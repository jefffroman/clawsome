"""The spawner surface the operator commands stand on.

9fdf319 (the sink extraction) deleted ``cancel_turn``, ``cancel_subtree``,
``running_for_session``, ``format_running`` and ``_gc_registry`` along with
``_deliver_completion`` — only the last was meant to move. Nothing tested
them, so %stop, %stop <task_id> and %subagents all raised AttributeError, and
every finished subagent raised one on its way out of ``_run_subagent``.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
from datetime import datetime, timedelta, timezone

import claw.agent
import claw.tools.subagent as subagent_mod
from claw.config import PersonaConfig, SubagentsConfig
from claw.tools.subagent import ChildTask, SubagentSpawner


def _spawner() -> SubagentSpawner:
    return SubagentSpawner(SubagentsConfig(
        max_concurrent=2, max_children_per_agent=2, default_model="test-model",
        personas={"grunt": PersonaConfig(model="test-model", role="grunt")},
    ))


def _ct(task_id, *, sid="matrix__room", turn="turn-1", parent="",
        status="running", age_s=0) -> ChildTask:
    return ChildTask(
        id=task_id, parent_id="agent-1", persona="grunt", prompt="p",
        origin_channel="matrix", origin_peer_id="!room:example.org",
        origin_session_key=sid,
        started_at=datetime.now(timezone.utc) - timedelta(seconds=age_s),
        status=status, spawn_turn_id=turn, parent_task_id=parent,
        task_name=task_id,
    )


def _called_spawner_methods(module) -> set[str]:
    """Every ``<x>.spawner.<name>`` / ``spawner.<name>`` / ``sp.<name>`` /
    ``self.<name>`` attribute used where the receiver is a spawner."""
    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        recv = node.value
        if isinstance(recv, ast.Attribute) and recv.attr == "spawner":
            names.add(node.attr)
        elif isinstance(recv, ast.Name) and recv.id in ("spawner", "sp"):
            names.add(node.attr)
    return names


def test_every_spawner_attribute_the_agent_uses_exists():
    sp = _spawner()
    missing = {
        n for n in _called_spawner_methods(claw.agent)
        if not hasattr(sp, n)
    }
    assert not missing, f"Agent calls spawner attributes that don't exist: {missing}"


def test_every_self_method_the_spawner_calls_exists():
    tree = ast.parse(inspect.getsource(SubagentSpawner))
    called = {
        n.func.attr for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"
    }
    missing = {n for n in called if not hasattr(SubagentSpawner, n)}
    assert not missing, f"SubagentSpawner calls missing self methods: {missing}"


async def test_cancel_turn_cancels_the_cascade_and_suppresses_it():
    sp = _spawner()
    hang = asyncio.Event()
    for ct in (
        _ct("a", turn="turn-1"),
        _ct("b", turn="turn-1", parent="a"),
        _ct("c", turn="turn-2"),
        _ct("d", turn="turn-1", status="completed"),
    ):
        if ct.status == "running":
            ct.aio_task = asyncio.create_task(hang.wait())
        sp.tasks[ct.id] = ct

    assert sorted(sp.cancel_turn("turn-1", suppress=True)) == ["a", "b"]
    await asyncio.sleep(0)
    assert sp.tasks["a"].aio_task.cancelled()
    assert sp.tasks["b"].aio_task.cancelled()
    assert sp.tasks["a"].suppress_delivery and sp.tasks["b"].suppress_delivery
    assert not sp.tasks["c"].aio_task.done()
    assert not sp.tasks["c"].suppress_delivery
    assert sp.cancel_turn("", suppress=True) == []
    sp.tasks["c"].aio_task.cancel()


async def test_cancel_subtree_delivers_the_target_only():
    sp = _spawner()
    hang = asyncio.Event()
    for ct in (_ct("a"), _ct("b", parent="a"), _ct("c", parent="b"),
               _ct("x")):
        ct.aio_task = asyncio.create_task(hang.wait())
        sp.tasks[ct.id] = ct

    assert sorted(sp.cancel_subtree("a")) == ["a", "b", "c"]
    assert not sp.tasks["a"].suppress_delivery
    assert sp.tasks["b"].suppress_delivery and sp.tasks["c"].suppress_delivery
    assert not sp.tasks["x"].aio_task.done()
    assert sp.cancel_subtree("nope") == []
    sp.tasks["x"].aio_task.cancel()
    await asyncio.sleep(0)


def test_running_for_session_lists_only_this_sessions_running_newest_first():
    sp = _spawner()
    for ct in (_ct("old", age_s=60), _ct("new", age_s=1),
               _ct("done", status="completed"), _ct("elsewhere", sid="other")):
        sp.tasks[ct.id] = ct
    cts = sp.running_for_session("matrix__room")
    assert [c.id for c in cts] == ["new", "old"]
    text = sp.format_running(cts)
    assert text.startswith("2 subagent(s) running:")
    assert "new" in text and "old" in text
    assert sp.format_running([]) == "No subagents running for this session."


def test_gc_registry_evicts_oldest_finished_never_running(monkeypatch):
    monkeypatch.setattr(subagent_mod, "_MAX_REGISTRY_SIZE", 2)
    sp = _spawner()
    now = datetime.now(timezone.utc)
    for i, status in enumerate(["completed", "completed", "running"]):
        ct = _ct(f"t{i}", status=status)
        if status != "running":
            ct.completed_at = now + timedelta(seconds=i)
        sp.tasks[ct.id] = ct
    sp._gc_registry()
    assert set(sp.tasks) == {"t1", "t2"}
