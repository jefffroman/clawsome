"""`think` is opt-in per call site, and only where the trace is discarded.

Three similarly-named knobs have been conflated before, so to be explicit:
this is Ollama's `think` request field (do not GENERATE a trace), not
`reasoning_effort` (measured a no-op) and not retaining `thinking` in history
(measured, rejected). `summarize()` has carried think=False since it was
written, but it is a non-tool call — it proved nothing about a tool-bearing
turn, which is what a flush is.

Measured 2026-08-24 on qwen3.6:35b-a3b, real 25-row flush prompt, n=3:
19.0s / 1162 eval tokens with thinking on vs 2.7s / 181 with it off, and the
append_file call emitted in every run of both arms.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from claw.config import OllamaConfig
from claw.ollama import OllamaClient
from claw.tools.base import Tool


def _client() -> OllamaClient:
    return OllamaClient(OllamaConfig(
        base_url="http://ollama.invalid",
        default_compaction_model="test-compact-model",
    ))


class _Recorder:
    """Captures each request body and replays scripted responses."""

    def __init__(self, statuses: list[int] | None = None) -> None:
        self.bodies: list[dict] = []
        self.statuses = statuses or []

    async def post(self, _url, json=None):  # noqa: ANN001
        # Snapshot: the retry path pops "think" from the SAME dict, so storing
        # the reference would make both recorded bodies look identical.
        self.bodies.append(dict(json or {}))
        status = self.statuses.pop(0) if self.statuses else 200
        req = httpx.Request("POST", "http://ollama.invalid/api/chat")
        return httpx.Response(
            status,
            json={"message": {"content": "done"}, "done_reason": "stop"},
            request=req,
        )


async def _run(client: OllamaClient, **kw):
    return await client.run_turn(
        model="test-model",
        history=[{"role": "user", "content": "hi"}],
        system=None,
        tools={},
        sid="s",
        workspace_dir=Path("/tmp"),
        label="test",
        **kw,
    )


async def test_think_is_absent_unless_asked():
    """Default must stay the model's own. A conversational turn surfaces its
    trace via %thinking, and the curator needs one to reason with."""
    c = _client()
    rec = _Recorder()
    c._client = rec  # type: ignore[assignment]
    await _run(c)
    assert "think" not in rec.bodies[0]


async def test_think_false_is_forwarded_when_asked():
    c = _client()
    rec = _Recorder()
    c._client = rec  # type: ignore[assignment]
    await _run(c, think=False)
    assert rec.bodies[0]["think"] is False


async def test_a_model_rejecting_think_is_retried_without_it():
    """Guard for a future non-hybrid model: a 400 on `think` must degrade to a
    normal call, not kill the turn."""
    c = _client()
    rec = _Recorder(statuses=[400, 200])
    c._client = rec  # type: ignore[assignment]
    _, final, _ = await _run(c, think=False)
    assert len(rec.bodies) == 2
    assert rec.bodies[0]["think"] is False
    assert "think" not in rec.bodies[1]
    assert final == "done"


async def test_the_flush_turn_asks_for_it():
    """THE point: the flush discards its prose, so the trace is pure waste.
    Asserted at the real call site, not on a hand-built body."""
    from claw import memory_flush

    seen: dict = {}

    class _FakeOllama:
        async def run_turn(self, **kw):
            seen.update(kw)
            return ([], "", "")

    async def _noop(_input):  # noqa: ANN001
        return ""

    await memory_flush.run_memory_flush(
        agent_id="agent-1",
        peer_label="user-1",
        ollama=_FakeOllama(),
        sid="s",
        workspace_dir=Path("/tmp"),
        rows=[{"role": "user", "content": "hello"}],
        primary_model="test-compact-model",
        tools={"append_file": Tool(
            name="append_file", description="t",
            input_schema={"type": "object", "properties": {}}, run=_noop,
        )},
        reason="periodic-growth",
    )
    assert seen["think"] is False
    # Bounded too: unset, these fell back to the client-wide 50 tool turns and
    # Ollama's unlimited num_predict, so a confused flush could consume its
    # entire turn_timeout_s deadline.
    assert seen["num_predict"] == 4096
    assert seen["max_tool_turns"] == 6
