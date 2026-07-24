"""Subagent completion is compact: the caller-supplied ``task_name`` is
echoed, the prompt + full result are spooled to a workspace file, the prompt
is NOT echoed, and the full result is NOT inlined into the parent's transcript.

Regression guard for the transcript-bloat leak: subagent completions used to
re-inject the full prompt + full result verbatim, bypassing the
persistence-side truncation that bounds every other content path. The prompt
is still preserved — in the spool file (its only on-disk home), not the
transcript.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from claw.config import SubagentsConfig
from claw.tools.subagent import (
    _RESULT_PREVIEW_CHARS,
    ChildTask,
    SubagentSpawner,
    _result_preview,
    _spool_result,
    _task_slug,
    build_subagent_spawn_tool,
)


def _spawner() -> SubagentSpawner:
    return SubagentSpawner(
        SubagentsConfig(
            max_concurrent=2,
            max_children_per_agent=2,
            default_model="m",
            personas={},
        )
    )


class _RecordingParent:
    """Minimal Agent stand-in: captures the synthetic completion inbound."""

    def __init__(self, workspace):
        self.id = "parent-1"
        self.allowed_spawn_personas = None
        self._active_inbound = ("matrix", "!room:x")
        self.agent_cfg = SimpleNamespace(workspace=workspace)
        self.received: list = []

    async def handle_inbound(self, msg):
        self.received.append(msg)


def _child(workspace_key: str, **overrides) -> ChildTask:
    base = dict(
        id="cristal-abcd1234",
        parent_id="parent-1",
        persona="cristal",
        prompt="PROMPT-SENTINEL should never reach the parent transcript",
        origin_channel="matrix",
        origin_peer_id="!room:x",
        origin_session_key=workspace_key,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        status="completed",
        task_name="Convert homepage HTML",
        result="RESULT-SENTINEL " + ("z" * 40_000),
    )
    base.update(overrides)
    return ChildTask(**base)


def test_task_slug():
    assert _task_slug("Convert homepage HTML") == "convert-homepage-html"
    assert _task_slug("  ***  ") == "task"
    assert _task_slug("x" * 100) == "x" * 40


def test_result_preview_caps():
    assert _result_preview("hi") == "hi"
    big = "a" * (_RESULT_PREVIEW_CHARS + 500)
    out = _result_preview(big)
    assert out.startswith("a" * _RESULT_PREVIEW_CHARS)
    assert "more chars in the file" in out
    assert len(out) < len(big)


def test_spool_writes_prompt_and_result(tmp_path):
    ct = _child("matrix__room")
    rel = _spool_result(ct, tmp_path)
    assert rel is not None
    contents = (tmp_path / rel).read_text()
    # BOTH the original prompt and the full result are preserved in the file...
    assert ct.prompt in contents
    assert ct.result in contents
    # ...under labeled sections, with the task header.
    assert "--- PROMPT ---" in contents
    assert "--- RESULT ---" in contents
    assert "Convert homepage HTML" in contents
    assert "convert-homepage-html" in rel
    assert "cristal-abcd1234" in rel


def test_completion_is_compact_and_spooled(tmp_path):
    parent = _RecordingParent(tmp_path)
    spawner = _spawner()
    ct = _child("matrix__room")
    ct.result_path = _spool_result(ct, parent.agent_cfg.workspace)

    asyncio.run(spawner._deliver_completion(parent, ct))

    assert len(parent.received) == 1
    body = parent.received[0].text
    # task_name is echoed so the parent can match result to request...
    assert "Convert homepage HTML" in body
    # ...but the prompt is NOT echoed...
    assert "PROMPT-SENTINEL" not in body
    # ...and the full result is NOT inlined (only a bounded preview + path).
    assert "z" * 1000 not in body
    assert ct.result_path in body
    assert "Preview (first" in body
    # the full result stays recoverable on disk...
    spooled = (tmp_path / ct.result_path).read_text()
    assert ct.result in spooled
    # ...and so does the prompt — preserved in the file though never echoed
    # into the transcript (body above).
    assert "PROMPT-SENTINEL" in spooled


def test_completion_falls_back_to_inline_when_spool_missing(tmp_path):
    parent = _RecordingParent(tmp_path)
    spawner = _spawner()
    ct = _child("matrix__room", result="short answer", result_path=None)

    asyncio.run(spawner._deliver_completion(parent, ct))

    body = parent.received[0].text
    assert "short answer" in body
    assert "Convert homepage HTML" in body


def test_task_name_required(tmp_path):
    parent = _RecordingParent(tmp_path)
    tool = build_subagent_spawn_tool(parent, _spawner(), depth=0)
    out = asyncio.run(tool.run({"persona": "cristal", "prompt": "do a thing"}))
    assert "task_name is required" in out
