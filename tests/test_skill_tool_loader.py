"""skills/loader.py — the ``tool.py`` path.

Zero coverage before 2026-08-08, which is exactly why a deprecation sat here
unnoticed: the module used ``asyncio.iscoroutinefunction`` (removed in Python
3.16), but the line only executes for a skill shipping a ``tool.py`` and
every skill in the estate is markdown-only. Nothing ran it, so nothing
warned — the visible DeprecationWarning in the suite came from chromadb, not
from here.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

import pytest

from claw.skills.loader import _load_tool_py


def _write(tmp_path, body: str):
    p = tmp_path / "tool.py"
    p.write_text(body)
    return p


def test_async_run_is_accepted(tmp_path):
    path = _write(tmp_path, """
TOOL_SPEC = {
    "name": "widget",
    "description": "does a thing",
    "input_schema": {"type": "object", "properties": {}},
}

async def run(input):
    return "ok"
""")
    tool = _load_tool_py(path, "fallback", "fallback desc")
    assert tool.name == "widget"
    assert tool.description == "does a thing"


def test_sync_run_is_rejected(tmp_path):
    """The check the deprecated call was performing. A plain def would be
    awaited by the tool runner and blow up at call time, so it must fail at
    load."""
    path = _write(tmp_path, """
TOOL_SPEC = {"name": "widget", "description": "d", "input_schema": {}}

def run(input):
    return "ok"
""")
    with pytest.raises(ValueError, match="async def"):
        _load_tool_py(path, "fallback", "fallback desc")


def test_missing_run_is_rejected(tmp_path):
    path = _write(tmp_path, """
TOOL_SPEC = {"name": "widget", "description": "d", "input_schema": {}}
""")
    with pytest.raises(ValueError, match="async def"):
        _load_tool_py(path, "fallback", "fallback desc")


def test_non_dict_tool_spec_is_rejected(tmp_path):
    path = _write(tmp_path, """
TOOL_SPEC = ["not", "a", "dict"]

async def run(input):
    return "ok"
""")
    with pytest.raises(ValueError, match="TOOL_SPEC"):
        _load_tool_py(path, "fallback", "fallback desc")


def test_spec_fields_fall_back_to_skill_defaults(tmp_path):
    path = _write(tmp_path, """
TOOL_SPEC = {}

async def run(input):
    return "ok"
""")
    tool = _load_tool_py(path, "skill-name", "skill description")
    assert tool.name == "skill-name"
    assert tool.description == "skill description"


async def test_loaded_tool_actually_runs(tmp_path):
    """End-to-end: the Tool's run is the module's coroutine function."""
    path = _write(tmp_path, """
TOOL_SPEC = {"name": "echo", "description": "d", "input_schema": {}}

async def run(input):
    return f"got {input['x']}"
""")
    tool = _load_tool_py(path, "fallback", "fallback desc")
    assert await tool.run({"x": 42}) == "got 42"
