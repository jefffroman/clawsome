"""Pure helpers across memory_flush / config / workspace_inject.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

import re

import pytest

from claw.config import _parse_can_spawn, _validate_can_spawn
from claw.memory_flush import today_iso_date
from claw.workspace_inject import render_extra_paths


def test_today_iso_date_shape_and_bad_zone():
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", today_iso_date(None))
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", today_iso_date("America/New_York"))
    with pytest.raises(Exception):
        today_iso_date("Definitely/Not/A/Zone")


def test_parse_can_spawn_lowercases():
    assert _parse_can_spawn(None) is None
    assert _parse_can_spawn(["Persona-1", "PERSONA-3"]) == ("persona-1", "persona-3")


def test_validate_can_spawn(make_cfg, make_agent_cfg, tmp_path):
    # Clean: default agent has can_spawn=None -> no raise.
    _validate_can_spawn(make_cfg(tmp_path))
    # Agent references a persona that doesn't exist -> ValueError.
    bad = make_cfg(
        tmp_path,
        agents=(make_agent_cfg(tmp_path, can_spawn=("ghost",)),),
    )
    with pytest.raises(ValueError, match="unknown persona"):
        _validate_can_spawn(bad)


def test_render_extra_paths(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "A.md").write_text("hello\n\n")   # trailing ws is rstripped
    (ws / "empty.md").write_text("")        # empty body -> skipped
    out = render_extra_paths(ws, ("A.md", "missing.md", "empty.md"))
    assert out == "## A.md\n\nhello"
