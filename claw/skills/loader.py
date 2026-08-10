"""Workspace skill discovery.

A skill lives at ``<workspace>/skills/<name>/``:

* ``SKILL.md`` (required) — markdown protocol injected into the system prompt
  inside a ``<skill_catalog>`` block. Optional YAML frontmatter sets
  ``name`` and ``description``.
* ``tool.py`` (optional) — Python module exporting:

  .. code-block:: python

      TOOL_SPEC = {
          "name": "calendar.list_today",
          "description": "...",
          "input_schema": {"type": "object", "properties": {...}, "required": []},
      }

      async def run(input: dict) -> str:
          ...

  When present, registered as a callable Tool. Failures (import error, bad
  TOOL_SPEC, non-async ``run``) are logged and the markdown still injects
  — markdown-only skills enact protocol via bash / read_file / write_file.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from claw.tools.base import Tool

log = logging.getLogger("claw.skills")


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    skill_md_body: str
    tool: Tool | None


def load_skills(workspace_dir: Path) -> list[Skill]:
    """Scan ``<workspace>/skills/<name>/`` and return a list of Skill objects."""
    skills_root = workspace_dir / "skills"
    if not skills_root.is_dir():
        return []
    out: list[Skill] = []
    for sd in sorted(skills_root.iterdir()):
        if not sd.is_dir():
            continue
        skill_md = sd / "SKILL.md"
        if not skill_md.is_file():
            continue
        try:
            frontmatter, body = _parse_frontmatter(skill_md.read_text())
        except Exception:
            log.exception("failed to parse %s; skipping skill", skill_md)
            continue
        name = frontmatter.get("name") or sd.name
        description = frontmatter.get("description") or ""

        tool: Tool | None = None
        tool_py = sd / "tool.py"
        if tool_py.is_file():
            try:
                tool = _load_tool_py(tool_py, name, description)
            except Exception:
                log.exception(
                    "failed to load tool.py for skill %s; injecting markdown only",
                    name,
                )

        out.append(Skill(
            name=name,
            description=description,
            skill_md_body=body,
            tool=tool,
        ))
    return out


def render_skill_catalog(skills: list[Skill]) -> str:
    """Render the skills as one ``<skill_catalog>...</skill_catalog>`` block
    for injection into the system prompt. Empty string if no skills.

    Only the skill ``name`` and ``description`` (from frontmatter) are
    rendered — the SKILL.md body is NOT injected. The agent is expected
    to ``read_file`` the relevant ``skills/<name>/SKILL.md`` on demand
    when it decides to use a skill. This keeps the per-request system
    prompt small (an 8-skill catalog drops from ~30 KB to ~1 KB).
    """
    if not skills:
        return ""
    parts = [
        "<skill_catalog>",
        "Available skills (read skills/<name>/SKILL.md for full instructions when invoking one):",
    ]
    for s in skills:
        desc = s.description.strip() if s.description else "(no description)"
        parts.append(f"- {s.name}: {desc}")
    parts.append("</skill_catalog>")
    return "\n".join(parts)


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter between leading ``---`` markers.

    Returns ``({}, text)`` if no frontmatter is present.
    """
    if not text.startswith("---\n") and not text.startswith("---\r\n"):
        return {}, text
    sep = "\n---\n"
    end_idx = text.find(sep, 4)
    if end_idx == -1:
        return {}, text
    fm_text = text[4:end_idx]
    body = text[end_idx + len(sep):]
    frontmatter = yaml.safe_load(fm_text) or {}
    if not isinstance(frontmatter, dict):
        return {}, text
    return frontmatter, body


def _load_tool_py(path: Path, default_name: str, default_description: str) -> Tool:
    spec = importlib.util.spec_from_file_location(f"claw.skill.{default_name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    tool_spec = getattr(mod, "TOOL_SPEC", None)
    run_fn = getattr(mod, "run", None)
    if not isinstance(tool_spec, dict):
        raise ValueError(f"{path}: TOOL_SPEC must be a dict")
    # inspect, not asyncio: asyncio.iscoroutinefunction is deprecated and
    # removed in Python 3.16. Equivalent here — the only behavioural
    # difference was @asyncio.coroutine, gone since 3.11, and both unwrap
    # functools.partial. This line never warned because it only runs for a
    # skill shipping a tool.py, and every current skill is markdown-only.
    if not inspect.iscoroutinefunction(run_fn):
        raise ValueError(f"{path}: run must be defined as `async def run(input: dict) -> str`")

    return Tool(
        name=tool_spec.get("name", default_name),
        description=tool_spec.get("description", default_description),
        input_schema=tool_spec.get(
            "input_schema",
            {"type": "object", "properties": {}, "required": []},
        ),
        run=run_fn,
    )
