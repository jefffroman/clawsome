"""Per-agent tool registry — merges built-in tools with workspace skills."""

from __future__ import annotations

import logging
from pathlib import Path

from claw.config import Config
from claw.skills.loader import load_skills, render_skill_catalog
from claw.tools.base import Tool
from claw.tools.builtin import build_builtin_tools
from claw.tools.web_search import build_web_search_tool

log = logging.getLogger("claw.skills.registry")


def build_agent_registry(
    cfg: Config,
    workspace_dir: Path,
) -> tuple[dict[str, Tool], str]:
    """Return ``(tools_dict, skill_catalog_block)`` for one agent.

    ``tools_dict`` merges built-in tools (bash, read_file, write_file,
    list_dir, web_search) with any Python tools registered by skills under
    ``<workspace>/skills/``. Skill tools win on name collision (so a
    workspace can override the default ``web_search`` if it wants).

    ``skill_catalog_block`` is the markdown for prompt injection; empty
    string if no skills.
    """
    tools: dict[str, Tool] = build_builtin_tools(workspace_dir)
    if cfg.searxng.base_url:
        ws = build_web_search_tool(cfg.searxng.base_url)
        tools[ws.name] = ws

    skills = load_skills(workspace_dir)
    skill_tool_names: list[str] = []
    for skill in skills:
        if skill.tool is None:
            continue
        if skill.tool.name in tools:
            log.info(
                "[skills] %s overrides builtin tool %s",
                skill.name, skill.tool.name,
            )
        tools[skill.tool.name] = skill.tool
        skill_tool_names.append(skill.tool.name)

    if skills:
        log.info(
            "[skills] loaded %d skill(s) from %s (%d with python tools: %s)",
            len(skills), workspace_dir / "skills",
            len(skill_tool_names),
            ", ".join(skill_tool_names) or "(markdown-only)",
        )

    return tools, render_skill_catalog(skills)
