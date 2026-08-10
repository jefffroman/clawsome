"""Tool registry per agent."""

from __future__ import annotations

from pathlib import Path

from claw.config import Config
from claw.tools.base import Tool, ollama_tool_spec
from claw.tools.builtin import build_builtin_tools
from claw.tools.web_search import build_web_search_tool

__all__ = ["Tool", "ollama_tool_spec", "build_registry"]


def build_registry(cfg: Config, workspace_dir: Path) -> dict[str, Tool]:
    tools = build_builtin_tools(workspace_dir)
    if cfg.searxng.base_url:
        ws = build_web_search_tool(cfg.searxng.base_url, workspace_dir)
        tools[ws.name] = ws
    return tools
