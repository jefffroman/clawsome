"""Render the per-agent ``extra_paths`` files (SOUL.md, AGENTS.md, etc.)
into one block to inject into the system prompt.
"""

from __future__ import annotations

from pathlib import Path


def render_extra_paths(workspace_dir: Path, extra_paths: tuple[str, ...]) -> str:
    parts: list[str] = []
    for rel in extra_paths:
        p = workspace_dir / rel
        if not p.is_file():
            continue
        try:
            content = p.read_text()
        except OSError:
            continue
        body = content.rstrip()
        if body:
            parts.append(f"## {rel}\n\n{body}")
    return "\n\n".join(parts)
