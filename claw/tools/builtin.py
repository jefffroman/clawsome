"""Built-in tools: bash, read_file, write_file, list_dir.

Per plan: no UID isolation. Bash runs in-process as the gateway user (claw).
OS-level isolation comes from claw being a non-admin standard user.
File tools enforce path scoping to the agent's workspace.
"""

from __future__ import annotations

import asyncio
import functools
import os
import subprocess
from pathlib import Path
from typing import Any

from claw.tools.base import Tool


class PathScopeError(ValueError):
    pass


def _resolve_scoped(workspace_dir: Path, rel_path: str) -> Path:
    if not rel_path or not rel_path.strip():
        raise PathScopeError("path must not be empty")
    if os.path.isabs(rel_path):
        candidate = Path(rel_path).resolve()
    else:
        candidate = (workspace_dir / rel_path).resolve()
    workspace_real = workspace_dir.resolve()
    if candidate != workspace_real and workspace_real not in candidate.parents:
        raise PathScopeError(f"path {rel_path!r} escapes workspace {workspace_dir}")
    return candidate


async def _run_bash(workspace_dir: Path, args: dict[str, Any]) -> str:
    cmd = (args.get("command") or "").strip()
    if not cmd:
        return "error: command is required"
    timeout = max(1, min(int(args.get("timeout_s", 60)), 300))

    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(Path.home()),
    }

    def _runner() -> tuple[int, str, str]:
        proc = subprocess.run(
            ["/bin/bash", "-c", cmd],
            cwd=str(workspace_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr

    loop = asyncio.get_running_loop()
    try:
        rc, out, err = await loop.run_in_executor(None, _runner)
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {timeout}s"

    body: list[str] = []
    if out:
        body.append(out.rstrip())
    if err:
        body.append(f"[stderr]\n{err.rstrip()}")
    body.append(f"[exit_code] {rc}")
    return "\n".join(body)


async def _run_read_file(workspace_dir: Path, args: dict[str, Any]) -> str:
    rel = args.get("path", "")
    try:
        path = _resolve_scoped(workspace_dir, rel)
    except PathScopeError as e:
        return f"error: {e}"
    if not path.exists():
        return f"error: not found: {rel}"
    if path.is_dir():
        return f"error: {rel} is a directory; use list_dir"
    try:
        return path.read_text()
    except UnicodeDecodeError:
        return f"error: {rel} is not a text file"
    except OSError as e:
        return f"error: {e}"


async def _run_write_file(workspace_dir: Path, args: dict[str, Any]) -> str:
    rel = args.get("path", "")
    content = args.get("content", "")
    if not isinstance(content, str):
        return "error: content must be a string"
    try:
        path = _resolve_scoped(workspace_dir, rel)
    except PathScopeError as e:
        return f"error: {e}"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)
    return f"wrote {len(content)} chars to {rel}"


async def _run_append_file(workspace_dir: Path, args: dict[str, Any]) -> str:
    rel = args.get("path", "")
    content = args.get("content", "")
    if not isinstance(content, str):
        return "error: content must be a string"
    try:
        path = _resolve_scoped(workspace_dir, rel)
    except PathScopeError as e:
        return f"error: {e}"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "a") as f:
            f.write(content)
            if not content.endswith("\n"):
                f.write("\n")
    except OSError as e:
        return f"error: {e}"
    return f"appended {len(content)} chars to {rel}"


async def _run_list_dir(workspace_dir: Path, args: dict[str, Any]) -> str:
    rel = args.get("path", ".")
    try:
        path = _resolve_scoped(workspace_dir, rel)
    except PathScopeError as e:
        return f"error: {e}"
    if not path.is_dir():
        return f"error: not a directory: {rel}"
    entries: list[str] = []
    for child in sorted(path.iterdir()):
        if child.is_dir():
            entries.append(f"{child.name}/")
        else:
            try:
                size = child.stat().st_size
                entries.append(f"{child.name} ({size} bytes)")
            except OSError:
                entries.append(child.name)
    return "\n".join(entries) if entries else "(empty)"


def build_builtin_tools(workspace_dir: Path) -> dict[str, Tool]:
    tools = [
        Tool(
            name="bash",
            description=(
                "Run a shell command inside the agent's workspace. "
                "stdout, stderr, and exit code are returned."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."},
                    "timeout_s": {"type": "integer", "description": "Timeout in seconds (1-300; default 60)."},
                },
                "required": ["command"],
            },
            run=functools.partial(_run_bash, workspace_dir),
        ),
        Tool(
            name="read_file",
            description=(
                "Read a UTF-8 text file from the agent's workspace. Path is "
                "relative to workspace; absolute paths must resolve inside it."
            ),
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            run=functools.partial(_run_read_file, workspace_dir),
        ),
        Tool(
            name="write_file",
            description=(
                "Atomically write content to a path inside the agent's workspace. "
                "Creates parent directories. Overwrites existing files."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            run=functools.partial(_run_write_file, workspace_dir),
        ),
        Tool(
            name="append_file",
            description=(
                "Atomically append content to a file inside the agent's "
                "workspace, creating the file if it doesn't exist. Adds a "
                "trailing newline if the content doesn't end with one. "
                "Prefer this over read_file+write_file when you just want "
                "to add new content at the end of an existing file — it's "
                "one tool call instead of two and preserves prior content "
                "by default."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            run=functools.partial(_run_append_file, workspace_dir),
        ),
        Tool(
            name="list_dir",
            description="List the entries of a directory inside the agent's workspace.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory path; defaults to '.'"}},
                "required": [],
            },
            run=functools.partial(_run_list_dir, workspace_dir),
        ),
    ]
    return {t.name: t for t in tools}
