"""Built-in tools: bash, read_file, write_file, list_dir.

Per plan: no UID isolation. Bash runs in-process as the gateway user (claw).
OS-level isolation comes from claw being a non-admin standard user.
File tools enforce path scoping to the agent's workspace.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any

from claw.runctx import current_task_id, current_turn_id
from claw.tools.spool import bound_result
from claw.tools.base import Tool

log = logging.getLogger("claw.tools.bash")


# --- bash subprocess registry --------------------------------------------
# Each bash command runs as its own session leader (start_new_session=True)
# so a single killpg takes the whole tree (bash + anything it spawned). We
# track every live process group keyed by pgid, tagged with the (turn_id,
# task_id) that owned the spawn, so %stop can kill exactly the right scope
# after cancelling a turn (cancelling the await abandons the executor
# thread but never the OS processes). There is deliberately NO session
# scope here — every %stop kill is bounded to a single turn or one
# subagent subtree within it:
#   - whole turn   -> kill_turn_bash(turn_id)      (main + the turn's cascade)
#   - one subagent -> kill_subagent_bash(task_ids) (that subtree only)
#
# Mutated from the event loop (registration, before offload) and worker
# threads (deregistration in the runner's finally) — hence the lock.
_BashCtx = tuple[str, str]  # (turn_id, task_id)
_BASH: dict[int, _BashCtx] = {}  # pgid -> ctx
_BASH_LOCK = threading.Lock()


def _bash_register(pgid: int, turn_id: str, task_id: str) -> None:
    with _BASH_LOCK:
        _BASH[pgid] = (turn_id, task_id)


def _bash_unregister(pgid: int) -> None:
    with _BASH_LOCK:
        _BASH.pop(pgid, None)


def _killpg_all(pgids: list[int], scope: str) -> int:
    """SIGKILL each process group; return the count signalled.

    A process exiting and being deregistered are not atomic, so a
    snapshotted pgid may already be gone. ``ProcessLookupError`` ("no such
    process", ESRCH) is therefore a *silent success* — the goal is "that
    tree is not running", and it isn't. Only an unexpected OSError (e.g.
    EPERM) is worth logging.
    """
    killed = 0
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
            killed += 1
        except ProcessLookupError:
            # Already gone between snapshot and kill — exactly the
            # end→deregister gap; treat as done, no noise.
            pass
        except OSError:
            log.exception("killpg(%s) failed (%s)", pgid, scope)
    return killed


def kill_turn_bash(turn_id: str) -> int:
    """SIGKILL every live bash process group whose spawn belonged to
    ``turn_id`` — the turn's own bash plus its entire subagent cascade
    (the turn id is inherited transitively into spawned tasks)."""
    if not turn_id:
        return 0
    with _BASH_LOCK:
        pgids = [p for p, (t, _) in _BASH.items() if t == turn_id]
    return _killpg_all(pgids, f"turn={turn_id}")


def kill_subagent_bash(task_ids: set[str]) -> int:
    """SIGKILL live bash process groups whose owning subagent is in
    ``task_ids`` (a targeted subtree). Empty/blank ids never match."""
    wanted = {t for t in task_ids if t}
    if not wanted:
        return 0
    with _BASH_LOCK:
        pgids = [p for p, (_, k) in _BASH.items() if k in wanted]
    return _killpg_all(pgids, f"tasks={sorted(wanted)}")


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
    timeout = max(1, min(int(args.get("timeout_s", 60)), 3600))

    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(Path.home()),
    }
    # Captured on the event loop (ContextVars aren't visible from the
    # worker thread); passed into _runner so the spawn is attributable to
    # the turn / subagent that owns it for %stop's scoped kill.
    turn_id = current_turn_id.get()
    task_id = current_task_id.get()

    def _runner() -> tuple[int, str, str]:
        # start_new_session=True → child is its own session/process-group
        # leader, so killpg(pgid) reaps bash AND anything it spawned. Popen
        # (not subprocess.run) so we can register the pgid *before* waiting.
        proc = subprocess.Popen(
            ["/bin/bash", "-c", cmd],
            cwd=str(workspace_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            # Exited before we could read its pgid — nothing to track;
            # fall back to pid-as-pgid (session leader => pgid == pid).
            pgid = proc.pid
        _bash_register(pgid, turn_id, task_id)
        try:
            try:
                out, err = proc.communicate(timeout=timeout)
                return proc.returncode, out, err
            except subprocess.TimeoutExpired:
                # Kill the whole tree on timeout (subprocess.run only
                # killed the immediate child; grandchildren leaked).
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    log.exception("killpg(%s) on timeout failed", pgid)
                proc.communicate()  # reap
                raise
        finally:
            _bash_unregister(pgid)

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
    # Bounded at write time — a verbose command would otherwise sit in the
    # transcript for the rest of the session. bound_result keeps a tail, so
    # the [exit_code] line above survives truncation.
    return bound_result(
        "\n".join(body), workspace_dir=workspace_dir, tool="bash",
    )


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
                "stdout, stderr, and exit code are returned. "
                "Default timeout is 60s; pass timeout_s explicitly for longer commands (max 3600s). "
                "For long-running work (builds, large clones, long test runs), prefer delegating "
                "to a subagent via spawn_subagent rather than blocking your own turn."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."},
                    "timeout_s": {"type": "integer", "description": "Timeout in seconds (1-3600; default 60)."},
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
                "Atomically write a file inside the agent's workspace, creating "
                "parent directories. Replaces the whole file — you must supply "
                "its complete new contents, so cost scales with total file size. "
                "For adding to the end of an existing (especially large) file, "
                "use append_file instead."
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
                "workspace, creating it if absent. Adds a trailing newline if "
                "the content doesn't end with one. Strongly prefer this over "
                "read_file+write_file for adding to the end of a file — "
                "especially a large or append-only one (logs, archives): append "
                "cost is fixed regardless of file size, whereas write_file must "
                "re-emit the file's entire contents. Preserves prior content."
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
