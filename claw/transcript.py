"""JSONL append-only transcript store keyed by session ID.

Each line is one Ollama-shaped message plus a ts field, e.g.

    {"role": "user", "content": "...", "ts": "<ISO>"}
    {"role": "assistant", "content": "...", "ts": "<ISO>"}
    {"role": "assistant", "content": "", "tool_calls": [...], "ts": "<ISO>"}
    {"role": "tool", "tool_call_id": "...", "content": "...", "ts": "<ISO>"}

Strip ``ts`` via ``as_message()`` before sending back to Ollama.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9_-]")


log = logging.getLogger("claw.transcript")


def sid_for_key(session_key: str) -> str:
    """Canonical sid: a filename-safe form of a message's ``session_key``.

    Idempotent for already-safe keys (e.g. ``"home"``). This is the single
    source of truth for the transcript key; ``session_id`` is the legacy
    ``(channel, peer_id)`` convenience that builds the default key.
    """
    return _FILENAME_SAFE.sub("_", session_key)


def session_id(channel: str, peer_id: str | int) -> str:
    return sid_for_key(f"{channel}_{peer_id}")


def _path(transcripts_dir: Path, sid: str) -> Path:
    return transcripts_dir / f"{sid}.jsonl"


def _flush_state_path(transcripts_dir: Path, sid: str) -> Path:
    return transcripts_dir / f"{sid}.flush.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def purge_scratch_sessions(scratch_dir: Path, agent_id: str = "") -> int:
    """Delete every subagent scratch session under ``scratch_dir``.

    Subagent sessions are reachable only through a live handle, and handles do
    not survive a gateway restart — so at boot everything here is unreachable
    by construction. Purging is what stops a post-restart spawn inheriting a
    dead task's context, and what stops the directory growing without bound.

    Returns the number removed. Non-.jsonl entries are left alone.
    """
    if not scratch_dir.is_dir():
        return 0
    removed = 0
    for entry in scratch_dir.iterdir():
        if entry.suffix != ".jsonl":
            continue
        try:
            entry.unlink()
            removed += 1
        except OSError:
            log.exception("[%s] could not purge %s", agent_id, entry)
    if removed:
        log.info(
            "[%s] purged %d orphaned subagent session(s) from %s",
            agent_id, removed, scratch_dir,
        )
    return removed


class TranscriptStore:
    def __init__(self, transcripts_dir: Path | str) -> None:
        self.dir = Path(transcripts_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def load(self, sid: str) -> list[dict[str, Any]]:
        path = _path(self.dir, sid)
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(obj)
        return rows

    def append(
        self,
        sid: str,
        message: dict[str, Any],
        ts: str | None = None,
    ) -> dict[str, Any]:
        row = {**message, "ts": ts or _now_iso()}
        path = _path(self.dir, sid)
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    def replace(self, sid: str, rows: list[dict[str, Any]]) -> None:
        path = _path(self.dir, sid)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, path)

    def archive(self, sid: str, suffix: str) -> Path | None:
        path = _path(self.dir, sid)
        if not path.exists():
            return None
        archived = self.dir / f"{sid}.{suffix}.jsonl"
        os.replace(path, archived)
        # Drop the flush-state sidecar — a new transcript at this sid starts
        # at row 0, and the next flush should slice from the start.
        flush_state = _flush_state_path(self.dir, sid)
        try:
            flush_state.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        return archived

    def read_flush_state(self, sid: str) -> int:
        """Return the row_count recorded by the previous successful memory
        flush, or 0 if no sidecar exists / file is unreadable / count is
        missing. Sidecar lives at ``<sid>.flush.json`` alongside the
        transcript.
        """
        path = _flush_state_path(self.dir, sid)
        if not path.exists():
            return 0
        try:
            with open(path) as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError):
            return 0
        rc = obj.get("row_count")
        return rc if isinstance(rc, int) and rc >= 0 else 0

    def write_flush_state(self, sid: str, row_count: int) -> None:
        """Atomically persist the row_count of the most recent successful
        flush. Atomic rewrite via tmp+rename mirrors ``replace()``.
        """
        path = _flush_state_path(self.dir, sid)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "row_count": int(row_count),
            "last_flushed_at": _now_iso(),
        }
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)

    def last_ts(self, sid: str) -> str | None:
        path = _path(self.dir, sid)
        if not path.exists():
            return None
        try:
            size = path.stat().st_size
            if size == 0:
                return None
            with open(path, "rb") as f:
                read_from = max(0, size - 4096)
                f.seek(read_from)
                tail = f.read().decode("utf-8", errors="replace")
            for line in reversed(tail.splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    return obj.get("ts")
                except json.JSONDecodeError:
                    continue
        except OSError:
            return None
        return None


def as_message(row: dict[str, Any]) -> dict[str, Any]:
    """Strip transcript-only fields so the row can go straight to Ollama."""
    return {k: v for k, v in row.items() if k != "ts"}


def estimate_tokens(rows: list[dict[str, Any]]) -> int:
    """Naive ~chars/4 estimator. Counts content + tool args + tool results."""
    chars = 0
    for r in rows:
        content = r.get("content")
        if isinstance(content, str):
            chars += len(content)
        for tc in r.get("tool_calls") or []:
            fn = tc.get("function", {})
            chars += len(fn.get("name", ""))
            args = fn.get("arguments", "")
            chars += len(args) if isinstance(args, str) else len(json.dumps(args))
    return chars // 4


def parse_iso(ts: str) -> float:
    try:
        return datetime.fromisoformat(ts).timestamp()
    except (TypeError, ValueError):
        return time.time()


def is_archived_transcript(filename: str) -> bool:
    return ".recap-" in filename or ".reset" in filename
