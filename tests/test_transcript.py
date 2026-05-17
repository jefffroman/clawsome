"""transcript.py — pure helpers + TranscriptStore round-trip.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

import time

from claw.transcript import (
    as_message,
    estimate_tokens,
    is_archived_transcript,
    parse_iso,
    session_id,
)


# --- pure ------------------------------------------------------------------

def test_session_id_sanitizes_unsafe_chars():
    # Anything outside [A-Za-z0-9_-] becomes "_".
    assert session_id("matrix", "!Room:srv") == "matrix__Room_srv"
    assert session_id("cron", 12345) == "cron_12345"


def test_as_message_strips_ts_only():
    row = {"role": "user", "content": "x", "ts": "2026-01-01T00:00:00+00:00"}
    assert as_message(row) == {"role": "user", "content": "x"}


def test_estimate_tokens_chars_over_4():
    rows = [
        {"role": "user", "content": "x" * 40},        # 40 chars
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "bash", "arguments": '{"a":1}'}},
        ]},  # name(4) + args(7) = 11 chars
    ]
    # (40 + 11) // 4 == 12
    assert estimate_tokens(rows) == (40 + len("bash") + len('{"a":1}')) // 4
    assert estimate_tokens([]) == 0


def test_parse_iso_valid_and_fallback():
    assert parse_iso("2026-05-03T07:30:00+00:00") == \
        __import__("datetime").datetime.fromisoformat(
            "2026-05-03T07:30:00+00:00").timestamp()
    before = time.time()
    fb = parse_iso("not-a-date")
    assert isinstance(fb, float) and fb >= before


def test_is_archived_transcript():
    assert is_archived_transcript("s.recap-2026-05-03T07-30-00.jsonl") is True
    assert is_archived_transcript("s.reset.jsonl") is True
    assert is_archived_transcript("s.jsonl") is False


# --- store (tmp_path fixture) ---------------------------------------------

def test_store_append_load_roundtrip_stamps_ts(transcripts):
    sid = "matrix_room"
    row = transcripts.append(sid, {"role": "user", "content": "hi"})
    assert row["content"] == "hi" and "ts" in row
    loaded = transcripts.load(sid)
    assert loaded == [row]


def test_store_replace_and_missing_load(transcripts):
    assert transcripts.load("nope") == []
    sid = "s"
    transcripts.append(sid, {"role": "user", "content": "a"})
    transcripts.replace(sid, [{"role": "user", "content": "b", "ts": "T"}])
    assert transcripts.load(sid) == [{"role": "user", "content": "b", "ts": "T"}]


def test_store_archive_moves_file_and_drops_flush_sidecar(transcripts):
    sid = "s"
    transcripts.append(sid, {"role": "user", "content": "a"})
    transcripts.write_flush_state(sid, 1)
    assert transcripts.read_flush_state(sid) == 1
    archived = transcripts.archive(sid, "reset")
    assert archived is not None and archived.exists()
    assert transcripts.load(sid) == []          # original gone
    assert transcripts.read_flush_state(sid) == 0  # sidecar dropped
    assert transcripts.archive("missing", "reset") is None


def test_store_last_ts(transcripts):
    sid = "s"
    assert transcripts.last_ts(sid) is None
    transcripts.append(sid, {"role": "user", "content": "a"}, ts="2026-01-01T00:00:00+00:00")
    assert transcripts.last_ts(sid) == "2026-01-01T00:00:00+00:00"
