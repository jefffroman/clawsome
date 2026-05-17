"""compaction.py — pure predicates + split logic.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

from claw.compaction import (
    _human_delta,
    _split_for_compaction,
    will_compact,
    will_mid_session_compact,
)
from claw.config import CompactionConfig


def test_human_delta_cutoffs():
    assert _human_delta(30) == "30s ago"
    assert _human_delta(-5) == "0s ago"            # clamped to 0
    assert _human_delta(120) == "about 2 min ago"
    assert _human_delta(3600) == "about 1.0 h ago"
    assert _human_delta(3 * 3600) == "about 3.0 h ago"
    assert _human_delta(2 * 86400) == "about 2.0 days ago"


def test_split_short_transcript_is_all_older():
    rows = [{"role": "user", "content": "a"}] * 3
    assert _split_for_compaction(rows, 48000) == (rows, [])


def test_split_lands_on_user_boundary():
    big = "x" * 40  # estimate_tokens([row]) == 10
    rows = [
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
        {"role": "user", "content": big},
    ]
    older, newer = _split_for_compaction(rows, reserve_tokens=15)
    assert older == rows[:4]
    assert newer == rows[4:]
    assert newer[0]["role"] == "user"          # cut never slices mid-tool-call
    assert older + newer == rows


def test_will_mid_session_compact(make_cfg, tmp_path):
    cfg = make_cfg(tmp_path)  # default threshold 96000
    assert will_mid_session_compact(cfg, []) is False
    low = make_cfg(tmp_path, compaction=CompactionConfig(
        mid_session_token_threshold=10, reserve_tokens=5))
    big_rows = [{"role": "user", "content": "x" * 100}]  # 25 tokens > 10
    assert will_mid_session_compact(low, big_rows) is True


def test_will_compact_gates(make_cfg, tmp_path):
    cfg = make_cfg(tmp_path)
    # Below threshold, no force -> the predicate gate short-circuits.
    assert will_compact(cfg, [], force=False) is False
    # force=True but empty rows -> _split returns ([], []) -> no older -> False.
    assert will_compact(cfg, [], force=True) is False
    # force=True with non-empty rows: n<4 -> _split returns (rows, []) so
    # `older` is truthy -> True. (The plan's "force/short -> False" was wrong;
    # this asserts the actual source behavior.)
    assert will_compact(cfg, [{"role": "user", "content": "hi"}], force=True) is True
    low = make_cfg(tmp_path, compaction=CompactionConfig(
        mid_session_token_threshold=10, reserve_tokens=5))
    assert will_compact(low, [{"role": "user", "content": "x" * 100}]) is True
