"""compaction.py — pure predicates + split logic.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

from claw.compaction import (
    _human_delta,
    _split_for_compaction,
    compaction_preview,
    will_compact,
    will_mid_session_compact,
)
from claw.transcript import estimate_tokens
from claw.config import CompactionConfig


def test_human_delta_cutoffs():
    assert _human_delta(30) == "30s ago"
    assert _human_delta(-5) == "0s ago"            # clamped to 0
    assert _human_delta(120) == "about 2 min ago"
    assert _human_delta(3600) == "about 1.0 h ago"
    assert _human_delta(3 * 3600) == "about 3.0 h ago"
    assert _human_delta(2 * 86400) == "about 2.0 days ago"


def test_split_short_transcript_compacts_nothing():
    """A transcript too small to hold an interior cut has nothing to
    compact. This asserted the opposite — (rows, []), i.e. summarize
    everything and keep nothing — until 2026-08-08. That contradicted
    _split_for_compaction's own contract and its behaviour one row later
    (n=4 on the same content returns ([], rows)), and %compact reached it
    with force=True, so a short session could be replaced wholesale by its
    own summary."""
    rows = [{"role": "user", "content": "a"}] * 3
    assert _split_for_compaction(rows, 48000) == ([], rows)


def test_split_is_continuous_across_the_old_n4_boundary():
    """Small n must agree with the general path, not invert it. Rows that fit
    inside the reserve compact nothing, at every length."""
    row = {"role": "user", "content": "a"}
    for n in (1, 2, 3, 4, 5):
        older, newer = _split_for_compaction([dict(row)] * n, 48000)
        assert older == [], f"n={n} should compact nothing"
        assert len(newer) == n


def test_small_transcript_over_reserve_still_splits():
    """Removing the n<4 early-out gains this: a short transcript that really
    does exceed the reserve gets a genuine split, where before it was
    summarize-everything-keep-nothing."""
    rows = [
        {"role": "user", "content": "x" * 400},       # 100 tokens
        {"role": "assistant", "content": "y" * 400},  # 100 tokens
    ]
    older, newer = _split_for_compaction(rows, reserve_tokens=100, min_older_tokens=0)
    assert older and newer
    assert older + newer == rows


def test_tiny_older_slice_is_not_worth_compacting():
    """The reserve floors the whole TRANSCRIPT; it does not floor `older`,
    which is what actually gets summarized. Because the cut lands *nearest*
    the reserve, a transcript barely over it yields a tiny older slice —
    measured: 48,200 tokens against a 48,000 reserve gives older=200, which
    the 300-word mid-session floor would 'compress' into ~399 tokens. That
    grows the transcript and destroys the rows to do it.

    Unreachable automatically (the 96,000 trigger guarantees a large older),
    but %compact passes force=True and skips that trigger.
    """
    rows = [{"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 400}
            for i in range(482)]                      # 100 tokens each = 48,200
    assert estimate_tokens(rows) == 48_200
    older, newer = _split_for_compaction(rows, reserve_tokens=48_000)
    assert older == []                                 # refused
    assert newer == rows                               # nothing destroyed
    # Same transcript, a reserve that leaves a worthwhile older slice.
    older, newer = _split_for_compaction(rows, reserve_tokens=40_000)
    assert estimate_tokens(older) >= 1_596


def test_split_partitions_rows_exactly():
    big = "x" * 40  # estimate_tokens([row]) == 10
    rows = [
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
        {"role": "user", "content": big},
    ]
    older, newer = _split_for_compaction(rows, reserve_tokens=15, min_older_tokens=0)
    assert older + newer == rows
    assert older and newer
    # Nearest to a 15-token reserve, and no tool_calls anywhere, so every
    # index is a legal cut. (This test asserted a *user* boundary until
    # 2026-08-08; that rule was replaced by tool-atomicity — see
    # test_split_never_orphans_a_tool_result.)
    assert estimate_tokens(newer) == 20


def test_will_mid_session_compact(make_cfg, tmp_path):
    cfg = make_cfg(tmp_path)  # default threshold 96000
    assert will_mid_session_compact(cfg, []) is False
    low = make_cfg(tmp_path, compaction=CompactionConfig(
        mid_session_token_threshold=10, reserve_tokens=5))
    big_rows = [{"role": "user", "content": "x" * 100}]  # 25 tokens > 10
    assert will_mid_session_compact(low, big_rows) is True


def test_will_mid_session_compact_counts_overhead(make_cfg, tmp_path):
    # Transcript alone is under threshold; the system-prompt overhead is what
    # pushes the real prompt over. Without overhead accounting the trigger
    # would never fire (this is the 2026-07-23 undercount bug).
    cfg = make_cfg(tmp_path, compaction=CompactionConfig(
        mid_session_token_threshold=100, reserve_tokens=5))
    small_rows = [{"role": "user", "content": "x" * 200}]  # 50 tokens < 100
    assert will_mid_session_compact(cfg, small_rows) is False
    assert will_mid_session_compact(cfg, small_rows, overhead_tokens=0) is False
    # + 60 tokens of system/memory overhead -> 110 > 100 -> fires.
    assert will_mid_session_compact(cfg, small_rows, overhead_tokens=60) is True


def test_will_compact_gates(make_cfg, tmp_path):
    cfg = make_cfg(tmp_path)
    # Below threshold, no force -> the predicate gate short-circuits.
    assert will_compact(cfg, [], force=False) is False
    # force=True but empty rows -> _split returns ([], []) -> no older -> False.
    assert will_compact(cfg, [], force=True) is False
    # force=True with one short row -> nothing to compact. This asserted True
    # until 2026-08-08, on the strength of the n<4 early-out returning
    # (rows, []); the comment here even noted the plan had said False and
    # called the plan wrong. The plan was right — that path summarized a
    # whole short session and kept none of it. See
    # test_split_short_transcript_compacts_nothing.
    assert will_compact(cfg, [{"role": "user", "content": "hi"}], force=True) is False
    low = make_cfg(tmp_path, compaction=CompactionConfig(
        mid_session_token_threshold=10, reserve_tokens=5))
    # One row can't be split at all — there is no interior cut point.
    assert will_compact(low, [{"role": "user", "content": "x" * 100}]) is False
    # Two rows clear the reserve and a cut exists — but `older` would be 25
    # tokens, which the ~300-word recap floor cannot compress. Clearing the
    # reserve is not the same as being worth compacting: the cut lands
    # NEAREST the reserve, so a transcript barely over it yields a tiny
    # older slice. See test_tiny_older_slice_is_not_worth_compacting.
    assert will_compact(low, [
        {"role": "user", "content": "x" * 100},
        {"role": "assistant", "content": "y" * 100},
    ]) is False
    # Large enough for the swap to actually free something -> True.
    assert will_compact(low, [
        {"role": "user", "content": "x" * 4000},       # 1000 tokens
        {"role": "assistant", "content": "y" * 4000},  # 1000 tokens
        {"role": "user", "content": "z" * 4000},       # 1000 tokens
    ]) is True


# --- cut placement: tool atomicity, nearest reserve --------------------------

def _tok_rows(spec):
    """Rows from (role, n_tokens) pairs. estimate_tokens is chars//4."""
    return [{"role": role, "content": "x" * (n * 4)} for role, n in spec]


def _cascade(spec):
    """Rows with real tool-call structure.

    ``("calls", tok, k)`` is an assistant issuing k tool_calls; ``("tool",
    tok)`` is one result. Function names are empty so they contribute no
    tokens and the arithmetic in each test stays exact.
    """
    out = []
    for item in spec:
        kind, tok = item[0], item[1]
        content = "x" * (tok * 4)
        if kind == "calls":
            out.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {"id": f"c{len(out)}-{k}", "function": {"name": "", "arguments": ""}}
                    for k in range(item[2])
                ],
            })
        elif kind == "tool":
            out.append({"role": "tool", "content": content, "tool_call_id": "c"})
        else:
            out.append({"role": kind, "content": content})
    return out


def test_cut_can_land_between_tool_rounds_inside_one_turn():
    """The capability the relaxed rule buys.

    One user turn, two completed tool rounds. The only ``user`` row is index
    0, so under the old rule there was no interior boundary at all and the
    whole 25-token turn was one atom. Now the gap between round 1 and round 2
    is a legal cut.
    """
    rows = _cascade([
        ("user", 100),          # 0
        ("calls", 5, 1),        # 1  round 1 opens
        ("tool", 5),            # 2  round 1 closes
        ("calls", 5, 1),        # 3  round 2 opens   <- cut lands here
        ("tool", 5),            # 4  round 2 closes
        ("assistant", 5),       # 5
    ])
    older, newer = _split_for_compaction(rows, reserve_tokens=15, min_older_tokens=0)
    assert older == rows[:3]
    assert newer[0]["role"] == "assistant"      # resumes mid-turn, not at a user row
    assert estimate_tokens(newer) == 15


def test_cut_never_orphans_a_tool_result():
    """The invariant the boundary rule exists to protect, swept across every
    reserve. A ``tool`` row must never be preserved without the assistant row
    that requested it."""
    rows = _cascade([
        ("user", 40),
        ("calls", 3, 2), ("tool", 12), ("tool", 9),
        ("assistant", 6),
        ("calls", 2, 1), ("tool", 20),
        ("user", 15),
        ("calls", 4, 3), ("tool", 7), ("tool", 5), ("tool", 11),
        ("assistant", 8),
    ])
    for reserve in range(1, 160):
        older, newer = _split_for_compaction(rows, reserve_tokens=reserve, min_older_tokens=0)
        assert older + newer == rows
        if not older:
            continue
        pending = 0
        for r in newer:
            if r["role"] == "assistant":
                pending += len(r.get("tool_calls") or [])
            elif r["role"] == "tool":
                assert pending > 0, f"orphaned tool result at reserve={reserve}"
                pending -= 1


def test_cut_lands_nearest_the_reserve_in_either_direction():
    """Nearest, not a fixed direction — otherwise the reserve is
    systematically over- or under-shot rather than merely approximate."""
    rows = _cascade([
        ("user", 100),      # 0   cutting here would compact nothing (excluded)
        ("calls", 30, 1),   # 1   legal, keeps 54
        ("tool", 4),        # 2   ILLEGAL — round still open
        ("assistant", 20),  # 3   legal, keeps 20
    ])
    # 54 is nearest -> the cut overshoots the reserve
    _older, newer = _split_for_compaction(rows, reserve_tokens=50, min_older_tokens=0)
    assert estimate_tokens(newer) == 54
    # 20 is nearest -> the cut undershoots it
    _older, newer = _split_for_compaction(rows, reserve_tokens=25, min_older_tokens=0)
    assert estimate_tokens(newer) == 20
    # index 2 sits closest to a tiny reserve but is mid-round, so never chosen
    _older, newer = _split_for_compaction(rows, reserve_tokens=5, min_older_tokens=0)
    assert newer[0]["role"] != "tool"


def test_split_nothing_to_compact_when_transcript_fits_reserve():
    rows = _tok_rows([
        ("user", 10), ("assistant", 10), ("user", 10), ("assistant", 10),
    ])
    older, newer = _split_for_compaction(rows, reserve_tokens=10_000, min_older_tokens=0)
    assert older == []
    assert newer == rows


def test_split_fits_reserve_check_does_not_depend_on_first_row_role():
    """The 'everything fits' case is now explicit. It used to fall out of the
    boundary search, which compacted a prefix when row 0 wasn't a user row."""
    rows = _tok_rows([
        ("assistant", 10), ("tool", 10), ("user", 10), ("assistant", 10),
    ])
    older, newer = _split_for_compaction(rows, reserve_tokens=10_000, min_older_tokens=0)
    assert older == []
    assert newer == rows


# --- %context reports the operation, not just the budget --------------------

def test_compaction_preview_reports_nothing_below_reserve(make_cfg, tmp_path):
    cfg = make_cfg(tmp_path)
    rows = _tok_rows([("user", 10), ("assistant", 10)] * 4)
    older_t, newer_t = compaction_preview(cfg, rows)
    assert older_t == 0
    assert newer_t == estimate_tokens(rows)


def test_compaction_preview_matches_actual_split(make_cfg, tmp_path):
    cfg = make_cfg(tmp_path)
    reserve = cfg.compaction.reserve_tokens
    # Comfortably larger than the reserve, with frequent user boundaries.
    rows = _tok_rows([("user", 500), ("assistant", 500)] * 200)
    older_t, newer_t = compaction_preview(cfg, rows)
    older, newer = _split_for_compaction(rows, reserve, min_older_tokens=0)
    assert (older_t, newer_t) == (estimate_tokens(older), estimate_tokens(newer))
    assert older_t > 0
    # And the preview is a faithful account of the whole transcript.
    assert older_t + newer_t == estimate_tokens(rows)
