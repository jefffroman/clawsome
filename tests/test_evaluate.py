"""The retrieval evaluation report. Pure scoring over labels — no index, no
model — so the arithmetic every tuning decision rests on is pinned.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

from claw.evaluate import report

# Query 0 has two relevant memories, one of them clearly so; query 1 has none.
GRADES = {
    "0": {"a": 2, "b": 1, "c": 0, "d": 0},
    "1": {"e": 0, "f": 0},
}


def test_a_perfect_arm():
    r = report([["a", "b"], []], GRADES)
    assert r["precision"] == 1.0
    assert r["recall_clear"] == 1.0
    assert r["hit_when_relevant"] == 1.0
    assert r["quiet_when_none"] == 1.0


def test_injecting_nothing_scores_no_precision_but_stays_quiet():
    """The degenerate arm: perfect on the turn with nothing to say, and it
    found none of what was there. Precision alone would not show that."""
    r = report([[], []], GRADES)
    assert r["per_query"] == 0.0
    assert r["recall_clear"] == 0.0
    assert r["hit_when_relevant"] == 0.0
    assert r["quiet_when_none"] == 1.0


def test_injecting_everything_finds_it_all_and_says_too_much():
    r = report([["a", "b", "c", "d"], ["e", "f"]], GRADES)
    assert r["per_query"] == 3.0
    assert r["precision"] == 2 / 6
    assert r["recall_clear"] == 1.0
    assert r["hit_when_relevant"] == 1.0
    assert r["quiet_when_none"] == 0.0, "it spoke on the turn that needed nothing"


def test_precision_rises_by_injecting_less():
    """Why precision is never read alone: dropping the weaker hit improves it
    while the turn is no better served."""
    both = report([["a", "b"], []], GRADES)
    one = report([["a"], []], GRADES)
    assert one["precision"] == both["precision"] == 1.0
    assert one["per_query"] < both["per_query"]
    assert one["recall_clear"] == both["recall_clear"] == 1.0


def test_an_ungraded_query_is_skipped_not_counted_as_empty():
    """A partial grading run must not read as an arm that returned nothing."""
    r = report([["a", "b"], [], ["x"]], GRADES)
    assert r["queries"] == 2


def test_a_wrong_answer_is_not_a_hit():
    r = report([["c", "d"], ["e"]], GRADES)
    assert r["precision"] == 0.0
    assert r["hit_when_relevant"] == 0.0
    assert r["quiet_when_none"] == 0.0
