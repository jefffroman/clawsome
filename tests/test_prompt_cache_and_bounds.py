"""Prefix-cache stability of the prompt, and bounded incidental tool output.

Both behaviours here exist for the same reason: the KV cache is a *prefix*
cache, so anything that perturbs an early token forces a full re-prefill of
everything after it. See ``Agent._retrieval_row``.
"""

from __future__ import annotations

from claw.agent import Agent
from claw.tools.spool import (
    HEAD_CHARS,
    MAX_INLINE_CHARS,
    TAIL_CHARS,
    bound_result,
)


# --- retrieved memory rides at the tail, not in the system prompt ----------

def _history(n: int = 8) -> list[dict]:
    rows: list[dict] = []
    for i in range(n):
        rows.append({"role": "user", "content": f"question {i}"})
        rows.append({"role": "assistant", "content": f"answer {i}"})
    rows.append({"role": "user", "content": "the current question"})
    return rows


def test_retrieval_row_empty_is_none():
    assert Agent._retrieval_row("") is None


def test_retrieval_row_wraps_block():
    row = Agent._retrieval_row("some recalled fact")
    assert row["role"] == "user"
    assert "<retrieved_memory>" in row["content"]
    assert "some recalled fact" in row["content"]
    assert "</retrieved_memory>" in row["content"]


def test_with_retrieval_noop_when_empty():
    history = _history()
    assert Agent._with_retrieval(history, "") is history


def test_with_retrieval_keeps_user_message_last():
    history = _history()
    out = Agent._with_retrieval(history, "recalled")
    # The user's own text must remain the final message — the retrieval block
    # is spliced in front of it, not after.
    assert out[-1] == history[-1]
    assert "<retrieved_memory>" in out[-2]["content"]
    assert len(out) == len(history) + 1


def test_with_retrieval_on_empty_history():
    out = Agent._with_retrieval([], "recalled")
    assert len(out) == 1
    assert "<retrieved_memory>" in out[0]["content"]


def test_changing_retrieval_leaves_conversation_prefix_identical():
    """The regression this whole change exists for.

    Two turns whose retrieved memory differs completely must still share a
    byte-identical prefix across the entire conversation. While the block sat
    at the end of the system prompt, a change here invalidated every
    subsequent token — on 2026-08-08 that meant re-prefilling 66.5k tokens
    (353 s) instead of ~200 (0.7 s).
    """
    history = _history(50)
    a = Agent._with_retrieval(history, "recalled set A")
    b = Agent._with_retrieval(history, "an entirely different set B")

    assert a[:-2] == b[:-2]          # whole conversation prefix untouched
    assert a[-1] == b[-1]            # and the current user message matches
    assert a[-2] != b[-2]            # only the retrieval row differs


# --- bounded incidental tool output ----------------------------------------

def test_short_result_passes_through_untouched(tmp_path):
    text = "a modest amount of output\n[exit_code] 0"
    assert bound_result(text, workspace_dir=tmp_path, tool="bash") == text


def test_result_at_cap_is_untouched(tmp_path):
    text = "x" * MAX_INLINE_CHARS
    assert bound_result(text, workspace_dir=tmp_path, tool="bash") == text


def test_oversized_result_is_bounded_and_spooled(tmp_path):
    text = "HEAD" + ("x" * 40_000) + "TAIL"
    out = bound_result(text, workspace_dir=tmp_path, tool="bash")

    assert len(out) < len(text)
    assert out.startswith("HEAD")
    assert out.endswith("TAIL")
    assert "[truncated:" in out
    assert str(len(text)) in out

    # The note must name a real file holding the *complete* original.
    spooled = list((tmp_path / ".tool-results").rglob("bash-*.txt"))
    assert len(spooled) == 1
    assert spooled[0].read_text() == text
    assert spooled[0].name in out


def test_bash_exit_code_survives_truncation(tmp_path):
    """``_run_bash`` appends ``[exit_code] N`` last, so a head-only truncation
    would hide whether the command actually succeeded."""
    text = ("noise\n" * 20_000) + "[exit_code] 137"
    out = bound_result(text, workspace_dir=tmp_path, tool="bash")
    assert "[exit_code] 137" in out


def test_bounded_result_size_is_predictable(tmp_path):
    text = "y" * 100_000
    out = bound_result(text, workspace_dir=tmp_path, tool="web_search")
    # head + tail + a short note, regardless of how large the input was.
    assert len(out) < HEAD_CHARS + TAIL_CHARS + 500


def test_identical_output_spools_once(tmp_path):
    text = "z" * 50_000
    first = bound_result(text, workspace_dir=tmp_path, tool="bash")
    second = bound_result(text, workspace_dir=tmp_path, tool="bash")
    assert first == second
    assert len(list((tmp_path / ".tool-results").rglob("bash-*.txt"))) == 1


def test_spool_failure_still_bounds_and_says_so(tmp_path):
    # A read-only workspace makes the spool write fail; the result must still
    # be bounded (bounded beats unbounded) and must not claim a path exists.
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / ".tool-results").write_text("not a directory")

    text = "q" * 50_000
    out = bound_result(text, workspace_dir=workspace, tool="bash")
    assert len(out) < len(text)
    assert "[truncated:" in out
    assert "unavailable" in out
    assert "read_file it" not in out
