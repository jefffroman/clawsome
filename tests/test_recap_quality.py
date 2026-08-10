"""Idle-recap sizing: tool-clip, word budget, size floor, summarize params.

All four were measured on a real 166-turn session (2026-08-08) whose recap
came out at ~140 words with `Open: none.` on a session that had in fact ended
mid-failure:

* tool results were clipped to 400 chars before the summarizer saw them, on
  rows the transcript had ALREADY capped at 8192 — 81% of the evidence
  discarded by a second, tighter truncation;
* the word cap was a flat 200 regardless of whether 4 turns or 166 were being
  compressed;
* there was no size floor, so a 2-turn session got recapped at boot;
* summarize() sent no options, inheriting the Modelfile's chat defaults
  (presence_penalty 1.5, temperature 1.0) and generating a reasoning trace
  that summarize() then discarded — 88% of its output tokens.

asyncio_mode=auto: no marker needed.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from claw.compaction import (
    _mid_session_instruction,
    _recap_word_budget,
    _rows_to_text,
    _MID_RECAP_WORDS_MAX,
    _MID_RECAP_WORDS_MIN,
    _RECAP_WORDS_MAX,
    _RECAP_WORDS_MIN,
    maybe_idle_recap,
)
from claw.config import CompactionConfig
from claw.ollama import TOOL_RESULT_THRESHOLD_CHARS


# --- tool clip -------------------------------------------------------------

def test_tool_results_are_not_double_truncated():
    """Transcript rows are already capped at TOOL_RESULT_THRESHOLD_CHARS, so
    the summarizer's clip must not cut below that — the old 400 threw away
    most of every non-trivial tool result."""
    body = "y" * TOOL_RESULT_THRESHOLD_CHARS
    text = _rows_to_text([{"role": "tool", "content": body}])
    assert len(text) >= TOOL_RESULT_THRESHOLD_CHARS
    assert "y" * 2000 in text


def test_tool_results_still_bounded():
    """Still a guard, for any caller handing us unbounded rows."""
    text = _rows_to_text([{"role": "tool", "content": "z" * 100_000}])
    assert len(text) < 100_000


# --- word budget -----------------------------------------------------------

@pytest.mark.parametrize("tokens,expected", [
    (0, _RECAP_WORDS_MIN),           # empty -> floor
    (4_000, _RECAP_WORDS_MIN),       # small session -> floor
    (24_000, 600),                   # the measured 166-turn session
    (10_000_000, _RECAP_WORDS_MAX),  # clamped
])
def test_recap_word_budget_scales_and_clamps(tokens, expected):
    assert _recap_word_budget(tokens) == expected


def test_budget_is_monotonic():
    prev = 0
    for t in (0, 5_000, 20_000, 50_000, 200_000):
        cur = _recap_word_budget(t)
        assert cur >= prev
        prev = cur


def test_mid_session_budget_is_more_generous():
    """Mid-session compacts LIVE working context, so it gets a higher floor
    and ceiling than an idle recap of a finished conversation."""
    for tokens in (0, 20_000, 500_000):
        idle = _recap_word_budget(tokens)
        mid = _recap_word_budget(
            tokens, lo=_MID_RECAP_WORDS_MIN, hi=_MID_RECAP_WORDS_MAX,
        )
        assert mid >= idle
    assert _recap_word_budget(0, lo=_MID_RECAP_WORDS_MIN,
                              hi=_MID_RECAP_WORDS_MAX) == _MID_RECAP_WORDS_MIN
    assert _recap_word_budget(10_000_000, lo=_MID_RECAP_WORDS_MIN,
                              hi=_MID_RECAP_WORDS_MAX) == _MID_RECAP_WORDS_MAX


def test_num_predict_covers_the_largest_budget():
    """A truncated recap is worst exactly where the budget is largest, so
    the generation cap must clear _MID_RECAP_WORDS_MAX with slack.
    ~1.33 tokens/word is the usual English ratio."""
    from claw.ollama import SUMMARY_NUM_PREDICT
    assert SUMMARY_NUM_PREDICT > _MID_RECAP_WORDS_MAX * 1.33


def test_mid_session_instruction_asks_for_resumable_state():
    """It differs from the idle one in kind: the agent keeps working from
    this text, so in-flight state and the word budget must both be stated."""
    text = _mid_session_instruction(1234)
    assert "1234" in text
    assert "Open:" in text
    assert "in-flight" in text


# --- size floor ------------------------------------------------------------

class _FakeTranscripts:
    def __init__(self, rows): self._rows = rows
    def last_ts(self, sid): return "2000-01-01T00:00:00+00:00"   # ancient
    def load(self, sid): return self._rows
    def archive(self, sid, tag): raise AssertionError("must not archive")


class _ExplodingOllama:
    async def summarize(self, **_kw):
        raise AssertionError("must not summarize below the floor")


def _cfg(**kw):
    class C: pass
    c = C()
    c.compaction = CompactionConfig(**kw)
    return c


async def test_tiny_session_is_left_verbatim():
    """THE regression: a 2-turn session was recapped at boot, replacing the
    rows with a paraphrase and archiving them. Below the floor we must not
    summarize and must not archive."""
    rows = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}]
    out = await maybe_idle_recap(
        cfg=_cfg(idle_recap_min_tokens=4000),
        ollama=_ExplodingOllama(),
        transcripts=_FakeTranscripts(rows),
        sid="s",
        compaction_model="m",
    )
    assert out is None


async def test_floor_of_zero_restores_old_behaviour():
    """The floor is configurable; 0 means recap anything (pre-change)."""
    rows = [{"role": "user", "content": "hi"}]
    calls = {}

    class _Ollama:
        async def summarize(self, *, model, instruction, transcript_text):
            calls["instruction"] = instruction
            return "summary body"

    class _T(_FakeTranscripts):
        def archive(self, sid, tag): return Path(f"/tmp/{sid}.{tag}.jsonl")

    out = await maybe_idle_recap(
        cfg=_cfg(idle_recap_min_tokens=0),
        ollama=_Ollama(),
        transcripts=_T(rows),
        sid="s",
        compaction_model="m",
    )
    assert out is not None
    assert "summary body" in out
    # The instruction carries a concrete word budget and demands the Open line.
    assert "Open:" in calls["instruction"]
    assert "words" in calls["instruction"]


# --- summarize params ------------------------------------------------------

async def test_summarize_disables_thinking_and_overrides_chat_defaults():
    from claw.config import OllamaConfig
    from claw.ollama import (
        SUMMARY_NUM_PREDICT,
        SUMMARY_PRESENCE_PENALTY,
        SUMMARY_TEMPERATURE,
        OllamaClient,
    )
    sent = {}

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": " out "}}

    class _Client:
        async def post(self, url, json):
            sent.update(json)
            return _Resp()

    c = OllamaClient(OllamaConfig(
        base_url="http://ollama.invalid",
        default_compaction_model="m",
    ))
    c._client = _Client()  # type: ignore[assignment]
    out = await c.summarize(model="m", instruction="i", transcript_text="t")

    assert out == "out"
    assert sent["think"] is False
    assert sent["options"]["presence_penalty"] == SUMMARY_PRESENCE_PENALTY == 0.0
    assert sent["options"]["temperature"] == SUMMARY_TEMPERATURE
    assert sent["options"]["num_predict"] == SUMMARY_NUM_PREDICT


async def test_summarize_retries_without_think_when_rejected():
    """A compaction_model with no thinking mode must not silently kill
    compaction — maybe_idle_recap swallows exceptions and returns None."""
    import httpx
    from claw.config import OllamaConfig
    from claw.ollama import OllamaClient

    bodies = []

    class _Resp:
        def __init__(self, ok): self.ok = ok
        def raise_for_status(self):
            if not self.ok:
                req = httpx.Request("POST", "http://ollama.invalid/api/chat")
                raise httpx.HTTPStatusError(
                    "400", request=req,
                    response=httpx.Response(400, request=req),
                )
        def json(self): return {"message": {"content": "recovered"}}

    class _Client:
        async def post(self, url, json):
            bodies.append(dict(json))
            return _Resp(ok="think" not in json)

    c = OllamaClient(OllamaConfig(
        base_url="http://ollama.invalid",
        default_compaction_model="m",
    ))
    c._client = _Client()  # type: ignore[assignment]
    out = await c.summarize(model="m", instruction="i", transcript_text="t")

    assert out == "recovered"
    assert len(bodies) == 2
    assert "think" in bodies[0] and "think" not in bodies[1]
    # options survive the retry — they are not the thing being rejected.
    assert bodies[1]["options"]["presence_penalty"] == 0.0
