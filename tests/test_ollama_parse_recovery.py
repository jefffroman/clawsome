"""The 5xx parse-recovery guard is SHARED across every ``chat_once`` in a
single ``run_turn`` — including the empty-reply/length recovery-branch
retry, not just the top-of-loop call.

Regression for the 2026-07-25 stacked-recovery gap: Ollama returns a 500
when it can't parse the model's raw tool-call output (malformed XML, e.g.
``<function>`` closed by ``</parameter>``). Before the fix, only the
top-of-loop ``chat_once`` was wrapped in the guarded re-prompt; a malformed
tool call emitted DURING empty-reply recovery raised an unhandled 500 and
crashed ``run_turn`` (observed live at 11:06). Now both call sites share the
one-shot ``parse_retry_used`` budget.

Drives the real ``OllamaClient.run_turn`` with a scripted ``chat_once`` — no
network. asyncio_mode=auto: no marker needed.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from claw.config import OllamaConfig
from claw.ollama import OllamaClient


def _http_500() -> httpx.HTTPStatusError:
    """A 500 shaped like Ollama's tool-call parser failure."""
    req = httpx.Request("POST", "http://ollama.invalid/api/chat")
    resp = httpx.Response(
        500,
        json={"error": "XML syntax error on line 5: element <function> "
                       "closed by </parameter>"},
        request=req,
    )
    return httpx.HTTPStatusError("500", request=req, response=resp)


def _reply(content: str, done_reason: str = "stop") -> dict:
    """A no-tool-call assistant response (ends the loop)."""
    return {"message": {"content": content}, "done_reason": done_reason}


def _client() -> OllamaClient:
    return OllamaClient(OllamaConfig(
        base_url="http://ollama.invalid",
        default_compaction_model="test-compact-model",
    ))


async def _run(client: OllamaClient, scripted: list) -> tuple:
    """Replace chat_once with a scripted sequence (values returned,
    Exceptions raised) and drive one run_turn. Returns (result, ncalls)."""
    n = {"i": 0}

    async def fake_chat_once(**_kw):
        item = scripted[n["i"]]
        n["i"] += 1
        if isinstance(item, Exception):
            raise item
        return item

    client.chat_once = fake_chat_once  # type: ignore[assignment]
    result = await client.run_turn(
        model="test-model",
        history=[{"role": "user", "content": "hi"}],
        system=None,
        tools={},
        sid="s",
        workspace_dir=Path("/tmp"),
        label="test",
    )
    return result, n["i"]


async def test_5xx_during_empty_reply_recovery_is_guarded():
    """The regression: empty-reply recovery retry hits a parser 500, and
    the shared guard re-prompts once and recovers instead of crashing."""
    (new_messages, final_text, _), ncalls = await _run(_client(), [
        _reply(""),          # turn 0: empty content -> empty-reply recovery
        _http_500(),         # recovery retry: malformed tool call -> 500
        _reply("recovered"),  # guarded re-prompt: clean final reply
    ])
    assert final_text == "recovered"
    assert ncalls == 3


async def test_5xx_at_top_of_loop_still_recovers():
    """The pre-existing top-of-loop guard is unchanged by the refactor."""
    (_, final_text, _), ncalls = await _run(_client(), [
        _http_500(),          # turn 0 top-of-loop: 500
        _reply("recovered"),  # guarded re-prompt
    ])
    assert final_text == "recovered"
    assert ncalls == 2


async def test_parse_recovery_is_one_shot_across_the_turn():
    """The guard fires at most once per run_turn: a second 5xx (here on the
    guarded retry itself) propagates rather than looping forever."""
    with pytest.raises(httpx.HTTPStatusError):
        await _run(_client(), [
            _http_500(),  # turn 0 top-of-loop: 500 -> guard spends its retry
            _http_500(),  # guarded retry ALSO 500 -> propagates
        ])
