"""Empty-reply recovery is one-shot; a SECOND empty must not end the turn
in silence.

Regression for 2026-08-08. ``run_turn``'s recovery branch fires at most once
per turn (``recovery_used``). On work-heavy turns the model can emit a second
empty completion — done_reason="stop" with a single end-of-turn token, so no
tool_calls and no text — and that landed on the unguarded
``if not tool_calls: return new_messages, content, ...`` with ``content``
still "". agent.py only sends on a truthy reply, so the turn ended in total
silence: indistinguishable, from the user's side, from being ignored.

Observed live twice in one afternoon (17- and 20-tool-turn sequences), each
after a first empty had already spent the recovery.

Two properties are asserted: the RETURNED text is the sentinel (so something
reaches the user), and the PERSISTED assistant row carries it too (an empty
assistant row is a degenerate example that would otherwise sit in the
transcript for the rest of the session).

Drives the real ``OllamaClient.run_turn`` with a scripted ``chat_once`` — no
network. asyncio_mode=auto: no marker needed.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

from pathlib import Path

from claw.config import OllamaConfig
from claw.ollama import _EMPTY_REPLY_FALLBACK, OllamaClient


def _reply(content: str, done_reason: str = "stop") -> dict:
    """A no-tool-call assistant response (ends the loop)."""
    return {"message": {"content": content}, "done_reason": done_reason}


def _tool_reply(name: str = "bash", content: str = "") -> dict:
    """An assistant response that calls a tool. Empty content here is the
    LEGITIMATE empty case — the tool call is the turn's output."""
    return {
        "message": {
            "content": content,
            "tool_calls": [{
                "function": {"name": name, "arguments": {"command": "true"}},
            }],
        },
        "done_reason": "stop",
    }


def _client() -> OllamaClient:
    return OllamaClient(OllamaConfig(
        base_url="http://ollama.invalid",
        default_compaction_model="test-compact-model",
    ))


async def _run(client: OllamaClient, scripted: list, tools: dict | None = None) -> tuple:
    """Replace chat_once with a scripted sequence and drive one run_turn.
    Returns (result, ncalls)."""
    n = {"i": 0}

    async def fake_chat_once(**_kw):
        item = scripted[n["i"]]
        n["i"] += 1
        return item

    client.chat_once = fake_chat_once  # type: ignore[assignment]
    result = await client.run_turn(
        model="test-model",
        history=[{"role": "user", "content": "hi"}],
        system=None,
        tools=tools or {},
        sid="s",
        workspace_dir=Path("/tmp"),
        label="test",
    )
    return result, n["i"]


async def test_second_empty_returns_sentinel_not_silence():
    """THE regression: recovery is spent on the first empty, the second has
    nothing left to try, and must not come back as ""."""
    (_, final_text, _), ncalls = await _run(_client(), [
        _reply(""),            # turn 0: empty -> recovery fires
        _reply(""),            # recovery retry: STILL empty -> fallback
    ])
    assert final_text == _EMPTY_REPLY_FALLBACK
    assert final_text.strip(), "must be truthy or agent.py sends nothing"
    assert ncalls == 2


async def test_second_empty_is_not_persisted_as_an_empty_row():
    """The transcript row carries the sentinel too — no empty assistant row
    left behind to condition later turns on."""
    (new_messages, final_text, _), _ = await _run(_client(), [
        _reply(""),
        _reply(""),
    ])
    assistants = [m for m in new_messages if m.get("role") == "assistant"]
    assert assistants, "the turn must still be recorded"
    assert all(m["content"].strip() for m in assistants)
    assert assistants[-1]["content"] == final_text


async def test_first_empty_still_recovers_normally():
    """Unchanged behavior: one empty is still repaired by the retry, and the
    fallback does not pre-empt it."""
    (_, final_text, _), ncalls = await _run(_client(), [
        _reply(""),             # turn 0: empty -> recovery
        _reply("recovered"),    # retry produces real text
    ])
    assert final_text == "recovered"
    assert ncalls == 2


async def test_empty_content_with_tool_calls_is_untouched():
    """The legitimate empty case: a model calling a tool normally emits no
    prose. That must not trip the fallback."""
    async def _noop(_args):
        return "ok"

    from claw.tools.base import Tool
    tools = {"bash": Tool(
        name="bash", description="d", input_schema={}, run=_noop,
    )}
    (new_messages, final_text, _), ncalls = await _run(_client(), [
        _tool_reply(),          # turn 0: tool call, empty content — fine
        _reply("all done"),     # turn 1: real reply
    ], tools=tools)
    assert final_text == "all done"
    assert ncalls == 2
    # The tool-calling row keeps its empty content; it is not a failure.
    tool_rows = [m for m in new_messages if m.get("tool_calls")]
    assert len(tool_rows) == 1
    assert tool_rows[0]["content"] == ""
