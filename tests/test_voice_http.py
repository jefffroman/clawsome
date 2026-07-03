"""Tests for claw's HTTP voice-turn endpoint (claw.voice_http).

Exercises the reply-future plumbing and the turn handler's resolution,
validation, and modality derivation against a fake agent handler — no real
Agent, Ollama, or network. The handler stands in for the agent path: it
records the InboundMessage it received and resolves the reply via the
HttpReplyChannel, exactly as the real agent does through channel.send.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from claw.channel.base import InboundMessage
from claw.config import DeviceConfig, EndpointConfig
from claw.voice_http import HttpReplyChannel, VoiceHttpServer

pytestmark = pytest.mark.asyncio


# --- fixtures / fakes ------------------------------------------------------

def _device(agent: str = "agent-1") -> DeviceConfig:
    return DeviceConfig(
        device_id="box-1",
        name="Box One",
        agent=agent,
        endpoints=(
            EndpointConfig(id="voice", type="voice", name="Living Room"),
            EndpointConfig(id="motion", type="event", name="Living Room Motion"),
        ),
    )


class _FakeRequest:
    """Minimal stand-in for aiohttp.web.Request — only ``.json()`` is used."""

    def __init__(self, payload) -> None:
        self._payload = payload

    async def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _server(handler, *, reply_timeout_s: float = 5.0, agent: str = "agent-1"):
    reply_channel = HttpReplyChannel()
    handlers = {agent: handler} if handler is not None else {}
    server = VoiceHttpServer(
        bind_host="127.0.0.1",
        bind_port=0,
        devices={"box-1": _device(agent)},
        handlers=handlers,
        reply_channel=reply_channel,
        reply_timeout_s=reply_timeout_s,
    )
    return server, reply_channel


async def _read(resp):
    """(status, parsed-json-body) from an aiohttp json_response."""
    return resp.status, json.loads(resp.body)


# --- HttpReplyChannel ------------------------------------------------------

async def test_reply_channel_expect_then_send_resolves():
    ch = HttpReplyChannel()
    fut = ch.expect("http:abc")
    assert not fut.done()
    await ch.send("http:abc", "hello there")
    assert fut.result() == "hello there"


async def test_reply_channel_send_unknown_peer_is_noop():
    ch = HttpReplyChannel()
    # No future registered — must not raise.
    await ch.send("http:nope", "ignored")


async def test_reply_channel_discard_drops_future():
    ch = HttpReplyChannel()
    ch.expect("http:abc")
    ch.discard("http:abc")
    # A late reply after discard is dropped, not delivered to a stale future.
    await ch.send("http:abc", "late")  # no raise


async def test_reply_channel_shutdown_cancels_pending():
    ch = HttpReplyChannel()
    fut = ch.expect("http:abc")
    await ch.shutdown()
    assert fut.cancelled()


# --- happy path ------------------------------------------------------------

async def test_turn_dispatches_and_returns_reply():
    received: list[InboundMessage] = []

    async def handler(msg: InboundMessage) -> None:
        received.append(msg)
        # Real agent replies later via channel.send; mimic with a task so the
        # handler returns first (fire-and-forget enqueue contract).
        async def _later():
            await server.reply_channel.send(msg.peer_id, f"echo: {msg.text}")
        asyncio.create_task(_later())

    server, _ = _server(handler)
    resp = await server._handle_turn(_FakeRequest(
        {"device_id": "box-1", "text": "what time is it"}
    ))
    status, body = await _read(resp)
    assert status == 200
    assert body == {"reply": "echo: what time is it", "agent_id": "agent-1"}
    assert len(received) == 1


async def test_turn_message_fields_and_modality_derivation():
    received: list[InboundMessage] = []

    async def handler(msg: InboundMessage) -> None:
        received.append(msg)
        await server.reply_channel.send(msg.peer_id, "ok")

    server, _ = _server(handler)
    await server._handle_turn(_FakeRequest(
        {"device_id": "box-1", "text": "hi"}
    ))
    msg = received[0]
    assert msg.channel == "voice"
    assert msg.session_key == "home"
    assert msg.sender_id == "box-1/voice"
    assert msg.sender_name == "Living Room"
    # Modality is derived from the endpoint's type, not asserted by the caller.
    assert msg.modality == "voice"
    assert msg.peer_id.startswith("http:")


async def test_explicit_endpoint_id_selects_endpoint():
    received: list[InboundMessage] = []

    async def handler(msg: InboundMessage) -> None:
        received.append(msg)
        await server.reply_channel.send(msg.peer_id, "ok")

    server, _ = _server(handler)
    # The 'motion' endpoint is type=event → modality should follow it.
    await server._handle_turn(_FakeRequest(
        {"device_id": "box-1", "endpoint_id": "motion", "text": "x"}
    ))
    msg = received[0]
    assert msg.sender_id == "box-1/motion"
    assert msg.modality == "event"


# --- validation / error paths ---------------------------------------------

@pytest.mark.parametrize("payload,status", [
    ({"text": "hi"}, 400),                              # missing device_id
    ({"device_id": "box-1"}, 400),                      # missing text
    ({"device_id": "box-1", "text": "   "}, 400),       # empty text
    ({"device_id": "ghost", "text": "hi"}, 404),        # unknown device
    ({"device_id": "box-1", "endpoint_id": "nope", "text": "hi"}, 404),
])
async def test_validation_errors(payload, status):
    async def handler(msg):  # never reached for these
        await server.reply_channel.send(msg.peer_id, "ok")

    server, _ = _server(handler)
    resp = await server._handle_turn(_FakeRequest(payload))
    st, body = await _read(resp)
    assert st == status
    assert "error" in body


async def test_non_dict_body_is_400():
    server, _ = _server(None)
    resp = await server._handle_turn(_FakeRequest([1, 2, 3]))
    st, body = await _read(resp)
    assert st == 400


async def test_bad_json_is_400():
    server, _ = _server(None)
    resp = await server._handle_turn(_FakeRequest(ValueError("bad json")))
    st, body = await _read(resp)
    assert st == 400


async def test_no_handler_for_agent_is_503():
    # Device binds to agent-1 but no handler registered.
    server, _ = _server(None)
    resp = await server._handle_turn(_FakeRequest(
        {"device_id": "box-1", "text": "hi"}
    ))
    st, body = await _read(resp)
    assert st == 503


async def test_reply_timeout_is_504_and_clears_future():
    async def silent_handler(msg):  # never replies
        pass

    server, reply_channel = _server(silent_handler, reply_timeout_s=0.05)
    resp = await server._handle_turn(_FakeRequest(
        {"device_id": "box-1", "text": "hi"}
    ))
    st, body = await _read(resp)
    assert st == 504
    # The abandoned future must not linger.
    assert not reply_channel._futures
