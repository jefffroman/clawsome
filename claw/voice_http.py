"""claw's HTTP voice-turn endpoint — the transport-decoupled voice inbound.

This is the "post-STT entry point" of a voice turn, expressed as request/
response. An **external voice stack** owns the audio — mic capture, wake, STT,
and TTS — and POSTs a *transcript* here, getting the agent's reply *text* back.
claw never touches audio.

The endpoint is transport-agnostic; any voice client can hit it:
- a thin client whose audio pipeline runs in an external voice stack (e.g. a
  hardware speaker that can't do STT/TTS), or
- a self-contained client that does its own native STT/TTS.

Wire contract (``POST <http_bind>/voice/turn``, JSON):

    request : {"device_id": str, "endpoint_id": str|null, "text": str}
    response: 200 {"reply": str, "agent_id": str}
              4xx/5xx {"error": str}

claw resolves ``device_id`` → bound agent + endpoint via the ``devices:``
config, derives the turn's **modality** from the endpoint's ``type`` (so the
voice speakable-reply hint is config-driven, not asserted by the caller),
dispatches through the normal agent path (shared ``"home"`` session,
coalescing, memory — all unchanged), and blocks on the reply.

The reply round-trips through the standard agent outbound path: the agent
calls ``channel.send(peer_id, text)`` on the :class:`HttpReplyChannel`
registered (by ``main.py``) under the ``"voice"`` channel name, which resolves
the per-request future this handler is awaiting. Each request gets a unique
``peer_id`` (``http:<uuid>``), so concurrent turns don't cross.

Stood up when ``HttpApiConfig.enabled``. claw runs no audio: mic capture, wake,
STT, and TTS all live in the external voice stack, which is one client of this
endpoint. The ``HttpReplyChannel`` is registered under the ``"voice"`` channel
name; nothing else claims it (claw has no WS voice channel).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import suppress
from typing import AsyncContextManager

from claw.channel.base import InboundHandler, InboundMessage, no_typing
from claw.config import DeviceConfig

log = logging.getLogger("claw.voice_http")

# Local reasoning-model voice turns run 15-45 s, far more when serialized behind
# another conversation. Sized to not strand a reply the caller is about to
# receive.
_REPLY_TIMEOUT_S = 300.0


class HttpReplyChannel:
    """Outbound ``Channel`` that resolves a per-request future on ``send``.

    Registered on each voice-bound agent under the ``"voice"`` channel name so
    a voice turn's reply (``channel.send(peer_id, text)``) lands here instead of
    matrix. ``send`` resolves the future keyed by the turn's unique ``peer_id``;
    the HTTP handler awaits it and returns the text as the response body.
    """

    name = "voice"

    def __init__(self) -> None:
        # peer_id -> future awaiting this turn's reply text.
        self._futures: dict[str, asyncio.Future[str]] = {}

    def expect(self, peer_id: str) -> asyncio.Future[str]:
        """Register and return the future for ``peer_id``'s reply."""
        fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._futures[peer_id] = fut
        return fut

    def discard(self, peer_id: str) -> None:
        """Drop a pending future (the handler timed out / gave up)."""
        self._futures.pop(peer_id, None)

    # -- Channel protocol -------------------------------------------------

    async def start(self, on_message: InboundHandler | None = None) -> None:
        _ = on_message  # voice uses per-agent registration, not this

    async def send(self, peer_id: str, text: str) -> None:
        """Deliver an agent reply: resolve the awaiting future.

        Drops silently if no one is waiting (the request already timed out and
        returned to the caller) — a lost late reply is better than a crash.
        """
        fut = self._futures.pop(peer_id, None)
        if fut is None:
            log.debug("http reply for unknown/expired peer %r; dropping", peer_id)
            return
        if not fut.done():
            fut.set_result(text)

    async def shutdown(self) -> None:
        for fut in self._futures.values():
            if not fut.done():
                fut.cancel()
        self._futures.clear()

    def typing(self, peer_id: str) -> AsyncContextManager[None]:
        return no_typing(peer_id)

    async def clear_typing(self, peer_id: str) -> None:
        _ = peer_id


def _err(status: int, message: str):
    from aiohttp import web

    return web.json_response({"error": message}, status=status)


class VoiceHttpServer:
    """Minimal aiohttp server exposing ``POST /voice/turn``."""

    def __init__(
        self,
        *,
        bind_host: str,
        bind_port: int,
        devices: dict[str, DeviceConfig],
        handlers: dict[str, InboundHandler],
        reply_channel: HttpReplyChannel,
        reply_timeout_s: float = _REPLY_TIMEOUT_S,
    ) -> None:
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.devices = dict(devices)
        self.handlers = dict(handlers)
        self.reply_channel = reply_channel
        self.reply_timeout_s = reply_timeout_s
        self._runner: object | None = None

    async def start(self) -> None:
        # Import here so claw without the http transport doesn't require aiohttp.
        from aiohttp import web

        app = web.Application()
        app.router.add_post("/voice/turn", self._handle_turn)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.bind_host, self.bind_port)
        await site.start()
        self._runner = runner
        log.info(
            "voice HTTP endpoint: listening on %s:%d (%d device(s): %s)",
            self.bind_host, self.bind_port, len(self.devices),
            ", ".join(sorted(self.devices)) or "(none)",
        )

    async def shutdown(self) -> None:
        if self._runner is not None:
            from aiohttp.web import AppRunner
            assert isinstance(self._runner, AppRunner)
            with suppress(Exception):
                await self._runner.cleanup()
            self._runner = None

    async def _handle_turn(self, request):
        from aiohttp import web

        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return _err(400, "body is not valid JSON")
        if not isinstance(body, dict):
            return _err(400, "body must be a JSON object")

        device_id = body.get("device_id")
        text = body.get("text")
        endpoint_id = body.get("endpoint_id")
        if not isinstance(device_id, str) or not device_id:
            return _err(400, "missing 'device_id'")
        if not isinstance(text, str) or not text.strip():
            return _err(400, "missing or empty 'text'")
        if endpoint_id is not None and not isinstance(endpoint_id, str):
            return _err(400, "'endpoint_id' must be a string or null")

        device = self.devices.get(device_id)
        if device is None:
            return _err(404, f"unknown device {device_id!r}")

        if endpoint_id is not None:
            endpoint = next(
                (e for e in device.endpoints if e.id == endpoint_id), None
            )
            if endpoint is None:
                return _err(
                    404, f"device {device_id!r} has no endpoint {endpoint_id!r}"
                )
        else:
            endpoint = next(
                (e for e in device.endpoints if e.type == "voice"), None
            )
            if endpoint is None:
                return _err(400, f"device {device_id!r} has no voice endpoint")

        handler = self.handlers.get(device.agent)
        if handler is None:
            return _err(
                503,
                f"device {device_id!r} binds to agent {device.agent!r} "
                f"with no registered handler",
            )

        peer_id = f"http:{uuid.uuid4()}"
        msg = InboundMessage(
            peer_id=peer_id,
            sender_name=endpoint.name,
            text=text,
            channel="voice",
            sender_id=f"{device_id}/{endpoint.id}",
            # Modality drives the speakable-reply hint — derived from the
            # endpoint's configured type, NOT asserted by the caller.
            modality=endpoint.type,
            session_key="home",
        )

        fut = self.reply_channel.expect(peer_id)
        await handler(msg)  # fire-and-forget enqueue into the agent drainer
        try:
            reply = await asyncio.wait_for(fut, timeout=self.reply_timeout_s)
        except asyncio.TimeoutError:
            self.reply_channel.discard(peer_id)
            log.warning(
                "voice HTTP: no agent reply within %.0fs (device=%s)",
                self.reply_timeout_s, device_id,
            )
            return _err(504, "agent reply timeout")
        except asyncio.CancelledError:
            self.reply_channel.discard(peer_id)
            raise

        return web.json_response({"reply": reply, "agent_id": device.agent})
