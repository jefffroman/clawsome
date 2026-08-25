"""Channel protocol — the minimal contract any inbound surface satisfies."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncContextManager, Awaitable, Callable, Protocol


@dataclass
class InboundMessage:
    """One inbound message from any channel.

    Four identifiers, deliberately distinct:

    - ``channel`` — transport / application protocol (``matrix``, ``voice``,
      and later ``event`` / ``status`` for device sensors). Routing only.
    - ``session_key`` — transcript grouping → the sid. Defaults to
      ``f"{channel}_{peer_id}"`` (matrix/cron unchanged), so each room/source
      is its own session; the voice channel sets it to the literal ``"home"``
      so every device folds into one household transcript.
    - ``sender_id`` / ``sender_name`` — *who/what spoke*. ``sender_id`` is the
      stable identifier — full MXID for matrix (``@user-1:example.org``),
      ``f"{device_id}/{endpoint_id}"`` for devices (e.g. ``speaker/voice``), task_id
      for subagent_completion synthetics, empty otherwise (cron /
      initial_prompt fall back to ``sender_name`` for log labelling).
      ``sender_name`` is the human display label.
    - ``peer_id`` — the reply-route handle (Matrix room id; voice
      per-connection ``voice:<device_id>:<uuid>``). Plumbing; the agent does
      not read it. It no longer doubles as the session key.

    ``modality`` (``text`` | ``voice`` | ``event`` | ``state``) is how the
    agent should read the input and form the reply; the speakable-reply prompt
    hint keys off ``modality == "voice"``, never off ``channel``.
    """
    peer_id: str
    sender_name: str
    text: str
    channel: str
    sender_id: str = ""
    modality: str = "text"
    session_key: str = ""
    # True only for the synthetic completion a finished subagent fires
    # back at its spawner. The agent may inject these into a RUNNING
    # turn (see Agent._process_batch); every other inbound waits for the
    # drainer and becomes its own conversational turn.
    is_subagent_completion: bool = False

    def __post_init__(self) -> None:
        # Default the session key to the legacy per-(channel, peer) form so
        # every existing producer (matrix, cron, initial_prompt, subagent
        # completion) keeps its current session unchanged. Channels that want
        # a shared session (voice → "home") pass session_key explicitly.
        if not self.session_key:
            self.session_key = f"{self.channel}_{self.peer_id}"


InboundHandler = Callable[[InboundMessage], Awaitable[None]]


class Channel(Protocol):
    name: str

    async def start(self, on_message: InboundHandler) -> None: ...
    async def send(self, peer_id: str, text: str) -> None: ...
    async def shutdown(self) -> None: ...
    def typing(self, peer_id: str) -> AsyncContextManager[None]: ...
    async def clear_typing(self, peer_id: str) -> None: ...

    # Optional capability — probed via ``getattr`` by the agent, not required.
    # Returns the ``session_key`` an inbound from ``peer_id`` would fold into,
    # so a synthetic turn (cron) can mirror its delivered reply into the
    # human-facing session. Only channels that carry a distinct human-facing
    # session (matrix) implement it; others simply omit it and mirroring
    # no-ops. See ``MatrixChannel.primary_session_key``.
    def primary_session_key(self, peer_id: str) -> str | None: ...


@asynccontextmanager
async def no_typing(_peer_id: str):
    """Fallback for channels that don't support typing indicators."""
    yield
