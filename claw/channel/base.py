"""Channel protocol — the minimal contract any inbound surface satisfies."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncContextManager, Awaitable, Callable, Protocol


@dataclass
class InboundMessage:
    """One inbound message from any channel.

    For Matrix: ``peer_id`` is the room ID (DMs have one peer; group rooms
    are routed by mention). ``sender_name`` is the human display name.
    ``sender_id`` is the stable identifier — full MXID for matrix
    (``@user-1:example.org``), task_id for subagent_completion synthetics,
    empty otherwise (cron / initial_prompt fall back to sender_name for
    log labelling).
    """
    peer_id: str
    sender_name: str
    text: str
    channel: str
    sender_id: str = ""


InboundHandler = Callable[[InboundMessage], Awaitable[None]]


class Channel(Protocol):
    name: str

    async def start(self, on_message: InboundHandler) -> None: ...
    async def send(self, peer_id: str, text: str) -> None: ...
    async def shutdown(self) -> None: ...
    def typing(self, peer_id: str) -> AsyncContextManager[None]: ...
    async def clear_typing(self, peer_id: str) -> None: ...


@asynccontextmanager
async def no_typing(_peer_id: str):
    """Fallback for channels that don't support typing indicators."""
    yield
