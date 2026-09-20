"""Reply functions for gate handlers.

A handler's reply is normally data — a fixed string or variants, or the tool's
own text relayed. When the reply should say something neither covers, a
handler names one of these instead (``reply_fn``).

Each function receives the tool's answer — its text output (text-shape
handlers) or its structured result (outcomes handlers) — the gate context and
the agent's tool registry, and returns the reply text, or ``None`` when it has
nothing to say, in which case the handler's ``fallback`` is used. A reply
function runs AFTER the action, so it must never decline the turn; anything it
cannot do is a ``None``.

Functions read structured fields (a tool's ``data`` twin), never a tool's
prose: that is written for the model and free to change.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

ReplyFn = Callable[[Any, Any, dict[str, Any]], Awaitable[str | None]]


def _track(fields: dict[str, Any] | None) -> str | None:
    """"<title> by <artist>", or the title alone; None when nothing nameable."""
    if not fields or not fields.get("title"):
        return None
    return f"{fields['title']} by {fields['artist']}" if fields.get("artist") else fields["title"]


async def skipping_to(result: Any, ctx: Any, tools: dict[str, Any]) -> str | None:
    """After a skip: "Skipping to <title> by <artist>." — from what the queue
    said was next BEFORE the skip, so nothing is read back or waited for."""
    nxt = result.get("next") if isinstance(result, dict) else None
    if nxt and nxt.get("link"):
        return "Skipping — the DJ's up next."
    track = _track(nxt)
    return f"Skipping to {track}." if track else None


async def playing_brief(output: Any, ctx: Any, tools: dict[str, Any]) -> str | None:
    """"What's playing?" in one line: "Now playing: <title> by <artist>.",
    "Paused: …", "Nothing is playing.", or the link between songs. None when
    the state cannot be told in a line (loading, player down, uncatalogued)."""
    tool = tools.get("music_status")
    if tool is None or tool.data is None:
        return None
    now = await tool.data({})
    state = now.get("state")
    if state == "idle":
        return "Nothing is playing."
    if state == "link":
        return "Between songs right now — that's the DJ talking."
    track = _track(now)
    if state == "playing" and track:
        return f"Now playing: {track}."
    if state == "paused" and track:
        return f"Paused: {track}."
    return None


REPLY_FUNCTIONS: dict[str, ReplyFn] = {
    "skipping_to": skipping_to,
    "playing_brief": playing_brief,
}
