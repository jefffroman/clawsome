"""Startup ``.initial_prompt.md`` hook.

For each agent, on boot — after the Matrix channel has started and the
scheduler is up — check ``<workspace>/.initial_prompt.md``. If present,
dispatch its contents as a synthetic ``InboundMessage`` and, on success,
remove the file. On failure, leave the file in place for the next attempt.

Used by the ``restart`` skill: an agent writes ``.initial_prompt.md``
inside its workspace before triggering a gateway restart, so the new
gateway process picks up the conversation thread instead of needing the
operator to send the next prompt manually.
"""

from __future__ import annotations

import logging
from typing import Any

from claw.channel.base import InboundMessage

log = logging.getLogger("claw.triggers.initial_prompt")


async def maybe_dispatch_initial_prompt(agent: Any) -> bool:
    """Returns True if a prompt was dispatched; False otherwise.

    Typed ``agent: Any`` to avoid the circular import — caller is main.py
    which already has the real ``Agent`` instances.
    """
    path = agent.agent_cfg.workspace / ".initial_prompt.md"
    if not path.is_file():
        return False
    try:
        content = path.read_text().strip()
    except OSError:
        log.exception("[%s] couldn't read %s", agent.id, path)
        return False
    if not content:
        log.info("[%s] %s is empty; removing", agent.id, path)
        try:
            path.unlink()
        except OSError:
            pass
        return False

    log.info("[%s] dispatching initial prompt (%d chars)", agent.id, len(content))
    msg = InboundMessage(
        peer_id="bootstrap",
        sender_name="restart",
        text=content,
        channel="initial_prompt",
    )
    try:
        await agent.handle_inbound(msg)
    except Exception:
        log.exception("[%s] initial_prompt dispatch failed; leaving %s in place", agent.id, path)
        return False

    try:
        path.unlink()
        log.info("[%s] consumed and removed %s", agent.id, path)
    except OSError:
        log.exception("[%s] couldn't remove %s after dispatch", agent.id, path)
    return True
