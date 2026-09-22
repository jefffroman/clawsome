"""Decision gate: the outer loop in front of the LLM.

Every human turn (matrix or voice) passes through here before the LLM sees it.
The gate asks a System One decision scorer one question: *which registered
direct handler, if any, fully handles this message?* The options are the
handlers' own descriptions plus ``none``. A handler runs only if it is the
scorer's pick with at least ``min_confidence``; everything else — ``none``,
an unsure pick, a scorer that is down or slow — goes to the LLM exactly as it
would without a gate. The gate can only ever *remove* an LLM turn, never
change one.

A direct handler picks; it never extracts. The scorer returns a choice among
options it was given, not text, so a handler's action must be fixed or chosen
from an enumerable set ("pause the music", not "play <whatever was asked>").

With no handlers registered there is nothing to choose between, so the scorer
is never called and every eligible turn passes through — the gate is then
pure plumbing, which is its first deployed form.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from claw.channel.base import InboundMessage
from claw.channel.envelope import strip_inbound_envelope
from claw.config import GateConfig, GateHandlerConfig
from claw.gate_replies import REPLY_FUNCTIONS
from claw.systemone import SystemOneClient

log = logging.getLogger(__name__)

# The option that sends the turn to the LLM. Reserved: no handler may use it.
NONE = "none"

# Human conversation only. cron turns are stateless scheduled events,
# initial_prompt is the agent's own continuation after a restart, and a
# subagent completion is machine output being fed back — none of them is a
# request a direct handler should answer.
ELIGIBLE_CHANNELS = ("matrix", "voice")

# Transcript rows written in the user slot that no user wrote: compaction
# recaps, cron mirrors, out-of-band notices. The scorer must not read them as
# things the user said.
_SYNTHETIC_PREFIXES = (
    "## Pre-compaction Recap",
    "⚙️ System note",
    "[SYSTEM (out-of-band",
)

_QUESTION = "route"
_INSTRUCTIONS = (
    "The last user message in the conversation is a new request. Which option "
    "fully handles that request on its own? Choose 'none' unless one option "
    "does everything the user asked."
)
_NONE_DESCRIPTION = (
    "None of these: the request needs a conversational reply, more than one "
    "action, or anything not listed."
)


class DirectHandler(Protocol):
    """A turn the gate may answer without the LLM.

    ``description`` is what the scorer reads to decide, so write it as the
    request it handles ("Pause the music or other audio that is playing"), not
    as the implementation. ``handle`` performs the action and returns the
    reply text, which is delivered and recorded exactly like an LLM reply — or
    ``None`` to decline, sending the turn to the LLM with nothing written.
    ``min_confidence`` (optional; None = the gate's) is this handler's bar.
    """

    id: str
    description: str
    min_confidence: float | None

    async def handle(self, ctx: "GateContext") -> str | None: ...


@dataclass(frozen=True)
class GateContext:
    agent_id: str
    sid: str
    msgs: list[InboundMessage]
    # The combined message body as the user wrote it (no envelope).
    text: str
    # What the scorer saw: prior user/assistant messages plus this one.
    state: list[dict[str, str]]


@dataclass(frozen=True)
class Decision:
    """Where a turn goes, and why.

    ``handler`` set means "answer directly"; ``None`` means the LLM answers.
    ``reason`` is one of: ``ineligible`` (not a human turn), ``no-handlers``,
    ``none`` (the scorer picked no handler), ``low-confidence``,
    ``scorer-error``, ``chosen``.
    """

    handler: DirectHandler | None
    reason: str
    choice: str | None = None
    confidence: float | None = None
    ctx: GateContext | None = field(default=None, compare=False)


# claw's tool convention for an answer that is not a success.
_FAILURE_PREFIXES = ("error:", "refused:")


class ToolHandler:
    """A declarative handler: one tool, fixed arguments, a shaped reply.

    Built from a GateHandlerConfig against the agent's own tool registry.

    * Text shape: the tool's text output must be the configured ``expect``
      (or, with none, anything not starting ``error:``/``refused:``).
    * Outcomes shape: the tool's structured ``data`` twin runs instead — once;
      it IS the action — and its ``result`` picks the reply.

    Anything not accepted declines (returns None) and the LLM takes the turn,
    so a failed action is explained by the LLM, never covered by a cheerful
    fixed reply.
    """

    def __init__(self, cfg: GateHandlerConfig, tools: dict[str, Any]) -> None:
        self.cfg = cfg
        self.id = cfg.id
        self.description = cfg.description
        self.min_confidence = cfg.min_confidence
        self._tools = tools
        self._tool = tools[cfg.tool]

    async def handle(self, ctx: "GateContext") -> str | None:
        if self.cfg.outcomes:
            result = await self._tool.data(dict(self.cfg.args))
            rc = self.cfg.outcomes.get(str(result.get("result")))
            if rc is None:
                log.info("[%s] handler %s declined: %s result %r", ctx.agent_id,
                         self.id, self.cfg.tool, result.get("result"))
                return None
            return await self._reply(rc, result, ctx, text=None)

        out = (await self._tool.run(dict(self.cfg.args))).strip()
        if out.lower().startswith(_FAILURE_PREFIXES) or (
            self.cfg.expect is not None and out != self.cfg.expect
        ):
            log.info("[%s] handler %s declined: %s returned %r",
                     ctx.agent_id, self.id, self.cfg.tool, out[:120])
            return None
        return await self._reply(self.cfg.reply, out, ctx, text=out)

    async def _reply(self, rc: Any, answer: Any, ctx: "GateContext",
                     text: str | None) -> str | None:
        if rc.relay:
            return text
        if rc.reply:
            return random.choice(rc.reply)
        said = None
        try:
            said = await REPLY_FUNCTIONS[rc.reply_fn](answer, ctx, self._tools)
        except Exception:  # noqa: BLE001 — the action already happened
            log.exception("[%s] reply_fn %s failed; using its fallback",
                          ctx.agent_id, rc.reply_fn)
        if said:
            return said
        if rc.fallback:
            return random.choice(rc.fallback)
        # Text shape with no fallback: the tool's own words. (An outcome
        # always has a fallback — config validation requires it.)
        return text


def build_tool_handlers(
    cfgs: tuple[GateHandlerConfig, ...], tools: dict[str, Any], agent_id: str,
) -> tuple[ToolHandler, ...]:
    """Bind handler configs to this agent's tools. A handler naming a tool the
    agent does not have, an outcomes handler on a tool with no structured
    ``data`` twin, or one passing an argument the tool does not declare, is
    dropped with an ERROR: the gate fails open, and a typo in one handler must
    not keep the gateway from starting."""
    bound = []
    for c in cfgs:
        tool = tools.get(c.tool)
        if tool is None:
            log.error("[%s] gate handler %r names tool %r, which this agent "
                      "does not have; handler dropped", agent_id, c.id, c.tool)
            continue
        if c.outcomes and getattr(tool, "data", None) is None:
            log.error("[%s] gate handler %r uses outcomes, but tool %r has no "
                      "structured data; handler dropped", agent_id, c.id, c.tool)
            continue
        # args are a plain dict in config and are passed straight to the tool,
        # which reads the keys it knows and ignores the rest — so a misspelled
        # key would otherwise load clean and fail once per turn, silently, as
        # a call with that argument simply missing.
        declared = (getattr(tool, "input_schema", None) or {}).get("properties")
        if declared is not None and (unknown := sorted(set(c.args) - set(declared))):
            log.error("[%s] gate handler %r passes %s to tool %r, which does not "
                      "declare %s; handler dropped (known: %s)", agent_id, c.id,
                      unknown, c.tool, "them" if len(unknown) > 1 else "it",
                      ", ".join(sorted(declared)) or "no arguments")
            continue
        bound.append(ToolHandler(c, tools))
    return tuple(bound)


def eligible(msgs: list[InboundMessage]) -> bool:
    """One human message on a conversational channel.

    A batch of several is left to the LLM: merged messages can carry more
    than one request, and a direct handler answers exactly one.
    """
    return (
        len(msgs) == 1
        and msgs[0].channel in ELIGIBLE_CHANNELS
        and not msgs[0].is_subagent_completion
    )


def build_state(
    rows: list[dict[str, Any]], current: str, context_turns: int,
) -> list[dict[str, str]]:
    """The last ``context_turns`` user/assistant messages, then ``current``.

    Tool traffic (tool rows, assistant rows that only carry tool_calls) and
    synthetic user-slot notes are dropped, and envelopes are stripped, so the
    scorer reads the conversation as the people in it had it.
    """
    history: list[dict[str, str]] = []
    for row in rows:
        role, content = row.get("role"), row.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        if role == "user":
            if content.startswith(_SYNTHETIC_PREFIXES):
                continue
            content = strip_inbound_envelope(content)
        history.append({"role": role, "content": content})
    kept = history[-context_turns:] if context_turns > 0 else []
    return kept + [{"role": "user", "content": current}]


class DecisionGate:
    def __init__(
        self, cfg: GateConfig, client: SystemOneClient | None,
        handlers: tuple[DirectHandler, ...] = (),
    ) -> None:
        ids = [h.id for h in handlers]
        if NONE in ids:
            raise ValueError(f"handler id {NONE!r} is reserved")
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate handler ids: {sorted(ids)}")
        if handlers and client is None:
            raise ValueError("handlers registered but no scorer client")
        self.cfg = cfg
        self.client = client
        self.handlers = {h.id: h for h in handlers}

    async def decide(
        self, agent_id: str, sid: str, msgs: list[InboundMessage], text: str,
        load_history: Callable[[], list[dict[str, Any]]],
    ) -> Decision:
        """Route one turn. Never raises: every failure is a pass-through."""
        if not eligible(msgs):
            return Decision(None, "ineligible")
        if not self.handlers:
            return Decision(None, "no-handlers")

        state = build_state(load_history(), text, self.cfg.context_turns)
        ctx = GateContext(agent_id=agent_id, sid=sid, msgs=msgs, text=text, state=state)
        criteria = {hid: h.description for hid, h in self.handlers.items()}
        criteria[NONE] = _NONE_DESCRIPTION
        try:
            answers = await self.client.ask(state, {_QUESTION: {
                "type": "choice", "instructions": _INSTRUCTIONS, "criteria": criteria,
            }}, timeout_s=self.cfg.timeout_s)
            answer = answers[_QUESTION]
            choice, confidence = answer["choice"], float(answer["confidence"])
        except Exception as e:  # noqa: BLE001 — fail open, whatever went wrong
            log.warning("[%s] gate scorer failed, passing through: %r", agent_id, e)
            return Decision(None, "scorer-error", ctx=ctx)

        if choice == NONE or choice not in self.handlers:
            return Decision(None, "none", choice=choice, confidence=confidence, ctx=ctx)
        handler = self.handlers[choice]
        floor = getattr(handler, "min_confidence", None)
        if confidence < (self.cfg.min_confidence if floor is None else floor):
            return Decision(None, "low-confidence", choice=choice,
                            confidence=confidence, ctx=ctx)
        return Decision(handler, "chosen", choice=choice,
                        confidence=confidence, ctx=ctx)
