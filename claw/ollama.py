"""Ollama Chat API client.

POSTs to ``/api/chat`` with ``stream: false`` (per plan: no streaming for v0
— Matrix delivers block-by-block on its own, streaming saves no
user-visible latency and adds ~80 LOC of producer/consumer machinery).

The tool-use loop runs at most ``cfg.max_tool_turns`` cycles per inbound
message: on each cycle, if the assistant message returned ``tool_calls``,
execute each, append a ``role: tool`` message carrying the result, and call
``/api/chat`` again. Otherwise return the assistant text.

Transcript shape is OpenAI-flat (the Ollama-native shape): each turn is
``{role, content, tool_calls?, tool_call_id?}``.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx

from claw.config import OllamaConfig
from claw.tools.base import Tool, ollama_tool_spec

log = logging.getLogger("claw.ollama")

# Truncate per-call args in verbose tool-call logs. Long enough to identify
# which write/bash/etc. is which from the leading characters; short enough
# to keep an ``append_file`` with a multi-KB body from blowing up the log.
_VERBOSE_ARG_PREVIEW_CHARS = 300

# Big models with no streaming can be slow on cold cache; allow generous
# wall time per request. Now configurable via OllamaConfig.request_timeout_s
# (default 1800s); this constant is the fallback when no cfg is provided
# (tests, ad-hoc usage).
DEFAULT_TIMEOUT_S = 1800.0

# When a single tool result exceeds this many chars, we keep the model's
# in-loop view intact (it sees the full result on the next chat_once in this
# run_turn invocation) but persist only a truncated preview into the
# transcript that the caller appends. The full text is spooled to disk for
# tools whose output isn't otherwise reachable; for ``read_file`` (where
# the source is already on disk) we skip the spool copy and just point at
# the original path.
TOOL_RESULT_THRESHOLD_CHARS = 8192

# summarize() generation settings. See the method docstring for why the
# Modelfile defaults (presence_penalty 1.5, temperature 1.0) are wrong for
# summarization.
#
# num_predict must clear the largest word budget compaction.py can ask for —
# currently _MID_RECAP_WORDS_MAX (1500 words ≈ 2000 tokens) — with room to
# spare, or the recap is truncated mid-sentence right at the size where it
# matters most. With think=False there is no reasoning trace competing for
# the budget. Raise this if either ceiling in compaction.py goes up.
SUMMARY_PRESENCE_PENALTY = 0.0
SUMMARY_TEMPERATURE = 0.3
SUMMARY_NUM_PREDICT = 4096

# Same idea on the *tool-call* side: tools like write_file / append_file
# can carry multi-KB content in their argument JSON. We keep full args in
# the in-loop ``messages`` list (so the model can chain on its own
# emissions within the current run_turn) and in DEBUG logs, but stub any
# string-typed arg value longer than this when persisting to the JSONL
# transcript. Stopping the bloat from leaking into the next turn's history
# also keeps later flush slices from re-capturing the same content.
TOOL_CALL_ARGS_MAX_CHARS = 200


def _truncate_persisted_tool_call_args(
    tc: dict[str, Any], max_chars: int,
) -> dict[str, Any]:
    """Return a copy of ``tc`` with any string-typed argument value longer
    than ``max_chars`` replaced by ``<truncated: N chars>``. Applied only
    when writing the assistant tool-call row to the persistent transcript;
    the in-loop ``messages`` list and the DEBUG log keep full args.
    """
    fn = tc.get("function") or {}
    args_str = fn.get("arguments")
    if not isinstance(args_str, str):
        return tc
    try:
        args = json.loads(args_str)
    except (json.JSONDecodeError, ValueError):
        return tc
    if not isinstance(args, dict):
        return tc
    changed = False
    truncated_args: dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > max_chars:
            truncated_args[k] = f"<truncated: {len(v)} chars>"
            changed = True
        else:
            truncated_args[k] = v
    if not changed:
        return tc
    new_fn = dict(fn)
    new_fn["arguments"] = json.dumps(truncated_args, ensure_ascii=False)
    return {**tc, "function": new_fn}


async def _spool_and_truncate(
    result: str,
    *,
    sid: str,
    call_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
    workspace_dir: Path,
) -> str:
    """Return the persistence-side tool message content for ``result``.

    Below the threshold: returns ``result`` unchanged.
    Above: truncated preview + reference to either the original path
    (``read_file``) or a transient spool file under
    ``<workspace>/.tool-results/<sid>/<call_id>.txt``.
    """
    if len(result) <= TOOL_RESULT_THRESHOLD_CHARS:
        return result
    head = result[: TOOL_RESULT_THRESHOLD_CHARS - 1200]
    tail = result[-400:]

    if tool_name == "read_file":
        rel = tool_args.get("path", "?")
        return (
            f"[read_file result was {len(result):,} chars; only this preview "
            f"is in transcript. Source file remains at {rel} — no spool copy "
            f"made. To re-read regions: bash: sed -n 'M,Np' {rel} | head -200, "
            f"or bash: grep -n -C3 'pattern' {rel} | head -50]\n\n"
            f"--- preview (first {len(head):,} chars) ---\n{head}\n"
            f"--- (snip) ---\n{tail}\n--- end preview ---"
        )

    spool_dir = workspace_dir / ".tool-results" / sid
    spool_dir.mkdir(parents=True, exist_ok=True)
    spool_path = spool_dir / f"{call_id}.txt"
    spool_path.write_text(result)
    rel = spool_path.relative_to(workspace_dir)
    return (
        f"[tool result was {len(result):,} chars; only this preview is in "
        f"transcript. Full text saved at {rel} (TRANSIENT — wiped at next "
        f"session rotate, files >24h old only). To inspect: bash: head/tail/"
        f"sed/grep on that path. To keep long-term: bash: mkdir -p research "
        f"&& cp {rel} research/<name>.md]\n\n"
        f"--- preview (first {len(head):,} chars) ---\n{head}\n"
        f"--- (snip) ---\n{tail}\n--- end preview ---"
    )

# One-shot system note injected after a 5xx so the model can recover from
# the most common parse failure (malformed XML in a tool-call parameter
# value, e.g. literal ``</parameter>`` text).
_PARSE_RECOVERY_NOTE = (
    "Your previous tool call could not be parsed by Ollama (likely a "
    "malformed XML tag, e.g. an unescaped special character in a "
    "parameter value). Re-emit a clean tool call now, or reply with "
    "plain text if no tool is needed."
)

# One-shot system note injected when Ollama returns done_reason="length"
# with safe (non-code) partial content. The partial gets re-fed as an
# assistant turn so the model sees its own truncated output, and this note
# tells it to choose a recovery strategy. The model's response to this
# note IS its decision — there's no separate "decision" mechanism.
_LENGTH_RECOVERY_NOTE_WITH_PARTIAL = (
    "Your previous reply was cut off — you reached the output token cap. "
    "The partial above is everything you generated so far. Now choose: "
    "(1) if the partial already conveys a complete usable answer, finish "
    "it cleanly in a few sentences; (2) if you were mid-thinking and the "
    "answer would be too long, restart with a tighter scope or shorter "
    "format; (3) if the question is genuinely too big, reply briefly "
    "explaining that and ask the user to narrow it down."
)

# Variant for the thinking-budget-exhaustion case: done_reason="length"
# with content="" — the model spent the entire generation budget inside
# its <think> trace and emitted no post-think prose. There's nothing to
# wrap up, so we drop the "finish the partial" option and the empty
# assistant turn isn't injected (some /api/chat parsers misbehave on
# empty assistant content, and there's no information in it anyway).
_LENGTH_RECOVERY_NOTE_EMPTY = (
    "Your previous reply produced no output before hitting the output "
    "token cap — your reasoning consumed the entire generation budget "
    "without producing a final answer. Now choose: (1) restart with a "
    "tighter scope or shorter format that fits within the budget, or "
    "(2) reply briefly explaining that the question is too big to "
    "answer at this depth and ask the user to narrow it down."
)

# One-shot system note for the "model finished naturally with empty
# content + no tool_calls" case (typically done_reason="stop"). The
# user only sees text replies, not tool calls or thinking traces, so
# an empty turn is invisible to them — they can't tell whether the
# work happened, failed, or is still in progress. Re-prompt for a
# brief summary or direct answer.
_EMPTY_REPLY_RECOVERY_NOTE = (
    "Your previous reply was empty — you finished the turn without "
    "sending any text to the user. The user can only see your text "
    "replies, not your tool calls or thinking. Reply now with a brief "
    "summary of what you did or the answer to their question."
)

# Terminal fallback for the same case when recovery has already been spent
# this run_turn (it is one-shot, so a SECOND empty lands here). There is no
# legitimate way to reach it: no tool_calls and no text means the model
# neither spoke nor acted.
#
# Deliberately not an exception. run_turn's first return value is the
# transcript rows, and agent.py's `except Exception` handler returns BEFORE
# appending them — raising would discard the whole turn's record, so the
# tool calls would have run with nothing to show they ever did. Same
# sentinel-string convention as the tool-loop ceiling and the code-fence
# discard.
#
# The wording must not invite the USER to re-ask: by this point the tools
# have already run and their side effects are real (files written, edits
# applied). Only the closing summary was lost, so "try again" would push
# them into duplicating completed work.
#
# It must not claim nothing was retried either — the one-shot recovery
# retry above is exactly how this line is reached. The internal retry and
# re-running the user's request are different things, and only the second
# is unnecessary, so the text speaks to that one.
_EMPTY_REPLY_FALLBACK = (
    "[claw: the work above completed, but I finished without writing a "
    "reply. Nothing needs re-running — ask me to summarize it.]"
)


def _label_prefix(label: str, verbose_suffix: str = "") -> str:
    """Format the caller-supplied label as a log prefix. Empty string when
    no label is provided.

    ``label`` is shown in every mode. ``verbose_suffix`` is appended
    (joined with ``:``) only when the ``claw.ollama`` logger is at DEBUG.
    Callers should put the always-useful information in ``label`` (agent
    id, kind, peer name, task id) and any verbose-only correlation
    handle (session id, internal sid) in ``verbose_suffix``.
    """
    if not label:
        return ""
    if verbose_suffix and log.isEnabledFor(logging.DEBUG):
        return f"[{label}:{verbose_suffix}] "
    return f"[{label}] "


def _has_unclosed_code_fence(text: str) -> bool:
    """Conservative truncation detector. Counts non-overlapping ``` markers
    and flags if odd. Errs toward false positives (over-flagging): false
    positives produce a structured error to the caller (safe), while false
    negatives would let truncated code through (dangerous). Tested against
    open/closed 3- and 4-backtick fences, mid-content truncation, and
    inline backticks; the only false-positive cases are 4+4 fences with
    a literal ``` inside, and unbalanced 5+ backtick blobs — both very
    rare in real LLM output and both fail-safe."""
    return text.count("```") % 2 == 1


class OllamaClient:
    def __init__(self, cfg: OllamaConfig, timeout_s: float | None = None) -> None:
        self.cfg = cfg
        self.max_tool_turns = cfg.max_tool_turns
        # Caller can pin an explicit timeout (tests); otherwise honor cfg.
        effective_timeout = timeout_s if timeout_s is not None else cfg.request_timeout_s
        self._client = httpx.AsyncClient(base_url=cfg.base_url, timeout=effective_timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat_once(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        label: str = "",
        think: bool | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            body["tools"] = tools
        if options:
            body["options"] = options
        # ``think`` is omitted unless a caller asks, so the default stays the
        # model's own (thinking on) for every conversational and curation
        # turn. Only a caller that DISCARDS the trace should pass False.
        if think is not None:
            body["think"] = think
        resp = await self._client.post("/api/chat", json=body)
        if resp.status_code == 400 and "think" in body:
            # Same guard summarize() carries: a model with no thinking mode
            # rejects the field outright. Every model we ship is a Qwen3
            # hybrid, so this is protection for a future swap, not a path we
            # expect to take.
            log.warning(
                "%s%s rejected think=%s; retrying without it",
                _label_prefix(label), model, body["think"],
            )
            body.pop("think")
            resp = await self._client.post("/api/chat", json=body)
        if resp.status_code >= 500:
            # 5xx from /api/chat is usually a tool-call parser failure
            # against the model's raw output (qwen3coder.go / qwen35.go
            # XML parsing). Capture enough state to diagnose without
            # logging the whole transcript.
            last_assistant = next(
                (m for m in reversed(messages) if m.get("role") == "assistant"),
                None,
            )
            preview = ""
            if last_assistant is not None:
                preview = (last_assistant.get("content") or "")[:500]
            log.warning(
                "%sollama 5xx model=%s status=%d body=%r last_assistant_content_preview=%r",
                _label_prefix(label), model, resp.status_code, resp.text[:500], preview,
            )
        resp.raise_for_status()
        return resp.json()

    async def run_turn(
        self,
        *,
        model: str,
        history: list[dict[str, Any]],
        system: str | None,
        tools: dict[str, Tool],
        sid: str,
        workspace_dir: Path,
        label: str,
        verbose_suffix: str = "",
        num_predict: int | None = None,
        max_tool_turns: int | None = None,
        on_thinking: Callable[[str], Awaitable[None]] | None = None,
        drain_inbox: Callable[[], list[dict[str, Any]]] | None = None,
        think: bool | None = None,
    ) -> tuple[list[dict[str, Any]], str, str]:
        """Drive ``/api/chat`` until the model stops requesting tools.

        Returns ``(new_messages, final_text, final_thinking)``.
        ``new_messages`` is the list of OpenAI-flat messages to append to the
        transcript (assistant turns with tool_calls, role=tool result turns,
        and the final assistant text). ``final_text`` is the text of the last
        assistant turn for sending to the user. ``final_thinking`` is that
        same final no-tool-call turn's ``message.thinking`` (Ollama's
        chain-of-trace), or ``""`` when the model emitted none or a
        synthetic-string branch fired (truncation discard / tool-loop
        ceiling). It is ephemeral — captured for optional out-of-band display
        only; it is **never** placed in ``new_messages`` and never persisted
        to the transcript.

        ``drain_inbox``, when given, is called once per tool turn and returns
        OpenAI-flat rows to append before the next model call — the mechanism
        by which a subagent completion reaches the turn that spawned it rather
        than the one after it. It must be synchronous and must not await, so
        the pop-and-mutate of the caller's queue stays atomic under asyncio.

        ``sid`` and ``workspace_dir`` are used to spool oversized tool
        results into ``<workspace>/.tool-results/<sid>/<call_id>.txt``: the
        in-loop ``messages`` list keeps the full result so the model can
        reason on it across this run_turn invocation, but ``new_messages``
        gets the truncated preview so subsequent inbounds don't re-pay for
        the full content. See ``_spool_and_truncate``.

        ``num_predict`` caps each /api/chat call's generated tokens. The
        recovery branch fires once per ``run_turn`` when the model emits
        no tool_calls AND either (a) hits the cap (``done_reason ==
        "length"``) — claw inspects the partial, fails loud on an unclosed
        code fence, otherwise re-feeds the partial + a length-recovery
        note so the model can wrap up, restart tighter, or bail; or (b)
        finishes with empty content under any other ``done_reason``
        (typically ``"stop"``) — claw re-prompts with
        ``_EMPTY_REPLY_RECOVERY_NOTE`` so the model produces something
        the user can actually see. The truncated partial only ever lives
        in the in-loop ``messages``, never in ``new_messages`` (transcript).

        Recovery is one-shot per ``run_turn``. A SECOND empty (no
        tool_calls, no text) has nothing left to try and is returned as
        ``_EMPTY_REPLY_FALLBACK`` rather than ``""`` — an empty string
        reaches the user as silence, which is indistinguishable from
        being ignored. The sentinel is substituted before the assistant
        record is built, so the transcript row matches what was sent.

        ``label`` is a caller-supplied prefix prepended to every
        ``claw.ollama`` log line emitted from this call. Always shown.
        Format convention: ``<agent_id>:<kind>[:<peer_or_task>]`` —
        e.g. ``"agent-1:main:user-1"``, ``"agent-1:flush:periodic-growth:user-1"``,
        ``"agent-1:subagent:persona-3-a1b2c3d4"``.

        ``verbose_suffix`` is an optional addendum (joined with ``:``)
        appended only when ``claw.ollama`` is at DEBUG. Use it for
        correlation handles that are noisy in normal operator scans —
        typically the matrix-room sid for the main / flush call sites,
        empty for subagents (their task_id is already in ``label``).

        ``max_tool_turns`` overrides the client-wide default (set from
        ``OllamaConfig.max_tool_turns`` at construction) for this call.
        Callers pass the agent's or persona's resolved value; ``None``
        falls back to the client-wide default.

        ``on_thinking`` is an optional **send-only** sink, awaited once per
        ``/api/chat`` call with that call's ``message.thinking`` (skipped
        when blank) — i.e. EVERY loop iteration's reasoning, not just the
        final turn's. It's the every-step (``%thinking full``) surface;
        ``final_thinking`` (the return value) is the final-only surface.
        Granularity ceiling is per call, not per token (we're
        ``stream: false``). The callback only ever *emits*: it is never
        appended to ``new_messages`` or the transcript, so ephemerality
        holds by construction even though it fires mid-loop (before the
        caller's transcript append) rather than after the cascade. A
        truncated turn that then recovers fires twice (pre- + post-recovery
        reasoning) — intended for the debug-honest ``full`` surface; the
        final-only return value still carries just the recovered turn's.
        """
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(history)
        new_messages: list[dict[str, Any]] = []
        # Final no-tool-call turn's chain-of-trace. Ephemeral: captured for
        # optional display by the caller, never appended to new_messages /
        # the transcript. Reassigned at each assistant-message parse so the
        # value reflects the LAST parsed turn (incl. a recovery re-parse).
        final_thinking: str = ""

        tool_specs = ollama_tool_spec(tools) if tools else None
        options: dict[str, Any] | None = None
        if num_predict is not None and num_predict != -1:
            options = {"num_predict": num_predict}
        parse_retry_used = False
        recovery_used = False
        effective_max_tool_turns = (
            max_tool_turns if max_tool_turns is not None else self.max_tool_turns
        )

        prefix = _label_prefix(label, verbose_suffix)

        async def chat_with_parse_recovery(turn_idx: int) -> dict[str, Any]:
            """``chat_once`` with a single guarded re-prompt on a 5xx.

            A 5xx from ``/api/chat`` is almost always Ollama failing to parse
            the model's raw tool-call output (malformed XML, e.g. a
            ``<function>`` closed by ``</parameter>``). Inject a one-shot
            recovery note and retry once. ``parse_retry_used`` is shared
            across EVERY ``chat_once`` in this ``run_turn`` — the top-of-loop
            call AND the empty-reply/length recovery-branch call — so a parser
            500 that lands on a recovery retry gets the same guarded re-prompt
            instead of propagating and killing the whole turn (the
            stacked-recovery gap fixed 2026-07-25: previously only the
            top-of-loop call was wrapped, so a malformed tool call emitted
            during empty-reply recovery crashed run_turn).
            """
            nonlocal parse_retry_used
            try:
                return await self.chat_once(
                    model=model, messages=messages, tools=tool_specs, options=options,
                    label=label, think=think,
                )
            except httpx.HTTPStatusError as e:
                if 500 <= e.response.status_code < 600 and not parse_retry_used:
                    parse_retry_used = True
                    log.info(
                        "%sturn %d: ollama 5xx; injecting recovery note and retrying once",
                        prefix, turn_idx,
                    )
                    messages.append({"role": "system", "content": _PARSE_RECOVERY_NOTE})
                    return await self.chat_once(
                        model=model, messages=messages, tools=tool_specs, options=options,
                        label=label, think=think,
                    )
                raise

        for turn_idx in range(effective_max_tool_turns):
            # Mid-turn inbox, drained before the model picks its next action.
            #
            # Without this the tool loop is a closed system: it can only learn
            # what was true when the turn started, plus its own tool results.
            # A subagent that finishes here has nowhere to report — its
            # completion sits in the agent's pending queue behind the session
            # lock this very turn is holding, and is not seen until the turn
            # ends. On a 50-tool-turn chain that is an hour late, by which
            # point the agent has usually redone or abandoned the work.
            #
            # Injected at the TAIL, so the KV prefix built by every preceding
            # row survives (same rule as retrieved memory). Injected at the TOP
            # of the iteration, so it lands after the previous iteration's tool
            # results are all appended — never between a tool_call and its
            # result, which would break tool atomicity and the compaction
            # cut-point invariant.
            if drain_inbox is not None:
                injected = drain_inbox()
                if injected:
                    messages.extend(injected)
                    new_messages.extend(injected)
                    log.info(
                        "%sturn %d: injected %d inbound row(s) mid-turn",
                        prefix, turn_idx, len(injected),
                    )
            response = await chat_with_parse_recovery(turn_idx)
            assistant_msg = response.get("message", {}) or {}
            content = assistant_msg.get("content", "") or ""
            final_thinking = assistant_msg.get("thinking", "") or ""
            # Per-iteration reasoning surface (%thinking full). Send-only;
            # the recovery re-parse below fires it again for a recovered
            # turn (pre- + post-recovery, by design for the debug surface).
            if on_thinking and final_thinking.strip():
                await on_thinking(final_thinking.strip())
            tool_calls = assistant_msg.get("tool_calls") or []
            done_reason = response.get("done_reason")

            # Recovery branch: model returned no tool_calls and no
            # actionable output. Two trigger cases:
            #   (a) done_reason="length": model hit the output token cap.
            #       Re-feed the partial (or just a note, if empty) so the
            #       model can wrap up, restart tighter, or bail.
            #   (b) empty content with any other done_reason (typically
            #       "stop"): model finished without sending text. The user
            #       sees nothing — re-prompt for a final reply.
            # When tool_calls are present, the model is mid-work; let the
            # loop proceed normally even if content is empty (the partial
            # content is just pre-tool-call narration). The flag bounds
            # recovery to once per run_turn so a still-degenerate retry
            # doesn't loop.
            empty_reply = not content.strip()
            if (
                not tool_calls
                and not recovery_used
                and (done_reason == "length" or empty_reply)
            ):
                if done_reason == "length" and _has_unclosed_code_fence(content):
                    log.warning(
                        "%sturn %d: done_reason=length with unclosed code fence "
                        "(content=%d chars); discarding partial",
                        prefix, turn_idx, len(content),
                    )
                    return new_messages, (
                        "[claw: output truncated mid-code-fence; partial discarded; "
                        "retry with smaller scope]"
                    ), ""
                recovery_used = True
                if done_reason == "length":
                    log.info(
                        "%sturn %d: done_reason=length (content=%d chars, no tool_calls); "
                        "injecting continuation note and retrying once",
                        prefix, turn_idx, len(content),
                    )
                    # Re-feed partial as assistant + recovery note as
                    # system. Empty-partial case (thinking-budget
                    # exhaustion) skips the empty assistant turn and uses
                    # the EMPTY-variant note — drops the nonsensical
                    # "finish the partial" option.
                    if content:
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "system",
                            "content": _LENGTH_RECOVERY_NOTE_WITH_PARTIAL,
                        })
                    else:
                        messages.append({
                            "role": "system",
                            "content": _LENGTH_RECOVERY_NOTE_EMPTY,
                        })
                else:
                    log.info(
                        "%sturn %d: done_reason=%s with empty content and no "
                        "tool_calls; injecting empty-reply recovery note "
                        "and retrying once",
                        prefix, turn_idx, done_reason,
                    )
                    messages.append({
                        "role": "system",
                        "content": _EMPTY_REPLY_RECOVERY_NOTE,
                    })
                response = await chat_with_parse_recovery(turn_idx)
                assistant_msg = response.get("message", {}) or {}
                content = assistant_msg.get("content", "") or ""
                # Recovery replaced content — reassign thinking from the SAME
                # message so a recovered reply carries its own trace, not the
                # pre-recovery turn's.
                final_thinking = assistant_msg.get("thinking", "") or ""
                if on_thinking and final_thinking.strip():
                    await on_thinking(final_thinking.strip())
                tool_calls = assistant_msg.get("tool_calls") or []
                # Likewise done_reason: it is read below by the terminal
                # empty-reply guard's log line, and the retry's reason is
                # the one that describes how this turn actually ended.
                done_reason = response.get("done_reason")

            # Terminal empty-reply guard. Recovery above is one-shot, so a
            # second empty in the same run_turn arrives here with nothing
            # left to try. Returning `content` unchanged would hand agent.py
            # an empty string, and it only sends on a truthy reply — so the
            # turn ends in silence and the user cannot distinguish it from
            # being ignored. Substitute the sentinel BEFORE assistant_record
            # is built so the persisted row carries it too: an empty
            # assistant row is a degenerate example that stays in the
            # transcript for the rest of the session, and the row should
            # match what the user was actually shown.
            if not tool_calls and not content.strip():
                log.warning(
                    "%sturn %d: empty reply with no tool_calls and recovery "
                    "already used (done_reason=%s); substituting fallback text",
                    prefix, turn_idx, done_reason,
                )
                content = _EMPTY_REPLY_FALLBACK

            assistant_record: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_record["tool_calls"] = [
                    _normalize_tool_call(tc, turn_idx, i) for i, tc in enumerate(tool_calls)
                ]
            messages.append(assistant_record)
            # Diverge: persist a copy with long string args stubbed so
            # subsequent turns (and later flush slices) don't carry the
            # full tool-call payload. The in-loop ``messages`` above keeps
            # full args for the model's own chaining within this run_turn.
            if tool_calls:
                persisted_record = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        _truncate_persisted_tool_call_args(tc, TOOL_CALL_ARGS_MAX_CHARS)
                        for tc in assistant_record["tool_calls"]
                    ],
                }
                new_messages.append(persisted_record)
            else:
                new_messages.append(assistant_record)

            if not tool_calls:
                return new_messages, content, final_thinking

            tc_counts = Counter(
                ((tc.get("function") or {}).get("name") or "?")
                for tc in assistant_record["tool_calls"]
            )
            tc_summary = ", ".join(
                f"{name}({n})" for name, n in tc_counts.most_common()
            )
            log.info(
                "%sturn %d: %d tool_call(s) requested: %s",
                prefix, turn_idx, len(tool_calls), tc_summary,
            )
            if log.isEnabledFor(logging.DEBUG):
                for tc in assistant_record["tool_calls"]:
                    fn = tc.get("function") or {}
                    name = fn.get("name") or "?"
                    args_str = json.dumps(
                        fn.get("arguments") or {},
                        default=str, ensure_ascii=False,
                    )
                    if len(args_str) > _VERBOSE_ARG_PREVIEW_CHARS:
                        args_str = (
                            args_str[: _VERBOSE_ARG_PREVIEW_CHARS - 3] + "..."
                        )
                    log.debug("%s  %s(%s)", prefix, name, args_str)
            for tc in assistant_record["tool_calls"]:
                result_text = await _run_one_tool(tc, tools)
                fn = tc.get("function", {}) or {}
                # Diverge: the model sees the full result on the next
                # chat_once in this loop; the transcript only carries the
                # truncated preview so subsequent inbounds don't re-pay.
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_text,
                })
                new_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": await _spool_and_truncate(
                        result_text,
                        sid=sid,
                        call_id=tc["id"],
                        tool_name=fn.get("name", ""),
                        tool_args=fn.get("arguments", {}) or {},
                        workspace_dir=workspace_dir,
                    ),
                })

        log.warning(
            "%shit max_tool_turns=%d without final assistant text",
            prefix, effective_max_tool_turns,
        )
        return new_messages, "[hit tool-use limit; please try again]", ""

    async def summarize(
        self,
        *,
        model: str,
        instruction: str,
        transcript_text: str,
    ) -> str:
        """One non-tool, non-streaming summary call (used by compaction).

        ``think=False``. Every chat model in the stack is a Qwen3 hybrid, and
        this call discards the reasoning trace entirely — it returns
        ``message.content`` only, never ``message.thinking``. Measured on a
        166-row transcript: 25.3 s / 1553 eval tokens with thinking on,
        against 5.4 s / 306 with it off. 88% of the generation was produced
        and thrown away, and the idle recap runs on the **boot path**, where
        that latency blocks startup.

        Explicit ``options`` because the Modelfile's defaults are tuned for
        chat, not summarization. ``presence_penalty`` is the important one:
        it ships at 1.5 and penalizes every token already emitted, while a
        faithful recap has to repeat the same filenames, paths and terms
        throughout. ``temperature`` is dropped for the same reason — a
        summary wants fidelity to its input, not variety.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": f"{instruction}\n\n--- transcript ---\n{transcript_text}",
            }],
            "stream": False,
            "think": False,
            "options": {
                "presence_penalty": SUMMARY_PRESENCE_PENALTY,
                "temperature": SUMMARY_TEMPERATURE,
                "num_predict": SUMMARY_NUM_PREDICT,
            },
        }
        try:
            resp = await self._client.post("/api/chat", json=body)
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            # `think` is rejected by models with no thinking mode. Every model
            # we ship is a hybrid, so this is a guard for a future
            # compaction_model, not a path we expect to take: retry without it
            # rather than let compaction silently stop producing recaps
            # (maybe_idle_recap swallows the exception and returns None).
            log.warning(
                "summarize: %s rejected think=False; retrying without it", model,
            )
            body.pop("think")
            resp = await self._client.post("/api/chat", json=body)
            resp.raise_for_status()
        return (resp.json().get("message", {}).get("content") or "").strip()


def _normalize_tool_call(tc: dict[str, Any], turn_idx: int, slot: int) -> dict[str, Any]:
    """Ollama may return tool_calls without ids and with arguments as either
    a dict or a JSON string. Normalize to ``{id, type: "function",
    function: {name, arguments: dict}}`` for transcript pairing.
    """
    func = tc.get("function", {}) or {}
    name = func.get("name", "")
    raw_args = func.get("arguments", {})
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {"_raw": raw_args}
    else:
        args = raw_args or {}
    tc_id = tc.get("id") or f"call_{turn_idx}_{slot}"
    return {
        "id": tc_id,
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


async def _run_one_tool(tc: dict[str, Any], tools: dict[str, Tool]) -> str:
    func = tc["function"]
    name = func["name"]
    args = func.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return f"error: tool {name!r} arguments not valid JSON: {args!r}"
    tool = tools.get(name)
    if tool is None:
        return f"error: unknown tool {name!r}"
    try:
        return await tool.run(args) or "(empty)"
    except Exception as e:
        log.exception("tool %s failed", name)
        return f"error: {e}"
