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
from pathlib import Path
from typing import Any

import httpx

from claw.config import OllamaConfig
from claw.tools.base import Tool, ollama_tool_spec

log = logging.getLogger("claw.ollama")

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
_LENGTH_RECOVERY_NOTE = (
    "Your previous reply was cut off — you reached the output token cap. "
    "The partial above is everything you generated so far. Now choose: "
    "(1) if the partial already conveys a complete usable answer, finish "
    "it cleanly in a few sentences; (2) if you were mid-thinking and the "
    "answer would be too long, restart with a tighter scope or shorter "
    "format; (3) if the question is genuinely too big, reply briefly "
    "explaining that and ask the user to narrow it down."
)


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
                "ollama 5xx model=%s status=%d body=%r last_assistant_content_preview=%r",
                model, resp.status_code, resp.text[:500], preview,
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
        num_predict: int | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """Drive ``/api/chat`` until the model stops requesting tools.

        Returns ``(new_messages, final_text)``. ``new_messages`` is the list
        of OpenAI-flat messages to append to the transcript (assistant turns
        with tool_calls, role=tool result turns, and the final assistant
        text). ``final_text`` is the text of the last assistant turn for
        sending to the user.

        ``sid`` and ``workspace_dir`` are used to spool oversized tool
        results into ``<workspace>/.tool-results/<sid>/<call_id>.txt``: the
        in-loop ``messages`` list keeps the full result so the model can
        reason on it across this run_turn invocation, but ``new_messages``
        gets the truncated preview so subsequent inbounds don't re-pay for
        the full content. See ``_spool_and_truncate``.

        ``num_predict`` caps each /api/chat call's generated tokens. When a
        call hits the cap (``done_reason == "length"``) the partial is
        inspected: an unclosed code fence triggers an immediate fail-loud
        return; otherwise claw injects the partial back as an assistant
        turn plus a recovery system note and re-calls /api/chat once so
        the model can wrap up, restart tighter, or bail. The truncated
        partial only ever lives in the in-loop ``messages``, never in
        ``new_messages`` (transcript).
        """
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(history)
        new_messages: list[dict[str, Any]] = []

        tool_specs = ollama_tool_spec(tools) if tools else None
        options: dict[str, Any] | None = None
        if num_predict is not None and num_predict != -1:
            options = {"num_predict": num_predict}
        parse_retry_used = False
        length_recovery_used = False

        for turn_idx in range(self.max_tool_turns):
            try:
                response = await self.chat_once(
                    model=model, messages=messages, tools=tool_specs, options=options,
                )
            except httpx.HTTPStatusError as e:
                if 500 <= e.response.status_code < 600 and not parse_retry_used:
                    parse_retry_used = True
                    log.info(
                        "turn %d: ollama 5xx; injecting recovery note and retrying once",
                        turn_idx,
                    )
                    messages.append({"role": "system", "content": _PARSE_RECOVERY_NOTE})
                    response = await self.chat_once(
                        model=model, messages=messages, tools=tool_specs, options=options,
                    )
                else:
                    raise
            assistant_msg = response.get("message", {}) or {}
            content = assistant_msg.get("content", "") or ""
            tool_calls = assistant_msg.get("tool_calls") or []
            done_reason = response.get("done_reason")

            # Length-recovery branch: only fires for the "model was producing
            # final-answer prose and got cut off" case (no tool_calls). When
            # tool_calls are present, the model is mid-work — let the tool
            # loop proceed normally; the partial content is just narration
            # leading up to the call. The flag bounds recovery to once per
            # run_turn so a still-truncated retry doesn't loop.
            if done_reason == "length" and not tool_calls and not length_recovery_used:
                if _has_unclosed_code_fence(content):
                    log.warning(
                        "turn %d: done_reason=length with unclosed code fence "
                        "(content=%d chars); discarding partial",
                        turn_idx, len(content),
                    )
                    return new_messages, (
                        "[claw: output truncated mid-code-fence; partial discarded; "
                        "retry with smaller scope]"
                    )
                length_recovery_used = True
                log.info(
                    "turn %d: done_reason=length (content=%d chars, no tool_calls); "
                    "injecting continuation note and retrying once",
                    turn_idx, len(content),
                )
                # Re-feed partial as assistant + recovery note as system.
                # This second call IS the agent's decision mechanism.
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "system", "content": _LENGTH_RECOVERY_NOTE})
                response = await self.chat_once(
                    model=model, messages=messages, tools=tool_specs, options=options,
                )
                assistant_msg = response.get("message", {}) or {}
                content = assistant_msg.get("content", "") or ""
                tool_calls = assistant_msg.get("tool_calls") or []

            assistant_record: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_record["tool_calls"] = [
                    _normalize_tool_call(tc, turn_idx, i) for i, tc in enumerate(tool_calls)
                ]
            new_messages.append(assistant_record)
            messages.append(assistant_record)

            if not tool_calls:
                return new_messages, content

            log.info("turn %d: %d tool_call(s) requested", turn_idx, len(tool_calls))
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

        log.warning("hit max_tool_turns=%d without final assistant text", self.max_tool_turns)
        return new_messages, "[hit tool-use limit; please try again]"

    async def summarize(
        self,
        *,
        model: str,
        instruction: str,
        transcript_text: str,
    ) -> str:
        """One non-tool, non-streaming summary call (used by compaction)."""
        resp = await self._client.post("/api/chat", json={
            "model": model,
            "messages": [{
                "role": "user",
                "content": f"{instruction}\n\n--- transcript ---\n{transcript_text}",
            }],
            "stream": False,
        })
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
