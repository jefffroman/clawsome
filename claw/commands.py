"""In-band admin commands (control plane).

Parsed out of Matrix message bodies *before* they reach the LLM turn
pipeline. A message is only ever treated as a command when the gateway has
commands enabled, the channel is matrix, the sender is on the command
allowlist, and the body parses here. Anything else is ordinary text.

This module is pure: parsing + the known-command set + usage string. The
side-effecting dispatch (clear / compact / verbose / context) lives on the
Agent so future commands are a one-line registry addition there.
"""

from __future__ import annotations

from dataclasses import dataclass

# Known command words. Dispatch for each lives in Agent._handle_command.
KNOWN: tuple[str, ...] = (
    "compact", "clear", "verbose", "context", "stop", "subagents",
)


@dataclass(frozen=True)
class ParsedCommand:
    """A parsed command invocation.

    ``name`` is the lowercased command word ("" for a bare prefix, or the
    raw unknown token if it isn't in ``KNOWN``). ``args`` is the remainder
    after the command word, stripped (may be "").
    """

    name: str
    args: str


def parse_command(text: str, prefix: str) -> ParsedCommand | None:
    """Return a ``ParsedCommand`` if ``text`` (stripped) starts with
    ``prefix``, else ``None``.

    A bare prefix yields ``name=""``; an unrecognized word yields that token
    as ``name`` — both non-``None`` so the caller answers with usage rather
    than letting the message fall through to the LLM. Returns ``None`` (not a
    command) when ``prefix`` is empty or the text doesn't start with it, so
    ordinary chat flows through untouched.
    """
    if not prefix:
        return None
    s = text.strip()
    if not s.startswith(prefix):
        return None
    rest = s[len(prefix):].strip()
    if not rest:
        return ParsedCommand(name="", args="")
    parts = rest.split(None, 1)
    return ParsedCommand(
        name=parts[0].lower(),
        args=parts[1].strip() if len(parts) > 1 else "",
    )


def usage(prefix: str) -> str:
    """One-line usage string, echoed back to authorized senders who type a
    bare prefix or an unknown command word."""
    return (
        f"commands: {prefix}compact, {prefix}clear, {prefix}context, "
        f"{prefix}stop [<task_id>] [--soft], {prefix}subagents, "
        f"{prefix}verbose <on|off>"
    )
