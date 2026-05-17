"""commands.py — in-band command parsing / usage (PR #50/#53 guard).

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

from claw.commands import (
    COMMAND_USAGE,
    KNOWN,
    ParsedCommand,
    command_usage,
    parse_command,
    usage,
)


def test_parse_command_not_a_command():
    assert parse_command("", "%") is None
    assert parse_command("hello world", "%") is None
    assert parse_command("hi", "") is None  # empty prefix disables


def test_parse_command_bare_prefix():
    assert parse_command("%", "%") == ParsedCommand(name="", args="")


def test_parse_command_lowercased_and_stripped():
    assert parse_command("  %Compact  ", "%") == ParsedCommand("compact", "")


def test_parse_command_with_args():
    assert parse_command("%verbose on", "%") == ParsedCommand("verbose", "on")


def test_parse_command_unknown_word_still_non_none():
    # An unrecognized word is returned (not None) so the caller answers with
    # usage rather than letting it fall through to the LLM.
    assert parse_command("%bogus a b", "%") == ParsedCommand("bogus", "a b")


def test_known_and_command_usage_single_source():
    assert set(KNOWN) == {
        "compact", "clear", "verbose", "context",
        "stop", "subagents", "thinking",
    }
    # COMMAND_USAGE must cover exactly KNOWN — a half-added command fails here.
    assert set(COMMAND_USAGE) == set(KNOWN)


def test_usage_one_liner_exact():
    assert usage("%") == (
        "commands: %compact, %clear, %context, "
        "%stop [<task_id>] [--soft], %subagents, "
        "%verbose <on|off>, %thinking <on|off|full>"
    )


def test_command_usage_no_arg_command():
    assert command_usage("clear", "%") == "%clear takes no arguments"
    assert command_usage("clear", "%", "extra") == (
        "unexpected argument 'extra' — %clear takes no arguments"
    )


def test_command_usage_arg_command():
    assert command_usage("verbose", "%") == "usage: %verbose <on|off>"
    assert command_usage("verbose", "%", "loud") == (
        "unexpected argument 'loud' — usage: %verbose <on|off>"
    )


def test_command_usage_unknown_falls_back_to_one_liner():
    assert command_usage("nope", "%") == usage("%")
