"""agent.py module helpers + the %thinking separator render guard.

Imports claw.agent / claw.channel.matrix (heavy deps) — run under the
gateway venv interpreter.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

from claw.agent import (
    _THINKING_ANSWER_SEP,
    _as_system_blockquote,
    _as_thinking_blockquote,
    _derive_peer_label,
    _has_visible_content,
    _quote_body,
)
from claw.channel.base import InboundMessage
from claw.channel.matrix import _to_html


def test_has_visible_content_suppresses_invisibles():
    assert _has_visible_content("") is False
    assert _has_visible_content("  \t\n") is False
    assert _has_visible_content("​") is False    # ZWSP (cat Cf)
    assert _has_visible_content("﻿") is False    # BOM
    assert _has_visible_content("­") is False    # soft hyphen
    assert _has_visible_content("ok") is True
    assert _has_visible_content("  x  ") is True


def test_thinking_answer_sep_value():
    assert _THINKING_ANSWER_SEP == "​\n"
    assert len(_THINKING_ANSWER_SEP) == 2
    assert _THINKING_ANSWER_SEP[0] == "​"


def test_thinking_sep_one_line_gap_and_block_leading_fallback():
    # ZWSP + one newline -> a single <br> inside ONE paragraph (a
    # one-line gap), not a whole empty paragraph. A plain "\n"/"\n\n"
    # edge is stripped by CommonMark and does NOT separate at all —
    # that's why the ZWSP is needed (agent.py comment).
    with_sep = _to_html(_THINKING_ANSWER_SEP + "answer")
    assert with_sep.count("<p>") == 1
    assert with_sep.count("<br") == 1
    assert "​" in with_sep.split("<br")[0]
    assert _to_html("\nanswer").count("<p>") == 1
    assert _to_html("\n\nanswer").count("<p>") == 1
    # Graceful fallback: when the answer opens with a block construct the
    # ZWSP paragraph is interrupted and the block renders intact (content
    # preserved — same as the prior two-line form).
    fenced = _to_html(_THINKING_ANSWER_SEP + "```py\nx=1\n```")
    assert "<pre>" in fenced and "<code" in fenced


def test_quote_body_and_blockquotes():
    assert _quote_body("a\n\nb") == "> a\n>\n> b"
    assert _as_thinking_blockquote("x") == "> 🧠 **reasoning**\n>\n> x"
    assert _as_system_blockquote("x") == "> 🖥️ **system**\n>\n> x"


def _msg(sender_id="", sender_name="", channel="matrix"):
    return InboundMessage(
        peer_id="!room:example.org",
        sender_name=sender_name,
        text="hi",
        channel=channel,
        sender_id=sender_id,
    )


def test_derive_peer_label():
    assert _derive_peer_label(_msg(sender_id="@user-1:example.org")) == "user-1"
    assert _derive_peer_label(_msg(sender_name="cron")) == "cron"
    assert _derive_peer_label(_msg(channel="cron")) == "cron"
