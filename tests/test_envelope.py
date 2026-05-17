"""channel/envelope.py — inbound-message envelope formatting.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from claw.channel.envelope import (
    _format_elapsed,
    _sanitize_header_part,
    format_inbound_envelope,
)


def test_format_elapsed_cutoffs():
    assert _format_elapsed(30) == "+30s"
    assert _format_elapsed(90) == "+1m"
    assert _format_elapsed(3600) == "+1h"
    assert _format_elapsed(47 * 3600) == "+47h"
    assert _format_elapsed(48 * 3600) == "+2d"
    assert _format_elapsed(-1) == "+0s"  # clamped to 0


def test_sanitize_header_part():
    assert _sanitize_header_part("a\r\nb[c]") == "a b(c)"


def test_format_inbound_envelope_basic():
    ts = datetime(2026, 5, 3, 7, 30, tzinfo=timezone.utc)
    # Derive the weekday from the datetime itself rather than hardcoding it.
    wd = ts.strftime("%a")
    out = format_inbound_envelope("matrix", "@user-1:example.org", "hi", ts=ts)
    assert out == f"[matrix @user-1:example.org {wd} 2026-05-03 07:30] hi"


def test_format_inbound_envelope_elapsed_after_sender():
    ts = datetime(2026, 5, 3, 7, 30, tzinfo=timezone.utc)
    prev = ts - timedelta(minutes=47)
    out = format_inbound_envelope(
        "matrix", "@user-1:example.org", "hi", ts=ts, prev_ts=prev
    )
    assert "@user-1:example.org +47m" in out


def test_format_inbound_envelope_no_sender_for_cron():
    ts = datetime(2026, 5, 3, 7, 30, tzinfo=timezone.utc)
    out = format_inbound_envelope("cron", None, "tick", ts=ts)
    wd = ts.strftime("%a")
    assert out == f"[cron {wd} 2026-05-03 07:30] tick"


def test_format_inbound_envelope_tz_shifts_clock():
    ts = datetime(2026, 5, 3, 7, 30, tzinfo=timezone.utc)
    local = ts.astimezone(ZoneInfo("America/New_York"))
    out = format_inbound_envelope(
        "matrix", "@user-1:example.org", "hi", ts=ts,
        tz_name="America/New_York",
    )
    assert local.strftime("%a %Y-%m-%d %H:%M") in out
