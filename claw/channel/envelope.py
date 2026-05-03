"""Envelope wrap for inbound messages.

Stamps every user-role message that reaches the agent with channel/sender/
timestamp/elapsed-time metadata, so the model has a per-turn anchor for
"today is", "what day of the week", and "how much time has passed since the
last interaction." Without it, small models (Qwen3.6-27B in particular)
hallucinate dates aggressively — they have no other source of ground truth
for the current date.

Format:

    [channel sender Sun 2026-05-03 07:30 +47m] body

- ``channel`` is always present. ``sender`` is omitted for cron / initial-
  prompt channels (no human sender). ``+elapsed`` only appears once a
  previous timestamp is known; the first inbound after a process restart
  has none.
- Weekday prefix is included specifically because small models are
  unreliable at deriving day-of-week from an absolute date.
- Elapsed-time suffix uses compact units: ``+30s``, ``+47m``, ``+3h``,
  ``+2d``. Cutoffs match OpenClaw's ``formatTimeAgo`` (seconds <1m,
  minutes <1h, hours <48h, then days).
- Timezone defaults to UTC if none supplied. Pass the operator's local
  IANA zone (e.g. ``"America/New_York"``) to keep envelope times aligned
  with cron / calendar / log timing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def _format_elapsed(seconds: float) -> str:
    """Compact elapsed-time formatter.

    Cutoffs: <60s → seconds, <60m → minutes, <48h → hours, else days.
    """
    s = max(0, int(seconds))
    if s < 60:
        return f"+{s}s"
    minutes = s // 60
    if minutes < 60:
        return f"+{minutes}m"
    hours = minutes // 60
    if hours < 48:
        return f"+{hours}h"
    return f"+{hours // 24}d"


def _sanitize_header_part(value: str) -> str:
    """Header parts must not be able to break the bracketed prefix.
    Collapse newlines/whitespace, neutralize brackets."""
    return (
        value.replace("\r\n", " ")
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("[", "(")
        .replace("]", ")")
        .strip()
    )


def format_inbound_envelope(
    channel: str,
    sender: str | None,
    body: str,
    ts: datetime,
    prev_ts: datetime | None = None,
    tz_name: str | None = None,
) -> str:
    """Wrap ``body`` with a date-stamped envelope.

    See module docstring for format. ``tz_name`` (IANA zone) controls the
    weekday + clock-time presentation; falls back to UTC.
    """
    tz = ZoneInfo(tz_name) if tz_name else timezone.utc
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    local = ts.astimezone(tz)
    weekday = local.strftime("%a")
    stamp = local.strftime("%Y-%m-%d %H:%M")

    elapsed: str | None = None
    if prev_ts is not None:
        if prev_ts.tzinfo is None:
            prev_ts = prev_ts.replace(tzinfo=timezone.utc)
        elapsed = _format_elapsed((ts - prev_ts).total_seconds())

    parts: list[str] = [_sanitize_header_part(channel)]
    if sender:
        s = _sanitize_header_part(sender)
        parts.append(f"{s} {elapsed}" if elapsed else s)
    elif elapsed:
        parts.append(elapsed)
    parts.append(f"{weekday} {stamp}")

    header = "[" + " ".join(parts) + "]"
    return f"{header} {body}"
