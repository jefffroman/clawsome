"""triggers/scheduler.py — crontab DOW normalization.

Regression guard for the APScheduler Sun=0 quirk: standard cron is
0=Sun..6=Sat (7=Sun legacy); APScheduler reads numeric DOW Mon=0..Sun=6,
so `0 3 * * 5` (cron Friday) would fire Saturday without this remap.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

from claw.triggers.scheduler import (
    _normalize_crontab_dow,
    _remap_dow_token,
    _remap_one_dow,
)


def test_remap_one_dow_numeric_and_named():
    assert _remap_one_dow("5") == "fri"   # the exact bug: cron Fri != Sat
    assert _remap_one_dow("0") == "sun"
    assert _remap_one_dow("7") == "sun"   # legacy alias
    assert _remap_one_dow("1") == "mon"
    assert _remap_one_dow("6") == "sat"
    assert _remap_one_dow("mon") == "mon"  # named pass-through
    assert _remap_one_dow("*") == "*"


def test_remap_dow_token_ranges_and_steps():
    # _remap_dow_token handles ONE comma-separated atom (comma splitting is
    # done by _normalize_crontab_dow, not here).
    assert _remap_dow_token("1-5") == "mon-fri"
    assert _remap_dow_token("*/2") == "*/2"
    assert _remap_dow_token("1-5/2") == "mon-fri/2"
    assert _remap_dow_token("0") == "sun"


def test_normalize_crontab_dow():
    assert _normalize_crontab_dow("0 3 * * 5") == "0 3 * * fri"
    assert _normalize_crontab_dow("0 8 * * 1-5") == "0 8 * * mon-fri"
    assert _normalize_crontab_dow("0 8 * * 0,6") == "0 8 * * sun,sat"
    assert _normalize_crontab_dow("0 3 * * mon") == "0 3 * * mon"
    # Non-5-field expression passes through so from_crontab raises its own error.
    assert _normalize_crontab_dow("garbage") == "garbage"
