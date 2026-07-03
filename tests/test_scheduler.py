"""triggers/scheduler.py — crontab DOW normalization.

Regression guard for the APScheduler Sun=0 quirk: standard cron is
0=Sun..6=Sat (7=Sun legacy); APScheduler reads numeric DOW Mon=0..Sun=6,
so `0 3 * * 5` (cron Friday) would fire Saturday without this remap.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from claw.config import CronConfig
from claw.triggers.scheduler import (
    JobRunner,
    _build_missed_review,
    _normalize_crontab_dow,
    _normalize_run_date,
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


# --- run_date normalization (Bug 2: seconds-less times APScheduler rejects) --

def test_normalize_run_date_adds_seconds():
    assert _normalize_run_date("2026-06-26T08:55") == "2026-06-26T08:55:00"
    assert _normalize_run_date("2026-06-26T08:55:00") == "2026-06-26T08:55:00"
    assert _normalize_run_date("2026-06-26T08:55:30") == "2026-06-26T08:55:30"


def test_build_missed_review_lists_jobs_and_target():
    out = _build_missed_review(
        [{"run_date": "2026-06-14T11:10:00", "message": "Eye Doc reminder"}],
        "@alice-phone:example.org",
    )
    assert "@alice-phone:example.org" in out
    assert "Eye Doc reminder" in out
    assert "2026-06-14T11:10:00" in out
    # always surface (never silently drop) — the heads-up is the feature
    assert "what slipped by" in out


# --- JobRunner: misfire detection, cleanup, review --------------------------

class _FakeAgent:
    def __init__(self) -> None:
        self.inbound: list = []

    async def handle_inbound(self, msg) -> None:
        self.inbound.append(msg)


def _runner(make_cfg, tmp_path: Path, grace: int = 3600, agents=None):
    jobs_file = tmp_path / "jobs.json"
    cfg = make_cfg(
        tmp_path, tz="UTC",
        cron=CronConfig(enabled=True, jobs_file=jobs_file, misfire_grace_time=grace),
    )
    runner = JobRunner(cfg, agents if agents is not None else {"a": _FakeAgent()})
    return runner, jobs_file


def _at_job(run_dt: datetime, job_id: str, **over) -> dict:
    j = {
        "id": job_id, "agent": "a", "kind": "at", "tz": "UTC",
        "message": "m", "deliver_to": "@d:x", "enabled": True,
        "deleteAfterRun": True,
        "run_date": run_dt.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    j.update(over)
    return j


def test_is_stale_at_job_classifies_by_grace(make_cfg, tmp_path):
    runner, _ = _runner(make_cfg, tmp_path, grace=3600)
    now = datetime.now(ZoneInfo("UTC"))
    assert runner._is_stale_at_job(_at_job(now + timedelta(hours=2), "f"), 3600) is False
    assert runner._is_stale_at_job(_at_job(now - timedelta(minutes=30), "r"), 3600) is False  # in grace
    assert runner._is_stale_at_job(_at_job(now - timedelta(hours=5), "s"), 3600) is True
    assert runner._is_stale_at_job({"kind": "cron", "cron": "0 8 * * *"}, 3600) is False


def test_load_collects_stale_and_prunes_disk(make_cfg, tmp_path):
    now = datetime.now(ZoneInfo("UTC"))
    runner, jobs_file = _runner(make_cfg, tmp_path)
    jobs_file.write_text(json.dumps([
        _at_job(now + timedelta(hours=2), "fut"),
        _at_job(now - timedelta(hours=5), "stale1"),
    ]))
    runner._load_jobs_from_disk()
    assert [j["id"] for j in runner._missed_at_load] == ["stale1"]
    assert [j["id"] for j in json.loads(jobs_file.read_text())] == ["fut"]  # stale pruned


def test_cleanup_one_shot_only_removes_delete_after_run(make_cfg, tmp_path):
    runner, jobs_file = _runner(make_cfg, tmp_path)
    jobs_file.write_text(json.dumps([
        {"id": "x", "deleteAfterRun": True},
        {"id": "y", "deleteAfterRun": False},
    ]))
    runner._cleanup_one_shot("x", "after execution")  # the EXECUTED path
    assert [j["id"] for j in json.loads(jobs_file.read_text())] == ["y"]
    runner._cleanup_one_shot("y", "missed at runtime")  # not deleteAfterRun → kept
    assert [j["id"] for j in json.loads(jobs_file.read_text())] == ["y"]


def test_is_exact_duplicate_at_same_minute():
    a = {"agent": "a", "kind": "at", "deliver_to": "@d", "message": "m",
         "run_date": "2026-07-01T12:00:00"}
    assert JobRunner._is_exact_duplicate(a, dict(a, run_date="2026-07-01T12:00:30")) is True   # same minute
    assert JobRunner._is_exact_duplicate(a, dict(a, run_date="2026-07-01T12:00")) is True       # seconds-less normalizes
    assert JobRunner._is_exact_duplicate(a, dict(a, run_date="2026-07-01T13:00:00")) is False   # different minute
    assert JobRunner._is_exact_duplicate(a, dict(a, message="m2")) is False                     # near-dup wording → allowed
    assert JobRunner._is_exact_duplicate(a, dict(a, deliver_to="@e")) is False                  # different target


def test_is_exact_duplicate_cron():
    a = {"agent": "a", "kind": "cron", "deliver_to": "@d", "message": "m", "cron": "0 8 * * *"}
    assert JobRunner._is_exact_duplicate(a, dict(a)) is True
    assert JobRunner._is_exact_duplicate(a, dict(a, cron="0 9 * * *")) is False


def test_add_job_rejects_exact_dup_keeps_near_dup(make_cfg, tmp_path):
    now = datetime.now(ZoneInfo("UTC"))
    runner, jobs_file = _runner(make_cfg, tmp_path)
    j1 = _at_job(now + timedelta(hours=2), "j1", message="Tribe reminder")
    j2 = _at_job(now + timedelta(hours=2), "j2", message="Tribe reminder")           # exact dup
    j3 = _at_job(now + timedelta(hours=2), "j3", message="Tribe reminder (1-hour)")  # near-dup
    assert runner.add_job(j1) is None        # added
    assert runner.add_job(j2) is not None    # exact dup → rejected (returns existing)
    assert runner.add_job(j3) is None        # near-dup → added
    msgs = [x["message"] for x in json.loads(jobs_file.read_text())]
    assert msgs.count("Tribe reminder") == 1            # not duplicated
    assert msgs.count("Tribe reminder (1-hour)") == 1   # near-dup kept


def test_review_missed_jobs_groups_by_deliver_to(make_cfg, tmp_path):
    agent = _FakeAgent()
    runner, _ = _runner(make_cfg, tmp_path, agents={"a": agent})
    runner._missed_at_load = [
        {"agent": "a", "deliver_to": "@alice:x", "run_date": "2026-06-14T11:10:00", "message": "r1"},
        {"agent": "a", "deliver_to": "@alice:x", "run_date": "2026-06-14T14:00:00", "message": "r2"},
        {"agent": "a", "deliver_to": "@bob:x", "run_date": "2026-06-10T19:17:00", "message": "r3"},
    ]
    asyncio.run(runner.review_missed_jobs())
    # one dispatch per (agent, deliver_to) group, replying to the original target
    assert sorted(m.peer_id for m in agent.inbound) == ["@alice:x", "@bob:x"]
    assert all(m.channel == "cron" for m in agent.inbound)
    alice_turn = next(m for m in agent.inbound if m.peer_id == "@alice:x")
    assert "r1" in alice_turn.text and "r2" in alice_turn.text  # both batched
    assert runner._missed_at_load == []  # drained
