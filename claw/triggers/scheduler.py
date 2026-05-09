"""APScheduler wrapper that fires agent turns on a schedule.

Reads ``jobs.json`` at boot and registers one ``AsyncIOScheduler`` job per
enabled entry. When a job fires, dispatch synthesizes an
``InboundMessage(channel="cron", peer_id="heartbeat", text=<configured
message>)`` and calls ``agent.handle_inbound`` — the agent then runs the
same pipeline as a Matrix DM (memory retrieval, compaction, ollama
tool-loop). Cron-driven replies typically reach the operator via tool
calls (e.g., a notification skill posting to Matrix) rather than the
channel-echo path.

``jobs.json`` schema (flat list of job entries on disk):

    [
      {
        "name": "morning-heartbeat",
        "agent": "morning-bot",
        "kind": "cron",
        "cron": "0 8 * * *",
        "tz": "America/New_York",
        "message": "Good morning. Run your morning heartbeat...",
        "enabled": true
      },
      {
        "name": "calendar-reminder-2026-05-15",
        "agent": "morning-bot",
        "kind": "at",
        "run_date": "2026-05-15T14:30:00-04:00",
        "message": "Reminder: ...",
        "enabled": true,
        "deleteAfterRun": true
      }
    ]

Numeric day-of-week values follow standard-cron convention (0=Sun..6=Sat,
7=Sun); ``_normalize_crontab_dow`` below translates them so APScheduler's
``from_crontab`` (which would otherwise read 5 as Saturday) sees the
expected day. Named days (``mon``..``sun``) pass through unchanged.

One-shot ``deleteAfterRun: true`` entries get removed from disk after
firing via an ``EVENT_JOB_EXECUTED`` listener.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from apscheduler.events import EVENT_JOB_EXECUTED, JobExecutionEvent
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from claw.channel.base import InboundMessage
from claw.config import Config
from claw.tools.base import Tool

log = logging.getLogger("claw.triggers.scheduler")

# Naive ISO 8601: YYYY-MM-DDTHH:MM[:SS] with no offset and no trailing Z.
# kind=at job times are interpreted in tz (per-job or gateway default), so the
# string must not embed its own zone — that's the whole point of the contract.
_NAIVE_ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?")


# APScheduler 3.x's ``CronTrigger.from_crontab`` passes the day-of-week
# field through as-is, but its CronTrigger reads numeric DOW in
# Mon=0..Sun=6 form, while the standard-cron convention every operator
# expects is Sun=0..Sat=6 (with 7=Sun as legacy alias). So a job written
# as ``0 3 * * 5`` (Friday in cron) would fire on Saturday under APS.
# This helper rewrites the 5th field to the equivalent APS-named-day
# expression before handing the string off; named days pass through.
_APS_WEEKDAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_CRON_TO_APS_DOW = {0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}


def _remap_one_dow(s: str) -> str:
    """Translate a single numeric or named DOW atom; pass non-digit
    tokens through unchanged."""
    if s.isdigit():
        n = int(s)
        if n in _CRON_TO_APS_DOW:
            return _APS_WEEKDAY_NAMES[_CRON_TO_APS_DOW[n]]
    return s


def _remap_dow_token(tok: str) -> str:
    """Translate one comma-separated DOW token, preserving range/step
    structure."""
    step = ""
    if "/" in tok:
        tok, step_part = tok.split("/", 1)
        step = "/" + step_part
    if tok == "*":
        return "*" + step
    if "-" in tok:
        a, b = tok.split("-", 1)
        return f"{_remap_one_dow(a)}-{_remap_one_dow(b)}{step}"
    return _remap_one_dow(tok) + step


def _normalize_crontab_dow(expr: str) -> str:
    """Pre-process a 5-field crontab expression so the day-of-week field
    follows standard-cron convention (0=Sun..6=Sat, 7=Sun) regardless of
    APScheduler's native numbering."""
    parts = expr.split()
    if len(parts) != 5:
        return expr  # let from_crontab raise the right error itself
    parts[4] = ",".join(_remap_dow_token(t) for t in parts[4].split(","))
    return " ".join(parts)


class JobRunner:
    def __init__(self, cfg: Config, agents: dict[str, Any]) -> None:
        # agents: dict[str, Agent] — typed Any to avoid circular import.
        self.cfg = cfg
        self.agents = agents
        self.scheduler = AsyncIOScheduler()
        self.scheduler.add_listener(self._on_job_executed, EVENT_JOB_EXECUTED)
        self._jobs_file = cfg.cron.jobs_file

    def start(self) -> None:
        if not self.cfg.cron.enabled:
            log.info("cron disabled in config; not loading jobs.json")
            return
        self._load_jobs_from_disk()
        self.scheduler.start()

    def shutdown(self) -> None:
        try:
            self.scheduler.shutdown(wait=False)
        except Exception:
            log.exception("scheduler shutdown raised")

    # --- loading ---------------------------------------------------------

    def _load_jobs_from_disk(self) -> None:
        if not self._jobs_file.exists():
            log.info("jobs file %s does not exist; starting with no jobs", self._jobs_file)
            return
        try:
            jobs = json.loads(self._jobs_file.read_text()) or []
        except (OSError, json.JSONDecodeError):
            log.exception("failed to read jobs file %s; starting with no jobs", self._jobs_file)
            return
        registered = 0
        for job in jobs:
            if not job.get("enabled", True):
                continue
            try:
                self._register_job(job)
                registered += 1
            except Exception:
                log.exception("scheduler: failed to register job %s", job.get("name"))
        log.info("scheduler: %d job(s) registered from %s", registered, self._jobs_file)

    def _register_job(self, job: dict[str, Any]) -> None:
        # Backfill id for legacy entries (pre-uuid era when 'name' served as
        # the key by convention). Mutates the dict in place so the caller's
        # read+write cycle persists the assignment.
        if "id" not in job or not job["id"]:
            job["id"] = uuid.uuid4().hex
        job_id = job["id"]
        agent_id = job.get("agent")
        if agent_id not in self.agents:
            log.warning("scheduler: skipping job %s — unknown agent %s", job_id, agent_id)
            return
        deliver_to = job.get("deliver_to")
        if not deliver_to:
            log.warning(
                "scheduler: skipping job %s — no deliver_to set (heartbeat "
                "retired; set deliver_to to a Matrix MXID/room id to enable)",
                job_id)
            return
        kind = job.get("kind", "cron")
        tz = job.get("tz") or self.cfg.tz
        if not tz:
            raise ValueError(
                f"job {job_id}: 'tz' is required (no per-job tz and no default in claw.yaml)"
            )
        if kind == "cron":
            expr = job.get("cron")
            if not expr:
                raise ValueError(f"job {job_id}: 'cron' field is required for kind=cron")
            trigger = CronTrigger.from_crontab(_normalize_crontab_dow(expr), timezone=tz)
        elif kind == "at":
            run_date = job.get("run_date")
            if not run_date:
                raise ValueError(f"job {job_id}: 'run_date' is required for kind=at")
            trigger = DateTrigger(run_date=run_date, timezone=tz)
        else:
            raise ValueError(f"job {job_id}: unknown kind {kind!r}")

        message = job.get("message", "")
        self.scheduler.add_job(
            func=self._dispatch,
            trigger=trigger,
            id=job_id,
            args=[agent_id, message, job_id, deliver_to],
            replace_existing=False,
            max_instances=self.cfg.cron.max_instances_per_job,
        )
        log.info(
            "scheduler: registered %s [%s] tz=%s -> agent=%s deliver_to=%s",
            job_id, kind, tz, agent_id, deliver_to,
        )

    # --- dispatch --------------------------------------------------------

    async def _dispatch(self, agent_id: str, message: str, job_id: str, deliver_to: str) -> None:
        agent = self.agents.get(agent_id)
        if agent is None:
            log.warning("scheduler: job %s fired but agent %s gone", job_id, agent_id)
            return
        log.info("scheduler: firing job %s -> agent %s deliver_to=%s", job_id, agent_id, deliver_to)
        msg = InboundMessage(
            peer_id=deliver_to,
            sender_name="cron",
            text=message,
            channel="cron",
        )
        try:
            await agent.handle_inbound(msg)
        except Exception:
            log.exception("scheduler: handle_inbound raised for job %s", job_id)

    def _on_job_executed(self, event: JobExecutionEvent) -> None:
        """If a fired job is marked deleteAfterRun, remove it from jobs.json."""
        job_id = event.job_id
        if not self._jobs_file.exists():
            return
        try:
            jobs = json.loads(self._jobs_file.read_text()) or []
        except (OSError, json.JSONDecodeError):
            return
        target = next((j for j in jobs if j.get("id") == job_id), None)
        if target is None or not target.get("deleteAfterRun"):
            return
        new_jobs = [j for j in jobs if j.get("id") != job_id]
        try:
            self._jobs_file.write_text(json.dumps(new_jobs, indent=2))
            log.info("scheduler: removed one-shot job %s after execution", job_id)
        except OSError:
            log.exception("scheduler: failed to rewrite jobs.json after firing %s", job_id)

    # --- live mutations (used by the cron_add / cron_remove tools) ------

    def _read_jobs(self) -> list[dict[str, Any]]:
        if not self._jobs_file.exists():
            return []
        try:
            return json.loads(self._jobs_file.read_text()) or []
        except (OSError, json.JSONDecodeError):
            return []

    def _write_jobs(self, jobs: list[dict[str, Any]]) -> None:
        self._jobs_file.parent.mkdir(parents=True, exist_ok=True)
        self._jobs_file.write_text(json.dumps(jobs, indent=2))

    def list_jobs(self) -> list[dict[str, Any]]:
        return self._read_jobs()

    def add_job(self, job: dict[str, Any]) -> None:
        """Append a new job to jobs.json AND register it live. Each job is
        keyed by an auto-generated uuid; agents identify jobs in cron_list /
        cron_remove by short id prefix."""
        if "id" not in job or not job["id"]:
            job["id"] = uuid.uuid4().hex
        existing = self._read_jobs()
        existing.append(job)
        self._write_jobs(existing)
        if job.get("enabled", True):
            self._register_job(job)

    def remove_job(self, job_id: str) -> dict[str, Any] | None:
        """Remove a single job by id (full uuid or unique short prefix as
        shown in cron_list). Returns the removed job, or None if no match.
        Raises if a short prefix is ambiguous."""
        if not job_id:
            raise ValueError("remove_job requires job_id")
        jobs = self._read_jobs()
        matches = [j for j in jobs if (j.get("id") or "").startswith(job_id)]
        if len(matches) > 1:
            ids = ", ".join((j.get("id") or "")[:12] for j in matches)
            raise ValueError(f"id prefix {job_id!r} is ambiguous: matches {ids}")
        if not matches:
            return None
        target = matches[0]
        self._write_jobs([j for j in jobs if j is not target])
        try:
            self.scheduler.remove_job(target.get("id") or "")
        except Exception:
            # Job may not have been registered (disabled, missing fields,
            # or not yet started). Removal from disk still valid.
            pass
        return target


# --- agent tools -----------------------------------------------------


def build_cron_add_tool(
    agent_id: str,
    runner: JobRunner,
    default_deliver_to: str | None,
    default_tz: str | None,
) -> Tool:
    """Tool an agent can call to schedule its own future turn.

    Forces the ``agent`` field to the calling agent's id so persona
    promotion isn't possible via this surface. Each call creates a new
    job with a fresh uuid — there's no name/key collision concept.
    ``default_deliver_to`` and ``default_tz`` (from config) fill in when
    the caller omits them; if no fallback is available, the call is
    rejected.
    """
    async def _run(args: dict[str, Any]) -> str:
        kind = (args.get("kind") or "at").lower()
        message = args.get("message") or ""
        tz = args.get("tz") or default_tz
        deliver_to = (args.get("deliver_to") or default_deliver_to or "").strip()
        if not message:
            return "error: message is required"
        if not deliver_to:
            return "error: deliver_to is required (no default configured)"
        if not tz:
            return "error: tz is required — no per-call value and no gateway default configured"
        job: dict[str, Any] = {
            "agent": agent_id,
            "kind": kind,
            "tz": tz,
            "message": message,
            "deliver_to": deliver_to,
            "enabled": True,
        }
        if kind == "cron":
            cron_expr = args.get("cron")
            if not cron_expr:
                return "error: 'cron' is required when kind='cron'"
            job["cron"] = cron_expr
        elif kind == "at":
            run_date = args.get("run_date")
            if not run_date:
                return "error: 'run_date' is required when kind='at'"
            if not _NAIVE_ISO_RE.fullmatch(run_date):
                return (
                    "error: run_date must be naive ISO 8601 (no offset, no Z), "
                    "e.g. '2026-05-15T14:30:00'; pass the zone in tz"
                )
            job["run_date"] = run_date
            job["deleteAfterRun"] = bool(args.get("deleteAfterRun", True))
        else:
            return f"error: kind must be 'cron' or 'at', got {kind!r}"

        try:
            runner.add_job(job)
        except Exception as e:
            log.exception("cron_add failed")
            return f"error: {e}"
        return f"job {job['id'][:8]} added (delivers to {deliver_to})"

    deliver_to_doc = (
        f"Matrix MXID or room id to send your reply to. "
        f"Defaults to {default_deliver_to}." if default_deliver_to
        else "Matrix MXID or room id to send your reply to. Required (no default configured)."
    )
    tz_doc = (
        f"IANA timezone. Defaults to {default_tz!r} (the gateway's configured "
        f"tz) — override per call to schedule in a different zone." if default_tz
        else "IANA timezone, e.g. 'America/New_York'. Required (no gateway default configured)."
    )
    return Tool(
        name="cron_add",
        description=(
            "Schedule a future turn for yourself. kind='at' fires once and "
            "auto-deletes. kind='cron' fires repeatedly on a crontab "
            "expression. The job's `message` becomes a synthetic user turn "
            "for you, and your reply is sent to `deliver_to` — write the "
            "message as a self-contained instruction. Returns a short id "
            "for the job."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["at", "cron"], "description": "One-shot or recurring."},
                "run_date": {"type": "string", "description": "Naive ISO 8601 datetime, e.g. '2026-05-15T14:30:00'. Only for kind='at'."},
                "cron": {"type": "string", "description": "Crontab expression, e.g. '0 8 * * *' or '*/15 * * * *'. Only for kind='cron'."},
                "tz": {"type": "string", "description": tz_doc},
                "message": {"type": "string", "description": "Synthetic user turn delivered to you when the job fires. Self-contained — no other context."},
                "deliver_to": {"type": "string", "description": deliver_to_doc},
                "deleteAfterRun": {"type": "boolean", "description": "Default true for kind=at, ignored for kind=cron."},
            },
            "required": ["kind", "message"],
        },
        run=_run,
    )


def build_cron_list_tool(runner: JobRunner) -> Tool:
    """Tool an agent can call to list all scheduled jobs (global)."""
    async def _run(args: dict[str, Any]) -> str:
        jobs = runner.list_jobs()
        if not jobs:
            return "no jobs scheduled"
        lines = []
        for j in jobs:
            jid = (j.get("id") or "")[:8] or "?"
            agent = j.get("agent", "?")
            kind = j.get("kind", "?")
            enabled = "" if j.get("enabled", True) else " [disabled]"
            if kind == "cron":
                sched = f"cron='{j.get('cron', '?')}'"
            elif kind == "at":
                sched = f"at={j.get('run_date', '?')}"
            else:
                sched = f"kind={kind}"
            tz = j.get("tz")
            tz_part = f" tz={tz}" if tz else ""
            deliver = j.get("deliver_to") or "(none)"
            msg = (j.get("message") or "").strip().replace("\n", " ")
            if len(msg) > 80:
                msg = msg[:77] + "..."
            lines.append(
                f"[{jid}] [{agent}] {sched}{tz_part} -> {deliver}{enabled}\n"
                f"        \"{msg}\""
            )
        return "\n".join(lines)

    return Tool(
        name="cron_list",
        description=(
            "List all scheduled cron jobs (across all agents). Each entry "
            "shows: short id (for cron_remove), owning agent, schedule, "
            "delivery target, and message preview. Same prompt scheduled "
            "twice = two distinct jobs with different ids."
        ),
        input_schema={"type": "object", "properties": {}},
        run=_run,
    )


def build_cron_remove_tool(runner: JobRunner) -> Tool:
    """Tool an agent can call to remove a scheduled job by id."""
    async def _run(args: dict[str, Any]) -> str:
        job_id = (args.get("id") or "").strip()
        if not job_id:
            return "error: id is required (use the short id from cron_list)"
        try:
            removed = runner.remove_job(job_id)
        except Exception as e:
            log.exception("cron_remove failed for id=%r", job_id)
            return f"error: {e}"
        if removed is None:
            return f"no job with id starting with {job_id!r}"
        return f"removed job {removed['id'][:8]}"

    return Tool(
        name="cron_remove",
        description=(
            "Remove a scheduled cron job by id. Use the short id from "
            "cron_list (or the full uuid). Affects the live scheduler "
            "immediately and rewrites the jobs file."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Job id (short prefix from cron_list, or full uuid)."},
            },
            "required": ["id"],
        },
        run=_run,
    )
