"""claw — Python gateway for a local AI estate.

Boot order:
1. Load YAML config.
2. Configure logging.
3. Per-agent: warmup MemoryIndex + initial reindex.
4. Per-agent: build tool registry, wire up Matrix channel, instantiate Agent.
5. Start each Matrix sync loop.
6. Start the cron scheduler.
7. Dispatch any pending .initial_prompt.md.
8. Run the maintenance loop (periodic flush + reindex) in the background.
9. Wait for SIGTERM/SIGINT.

Shutdown:
1. Stop the cron scheduler.
2. Stop Matrix sync loops.
3. Cancel maintenance loop.
4. Close shared Ollama HTTP client.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any

from claw import __version__
from claw import logsetup
from claw.agent import Agent
from claw.channel.matrix import MatrixChannel
from claw.config import Config, load
from claw.voice_http import HttpReplyChannel, VoiceHttpServer
from claw.memory import MemoryIndex
from claw.memory_curate import curate_all_agents
from claw.ollama import OllamaClient
from claw.skills import build_agent_registry
from claw.tools.subagent import SubagentSpawner
from claw.transcript import TranscriptStore
from claw.triggers.initial_prompt import maybe_dispatch_initial_prompt
from claw.triggers.scheduler import JobRunner

log = logging.getLogger("claw.main")


def _configure_logging(verbose: bool) -> None:
    # Install handlers/formatter once; level logic + library taming lives in
    # logsetup so the %verbose command can re-apply it at runtime.
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logsetup.apply_log_levels(verbose)


async def _serve(cfg: Config) -> int:
    _configure_logging(cfg.verbose)
    log.info("starting claw gateway v%s", __version__)

    ollama = OllamaClient(cfg.ollama)
    spawner = SubagentSpawner(cfg.subagents)
    # Shared by-reference dict the JobRunner reads at fire time. We populate
    # it as Agents are constructed below so the runner's dispatch closure
    # always sees the live set.
    agents_by_id: dict[str, Agent] = {}
    job_runner = JobRunner(cfg, agents_by_id)
    memory_indexes: list[MemoryIndex] = []
    agents: list[Agent] = []
    channels: list[MatrixChannel] = []

    for ac in cfg.agents:
        log.info("[%s] initializing", ac.id)
        memory = MemoryIndex(ac.id, ac.workspace)
        await memory.warmup_async()
        try:
            result = await memory.reindex_if_stale()
            log.info("[%s] initial reindex: %s", ac.id, result)
        except Exception:
            log.exception("[%s] initial reindex failed; continuing", ac.id)
        memory_indexes.append(memory)

        tools, skill_catalog = build_agent_registry(cfg, ac.workspace)
        transcripts = TranscriptStore(ac.workspace / "transcripts")
        channel = MatrixChannel(ac.matrix)
        channels.append(channel)

        agent = Agent(
            cfg=cfg,
            agent_cfg=ac,
            ollama=ollama,
            memory=memory,
            tools=tools,
            transcripts=transcripts,
            channel=channel,
            skill_catalog=skill_catalog,
            spawner=spawner,
            depth=0,
            job_runner=job_runner,
            remaining_spawn_budget=ac.max_spawn_depth,
        )
        agents.append(agent)
        agents_by_id[ac.id] = agent

    for agent, channel in zip(agents, channels):
        await channel.start(agent.handle_inbound)

    # HTTP turn endpoint: claw's transport-decoupled voice inbound. Stood up
    # when http_api is enabled AND at least one device is configured. A caller
    # (an external voice stack fronting a thin client, or a client with its own
    # STT/TTS) POSTs {device_id, endpoint_id, text}; the device resolves to its
    # bound agent, and the outbound reply routes back via HttpReplyChannel
    # (resolving the request's future) rather than matrix. claw runs no audio —
    # mic capture, wake, STT, and TTS live in the external voice stack.
    voice_http: VoiceHttpServer | None = None
    if cfg.http_api is not None and cfg.http_api.enabled and cfg.devices:
        devices = {d.device_id: d for d in cfg.devices}
        bound_agents = {d.agent for d in cfg.devices}
        reply_channel = HttpReplyChannel()
        handlers: dict[str, Any] = {}
        for agent in agents:
            if agent.id in bound_agents:
                agent.register_channel(reply_channel.name, reply_channel)
                handlers[agent.id] = agent.handle_inbound
        voice_http = VoiceHttpServer(
            bind_host=cfg.http_api.bind_host,
            bind_port=cfg.http_api.bind_port,
            devices=devices,
            handlers=handlers,
            reply_channel=reply_channel,
        )
        await voice_http.start()

    # Eagerly recap any session whose last activity was > idle_recap_seconds
    # ago, so the first live turn after boot doesn't pay summarizer latency.
    # Run it in the BACKGROUND, AFTER channels are listening: a multi-minute
    # summarize must not hold off channel bring-up, or the voice box (and matrix
    # clients) can't (re)connect until it finishes — which stranded the box for
    # ~90s after every restart, well past its ws keepalive window. The recap
    # takes the session lock, so a live turn that lands first just serializes
    # ahead of it (paying the latency it would have paid anyway).
    async def _boot_recap(agent: Agent) -> None:
        try:
            await agent.boot_recap_known_sessions()
        except Exception:
            log.exception("[%s] boot recap pass raised; continuing", agent.id)

    recap_tasks = [
        asyncio.create_task(_boot_recap(agent), name=f"boot-recap-{agent.id}")
        for agent in agents
    ]

    # Start the cron scheduler now that all agents are wired and channels
    # are syncing. Jobs may begin firing immediately if their next trigger
    # time has already passed.
    job_runner.start()

    # If any agent has a .initial_prompt.md left by a prior restart, deliver
    # it now so the new process picks up where the old one left off.
    for agent in agents:
        try:
            await maybe_dispatch_initial_prompt(agent)
        except Exception:
            log.exception("[%s] initial_prompt pass raised", agent.id)

    # Surface any one-shot reminders missed beyond grace while we were down
    # (collected during job_runner.start()) to their owning agent for
    # deliver-late-or-discard review.
    try:
        await job_runner.review_missed_jobs()
    except Exception:
        log.exception("missed-reminder review pass raised")

    maintenance_task = asyncio.create_task(
        _maintenance_loop(agents, memory_indexes),
        name="maintenance-loop",
    )

    rotate_task: asyncio.Task | None = None
    if cfg.lifecycle.daily_session_rotate_hour is not None:
        rotate_task = asyncio.create_task(
            _daily_session_rotate_loop(
                agents, cfg.lifecycle.daily_session_rotate_hour, cfg.tz,
            ),
            name="daily-session-rotate",
        )

    curation_task: asyncio.Task | None = None
    if cfg.memory_curation.enabled:
        curation_task = asyncio.create_task(
            _nightly_curation_loop(agents, cfg),
            name="nightly-curation",
        )

    stop_event = asyncio.Event()

    def _request_stop(_sig: int = 0, _frame: Any = None) -> None:
        log.info("shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            signal.signal(sig, _request_stop)

    log.info("ready (%d agent%s)", len(agents), "s" if len(agents) != 1 else "")
    await stop_event.wait()

    log.info("shutting down")
    job_runner.shutdown()
    for t in recap_tasks:
        t.cancel()
    if voice_http is not None:
        try:
            await voice_http.shutdown()
        except Exception:
            log.exception("voice HTTP endpoint shutdown raised")
    for channel in channels:
        try:
            await channel.shutdown()
        except Exception:
            log.exception("channel shutdown raised")
    maintenance_task.cancel()
    try:
        await maintenance_task
    except (asyncio.CancelledError, Exception):
        pass
    if rotate_task is not None:
        rotate_task.cancel()
        try:
            await rotate_task
        except (asyncio.CancelledError, Exception):
            pass
    if curation_task is not None:
        curation_task.cancel()
        try:
            await curation_task
        except (asyncio.CancelledError, Exception):
            pass
    try:
        await ollama.aclose()
    except Exception:
        pass
    log.info("clean exit")
    return 0


async def _maintenance_loop(
    agents: list[Agent],
    indexes: list[MemoryIndex],
    interval_s: int = 300,
) -> None:
    """Periodic maintenance: trigger memory_flush per active session that
    has grown enough since its last flush, gather, then reindex so the
    new memory file content gets picked up in the same tick.

    Runs forever until cancelled.
    """
    while True:
        try:
            await asyncio.sleep(interval_s)

            flush_tasks: list[asyncio.Task] = []
            for agent in agents:
                try:
                    flush_tasks.extend(agent.periodic_flush_pass())
                except Exception:
                    log.exception("[%s] periodic_flush_pass raised", agent.id)
            if flush_tasks:
                log.info(
                    "maintenance: %d periodic flush task(s) running",
                    len(flush_tasks),
                )
                await asyncio.gather(*flush_tasks, return_exceptions=True)

            for index in indexes:
                try:
                    result = await index.reindex_if_stale()
                    if result.get("status") == "reindexed":
                        log.info(
                            "[%s] periodic reindex: +%d changed (%s) -%d removed (%s)",
                            index.agent_id,
                            len(result.get("changed", [])),
                            ", ".join(result.get("changed", [])) or "—",
                            len(result.get("removed", [])),
                            ", ".join(result.get("removed", [])) or "—",
                        )
                except Exception:
                    log.exception("[%s] periodic reindex failed", index.agent_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("maintenance loop iteration raised; continuing")


def _seconds_until_next(hour: int, tz_name: str | None) -> float:
    """Seconds from now until the next ``hour:00:00`` in the given IANA TZ.
    Always > 0 — if we're already past today's hour:00, returns
    time-to-tomorrow's. ``tz_name=None`` falls back to UTC.
    """
    tz = ZoneInfo(tz_name) if tz_name else timezone.utc
    now = datetime.now(tz)
    target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


async def _daily_session_rotate_loop(
    agents: list[Agent], hour: int, tz_name: str | None,
) -> None:
    """Sleep until the configured ``hour`` in ``tz_name``, rotate every
    agent's sessions (memory_flush + transcript archive + cache wipe), repeat.

    Runs forever until cancelled.
    """
    while True:
        try:
            sleep_s = _seconds_until_next(hour, tz_name)
            log.info(
                "daily rotate: next run in %.0f s (hour=%02d:00 %s)",
                sleep_s, hour, tz_name or "UTC",
            )
            await asyncio.sleep(sleep_s)
            for agent in agents:
                try:
                    rotated = await agent.clear_all_sessions()
                    log.info("[%s] daily rotate: %d session(s) archived", agent.id, rotated)
                except Exception:
                    log.exception("[%s] daily rotate raised", agent.id)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("daily rotate loop iteration raised; continuing")
            # Don't busy-loop on persistent failure.
            await asyncio.sleep(60)


async def _nightly_curation_loop(agents: list[Agent], cfg: Config) -> None:
    """Sleep until ``memory_curation.hour`` in ``cfg.tz``, run the nightly
    memory curator for every agent (dedup / supersession / archive lapsed
    ephemera), repeat. A quiet hour keeps the larger curator model off the
    user-reply path and avoids racing the collector on today's daily note
    (which the curator also skips).

    Runs forever until cancelled.
    """
    hour = cfg.memory_curation.hour
    while True:
        try:
            sleep_s = _seconds_until_next(hour, cfg.tz)
            log.info(
                "nightly curation: next run in %.0f s (hour=%02d:00 %s)",
                sleep_s, hour, cfg.tz or "UTC",
            )
            await asyncio.sleep(sleep_s)
            await curate_all_agents(agents, cfg)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("nightly curation loop iteration raised; continuing")
            await asyncio.sleep(60)


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="claw",
        description="Minimalist Python gateway for a local AI estate.",
    )
    parser.add_argument("--config", help="path to claw.yaml")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()

    if not args.config:
        raise SystemExit("--config <path> is required")

    try:
        cfg = load(args.config)
    except FileNotFoundError:
        raise SystemExit(f"config not found: {args.config}")
    except Exception as e:
        raise SystemExit(f"config load failed: {e}")

    try:
        sys.exit(asyncio.run(_serve(cfg)))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    cli()
