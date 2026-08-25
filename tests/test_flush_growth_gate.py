"""Both background gates are decided in one place, at turn-end.

The growth gate used to live in a 300s maintenance timer
(``periodic_flush_pass``). That timer fired blind to turn state, so a tick
landing mid-turn started a second GPU-bound /api/chat alongside a live reply —
the exact failure d352b0e moved the COMPACTION path to turn-end to prevent,
reintroduced through the other door. Observed 2026-08-24: a flush spawned 71s
into a turn, ran the full 300s deadline against a contended GPU, returned 500,
and discarded the work, while the reply took 8m22s against a ~30s baseline.

Retiring the timer is safe because transcript growth only ever comes from a
turn appending rows, so a turn-boundary check catches every growth event.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

import claw.agent as agent_mod
from claw.agent import Agent


def _agent(tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel,
           transcripts, **over) -> Agent:
    cfg = make_cfg(tmp_path, **over)
    agent_cfg = cfg.agents[0]
    agent_cfg.workspace.mkdir(parents=True, exist_ok=True)
    return Agent(
        cfg=cfg, agent_cfg=agent_cfg, ollama=fake_ollama, memory=fake_memory,
        tools={}, transcripts=transcripts, channel=fake_channel,
        spawner=None, job_runner=None,
    )


def _rows(tokens: int) -> list[dict]:
    """Rows whose estimate_tokens() lands near `tokens`."""
    from claw.transcript import estimate_tokens
    row = {"role": "user", "content": "word " * 200}
    n = max(1, tokens // max(1, estimate_tokens([row])))
    return [dict(row) for _ in range(n)]


async def test_growth_alone_fires_a_flush_at_turn_end(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    """THE replacement for the timer: enough new material to be worth
    remembering, nowhere near the compaction threshold."""
    a = _agent(tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel,
               transcripts)
    rows = _rows(a.cfg.memory_flush.periodic_growth_threshold * 2)
    assert a._spawn_bg_maintenance_if_needed("s", rows) == "flush"
    assert "s" in a._bg_compaction


async def test_growth_below_threshold_spawns_nothing(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    a = _agent(tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel,
               transcripts)
    rows = _rows(a.cfg.memory_flush.periodic_growth_threshold // 4)
    assert a._spawn_bg_maintenance_if_needed("s", rows) == ""
    assert not a._bg_compaction


async def test_compaction_wins_when_both_gates_trip(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts,
    monkeypatch,
):
    """One task, not two — compaction already flushes first, so firing a
    growth flush as well would duplicate the work it is about to do."""
    a = _agent(tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel,
               transcripts)
    monkeypatch.setattr(agent_mod, "will_mid_session_compact", lambda *a_, **k: True)
    rows = _rows(a.cfg.memory_flush.periodic_growth_threshold * 2)
    assert a._spawn_bg_maintenance_if_needed("s", rows) == "compact"
    assert len(a._bg_compaction) == 1


async def test_growth_is_measured_since_the_last_flush_not_from_zero(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    """A large session that has already been flushed must not re-fire every
    turn — the gate is delta, not absolute size."""
    a = _agent(tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel,
               transcripts)
    from claw.transcript import estimate_tokens
    rows = _rows(a.cfg.memory_flush.periodic_growth_threshold * 3)
    a._last_periodic_flush_tokens["s"] = estimate_tokens(rows)
    assert a._spawn_bg_maintenance_if_needed("s", rows) == ""


async def test_an_in_flight_background_task_blocks_a_second(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    a = _agent(tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel,
               transcripts)
    rows = _rows(a.cfg.memory_flush.periodic_growth_threshold * 2)
    assert a._spawn_bg_maintenance_if_needed("s", rows) == "flush"
    assert a._spawn_bg_maintenance_if_needed("s", rows) == ""


async def test_disabled_flush_does_not_fire_the_growth_gate(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    from claw.config import MemoryFlushConfig
    a = _agent(tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel,
               transcripts, memory_flush=MemoryFlushConfig(enabled=False))
    rows = _rows(a.cfg.memory_flush.periodic_growth_threshold * 2)
    assert a._spawn_bg_maintenance_if_needed("s", rows) == ""


def test_the_timer_driven_pass_is_gone():
    """Regression guard. Re-adding a timer would reintroduce the contention
    this change exists to remove — the gates belong at a turn boundary."""
    assert not hasattr(Agent, "periodic_flush_pass")
