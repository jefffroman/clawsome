"""A subagent is an agent with a scratch session, not a callable.

Before this, ``run_one_shot`` discarded ``new_messages``: a subagent had no
transcript, no inbox, and was dropped when it returned. That was correct while
spawn was synchronous (the parent awaited the child and got its reply as a
return value), and quietly became incomplete when spawn went async — the result
now has to *arrive* somewhere, and only top-level agents had anywhere for it to
land. Chains were still advertised by config (max_spawn_depth > 1) that the
runtime could not honour.

These cover the session, its lifetime, and the guard that stops a subagent
behaving like a participant.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

from claw.agent import Agent
from claw.channel.base import InboundMessage
from claw.config import PersonaConfig, SubagentsConfig
from claw.transcript import purge_scratch_sessions
from claw.sink import ParentSink, completion_body
from claw.tools.subagent import ChildTask, SubagentSpawner

from datetime import datetime, timezone


def _subagents(**over):
    d = dict(
        max_concurrent=2, max_children_per_agent=2, default_model="test-model",
        personas={"grunt": PersonaConfig(model="test-model", role="grunt")},
    )
    d.update(over)
    return SubagentsConfig(**d)


def _agent(tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel,
           transcripts) -> Agent:
    cfg = make_cfg(tmp_path, subagents=_subagents())
    agent_cfg = cfg.agents[0]
    agent_cfg.workspace.mkdir(parents=True, exist_ok=True)
    return Agent(
        cfg=cfg, agent_cfg=agent_cfg, ollama=fake_ollama, memory=fake_memory,
        tools={}, transcripts=transcripts, channel=fake_channel,
        spawner=None, job_runner=None,
    )


def _ct(task_id="grunt-1", sid="matrix__room") -> ChildTask:
    return ChildTask(
        id=task_id, parent_id="agent-1", persona="grunt", prompt="do a thing",
        origin_channel="matrix", origin_peer_id="!room:example.org",
        origin_session_key=sid, started_at=datetime.now(timezone.utc),
        task_name="a task",
    )


# --- the session ------------------------------------------------------

async def test_context_survives_between_a_subagents_own_turns(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    """THE point of giving a subagent a session: a second turn sees the first.

    run_one_shot discarded new_messages, so a subagent woken by its own child
    would have been prompted with an empty history about work it had no record
    of requesting."""
    parent = _agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                    fake_channel, transcripts)
    child = parent.fork("grunt", task_id="grunt-1")

    await child.run_task("first prompt")
    await child.run_task("second prompt")

    history = fake_ollama.turns[-1]["history"]
    contents = [m.get("content") for m in history]
    assert "first prompt" in contents
    assert "second prompt" in contents
    assert contents.index("first prompt") < contents.index("second prompt")


def test_scratch_sessions_live_outside_the_participants_transcript_dir(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    """A subagent's context must never sit where rotation, boot recap or the
    flush pass would walk it."""
    parent = _agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                    fake_channel, transcripts)
    child = parent.fork("grunt", task_id="grunt-1")
    assert child.transcripts.dir == parent.transcripts.dir / "subagent"
    # Every walker in agent.py filters on *.jsonl, so a subdirectory entry is
    # skipped. Assert the shape those walkers rely on.
    assert child.transcripts.dir.is_dir()
    assert not child.transcripts.dir.name.endswith(".jsonl")


def test_a_fork_of_a_fork_reuses_the_store_rather_than_nesting(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    parent = _agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                    fake_channel, transcripts)
    child = parent.fork("grunt", task_id="grunt-1")
    grandchild = child.fork("grunt", task_id="grunt-2")
    assert grandchild.transcripts.dir == child.transcripts.dir


def test_sid_is_keyed_by_task_not_agent_id(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    """Two concurrent spawns of one persona from one parent share an agent id;
    keyed by that they would collide in a single transcript."""
    parent = _agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                    fake_channel, transcripts)
    a = parent.fork("grunt", task_id="grunt-1")
    b = parent.fork("grunt", task_id="grunt-2")
    assert a.id == b.id
    assert a.scratch_sid != b.scratch_sid


# --- lifetime ---------------------------------------------------------

async def test_reaping_leaves_no_context_for_the_next_spawn(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    """A handle's death is the end of its session. A later spawn must start
    clean rather than inherit a dead task's conversation."""
    parent = _agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                    fake_channel, transcripts)
    child = parent.fork("grunt", task_id="grunt-1")
    await child.run_task("secret from the previous task")
    assert (child.transcripts.dir / f"{child.scratch_sid}.jsonl").exists()

    child.discard_scratch_session()
    assert not (child.transcripts.dir / f"{child.scratch_sid}.jsonl").exists()

    reborn = parent.fork("grunt", task_id="grunt-1")
    await reborn.run_task("fresh prompt")
    contents = [m.get("content") for m in fake_ollama.turns[-1]["history"]]
    assert "secret from the previous task" not in contents


def test_boot_purge_clears_orphaned_sessions(tmp_path):
    """Handles do not survive a restart, so anything left is unreachable."""
    d = tmp_path / "subagent"
    d.mkdir()
    (d / "subagent-grunt-1.jsonl").write_text("{}\n")
    (d / "keep.txt").write_text("not a session")
    purge_scratch_sessions(d, "agent-1")
    assert not (d / "subagent-grunt-1.jsonl").exists()
    assert (d / "keep.txt").exists()


def test_boot_purge_tolerates_a_missing_dir(tmp_path):
    purge_scratch_sessions(tmp_path / "nope", "agent-1")


# --- a subagent is not a participant ----------------------------------

async def test_a_subagent_never_starts_a_conversational_drainer(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    """THE latent bug this closes. A drainer on a fork would take the FORK's
    session lock — a different object from the participant's, so no mutual
    exclusion — run a full turn against the human's transcript, and post to
    the room."""
    parent = _agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                    fake_channel, transcripts)
    child = parent.fork("grunt", task_id="grunt-1")
    child.attach_sink(ParentSink(parent, _ct()))

    await child.handle_inbound(InboundMessage(
        peer_id="!room:example.org", sender_name="subagent:grunt:grunt-2",
        text="Subagent task 'x' finished.", channel="matrix",
        session_key=child.scratch_sid, is_subagent_completion=True,
    ))

    assert not child._drainer_tasks, "a fork must never drain conversationally"
    assert not fake_channel.sent, "and must never reach the room"
    # But the report is queued, not dropped.
    assert child._pending_inbound[child.scratch_sid]


async def test_a_report_that_lands_between_turns_is_picked_up(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    """A child that finishes while its spawner is idle must still be seen when
    the spawner next runs."""
    parent = _agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                    fake_channel, transcripts)
    child = parent.fork("grunt", task_id="grunt-1")
    child.attach_sink(ParentSink(parent, _ct()))
    await child.run_task("go")

    await child.handle_inbound(InboundMessage(
        peer_id="!room:example.org", sender_name="subagent:grunt:grunt-2",
        text="GRANDCHILD-REPORT", channel="matrix",
        session_key=child.scratch_sid, is_subagent_completion=True,
    ))
    await child.run_task()

    blob = "\n".join(
        str(m.get("content")) for m in fake_ollama.turns[-1]["history"]
    )
    assert "GRANDCHILD-REPORT" in blob


async def test_a_subagents_turn_drains_its_inbox_every_tool_turn(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    """Not just between turns — the same per-tool-turn hook participants got."""
    parent = _agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                    fake_channel, transcripts)
    child = parent.fork("grunt", task_id="grunt-1")
    await child.run_task("go")
    assert fake_ollama.turns[-1]["drain_inbox"] is not None


# --- multi-emission ---------------------------------------------------

def test_outstanding_children_gate_terminality():
    spawner = SubagentSpawner(_subagents())
    parent_ct = _ct("grunt-1")
    kid = _ct("grunt-2")
    kid.parent_task_id = "grunt-1"
    spawner.tasks = {"grunt-1": parent_ct, "grunt-2": kid}

    kid.status = "running"
    assert spawner.has_outstanding_children("grunt-1")
    kid.status = "completed"
    assert not spawner.has_outstanding_children("grunt-1")


def test_an_interim_report_does_not_claim_to_be_finished():
    """'finished ... status=reporting' would be a contradiction."""
    ct = _ct()
    ct.result = "partial"
    ct.status = "reporting"
    assert "still working" in completion_body(ct)
    ct.status = "completed"
    assert "finished" in completion_body(ct)
    assert "still working" not in completion_body(ct)
