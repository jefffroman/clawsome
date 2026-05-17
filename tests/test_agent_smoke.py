"""One Agent happy-path smoke — proves the fake scaffold wires up.

Calls Agent._process_batch directly (the awaitable turn core);
handle_inbound is fire-and-forget (spawns a drainer task) and would be
racy. Imports claw.agent (heavy deps) — run under the gateway venv.
asyncio_mode=auto: no @pytest.mark.asyncio needed.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

from claw.agent import Agent, _THINKING_ANSWER_SEP
from claw.channel.base import InboundMessage
from claw.transcript import session_id


def _make_agent(tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel,
                 transcripts) -> Agent:
    cfg = make_cfg(tmp_path)
    agent_cfg = cfg.agents[0]
    agent_cfg.workspace.mkdir(parents=True, exist_ok=True)
    return Agent(
        cfg=cfg,
        agent_cfg=agent_cfg,
        ollama=fake_ollama,
        memory=fake_memory,
        tools={},
        transcripts=transcripts,
        channel=fake_channel,
        spawner=None,
        job_runner=None,
    )


async def test_process_batch_happy_path(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    cfg = make_cfg(tmp_path)
    agent_cfg = cfg.agents[0]
    agent_cfg.workspace.mkdir(parents=True, exist_ok=True)

    agent = Agent(
        cfg=cfg,
        agent_cfg=agent_cfg,
        ollama=fake_ollama,
        memory=fake_memory,
        tools={},
        transcripts=transcripts,
        channel=fake_channel,
        spawner=None,
        job_runner=None,
    )

    sid = "matrix__room_test"
    msg = InboundMessage(
        peer_id="!room:example.org",
        sender_name="user-1",
        text="hello",
        channel="matrix",
        sender_id="@user-1:example.org",
    )

    await agent._process_batch(sid, [msg], turn_id="t1")

    # run_turn invoked, memory queried.
    assert fake_ollama.turns, "ollama.run_turn was not called"
    assert fake_memory.queries, "memory.retrieve_markdown was not called"

    # Transcript captured the user turn + the assistant turn.
    rows = transcripts.load(sid)
    assert rows[0]["role"] == "user" and "hello" in rows[0]["content"]
    assert any(r["role"] == "assistant" for r in rows)

    # Final answer was sent back to the originating peer.
    assert fake_channel.sent, "no reply sent"
    peer, text = fake_channel.sent[-1]
    assert peer == "!room:example.org"
    assert "hello back" in text


async def test_system_block_gaps_the_answer(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    """A 🖥️ system block emitted while a turn is in flight makes that
    turn's answer carry the same one-line gap a %thinking block earns.
    """
    agent = _make_agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                         fake_channel, transcripts)
    peer = "!room:example.org"
    sid = session_id("matrix", peer)
    msg = InboundMessage(
        peer_id=peer, sender_name="user-1", text="hi",
        channel="matrix", sender_id="@user-1:example.org",
    )

    # No in-flight turn -> _cmd_reply sends a 🖥️ block but tags nothing.
    await agent._cmd_reply(peer, "status read")
    assert agent._system_emitted_turns == set()
    assert "🖥️" in fake_channel.sent[-1][1]

    # Turn in flight -> _cmd_reply tags exactly that turn id.
    agent._inflight_turn[sid] = "t1"
    await agent._cmd_reply(peer, "context: ...")
    assert "t1" in agent._system_emitted_turns

    # That turn's answer is prefixed with the separator.
    await agent._process_batch(sid, [msg], turn_id="t1")
    assert fake_channel.sent[-1][1].startswith(_THINKING_ANSWER_SEP)
    assert "hello back" in fake_channel.sent[-1][1]

    # Negative control: an untagged turn's answer gets no prefix.
    fake_channel.sent.clear()
    await agent._process_batch(sid, [msg], turn_id="t2")
    assert not fake_channel.sent[-1][1].startswith(_THINKING_ANSWER_SEP)
