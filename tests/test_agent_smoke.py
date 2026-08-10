"""One Agent happy-path smoke — proves the fake scaffold wires up.

Calls Agent._process_batch directly (the awaitable turn core);
handle_inbound is fire-and-forget (spawns a drainer task) and would be
racy. Imports claw.agent (heavy deps) — run under the gateway venv.
asyncio_mode=auto: no @pytest.mark.asyncio needed.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

import logging

from claw.agent import Agent, _THINKING_ANSWER_SEP
from claw.channel.base import InboundMessage
from claw.commands import parse_command
from claw.config import VoiceServiceConfig
from claw.transcript import session_id

# A distinctive, stable fragment of the default voice modality hint.
_HINT_MARKER = "read aloud by"


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


async def test_voice_reply_routes_to_voice_channel(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    """A voice inbound's reply is delivered to the registered voice channel,
    not the agent's primary (matrix) channel. Regression for the voice
    redesign's missing outbound wiring: replies went to matrix.send() with the
    voice session id as peer_id and were dropped as a 'non-existent channel',
    so no TTS ever reached the box.
    """
    agent = _make_agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                        fake_channel, transcripts)
    voice_channel = type(fake_channel)()          # a second fake, registered as "voice"
    agent.register_channel("voice", voice_channel)

    sid = "voice__box3"
    msg = InboundMessage(
        peer_id="voice:box3-main-area:sess-1",
        sender_name="box3",
        text="what's the temperature?",
        channel="voice",
        sender_id="box3-main-area/voice",
    )
    await agent._process_batch(sid, [msg], turn_id="t1")

    # Reply delivered to the voice channel, keyed by the session peer_id...
    assert voice_channel.sent, "reply was not delivered to the voice channel"
    peer, text = voice_channel.sent[-1]
    assert peer == "voice:box3-main-area:sess-1"
    assert "hello back" in text
    # ...and NOT leaked to the primary (matrix) channel.
    assert not fake_channel.sent, "reply leaked to the matrix channel"


def test_channel_for_falls_back_to_primary(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    """_channel_for returns the registered channel for a known inbound channel
    and the primary (matrix) channel for everything else — matrix itself and
    synthetic channels (cron / initial_prompt / subagent_completion)."""
    agent = _make_agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                        fake_channel, transcripts)
    voice_channel = type(fake_channel)()
    agent.register_channel("voice", voice_channel)
    assert agent._channel_for("voice") is voice_channel
    assert agent._channel_for("matrix") is fake_channel
    assert agent._channel_for("cron") is fake_channel


async def test_cron_turn_is_stateless_and_mirrors_trigger_and_reply(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    """A cron turn is stateless — it writes NO transcript of its own — and its
    trigger + delivered reply are mirrored into the human-facing (matrix)
    session for the same peer. The mirror is a user-slot provenance note
    carrying the trigger prompt, then the pristine assistant reply, so a
    follow-up from the human reads against full context.
    """
    agent = _make_agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                        fake_channel, transcripts)

    # deliver_to is a room id here (the fake can't resolve MXID->DM room).
    peer = "!room:example.org"
    cron_sid = session_id("cron", peer)
    trigger = "Remind the user that trash day is tomorrow."
    msg = InboundMessage(
        peer_id=peer, sender_name="cron", text=trigger, channel="cron",
    )
    await agent._process_batch(cron_sid, [msg], turn_id="c1")

    # The reply was delivered over the (matrix) primary channel to the peer.
    assert fake_channel.sent, "cron reply was not delivered"
    assert fake_channel.sent[-1][0] == peer

    # Stateless: the cron session accumulated NO transcript.
    assert transcripts.load(cron_sid) == [], "cron turn should persist nothing"

    # Mirrored into the matrix primary session: a user-slot note carrying the
    # trigger prompt, then the pristine assistant reply.
    matrix_sid = session_id("matrix", peer)
    assert matrix_sid != cron_sid
    mirrored = transcripts.load(matrix_sid)
    assert len(mirrored) == 2, f"expected note + reply, got {mirrored}"
    assert mirrored[0]["role"] == "user"
    assert "cron" in mirrored[0]["content"].lower()
    assert trigger in mirrored[0]["content"], "trigger prompt missing from note"
    assert mirrored[1]["role"] == "assistant" and "hello back" in mirrored[1]["content"]


async def test_matrix_turn_not_mirrored(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    """A normal matrix turn is NOT mirrored: its own sid already equals the
    channel's primary session key for the peer, so the equality guard makes
    the mirror a no-op (no duplicate rows, no provenance note). It also
    persists normally (not stateless)."""
    agent = _make_agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                        fake_channel, transcripts)
    peer = "!room:example.org"
    sid = session_id("matrix", peer)
    msg = InboundMessage(
        peer_id=peer, sender_name="user-1", text="hi",
        channel="matrix", sender_id="@user-1:example.org",
    )
    await agent._process_batch(sid, [msg], turn_id="m1")

    rows = transcripts.load(sid)
    assert not any("System note" in (r.get("content") or "") for r in rows)
    # Persisted normally: exactly one user turn (the real one) + assistant.
    assert sum(1 for r in rows if r["role"] == "user") == 1
    assert any(r["role"] == "assistant" for r in rows)
    # Exactly one user turn (the real one) + assistant reply — no mirror dupes.
    assert sum(1 for r in rows if r["role"] == "user") == 1


async def test_compaction_deferred_to_turn_end_with_notice(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    """When the compaction predicate trips, background compaction is spawned
    at turn-END (after the reply) — never before run_turn, where it would
    contend for GPU with the reply — and a 🖥️ system notice is dropped into
    the room. The reply goes out first, the notice second.
    """
    from claw.config import CompactionConfig
    # Threshold of 1 token guarantees the (overhead-inclusive) predicate trips.
    cfg = make_cfg(tmp_path, compaction=CompactionConfig(
        mid_session_token_threshold=1, reserve_tokens=1))
    agent_cfg = cfg.agents[0]
    agent_cfg.workspace.mkdir(parents=True, exist_ok=True)
    agent = Agent(
        cfg=cfg, agent_cfg=agent_cfg, ollama=fake_ollama, memory=fake_memory,
        tools={}, transcripts=transcripts, channel=fake_channel,
        spawner=None, job_runner=None,
    )
    peer = "!room:example.org"
    sid = session_id("matrix", peer)
    msg = InboundMessage(
        peer_id=peer, sender_name="user-1", text="hello",
        channel="matrix", sender_id="@user-1:example.org",
    )
    await agent._process_batch(sid, [msg], turn_id="t1")

    texts = [t for _, t in fake_channel.sent]
    reply_idx = next(i for i, t in enumerate(texts) if "hello back" in t)
    notice_idx = next(
        i for i, t in enumerate(texts) if "Auto-compaction in progress" in t
    )
    # Reply first, compaction notice second (proves turn-END spawn).
    assert notice_idx > reply_idx
    assert "🖥️" in texts[notice_idx]


async def test_no_compaction_notice_below_threshold(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    """A short turn well under the threshold neither spawns compaction nor
    emits the notice (the notice fires only when a compaction actually starts).
    """
    agent = _make_agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                        fake_channel, transcripts)  # default 96k threshold
    peer = "!room:example.org"
    sid = session_id("matrix", peer)
    msg = InboundMessage(
        peer_id=peer, sender_name="user-1", text="hi",
        channel="matrix", sender_id="@user-1:example.org",
    )
    await agent._process_batch(sid, [msg], turn_id="t1")
    assert not any(
        "Auto-compaction" in t for _, t in fake_channel.sent
    ), "notice sent despite being far below the compaction threshold"


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


async def test_cmd_stop_noop_emits_log(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts,
    caplog,
):
    """%stop with nothing running used to write zero claw.log output (only the
    operator-facing reply), so a stop was invisible server-side. Regression: it
    must emit an INFO line on claw.agent — symmetric with the turn-start/-complete
    anchors — while still replying to the operator."""
    agent = _make_agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                        fake_channel, transcripts)
    peer = "!room:example.org"
    sid = session_id("matrix", peer)
    msg = InboundMessage(
        peer_id=peer, sender_name="user-1", text="%stop",
        channel="matrix", sender_id="@user-1:example.org",
    )
    cmd = parse_command(agent.cfg.commands.prefix + "stop",
                        agent.cfg.commands.prefix)

    with caplog.at_level(logging.INFO, logger="claw.agent"):
        await agent._cmd_stop(sid, msg, cmd)

    assert any(
        "no in-flight turn to cancel" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]
    # The operator still gets the user-facing reply.
    assert fake_channel.sent, "no reply sent"
    assert "Nothing running" in fake_channel.sent[-1][1]


# --- G: voice-modality system hint -----------------------------------------

def _voice_agent(tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel,
                 transcripts, *, voice=None, **over) -> Agent:
    cfg = make_cfg(tmp_path, voice=voice if voice is not None else VoiceServiceConfig(),
                   **over)
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


def test_voice_hint_injected_for_voice_modality(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    agent = _voice_agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                         fake_channel, transcripts)
    sysprompt = agent._build_system_prompt(None, modality="voice")
    assert _HINT_MARKER in sysprompt


def test_voice_hint_absent_for_text_modality(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    agent = _voice_agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                         fake_channel, transcripts)
    # Same agent, non-voice modality -> the hint must not leak in.
    assert _HINT_MARKER not in agent._build_system_prompt(None, modality="text")
    # Default modality (None) is also non-voice.
    assert _HINT_MARKER not in agent._build_system_prompt(None)


def test_voice_hint_disabled_by_empty_config(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    # An operator can disable the hint by configuring it to "".
    agent = _voice_agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                         fake_channel, transcripts,
                         voice=VoiceServiceConfig(modality_hint=""))
    assert _HINT_MARKER not in agent._build_system_prompt(None, modality="voice")


# --- language steering (all modalities) ------------------------------------

def test_language_injected_for_all_modalities(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    # Language steering is channel/modality-agnostic — it must appear on voice,
    # text, and the default-modality prompt (a bilingual model can code-switch
    # anywhere; voice is just where it breaks).
    agent = _voice_agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                         fake_channel, transcripts, language="English (en_US)")
    for modality in ("voice", "text", None):
        assert "Always reply in English (en_US)." in \
            agent._build_system_prompt(None, modality=modality)


def test_language_absent_when_unset(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    # Default (no language configured) -> no steering line.
    agent = _voice_agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                         fake_channel, transcripts)
    assert "Always reply in" not in agent._build_system_prompt(None, modality="text")
