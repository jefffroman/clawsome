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
    sysprompt = agent._build_system_prompt("", None, modality="voice")
    assert _HINT_MARKER in sysprompt


def test_voice_hint_absent_for_text_modality(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    agent = _voice_agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                         fake_channel, transcripts)
    # Same agent, non-voice modality -> the hint must not leak in.
    assert _HINT_MARKER not in agent._build_system_prompt("", None, modality="text")
    # Default modality (None) is also non-voice.
    assert _HINT_MARKER not in agent._build_system_prompt("", None)


def test_voice_hint_disabled_by_empty_config(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    # An operator can disable the hint by configuring it to "".
    agent = _voice_agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                         fake_channel, transcripts,
                         voice=VoiceServiceConfig(modality_hint=""))
    assert _HINT_MARKER not in agent._build_system_prompt("", None, modality="voice")


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
            agent._build_system_prompt("", None, modality=modality)


def test_language_absent_when_unset(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    # Default (no language configured) -> no steering line.
    agent = _voice_agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                         fake_channel, transcripts)
    assert "Always reply in" not in agent._build_system_prompt("", None, modality="text")
