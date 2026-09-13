"""%clear <session> — reaching a session you are not typing in.

Voice turns all share the literal "home" sid and cannot issue commands: the
control plane is matrix-only, and "%clear" would not survive STT in any case.
Before this, nothing could clear the voice transcript short of waiting for the
nightly rotate or archiving the file by hand.
"""
from claw.agent import Agent
from claw.channel.base import InboundMessage
from claw.commands import parse_command


def _make_agent(tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel,
                transcripts) -> Agent:
    cfg = make_cfg(tmp_path)
    agent_cfg = cfg.agents[0]
    agent_cfg.workspace.mkdir(parents=True, exist_ok=True)
    return Agent(
        cfg=cfg, agent_cfg=agent_cfg, ollama=fake_ollama, memory=fake_memory,
        tools={}, transcripts=transcripts, channel=fake_channel,
        spawner=None, job_runner=None,
    )


def _msg(text: str) -> InboundMessage:
    return InboundMessage(
        peer_id="!room:example.org", sender_name="alice", text=text,
        channel="matrix", sender_id="@alice:example.org",
    )


def _replies(fake_channel) -> str:
    return " | ".join(t for _p, t in fake_channel.sent)


async def test_clears_a_named_session_not_the_current_one(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    agent = _make_agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                        fake_channel, transcripts)
    transcripts.append("home", {"role": "user", "content": "spoken turn"})
    transcripts.append("matrix__room", {"role": "user", "content": "typed turn"})

    await agent._cmd_clear("matrix__room", _msg("%clear home"),
                           parse_command("%clear home", "%"))

    assert transcripts.load("home") == [], "the named session should be cleared"
    assert transcripts.load("matrix__room"), \
        "the session the command was typed in must be untouched"
    assert "'home'" in _replies(fake_channel), \
        "the reply should name which session was cleared, since it may not be this one"


async def test_bare_clear_still_clears_the_current_session(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    """The existing behaviour is unchanged — the argument is optional."""
    agent = _make_agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                        fake_channel, transcripts)
    transcripts.append("home", {"role": "user", "content": "spoken"})
    transcripts.append("matrix__room", {"role": "user", "content": "typed"})

    await agent._cmd_clear("matrix__room", _msg("%clear"),
                           parse_command("%clear", "%"))

    assert transcripts.load("matrix__room") == []
    assert transcripts.load("home"), "an unnamed clear must not touch other sessions"


async def test_unknown_session_lists_the_valid_ones(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    """Discoverable rather than memorised — you should not have to know the
    literal 'home' to use this."""
    agent = _make_agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                        fake_channel, transcripts)
    transcripts.append("home", {"role": "user", "content": "x"})

    await agent._cmd_clear("home", _msg("%clear nope"),
                           parse_command("%clear nope", "%"))

    reply = _replies(fake_channel)
    assert "Unknown session" in reply and "home" in reply
    assert transcripts.load("home"), "a typo must not clear anything"


def test_active_sids_excludes_archives(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts
):
    """The nameable set and the rotating set are the same enumeration — a name
    accepted by one and unknown to the other is a confusing way to lose a
    transcript."""
    agent = _make_agent(tmp_path, make_cfg, fake_ollama, fake_memory,
                        fake_channel, transcripts)
    transcripts.append("home", {"role": "user", "content": "x"})
    archived = transcripts.dir / "home.reset-2026-09-05T04-00-00+00-00.jsonl"
    archived.write_text('{"role": "user", "content": "old"}\n')

    sids = agent.active_sids()
    assert "home" in sids
    assert not any("reset-" in s for s in sids), "archives are not live sessions"
