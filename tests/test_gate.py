"""Decision gate — the outer loop in front of the LLM.

Covers the deployed zero-handler form (pure pass-through, scorer never
called), the direct-answer path it is plumbing for (exercised with a fake
handler and a fake scorer), every fail-open branch, channel eligibility, the
state the scorer reads, and the config block. Imports claw.agent (heavy deps)
— run under the gateway venv. asyncio_mode=auto.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime, timezone

import httpx
import pytest

from claw.agent import Agent
from claw.channel.base import InboundMessage
from claw.channel.envelope import format_inbound_envelope, strip_inbound_envelope
from claw.config import (GateConfig, GateHandlerConfig, GateReplyConfig, SystemOneConfig,
                         _parse_gate,
                         _parse_gate_handler, _parse_systemone, _validate_gate)
from claw.gate import (NONE, DecisionGate, GateContext, ToolHandler, build_state,
                       build_tool_handlers, eligible)
from claw.systemone import SystemOneClient, SystemOneError
from claw.voice_http import HttpReplyChannel


class FakeScorer:
    """Stands in for SystemOneClient: records calls, answers as configured."""

    def __init__(self, choice=NONE, confidence=0.99, error=None):
        self.calls: list[tuple[object, dict]] = []
        self.choice, self.confidence, self.error = choice, confidence, error

    async def ask(self, state, questions, model="local"):
        self.calls.append((state, questions))
        if self.error is not None:
            raise self.error
        (qid,) = questions
        return {qid: {"type": "choice", "choice": self.choice,
                      "probabilities": {}, "confidence": self.confidence}}


class FakeHandler:
    def __init__(self, hid="music-pause", description="Pause the music",
                 reply="Paused.", error=None):
        self.id, self.description = hid, description
        self.reply, self.error = reply, error
        self.contexts = []

    async def handle(self, ctx):
        self.contexts.append(ctx)
        if self.error is not None:
            raise self.error
        return self.reply


def _msg(text="pause the music", channel="matrix", **kw):
    return InboundMessage(
        peer_id=kw.pop("peer_id", "!room:example.org"), sender_name="user-1",
        text=text, channel=channel, sender_id="@user-1:example.org", **kw,
    )


@pytest.fixture
def make_agent(tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts):
    def _make(scorer=None, handlers=(), **gate_over):
        gate_cfg = GateConfig(enabled=True, exposed_to=("agent-1",), **gate_over)
        cfg = make_cfg(tmp_path, gate=gate_cfg)
        agent_cfg = cfg.agents[0]
        agent_cfg.workspace.mkdir(parents=True, exist_ok=True)
        gate = DecisionGate(gate_cfg, scorer if scorer is not None else FakeScorer(),
                            handlers=tuple(handlers))
        return Agent(cfg=cfg, agent_cfg=agent_cfg, ollama=fake_ollama,
                     memory=fake_memory, tools={}, transcripts=transcripts,
                     channel=fake_channel, spawner=None, job_runner=None, gate=gate)
    return _make


SID = "matrix__room_test"


# --- the deployed form: zero handlers -------------------------------------

async def test_empty_registry_passes_through_without_calling_the_scorer(
    make_agent, fake_ollama, fake_channel, transcripts, caplog,
):
    scorer = FakeScorer(choice="anything")
    agent = make_agent(scorer=scorer)
    with caplog.at_level(logging.INFO, logger="claw.agent"):
        await agent._process_batch(SID, [_msg()], turn_id="t1")

    assert scorer.calls == [], "no handlers: the scorer must never be consulted"
    assert len(fake_ollama.turns) == 1, "the LLM answers exactly as without a gate"
    assert "hello back" in fake_channel.sent[-1][1]
    assert any("-> llm (no-handlers)" in r.getMessage() for r in caplog.records)
    rows = transcripts.load(SID)
    assert rows[0]["role"] == "user" and rows[-1]["role"] == "assistant"


async def test_ungated_agent_is_unchanged(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts, caplog,
):
    cfg = make_cfg(tmp_path)
    cfg.agents[0].workspace.mkdir(parents=True, exist_ok=True)
    agent = Agent(cfg=cfg, agent_cfg=cfg.agents[0], ollama=fake_ollama,
                  memory=fake_memory, tools={}, transcripts=transcripts,
                  channel=fake_channel, spawner=None, job_runner=None)
    with caplog.at_level(logging.INFO, logger="claw.agent"):
        await agent._process_batch(SID, [_msg()], turn_id="t1")
    assert len(fake_ollama.turns) == 1
    assert not any(":gate:" in r.getMessage() for r in caplog.records)


# --- the direct path the gate is plumbing for ------------------------------

async def test_chosen_handler_answers_instead_of_the_llm(
    make_agent, fake_ollama, fake_memory, fake_channel, transcripts, caplog,
):
    handler = FakeHandler()
    agent = make_agent(scorer=FakeScorer(choice="music-pause", confidence=0.95),
                       handlers=[handler])
    with caplog.at_level(logging.INFO, logger="claw.agent"):
        await agent._process_batch(SID, [_msg()], turn_id="t1")

    assert fake_ollama.turns == [], "a direct answer must not reach the LLM"
    assert fake_memory.queries == [], "a direct answer costs no retrieval"
    assert fake_channel.sent == [("!room:example.org", "Paused.")]
    rows = transcripts.load(SID)
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert rows[0]["content"].endswith("pause the music")   # enveloped, like any turn
    assert rows[1]["content"] == "Paused."
    (ctx,) = handler.contexts
    assert ctx.text == "user-1: pause the music"
    msgs = [r.getMessage() for r in caplog.records]
    assert any("-> music-pause (chosen, music-pause @ 0.95)" in m for m in msgs)
    assert any("turn complete (direct: music-pause" in m for m in msgs)


async def test_next_turn_sees_the_direct_exchange(
    make_agent, fake_ollama, transcripts,
):
    agent = make_agent(scorer=FakeScorer(choice="music-pause"), handlers=[FakeHandler()])
    await agent._process_batch(SID, [_msg()], turn_id="t1")
    agent.gate.client.choice = NONE
    await agent._process_batch(SID, [_msg("why did it stop?")], turn_id="t2")

    (turn,) = fake_ollama.turns
    history = " | ".join(str(m.get("content")) for m in turn["history"])
    assert "pause the music" in history and "Paused." in history


async def test_voice_direct_answer_resolves_the_waiting_request(
    make_agent, fake_ollama, fake_channel,
):
    agent = make_agent(scorer=FakeScorer(choice="music-pause"), handlers=[FakeHandler()])
    voice = HttpReplyChannel()
    agent.register_channel("voice", voice)
    peer = "http:turn-1"
    fut = voice.expect(peer)

    await agent._process_batch("home", [_msg(channel="voice", peer_id=peer)], turn_id="t1")

    assert fut.done() and fut.result() == "Paused."
    assert fake_ollama.turns == [] and fake_channel.sent == []


async def test_scorer_sees_recent_conversation(make_agent, transcripts):
    scorer = FakeScorer(choice=NONE)
    agent = make_agent(scorer=scorer, handlers=[FakeHandler()], context_turns=2)
    for i in range(3):
        transcripts.append(SID, {"role": "user", "content": f"[matrix user-1 Sun 2026-05-03 07:3{i}] q{i}"})
        transcripts.append(SID, {"role": "assistant", "content": f"a{i}"})
    await agent._process_batch(SID, [_msg("skip this one")], turn_id="t1")

    (state, questions), = scorer.calls
    assert state == [
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "user-1: skip this one"},
    ]
    (q,) = questions.values()
    assert q["type"] == "choice"
    assert set(q["criteria"]) == {"music-pause", NONE}
    assert q["criteria"]["music-pause"] == "Pause the music"


# --- fail open ---------------------------------------------------------------

@pytest.mark.parametrize("scorer, reason", [
    (FakeScorer(choice=NONE), "none"),
    (FakeScorer(choice="music-pause", confidence=0.5), "low-confidence"),
    (FakeScorer(choice="not-a-handler"), "none"),
    (FakeScorer(error=httpx.ReadTimeout("slow")), "scorer-error"),
    (FakeScorer(error=httpx.ConnectError("down")), "scorer-error"),
    (FakeScorer(error=SystemOneError("HTTP 422")), "scorer-error"),
])
async def test_every_non_choice_goes_to_the_llm(
    make_agent, fake_ollama, fake_channel, caplog, scorer, reason,
):
    handler = FakeHandler()
    agent = make_agent(scorer=scorer, handlers=[handler])
    with caplog.at_level(logging.INFO, logger="claw.agent"):
        await agent._process_batch(SID, [_msg()], turn_id="t1")
    assert len(fake_ollama.turns) == 1
    assert handler.contexts == []
    assert "hello back" in fake_channel.sent[-1][1]
    assert any(f"-> llm ({reason}" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("handler", [
    FakeHandler(error=RuntimeError("device offline")),
    FakeHandler(reply="   "),
])
async def test_failed_handler_falls_through_without_writing(
    make_agent, fake_ollama, fake_channel, transcripts, handler,
):
    agent = make_agent(scorer=FakeScorer(choice="music-pause"), handlers=[handler])
    await agent._process_batch(SID, [_msg()], turn_id="t1")

    assert len(fake_ollama.turns) == 1
    rows = transcripts.load(SID)
    assert [r["role"] for r in rows].count("user") == 1, "no duplicate user row"
    assert "hello back" in fake_channel.sent[-1][1]


# --- who is gated ------------------------------------------------------------

@pytest.mark.parametrize("msgs", [
    [_msg(channel="initial_prompt", peer_id="bootstrap")],
    [_msg(is_subagent_completion=True)],
    [_msg("pause"), _msg("and then tell me a story")],
])
async def test_non_human_and_batched_turns_never_consult_the_gate(
    make_agent, fake_ollama, caplog, msgs,
):
    scorer = FakeScorer(choice="music-pause")
    agent = make_agent(scorer=scorer, handlers=[FakeHandler()])
    with caplog.at_level(logging.INFO, logger="claw.agent"):
        await agent._process_batch(SID, msgs, turn_id="t1")
    assert scorer.calls == []
    assert len(fake_ollama.turns) == 1
    assert not any(":gate:" in r.getMessage() for r in caplog.records)


async def test_cron_turns_never_consult_the_gate(make_agent, fake_ollama):
    scorer = FakeScorer(choice="music-pause")
    agent = make_agent(scorer=scorer, handlers=[FakeHandler()])
    await agent._process_batch("cron_job-1", [_msg("reminder", channel="cron")], turn_id="t1")
    assert scorer.calls == []
    assert len(fake_ollama.turns) == 1


def test_eligible():
    assert eligible([_msg()])
    assert eligible([_msg(channel="voice")])
    assert not eligible([_msg(channel="cron")])
    assert not eligible([_msg(), _msg()])


# --- state the scorer reads --------------------------------------------------

def test_build_state_drops_tool_traffic_and_synthetic_notes():
    rows = [
        {"role": "user", "content": "## Pre-compaction Recap\nCovers turns ..."},
        {"role": "user", "content": "[matrix user-1 +47m Sun 2026-05-03 07:30] play something"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "x"}}]},
        {"role": "tool", "content": "tool output"},
        {"role": "assistant", "content": "Playing an album."},
        {"role": "user", "content": "⚙️ System note — not from the user. ..."},
        {"role": "user", "content": "[SYSTEM (out-of-band notice — not a user message ...)]"},
        {"role": "assistant", "content": "A proactive reminder."},
    ]
    assert build_state(rows, "louder", context_turns=6) == [
        {"role": "user", "content": "play something"},
        {"role": "assistant", "content": "Playing an album."},
        {"role": "assistant", "content": "A proactive reminder."},
        {"role": "user", "content": "louder"},
    ]
    assert build_state(rows, "louder", context_turns=0) == [{"role": "user", "content": "louder"}]


def test_strip_envelope_inverts_format():
    ts = datetime(2026, 5, 3, 11, 30, tzinfo=timezone.utc)
    for sender, prev in ((None, None), ("user-1", None), ("user-1", ts)):
        wrapped = format_inbound_envelope("matrix", sender, "body [with brackets]", ts, prev)
        assert strip_inbound_envelope(wrapped) == "body [with brackets]"
    assert strip_inbound_envelope("[not an envelope] text") == "[not an envelope] text"


# --- construction and config -------------------------------------------------

def test_gate_rejects_reserved_and_duplicate_ids():
    cfg = GateConfig(enabled=True)
    with pytest.raises(ValueError, match="reserved"):
        DecisionGate(cfg, FakeScorer(), handlers=(FakeHandler(hid=NONE),))
    with pytest.raises(ValueError, match="duplicate"):
        DecisionGate(cfg, FakeScorer(), handlers=(FakeHandler(), FakeHandler()))
    with pytest.raises(ValueError, match="no scorer"):
        DecisionGate(cfg, None, handlers=(FakeHandler(),))
    DecisionGate(cfg, None)   # zero handlers needs no scorer


def test_parse_gate():
    assert _parse_gate(None) == GateConfig()
    g = _parse_gate({"enabled": True, "exposed_to": ["agent-1"],
                     "min_confidence": "0.9", "context_turns": "4"})
    assert g == GateConfig(enabled=True, exposed_to=("agent-1",),
                           min_confidence=0.9, context_turns=4)
    with pytest.raises(TypeError):
        _parse_gate({"enabled": True, "min_confidance": 0.9})
    with pytest.raises(TypeError):   # moved to the systemone block
        _parse_gate({"enabled": True, "base_url": "http://scorer.invalid"})


@pytest.mark.parametrize("over, fragment", [
    ({"exposed_to": ("nobody",)}, "unknown agent"),
    ({"min_confidence": 1.5}, "min_confidence"),
    ({"context_turns": -1}, "context_turns"),
])
def test_validate_gate(tmp_path, make_cfg, over, fragment):
    cfg = make_cfg(tmp_path)
    gate = dataclasses.replace(
        GateConfig(enabled=True, exposed_to=(cfg.agents[0].id,)), **over)
    cfg = dataclasses.replace(cfg, gate=gate)
    with pytest.raises(ValueError, match=fragment):
        _validate_gate(cfg)


def test_disabled_gate_is_not_validated(tmp_path, make_cfg):
    _validate_gate(make_cfg(tmp_path, gate=GateConfig(exposed_to=("nobody",))))


# --- client ------------------------------------------------------------------

def _client(handler) -> SystemOneClient:
    c = SystemOneClient("http://scorer.invalid", 1.0)
    c._client = httpx.AsyncClient(base_url="http://scorer.invalid",
                                  transport=httpx.MockTransport(handler))
    return c


async def test_client_returns_answers_and_sends_the_wire_shape():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"model": "m", "answers": {
            "q": {"type": "noul", "noul": 0.9}}, "usage": {}})

    c = _client(handler)
    answers = await c.ask("state", {"q": {"type": "noul", "instructions": "?"}})
    assert answers == {"q": {"type": "noul", "noul": 0.9}}
    assert seen["state"] == "state" and "q" in seen["questions"]
    await c.aclose()


@pytest.mark.parametrize("response", [
    httpx.Response(422, json={"error": {"message": "bad"}}),
    httpx.Response(200, json={"answers": {}}),
])
async def test_client_raises_on_unusable_response(response):
    c = _client(lambda request: response)
    with pytest.raises(SystemOneError):
        await c.ask("state", {"q": {"type": "noul", "instructions": "?"}})
    await c.aclose()


# --- the scorer is optional: systemone block ---------------------------------

def test_parse_systemone():
    assert _parse_systemone(None) is None
    assert _parse_systemone({}) == SystemOneConfig()
    assert _parse_systemone({"base_url": "http://scorer.invalid", "timeout_s": "3"}) \
        == SystemOneConfig(base_url="http://scorer.invalid", timeout_s=3.0)
    with pytest.raises(TypeError):
        _parse_systemone({"base_uri": "x"})


def test_handlers_require_a_scorer(tmp_path, make_cfg):
    cfg = make_cfg(tmp_path)
    gate = GateConfig(enabled=True, exposed_to=(cfg.agents[0].id,),
                      handlers=(_hcfg(),))
    with pytest.raises(ValueError, match="need a scorer"):
        _validate_gate(dataclasses.replace(cfg, gate=gate))
    _validate_gate(dataclasses.replace(cfg, gate=gate, systemone=SystemOneConfig()))
    # A gate with no handlers needs no scorer at all.
    _validate_gate(dataclasses.replace(cfg, gate=dataclasses.replace(gate, handlers=())))


def test_handler_ids_reserved_and_unique(tmp_path, make_cfg):
    cfg = dataclasses.replace(make_cfg(tmp_path), systemone=SystemOneConfig())
    for handlers, fragment in (
        ((_hcfg(hid=NONE),), "reserved"),
        ((_hcfg(), _hcfg()), "duplicate"),
    ):
        gate = GateConfig(enabled=True, exposed_to=(cfg.agents[0].id,), handlers=handlers)
        with pytest.raises(ValueError, match=fragment):
            _validate_gate(dataclasses.replace(cfg, gate=gate))


async def test_no_scorer_means_agent_runs_as_before(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts, caplog,
):
    """No systemone block: no client, and a configured gate is pure
    pass-through — the claw behaves exactly as without the feature."""
    cfg = make_cfg(tmp_path)
    cfg = dataclasses.replace(cfg, gate=GateConfig(enabled=True, exposed_to=(cfg.agents[0].id,)))
    cfg.agents[0].workspace.mkdir(parents=True, exist_ok=True)
    agent = Agent(cfg=cfg, agent_cfg=cfg.agents[0], ollama=fake_ollama,
                  memory=fake_memory, tools={}, transcripts=transcripts,
                  channel=fake_channel, spawner=None, job_runner=None)
    assert agent.systemone is None
    assert agent.gate is not None and agent.gate.handlers == {}
    with caplog.at_level(logging.INFO, logger="claw.agent"):
        await agent._process_batch(SID, [_msg()], turn_id="t1")
    assert len(fake_ollama.turns) == 1
    assert any("-> llm (no-handlers)" in r.getMessage() for r in caplog.records)


# --- declarative handlers ------------------------------------------------------

def _hcfg(hid="audio-pause", **kw):
    """A text-shape handler: expect + one reply form."""
    base = dict(id=hid, description="Pause the audio", tool="music_control",
                args={"action": "pause"}, expect="paused",
                reply=GateReplyConfig(reply=("Paused.",)))
    base.update(kw)
    return GateHandlerConfig(**base)


def _skip_cfg(**kw):
    """The outcomes-shape skip handler, as deployed."""
    base = dict(
        id="audio-next", description="Skip to the next track", tool="music_control",
        args={"action": "next"},
        outcomes={
            "skipped": GateReplyConfig(reply_fn="skipping_to", fallback=("Skipping.",)),
            "end_of_queue": GateReplyConfig(reply=("That was the last one.",)),
        })
    base.update(kw)
    return GateHandlerConfig(**base)


class FakeTool:
    data = None     # no structured twin, like most real tools

    def __init__(self, output="paused", error=None):
        self.output, self.error = output, error
        self.calls: list[dict] = []

    async def run(self, args):
        self.calls.append(args)
        if self.error is not None:
            raise self.error
        return self.output


class FakeControl(FakeTool):
    """music_control with its structured twin — which IS the action."""

    def __init__(self, result):
        super().__init__("prose for the model")
        self.result = result
        self.data_calls: list[dict] = []

    async def data(self, args):
        self.data_calls.append(args)
        return self.result


class FakeStatus(FakeTool):
    def __init__(self, now=None, error=None):
        super().__init__("prose for the model, which code must not parse")
        self.now, self.data_error = now or {}, error

    async def data(self, args):
        if self.data_error is not None:
            raise self.data_error
        return self.now


def _ctx():
    return GateContext(agent_id="agent-1", sid=SID, msgs=[_msg()], text="x", state=[])


NEXT = {"title": "Marcus Garvey", "artist": "Burning Spear", "album": "Marcus Garvey"}


# config shapes

def test_parse_reply_forms():
    parse = _parse_gate_handler
    base = {"id": "h", "description": "d", "tool": "t"}
    assert parse({**base, "reply": "Paused."}).reply == GateReplyConfig(reply=("Paused.",))
    assert parse({**base, "reply": ["a", "b"]}).reply.reply == ("a", "b")
    assert parse({**base, "relay": True}).reply.relay is True
    fn = parse({**base, "reply_fn": "playing_brief", "fallback": ["x"]}).reply
    assert fn == GateReplyConfig(reply_fn="playing_brief", fallback=("x",))
    assert parse({**base, "reply_fn": "playing_brief"}).reply.fallback == ()   # text shape: tool's words
    for bad, fragment in (
        ({**base}, "exactly one"),
        ({**base, "reply": "x", "reply_fn": "playing_brief"}, "exactly one"),
        ({**base, "relay": True, "reply": "x"}, "exactly one"),
        ({**base, "reply": "x", "fallback": "y"}, "fallback only"),
        ({**base, "reply_fn": "nope"}, "unknown reply_fn"),
        ({**base, "reply": "x", "min_confidence": 2}, "min_confidence"),
    ):
        with pytest.raises(ValueError, match=fragment):
            parse(bad)
    with pytest.raises(TypeError):
        parse({**base, "reply": "x", "tools": "t2"})


def test_parse_outcomes():
    parse = _parse_gate_handler
    base = {"id": "h", "description": "d", "tool": "t"}
    h = parse({**base, "outcomes": {
        "skipped": {"reply_fn": "skipping_to", "fallback": ["Skipping."]},
        "end_of_queue": {"reply": "That was the last one."}}})
    assert h.reply is None and set(h.outcomes) == {"skipped", "end_of_queue"}
    for bad, fragment in (
        ({**base, "outcomes": {}}, "empty"),
        ({**base, "reply": "x", "outcomes": {"a": {"reply": "y"}}}, "replaces"),
        ({**base, "expect": "a", "outcomes": {"a": {"reply": "y"}}}, "replaces"),
        ({**base, "outcomes": {"a": {"relay": True}}}, "no text output"),
        ({**base, "outcomes": {"a": {"reply_fn": "skipping_to"}}}, "needs a fallback"),
        ({**base, "outcomes": {"a": {"reply": "y", "expect": "z"}}}, "unknown key"),
    ):
        with pytest.raises(ValueError, match=fragment):
            parse(bad)


# text shape

async def test_text_handler_success_and_varied_reply():
    tool = FakeTool("paused")
    h = ToolHandler(_hcfg(reply=GateReplyConfig(reply=("Paused.", "Pausing."))),
                    {"music_control": tool})
    assert {await h.handle(_ctx()) for _ in range(40)} == {"Paused.", "Pausing."}
    assert tool.calls[0] == {"action": "pause"}


@pytest.mark.parametrize("output", [
    "error: the music player is not running",
    "refused: speaker unreachable",
    "something other than paused",
])
async def test_text_handler_declines_on_anything_but_success(output):
    h = ToolHandler(_hcfg(), {"music_control": FakeTool(output)})
    assert await h.handle(_ctx()) is None


async def test_text_handler_relays_output():
    h = ToolHandler(_hcfg(hid="s", tool="music_status", args={}, expect=None,
                          reply=GateReplyConfig(relay=True)),
                    {"music_status": FakeTool("playing A — B")})
    assert await h.handle(_ctx()) == "playing A — B"


# outcomes shape: skip

async def test_skip_names_the_next_track_from_the_tools_own_result():
    ctl = FakeControl({"result": "skipped", "next": NEXT})
    h = ToolHandler(_skip_cfg(), {"music_control": ctl})
    assert await h.handle(_ctx()) == "Skipping to Marcus Garvey by Burning Spear."
    assert ctl.data_calls == [{"action": "next"}] and ctl.calls == [], "the action ran once"


@pytest.mark.parametrize("result, reply", [
    ({"result": "skipped", "next": {"link": True}}, "Skipping — the DJ's up next."),
    ({"result": "skipped", "next": {"title": "stream.mp3", "artist": None}}, "Skipping to stream.mp3."),
    ({"result": "skipped", "next": None}, "Skipping."),          # uncatalogued → fallback
    ({"result": "end_of_queue", "next": None}, "That was the last one."),
])
async def test_skip_outcomes(result, reply):
    h = ToolHandler(_skip_cfg(), {"music_control": FakeControl(result)})
    assert await h.handle(_ctx()) == reply


@pytest.mark.parametrize("result", ["idle", "down", "error"])
async def test_unlisted_outcomes_go_to_the_llm(result):
    h = ToolHandler(_skip_cfg(), {"music_control": FakeControl({"result": result})})
    assert await h.handle(_ctx()) is None


# outcomes/reply_fn failure handling

async def test_a_failing_reply_fn_uses_its_fallback(monkeypatch):
    async def boom(*a):
        raise RuntimeError("bug")
    import claw.gate as gate_mod
    monkeypatch.setitem(gate_mod.REPLY_FUNCTIONS, "skipping_to", boom)
    h = ToolHandler(_skip_cfg(), {"music_control": FakeControl({"result": "skipped", "next": NEXT})})
    assert await h.handle(_ctx()) == "Skipping."


# status

def _status_handler(tool):
    return ToolHandler(_hcfg(hid="audio-status", tool="music_status", args={}, expect=None,
                             reply=GateReplyConfig(reply_fn="playing_brief")),
                       {"music_status": tool})


@pytest.mark.parametrize("now, reply", [
    ({**NEXT, "state": "playing"}, "Now playing: Marcus Garvey by Burning Spear."),
    ({**NEXT, "state": "paused"}, "Paused: Marcus Garvey by Burning Spear."),
    ({"state": "playing", "title": "stream.mp3", "artist": None}, "Now playing: stream.mp3."),
    ({"state": "idle"}, "Nothing is playing."),
    ({"state": "link"}, "Between songs right now — that's the DJ talking."),
])
async def test_playing_brief(now, reply):
    assert await _status_handler(FakeStatus(now)).handle(_ctx()) == reply


@pytest.mark.parametrize("tool", [
    FakeStatus({"state": "loading"}), FakeStatus({"state": "down"}),
    FakeStatus(error=RuntimeError("socket gone")), FakeTool("playing something"),
])
async def test_playing_brief_falls_back_to_the_tools_own_words(tool):
    assert await _status_handler(tool).handle(_ctx()) == tool.output


# binding and thresholds

def test_binding_drops_missing_tools_and_outcomes_without_data(caplog):
    with caplog.at_level(logging.ERROR, logger="claw.gate"):
        bound = build_tool_handlers(
            (_hcfg(), _hcfg(hid="h2", tool="no_such_tool"), _skip_cfg()),
            {"music_control": FakeTool()}, "agent-1")
    assert [h.id for h in bound] == ["audio-pause"]
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "no_such_tool" in msgs and "no structured data" in msgs


def test_binding_drops_a_handler_passing_an_undeclared_argument(caplog):
    """args go straight to the tool, which reads what it knows and ignores the
    rest — so a misspelled key would otherwise load clean and fail per turn,
    silently, as a call with that argument simply missing."""
    class Schemad(FakeTool):
        input_schema = {"type": "object", "properties": {"action": {"type": "string"}}}

    with caplog.at_level(logging.ERROR, logger="claw.gate"):
        bound = build_tool_handlers(
            (_hcfg(), _hcfg(hid="typo", args={"actoin": "pause"})),
            {"music_control": Schemad()}, "agent-1")
    assert [h.id for h in bound] == ["audio-pause"]
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "actoin" in msgs and "typo" in msgs


def test_binding_leaves_a_tool_without_a_schema_alone():
    """Nothing declared, nothing to check against — not an error."""
    bound = build_tool_handlers((_hcfg(args={"anything": 1}),),
                                {"music_control": FakeTool()}, "agent-1")
    assert [h.id for h in bound] == ["audio-pause"]


async def test_per_handler_threshold():
    cfg = GateConfig(enabled=True, min_confidence=0.8)
    strict = ToolHandler(_hcfg(hid="audio-stop", min_confidence=0.95), {"music_control": FakeTool()})
    gate = DecisionGate(cfg, FakeScorer(choice="audio-stop", confidence=0.9), handlers=(strict,))
    d = await gate.decide("agent-1", SID, [_msg()], "stop", lambda: [])
    assert d.reason == "low-confidence"
    gate.client.confidence = 0.97
    d = await gate.decide("agent-1", SID, [_msg()], "stop", lambda: [])
    assert d.reason == "chosen"


# end to end through the Agent

async def test_agent_builds_gate_from_config_and_handles_a_turn(
    tmp_path, make_cfg, fake_ollama, fake_memory, fake_channel, transcripts,
):
    cfg = make_cfg(tmp_path)
    aid = cfg.agents[0].id
    cfg = dataclasses.replace(
        cfg, systemone=SystemOneConfig(),
        gate=GateConfig(enabled=True, exposed_to=(aid,), handlers=(_skip_cfg(),)))
    cfg.agents[0].workspace.mkdir(parents=True, exist_ok=True)
    ctl = FakeControl({"result": "skipped", "next": NEXT})
    agent = Agent(cfg=cfg, agent_cfg=cfg.agents[0], ollama=fake_ollama,
                  memory=fake_memory, tools={"music_control": ctl},
                  transcripts=transcripts, channel=fake_channel, spawner=None,
                  job_runner=None, systemone=FakeScorer(choice="audio-next"))
    assert set(agent.gate.handlers) == {"audio-next"}
    await agent._process_batch(SID, [_msg("skip this song")], turn_id="t1")
    assert ctl.data_calls == [{"action": "next"}]
    assert fake_ollama.turns == []
    assert fake_channel.sent[-1][1] == "Skipping to Marcus Garvey by Burning Spear."


async def test_declining_handler_leaves_the_turn_to_the_llm(
    make_agent, fake_ollama, fake_channel, transcripts,
):
    tool = FakeTool("error: the music player is not running")
    handler = ToolHandler(_hcfg(), {"music_control": tool})
    agent = make_agent(scorer=FakeScorer(choice="audio-pause"), handlers=[handler])
    await agent._process_batch(SID, [_msg("pause")], turn_id="t1")
    assert tool.calls, "the action was attempted"
    assert len(fake_ollama.turns) == 1, "and the LLM then answered"
    assert [r["role"] for r in transcripts.load(SID)].count("user") == 1
