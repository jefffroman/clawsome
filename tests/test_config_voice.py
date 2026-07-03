"""config.load() voice + http_api + devices parsing and validation.

The audio pipeline left clawsome (voice-gateway sibling); claw keeps only the
claw-side voice config (the modality hint), the HTTP turn endpoint config, and
device routing. PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

import textwrap

import pytest

from claw import config


def _yaml(
    tmp_path,
    *,
    with_voice: bool = False,
    with_http: bool = False,
    with_devices: bool = False,
    device_agent: str = "example",
    device_endpoints: bool = True,
) -> str:
    base = textwrap.dedent(f"""\
    ollama:
      base_url: http://ollama.invalid
      default_compaction_model: test-compact-model
    searxng:
      base_url: http://searx.invalid
    cron:
      enabled: false
      jobs_file: {tmp_path}/jobs.json
    subagents:
      max_concurrent: 1
      max_children_per_agent: 1
      default_model: test-model
      personas:
        persona-3: {{ model: m, role: grunt }}
    """)

    voice = textwrap.dedent("""\
    voice:
      modality_hint: "speak briefly"
    """) if with_voice else ""

    http = textwrap.dedent("""\
    http_api:
      enabled: true
      bind_host: 0.0.0.0
      bind_port: 11599
    """) if with_http else ""

    # Indented to sit under the device list item (its keys are at 4 spaces).
    eps = (
        "    endpoints:\n"
        "      - { id: voice, type: voice, name: Box3 }\n"
    ) if device_endpoints else "    endpoints: []\n"
    devices = (textwrap.dedent(f"""\
    devices:
      - device_id: box3
        name: Box3
        agent: {device_agent}
    """) + eps) if with_devices else ""

    agents_head = textwrap.dedent(f"""\
    agents:
      - id: example
        workspace: {tmp_path}/ws
        primary_model: test-model
        matrix:
          user_id: "@bot:example.org"
          homeserver: https://example.org
          access_token_file: {tmp_path}/bot.token
          device_id: CLAW_TEST
          device_name: Claw Test
          store_path: {tmp_path}/store
    """)

    return base + voice + http + devices + agents_head


def _load(tmp_path, **kw):
    p = tmp_path / "claw.yaml"
    p.write_text(_yaml(tmp_path, **kw))
    return config.load(p)


# --- voice (modality) config ----------------------------------------------

def test_voice_block_parses_modality_hint(tmp_path):
    cfg = _load(tmp_path, with_voice=True)
    assert cfg.voice is not None
    assert cfg.voice.modality_hint == "speak briefly"


def test_no_voice_block_is_none(tmp_path):
    cfg = _load(tmp_path)
    assert cfg.voice is None
    assert cfg.http_api is None


def test_voice_service_default_hint_present():
    svc = config.VoiceServiceConfig()
    assert "voice channel" in svc.modality_hint.lower()


# --- http_api config -------------------------------------------------------

def test_http_api_block_parses(tmp_path):
    cfg = _load(tmp_path, with_http=True)
    assert cfg.http_api is not None
    assert cfg.http_api.enabled is True
    assert cfg.http_api.bind_host == "0.0.0.0"
    assert cfg.http_api.bind_port == 11599


def test_http_api_defaults():
    c = config.HttpApiConfig()
    assert c.enabled is False
    assert c.bind_host == "0.0.0.0"
    assert c.bind_port == 11501


def test_http_api_parse_empty_is_none_minimal_defaults():
    # An empty/absent block -> None (not stood up). A non-empty one defaults.
    assert config._parse_http_api({}) is None
    c = config._parse_http_api({"enabled": True})
    assert c.enabled is True
    assert c.bind_port == 11501


# --- devices + validation --------------------------------------------------

def test_devices_parse_and_validate(tmp_path):
    cfg = _load(tmp_path, with_devices=True)
    assert len(cfg.devices) == 1
    dev = cfg.devices[0]
    assert dev.device_id == "box3"
    assert dev.agent == "example"
    assert dev.endpoints[0].id == "voice"
    assert dev.endpoints[0].type == "voice"


def test_device_unknown_agent_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown agent"):
        _load(tmp_path, with_devices=True, device_agent="ghost")


def test_device_without_endpoints_rejected(tmp_path):
    with pytest.raises(ValueError, match="no endpoints"):
        _load(tmp_path, with_devices=True, device_endpoints=False)


def test_agent_voice_key_ignored(tmp_path):
    # Voice rendering is gateway-side: claw no longer has a per-agent voice
    # setting. A stray agents[].voice key is simply ignored (forward-compat),
    # not parsed into config.
    p = tmp_path / "claw.yaml"
    body = _yaml(tmp_path)
    body += "    voice:\n      piper_voice: en_US-libritts-high\n"
    p.write_text(body)
    cfg = config.load(p)
    assert not hasattr(cfg.agents[0], "voice")
