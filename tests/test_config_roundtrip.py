"""config.load() round-trip from a PII-clean in-test YAML.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

import textwrap

import pytest

from claw import config


def _yaml(tmp_path, *, bad_can_spawn: bool = False) -> str:
    cs = "[persona-9]" if bad_can_spawn else "[persona-3]"
    return textwrap.dedent(f"""
    verbose: false
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
        Persona-1: {{ model: m, role: researcher, max_spawn_depth: 1, can_spawn: {cs} }}
        persona-3: {{ model: m, role: grunt }}
    commands:
      enabled: true
      prefix: "%"
      allow:
        - "@user-1:example.org"
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


def test_config_load_roundtrip(tmp_path):
    p = tmp_path / "claw.yaml"
    p.write_text(_yaml(tmp_path))
    cfg = config.load(p)

    assert cfg.ollama.base_url == "http://ollama.invalid"
    assert cfg.commands.prefix == "%"
    assert cfg.commands.allow == ("@user-1:example.org",)
    assert cfg.agents[0].id == "example"
    assert cfg.agents[0].matrix.user_id == "@bot:example.org"
    # persona keys are lowercased on load.
    assert set(cfg.subagents.personas) == {"persona-1", "persona-3"}
    assert cfg.subagents.personas["persona-1"].can_spawn == ("persona-3",)


def test_config_load_rejects_unknown_can_spawn(tmp_path):
    p = tmp_path / "claw.yaml"
    p.write_text(_yaml(tmp_path, bad_can_spawn=True))
    with pytest.raises(ValueError, match="unknown persona"):
        config.load(p)


def _yaml_bare_optionals(tmp_path) -> str:
    # Every optional all-defaulted block present-but-null (YAML `key:` with no
    # value -> None). Regression guard: _parse must coalesce None -> defaults,
    # not splat ``**None`` (the claw.example.yaml bare-`lifecycle:` crash).
    return textwrap.dedent(f"""
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
    memory_retrieval:
    compaction:
    memory_flush:
    lifecycle:
    commands:
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


def test_config_load_present_but_null_optionals_use_defaults(tmp_path):
    p = tmp_path / "claw.yaml"
    p.write_text(_yaml_bare_optionals(tmp_path))
    cfg = config.load(p)  # must not raise on `lifecycle:` / siblings == None

    assert cfg.memory_retrieval == config.MemoryRetrievalConfig()
    assert cfg.compaction == config.CompactionConfig()
    assert cfg.memory_flush == config.MemoryFlushConfig()
    assert cfg.lifecycle == config.LifecycleConfig()
    assert cfg.commands == config.CommandsConfig()


# --- smart retrieval -----------------------------------------------------------

def _yaml_smart(tmp_path, *, enabled: bool, scorer: bool) -> str:
    extra = "systemone:\n  base_url: http://scorer.invalid\n" if scorer else ""
    return _yaml(tmp_path) + textwrap.dedent(f"""
    memory_retrieval:
      smart_retrieval:
        enabled: {str(enabled).lower()}
    """) + extra


def test_smart_retrieval_without_a_scorer_is_a_config_error(tmp_path):
    """It needs a scorer, and it says so at load rather than at the first turn.

    The runtime is forgiving on purpose — a scorer that errors mid-turn drops
    back to ordinary retrieval — but a config asking for the feature with no
    scorer is a mistake, and it would otherwise present as the feature
    silently never running.
    """
    p = tmp_path / "claw.yaml"
    p.write_text(_yaml_smart(tmp_path, enabled=True, scorer=False))
    with pytest.raises(ValueError, match="smart_retrieval.enabled is true but no scorer"):
        config.load(p)


def test_smart_retrieval_loads_with_a_scorer(tmp_path):
    p = tmp_path / "claw.yaml"
    p.write_text(_yaml_smart(tmp_path, enabled=True, scorer=True))
    cfg = config.load(p)
    sr = cfg.memory_retrieval.smart_retrieval
    assert sr.enabled and cfg.systemone is not None
    # The shipped defaults are the measured operating point, not round numbers.
    assert (sr.note_chars, sr.promote_margin, sr.review_margin) == (200, 0.10, 0.30)
    # Retrieval's deadline is its own, and longer than the gate's: one batched
    # request decides what the turn is told, and a miss costs only precision,
    # while the gate must fail open in front of the LLM. Neither inherits from
    # `systemone`, which no longer carries a deadline at all.
    assert sr.timeout_s == 5.0
    assert sr.timeout_s > cfg.gate.timeout_s == 2.0
    assert not hasattr(cfg.systemone, "timeout_s")


def test_smart_retrieval_off_needs_no_scorer(tmp_path):
    p = tmp_path / "claw.yaml"
    p.write_text(_yaml_smart(tmp_path, enabled=False, scorer=False))
    assert config.load(p).memory_retrieval.smart_retrieval.enabled is False


def test_smart_retrieval_rejects_an_incoherent_pair_of_thresholds(tmp_path):
    """Retaining must never be a higher bar than promoting: that would mean a
    retrieved candidate needs a better score to stay than a rejected one needs
    to get in."""
    p = tmp_path / "claw.yaml"
    p.write_text(_yaml(tmp_path) + textwrap.dedent("""
    memory_retrieval:
      smart_retrieval:
        retain_threshold: 0.95
        promote_threshold: 0.90
    """))
    with pytest.raises(ValueError, match="retain_threshold must not exceed"):
        config.load(p)
