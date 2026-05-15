from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    default_compaction_model: str
    # Per-inbound tool-use loop ceiling. The model can request tools, get
    # results back, and repeat — this caps the round-trips before claw
    # forces a stub final reply. 30 is comfortable for cron-driven research
    # tasks (web_search × N + reads + write_file + summarize); raise per
    # deployment if needed.
    max_tool_turns: int = 30
    # Per-/api/chat-call generation cap (Ollama options.num_predict). When
    # the model hits this, claw inspects the partial: dangerous truncation
    # (unclosed code fence) → discard partial + return structured error;
    # otherwise → re-feed partial + recovery system note and call once
    # more so the model can wrap up, restart tighter, or bail. None or -1
    # disables (unbounded — original behavior, not recommended; runaway
    # reasoning traces can hang the call until request_timeout_s fires).
    num_predict: int | None = 8192
    # httpx read timeout per /api/chat call. Should comfortably exceed the
    # worst-case wall time for one num_predict-bounded generation on the
    # slowest agent's model. Bumped 2026-05-03 from 900 to 1800 to give
    # 16K-cap turns on qwen3.5:122b (Thesa) headroom; recovery branch
    # double-budgets a length-truncated turn (initial + recovery), each
    # call independently bounded by this timeout.
    request_timeout_s: float = 1800.0


@dataclass(frozen=True)
class MemoryRetrievalConfig:
    top_n: int = 5
    compact: bool = True


@dataclass(frozen=True)
class SearxngConfig:
    base_url: str


@dataclass(frozen=True)
class CronConfig:
    enabled: bool
    jobs_file: Path
    max_instances_per_job: int = 1
    # Agent ids that get cron_add / cron_list / cron_remove. Empty = no agent
    # has scheduling tools (cron jobs in jobs.json still fire, but no agent
    # can mutate them via tool calls). Per-deployment policy: gate by
    # responsibility, not name in code.
    exposed_to: tuple[str, ...] = ()
    # Fallback delivery target when cron_add is called without an explicit
    # deliver_to. Empty/None = require agent to specify per call. Used to
    # hardcode a deployment-specific phone MXID; pulled into config so the
    # source stays deployment-neutral.
    default_deliver_to: str | None = None


@dataclass(frozen=True)
class PersonaConfig:
    model: str
    role: str
    # Max chain depth this persona is willing to root. 0 = leaf (cannot spawn);
    # 1 = can spawn one level under itself; etc. Effective budget when this
    # persona is forked is min(parent_remaining - 1, this value).
    max_spawn_depth: int = 0
    # Persona names this persona is allowed to spawn. None = no restriction
    # (subject to max_spawn_depth). Empty tuple = explicitly nothing (also
    # subject to max_spawn_depth). Validated at config load against the
    # persona registry.
    can_spawn: tuple[str, ...] | None = None
    # Per-persona override of OllamaConfig.num_predict. None inherits global.
    num_predict: int | None = None
    # Per-persona override of OllamaConfig.max_tool_turns. None inherits global.
    # Tune by role: researchers (web_search + reads) need headroom; grunts
    # should stay tight; coders sit in the middle.
    max_tool_turns: int | None = None


@dataclass(frozen=True)
class SubagentsConfig:
    max_concurrent: int
    max_children_per_agent: int
    default_model: str
    personas: dict[str, PersonaConfig]


@dataclass(frozen=True)
class CompactionConfig:
    idle_recap_seconds: int = 3600
    # Mid-session compaction fires when estimated transcript tokens exceed
    # this. Defaults are tuned for a 192K context window — set so compaction
    # triggers around 50% of context, leaving generation headroom and a
    # comfortable reserve.
    mid_session_token_threshold: int = 96000
    # When compaction fires, the newest ``reserve_tokens`` worth of rows are
    # preserved verbatim; the older portion is summarized into a single
    # recap turn. The split walks newest-first to accumulate this budget
    # then advances to the next ``user`` boundary so it doesn't slice mid
    # tool-call sequence. Default = 1/4 of a 192K window.
    reserve_tokens: int = 48000


@dataclass(frozen=True)
class MemoryFlushConfig:
    enabled: bool = True
    # Periodic flush triggers from the maintenance loop (paired with reindex)
    # whenever a session's transcript has grown by this many tokens since its
    # last periodic flush. Lets long-running sessions capture durable info
    # regularly instead of waiting for compaction to be imminent. With a
    # 96K compaction trigger, 4K of growth ≈ 4% — a comfortable cadence.
    periodic_growth_threshold: int = 4000
    # Per-flush deadline. The flush runs as a background task off the user's
    # critical path, but a wedged Ollama generation can otherwise hold it for
    # the full httpx 900s × tool-loop iterations. On timeout we drop the flush
    # and let compaction proceed anyway — losing one flush is cheaper than
    # delaying compaction.
    turn_timeout_s: float = 300.0


@dataclass(frozen=True)
class LifecycleConfig:
    # Hour (0-23, local time) at which to wipe in-flight session transcripts
    # for every agent and start fresh. None disables. Memory_flush runs once
    # per session before the wipe so durable knowledge lands in
    # memory/YYYY-MM-DD.md first; the JSONL is then archived with a `.reset-*`
    # suffix and the next inbound message starts a clean session. Useful as
    # a daily reset so transcripts don't grow indefinitely.
    daily_session_rotate_hour: int | None = None


@dataclass(frozen=True)
class CommandsConfig:
    # In-band admin commands parsed out of Matrix message bodies before they
    # reach the LLM. A message is only treated as a command when enabled, the
    # channel is matrix, the sender is in `allow`, and the body starts with
    # `prefix`. Anything else (incl. an unauthorized sender's prefixed text)
    # flows to the LLM as ordinary text — no command, no reply, no indication.
    # Defaults true, but `allow` is still fail-closed (empty = nobody), so a
    # deployment that never sets `allow` has no usable commands regardless.
    enabled: bool = True
    # Sigil that prefixes a command word. "%" is safe: Matrix clients don't
    # intercept it (unlike "/"), it isn't Markdown, and it's improbable as a
    # natural first character. Configurable per deployment.
    prefix: str = "%"
    # MXIDs allowed to run commands. Empty tuple = nobody (fail closed).
    # Deliberately separate from per-agent matrix.allow_from so a sender who
    # may DM an agent does not automatically gain control-plane access.
    allow: tuple[str, ...] = ()


@dataclass(frozen=True)
class MatrixAccountConfig:
    user_id: str
    homeserver: str
    access_token_file: Path
    device_id: str
    device_name: str
    store_path: Path
    encryption: bool = True
    auto_join: Literal["always", "never"] = "always"
    allow_bots: Literal["mentions", "all", "none"] = "mentions"
    allow_from: tuple[str, ...] = ()
    # Password file is needed to satisfy the UIA challenge on
    # /keys/device_signing/upload during cross-signing bootstrap. If unset,
    # cross-signing is skipped — the bot still works but appears as
    # "user verification unavailable" in Element. Mode 0600.
    password_file: Path | None = None
    # One-shot migration flag: replace any existing cross-signing keys on
    # the homeserver with freshly-generated ones. Use this when migrating
    # an account that was previously bootstrapped by another client and we
    # don't have access to the original SSK private key. *Destructive* —
    # invalidates prior device signatures and forces every other user to
    # re-verify this account in their Element client. Set true once for
    # the migration boot, then remove from config (or set false) on
    # subsequent runs.
    force_cross_signing_replace: bool = False


@dataclass(frozen=True)
class AgentConfig:
    id: str
    workspace: Path
    primary_model: str
    matrix: MatrixAccountConfig
    compaction_model: str | None = None
    extra_paths: tuple[str, ...] = ()
    # Max chain depth this top-level agent is willing to root. 0 = no spawning.
    # Subagent forks inherit a budget capped by this value (and further by the
    # persona's own max_spawn_depth).
    max_spawn_depth: int = 1
    # Persona allowlist. None = any persona (the default for top-level agents).
    # Tuple = restricted set. Same semantics as PersonaConfig.can_spawn.
    can_spawn: tuple[str, ...] | None = None
    # Per-agent override of OllamaConfig.num_predict. None inherits global.
    num_predict: int | None = None
    # Per-agent override of OllamaConfig.max_tool_turns. None inherits global.
    max_tool_turns: int | None = None


@dataclass(frozen=True)
class Config:
    verbose: bool
    ollama: OllamaConfig
    memory_retrieval: MemoryRetrievalConfig
    searxng: SearxngConfig
    cron: CronConfig
    subagents: SubagentsConfig
    compaction: CompactionConfig
    memory_flush: MemoryFlushConfig
    lifecycle: LifecycleConfig
    commands: CommandsConfig
    agents: tuple[AgentConfig, ...]
    # IANA timezone name (e.g. "America/New_York", "Europe/Berlin") used for
    # every operator-facing timestamp claw renders: inbound-message envelope,
    # daily-note filename `memory/YYYY-MM-DD.md`, daily-session-rotate hour.
    # ``None`` falls back to UTC. Internal/persisted timestamps (transcripts,
    # memory sync state, compaction bookkeeping) stay in UTC regardless —
    # this knob only affects what humans (and the model) see.
    tz: str | None = None


def load(path: Path | str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = _parse(raw)
    _validate_can_spawn(cfg)
    return cfg


def _parse_can_spawn(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(s.lower() for s in value)


def _parse_persona(spec: dict[str, Any]) -> "PersonaConfig":
    return PersonaConfig(
        model=spec["model"],
        role=spec["role"],
        max_spawn_depth=spec.get("max_spawn_depth", 0),
        can_spawn=_parse_can_spawn(spec.get("can_spawn")),
        num_predict=spec.get("num_predict"),
        max_tool_turns=spec.get("max_tool_turns"),
    )


def _validate_can_spawn(cfg: "Config") -> None:
    """Each can_spawn entry must reference an actual persona."""
    persona_names = set(cfg.subagents.personas.keys())
    for ac in cfg.agents:
        if ac.can_spawn is not None:
            unknown = set(ac.can_spawn) - persona_names
            if unknown:
                raise ValueError(
                    f"agent {ac.id!r} can_spawn references unknown persona(s): "
                    f"{sorted(unknown)}; available: {sorted(persona_names)}"
                )
    for name, pc in cfg.subagents.personas.items():
        if pc.can_spawn is not None:
            unknown = set(pc.can_spawn) - persona_names
            if unknown:
                raise ValueError(
                    f"persona {name!r} can_spawn references unknown persona(s): "
                    f"{sorted(unknown)}; available: {sorted(persona_names)}"
                )


def _parse_commands(d: dict[str, Any]) -> CommandsConfig:
    return CommandsConfig(
        enabled=d.get("enabled", True),
        prefix=d.get("prefix", "%"),
        allow=tuple(d.get("allow", [])),
    )


def _parse(d: dict[str, Any]) -> Config:
    return Config(
        verbose=d.get("verbose", False),
        ollama=OllamaConfig(**d["ollama"]),
        memory_retrieval=MemoryRetrievalConfig(**d.get("memory_retrieval", {})),
        searxng=SearxngConfig(**d["searxng"]),
        cron=CronConfig(
            enabled=d["cron"]["enabled"],
            jobs_file=Path(d["cron"]["jobs_file"]),
            max_instances_per_job=d["cron"].get("max_instances_per_job", 1),
            exposed_to=tuple(d["cron"].get("exposed_to", ())),
            default_deliver_to=d["cron"].get("default_deliver_to"),
        ),
        subagents=SubagentsConfig(
            max_concurrent=d["subagents"]["max_concurrent"],
            max_children_per_agent=d["subagents"]["max_children_per_agent"],
            default_model=d["subagents"]["default_model"],
            personas={
                name.lower(): _parse_persona(spec)
                for name, spec in d["subagents"]["personas"].items()
            },
        ),
        compaction=CompactionConfig(**d.get("compaction", {})),
        memory_flush=MemoryFlushConfig(**d.get("memory_flush", {})),
        lifecycle=LifecycleConfig(**d.get("lifecycle", {})),
        commands=_parse_commands(d.get("commands", {})),
        agents=tuple(_parse_agent(a) for a in d["agents"]),
        tz=d.get("tz"),
    )


def _parse_agent(d: dict[str, Any]) -> AgentConfig:
    m = d["matrix"]
    return AgentConfig(
        id=d["id"],
        workspace=Path(d["workspace"]),
        primary_model=d["primary_model"],
        compaction_model=d.get("compaction_model"),
        extra_paths=tuple(d.get("extra_paths", [])),
        max_spawn_depth=d.get("max_spawn_depth", 1),
        can_spawn=_parse_can_spawn(d.get("can_spawn")),
        num_predict=d.get("num_predict"),
        max_tool_turns=d.get("max_tool_turns"),
        matrix=MatrixAccountConfig(
            user_id=m["user_id"],
            homeserver=m["homeserver"],
            access_token_file=Path(m["access_token_file"]),
            device_id=m["device_id"],
            device_name=m["device_name"],
            store_path=Path(m["store_path"]),
            encryption=m.get("encryption", True),
            auto_join=m.get("auto_join", "always"),
            allow_bots=m.get("allow_bots", "mentions"),
            allow_from=tuple(m.get("allow_from", [])),
            password_file=Path(m["password_file"]) if m.get("password_file") else None,
            force_cross_signing_replace=bool(m.get("force_cross_signing_replace", False)),
        ),
    )
