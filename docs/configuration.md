# Configuration reference

Every key in `claw.yaml`. The shipped `claw.example.yaml` is a runnable
template; this doc is the exhaustive reference. Schema source of truth is
the dataclasses in `claw/config.py`.

`claw.yaml` is parsed once at startup. Editing requires a process
restart — no live-reload.

## Top-level

| Key | Type | Default | Notes |
|---|---|---|---|
| `verbose` | bool | `false` | Enables DEBUG-level logging across `claw.*` loggers. Verbose; flip on when investigating. |

## `ollama:`

The `/api/chat` client used for inference and compaction.

| Key | Type | Default | Notes |
|---|---|---|---|
| `base_url` | str | required | Base URL of the Ollama HTTP server, e.g. `http://127.0.0.1:11434`. |
| `default_compaction_model` | str | required | Fallback model used for background compaction and memory_flush turns when an agent doesn't override. Pick a fast non-reasoning model — these turns aren't user-visible and shouldn't compete with primary inference for slots. |
| `max_tool_turns` | int | `30` | Cap on tool round-trips per inbound message before the loop forces a stub final reply. Cron-driven research tasks can need 15–25; raise per deployment. |
| `num_predict` | int \| null | `8192` | Per-call generation cap (Ollama `options.num_predict`). When the model hits this, claw inspects the partial: if the content has an unclosed Markdown code fence (odd count of ` ``` ` markers), the partial is discarded and `[claw: output truncated mid-code-fence; partial discarded; retry with smaller scope]` is returned to the caller; otherwise claw injects the partial back as an assistant turn plus a recovery system note and re-calls `/api/chat` once so the model can wrap up, restart with tighter scope, or bail. `null` or `-1` disables (unbounded — original behavior; runaway reasoning traces can hang the call until `request_timeout_s` fires). Override per agent / per persona below. |
| `request_timeout_s` | float | `1800.0` | Per-call httpx read timeout for `/api/chat`. Should comfortably exceed the worst-case wall time for one `num_predict`-bounded generation on the slowest agent's model. |

## `memory_retrieval:`

In-process memory retrieval. The whole block is optional — omit to use defaults.

| Key | Type | Default | Notes |
|---|---|---|---|
| `top_n` | int | `5` | Max chunks injected into the per-turn prompt. Above ~10 the prompt grows fast and retrieval stops adding signal. |
| `compact` | bool | `true` | Collapse the retrieval block to a single header + bullets instead of one block per chunk. Reduces token overhead. |

## `searxng:`

Backend for the `web_search` built-in tool.

| Key | Type | Default | Notes |
|---|---|---|---|
| `base_url` | str | required | URL of a SearXNG instance. The `web_search` tool calls `<base_url>/search?q=...&format=json`. |

## `cron:`

In-process APScheduler that fires `InboundMessage(channel="cron", ...)` on schedule.

| Key | Type | Default | Notes |
|---|---|---|---|
| `enabled` | bool | required | Master switch. `false` skips the scheduler entirely. |
| `jobs_file` | path | required | Path to the JSON file holding job definitions. See "jobs.json schema" below. |
| `max_instances_per_job` | int | `1` | Per-job concurrency cap. With `1`, a still-running job blocks its own next firing. |
| `exposed_to` | list[str] | `[]` | Agent ids granted the `cron_add` / `cron_list` / `cron_remove` tools. Empty disables agent-driven scheduling (jobs in `jobs_file` still fire). |
| `default_deliver_to` | str \| null | `null` | Fallback delivery target (e.g. an MXID) when `cron_add` is called without an explicit `deliver_to`. `null` requires the agent to specify per call. |

### `jobs.json` schema

Flat list of job entries. Two kinds:

```json
[
  {
    "name": "morning-heartbeat",
    "agent": "morning-bot",
    "kind": "cron",
    "cron": "0 8 * * *",
    "tz": "America/New_York",
    "message": "Run your morning heartbeat...",
    "enabled": true
  },
  {
    "name": "calendar-reminder-2026-05-15",
    "agent": "morning-bot",
    "kind": "at",
    "run_date": "2026-05-15T14:30:00-04:00",
    "message": "Reminder: ...",
    "enabled": true,
    "deleteAfterRun": true
  }
]
```

| Field | Required | Notes |
|---|---|---|
| `name` | yes | Human-readable id; appears in logs. |
| `agent` | yes | Agent id from `agents:` block in `claw.yaml`. |
| `kind` | yes | `cron` or `at`. |
| `enabled` | yes | `false` disables without removing the entry. |
| `message` | yes | Synthesized as the inbound message text. |
| `cron` | kind=cron | Standard 5-field cron expression. Numeric DOW follows cron convention (0=Sun..6=Sat); the scheduler normalizes to APScheduler's (0=Mon). Named days (`mon`..`sun`) pass through unchanged. |
| `tz` | optional | IANA zone for the schedule. Defaults to UTC if omitted. |
| `run_date` | kind=at | ISO 8601 datetime, e.g. `"2026-05-15T14:30:00-04:00"`. |
| `deleteAfterRun` | optional, kind=at | `true` removes the entry from disk after firing. |
| `deliver_to` | optional | Routing hint for the agent's reply. Read by the notification skill / channel; not interpreted by the scheduler itself. |

## `subagents:`

Persona-based child agents spawned via the `spawn_subagent` tool.

| Key | Type | Default | Notes |
|---|---|---|---|
| `max_concurrent` | int | required | Global cap on simultaneously-running subagents. |
| `max_children_per_agent` | int | required | Per-parent cap on concurrent children. |
| `default_model` | str | required | Model used when a spawn doesn't name a persona. |
| `personas` | dict[str, persona] | required | Named persona registry. Keys are case-insensitive (lowercased at load). |

### Per-persona keys

| Key | Type | Default | Notes |
|---|---|---|---|
| `model` | str | required | Ollama model tag for this persona. |
| `role` | str | required | One-word role label injected into the persona's system prompt context. |
| `max_spawn_depth` | int | `0` | Deepest chain this persona will root. `0` = leaf (cannot spawn). |
| `can_spawn` | list[str] \| null | `null` | Persona allowlist. `null` = any persona; `[]` = explicitly nothing; otherwise restricted set. Validated at config load against the persona registry. |
| `num_predict` | int \| null | `null` | Per-persona override of `ollama.num_predict`. `null` inherits the global. Tune by role: researchers benefit from headroom for deep reasoning (e.g. `16384`); grunts should stay tight (e.g. `4096`). |

The effective spawn budget when persona X is forked under parent Y is
`min(parent_remaining - 1, X.max_spawn_depth)`, AND the spawn must be in
the parent's `can_spawn` list (if any).

## `compaction:`

Mid-session compaction + idle recap. Whole block optional — defaults
target a 192K Ollama context.

| Key | Type | Default | Notes |
|---|---|---|---|
| `idle_recap_seconds` | int | `3600` | If the prior session's last turn is older than this on agent boot, prepend a `## Last Session Recap` row. |
| `mid_session_token_threshold` | int | `96000` | Estimated transcript tokens past which mid-session compaction fires. Default fires at ~50% of a 192K context. |
| `reserve_tokens` | int | `48000` | Newest-tokens budget preserved verbatim when compaction fires; older portion is summarized. Walk advances to the next `user`-role boundary so it doesn't slice mid-tool-call sequence. Default = 1/4 of 192K. |

Rule of thumb when scaling to a different context window: trigger ≈ 2 × reserve.

## `memory_flush:`

Flush turns that ask the agent to capture durable knowledge into
`memory/YYYY-MM-DD.md`. Three trigger paths share the same turn — see
`docs/operations.md` for when each fires. Whole block optional.

| Key | Type | Default | Notes |
|---|---|---|---|
| `enabled` | bool | `true` | Master switch. `false` disables all three trigger paths. |
| `soft_threshold_tokens` | int | `12000` | Pre-compaction trigger margin. Flush fires when transcript is within this many tokens of `compaction.mid_session_token_threshold`. Tracks the 192K tuning (was 4000 when threshold=24000). |
| `force_flush_transcript_bytes` | int | `2_097_152` | Hard fallback. If the on-disk transcript JSONL crosses this size, flush fires regardless of token estimate. Catches pathological transcripts where token estimate underreports (e.g. lots of base64 in tool results). |
| `periodic_growth_threshold` | int | `4000` | Periodic flush triggers when a session's transcript has grown by this many tokens since its last flush. With a 96K compaction trigger, 4K growth ≈ 4% — comfortable cadence. |
| `turn_timeout_s` | float | `300.0` | Per-flush deadline. On timeout the flush is dropped and compaction proceeds anyway. |

## `tz:` (top-level)

Optional, top-level (not nested). IANA zone name used for every operator-facing timestamp claw renders: the inbound-message envelope (so the model sees today's date + day-of-week on every turn), the `memory/YYYY-MM-DD.md` daily-note filename, and the `lifecycle.daily_session_rotate_hour`. Internal/persisted timestamps (transcripts, memory sync state, compaction bookkeeping) stay UTC regardless. Omit (or `null`) for UTC.

## `lifecycle:`

Whole block optional.

| Key | Type | Default | Notes |
|---|---|---|---|
| `daily_session_rotate_hour` | int \| null | `null` | Hour (0–23, in `tz`) at which every active session gets a final memory_flush + the JSONL is archived with a `.reset-<ts>` suffix + per-session caches clear. `null` disables. Memory files under `<workspace>/memory/` are NOT touched, so durable knowledge persists across rotate. |

## `agents:`

List of agent blocks. At least one required. Each block:

| Key | Type | Default | Notes |
|---|---|---|---|
| `id` | str | required | Agent identifier. Used in logs, transcript naming, `cron.exposed_to`, etc. Must be unique across the deployment. |
| `workspace` | path | required | Workspace directory. Must exist and be writable by the user running claw. |
| `primary_model` | str | required | Default Ollama model for user-reply turns. |
| `compaction_model` | str \| null | `null` | Model for background compaction + memory_flush. `null` falls back to `ollama.default_compaction_model`. |
| `max_spawn_depth` | int | `1` | Deepest subagent chain this top-level agent can root. `0` = no spawning. |
| `can_spawn` | list[str] \| null | `null` | Persona allowlist. Same semantics as persona-level `can_spawn`. |
| `num_predict` | int \| null | `null` | Per-agent override of `ollama.num_predict`. `null` inherits the global. |
| `extra_paths` | list[str] | `[]` | Workspace-relative paths injected into the system prompt every turn. Conventionally `IDENTITY.md`, `USER.md`, `SOUL.md`, `AGENTS.md`, `TOOLS.md`. Missing files silently skipped. Keep this list disjoint from indexed memory sources. |
| `matrix` | block | required | Per-agent Matrix account. See below. |

### Per-agent `matrix:`

| Key | Type | Default | Notes |
|---|---|---|---|
| `user_id` | str (MXID) | required | Bot's full Matrix ID, e.g. `@bot:matrix.example.com`. |
| `homeserver` | URL | required | Homeserver base URL. |
| `access_token_file` | path | required | One-line file with the bot's access token. Mode 0600. Mint via `POST /_matrix/client/r0/login`. |
| `device_id` | str | required | Stable device id for this bot. Conventionally uppercase. |
| `device_name` | str | required | Human-facing device label visible in Element's device list. |
| `store_path` | path | required | matrix-nio crypto store directory. Holds Olm sessions, group sessions. Conventionally `<workspace>/.matrix-store`. |
| `encryption` | bool | `true` | Enable E2E encryption. Disable only if testing against a non-encrypted room. |
| `auto_join` | `"always"` \| `"never"` | `"always"` | Whether to auto-join rooms the bot is invited to. |
| `allow_bots` | `"mentions"` \| `"all"` \| `"none"` | `"mentions"` | Filter for group-room messages. `"mentions"` = reply only when explicitly `@`-mentioned. |
| `allow_from` | list[str] (MXIDs) | `[]` | DM allowlist. The bot accepts DMs only from these MXIDs. Group rooms ignore this list. |
| `password_file` | path \| null | `null` | One-line file with the bot account login password. Needed only for the one-time cross-signing UIA challenge on `/keys/device_signing/upload`. If unset, cross-signing is skipped — the bot still works but appears as "user verification unavailable" in Element. Mode 0600. |
| `force_cross_signing_replace` | bool | `false` | One-shot operator escape hatch: replace any existing cross-signing keys on the homeserver with freshly-generated ones. Use when migrating an account previously bootstrapped by another client. **Destructive** — invalidates prior device signatures and forces every other user to re-verify this account. Set `true` for one boot, then revert. |

## Validation

`load()` validates `can_spawn` entries against the persona registry —
unknown persona names raise `ValueError` at startup. Other shape errors
surface as standard YAML / dataclass errors with the offending field.
