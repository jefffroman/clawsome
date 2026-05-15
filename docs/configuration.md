# Configuration reference

Every key in `claw.yaml`. The shipped `claw.example.yaml` is a runnable
template; this doc is the exhaustive reference. Schema source of truth is
the dataclasses in `claw/config.py`.

`claw.yaml` is parsed once at startup. Editing requires a process
restart — no live-reload.

## Top-level

| Key | Type | Default | Notes |
|---|---|---|---|
| `verbose` | bool | `false` | Enables DEBUG-level logging across `claw.*` loggers. Also expands the per-turn tool-call log line to include the full `kind:sid` label and adds one DEBUG line per tool call with truncated JSON arguments — so flipping this on will write tool args (Matrix room ids, bash commands, file contents being written, web search queries) to `/var/log/claw.log`. The log file is mode 644 owned by `claw`. Verbose; flip on when investigating. |

## `ollama:`

The `/api/chat` client used for inference and compaction.

| Key | Type | Default | Notes |
|---|---|---|---|
| `base_url` | str | required | Base URL of the Ollama HTTP server, e.g. `http://127.0.0.1:11434`. **Must match the daemon's actual bind URL** (`OLLAMA_HOST`). Every model tag referenced elsewhere in this config (`agents[*].primary_model`, `agents[*].compaction_model`, `default_compaction_model`, `subagents.personas[*].model`, `subagents.default_model`) must already be pulled on this server (`ollama pull <tag>`). |
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
| `base_url` | str | required | URL of a SearXNG instance. The `web_search` tool calls `<base_url>/search?q=...&format=json` — **the SearXNG instance must have JSON output enabled** (`search.formats:` in `settings.yml` must include `json`; upstream defaults ship HTML-only). |

## `cron:`

In-process APScheduler that fires `InboundMessage(channel="cron", ...)` on schedule.

| Key | Type | Default | Notes |
|---|---|---|---|
| `enabled` | bool | required | Master switch. `false` skips the scheduler entirely. |
| `jobs_file` | path | required | Path to the JSON file holding job definitions. **Must be writable by the claw user if `exposed_to` is non-empty** — agents rewrite the file via `cron_add` / `cron_remove`. See "jobs.json schema" below. |
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

Persona-based child agents spawned via the `subagent_spawn` tool. Spawns
are async — the call returns a `task_id` immediately and the result is
delivered later as a synthetic completion message; companion tools
`subagent_status`, `subagent_list`, and `subagent_stop` cover polling,
roster inspection, and cancellation. See `docs/architecture.md` for the
full lifecycle.

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
| `idle_recap_seconds` | int | `3600` | Boundary between resume and recap on agent boot. Younger: the existing transcript stays loaded and the agent resumes mid-conversation. Older: the JSONL is archived (`.recap-<ts>`) and a fresh session opens with a `## Last Session Recap` system block. |
| `mid_session_token_threshold` | int | `96000` | Estimated transcript tokens past which mid-session compaction fires. Default fires at ~50% of a 192K context. |
| `reserve_tokens` | int | `48000` | Newest-tokens budget preserved verbatim when compaction fires; older portion is summarized. Walk advances to the next `user`-role boundary so it doesn't slice mid-tool-call sequence. Default = 1/4 of 192K. |

Rule of thumb when scaling to a different context window: trigger ≈ 2 × reserve.

## `memory_flush:`

Flush turns that ask the agent to capture durable knowledge into
`memory/YYYY-MM-DD.md`. Three trigger paths share the same turn — see
`docs/operations.md` for when each fires. Whole block optional.

| Key | Type | Default | Notes |
|---|---|---|---|
| `enabled` | bool | `true` | Master switch. `false` disables both trigger paths (pre-compact bg + periodic). |
| `periodic_growth_threshold` | int | `4000` | Periodic flush triggers when a session's transcript has grown by this many tokens since its last flush. With a 96K compaction trigger, 4K growth ≈ 4% — comfortable cadence. |
| `turn_timeout_s` | float | `300.0` | Per-flush deadline. On timeout the flush is dropped and compaction proceeds anyway. |

## `tz:` (top-level)

Optional, top-level (not nested). IANA zone name used for every operator-facing timestamp claw renders: the inbound-message envelope (so the model sees today's date + day-of-week on every turn), the `memory/YYYY-MM-DD.md` daily-note filename, and the `lifecycle.daily_session_rotate_hour`. Internal/persisted timestamps (transcripts, memory sync state, compaction bookkeeping) stay UTC regardless. Omit (or `null`) for UTC. **Must be a valid IANA zone name installed in the host's tzdata** (e.g., `America/New_York`, `Europe/Berlin`).

## `lifecycle:`

Whole block optional.

| Key | Type | Default | Notes |
|---|---|---|---|
| `daily_session_rotate_hour` | int \| null | `null` | Hour (0–23, in `tz`) at which every active session gets a final memory_flush + the JSONL is archived with a `.reset-<ts>` suffix + per-session caches clear. `null` disables. Memory files under `<workspace>/memory/` are NOT touched, so durable knowledge persists across rotate. |

## `commands:`

Whole block optional. In-band operator commands parsed out of Matrix
message bodies before they reach the model (see Architecture → Control
plane). Applies process-wide, not per-agent. `enabled` defaults true,
but `allow` is fail-closed (empty = nobody) — so the feature is
effectively off until you list operator MXIDs.

| Key | Type | Default | Notes |
|---|---|---|---|
| `enabled` | bool | `true` | Master switch. When `false`, a prefixed message is always ordinary text. Defaults `true`; `allow` (below) is the real gate. |
| `prefix` | str | `"%"` | Sigil that marks a command. Pick one Matrix clients don't intercept (`/` is reserved client-side) and that won't collide with the agents' natural prose. |
| `allow` | list[str] | `[]` | MXIDs permitted to run commands. **Empty = nobody (fail closed).** Deliberately separate from per-agent `matrix.allow_from`: being able to DM an agent does not grant control-plane access. A prefixed message from a non-listed sender is treated as ordinary text — no reply, no hint. |

In a group room with `allow_bots: mentions`, a command must still
`@`-mention the bot to reach the gateway at all.

```yaml
commands:
  enabled: true
  prefix: "%"
  allow:
    - "@operator:example.org"
```

## `agents:`

List of agent blocks. At least one required. Each block:

| Key | Type | Default | Notes |
|---|---|---|---|
| `id` | str | required | Agent identifier. Used in logs, transcript naming, `cron.exposed_to`, etc. Must be unique across the deployment. |
| `workspace` | path | required | Workspace directory. **Must exist and be writable by the user running claw before first boot.** Subdirs (`memory/`, `transcripts/`, `.memory/`, `.matrix-store/`, `.tool-results/`) are created on demand. |
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
| `user_id` | str (MXID) | required | Bot's full Matrix ID, e.g. `@bot:matrix.example.com`. The localpart must be a registered user on the homeserver. **The server-name portion (after `:`) must match the homeserver's `server_name`** (Synapse: `server_name:` in `homeserver.yaml`) — *not* the URL host of `homeserver` below. The two are independent values. |
| `homeserver` | URL | required | Homeserver base URL — *where to reach* the server, e.g. `http://127.0.0.1:6167` for a same-host Synapse. **Must match the homeserver's actual bind URL.** Independent from the server-name in `user_id`. |
| `access_token_file` | path | required | One-line file with the bot's access token. Mode 0600. Token must be valid for `user_id` on `homeserver`. Mint via `POST /_matrix/client/r0/login` — see `docs/operations.md` *Matrix bot first-deploy*. |
| `device_id` | str | required | Stable device id for this bot. Conventionally uppercase. **Must be unique per `user_id` for fresh crypto state** — reusing an existing device id binds to that device's existing Olm sessions on the homeserver. |
| `device_name` | str | required | Human-facing device label visible in Element's device list. |
| `store_path` | path | required | matrix-nio crypto store directory. Holds Olm sessions, group sessions. Conventionally `<workspace>/.matrix-store`. **Must persist across restarts** — losing it forces a fresh crypto handshake and invalidates ongoing E2E sessions. |
| `encryption` | bool | `true` | Enable E2E encryption. Disable only if testing against a non-encrypted room. |
| `auto_join` | `"always"` \| `"never"` | `"always"` | Whether to auto-join rooms the bot is invited to. |
| `allow_bots` | `"mentions"` \| `"all"` \| `"none"` | `"mentions"` | Filter for group-room messages. `"mentions"` = reply only when explicitly `@`-mentioned. |
| `allow_from` | list[str] (MXIDs) | `[]` | DM allowlist. The bot accepts DMs only from these MXIDs (e.g. `@alice:localhost.localnet`). Group rooms ignore this list. **Each entry must be a real account on the homeserver** — clawsome ships no signup flow, so register human users out-of-band (Synapse: `register_new_matrix_user`) before listing them here. |
| `password_file` | path \| null | `null` | One-line file with the bot account login password. Needed only for the one-time cross-signing UIA challenge on `/keys/device_signing/upload`. If unset, cross-signing is skipped — the bot still works but appears as "user verification unavailable" in Element. Mode 0600. |
| `force_cross_signing_replace` | bool | `false` | One-shot operator escape hatch: replace any existing cross-signing keys on the homeserver with freshly-generated ones. Use when migrating an account previously bootstrapped by another client. **Destructive** — invalidates prior device signatures and forces every other user to re-verify this account. Set `true` for one boot, then revert. |

## Validation

`load()` validates `can_spawn` entries against the persona registry —
unknown persona names raise `ValueError` at startup. Other shape errors
surface as standard YAML / dataclass errors with the offending field.
