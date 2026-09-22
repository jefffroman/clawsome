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
| `candidates` | int | `20` | Search results (fused vector + keyword ranking) that get the relevance check. Must be ≥ `top_n`. |
| `common_word_max_share` | float | `0.10` | A word in more than this share of the store's memories is ignored by keyword search. Derived from the store on every reindex (stores under 20 memories treat no word as common). |
| `relevance.threshold` | float | `0.45` | Minimum relevance score (0–1) for a candidate to be injected. `0` disables the gate: a plain `top_n` slice. |
| `relevance.offset` / `distance` / `keyword` / `keyword_share` / `traversal` | float | `2.40` / `-2.99` / `0.64` / `0.24` / `0.0` | Weights of the relevance score, `sigmoid(offset + distance·d + keyword·log(1+kw) + keyword_share·kw/kw_best + traversal·g)`, where `g` is 1 for a candidate the graph reached. Calibrated defaults — see `docs/operations.md` *Memory retrieval* before changing. `traversal` defaults to 0: being reachable is not by itself evidence. |
| `calibration.best_distance` / `tolerance` | float | `0.973` / `0.25` | Where the weights were calibrated. A WARNING is logged when recent messages' median best vector distance strays further than `tolerance` from `best_distance`. |
| `graph.enabled` | bool | `true` | Expand the candidate pool one hop along the knowledge graph. `false` removes the step. ⚠ With the defaults (`relevance.traversal: 0.0`, smart retrieval off) this knob is **observably inert** — see `docs/operations.md` *When the graph changes anything*. |
| `graph.seeds` | int | `10` | How many of the fused candidates to expand from, best first. |
| `graph.max_expand` | int | `20` | Hard cap on chunks added per query, whatever the graph's shape. |
| `smart_retrieval.enabled` | bool | `false` | Ask the decision scorer about the candidates *near* `relevance.threshold`, where the formula is guessing. **Requires a `systemone` block** — enabling it without one is a load-time error, not a silent no-op. |
| `smart_retrieval.promote_margin` / `review_margin` | float | `0.10` / `0.30` | How far below / above `threshold` the asked band reaches. Both are **relative to `threshold`**, so tuning it moves the band. |
| `smart_retrieval.promote_threshold` / `retain_threshold` | float | `0.905` / `0.107` | P(relevant) a candidate needs to come **in** from below, and to **survive** from above. `retain_threshold` may not exceed `promote_threshold` — a retrieved candidate needing a higher score to stay than a rejected one needs to enter is not a policy anyone means to write. |
| `smart_retrieval.graph_promote_threshold` | float | `0.10` | Own bar for graph-reached candidates, which score low across the board because their relevance is usually indirect. |
| `smart_retrieval.note_chars` | int | `200` | How much of a note the scorer reads. ⚠ **One calibration with the two thresholds** — clipping moves the whole score distribution. Never change it alone. |
| `smart_retrieval.context_turns` | int | `6` | Prior messages the scorer reads with the new one. |
| `smart_retrieval.timeout_s` | float | `5.0` | Retrieval's **own** scorer deadline, independent of `gate.timeout_s`. It asks about every borderline candidate in one batched request and a miss costs precision rather than memory, so it can afford to wait; measured, a 2 s bound truncated the tail and cost ~5 points on every retrieval metric. Must be > 0. |

Partially overriding `relevance:` keeps the defaults for the keys you leave out, which mixes one fitted model with another. Set all of them or none.

The explicit `memory_search` tool is not gated, and is not expanded either: an agent asking for a search gets what the two ranked legs found.

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

## `bluetooth:`

Agent-facing control of the host's Bluetooth radio, via
[`blueutil`](https://github.com/toy/blueutil). **macOS only**, and optional:
omit the whole block and no Bluetooth tools are built.

```yaml
bluetooth:
  enabled: true
  binary: /opt/homebrew/bin/blueutil
  exposed_to: [agent-1]
  verbs: [list, scan, pair, connect, disconnect]
  max_scan_seconds: 20
  timeout_s: 90
```

| Key | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch. `false` builds no tools regardless of the rest. |
| `binary` | path | `/opt/homebrew/bin/blueutil` | Path to `blueutil`. |
| `exposed_to` | list[str] | `[]` | Agent ids granted the family. Empty means nobody. |
| `verbs` | list[str] | `[list, scan]` | Which verbs exist. Any of `list`, `scan`, `pair`, `unpair`, `connect`, `disconnect`, `power`. An unknown verb **fails the config load**. |
| `max_scan_seconds` | int | `20` | Ceiling on one scan. The radio is occupied for its whole duration. |
| `timeout_s` | float | `90` | Wall-clock ceiling on one `blueutil` invocation. |

**Two gates, because they answer different questions.** `exposed_to` says *who*
may drive the radio; `verbs` says *what* may be done to it. A verb that is not
listed is never built into a `Tool`, so the model is not shown a capability it
would only be refused — and a verb can be taken away by editing config, without
touching code. The riskiest two are `power` (a controller powered off on a
headless host can only be turned back on by this same tool) and `unpair`
(undoing it needs physical access to put the device back into pairing mode);
both are off by default.

Tools built, one per enabled verb: `bluetooth_list`, `bluetooth_scan`,
`bluetooth_pair`, `bluetooth_unpair`, `bluetooth_connect`,
`bluetooth_disconnect`, `bluetooth_power`. The family is stripped on subagent
spawn, like `cron_*`.

> Host requirements — a logged-in console session, and a TCC grant pinned to
> the binary — are in [Audio](audio.md#bluetooth-host-requirements). Both bite
> before anything works.

## `music:`

Playback of a local music library through a long-lived, idle
[mpv](https://mpv.io) held open on a JSON IPC socket. Optional: omit the whole
block and no music tools are built. Requires `mpv` >= 0.38 on the host, and
`ffmpeg` for ingest.

```yaml
music:
  enabled: true
  exposed_to: [agent-1]
  library_root: /path/to/music          # Artist/Album/NN - Title.ext
  db_path: /path/to/state/library.db
  mpv_binary: /usr/local/bin/mpv
  mpv_socket: /path/to/run/mpv.sock
  switchaudio_binary: /usr/local/bin/SwitchAudioSource
  connect_timeout_s: 15
  loudness:
    normalize: true
    target_lufs: -13            # your collection's mode — the music CLI's `stats`
    boost_ceiling_dbtp: 0
    assumed_lufs: -8
    furniture_below_album_lu: 8
    furniture_max_s: 120
  candidates:
    fresh_hours: 24
    exclude_genres: [Speech]
  outputs:
    - id: room-a
      name: Room A
      mpv_device: coreaudio/AA-BB-CC-DD-EE-FF:output
      coreaudio_name: Some BT Adapter
      bluetooth_address: aa-bb-cc-dd-ee-ff
      default: true
    - id: room-b
      name: Room B
      mpv_device: coreaudio/BuiltInSpeakerDevice
```

| Key | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch. |
| `exposed_to` | list[str] | `[]` | Agent ids granted the family. Empty means nobody. |
| `library_root` | path | — | Root of the tree. Artist and album come from the directory layout. |
| `db_path` | path | — | The catalogue. See the warning below about where to put it. |
| `mpv_socket` | path | — | The IPC socket mpv holds. The gateway connects; it never starts mpv. |
| `switchaudio_binary` | path | — | Only used to restore the machine-wide default output after a Bluetooth connect steals it. |
| `connect_timeout_s` | float | `15` | How long to wait for a Bluetooth output to appear after asking for it. Past this the tool refuses. |
| `loudness` | map | see below | How playback level is set. Every value in it is a property of *your* collection. |
| `candidates` | map | see below | What `music_candidates` leaves out, and how big a pool it offers. |
| `outputs` | list | `[]` | Named speakers. At most one `default: true`; duplicate ids **fail the config load**. |

`loudness:` keys — strict, like the rest of the block:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `normalize` | bool | `false` | Off = every entry plays at unity. |
| `target_lufs` | float | `-18` | Where gain aims. Set it to your collection's **mode**; −18 (the ReplayGain 2.0 reference) is only a placeholder until you have measured. |
| `boost_ceiling_dbtp` | float | `0` | A quieter track is boosted only until its true peak would reach this. Must be ≤ 0. |
| `assumed_lufs` | float | `-8` | Loudness assumed for an unmeasured track — near your loud end, so it is cut rather than blasted. Never boosted. |
| `furniture_below_album_lu` | float | `8` | Album furniture: this far below its own record's mean… |
| `furniture_max_s` | float | `120` | …and no longer than this. |

`candidates:` keys — strict:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `fresh_hours` | float | `24` | A track queued this recently is left out of a pool. Beyond it, never-queued tracks are still offered first. |
| `exclude_genres` | list[str] | `[]` | Genres that are not music for a set (spoken word…). Whole-name, case-insensitive; asking for one explicitly overrides. |
| `pool_factor` | float | `2` | Minutes offered per minute asked for. Must be ≥ 1. |
| `default_minutes` | float | `60` | The set length when none is given. |
| `max_tracks` | int | `60` | Hard ceiling on a pool. |

Tools built: `music_play`, `music_control`, `music_status`, `music_outputs`,
`music_curate`, `music_search`, `music_candidates`, `music_history`, and
`music_dj` when `dj.enabled`. Stripped on subagent spawn, like `cron_*` and `bluetooth_*`.


> The behaviour these keys select — the DJ run sheet, the opposite contracts of
> search and candidates, output routing, and how loudness is decided — is in
> [Audio](audio.md).

## `systemone:`

Optional client for a System One decision service (typed decisions, no
generated text). Omit the block to run without one — every feature that uses
it then behaves as if the feature were off.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `base_url` | str | `http://127.0.0.1:11502` | `POST {base_url}/v1/systemone`. |

**There is no `timeout_s` here** — each caller states its own deadline at the
call site (`gate.timeout_s`, `memory_retrieval.smart_retrieval.timeout_s`),
because they sit on different paths and want different bounds. A client-level
default could never be reached anyway (a per-request timeout overrides it), so
it would only let the next caller inherit someone else's latency policy by
accident. A config still carrying the key **fails to load** rather than being
quietly ignored.

## `gate:`

The decision gate: an outer loop in front of the LLM that answers simple,
command-like human turns with a direct tool action. Optional; absent =
disabled.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch. |
| `exposed_to` | list[str] | `[]` | Agent ids that get a gate. |
| `timeout_s` | float | `2.0` | The gate's own scorer deadline. It runs before every eligible turn and a timeout means "let the LLM answer", so this is a latency guarantee to the user rather than a transport setting. Must be > 0. |
| `min_confidence` | float | `0.8` | Default bar a handler's pick must reach. |
| `context_turns` | int | `6` | Prior user/assistant messages the service sees with the new one. |
| `handlers` | list | `[]` | Declarative handlers — each one tool, fixed args, a success check and a reply form. Requires `systemone:`. |

The handler schema (text vs outcomes shape, reply forms, `fallback`,
per-handler `min_confidence`), validation rules, log lines and tuning method
are in [`docs/decisions.md`](decisions.md). Writing one of your own — whether
a request fits at all, and how to tell why one is not firing — is in
[`docs/extending.md`](extending.md#gate-handlers).

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

## `memory_curation:`

The nightly **"forgetory" curator** — a heavier, once-a-night pass that grooms
each agent's markdown memory (dedups recurring churn, marks superseded facts,
archives lapsed ephemera). Distinct from the frequent `memory_flush` *collector*;
see `docs/operations.md` for how it selects files and what it produces. Whole
block optional; **coupled to `memory_flush`** — it only runs for agents whose
`memory_flush` is enabled (nothing collected → nothing to curate).

| Key | Type | Default | Notes |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch. Off by default — opt in per deployment. |
| `hour` | int | `4` | Local-tz hour (0–23) for the nightly pass. A quiet hour keeps the big model off the reply path and sidesteps the collector/curator file race (the curator also skips today's daily note). |
| `model` | str | `qwen3.5:122b` | Model for the curator turn — deliberately bigger than the collector; supersession/dedup is relational judgment worth the cost. Per-deployment (source stays model-neutral). |
| `near_neighbor_k` | int | `8` | Near-neighbours fetched per changed memory (via the existing hybrid search) so the curator can judge dedup/supersession against them. |
| `max_files_per_run` | int | `0` | Caps candidate daily-note files per invocation (`0` = unlimited). The curator processes **one file per turn** regardless; this lets a large first bootstrap chunk across nights. |
| `max_tool_turns` | int | `80` | Tool round-trips allowed per turn — the curator reads old notes/files before judging, so it needs more latitude than a collector flush. |
| `num_predict` | int \| null | `16384` | Per-generation token cap for the curator turn (2× the global `ollama.num_predict`). A single `write_file` rewriting a whole daily note is large, and truncating it mid-file would corrupt memory. `null` inherits the global. |
| `turn_timeout_s` | float | `1800.0` | Per-curation deadline. Generous — a full nightly groom on a large model legitimately takes a while. On timeout the pass is abandoned and retried next night (markdown is left valid). |
| `superseded_archive_days` | int | `30` | The nightly whole-corpus supersession-review turn is handed the complete superseded list with each entry's age and archives ones superseded ~this many+ days ago, case-by-case. Age is measured from the *superseding* memory's `ts`. |

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
    - "@user-1:example.org"
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
| `allow_from` | list[str] (MXIDs) | `[]` | DM allowlist. The bot accepts DMs only from these MXIDs (e.g. `@user-1:localhost.localnet`). Group rooms ignore this list. **Each entry must be a real account on the homeserver** — clawsome ships no signup flow, so register human users out-of-band (Synapse: `register_new_matrix_user`) before listing them here. |
| `password_file` | path \| null | `null` | One-line file with the bot account login password. Needed only for the one-time cross-signing UIA challenge on `/keys/device_signing/upload`. If unset, cross-signing is skipped — the bot still works but appears as "user verification unavailable" in Element. Mode 0600. |
| `force_cross_signing_replace` | bool | `false` | One-shot operator escape hatch: replace any existing cross-signing keys on the homeserver with freshly-generated ones. Use when migrating an account previously bootstrapped by another client. **Destructive** — invalidates prior device signatures and forces every other user to re-verify this account. Set `true` for one boot, then revert. |

## Validation

`load()` validates `can_spawn` entries against the persona registry —
unknown persona names raise `ValueError` at startup. The `gate:` and
`systemone:` blocks are validated strictly (see `docs/decisions.md`
*Validation*); gate handlers without a `systemone:` block fail the load. Other shape errors
surface as standard YAML / dataclass errors with the offending field.

An optional block written but left empty (e.g. `lifecycle:` with no
value, which YAML parses as `null`) is treated identically to omitting
it — defaults apply. The required blocks (`ollama`, `searxng`, `cron`,
`subagents`, `agents`) still error if missing or empty.
