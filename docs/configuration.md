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

### Two host requirements worth knowing before enabling this

**A logged-in console session.** Every call runs inside the console GUI session
as a one-shot launchd job, not in the gateway's own daemon context, because the
two are not equivalent — see [Extending](extending.md#tools-that-need-a-gui-session)
for the measurement and the reasoning. If no console session exists, the tools
say so rather than reporting a Bluetooth fault.

**A TCC grant.** On modern macOS, reaching IOBluetooth at all requires
`kTCCServiceBluetoothAlways`. Without it `blueutil` does not error — it reports
the controller as powered off and lists no devices, which is indistinguishable
from a radio that really is off. The tools disambiguate by cross-checking
`system_profiler`, which reads the IORegistry and needs neither a session nor a
grant, and say "grant, not radio" when the two disagree. Note that such a grant
is typically pinned to the binary's code signature, so upgrading `blueutil`
can silently revoke it.

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

### `music_dj` — speech between the songs

The agent picks a set (usually from `music_candidates`) and says something
between the records. The call is a **run sheet**: entries with `say`, `play` (a
`t:` track handle, or an `al:` record handle) or both, `say` spoken before what
it plays; a `say` alone at the start is an opening, at the end a sign-off.

- **Insert, never duck.** A link is rendered to a WAV and queued as its own
  playlist entry, so it *cannot* overlap music — the whole-song rule is
  structural.
- **A record is one entry.** An `al:` entry plays the whole album, in its own
  order, codas kept, at one album gain — exactly as `music_play` plays it —
  while single tracks in the same set keep per-track gain. So a `say` on it
  introduces the record, and nothing can be queued inside its running order.
  Where links belong around a record is taste, left to the agent's skill.
  Entries take handles, never free text, so an entry cannot mis-resolve.
- **All or nothing.** Every handle resolves and every clip renders before the
  first entry is queued; one failure queues nothing, because the words assert
  what plays next.
- **`minutes` is checked before rendering**, within 10%, so a set of the wrong
  length costs milliseconds rather than a synthesis.
- **Appending drops a trailing sign-off** — unless it is being spoken — so a
  goodbye never lands mid-evening. `music_status` reports what is next and what
  the queue ends with, and recognises a link as a link.
- **`target_lufs` sets the voice level; links then play at unity.** Measured on
  the finished clip in the library's format: a louder link is cut to it, a
  quieter one raised toward it only as far as `max_true_peak_dbfs` allows — the
  music's own rule, no limiter. It also evens out Piper's render-to-render
  spread. Unset, a link keeps the flat render's level (~-20 LUFS). Independent
  of `loudness.target_lufs` and of `normalize` — choose it against where your
  music actually plays, which with normalisation on is near
  `loudness.target_lufs`.
- **`drive` shapes the voice, never its level.** Saturation adds loudness as
  well as density. With `target_lufs` set, the target pins the level anyway;
  without it, each saturated link is measured against the same Piper render
  flat and the difference is cut back off (never boosted): within 0.2 LU of
  flat, with more peak headroom. A ladder of drives therefore compares only
  timbre. Piper samples noise — two renders of one line differ by up to
  ~0.7 LU — so compare on one render.

`dj:` keys: `enabled`, `piper_binary`, `voice_model`, `render_dir`,
`length_scale`, `sentence_silence`, `drive` (1.0 = off; timbre only),
`target_lufs` (null = the render's own level), `max_true_peak_dbfs`,
`pad_ms`, `sample_rate`, `channels`, `max_links`, `keep_hours`. Rendering uses
the Piper CLI, so the DJ's voice and speaking rate are independent of any live
TTS daemon; one process per run sheet (model load dominates).

### Search and candidates — opposite contracts on purpose

**`music_search` looks and never acts, and hides nothing.** It matches artist,
album, title, genre, and words in notes and moods, and returns artists, albums
and tracks. Interludes, spoken word, recently played and annotated records all
appear, labelled rather than removed; the only narrowing is what the caller
passes, and the reply restates it. A display cap, when hit, states the true
total. A miss is a miss — near misses appear only when nothing matched, marked
as not being matches. Notes and moods are attributed to the level they were
written at, so an artist-level note is one artist row, not one per track.
**`scope` returns only one kind, and never finds less:** every hit is reported
as that kind — an artist that matches comes back as their tracks or records, a
track as its record or artist — with the kind's own matches ranked first. It is
not a narrowing (genre and years are), so it cannot turn a match into a miss.

**Notes add up across levels; everything else takes the most specific.** A
track's mood overrides its album's, but a track's note is *added to* its
album's and its artist's — each level carries its own facts, and all of them
apply, lowest level first. List rows show each level's note cut to 90
characters; **passing a handle as the query** returns the detail view — every
level's own curation in full, and for an album its tracks, for an artist their
records.

**`music_candidates` filters on purpose, and counts what it filtered.** It
offers a pool about `pool_factor`× the requested length, leaving out recent
plays, interludes and `exclude_genres`, and its reply says how many of each.
The pool is taken from the top of an **interleave** — round-robin across
artists, then across each artist's records, never-queued tracks first — so any
prefix is as varied as the request allows: a broad genre yields one track each
from many artists, a single artist is spread across their records, and no
per-artist cap needs tuning. Choosing from the pool is left to the agent: the
catalogue knows what exists, how long it is, and when it last played, but not
what suits an evening.

If the set-builder were the only lookup, "not in the pool" would read as "not in
the library". Keeping the two apart — and having each say which it is — is the
point.

**`music_history`** reads the curation log — every annotation, correction,
rename and merge, with who, when and what it replaced — for one artist, album or
track (with what is inside it and the levels above it), or the most recent edits
everywhere. The log also records the operations that *discard* curation (a
repopulate, or a file gone from disk), with what was lost, so it can be put
back. There is no stored "last curated" field; the log is the record.

**Handles** carry an answer from one tool into the next exactly: `ar:<id>`,
`al:<id>`, `t:<hash of the path>`. `music_play` and `music_curate` resolve a
handle exactly and refuse an unknown one rather than matching the nearest thing;
`music_play(handles=[...])` plays a chosen set in the given order, all or
nothing.

**Outputs are a named list, and that is the point.** The agent asks for
`room-a`; it never handles a CoreAudio device string, a MAC address, or a
filesystem path. Matching happens inside the tool, over the catalogue — a real
music collection is full of apostrophes, accents and percent-encoded slashes,
and a path composed by a language model onto a command line is the worst place
to discover that.

An output with **no** `bluetooth_address` is a wired or built-in device with no
connect step. The key being absent is the signal; it is never null-for-none.

### Playback routing never moves the system default

mpv is told `audio-device` per instance, so playback cannot disturb anything
else on the host. That began as isolation and became a safety property: only
explicitly-targeted audio should reach a speaker somebody is sitting next to.

Which creates one non-obvious obligation. **macOS makes a Bluetooth audio
device the system default output the moment it connects** — so the reconnect
that `ensure_output` performs would itself re-point system audio, including
alert sounds with their own volume, at that speaker. The reconnect therefore
captures the current default before connecting and restores it afterwards.

A speaker that cannot be woken produces a refusal naming it and why. Playback
never falls back to a different output: audio arriving in a room nobody asked
about is worse than audio not arriving.

### Loudness — aim at the mode, cut freely, boost only into headroom

Three measurements, each answering one question, and it pays to keep them
apart:

| Metric | Tells you | Does not tell you |
|---|---|---|
| Integrated LUFS | how loud a track **sounds** | how much gain it can take |
| True peak | **headroom** — gain available before it clips | how loud it sounds |
| LRA | dynamics | either of the above |

In a collection that spans the loudness war the first two come apart. Measured
across one real collection of 4,500 tracks: integrated loudness spans 31.9 dB,
yet **93% of tracks peak within 3 dB of full scale** and 57% already peak above
it. The loud records are loud *by compression*. Two traps follow, and both are
easy to walk into:

- **A LUFS gap is not gain you can apply.** Gain moves loudness and peak
  together, so closing a gap *upward* needs headroom the quiet, dynamic
  records mostly do not have. Upward means a limiter, which changes the music
  rather than its level. **Downward is free.**
- **Nor does matched peak mean matched loudness.** Peak-matched masters that are
  more compressed sound louder — that is the whole mechanism of the loudness
  war. A LUFS gap between records is a level difference a listener hears.

So, with `normalize` on:

- **The target is the collection's mode**, not its mean or a broadcast
  standard. At the mode the typical record is untouched, so switching
  normalisation on does not make the room quieter. The music CLI's `stats` prints
  the measured mode beside the configured target and flags drift.
- **Louder tracks are cut to the target.**
- **Quieter tracks are boosted toward it only as far as their own true peak
  allows**, up to `boost_ceiling_dbtp`. In a collection like the one above that
  is often not far — the median boost was +0.5 dB — but together with the cuts
  it took the 5th–95th percentile spread of a shuffle from 12.8 LU to 4.6.
- **A shuffle gets per-track gain; an album in order gets one figure** for the
  whole record — energy-weighted loudness, headroom from its loudest peak — so
  its own quiet/loud relationships survive.
- An **unmeasured** track is assumed loud (`assumed_lufs`) and never boosted.

Why the ceiling defaults to 0 dBTP rather than the textbook −1 (the margin for
lossy codecs): the ceiling bounds *our boost only*. Where most of a collection
already peaks above 0 at unity, −1 protects only the boosted tracks — which
would then be the cleanest ones playing — while roughly halving how many get
any boost at all.

⚠ **mpv's `volume` is cubic, not a percentage of amplitude** — 50 is −18 dB,
not −6. Gain is therefore sent as the per-entry `volume-gain` option, which is
in dB. And mpv's `--volume-gain-max` does **not** cap a per-file option, so the
never-past-the-peak rule is enforced in claw, not by mpv. A dB-to-`volume`
conversion done as if it were linear makes every cut three times deeper than
intended; that bug once produced a listening verdict against normalisation
that had to be withdrawn.

There is still **no per-output volume**. A standing attenuation happens before
a lossy encoder and the amplifier then raises music and codec noise together.
Normalisation is not that: it brings a loud record down to where the typical
one already enters the encoder.

ReplayGain tags are the portable alternative and are not used: they must be
written into the files, and a catalogue can hold the same measurement without
modifying anything the operator owns.

**Album furniture** — a track far quieter than its own record and short
(`furniture_below_album_lu`, `furniture_max_s`). Codas, segues, spoken intros:
under 1% of a collection, quiet on purpose. Dropped from a *shuffle*, where they
are a dead half-minute; always kept in album order, where they are the joins.
Never lifted on their own.

### The catalogue

`db_path` is SQLite (WAL). Artists contain albums contain tracks, each a row
with an id, each carrying the same curatable fields. It holds the loudness
measurement, what the tags said, and **curation** — mood, energy, notes,
corrections — resolved most specific first: the track's value, else the
album's, else the artist's. A track can override its album's artist, which is
how a compilation credits who actually played.

> ⚠ **This file is state, not cache. Put it where your backups reach.**
> Everything measured could be rebuilt from the files in minutes, but curation
> exists nowhere else. Do not put it beside the IPC socket in a runtime
> directory; that placement quietly says "disposable", and it is not.

Ingest is per file; a collection sweep is a loop over the same function. A file
whose size, mtime and measure version all match is never opened, so a re-sweep
of a large library costs seconds. See
[Extending](extending.md#curation-must-outlive-what-produced-it) for the write
rules, which are the part worth copying.

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
unknown persona names raise `ValueError` at startup. Other shape errors
surface as standard YAML / dataclass errors with the offending field.

An optional block written but left empty (e.g. `lifecycle:` with no
value, which YAML parses as `null`) is treated identically to omitting
it — defaults apply. The required blocks (`ollama`, `searxng`, `cron`,
`subagents`, `agents`) still error if missing or empty.
