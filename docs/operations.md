# Operations

Runbook for keeping a clawsome deployment healthy. Assumes you've read
`docs/architecture.md` and have a working `claw.yaml`. Cross-references
to specific keys point at `docs/configuration.md`.

## Background tasks at a glance

| Task | Trigger | Where it runs | What it produces |
|---|---|---|---|
| Periodic memory_flush | Maintenance loop, every 5 min, per-session if grew by `memory_flush.periodic_growth_threshold` since last flush | Background asyncio task | Bullets appended to `<workspace>/memory/YYYY-MM-DD.md` |
| Pre-compaction memory_flush | Request path, when transcript within `memory_flush.soft_threshold_tokens` of compaction trigger | Background asyncio task (off the user-reply critical path) | Same as above |
| Pre-rotate memory_flush | At `lifecycle.daily_session_rotate_hour`, once per active session before wipe | Synchronous under per-session lock | Same as above |
| Mid-session compaction | Request path, when estimated transcript tokens > `compaction.mid_session_token_threshold` | Background asyncio task | Older portion of transcript collapsed to a `## Pre-compaction Recap` row |
| Idle recap | Agent boot. Older than `compaction.idle_recap_seconds` → archive + recap; younger → transcript resumes intact, no recap | Synchronous, pre-live | `## Last Session Recap` row prepended on the fresh session; prior JSONL archived `.recap-<ts>` |
| Periodic reindex | Maintenance loop, every 5 min, if memory source files' hash changed | Background asyncio task | Refreshed ChromaDB + BM25 + graph |
| Daily session rotate | At `lifecycle.daily_session_rotate_hour` | Synchronous per session | Final memory_flush, JSONL archived `.reset-<ts>`, caches cleared |

## Memory_flush in depth

A flush turn loads the full session transcript, prepends a synthetic user
message asking the agent to summarize durable knowledge into today's
memory file via the `append_file` tool, and runs the agent's compaction
model end-to-end.

The flush turn is **not persisted to the user-visible transcript** — only
the side effect (the appended bullets) survives. This means:

- The user never sees the flush happen.
- The agent's primary inference slot stays free for the next user reply.
- Quality is bounded by the compaction model. Set `agents[*].compaction_model`
  per agent if a specific model handles flush better than the default.

### Tuning thresholds

Defaults target a 192K Ollama context. Scale together when changing
context size — the relationships matter more than absolute values:

| Context | mid_session_token_threshold | reserve_tokens | soft_threshold_tokens | periodic_growth_threshold |
|---|---|---|---|---|
| 64K | 32000 | 16000 | 4000 | 1300 |
| 128K | 64000 | 32000 | 8000 | 2700 |
| 192K | 96000 | 48000 | 12000 | 4000 |

Rule of thumb: `mid_session_token_threshold ≈ 50%` of context,
`reserve_tokens ≈ 25%`, `soft_threshold_tokens ≈ trigger / 8`,
`periodic_growth_threshold ≈ trigger / 24`.

### When flushes time out

`turn_timeout_s` (default 300s) bounds each flush. On timeout the flush
is dropped and any pending compaction proceeds anyway — losing one
flush is cheaper than delaying compaction. If you see frequent timeouts:

- Reduce the model's load (smaller compaction model, fewer concurrent
  agents).
- Shorten flush prompt expectations (the prompt asks for one `append_file`
  call; if the model is doing many, the agent may need re-tuning).
- Confirm Ollama isn't wedged — `curl <ollama.base_url>/api/tags` should
  respond quickly.

## Mid-session compaction

When a transcript crosses `compaction.mid_session_token_threshold`,
clawsome:

1. Snapshots the rows under the per-session lock.
2. Walks newest-first to accumulate `reserve_tokens`, then advances to
   the next `user`-role boundary so the split doesn't slice mid-tool-call.
3. Asks the compaction model to summarize the older portion into a
   `## Pre-compaction Recap` row.
4. Atomically swaps the new transcript (`recap row + reserved tail`)
   under the session lock. Any user turns that arrived during summarize
   are preserved (the swap rechecks row count against its snapshot and
   bails on mismatch).

The recap row is stable across subsequent turns so prompt caches stay
warm.

## Daily session rotate

If `lifecycle.daily_session_rotate_hour` is set, every active session
gets:

1. A final synchronous `memory_flush` so durable knowledge lands in
   `memory/YYYY-MM-DD.md`.
2. The transcript JSONL renamed `<sid>.jsonl.reset-<ts>` (preserved on
   disk, just no longer the active transcript).
3. Per-session caches cleared (`_pending_inbound`, `_idle_recapped`, etc.).

The next inbound message starts a fresh session. Memory files under
`<workspace>/memory/` are never touched, so durable knowledge persists.

Useful as a daily reset to keep transcripts from growing indefinitely.
Continuity across the wipe comes from the pre-rotate `memory_flush`,
not from idle recap: the flush appends durable knowledge to
`memory/YYYY-MM-DD.md`, which surfaces via retrieval on the next turn.
Idle recap does *not* fire after a rotate — the JSONL is already
archived, so `maybe_idle_recap` finds no `last_ts` and bails. If you
want a recap on resume rather than a hard reset, leave
`daily_session_rotate_hour` unset and let `idle_recap_seconds` cover
the conversational gap instead.

## Memory retrieval

Per agent, `MemoryIndex` indexes:

- `<workspace>/MEMORY.md` (top-level)
- `<workspace>/memory/YYYY-MM-DD.md` (daily notes — bare-date filename only)

**Bifurcation contract.** `memory/YYYY-MM-DD-<slug>.md` is journal-only
and **not** indexed (the regex anchors on a bare date). Use the slug
form for transient reasoning notes; use the bare form for distillations
worth retrieving.

**Retrieval.** Hybrid: ChromaDB vector + BM25 + NetworkX co-occurrence
graph, fused by Reciprocal Rank Fusion. Vector floor at distance 1.50
(calibrated against MiniLM-L6-v2's distribution: topical hits cluster
around 1.41–1.44, gibberish plateaus at 1.50+).

**Reindex cadence.** Maintenance loop checks `sourcesHash` every 5 min
and reindexes if changed. Source-file edits (e.g., a memory_flush
appending to today's file) get picked up within 5 min.

**Tuning the floor.** If retrieval is too noisy or too sparse, recompute
distances against a known-good query against your actual corpus and
adjust `VECTOR_DISTANCE_MAX` in `claw/memory.py`. Keep it in source —
this is calibration, not config.

## Matrix bot first-deploy

Per agent, you need:

1. **A user account on the homeserver.** Create via your homeserver's
   admin tool (Synapse: `register_new_matrix_user`).
2. **An access token.** Mint with:
   ```
   curl -X POST <homeserver>/_matrix/client/r0/login \
     -H 'Content-Type: application/json' \
     -d '{"type":"m.login.password","identifier":{"type":"m.id.user","user":"<localpart>"},"password":"<pwd>","device_id":"<DEVICE_ID>","initial_device_display_name":"<Device Name>"}'
   ```
   Save `access_token` to a one-line file, mode 0600. Reference from
   `agents[*].matrix.access_token_file`.
3. **A password file (optional but strongly recommended).** Same
   password, one line, mode 0600. Reference from
   `agents[*].matrix.password_file`. Without this, cross-signing
   bootstrap is skipped and the bot shows "user verification
   unavailable" in Element.
4. **`store_path`** at `<workspace>/.matrix-store` — directory will be
   created if missing; matrix-nio writes Olm + group session state here.

On first boot, claw POSTs master / self-signing / user-signing keys to
`/keys/device_signing/upload` (UIA challenge satisfied via
`password_file`), then writes `cross_signing.json` to the store. Any
subsequent boot detects the existing keys and skips bootstrap.

### Allowlists

- `allow_from` — DM senders. Bot accepts DMs only from these MXIDs.
  Group rooms ignore this.
- `allow_bots` — group-room behavior. `"mentions"` is the safe default:
  bot replies only when explicitly `@`-mentioned. `"all"` means it'll
  reply to any group message (chatty); `"none"` means it ignores group
  rooms entirely.

### Ghost DM cleanup

If a sender's Element shows an empty DM room with the bot that won't go
away, two layers need fixing:

1. **`m.direct` account data** on the homeserver — admin masquerade PUT
   to remove the stale entry.
2. **Element local cache** — clear cache (or sign out / sign in). Element
   X usually needs a recovery key to sign back in cleanly.

Both layers must be addressed; either alone leaves the ghost.

## Troubleshooting

### Increase log verbosity

Set `verbose: true` in `claw.yaml` and restart. Switches `claw.*`
loggers to DEBUG. Verbose; flip back off when done investigating.

### Common log signatures

| Signature | Means |
|---|---|
| `claw.main INFO ready (N agents)` | Boot complete. |
| `claw.main INFO maintenance: K periodic flush task(s) running` | K active sessions met growth threshold this tick. Silence = no qualifying sessions. |
| `claw.memory_flush INFO [<sid>] memory flush starting (reason=<r>, N rows)` | Flush turn beginning. |
| `claw.memory_flush INFO [<sid>] memory flush done (reason=<r>)` | Flush turn complete; durable bullets written. |
| `claw.memory_flush ERROR [<sid>] memory flush turn failed (reason=<r>)` | Flush threw an exception (Traceback follows). Compaction will proceed regardless. |
| `claw.ollama INFO [<label>] turn N: K tool_call(s) requested` | Tool round-trip. `<label>` shape: `<agent_id>:<kind>[:<peer_or_task>]` — e.g. `quint:main:alice` (user-facing turn from `@alice:example.org`), `quint:flush:periodic-growth:alice` (background memory flush of that user's session), `quint:subagent:chop-chop-a1b2c3d4` (subagent one-shot, parent's id + kind + the spawned task_id). At DEBUG verbosity an additional `:<sid>` correlation handle is appended for the matrix call sites (subagent labels stay as-is — the task_id is already a stable correlation handle). If N approaches `max_tool_turns`, the model is in a tool loop. |
| `WARNING [<agent>] background flush timed out after Xs` | Flush exceeded `turn_timeout_s`. |
| `claw.main INFO firing job <name>` | Cron-driven inbound being dispatched. |

### Flush isn't firing

Check the transcript size. A periodic flush only fires if the session
has grown by `periodic_growth_threshold` tokens since its last flush.
Quiet sessions stay below the bar. Drop the threshold or wait for
activity.

### Compaction fired but the model still reports "context full"

Check `compaction.reserve_tokens` against your context window. The
preserved tail must fit alongside the system prompt + retrieval + new
turns. With a 192K context and 48K reserve, plus ~10K of identity
injects + ~10K retrieval headroom, you have ~120K for new turns —
plenty. With a 64K context and 32K reserve, you have ~10K left for new
turns, which can fill again fast.

### Bot replies to DM but not to group `@`-mentions

Confirm `allow_bots: mentions` (not `none`) and that the mention is a
proper `m.mention` event (Element's `@` autocomplete produces these;
plain text `@bot` does not).

### "user verification unavailable" in Element

`password_file` was unset on first boot, so cross-signing was skipped.
Add `password_file`, then either restart (it'll bootstrap automatically)
or set `force_cross_signing_replace: true` for one boot to force a
replace.

### Bot missing from invited room

Check `auto_join: always` is set. With `"never"`, invites must be
accepted manually via the Matrix admin API.

## Health checks

A minimal health monitor should watch for:

- `claw.main INFO ready (N agents)` after restart (confirms boot).
- Absence of `ERROR` / `Traceback` over the last hour (catches silent
  background failures).
- For each active session, `memory flush done` events appearing at
  roughly the expected cadence (catches wedged maintenance loop).
- For each cron job, the expected `firing job <name>` line on schedule
  (catches scheduler regressions).
