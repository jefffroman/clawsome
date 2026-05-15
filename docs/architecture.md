# Architecture

Mental model and request lifecycle. Read this first — the other docs use the
vocabulary it sets.

## The four abstractions

A clawsome deployment runs **one process** managing **N agents**, each with
**one or more inbound channels**. Agents may transiently fork into
**subagents** drawn from a shared registry of **personas**.

**Agent.** A persistent identity. Owns: a workspace directory, a primary
inference model, a tool registry, a memory index, a transcript per session,
and one or more channel client(s). Configured under `agents:` in `claw.yaml`.

**Workspace.** A directory on disk that holds everything an agent reads and
writes. Identity files (injected per turn), memory files (indexed), skills
(loaded as tools), per-agent state (chromadb persistence, matrix crypto
store, tool-result spool, transcripts). One workspace per agent. Path
declared by `workspace:` under each agent block.

**Channel.** An inbound surface that delivers `InboundMessage`s and accepts
outbound text. The shipped channel is Matrix (`matrix-nio[e2e]`, one
account per agent). The `Channel` protocol lives in
`claw/channel/base.py` — `start`, `send`, `shutdown`, `typing`. New
surfaces (Slack, IRC, HTTP webhook, CLI) implement that protocol and
register on the agent.

**Subagent (persona).** A *persona* is a named subagent template — model,
role label, spawn budget, optional `can_spawn` allowlist — declared once
under `subagents.personas:` in `claw.yaml` and shared across every agent
in the deployment. A *subagent* is a transient `Agent` fork instantiated
from a persona by the `subagent_spawn` tool: one-shot, no transcript
persistence, no compaction, sharing the parent's workspace, memory
index, and tool registry (minus the `subagent_*` family and `cron_*`).
Spawns are **async**: the fork's `run_one_shot(prompt)` runs as a
detached `asyncio.Task` held by the spawner registry, and the
`subagent_spawn` tool returns immediately with a `task_id` — freeing
the parent's drainer to handle other inbound while the child works.
On completion (success, failure, or cancellation) the spawner fires a
synthetic `InboundMessage` whose `(channel, peer_id)` match the
original spawn site, so the result arrives as the parent's next turn
with the original prompt + result body already in context. The
companion tools `subagent_status`, `subagent_list`, and `subagent_stop`
cover polling, roster inspection, and cancellation. Spawn budgets
shrink strictly down each chain (`min(parent_remaining - 1,
persona.max_spawn_depth)`); global `max_concurrent` and per-parent
`max_children_per_agent` cap live fan-out (counts include both pending
and running children, so concurrent spawns can't race past the cap);
`ABSOLUTE_MAX_CHAIN_DEPTH=5` is the runtime safety net. The registry
is in-memory only — gateway restart kills any in-flight subagents and
forgets completed results. The shape — a pool of named (model, role)
pairs any agent can delegate into and discard — is content-neutral;
the example `researcher / coder / grunt` triple is one defaulting,
not a fixed taxonomy.

## Request lifecycle

For every inbound message:

1. **Channel** decodes the inbound and calls `agent.handle_inbound(InboundMessage)`.
2. **Session lookup.** Each peer (Matrix room, etc.) maps to a session id
   `sid`. Per-session lock acquired so concurrent inbounds for the same
   peer serialize.
3. **Transcript load.** JSONL at `<workspace>/transcripts/<sid>.jsonl` is
   read into `rows`.
4. **System prompt assembly.** Three sources concatenated:
   - **Injected files** — every path in the agent's `extra_paths` is read
     and rendered as a `## <filename>\n\n<contents>` block.
     Conventionally `IDENTITY.md`, `USER.md`, `SOUL.md`, `AGENTS.md`,
     `TOOLS.md`. Missing files are silently skipped.
   - **Skill catalog** — for each loaded skill, a one-line entry with the
     skill name + description. Full SKILL.md contents are not injected.
   - **Retrieved memory** — `MemoryIndex.retrieve_markdown(query=last_user_text)`
     returns top-k matches from MEMORY.md + dated daily notes; result is
     wrapped in `<retrieved_memory>...</retrieved_memory>`.
5. **Ollama tool loop.** `OllamaClient.run_turn` posts to `/api/chat` with
   the system prompt + history. If the response has `tool_calls`, execute
   each, append `role: tool` results, call again. Loop bounded by
   `ollama.max_tool_turns`.
6. **Persist & reply.** Final assistant text goes to the transcript and to
   `channel.send(peer_id, text)`.

Steps 1–4 happen on the user-reply critical path. Steps 5–6 dominate
latency (Ollama generation). Background tasks (compaction, memory_flush,
reindex) run off this path.

## Control plane (in-band admin commands)

A configured prefix (default `%`) turns a Matrix message into an
operator command instead of a conversational turn. Commands are
intercepted in `handle_inbound` **before** step 2 above — never
enqueued, never written to the transcript, never sent to the model. A
message is a command only if **all** hold: `commands.enabled`, the
channel is `matrix`, the sender's MXID is in `commands.allow`, and the
body parses. Any miss → it flows through as an ordinary turn with no
reply and no hint a command was attempted (an unauthorized sender can't
even probe the command set). The handler runs as a detached task so
`handle_inbound` keeps its fast return; it takes the per-session lock
itself, serializing behind any in-flight turn.

**Scope vocabulary.** A *session* is one conversation (one peer/room,
one `sid`, one transcript) and lives for days. A *turn* is one
`_process_batch` — one (coalesced) inbound → one `run_turn` → one
reply. A session has many turns. Every command *action* is bounded to a
single turn, or one subagent subtree within it; nothing is session-wide
except the read-only `%subagents` listing.

| Command | Action |
|---|---|
| `%context` | Report the next turn's starting token floor (system prompt + pending recap + transcript rows) vs. the compaction threshold. Read-only. |
| `%compact` | If a forced compaction would actually swap (transcript exceeds the reserve/keep window), run a pre-compact memory_flush then force-compact; else no-op *without* flushing. |
| `%clear` | Final memory_flush, then archive the transcript + reset per-session state (same machinery as the daily rotate). |
| `%stop [<task_id>] [--soft]` | Cancel the in-flight turn + its entire spawned cascade, suppress those subagents' completion delivery, and SIGKILL their bash trees. `<task_id>` instead cancels just that subagent + its descendant subtree (parent/siblings untouched). `--soft` skips only the bash kill. |
| `%subagents` | List this session's running subagents (discovery). Read-only. |
| `%verbose <on\|off>` | Toggle DEBUG logging process-wide at runtime (no restart). |

**Turn / cascade identity.** Each turn gets a `turn_id` that
`_process_batch` publishes via a ContextVar. `asyncio.create_task`
copies the context, so a spawned subagent — and anything *it* spawns,
at any depth — inherits the rooting turn's id. One `turn_id` therefore
identifies an entire cascade with no tree-walking, which is how `%stop`
cancels/kills exactly the stopped turn's subtree and nothing else. Bash
runs in its own process group (`start_new_session=True`) tagged with
`(turn_id, task_id)`, so one `killpg` reaps a whole tree and the kill
scopes to one turn or one subagent. There is no session scope in the
bash registry by construction.

**`%stop` mechanics.** Cancelling the per-session drainer raises
`CancelledError` (a `BaseException`, so `_process_batch`'s
`except Exception` can't swallow it) inside `run_turn`; it unwinds the
session-lock + typing context managers and the drainer ends.
`handle_inbound` lazily recreates the drainer on the next inbound.
Cancellation lands at the next `await` (Ollama HTTP / tool I/O) —
prompt for I/O-bound turns; a turn wedged in a non-`await`ing section
can only be ended by restarting the process. The triggering user
message was persisted before `run_turn`, so a cancelled turn would
otherwise leave an orphaned unanswered instruction the next turn
re-attempts; `%stop` appends a synthetic user-role marker recording the
operator cancellation so the agent treats it as abandoned rather than
looping. Subagent completions normally re-enter the session as a
synthetic inbound; `%stop` suppresses that for the cancelled cascade so
a zombie can't resurrect a stopped conversation.

## Workspace contract

| Path | Role | Read by |
|---|---|---|
| `IDENTITY.md`, `USER.md`, `SOUL.md`, `AGENTS.md`, `TOOLS.md` | Identity / role / operating instructions. Injected verbatim into the system prompt every turn via `extra_paths`. | Per turn |
| `MEMORY.md` | Top-level durable knowledge. Indexed by the memory system. | On reindex |
| `memory/YYYY-MM-DD.md` | Daily memory notes. Indexed (filename matches `^\d{4}-\d{2}-\d{2}\.md$`). | On reindex |
| `memory/YYYY-MM-DD-<slug>.md` | Journal-only daily notes. **Not** indexed (anchored regex requires bare date). | (never) |
| `skills/<name>/SKILL.md` | Skill description + usage. Catalog entry injected per turn. | Per turn (catalog) |
| `skills/<name>/tool.py` | Optional tool implementation. `TOOL_SPEC` + `async def run(input)`. | At skill load |
| `transcripts/<sid>.jsonl` | OpenAI-flat message log per session. | Per turn (load) |
| `.memory/chroma_db/` | ChromaDB persistence (vector + metadata). | On reindex / retrieve |
| `.memory/bm25_corpus.json` | Lexical index. | On reindex / retrieve |
| `.memory/memory_graph.json` | NetworkX co-occurrence graph for RRF fusion. | On reindex / retrieve |
| `.memory/sync_state.json` | Source files hash + last reindex marker. | On boot / reindex |
| `.matrix-store/` | matrix-nio crypto store (Olm sessions, group sessions). | Continuously |
| `.matrix-store/cross_signing.json` | Master / SSK / USK seeds. Mode 0600. | On boot |
| `.tool-results/<sid>/<call_id>.txt` | Spool for tool results > 8 KB. The model sees full results in-loop; transcripts get a truncated preview. | Per turn (when oversized) |
| `.initial_prompt.md` | One-shot opening turn dispatched at boot. Removed on success. | On boot |

The **disjointness rule** for top-level files: a `.md` file is either
*injected* (listed in `extra_paths`) or *indexed* (matches the indexed-set
patterns), never both — otherwise retrieval slots get spent on content the
model already sees verbatim.

## Background tasks

Run off the user-reply critical path so latency stays bounded.

- **Periodic memory_flush.** Maintenance loop fires every 5 min; for each
  active session that has grown by `memory_flush.periodic_growth_threshold`
  tokens since its last flush, spawn a flush turn over *the rows added
  since that last flush*, asking the agent to append durable knowledge
  to today's `memory/YYYY-MM-DD.md`. At most one flush is in flight per
  session; concurrent triggers skip rather than queue.
- **Pre-compact memory_flush + mid-session compaction.** When a
  transcript crosses `compaction.mid_session_token_threshold`, the
  request path spawns a single background task that runs the flush
  (incremental, locked the same way as the periodic path) and then the
  mid-session compaction in sequence. Both run on the compaction model
  concurrently with the user's main reply. Compaction summarizes the
  older portion into a single `## Pre-compaction Recap` row and
  atomically swaps it in under the per-session lock — preserving rows
  appended during the work.
- **Idle recap.** On agent boot, the prior session's last-turn age
  decides resume vs. recap. *Younger* than
  `compaction.idle_recap_seconds` → the existing transcript stays loaded
  and the agent picks up mid-conversation (no recap, no archive).
  *Older* → the JSONL is archived (`.recap-<ts>`) and a fresh session
  opens prefixed with a `## Last Session Recap` system block summarizing
  the prior turns.
- **Periodic reindex.** Maintenance loop reindexes each agent's memory
  source files if their hash changed since last reindex.
- **Daily session rotate.** At `lifecycle.daily_session_rotate_hour`
  (local time), every active session gets a final memory_flush, the
  JSONL is archived with a `.reset-<ts>` suffix, and per-session caches
  clear. Next inbound starts fresh.
- **Cron / scheduled.** `triggers/scheduler.py` reads `cron.jobs_file` at
  boot; jobs synthesize `InboundMessage(channel="cron", ...)` and reach
  the same `handle_inbound` path as channel messages.

## Where state lives

- **Live config:** wherever `--config` points (e.g.,
  `~/.claw/claw.yaml`). Not in this repo.
- **Workspaces:** per-agent dirs declared by `workspace:` in `claw.yaml`.
- **Cron jobs file:** `cron.jobs_file` path in `claw.yaml`.
- **Credentials:** files referenced by `*_file:` keys in the agent's
  `matrix:` block (mode 0600). Tokens and bot passwords. Not in any repo.

The clawsome source (`claw/`) is stateless — no read/write of paths
outside what `claw.yaml` declares.

## Further reading

- `docs/setup.md` — prerequisites (Ollama, SearXNG, Matrix homeserver),
  Python install, first-boot checklist.
- `docs/configuration.md` — full key-by-key `claw.yaml` reference.
- `docs/operations.md` — runbook (compaction tuning, matrix bot setup,
  troubleshooting).
- `docs/extending.md` — adding a tool, skill, or channel.
