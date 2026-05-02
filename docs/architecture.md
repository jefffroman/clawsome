# Architecture

Mental model and request lifecycle. Read this first — the other docs use the
vocabulary it sets.

## The three abstractions

A clawsome deployment runs **one process** managing **N agents**, each with
**one or more inbound channels**.

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
  tokens since its last flush, spawn a flush turn that asks the agent to
  append durable knowledge to today's `memory/YYYY-MM-DD.md`.
- **Pre-compaction memory_flush.** Same flush, fired from the request path
  when the transcript is within `memory_flush.soft_threshold_tokens` of
  the compaction trigger. Captures durable info before older turns get
  summarized away.
- **Mid-session compaction.** When a transcript crosses
  `compaction.mid_session_token_threshold`, the older portion is
  summarized into a single `## Pre-compaction Recap` row and atomically
  swapped in under the per-session lock.
- **Idle recap.** On agent boot, if the prior session's last turn is older
  than `compaction.idle_recap_seconds`, a `## Last Session Recap` row is
  prepended.
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

- `docs/configuration.md` — full key-by-key `claw.yaml` reference.
- `docs/operations.md` — runbook (compaction tuning, matrix bot setup,
  troubleshooting).
- `docs/extending.md` — adding a tool, skill, or channel.
