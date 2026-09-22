# Operations

Runbook for keeping a clawsome deployment healthy. Assumes you've read
`docs/architecture.md` and have a working `claw.yaml`. Cross-references
to specific keys point at `docs/configuration.md`.

## Background tasks at a glance

| Task | Trigger | Where it runs | What it produces |
|---|---|---|---|
| Periodic memory_flush | Maintenance loop, every 5 min, per-session if grew by `memory_flush.periodic_growth_threshold` since last flush | Background asyncio task | Bullets appended to `<workspace>/memory/YYYY-MM-DD.md` |
| Pre-compact memory_flush + mid-session compaction | Request path, when estimated transcript tokens > `compaction.mid_session_token_threshold`. A single bg task runs flush, then compaction, in sequence — flush captures durable items before older turns are summarized away. | Background asyncio task (off the user-reply critical path) | Bullets appended to `<workspace>/memory/YYYY-MM-DD.md`, then older portion of transcript collapsed to a `## Pre-compaction Recap` row |
| Pre-rotate memory_flush | At `lifecycle.daily_session_rotate_hour`, once per active session before wipe | Synchronous under per-session lock | Same as above |
| Idle recap | Agent boot. Older than `compaction.idle_recap_seconds` → archive + recap; younger → transcript resumes intact, no recap | Synchronous, pre-live | `## Last Session Recap` row prepended on the fresh session; prior JSONL archived `.recap-<ts>` |
| Periodic reindex | Maintenance loop, every 5 min, if memory source files' hash changed | Background asyncio task | Refreshed ChromaDB + BM25 + graph |
| Daily session rotate | At `lifecycle.daily_session_rotate_hour` | Synchronous per session | Final memory_flush, JSONL archived `.reset-<ts>`, caches cleared |
| Nightly curation (forgetory) | At `memory_curation.hour` (local tz), if `memory_curation.enabled` and the agent collects | Background asyncio task, one bounded turn per file | Deduped/superseded/archived markdown; ephemera moved to `memory/archive/YYYY-MM.md`; a reasoned `claw.curator` line per decision in `claw.log` |

## Memory_flush in depth

A flush turn passes the agent the *slice of rows added since the previous
successful flush for that session* — not the full transcript — followed
by a synthetic user message asking the agent to summarize durable
knowledge into today's memory file via the `append_file` tool. The
agent's compaction model runs end-to-end.

The "rows since last flush" marker is persisted per-session at
`<workspace>/transcripts/<sid>.flush.json` so the marker survives claw
restarts. Mid-session compaction shrinks the transcript via atomic
replace; the next flush detects the shrink and starts over from row 0.
At most one flush is in flight per session: background flushes
(periodic, pre-compact) skip if another is running; the synchronous
pre-rotate flush waits on the per-session flush lock.

Long tool-call argument strings (e.g., a `write_file`/`append_file` with
multi-KB content) are stubbed in the persisted assistant row at
`ollama.TOOL_CALL_ARGS_MAX_CHARS` (default 200). The in-loop `messages`
list and the `claw.ollama` DEBUG log keep the full args, so the model
can chain on its own emissions during a single `run_turn` and verbose
operators can see what was called. Stubbing in persistence prevents the
flush model from re-seeing the prior content and re-capturing it on the
next slice.

The flush turn is **not persisted to the user-visible transcript** — only
the side effect (the appended bullets) survives. This means:

- The user never sees the flush happen.
- The agent's primary inference slot stays free for the next user reply.
- Quality is bounded by the compaction model. Set `agents[*].compaction_model`
  per agent if a specific model handles flush better than the default.

### Tuning thresholds

Defaults target a 192K Ollama context. Scale together when changing
context size — the relationships matter more than absolute values:

| Context | mid_session_token_threshold | reserve_tokens | periodic_growth_threshold |
|---|---|---|---|
| 64K | 32000 | 16000 | 1300 |
| 128K | 64000 | 32000 | 2700 |
| 192K | 96000 | 48000 | 4000 |

Rule of thumb: `mid_session_token_threshold ≈ 50%` of context,
`reserve_tokens ≈ 25%`, `periodic_growth_threshold ≈ trigger / 24`.

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

**What is indexed.** A section's **heading is indexed with its body**. The
parser strips a `##` heading into metadata, so for a long time a heading's
words reached neither leg and a note titled "Bluetooth pairing" could not be
found by searching for those words. A title is the most topical line a section
has. The stored text stays body-only, so the rendered snippet does not repeat
the heading printed above it. A heading that *opens* a file — a `#` title, or a
`###` before any `##` — names its chunk too, rather than leaving the chunk
called "Intro" with the title buried in its body.

**A chunk of pure annotation is a node, not a result.** A chunk whose content
is nothing but headings and/or HTML comments — a file title, a placeholder, a
section that only introduces its children — is marked `traversal_only`: it
stays in the corpus and in the graph, and the walk passes *through* it to reach
what it connects, but it is never itself a candidate. Injecting one tells the
model a section exists and nothing about what it says, which is the same defect
as rendering graph node names with no text.

**Retrieval.** ChromaDB vector + BM25, fused by Reciprocal Rank Fusion, with
the knowledge graph expanding the result. Each turn:

1. **Search.** The keyword side searches the message *without* its envelope
   header, tokenized with punctuation stripped, and without the store's
   **common words** — any word in more than `common_word_max_share` of the
   memories, derived from the store at every reindex, no word list. The vector
   side embeds the full message. The fused top `candidates` (20) go on.
2. **Graph expansion.** One hop out of the sections of the best `graph.seeds`
   candidates adds up to `graph.max_expand` more, each marked `traversal`.
   The graph does **not** match the query — see below.
3. **Relevance gate.** Each candidate is scored from its vector distance `d`,
   keyword score `kw`, and whether the graph reached it (`g`):
   `sigmoid(offset + distance·d + keyword·log(1+kw) + keyword_share·kw/kw_best + traversal·g)`.
   Those at or above `relevance.threshold` are injected, up to `top_n`, **best
   score first**. A turn where nothing fits injects **no row at all** — not a
   "nothing found" row, which would cost tokens to report an absence and
   re-rank every turn. An explicit `memory_search` does say "No strong
   matches", because someone asked.
4. **Supersession** heads are appended (below).

**Index freshness is never reported to the agent.** Source hashes differ from
the index for the few minutes between an agent writing a memory and the next
periodic reindex, so a freshness check fires *because* the agent just saved
something — telling it its own memory is "stale" at the moment that memory is
most trustworthy, and clearing itself minutes later. The agent can take no
action on it in any case. The block carried a `**Sync:**` line, a
`WARNING: MEMORY OUT OF SYNC` banner, and an exemption that made an
otherwise-silent turn emit a block anyway; all three were removed. Index health
is an operator's question, and the `periodic reindex:` log lines answer it.

**Why the graph expands rather than searches.** It used to score graph node
*names* against the query by summed IDF and render the top few with their
neighbours. That is a second retrieval mechanism, and nothing calibrated it:
it answered "how does compaction work" with an unrelated project section and
two calendar entries, every hit matching `work`. Choosing a good anchor is
only hard because the graph was choosing one by itself; expanding from
candidates the two ranked legs already endorsed removes the problem instead of
tuning it.

Only **section** nodes nominate chunks. A concept node — a bolded phrase — has
no chunk of its own, so the only way to give it content is to walk on to the
sections carrying it; since a phrase like "Notes" is bolded throughout a store,
that walk nominated a median 48 of 443 chunks per message on one store against
1–6 for sections alone. Concepts remain as connective tissue the walk passes
through.

`relevance.traversal` defaults to **0.0**: being reachable is not by itself
evidence, so a graph-reached candidate must stand on its content until a fit
against graded data earns it otherwise. `graph.enabled: false` removes the
step.

**When the graph changes anything.** With the shipped defaults it does not.
Expansion still runs and still adds candidates, but at `traversal: 0.0` a
graph-reached chunk is judged on its content alone — and an expanded chunk is
precisely one that neither ranked leg scored highly, so it rarely clears the
threshold. Measured on one store it injected **zero**. So `graph.enabled: true`
and `false` produce the same injected set, and the knob reads as live while
being observably inert.

It becomes live under either of two conditions, and it is worth knowing which
one you are relying on:

| condition | effect |
|---|---|
| `relevance.traversal` given a positive **weight** | reachability becomes evidence; measured 6 graph memories injected at `1.0`, 80 at `3.0` |
| `smart_retrieval.enabled: true` | graph-reached candidates are **always asked**, whatever they scored, so the scorer can admit them on content the formula cannot judge |

⚠ **`traversal` is a regression coefficient, not a hop count.** The feature it
multiplies is binary — 1.0 for any chunk the graph reached, 0.0 for one the two
ranked legs found — so the weight lives in log-odds beside `distance` and
`keyword`, and is best read as a *discount on the bar* for graph-reached
chunks. Against the default `threshold: 0.45`:

| `traversal` | a graph-reached chunk clears the bar at a content-only score of |
|---|---|
| `0.0` | 0.450 — no discount; it competes like any other candidate |
| `1.0` | 0.231 |
| `3.0` | 0.039 — near enough everything the graph touches |

**The walk is always exactly one hop, and that is not configurable.** It visits
each seed section's immediate neighbours, in or out, and stops. `graph.seeds`
chooses how many candidates to walk from and `graph.max_expand` caps what comes
back, but neither is a depth.

This is deliberately **not** a validation error. A config option that cannot
produce a result under any setting of other knobs would not be an option at
all — but `relevance.traversal` is such a knob, so `graph.enabled` is a real
choice whose effect is conditional, which is a thing to document rather than
to reject at load.

**One list, not two.** What the graph reached arrives as an ordinary candidate
with its text, competing for the same `top_n` slots. It previously had a
section of its own containing node names and arrows and no text at all, which
told the model those sections existed and nothing about what they said — so
acting on one cost a `memory_search` turn, the thing the block exists to avoid.
Because the block is rebuilt per turn it never hits the prefix cache, so those
tokens were paid in prefill every time.

Why a gate and not a distance floor: vector distance is only a weak relevance
signal (a similar-but-off-topic memory sits as close as a relevant one), so any
floor strict enough to keep junk out also starves turns that needed memory.
Keyword overlap on *meaningful* words is a second, largely independent signal;
together they separate relevant from irrelevant memories about as well as an
LLM judge did, in milliseconds. The keyword index is built once per reindex
and kept in memory.

On the store the defaults were fitted to (443 chunks, MiniLM embeddings, 50
real messages with 1,713 graded message-memory pairs — headings were in the
keyword index at the time but not yet in every vector, so `distance` was fitted
against a slightly weaker vector leg than it now scores; correcting that moved
the median best distance by 0.008 and did not justify a re-fit): 3.8 memories
per turn at 69% relevant, against an ungated always-5 slice at ~50%, with the same share of
relevant memories found — and a turn with nothing relevant stays silent half the
time rather than never.

Those figures are not comparable to ones taken against an earlier, narrower
label set; widening the graded pool changes the denominator. Compare arms within
one set, never across two.

**Reindex cadence.** Maintenance loop checks `sourcesHash` every 5 min
and reindexes if changed. Source-file edits (e.g., a memory_flush
appending to today's file, or the nightly curator rewriting a note) get
picked up within 5 min.

⚠ **A source hash cannot notice that the *code* reading it changed.** If you
alter what gets indexed — the text handed to the embedder, how a note is split
into chunks, what the keyword index sees — the sources are identical, no
reindex fires, and the change ships and does nothing until something forces a
rebuild by hand. Bump `INDEX_VERSION` in `memory.py` with any such change: it
is recorded in `sync_state.json` and a mismatch forces a full rebuild, so the
deploy reindexes itself. Note that a full rebuild and an incremental update are
two code paths over the same store — keep them producing identical documents,
or a chunk's vector will depend on which path last touched it.

**Supersession.** A memory the curator has superseded carries a
`supersededBy=<id>` marker; retrieval auto-follows the old→new chain and
renders the current head last, so a stale fact stays searchable as a
visible timeline without outranking its replacement. See curation below.

**Tuning the gate.** The weights are calibrated to a store's shape — how
its memories are written and chunked, its vocabulary, its size — and they
drift as the store changes. A WARNING like `memory retrieval drift: median
best distance … calibrated at …` means the store has moved away from where
they were fitted.

Two things it is **not**. It is not a deploy-time check: it fires only after
100 live retrievals, i.e. roughly 100 real turns, so after a change to the
indexed representation compute the median best vector distance over a frozen
query set directly instead of waiting for it. And it is not an instruction to
re-fit — it is a prompt to re-measure. **A re-fit must beat the shipped weights
at matched volume before it is taken.** One measured example: a re-fit prompted
this way showed precision 0.59 → 0.69 and was rejected, because it bought that
purely by injecting 2.04 memories a turn instead of 3.76, with recall falling
0.57 → 0.47. At equal volume it was noise.

To re-calibrate:

1. Take ~50 real messages from transcripts. For each, collect the ungated top
   `candidates` (e.g. `_candidates()` with `threshold: 0`).
2. Grade each message–memory pair 0/1/2 for relevance (by hand, or with a
   strong model as referee — spot-check it).
3. Fit a logistic regression of *relevant (grade ≥ 1)* on
   `[d, log(1+kw), kw/kw_best, g]`, cross-validated by message so it cannot
   memorise; set `relevance.*` to the fitted weights and choose `threshold`
   from the precision/recall trade-off you want.
4. Set `calibration.best_distance` to the median best vector distance per
   message on that sample.

All four are `python -m claw.evaluate collect`, `grade`, `fit` and `score`.
`fit` groups its folds by message — a random split puts candidates from one
message on both sides, and since they share `kw_best` and much of their
subject matter, that leaks. It ridge-penalises, because `traversal` is zero
for most rows and non-zero for a clustered minority, which is the setup where
an unpenalised coefficient runs off to infinity and reports a model that looks
superb and predicts nothing.

**Re-pick `threshold` after any re-fit.** New coefficients mean a new
probability scale, so carrying the old number across silently changes
behaviour on the *existing* population, not just the new one.

**Widen, do not redraw.** `collect --extend` keeps the existing messages and
only lengthens their candidate lists, so earlier grades still describe the same
pairs and `grade` fills the gaps. Collect wider than production injects: the
referee costs one call per message whatever the list length, so a pool that
already covers the next experiment is far cheaper than a second run.

**Keep what step 2 produces.** The grades are the expensive part and the
reusable part: a label says whether this memory helps answer this message,
which does not change when scoring, term selection or fusion changes. So they
are not only for re-fitting — any later change to the retrieval path can be
scored against them with **no model calls at all**, which is the difference
between validating a change and asserting it. Regrading is not a cheap redo
either: a referee is not deterministic, so fresh labels are a different
baseline and earlier measurements stop being comparable.

**But a label set decays, so check it rather than trusting it.** A chunk id is
positional (`{source}:{index}`), so editing or re-sectioning a note re-points
an id at content nobody graded and the label silently describes text that is
no longer there. On one store 31 labels across 8 ids went stale in a single
day, because an agent kept writing to that day's note. `collect --extend`
compares the frozen text of every labelled chunk against the live corpus and
refuses rather than building on it; `--drop-drifted` discards just those pairs
so the next `grade` re-labels them, which costs no extra calls when those
messages are being visited anyway.

`score` reports the share of injected memories carrying **no** label. Every
rate treats an unlabelled memory as irrelevant, so a non-zero share makes
precision a floor rather than a measurement — and that is exactly what a change
widening the candidate pool produces, which is when the number is most likely
to be misread as a regression.

Tests will not catch this class of change. `relevance` is a fitted model whose
inputs include keyword scores, so altering which words are searched moves the
distribution its coefficients were fitted on while every test stays green —
and the drift WARNING watches vector distance, so it does not fire either.

The set holds real messages and real memory text. Keep it where backups reach,
out of version control, and far from any public mirror.

Quick adjustments without re-fitting: raise `threshold` for fewer, cleaner
memories; lower it (or `0`) for more. The explicit `memory_search` tool is
never gated.

### Smart retrieval

`memory_retrieval.smart_retrieval.enabled: true` adds a second opinion on the
candidates near the threshold. It **requires a `systemone` block** — enabling
it without one is a load-time error, not a silent no-op.

The relevance formula is calibrated and free, but it is four coefficients, and
near its own threshold it is guessing. Smart retrieval sends just that band to
the scorer, one question per candidate, and acts on the answers:

- a **rejected** candidate the scorer calls relevant is **promoted in**;
- a **retrieved** candidate it calls irrelevant is **dropped**;
- everything surviving is injected, with **no `top_n` slice** — a turn may
  inject nothing, or everything that passed.

**Only the borderline is asked about**, which is what makes it affordable, and
each bound was measured rather than chosen:

| band | behaviour | why |
|---|---|---|
| below `threshold - promote_margin` | dropped unasked | of 170 candidates that far below, the scorer promoted **zero** |
| the band, either side | asked | where the formula is guessing |
| above `threshold + review_margin` | kept unasked | the formula is confident there and the scorer is worse than it — it wanted to drop 17 such entries and 13 were useful |
| graph-reached, any score | **always asked** | the fitted weights lean on a keyword score an expanded chunk has no reason to have, so the formula's opinion of one is not evidence |

Both margins are **relative to `threshold`**, so tuning the threshold moves the
band with it. Absolute values would drift off the boundary they exist to police.

**Cost.** Roughly `fixed + questions x per-question`, the per-question part
scaling with `note_chars` (measured 59 ms at 400 characters, 42 at 200, 35 at
100). On one store the whole pass ran ~1.4 s a turn, median 1.3 s with a tail
to ~3.9 s. A turn whose candidates all sit outside the band asks nothing and
costs nothing.

`smart_retrieval.timeout_s` (default 5.0) bounds that batched request, and is
**retrieval's own deadline, not the gate's**. The two share one client but sit
on different paths: the gate answers in front of the LLM and must fail open
quickly, while this asks about every borderline candidate at once and its
failure costs precision rather than memory. Setting it to the gate's 2 s
truncates the tail — those turns fall back to ordinary retrieval — and cost
~5 points on **every** metric below.

Against that, more accurate retrieval can pay for itself in turns the agent no
longer spends hunting for what it should already have been told — a memory that
arrives unbidden is one the agent never has to suspect exists and go searching
for. Treat the latency as a ceiling on the cost, not as the net.

**Measured** on one store, 50 real messages graded by the consuming model with
the same conversation context the scorer reads, through the shipped code path
at the shipped deadline:

| | ordinary | smart |
|---|---|---|
| memories per turn | 3.64 | **3.16** |
| relevant (grade ≥ 1) | 0.50 | **0.64** |
| clearly relevant (= 2) | 0.32 | **0.45** |
| recall of clearly relevant | 0.62 | **0.73** |
| stayed quiet when nothing was relevant | 0.10 | **0.70** |
| got something when there was something | 0.93 | 0.82 |

**It injects less while recalling more**, which is the point: shrinking the
injected set raises precision on its own, so an arm that injects less normally
has to be read against a random cut of the same size before it can claim
anything — but a random cut lowers recall in proportion and cannot raise it.
By grade, the exchange is **34 net junk chunks shed** and **12 net clearly
relevant gained**, against 2 net marginals lost.

The one metric that moves the wrong way is *got something when there was
something*, 0.93 → 0.82. Read it with the grades attached: all four turns it
counts as losses had only **marginally** relevant memories available, so no
turn lost a clearly relevant memory to silence — which is why recall rises at
the same time.

> ⚠ **Measure this in the shape you deploy.** Two dimensions each move the
> result by ~5 points and neither is visible in the output. A graded set
> collected **without conversation context** has the scorer judging a bare
> message, which is not what it sees at runtime (it reads
> `smart_retrieval.context_turns`); collect with context and show the referee
> the same thing. And a bench **deadline** more generous than the deployed one
> flatters the result by letting the tail through. Both were walked into.
> `evaluate smart --timeout` defaults to the shipped value for this reason.

> ⚠ **`note_chars` and the two thresholds are one calibration.** Clipping moves
> the scorer's whole score distribution, so the same thresholds cut somewhere
> else: between 400 and 200 characters `retain_threshold` had to move
> 0.149 → 0.107 to hold the same operating point. Changing `note_chars` alone
> is a silent regression, not a tuning. 200 was measured to be
> indistinguishable from 400 once the thresholds were re-fitted; 100 was not.

**Failure is always open.** No scorer, a transport error, a timeout or a
malformed answer all fall back to ordinary retrieval for that turn. An outage
costs precision, never memory.

Score it against a graded set with `python -m claw.evaluate smart`, which runs
the shipped path with a real scorer and reports ordinary retrieval beside it.

## Curation (forgetory) in depth

The **collector** (`memory_flush`) captures durable knowledge frequently
and cheaply. The **curator** (`memory_curation`, "forgetory") is the
heavier, once-a-night counterpart that keeps the accumulated memory tight.
It's off by default (`memory_curation.enabled: false`) and **coupled to
`memory_flush`** — it only runs for agents that collect.

**Model of record.** Memory is markdown; ChromaDB/BM25/graph are re-derived
from it (`rm -rf <workspace>/.memory/` rebuilds intact). Each memory section
carries an HTML-comment marker under its heading —
`<!-- mem ts=<date> id=<id> status=<active|…> supersededBy=<id?> -->`. The
collector seeds `ts`; the curator fills in the rest. Because every edit the
curator makes is atomic markdown, a partial or timed-out pass always leaves
memory in a valid state.

**What a pass does.** Per night, for each selected daily-note file (one
bounded turn each), the curator may:

- **dedup** — collapse near-identical memories (e.g., recurring-cron churn)
  to one canonical entry;
- **supersede** — mark a stale long-term fact `supersededBy=<id>` pointing at
  the memory that replaced it (retrieval follows the chain);
- **archive** — move lapsed ephemera (past appointments, "today is X") out of
  the indexed daily notes into monthly shards `memory/archive/YYYY-MM.md`
  (preserved with provenance, never indexed — they live in a subdirectory the
  `memory/*.md` source glob doesn't recurse into). Sharding keeps each archive
  file small so the curator appends cheaply instead of rewriting a monolith;
- **uncertain** — leave a memory in place with an inline caveat when a
  judgment (e.g., a dangling `supersededBy`) can't be made safely.

**File selection (forward-only date cursor).** Nightly the curator grooms only
daily notes dated **after** `curatedThrough` and **before** today, then advances
the cursor to the last contiguously-curated date (a timeout pins it so that note
retries next night rather than being skipped). `curatedThrough` (a `YYYY-MM-DD`
date at `<workspace>/.memory/curation_state.json`) makes the curator monotonic:
files at/before it are never re-scanned — they resurface only as near-neighbours
when a newer note supersedes them — so the curator's own edits to a prior note
never re-enqueue it. In steady state that's just yesterday's completed note.
`MEMORY.md` carries no filename date and is **never** a nightly candidate; it's
maintained as a side effect of curating the notes that supersede it (a
full-corpus bootstrap still grooms it). Separately, a **whole-corpus
supersession-review turn** runs every pass — handed the complete superseded list
(any age, not window-limited) with each entry's age, archiving ones stale for
~`superseded_archive_days`+, case-by-case. Today's daily note is excluded from
curation (it's still being written by the collector).

**Audit trail.** The curator narrates its own decisions via a `record_action`
tool — one reasoned `claw.curator` line per `archive`/`supersede`/`dedup`/
`uncertain` in `claw.log` (grep `claw.curator`). This is the primary window into
*why* it did what it did; watch it for the first few nights after enabling.

**Bootstrap the backlog, then seed the cursor.** There's no date cursor before
the first-ever pass. Run the one-off full-corpus bootstrap
(`python -m claw.memory_curate --config … --agent … [--dry-run]`), which grooms
every not-yet-id-marked file (resumable — it skips fully-marked ones), optionally
capped with `max_files_per_run` to chunk across runs. Once the backlog is clean,
seed `curatedThrough` to the last curated date so the first nightly starts from
there; if the cursor is ever missing, the nightly pass conservatively assumes
everything through yesterday is done rather than re-curating the corpus.

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

## Admin commands

In-band operator commands over Matrix. Requires the `commands:` config
block (see Configuration); a prefixed message from a non-allowlisted
sender is silently treated as ordinary text. Every action is bounded to
a single turn (or one subagent subtree); the only session-wide command
is the read-only `%subagents`.

| Command | When to use |
|---|---|
| `%context` | See how close the live transcript is to auto-compaction. Reports the next turn's starting floor. Read-only. |
| `%compact` | Force a compaction now (e.g. before a long task) without waiting for the threshold. No-ops *without* flushing if the transcript already fits the keep window. |
| `%clear` | Hard reset this conversation. Runs a final memory_flush first so durable knowledge survives; transcript archived `.reset-<ts>`. |
| `%stop` | Panic button: a runaway/looping turn, or one you don't want to finish. Cancels the turn + its whole subagent cascade and SIGKILLs their bash. |
| `%stop --soft` | Same, but leave in-flight shell commands running — when a non-idempotent command (DB dump/restore, migration, package install, large write) is mid-flight and a SIGKILL would corrupt it. You're stopping the agent, not that command. |
| `%stop <task_id>` | Kill one specific subagent + its descendants without touching the parent turn or siblings. Get the id from `%subagents`. `--soft` spares its bash. |
| `%subagents` | List this conversation's running subagents (task_id, persona, elapsed) — discovery for targeted `%stop`. |
| `%verbose on` / `off` | Set DEBUG logging at runtime, no restart. Process-wide. Bare `%verbose` reports state. |
| `%thinking on` / `off` / `full` | Surface the model's reasoning trace before replies. `on` = final answer turn's; `full` = every tool-loop iteration's. Per-**conversation** (unlike `%verbose`'s process-wide scope), runtime, ephemeral. Bare `%thinking` reports state. Default off. |

`%stop` acts immediately and concurrently — it is not queued behind the
turn. Cancellation lands at the running turn's next `await` (sub-second
for I/O-bound work). A turn wedged with no `await` can only be ended by
restarting the gateway (the realistic wedge — a long `bash` — is killed
by the default `%stop`). After a cancel a synthetic
`[SYSTEM … cancelled-by-user …]` marker is appended to the transcript
so the agent treats the stopped instruction as abandoned instead of
re-attempting it (which would loop). `%stop` suppresses the cancelled
cascade's completion delivery so a zombie subagent can't resurrect a
stopped conversation; a targeted `%stop <task_id>` lets the target's
own (cancelled) completion deliver — the session is alive and should
learn it died — but suppresses collateral descendants.

Relevant log lines: the `cmd-<agent>-<sid>-<name>` command-handler task
name; `claw.tools.bash` `killpg(...)` entries; subagent `cancelled` /
`completion suppressed (session stopped)`.

## Troubleshooting

### Increase log verbosity

`%verbose on` flips `claw.*` loggers to DEBUG at runtime — process-wide,
no restart; `%verbose off` reverts; bare `%verbose` reports the current
state (see Admin commands). For boot-time verbosity instead, set
`verbose: true` in `claw.yaml` and restart. DEBUG is noisy — flip back
off when done investigating.

To inspect a model's chain-of-thought instead of log internals, use
`%thinking on` — it surfaces the final answer turn's reasoning to
Matrix for *this conversation* (not process-wide, no config key),
leaving logs and transcripts untouched. `%thinking full` surfaces every
tool-loop iteration's reasoning instead (verbose; for debugging a
multi-step turn). Flip back `off` when done.

Both `%verbose` and `%thinking` are in-memory runtime state, not
persisted: a gateway restart resets `%verbose` to the `claw.yaml`
`verbose` boot value and `%thinking` to off for every conversation.
Re-issue after a restart if you need it back.

### Common log signatures

| Signature | Means |
|---|---|
| `claw.main INFO ready (N agents)` | Boot complete. |
| `claw.main INFO maintenance: K periodic flush task(s) running` | K active sessions met growth threshold this tick. Silence = no qualifying sessions. |
| `claw.memory_flush INFO [<sid>] memory flush starting (reason=<r>, N rows)` | Flush turn beginning. |
| `claw.memory_flush INFO [<sid>] memory flush done (reason=<r>)` | Flush turn complete; durable bullets written. |
| `claw.memory_flush ERROR [<sid>] memory flush turn failed (reason=<r>)` | Flush threw an exception (Traceback follows). Compaction will proceed regardless. |
| `claw.ollama INFO [<label>] turn N: K tool_call(s) requested` | Tool round-trip. `<label>` shape: `<agent_id>:<kind>[:<peer_or_task>]` — e.g. `agent-1:main:user-1` (user-facing turn from `@user-1:example.org`), `agent-1:flush:periodic-growth:user-1` (background memory flush of that user's session), `agent-1:subagent:persona-3-a1b2c3d4` (subagent one-shot, parent's id + kind + the spawned task_id). At DEBUG verbosity an additional `:<sid>` correlation handle is appended for the matrix call sites (subagent labels stay as-is — the task_id is already a stable correlation handle). If N approaches `max_tool_turns`, the model is in a tool loop. |
| `WARNING [<agent>] background flush timed out after Xs` | Flush exceeded `turn_timeout_s`. |
| `claw.main INFO firing job <name>` | Cron-driven inbound being dispatched. |
| `[<agent>:gate:<peer>] -> <handler> (chosen, <handler> @ 0.92)` | The decision gate answered directly; `turn complete (direct: <handler>, …)` follows. `-> llm (<reason>, …)` means it passed the turn on — `no-handlers`, `none`, `low-confidence` (with the score to tune against) or `scorer-error`. See `docs/decisions.md`. |

### A gate handler never fires

Check, in order: the boot line `[<agent>] gate handlers: …` lists it (a handler
naming a tool the agent lacks is dropped with an `ERROR`); the turn is
eligible (one message, `matrix`/`voice`); and the `:gate:` line's reason. A
steady `low-confidence, <handler> @ 0.7x` means the right pick under the bar —
tune the description or that handler's `min_confidence` against a measured set
of utterances and near-misses (`docs/decisions.md` *Tuning*). `scorer-error`
means the decision service is down, slow (past `gate.timeout_s`) or
answering in the wrong format; the turn went to the LLM, as designed.

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

### Music tools report "the player is not running"

The gateway connects to an mpv IPC socket; it never starts mpv. That process is
the deployment's to supervise, and this message means the socket could not be
reached — mpv is down, or `mpv_socket` in `claw.yaml` does not match the path
mpv was actually given.

Two things make this look like a permissions problem when it is not. mpv
creates the socket `srwxr-xr-x`, and connecting to a unix socket needs **write**
permission on it — so anything probing the socket must run as the user mpv runs
as, not merely as a user who can read the path. And mpv silently keeps a stale
socket file across an unclean exit, so the file existing proves nothing; the
check that means something is a property read that comes back
`{"error":"success"}`.

### Music plays at wildly different volumes

Expected, if `normalize` is off — which is the default, and usually right. Two
records mastered thirty years apart genuinely differ in average loudness while
peaking in the same place; that is the mastering, not a fault. Switch
`normalize` on if the room's noise floor makes average level matter more than
peaks.

If it is on and volumes still vary, the catalogue has no measurement for those
tracks. An unmeasured file is deliberately assumed *loud*, so it plays quiet
rather than deafening — a whole album sounding oddly reticent is the signature.
Run ingest and check the track count matches the library.

Loudness comes from the file's measurement, not from any tag, so retagging
changes nothing. A single mis-measured track is a fact about the file rather
than something to override: the measured columns are not curatable by design.

### A track is missing from a shuffle

Album furniture — a short track far quieter than its own record — is left out
of shuffles on purpose, and kept in album order. Play the record in sequence
and it comes back.

### Curation seems to have vanished after a re-scan

A normal ingest cannot overwrite curation; the conflict clause does not name
those columns. Two things legitimately clear it:

- **`--repopulate`** on that track, which is the deliberate "restore defaults"
  and discards curation by design.
- **A renamed or moved file.** The catalogue is keyed on path, so a rename is a
  delete plus a fresh insert and nothing can tell it from a swap. Do bulk
  renames *before* investing in annotation, not after. The append-only edit log
  survives and is keyed on the old path, so a mistaken rename is recoverable by
  hand.

Curation attached to an album or an artist is unaffected by either, since it is
keyed on the name rather than the path — which is a further reason to put a
fact at the highest level where it is true.

### Bot replies to DM but not to group `@`-mentions

Confirm `allow_bots: mentions` (not `none`) and that the mention is a
proper `m.mention` event (Element's `@` autocomplete produces these;
plain text `@bot` does not).

### "user verification unavailable" in Element

`password_file` was unset on first boot, so cross-signing was skipped.
Add `password_file`, then either restart (it'll bootstrap automatically)
or set `force_cross_signing_replace: true` for one boot to force a
replace.

### SAS verification stalls on "waiting for other user", then cancels

The peer sends its key, nothing visibly happens for ~30 s, and it
cancels with `m.key_mismatch` — having never sent a MAC. Despite the
cancel code, this is **not** a MAC problem: the verification never got
that far.

matrix-nio builds the SAS commitment with `sha256(...).hexdigest()`,
but the spec requires the hash as **unpadded base64** (43 characters,
not 64 hex). The peer stores the commitment and checks it when our
ephemeral key arrives, so a hex string can never match. claw re-encodes
it in `claw/channel/sas_compat.py`; the fix is a no-op if a future
matrix-nio emits base64 itself.

The same module handles a second nio defect: `chosen_mac_method` is
pinned to legacy `hkdf-hmac-sha256` while MACs are computed with the
*corrected* base64. claw prefers `hkdf-hmac-sha256.v2` when the peer
offers it (the spec forbids v1 in that case), and falls back to forcing
libolm's encoding for v1-only peers.

Because these events are E2E-encrypted, a failed exchange **cannot** be
reconstructed from the room timeline. claw logs every SAS event in both
directions — grep the log for `SAS <<<` / `SAS >>>` and read the actual
`accept` content before theorising.

### Upgrading matrix-nio across the vodozemac boundary

nio 0.26 replaced libolm with vodozemac. An existing store migrates
automatically and losslessly — same device fingerprint, sessions intact
— but the migration is **one-way, and merely opening a store performs
it**. A 0.26 process that only *reads* a store rewrites its pickles;
0.25 then fails on it with `OlmAccountError: BAD_ACCOUNT_KEY`.

Back up each `.matrix-store/` before upgrading, and never point a
different-version interpreter at a live store to "just check" — verify
on copies.

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
