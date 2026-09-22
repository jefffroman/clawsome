# Fast decisions: the System One client and the decision gate

Most of what an agent decides is small: *which of these is the user asking
for?*, *is this relevant?*, *how urgent is it?*. An LLM can answer, but it
spends seconds generating text that code then parses back into an `if`.
A **System One** decision service answers the same kind of question
directly — typed options in, a probability for each option out, nothing
generated — in tens to hundreds of milliseconds.

Clawsome can use one. It is **optional**: with no decision service
configured, clawsome behaves exactly as it does without these features, and
that is a fully supported way to run.

Two pieces:

- **The client** (`systemone:` block, `claw/systemone.py`) — one shared
  connection to a decision service, available to every agent and every
  feature.
- **The decision gate** (`gate:` block, `claw/gate.py`) — its first user: an
  outer loop in front of the LLM that answers simple, command-like turns
  ("pause the music", "what's playing?") with a direct action instead of an
  LLM turn.

## The external service

### What it must speak

`POST {base_url}/v1/systemone` with a JSON body

```json
{
  "state": "...",                       // string, object or array: the context
  "model": "local",
  "questions": {
    "<id>": {"type": "choice", "instructions": "...", "criteria": {"opt-a": "description", "opt-b": null}},
    "<id>": {"type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}},
    "<id>": {"type": "score",  "instructions": "...", "criteria": ["level 0", "level 1", "level 2"]}
  }
}
```

answered with

```json
{"model": "...",
 "answers": {
   "<id>": {"type": "choice", "choice": "opt-a", "probabilities": {"opt-a": 0.93, "opt-b": 0.07}, "confidence": 0.63},
   "<id>": {"type": "noul", "noul": 0.91},
   "<id>": {"type": "score", "score": 1.4, "legend": {"0": "...", "1": "..."}, "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5}, "confidence": 0.3}},
 "usage": {"input_tokens": 312, "output_tokens": 0}}
```

This is the wire format of TypeSafe's hosted **Jev** model ("System One
API"). The three question types:

| Type | Asks | Answer |
|---|---|---|
| `choice` | pick one of the listed options | `choice`, `probabilities`, `confidence` |
| `noul` | yes or no | `noul` = P(yes) |
| `score` | rate against ordered levels | `score` (probability-weighted level), `probabilities`, `confidence` |

`confidence` summarises how concentrated the distribution is (1.0 = all mass
on one option). The gate compares it against its thresholds, so a service
must return it on `choice` answers.

### What to run

Clawsome ships **no** decision service, just as it ships no inference or
search. Any service that implements the endpoint above works. Two notes:

- **The client sends no `Authorization` header.** It is built for a service
  on the same host or a trusted private network, like the other services
  clawsome talks to. The hosted Jev API requires a bearer token and is
  therefore not usable as-is.
- **Open reproductions** of the interface run a small open-weights model
  locally and read each option's probability straight from the model's
  logits after one pass over the state (no decoding). They typically cap the
  number of options per question (e.g. 16, one answer-slot token per option)
  — keep the gate's handler count under that cap (the gate adds one option,
  `none`). Their probabilities are generally **uncalibrated**: set thresholds
  from measured routing, not from theory (see *Tuning* below).

Quick check:

```bash
curl -s http://127.0.0.1:11502/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {"urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}
}' | jq .answers
```

Latency matters more than throughput here: the gate asks one question per
eligible turn, on the reply path. A service sharing a GPU with the LLM will
answer more slowly while the LLM is generating; that only matters if it
approaches `gate.timeout_s`. Note that each caller sets its own deadline —
retrieval's batched call deliberately waits longer than the gate's, so "slow"
means different things to them.

## The contract for code that uses the client

Every feature that asks the decision service anything must:

1. **Check that a client exists.** Top-level agents receive it as
   `Agent.systemone`; it is `None` when no `systemone:` block is configured.
   `None` means: behave exactly as the feature would without a decision
   service.
2. **Treat every failure the same way** — timeout, connection refused, a
   non-200, a missing answer. `SystemOneClient.ask` raises on all of them; the
   caller catches and falls back.
3. **Have its own switch**, so a deployment can enable one use without
   another.

The gate below is written to this contract. It is also the model for future
internal uses (e.g. judging whether a retrieved memory chunk is relevant, a
`noul` per chunk over one shared state).

## The decision gate

### Where it sits

For each inbound turn, after in-band `%` commands are handled and the
timestamp envelope is applied, **before** memory retrieval and the LLM:

1. **Eligibility.** Exactly one message, on the `matrix` or `voice` channel,
   not a subagent completion. Cron turns, `initial_prompt` turns, subagent
   completions and merged batches never consult the gate.
2. **No handlers → pass through**, without calling the service. The log says
   `-> llm (no-handlers)`.
3. **One `choice` question.** State = the last `context_turns` user/assistant
   messages (tool traffic, synthetic notes and envelopes removed) plus the new
   message. Options = each handler's `description`, plus `none` ("needs a
   conversational reply, more than one action, or anything not listed").
4. **Route.** A handler runs only if it is the pick with `confidence` at or
   above its bar (its own `min_confidence`, else the gate's). Everything else —
   `none`, a low-confidence pick, any service failure — continues into the
   ordinary LLM turn, unchanged.
5. **Direct answer.** The handler performs its action and produces the reply.
   The user message and the reply are written to the transcript like any turn
   (so the next turn — direct or LLM — has the context), the reply goes out
   through the turn's normal sink (for voice, this resolves the waiting HTTP
   request), and the usual end-of-turn maintenance runs. If the handler
   *declines* (its tool did not report success), nothing is written and the
   LLM takes the turn.

The gate can only ever **remove** an LLM turn. It never alters one: when it
passes a turn through, the LLM sees exactly what it would have without a gate.

### Why context matters

The state includes recent conversation so that short follow-ups route
correctly. "restart the music" right after the agent said "Paused." is
clearly *resume*; the same words after "Now playing …" are ambiguous (start
over? stop?), and a well-behaved service will be less confident. The context
is the **conversation**, not the world: a change made outside this chat
(another client, another room) is invisible to it.

### Handlers pick; they never extract

The service returns a choice among options it was given — not text. So a
handler's action must be **fixed**: "pause", "skip", "what's playing". A
request that needs something *taken from the message* — "play some reggae",
"turn it down a bit" — is not a handler; `none` sends it to the LLM.

## Configuration

```yaml
systemone:
  base_url: http://127.0.0.1:11502

gate:
  enabled: true
  exposed_to: [agent-1]
  timeout_s: 2.0
  min_confidence: 0.8
  context_turns: 6
  handlers:
    - id: audio-pause
      description: "Pause the music or other audio that is playing, so it can be resumed later"
      tool: music_control
      args: {action: pause}
      expect: paused
      reply: ["Paused.", "Pausing.", "Okay, paused."]
    - id: audio-next
      description: "Skip the current song or audio track and go on to the next one"
      tool: music_control
      args: {action: next}
      outcomes:
        skipped:
          reply_fn: skipping_to
          fallback: ["Skipping.", "Next one."]
        end_of_queue:
          reply: "That was the last one — the queue's finished."
      min_confidence: 0.75
    - id: audio-status
      description: "Identify the song, artist or other audio that is playing right now"
      tool: music_status
      reply_fn: playing_brief
```

### `systemone:`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `base_url` | str | `http://127.0.0.1:11502` | Where the service listens. |

Omit the block entirely to run without a decision service.

**No timeout lives here.** A deadline is a property of the question being
asked, not of the transport: the gate must fail open fast in front of the LLM,
while a retrieval batch asks about many candidates at once and is worth
waiting longer for. One client is shared, so a default here became whichever
caller's policy got there first — and the caller that had not stated one
silently inherited it. `ask()` therefore requires the deadline, and since every
caller passes one, a client-level default would be unreachable regardless: a
per-request timeout overrides it for connect, read, write and pool alike.

### `gate:`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch. |
| `exposed_to` | list[str] | `[]` | Agent ids that get a gate. Top-level agents only; subagents never do. |
| `min_confidence` | float | `0.8` | Default bar, within [0, 1]. |
| `context_turns` | int | `6` | Prior user/assistant messages included in the state. `0` = the new message alone. |
| `handlers` | list | `[]` | See below. **Requires a `systemone:` block**; a gate with no handlers needs none. |

### Handlers

Every handler names **one tool from the agent's own registry** (a built-in, a
config-gated family such as `music_*`, or a skill `tool.py`) and calls it with
**fixed** `args`.

| Key | Required | Meaning |
|---|---|---|
| `id` | yes | Unique; `none` is reserved. Appears in logs. |
| `description` | yes | **What the service reads to decide.** Write it as the request it handles, not as the implementation. This is the main tuning surface. |
| `tool` | yes | Tool name. A handler naming a tool the agent does not have is dropped at startup with an `ERROR` log (the gateway still starts). |
| `args` | no | Fixed arguments for the tool. |
| `min_confidence` | no | This handler's own bar — e.g. higher for a destructive action. |

Then **one** of two shapes:

**Text shape** — `expect` (optional) plus one reply form at the top level.
The tool runs normally. Its text output must equal `expect`, or, with no
`expect`, merely not start with `error:` / `refused:`. Anything else declines.

**Outcomes shape** — `outcomes:`, a map from result to reply form. The tool's
**structured result** (its `data` twin, below) is used instead of its text,
and the map is keyed on that result's `result` field. A result not listed
declines. Use this when *what happened* changes what to say — e.g. a skip
that reports the next track, or that the queue has ended. A handler using
`outcomes` on a tool without a `data` twin is dropped at startup with an
`ERROR`.

### Reply forms

Exactly one per handler (text shape) or per outcome:

| Form | Says |
|---|---|
| `reply: "text"` or `reply: [a, b, c]` | That text, or one of the variants chosen at random each time. |
| `reply_fn: <name>` (+ `fallback: [...]`) | What the named reply function returns. When it returns nothing: a `fallback` variant — or, in the text shape with no fallback, the tool's own words. In the outcomes shape a `fallback` is **required**: the action has already happened, so there must always be something to say. |
| `relay: true` | The tool's own text output. Text shape only. |

`fallback` is valid only beside `reply_fn`. Reply functions live in
`claw/gate_replies.py`; shipped ones:

| Name | Use with | Says |
|---|---|---|
| `skipping_to` | `music_control` `next`, outcome `skipped` | "Skipping to \<title\> by \<artist\>." / "Skipping — the DJ's up next." |
| `playing_brief` | `music_status` | "Now playing: \<title\> by \<artist\>." / "Paused: …" / "Nothing is playing." |

### Validation

At load: unknown keys anywhere in the two blocks raise; `min_confidence`
outside [0, 1], negative `context_turns`, duplicate or reserved (`none`)
handler ids, an unknown `reply_fn`, zero or several reply forms, `fallback`
without `reply_fn`, `relay` in an outcome, and handlers without a `systemone`
block all fail the load. Binding a handler to a tool happens at agent
construction, where a missing tool drops that handler with an `ERROR` rather
than failing the boot.

## Logs

```
[agent-1:main:user-1] turn starting (1 inbound)
[agent-1:gate:user-1] -> audio-pause (chosen, audio-pause @ 0.92)
[agent-1:main:user-1] turn complete (direct: audio-pause, reply 7 chars, 0.6s)
```

| Line | Means |
|---|---|
| `[<agent>] gate handlers: a, b, c` | At boot: the handlers bound for this agent. |
| `-> llm (no-handlers)` | Eligible turn, no handlers; the service was not called. |
| `-> llm (none, none @ 0.86)` | The service picked `none`. |
| `-> llm (low-confidence, <id> @ 0.79)` | Picked a handler, under its bar. The number is what to tune against. |
| `-> llm (scorer-error)` | Service unreachable, slow or erroring (a WARNING line has the detail). |
| `-> <id> (chosen, <id> @ 0.92)` | Direct answer follows. |
| `turn complete (direct: <id>, …)` | The direct turn finished. |
| `handler <id> declined: <tool> returned …` | The tool did not report success; the LLM takes the turn. |

No gate line at all on a turn means it was not eligible (cron, merged batch,
…) or the agent has no gate.

## Tuning

Descriptions and bars are tuned against **measured routing**, not intuition:

1. Write a set of utterances: several phrasings per handler (short and long,
   typed and spoken style, with and without a preceding exchange), and — more
   important — **near-misses** that must go to the LLM: requests that mention
   the same things but need extraction or more than one action ("play the next
   album by them", "remind me to stop the music at 10pm", "pause the music and
   tell me the time").
2. Run them through `DecisionGate.decide` against the real service, with
   realistic conversation state. Count correct routes and, separately,
   **wrong handlers above their bar** — the only dangerous outcome. A miss
   costs an LLM turn; a wrong handler does the wrong thing.
3. Adjust descriptions or per-handler bars until wrong-above-bar is zero, then
   watch the `-> llm (low-confidence, …)` lines in real use.

Options compete: changing one description can move another handler's
scores. Re-run the whole set after every change.

## Extending

**A reply function** is an async function in `claw/gate_replies.py`,
registered in `REPLY_FUNCTIONS`:

```python
async def my_reply(answer, ctx, tools) -> str | None:
    ...
```

`answer` is the tool's text (text shape) or its structured result (outcomes
shape); `ctx` is the `GateContext` (agent id, session, message, the state the
service saw); `tools` is the agent's tool registry. Return the reply, or
`None` to use the fallback. It runs **after** the action: it must never
decline the turn, and an exception is logged and treated as `None`. Read
tools' structured `data`, never their prose.

**A tool's structured twin.** `Tool` has an optional
`data: async (args) -> dict`, used by code and never shown to the model. A
tool's text is written for the model and may be reworded for it at any time,
so code that needs fields reads them from `data`:

- For a **read-only** tool, `data` can be a separate call (e.g.
  `music_status.data` → `{"state": "playing", "title": …, "artist": …}`).
- For an **action** tool, `data` **is** the action — calling it once performs
  it — and `run` must render its text from the same single call. An action
  must never run twice to be described twice. `music_control` shows the
  pattern (`_do_control` returns both faces).

An outcomes handler requires a `data` whose result carries a `result` field.

**A handler that is not a tool call** can be written as a Python class with
`id`, `description`, `min_confidence` and `async handle(ctx) -> str | None`
(the `DirectHandler` protocol) and passed to `DecisionGate` directly; the
declarative form covers the common case.
