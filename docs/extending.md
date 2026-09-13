# Extending

Three extension points: **skills** (workspace-local), **built-in tools**
(in-tree), and **channels** (in-tree). Skills are the right answer for
99% of "I want to add capability X" use cases.

## Skills

A skill is a directory under `<workspace>/skills/<name>/` with a
`SKILL.md` and an optional `tool.py`. Skills are loaded per-agent at
agent startup; reload requires a process restart.

### Markdown-only skill

The simplest skill is a SKILL.md describing a protocol the agent enacts
via existing built-in tools (`bash`, `read_file`, `write_file`,
`append_file`, `web_search`, etc.).

```markdown
---
name: morning-brief
description: Pull yesterday's calendar entries and summarize them as a morning briefing.
---

# Morning brief protocol

When invoked:

1. Read `~/calendar/<yesterday>.ics` via `read_file`.
2. Extract event titles and times.
3. Summarize as 3-5 bullet points.
4. Send the result via the appropriate channel.
```

The catalog injects only the `name` + `description`. The agent reads
the SKILL.md body on demand via `read_file` when it decides to invoke
the skill — keeps the per-turn system prompt small even with many
skills loaded.

### Skill with a Python tool

Add `tool.py` alongside `SKILL.md`:

```python
"""Calendar — list today's events."""

from datetime import date
from pathlib import Path

TOOL_SPEC = {
    "name": "calendar.list_today",
    "description": "Return today's calendar events as a markdown bullet list.",
    "input_schema": {
        "type": "object",
        "properties": {
            "tz": {
                "type": "string",
                "description": "IANA timezone, e.g. 'America/New_York'. Defaults to UTC.",
            },
        },
        "required": [],
    },
}


async def run(input: dict) -> str:
    tz = input.get("tz") or "UTC"
    today = date.today().isoformat()
    ics_path = Path.home() / "calendar" / f"{today}.ics"
    if not ics_path.is_file():
        return f"no calendar file for {today}"
    # ... parse + return ...
    return "- 09:00 standup\n- 14:00 design review"
```

Constraints enforced by the loader:

- `TOOL_SPEC` must be a dict with at least `name` (defaults to the
  skill directory name if missing).
- `run` must be `async def run(input: dict) -> str`. Synchronous
  functions are rejected at load.
- A failure (import error, malformed `TOOL_SPEC`, non-async `run`) logs
  the error and proceeds with the markdown-only catalog entry — the
  skill stays available, just without the Python tool.

The tool is registered under `TOOL_SPEC["name"]` (e.g. `calendar.list_today`)
and joins the agent's tool registry alongside the built-ins. The model
can call it like any other tool.

### Skill conventions

- One responsibility per skill. A `calendar` skill that does both
  reading and writing is fine; a kitchen-sink `productivity` skill
  isn't.
- Document inputs/outputs in SKILL.md plain text in addition to
  `input_schema`. The `input_schema` is for tool-call validation; the
  SKILL.md is for the model's reasoning about *when* to invoke.
- Keep tool runtime short. A long-running tool blocks the agent's tool
  loop. For long jobs, return a job id and have the agent poll via a
  separate tool.

## Built-in tools

Built-ins live in `claw/tools/` and are wired into every agent via
`build_registry()` in `claw/tools/__init__.py`. Add a built-in when:

- The capability is universal (every agent should have it, no skill
  customization needed).
- It needs claw-internal access not exposed to skills (e.g., session
  state, agent registry).

Pattern (matching `claw/tools/web_search.py`):

```python
"""my_tool — does X."""

from __future__ import annotations

from claw.tools.base import Tool


async def _run(input: dict) -> str:
    # ...
    return "result string"


def build_my_tool() -> Tool:
    return Tool(
        name="my_tool",
        description="Does X. Returns a markdown summary.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
        run=_run,
    )
```

Then wire into `claw/tools/__init__.py:build_registry`:

```python
from claw.tools.my_tool import build_my_tool

def build_registry(cfg, workspace_dir):
    tools = build_builtin_tools(workspace_dir)
    if cfg.searxng.base_url:
        ws = build_web_search_tool(cfg.searxng.base_url)
        tools[ws.name] = ws
    mt = build_my_tool()
    tools[mt.name] = mt
    return tools
```

If the tool has config, add a dataclass to `claw/config.py` and read
from `cfg`. Don't read environment variables or other config sources
inside tool implementations — `claw.yaml` is the single source of
truth.

### Config-gated tool families

Some tools should not exist for every agent, and some verbs should not exist
for every deployment. Two families ship gated this way (`cron_*`,
`bluetooth_*`), and the pattern is worth copying:

- **Gate the family by agent** with an `exposed_to: [agent-id]` list, rather
  than hardcoding names in the tool. Which agent owns a responsibility is a
  deployment's policy, not the code's.
- **Gate the individual verbs too**, when they differ in risk. "Who may act"
  and "what may be done" are separate questions, and collapsing them forces an
  operator to grant a destructive verb in order to get a harmless one.
- **Do not build a Tool for a disabled verb.** A model shown a tool it will
  always be refused will keep trying it, and will reason about capabilities it
  does not have. Absence is a clearer signal than a refusal.
- **Fail the config load on an unknown verb.** A typo that silently drops a
  capability the operator believed they had granted is worse than a crash at
  boot. Validate against the tool module's own list so the two cannot drift.
- **Strip the family on subagent spawn** if it is privileged, the way
  `subagent_*` and `cron_*` are stripped — otherwise a persona inherits a
  capability the deployment granted to a specific agent.

### Curation must outlive what produced it

A tool that builds an index over files the user owns will eventually want to
hold something the files do not say — a label, a correction, a judgement. The
moment it does, two kinds of data share a table: what a scan derived, and what
a person or an agent decided. Re-running the scan must not destroy the second
kind.

The instinct is a flag — a `curated_fields` list, or a per-field provenance
column, consulted before each write. Resist it. A rule that lives in an
`if` statement is a rule someone will refactor around. **Make the unsafe write
not exist:**

```sql
INSERT INTO tracks (...) VALUES (...)
ON CONFLICT(path) DO UPDATE SET
    duration_s = excluded.duration_s,
    lufs       = excluded.lufs,
    size = excluded.size, mtime = excluded.mtime
    -- Curated and seeded columns are simply ABSENT. Not guarded, not
    -- conditional: an UPDATE that never names a column cannot clobber it.
```

Three properties follow, none of which needs enforcing:

- **Draw the line at what is verifiable from the artifact itself.** A
  measurement is a fact about the bytes; everything else is an opinion, and
  opinions are curatable. If a measurement looks wrong, the artifact is wrong —
  that is information, not something to override.
- **Values seeded from the artifact are written on `INSERT` only.** They start
  as what the file said and become the operator's. Picking up a later change to
  the file is then an explicit act, not something a routine scan decides.
- **"Restore defaults" is a different statement, not a mode.** Delete and
  re-insert. Because it is a separate call site, no flag threaded through the
  scan path can reach it by accident.

The provenance question — *who* decided this, and when — is better answered by
an append-only edit log than by a column on every row. The log also records
what the value displaced, which a column cannot, and it cannot drift out of
step with the data because it is never updated.

#### Inheritance, and the `NOT NULL DEFAULT` trap

Curation usually wants levels: a label true of a whole group, overridable for
one member. **Model the levels as entities with ids**, each row carrying the
same curatable fields, and resolve in a view — most specific first — so every
reader goes through one definition of precedence:

```sql
COALESCE(item.mood, group.mood, supergroup.mood) AS r_mood
```

Ids rather than names as keys, even when the names look stable. Renaming a
group is then one row and cannot orphan an annotation. An earlier draft here
keyed on names, and needed a fan-out across every child row plus a re-keying
pass to keep annotations attached — both deleted by the change to ids.

Two consequences worth planning for. Renaming onto a name that **already
exists** should merge rather than fail, because "the same thing is filed twice"
is the usual reason to rename; decide up front whose annotations win (the
survivor's) and what happens to one only the loser had (carry it — losing an
annotation is the worst outcome of tidying up). And a value that belongs to the
group should be **written only at the group level**: if a scan also writes it
onto each child "when it differs", a child added after a correction will carry
the file's value and silently shadow the correction — the exact defect levels
were introduced to fix.

⚠ **Every inheritable column must be nullable.** `NULL` is "nobody said", and
that is the whole mechanism. A column declared `NOT NULL DEFAULT 0` stores a
literal 0 on every row, `COALESCE` stops at the first argument, and the levels
above become unreachable — silently, because the value it returns is perfectly
plausible. A column default becomes an *assertion* the moment inheritance is
put underneath it.

A related trap when a column holds JSON: `json.dumps(None)` is the string
`"null"`, which is neither SQL NULL nor falsy. Clearing such a field writes a
value that reads as set and then decodes to `None` further downstream. Encode
only when the value is not `None`.

### Measure at ingest, apply per item

Metadata that ships inside a file is written by whoever made it, to no
particular standard, and is frequently absent. When a property is *derivable
from the bytes*, deriving it once at ingest is better than trusting a tag:
measured values are reproducible, comparable across a whole collection, and can
be re-derived when the way they are computed changes — keep a `measure_version`
alongside them so bumping it re-opens everything.

Prefer a **hard identity check** to decide whether a file needs re-opening —
size, mtime and measure version together. On a large collection this is the
difference between a sweep costing minutes and costing seconds, and it makes
"ingest one file" and "ingest everything" the same code path.

A measurement is worth taking even when you end up not acting on it. The one
that prompted this section was loudness, and the measurement's most valuable
result was showing that the normalisation it was gathered for should be left
**off**: peaks already matched across the collection, so normalising by average
loudness would have created a mismatch that doing nothing did not have. The
numbers still earn their place — they identify outliers, and they feed a
different feature entirely. Measuring is cheap; applying is the decision.

Applying a per-item value at playback or render time raises a separate
question: how does the value follow the item without something watching a
cursor? If the underlying player takes **per-item options at queue time**, use
them. mpv's `loadfile` accepts a fourth argument of file-local options
(`volume=42`), applied for that entry only and dropped when it ends. The
alternative — setting a property on the player and watching for item changes —
means a long-lived connection draining an event stream, and a race between the
change arriving and the value being set.

### Tools that need a GUI session

A tool that shells out to a host utility can behave differently depending on
the launchd context it runs in, and the difference is not always an error.

Measured on macOS 26 with `blueutil`: called from a `LaunchDaemon`, `--power`
and `--paired` answer correctly, while `--inquiry` returns **empty with exit
0**. A device that a session-context scan finds on its first sweep stayed
invisible across four minutes of daemon-context sweeps. Discovery and pairing
are per-user operations in `bluetoothd`, and with no console user it reports
their absence as "nothing there" rather than as a failure.

That shape — right answers for some calls, plausible wrong answers for others,
no error anywhere — is the one to design against. `claw/tools/bluetooth.py`
routes **every** verb through the console session rather than splitting by
verb, because a split encodes a belief about which calls are session-scoped
that no runtime check can verify, and the failure mode of getting it wrong is
silent.

The routing itself needs no privilege: launchd lets a process manage its own
user's GUI domain, so a gateway running as the console user can bootstrap a
one-shot job into `gui/<uid>` with no sudo and no helper daemon. Two details
matter. Exit status must travel back through a **sentinel file** — a one-shot
job is torn down the moment it exits, so "the job is gone" and "the job never
started" are otherwise indistinguishable from outside. And the dependency is
now the console session itself, so its absence should be reported as what it
is, rather than surfacing as a puzzling failure of the underlying feature.

### Tool result spooling

Built-in tool implementations don't need to think about result size.
The Ollama loop (`claw/ollama.py`) intercepts results larger than 8 KB
and spools them to `<workspace>/.tool-results/<sid>/<call_id>.txt` —
the model sees the full content during the in-loop run, but persisted
transcripts get a truncated preview + the spool path. `read_file`
specifically skips this spooling so the agent can re-read its own
spooled results.

Workspace cleanup happens at the daily session rotate (per-workspace
sweep, drops anything >24h old).

## Channels

A channel is an inbound surface that satisfies the `Channel` protocol
in `claw/channel/base.py`:

```python
class Channel(Protocol):
    name: str
    async def start(self, on_message: InboundHandler) -> None: ...
    async def send(self, peer_id: str, text: str) -> None: ...
    async def shutdown(self) -> None: ...
    def typing(self, peer_id: str) -> AsyncContextManager[None]: ...
```

`InboundMessage(peer_id, sender_name, text, channel)` is the structure
your channel hands to the agent's inbound handler.

Adding a channel today is more involved than adding a skill or tool —
the agent only knows about Matrix, the per-agent matrix block in
`claw.yaml` is wired through `_parse_agent`, and `main.py` constructs
matrix clients explicitly. Adding (say) Slack would mean:

1. New module `claw/channel/slack.py` implementing the protocol.
2. New per-agent config block (e.g., `slack:`) parsed in
   `_parse_agent` (`claw/config.py`).
3. Boot wiring in `main.py` — instantiate the channel client per agent
   that has the block, call `channel.start(agent.handle_inbound)`.
4. Outbound delivery routing — currently `agent` writes back to the
   channel that delivered the inbound. For multi-channel agents you may
   want explicit routing (e.g., a `deliver_to` hint).

For one-off integrations (push a single notification on demand), prefer
a skill that calls a webhook over a full channel implementation.

## What not to do

- **Don't read agent state from inside a tool.** The tool's `input` is
  the contract. Side-channels through globals or filesystem state
  outside the workspace make tools non-portable across agents.
- **Don't hardcode agent ids in tools or channels.** If a tool only
  applies to certain agents, gate via config (see `cron.exposed_to` for
  the established pattern).
- **Don't write outside the workspace.** Tool implementations can
  technically write anywhere the claw process has permission to, but by
  convention they stay inside `workspace_dir`. Memory, transcripts,
  and skill state all live there. Operators expect to be able to
  archive/move a workspace as a unit.
