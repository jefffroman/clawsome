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
