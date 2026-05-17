<img src="https://raw.githubusercontent.com/jefffroman/clawsome/2026.5.17/clawsome-logo.png" alt="Clawsome" width="80" align="left">

# Clawsome
A lean python harness for running persistent AI agents.

> Clawsome is under early, active development. Expect rough edges, breaking changes, and missing features. Feedback and contributions are welcome.

<br clear="left">


## Setup

Clawsome talks to a few external services (Ollama, SearXNG, a Matrix
homeserver) — get those running first, then `pip install -e .` and
point claw at a populated `claw.yaml`.

→ See [`docs/setup.md`](https://github.com/jefffroman/clawsome/blob/2026.5.17/docs/setup.md) for prerequisites, install notes
(including the macOS libolm wrinkle), and a first-boot checklist.

## Architecture

A single clawsome process manages **multiple agents**, each with a **workspace** (state on
disk) and **one or more channels** (inbound surfaces — Matrix is the shipped
one). Per-turn, the agent loads its transcript, assembles a system prompt
from injected identity files + retrieved memory, runs an Ollama tool loop,
and replies. Background tasks (compaction, memory_flush, reindex, cron,
session rotate) run **off** the user-reply path. An optional
**control plane** lets allowlisted operators run in-band admin commands
(`%stop`, `%compact`, `%clear`, …) that are intercepted before the
model and never enter the transcript.

→ See [`docs/architecture.md`](https://github.com/jefffroman/clawsome/blob/2026.5.17/docs/architecture.md) for the full mental
model, request lifecycle, and workspace file contract.

## Configuration

`claw.yaml` is parsed once at startup; editing requires a restart. The
shipped `claw.example.yaml` is a runnable template with inline comments.

→ See [`docs/configuration.md`](https://github.com/jefffroman/clawsome/blob/2026.5.17/docs/configuration.md) for the full key-by-key
reference.

## Operations

Background tuning, daily session rotate, memory retrieval cadence, Matrix
bot first-deploy (token + cross-signing UIA + ghost-DM avoidance), the
admin-command runbook (`%stop`/`%compact`/`%clear`/`%subagents`/
`%verbose`/`%thinking`), and log-based troubleshooting.

→ See [`docs/operations.md`](https://github.com/jefffroman/clawsome/blob/2026.5.17/docs/operations.md).

## Extending

Three extension points: **skills** (workspace-local; markdown protocol +
optional `tool.py`), **built-in tools** (in-tree; universal capability),
**channels** (in-tree; new inbound surface). Skills cover 99% of cases.

→ See [`docs/extending.md`](https://github.com/jefffroman/clawsome/blob/2026.5.17/docs/extending.md).

## Deployment

`pip install` exposes a `claw` console script that takes
`--config <path/to/claw.yaml>` and runs in the foreground. Wire it into
launchd / systemd / your process manager of choice as a long-lived service
running under a dedicated user. This repo intentionally does not ship
example service files — paths, log destinations, and label/unit names all
depend on the host layout.

---

## Contributing

Issues and pull requests are welcome on [GitHub](https://github.com/jefffroman/clawsome).

### Tests

```bash
pip install -e ".[dev]"   # pytest + pytest-asyncio
pytest -q
```

Pure-logic unit tests plus one `Agent` smoke test — no live Ollama,
Matrix, or network needed (fakes + `tmp_path` throughout); runs in
under a second. Tests use neutral placeholders only (no real hostnames
or MXIDs) — please keep new ones that way.
