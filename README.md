<img src="https://raw.githubusercontent.com/jefffroman/clawsome/2026.9.20/clawsome-logo.png" alt="Clawsome" width="80" align="left">

# Clawsome
A lean python harness for running persistent AI agents.

> Clawsome is under early, active development. Expect rough edges, breaking changes, and missing features. Feedback and contributions are welcome.

<br clear="left">


## Setup

Clawsome talks to a few external services (Ollama, SearXNG, a Matrix
homeserver) — get those running first, then `pip install -e .` and
point claw at a populated `claw.yaml`.

→ See [`docs/setup.md`](https://github.com/jefffroman/clawsome/blob/2026.9.20/docs/setup.md) for prerequisites, install notes
(including why the `matrix-nio[e2e]` extra is mandatory), and a
first-boot checklist.

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

→ See [`docs/architecture.md`](https://github.com/jefffroman/clawsome/blob/2026.9.20/docs/architecture.md) for the full mental
model, request lifecycle, and workspace file contract.

Optionally, a **System One decision service** (typed choices with
probabilities, no generated text) lets a **decision gate** answer simple,
command-like turns — "pause the music", "what's playing?" — with a direct tool
action in well under a second instead of a full LLM turn. Clawsome runs the
same with or without one.

→ See [`docs/decisions.md`](https://github.com/jefffroman/clawsome/blob/2026.9.20/docs/decisions.md).

## Configuration

`claw.yaml` is parsed once at startup; editing requires a restart. The
shipped `claw.example.yaml` is a runnable template with inline comments.

→ See [`docs/configuration.md`](https://github.com/jefffroman/clawsome/blob/2026.9.20/docs/configuration.md) for the full key-by-key
reference.

## Audio

Bluetooth and music: the host requirements that bite before either works, the
DJ run sheet, why search and candidates have opposite contracts, output
routing, and how playback loudness is decided.

→ See [`docs/audio.md`](https://github.com/jefffroman/clawsome/blob/2026.9.20/docs/audio.md).

## Operations

Background tuning, daily session rotate, memory retrieval cadence, Matrix
bot first-deploy (token + cross-signing UIA + ghost-DM avoidance), the
admin-command runbook (`%stop`/`%compact`/`%clear`/`%subagents`/
`%verbose`/`%thinking`), and log-based troubleshooting.

→ See [`docs/operations.md`](https://github.com/jefffroman/clawsome/blob/2026.9.20/docs/operations.md).

## Extending

Four extension points: **skills** (workspace-local; markdown protocol +
optional `tool.py`), **built-in tools** (in-tree; universal capability),
**gate handlers** (config-only; answer a fixed, command-like turn without the
LLM), **channels** (in-tree; new inbound surface). Skills cover 99% of cases.

→ See [`docs/extending.md`](https://github.com/jefffroman/clawsome/blob/2026.9.20/docs/extending.md).

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
