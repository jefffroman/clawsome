# Setup

Clawsome is an orchestrator. It does **not** ship inference, search, or
messaging — it talks to existing local services. Stand those up first,
then point claw at them via `claw.yaml`.

## Required external services

### Ollama

Used for every model claw runs against — primary inference per agent,
background compaction + memory_flush turns, and subagent personas.
Installed and managed independently.

- Default bind: `http://127.0.0.1:11434`. Pointed at by `ollama.base_url`.
- Every model tag referenced in `claw.yaml` must already be pulled —
  `ollama pull <tag>`. The referenced spots:
  - `agents[*].primary_model`
  - `agents[*].compaction_model` (per-agent override; optional)
  - `ollama.default_compaction_model` (fallback for the above)
  - `subagents.personas[*].model`
  - `subagents.default_model`
- Tune the daemon's `OLLAMA_NUM_PARALLEL`, `OLLAMA_KV_CACHE_TYPE`,
  `OLLAMA_CONTEXT_LENGTH`, `OLLAMA_KEEP_ALIVE`,
  `OLLAMA_MAX_LOADED_MODELS` to match your hardware and the
  context-window math in `docs/operations.md` (the compaction +
  memory_flush thresholds in `claw.yaml` are calibrated against
  `OLLAMA_CONTEXT_LENGTH`).

Quick check:

```bash
curl -s http://127.0.0.1:11434/api/tags | jq '.models[].name'
```

### SearXNG

Backs the `web_search` built-in tool.

- Default bind: `http://127.0.0.1:8080`. Pointed at by `searxng.base_url`.
- The instance **must have JSON output enabled** — `web_search` calls
  `<base_url>/search?q=...&format=json`. In `settings.yml`, ensure
  `search.formats:` includes `json` (upstream default ships HTML-only).

Quick check:

```bash
curl -s "http://127.0.0.1:8080/search?q=hello&format=json" | jq '.results | length'
```

### Matrix homeserver

Used by every agent's Matrix channel. One bot user per agent.

- Tested against Synapse.
- The `agents[*].matrix.homeserver` URL is *where to reach* the server
  (e.g., `http://127.0.0.1:6167` for a same-host Synapse). It is
  independent of the server-name portion of `agents[*].matrix.user_id`,
  which must match the homeserver's `server_name` setting (Synapse:
  `server_name:` in `homeserver.yaml`). The two values rarely look
  alike — the URL is a network address, the server-name is the bot's
  identity domain.
- Each bot needs a unique `device_id` per `user_id` for fresh crypto
  state. Reusing an existing device id binds to that device's existing
  Olm sessions on the homeserver.
- Token minting, cross-signing bootstrap, and allowlist semantics are
  covered in detail in `docs/operations.md` *Matrix bot first-deploy*.

## Python runtime

- Python ≥ 3.12.
- Install: `pip install -e .` from the clawsome repo root.
- The dependency that typically dominates install time is
  `matrix-nio[e2e]`, which pulls in `python-olm`, which builds `libolm`
  from source on platforms without wheels.

### macOS install

`python-olm` has no macOS wheels at time of writing, and the bundled
libolm needs adjustments for current cmake + Apple Clang:

1. `brew install libolm` — provides the system libolm headers.
2. Set `CMAKE_POLICY_VERSION_MINIMUM=3.5` in the install env so the
   bundled cmake config accepts current cmake.
3. The bundled `libolm/include/olm/list.hh` may need a const-qualifier
   fix to compile under current Apple Clang. If `pip install` fails
   inside the python-olm build, check the python-olm and libolm GitHub
   issue trackers — the exact patch tracks upstream.

A typical incantation:

```bash
brew install libolm
CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install -e .
```

### Linux install

Install the libolm development package via your distro
(`apt install libolm-dev`, `dnf install libolm-devel`, etc.), then
`pip install -e .`. No additional patching usually required.

## Workspace prep

For each agent, the path under `agents[*].workspace` must exist and be
writable by the user running claw **before first boot**. Subdirectories
(`memory/`, `transcripts/`, `.memory/`, `.matrix-store/`,
`.tool-results/`) are created on demand. Identity files
(`IDENTITY.md`, `USER.md`, `SOUL.md`, `AGENTS.md`, `TOOLS.md`,
`MEMORY.md`) are authored by you — claw doesn't generate them, just
reads them. The disjointness rule in `docs/architecture.md` (workspace
contract) is the guide for which files go where.

## First-boot checklist

- [ ] Ollama running. Every model tag referenced in `claw.yaml` pulled
      and warm.
- [ ] SearXNG running with JSON output enabled.
- [ ] Matrix homeserver running. Bot users registered (one per agent).
      **Human users registered too** — clawsome ships no HTTP plane
      and no self-service signup; every operator who'll talk to the
      bots needs an account on the homeserver, created out-of-band
      (Synapse: `register_new_matrix_user`). Their MXIDs go in
      `agents[*].matrix.allow_from`. Access tokens for the bots minted
      and saved to `access_token_file` paths (mode 0600). Password
      files saved (mode 0600) for cross-signing.
- [ ] `claw.yaml` populated. Every `*_file:` and path key resolves to
      an existing path or a writable parent.
- [ ] Each agent's `workspace` directory exists and is writable.
- [ ] `pip install -e .` complete; `claw --help` runs.
- [ ] First boot: run `claw --config <path>` in the foreground. Watch
      for `claw.main INFO ready (N agents)`. On first boot per agent,
      cross-signing bootstrap runs against the homeserver's UIA
      endpoint — expect a brief delay before the agent is interactive.

Once steady-state, wire `claw --config <path>` into systemd / launchd /
your service manager of choice. Operational tuning (compaction
thresholds, retrieval cadence, Matrix allowlists, troubleshooting) is
covered in `docs/operations.md`.
