"""Subagent spawning — persona-based one-shot agent forks.

Personas are configured in ``claw.yaml`` under ``subagents.personas:``
(case-insensitive lookup). Each spawn creates a transient ``Agent``
sharing the parent's workspace, tools (minus ``spawn_subagent`` and
``cron_*``), and memory. Transcripts are NOT persisted — a subagent's
job is one-shot.

Spawn permission is per-agent: each agent carries a
``remaining_spawn_budget`` (top-level agents seed it from
``AgentConfig.max_spawn_depth``; forks compute
``min(parent_remaining - 1, persona.max_spawn_depth)``). A hardcoded
``ABSOLUTE_MAX_CHAIN_DEPTH`` is the runtime safety net.

A single ``SubagentSpawner`` instance lives in the gateway process and
enforces global ``max_concurrent`` and per-parent
``max_children_per_agent`` semantics across all simultaneous spawns.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from claw.config import SubagentsConfig
from claw.tools.base import Tool

if TYPE_CHECKING:
    from claw.agent import Agent

log = logging.getLogger("claw.tools.subagent")

# Absolute chain-depth ceiling. Per-agent budgets handle the common case;
# this is a runtime safety so a buggy config can't cause runaway recursion.
ABSOLUTE_MAX_CHAIN_DEPTH = 5


class SubagentSpawner:
    def __init__(self, cfg: SubagentsConfig) -> None:
        self.cfg = cfg
        self.semaphore = asyncio.Semaphore(cfg.max_concurrent)
        # parent_id -> currently-running child count
        self.children_by_parent: dict[str, int] = {}

    async def spawn(
        self,
        parent: "Agent",
        persona: str,
        prompt: str,
        depth: int,
    ) -> str:
        """Spawn a subagent of ``persona`` to execute ``prompt`` once.
        Returns the subagent's final reply text (or an error string).
        """
        if depth > ABSOLUTE_MAX_CHAIN_DEPTH:
            return (
                f"error: subagent chain hit absolute safety ceiling "
                f"(depth={depth}, max={ABSOLUTE_MAX_CHAIN_DEPTH})"
            )

        persona_key = persona.lower()
        persona_cfg = self.cfg.personas.get(persona_key)
        if persona_cfg is None:
            available = ", ".join(sorted(self.cfg.personas.keys())) or "(none configured)"
            return f"error: unknown persona {persona!r}; available: {available}"

        # Per-parent allowlist enforcement (defense in depth — the tool's
        # persona schema also lists only allowed entries).
        allowed = parent.allowed_spawn_personas
        if allowed is not None and persona_key not in allowed:
            allowed_str = ", ".join(sorted(allowed)) or "(none)"
            return (
                f"error: parent {parent.id!r} is not allowed to spawn "
                f"persona {persona_key!r}; allowed: {allowed_str}"
            )

        count = self.children_by_parent.get(parent.id, 0)
        if count >= self.cfg.max_children_per_agent:
            return (
                f"error: parent {parent.id} has hit max_children_per_agent="
                f"{self.cfg.max_children_per_agent} concurrent children"
            )

        async with self.semaphore:
            self.children_by_parent[parent.id] = count + 1
            log.info(
                "[%s] spawning subagent persona=%s depth=%d model=%s",
                parent.id, persona_key, depth, persona_cfg.model,
            )
            try:
                child = parent.fork(persona_key)
                return await child.run_one_shot(prompt)
            except Exception as e:
                log.exception("[%s] subagent persona=%s raised", parent.id, persona_key)
                return f"error: subagent failed: {e}"
            finally:
                self.children_by_parent[parent.id] = max(
                    0, self.children_by_parent.get(parent.id, 1) - 1,
                )


def build_subagent_tool(
    parent: "Agent",
    spawner: SubagentSpawner,
    depth: int,
) -> Tool:
    """Build a ``spawn_subagent`` Tool bound to a specific parent agent and
    spawn depth. The tool's closure captures both, so each fork gets its
    own correctly-scoped tool. The persona schema lists only the personas
    the parent is actually allowed to spawn.
    """
    all_keys = set(spawner.cfg.personas.keys())
    if parent.allowed_spawn_personas is None:
        persona_keys = sorted(all_keys)
    else:
        persona_keys = sorted(all_keys & set(parent.allowed_spawn_personas))
    persona_list = ", ".join(persona_keys) if persona_keys else "(none configured)"

    async def _run(args: dict[str, Any]) -> str:
        prompt = (args.get("prompt") or "").strip()
        persona = (args.get("persona") or "").strip()
        if not prompt:
            return "error: prompt is required"
        if not persona:
            return "error: persona is required"
        return await spawner.spawn(parent, persona, prompt, depth + 1)

    return Tool(
        name="spawn_subagent",
        description=(
            "Spawn a subagent persona for a one-shot task. The child runs "
            "against its persona's configured model and shares your workspace, "
            "tools, and memory. Persona is case-insensitive. Returns the "
            "child's final reply. Available personas: " + persona_list
        ),
        input_schema={
            "type": "object",
            "properties": {
                "persona": {
                    "type": "string",
                    "description": (
                        "Persona name (case-insensitive). One of: "
                        + (", ".join(persona_keys) if persona_keys else "(none)")
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "What you want the subagent to do. Be specific — "
                        "they have no other context beyond the workspace."
                    ),
                },
            },
            "required": ["persona", "prompt"],
        },
        run=_run,
    )
