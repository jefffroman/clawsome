"""Tool dataclass + Ollama tool-spec adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    run: Callable[[dict[str, Any]], Awaitable[str]]
    # Optional structured twin of ``run`` for code callers (gate reply
    # functions), never shown to the model. ``run``'s text is written for the
    # model and may be reworded for it at any time; code that needs fields
    # reads them here instead of parsing that prose.
    data: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None

    def as_ollama_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


def ollama_tool_spec(registry: dict[str, Tool]) -> list[dict[str, Any]]:
    return [t.as_ollama_tool() for t in registry.values()]
