"""Explicit memory retrieval tool — wraps MemoryIndex.retrieve_markdown.

Auto-retrieval already prepends a relevance block to every turn. This tool
exists for cases where the agent needs to query for something the
auto-retrieval didn't surface — a follow-up search, a different framing,
or a deeper top_n.
"""

from __future__ import annotations

import logging
from typing import Any

from claw.memory import MemoryIndex
from claw.tools.base import Tool

log = logging.getLogger("claw.tools.memory_search")


def build_memory_search_tool(memory: MemoryIndex) -> Tool:
    async def _run(args: dict[str, Any]) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return "error: query is required"
        top_n = max(1, min(int(args.get("top_n", 5)), 20))
        compact = bool(args.get("compact", True))
        try:
            result = await memory.retrieve_markdown(query, top_n=top_n, compact=compact)
        except Exception as e:
            log.exception("memory_search failed")
            return f"error: {e}"
        return result or "(no matches)"

    return Tool(
        name="memory_search",
        description=(
            "Search your own memory index (ChromaDB + BM25 + graph) for relevant "
            "chunks. Returns formatted markdown with the top matches and any "
            "linked graph neighbors. Auto-retrieval already runs every turn, so "
            "use this only for follow-up queries the auto-block didn't cover, "
            "or when you want a higher top_n."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language query."},
                "top_n": {
                    "type": "integer",
                    "description": "Number of chunks to return (1-20; default 5).",
                },
                "compact": {
                    "type": "boolean",
                    "description": "If true (default), return a condensed view; false for fuller chunks.",
                },
            },
            "required": ["query"],
        },
        run=_run,
    )
