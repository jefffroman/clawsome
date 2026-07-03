"""Explicit memory retrieval tool — wraps MemoryIndex.retrieve_markdown.

Auto-retrieval already prepends a relevance block to every turn. This tool
exists for cases where the agent needs to query for something the
auto-retrieval didn't surface — a follow-up search, a different framing,
or a deeper top_n.
"""

from __future__ import annotations

import asyncio
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


def build_curator_search_tool(memory: MemoryIndex, today: str) -> Tool:
    """Curator-only variant of ``memory_search``. The agent-facing tool returns
    the retrieval block with markers stripped (ids/ts hidden); the curator
    instead needs the marker metadata — ``id`` (to write ``supersededBy``
    pointers), ``supersededBy`` (to spot existing chains), and the source file
    — for any memory in the WHOLE corpus, any age. So this exposes them inline.

    Today's note is filtered out: it is off-limits to the curator AND its
    memories are seeded by the collector with ``ts`` only (no ``id``), so they
    can be neither edited nor referenced as a supersession target. Surfacing
    them would only tempt a bad cross-day dedup/supersession — tomorrow's pass
    handles today once it has ids. Read-only; not audit-wrapped. Registered
    under the same name so it overrides the agent tool in the curator's set."""
    today_src = f"memory/{today}.md"

    async def _run(args: dict[str, Any]) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return "error: query is required"
        n = max(1, min(int(args.get("top_n", 8)), 20))
        try:
            loop = asyncio.get_running_loop()
            # Hold the index lock for the read — same as retrieve_markdown — so
            # the sidecar reads (bm25/graph) stay consistent against the
            # concurrent maintenance-loop reindex, which can fire mid-curation.
            async with memory.lock:
                hits = await loop.run_in_executor(None, memory.search, query, n)
        except Exception as e:
            log.exception("curator search failed")
            return f"error: {e}"
        lines: list[str] = []
        for h in hits:
            src = h["id"].rsplit(":", 1)[0]
            if src == today_src:
                continue  # today's note: un-editable + un-referenceable here
            mid = h.get("mem_id") or "—"
            sup = f" supersededBy={h['superseded_by']}" if h.get("superseded_by") else ""
            cur = " [current head]" if h.get("_current") else ""
            lines.append(
                f"- [{src}] **{h['section']}** (id={mid}{sup}){cur}: {h['text'][:240]}"
            )
        return "\n".join(lines) if lines else "(no matches)"

    return Tool(
        name="memory_search",
        description=(
            "Search the whole memory index (BM25 + vector + RRF) for memories "
            "similar to a query — any age, any file. Returns each match with its "
            "source file, section, marker `id`, and any `supersededBy` pointer, "
            "so you can find supersession/dedup candidates beyond the file in "
            "front of you and reference them by id."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language query."},
                "top_n": {
                    "type": "integer",
                    "description": "Number of matches to return (1-20; default 8).",
                },
            },
            "required": ["query"],
        },
        run=_run,
    )
