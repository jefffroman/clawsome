"""SearXNG-backed web_search tool."""

from __future__ import annotations

import functools
import logging
import urllib.parse
from pathlib import Path
from typing import Any

import httpx

from claw.tools.base import Tool
from claw.tools.spool import bound_result

log = logging.getLogger("claw.tools.web_search")


async def _run_web_search(
    searxng_url: str, workspace_dir: Path, args: dict[str, Any],
) -> str:
    query = (args.get("query") or "").strip()
    if not query:
        return "error: query is required"
    n = max(1, min(int(args.get("num", 10)), 30))
    category = args.get("category", "general")

    qs = urllib.parse.urlencode({"q": query, "format": "json", "categories": category})
    url = searxng_url.rstrip("/") + "/search?" + qs

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            payload = resp.json()
    except Exception as e:
        return f"error: searxng request failed: {e}"

    results = (payload.get("results") or [])[:n]
    if not results:
        return "(no results)"
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or "(untitled)"
        href = r.get("url") or ""
        snippet = (r.get("content") or "").strip()
        lines.append(f"{i}. {title}\n   {href}\n   {snippet}")
    # Snippet length is SearXNG's call, not ours — a broad query across 30
    # results can run long, so bound it like any other incidental output.
    return bound_result(
        "\n".join(lines), workspace_dir=workspace_dir, tool="web_search",
    )


def build_web_search_tool(searxng_url: str, workspace_dir: Path) -> Tool:
    return Tool(
        name="web_search",
        description=(
            "Search the public web via the local SearXNG endpoint. "
            "Returns up to N results with title, URL, and snippet."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "num": {"type": "integer", "description": "Number of results (1-30; default 10)."},
                "category": {
                    "type": "string",
                    "description": "SearXNG category (general, news, images, videos, science, ...).",
                },
            },
            "required": ["query"],
        },
        run=functools.partial(_run_web_search, searxng_url, workspace_dir),
    )
