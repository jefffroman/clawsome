"""In-process memory retrieval — multi-agent.

ChromaDB + BM25 + NetworkX hybrid retrieval with RRF fusion. Source files live
under ``<workspace>/`` (top-level: MEMORY.md only) and ``<workspace>/memory/``
(daily notes ``YYYY-MM-DD.md`` per the bifurcation contract). Per-agent
ChromaDB persists at ``<workspace>/.memory/``.

The module owns indexes per-agent: ``Agent.handle_inbound`` calls
``retrieve_markdown`` on every turn before assembling the prompt. No
separate retrieval service — collapsing the boundary into a function call
keeps deployment minimal and removes a moving part.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
import networkx as nx
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

log = logging.getLogger("claw.memory")

MODEL_NAME = "all-MiniLM-L6-v2"
RRF_K = 60

# Calibrated against MiniLM-L6-v2 distance distributions: topical hits cluster
# at 1.41-1.44, weak/gibberish queries plateau at 1.50+. Calibrated against
# an active agent's index.
VECTOR_DISTANCE_MAX = 1.50

EXCLUDED_SECTION_PATTERNS = re.compile(
    r"^(Auto-Retrieved Memory Context|Conversation Summary|Hybrid Search|Knowledge Graph)\b"
)

DAILY_NOTE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")

# Indexing is disjoint from prompt injection (extra_paths in claw.yaml).
# Files the model already sees verbatim every turn (IDENTITY/USER/SOUL/AGENTS/
# TOOLS) shouldn't compete for retrieval slots — only large/growing content
# belongs here. MEMORY.md is the boundary case; daily notes scale.
ROOT_SOURCES = ("MEMORY.md",)


class MemoryIndex:
    """One per agent. Owns its own ChromaDB collection plus BM25/graph sidecars."""

    def __init__(self, agent_id: str, workspace_dir: Path) -> None:
        self.agent_id = agent_id
        self.workspace_dir = Path(workspace_dir)
        self.data_dir = self.workspace_dir / ".memory"
        self.embedder: Any | None = None
        self.chroma_client: chromadb.api.ClientAPI | None = None
        # Lock prevents reindex from racing concurrent retrieval reads of the
        # JSON sidecars (bm25_corpus.json, memory_graph.json).
        self.lock = asyncio.Lock()

    # --- lifecycle ----------------------------------------------------------

    def warmup(self) -> None:
        log.info("[%s] loading embedding function (%s via ONNX)", self.agent_id, MODEL_NAME)
        self.embedder = embedding_functions.DefaultEmbeddingFunction()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(self.data_dir / "chroma_db")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        log.info("[%s] chromadb ready at %s", self.agent_id, db_path)

    async def warmup_async(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.warmup)

    # --- source collection --------------------------------------------------

    def _daily_notes(self) -> list[Path]:
        return [
            p for p in sorted(self.workspace_dir.glob("memory/*.md"))
            if DAILY_NOTE_PATTERN.match(p.name)
        ]

    def _source_paths(self) -> list[Path]:
        paths: list[Path] = []
        for name in ROOT_SOURCES:
            p = self.workspace_dir / name
            if p.exists():
                paths.append(p)
        paths.extend(self._daily_notes())
        return paths

    def _source_key(self, path: Path) -> str:
        """Stable key for a source path — matches the ``metadata.source``
        strings emitted by ``_collect_sources`` so chunks can be grouped/
        deleted by source consistently."""
        if path.parent == self.workspace_dir:
            return path.name
        return f"memory/{path.name}"

    def _compute_source_hashes(self) -> dict[str, str]:
        """Per-file SHA1 over current source bytes. Per-file (not global)
        so a single daily-note append doesn't invalidate every other file."""
        out: dict[str, str] = {}
        for path in self._source_paths():
            try:
                out[self._source_key(path)] = hashlib.sha1(path.read_bytes()).hexdigest()
            except FileNotFoundError:
                pass
        return out

    @staticmethod
    def _parse_markdown(path: Path) -> list[dict[str, Any]]:
        content = path.read_text()
        chunks: list[dict[str, Any]] = []
        sections = re.split(r"(^##\s+.*$)", content, flags=re.MULTILINE)
        if sections[0].strip():
            chunks.append({"content": sections[0].strip(), "metadata": {"section": "Intro"}})
        for i in range(1, len(sections), 2):
            header = sections[i].strip().lstrip("#").strip()
            body = sections[i + 1].strip() if i + 1 < len(sections) else ""
            if body and not EXCLUDED_SECTION_PATTERNS.match(header):
                chunks.append({"content": body, "metadata": {"section": header}})
        return chunks

    def _collect_sources(self) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for fname in ROOT_SOURCES:
            fpath = self.workspace_dir / fname
            if fpath.exists():
                for c in self._parse_markdown(fpath):
                    c["metadata"]["source"] = fname
                    chunks.append(c)
        for fpath in self._daily_notes():
            for c in self._parse_markdown(fpath):
                c["metadata"]["source"] = f"memory/{fpath.name}"
                chunks.append(c)
        return chunks

    # --- indexing -----------------------------------------------------------

    @staticmethod
    def _build_graph(chunks: list[dict[str, Any]]) -> nx.DiGraph:
        G: nx.DiGraph = nx.DiGraph()
        for c in chunks:
            G.add_node(c["metadata"]["section"], type="section")
            for concept in re.findall(r"\*\*(.*?)\*\*", c["content"]):
                if 3 <= len(concept) <= 50:
                    G.add_node(concept, type="concept")
                    G.add_edge(c["metadata"]["section"], concept, relation="contains")
        nodes = set(G.nodes())
        for c in chunks:
            for target in nodes:
                if target != c["metadata"]["section"] and target in c["content"]:
                    G.add_edge(c["metadata"]["section"], target, relation="mentions")
        return G

    @staticmethod
    def _atomic_write_json(path: Path, obj: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)

    def _load_prev_hashes(self) -> dict[str, str] | None:
        """Returns the per-file hash map from sync_state.json, or None if
        the state file is missing, malformed, or in the legacy single-
        ``sourcesHash`` format. Caller treats None as "everything changed"
        and triggers a full rebuild that upgrades the file."""
        state_path = self.data_dir / "sync_state.json"
        if not state_path.exists():
            return None
        try:
            with open(state_path) as f:
                state = json.load(f)
        except Exception:
            return None
        prev = state.get("sourceHashes")
        if not isinstance(prev, dict):
            return None
        return prev

    def needs_reindex(self) -> tuple[list[str], list[str]]:
        """Returns (changed, removed) source-key lists. Empty tuple
        ``([], [])`` means in-sync. A None ``prev_hashes`` (missing/legacy
        state) returns ``(all_current_sources, [])`` so the caller takes
        the full-rebuild path and upgrades the state file."""
        current = self._compute_source_hashes()
        if not current:
            return ([], [])
        prev = self._load_prev_hashes()
        if prev is None:
            return (sorted(current.keys()), [])
        changed = sorted(k for k, v in current.items() if prev.get(k) != v)
        removed = sorted(k for k in prev.keys() if k not in current)
        return (changed, removed)

    @staticmethod
    def _assign_chunk_ids(chunks: list[dict[str, Any]]) -> list[str]:
        """Per-source enumeration: ``{source}:{i}`` where i is local to
        that source. Stable across reindexes — adding a section to one
        file doesn't shift any other file's ids — which makes the
        incremental path's surviving-chunk ids match the freshly-built
        BM25 corpus ids."""
        ids: list[str] = []
        per_source_idx: dict[str, int] = {}
        for c in chunks:
            src = c["metadata"]["source"]
            i = per_source_idx.get(src, 0)
            ids.append(f"{src}:{i}")
            per_source_idx[src] = i + 1
        return ids

    def do_reindex(self, changed: list[str], removed: list[str]) -> dict[str, Any]:
        assert self.chroma_client is not None and self.embedder is not None, "warmup() not called"
        chunks = self._collect_sources()
        if not chunks:
            return {"status": "skipped", "reason": "no sources"}

        bm25_path = self.data_dir / "bm25_corpus.json"
        graph_path = self.data_dir / "memory_graph.json"
        state_path = self.data_dir / "sync_state.json"

        current_sources = sorted({c["metadata"]["source"] for c in chunks})
        all_ids = self._assign_chunk_ids(chunks)
        prev_hashes = self._load_prev_hashes()
        # Full rebuild path: missing/legacy state, or every current source
        # is in the changed set with no surviving entries.
        full_rebuild = prev_hashes is None or (
            set(changed) == set(current_sources) and not removed
        )

        col_name = f"memory_{self.agent_id}"
        if full_rebuild:
            log.info("[%s] full reindex: %d chunks from %d sources",
                     self.agent_id, len(chunks), len(current_sources))
            try:
                self.chroma_client.delete_collection(col_name)
            except Exception:
                pass
            col = self.chroma_client.create_collection(col_name, embedding_function=self.embedder)
            documents = [c["content"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]
            col.upsert(ids=all_ids, documents=documents, metadatas=metadatas)
        else:
            log.info("[%s] incremental reindex: +%d changed -%d removed",
                     self.agent_id, len(changed), len(removed))
            col = self.chroma_client.get_or_create_collection(
                col_name, embedding_function=self.embedder
            )
            # Delete all chunks belonging to changed-or-removed sources;
            # then upsert fresh chunks for the changed sources only. The
            # per-source id scheme (_assign_chunk_ids) means surviving
            # chunks' ids match what BM25 will write below.
            for src in list(changed) + list(removed):
                try:
                    col.delete(where={"source": src})
                except Exception:
                    log.exception("[%s] failed to delete chunks for %s",
                                  self.agent_id, src)
            changed_set = set(changed)
            up_ids: list[str] = []
            up_docs: list[str] = []
            up_metas: list[dict[str, Any]] = []
            for chunk_id, c in zip(all_ids, chunks):
                if c["metadata"]["source"] not in changed_set:
                    continue
                up_ids.append(chunk_id)
                up_docs.append(c["content"])
                up_metas.append(c["metadata"])
            if up_ids:
                col.upsert(ids=up_ids, documents=up_docs, metadatas=up_metas)

        # BM25 corpus is rebuilt from all current chunks every time —
        # JSON dump is sub-millisecond and avoids any drift between the
        # indexed set and the searched set.
        corpus = [
            {"id": all_ids[i], "text": chunks[i]["content"],
             "section": chunks[i]["metadata"]["section"]}
            for i in range(len(chunks))
        ]
        self._atomic_write_json(bm25_path, corpus)

        # Graph also rebuilt fully — cross-file "mentions" edges scan the
        # global node set (see _build_graph), so any change can ripple.
        G = self._build_graph(chunks)
        self._atomic_write_json(graph_path, nx.node_link_data(G))

        state = {
            "agent_id": self.agent_id,
            "sourceHashes": self._compute_source_hashes(),
            "chromadbChunks": len(chunks),
            "graphNodes": G.number_of_nodes(),
            "graphEdges": G.number_of_edges(),
            "lastSync": datetime.now(timezone.utc).isoformat(),
            "status": "synced",
        }
        self._atomic_write_json(state_path, state)
        return {
            "status": "reindexed",
            "chunks": len(chunks),
            "graphNodes": G.number_of_nodes(),
            "changed": changed if not full_rebuild else current_sources,
            "removed": removed,
        }

    async def reindex_if_stale(self, force: bool = False) -> dict[str, Any]:
        async with self.lock:
            loop = asyncio.get_running_loop()
            if force:
                current = await loop.run_in_executor(
                    None, lambda: sorted(self._compute_source_hashes().keys())
                )
                return await loop.run_in_executor(
                    None, self.do_reindex, current, []
                )
            changed, removed = await loop.run_in_executor(None, self.needs_reindex)
            if not changed and not removed:
                return {"status": "in_sync"}
            return await loop.run_in_executor(None, self.do_reindex, changed, removed)

    # --- retrieval ----------------------------------------------------------

    @staticmethod
    def _rrf_fuse(bm25_ranked: list[str], vector_ranked: list[str]) -> list[str]:
        scores: dict[str, float] = {}
        for rank, doc_id in enumerate(bm25_ranked):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)
        for rank, doc_id in enumerate(vector_ranked):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)
        return sorted(scores, key=lambda x: scores[x], reverse=True)

    def _hybrid_search(self, query: str, n: int = 5) -> list[dict[str, Any]]:
        bm25_path = self.data_dir / "bm25_corpus.json"
        try:
            with open(bm25_path) as f:
                corpus = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

        tokenized = [doc["text"].lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores(query.lower().split())
        bm25_ranked = [
            corpus[i]["id"]
            for i in sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True)
            if bm25_scores[i] > 0
        ]

        if self.chroma_client is None:
            return []
        col_name = f"memory_{self.agent_id}"
        try:
            col = self.chroma_client.get_collection(col_name, embedding_function=self.embedder)
        except Exception:
            return []

        total = max(col.count(), 1)
        results = col.query(
            query_texts=[query],
            n_results=min(n * 2, total),
            include=["documents", "metadatas", "distances"],
        )
        vector_ranked = [
            results["ids"][0][i]
            for i in sorted(range(len(results["ids"][0])), key=lambda x: results["distances"][0][x])
            if results["distances"][0][i] < VECTOR_DISTANCE_MAX
        ]

        fused = self._rrf_fuse(bm25_ranked, vector_ranked)[:n]
        id_to_doc = {doc["id"]: doc for doc in corpus}
        return [id_to_doc[fid] for fid in fused if fid in id_to_doc]

    def _query_graph(self, query: str, top_n: int = 5) -> dict[str, Any]:
        graph_path = self.data_dir / "memory_graph.json"
        try:
            with open(graph_path) as f:
                G = nx.node_link_graph(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            return {"nodes": 0, "related": []}
        terms = query.lower().split()
        hits = [n for n in G.nodes() if any(t in n.lower() for t in terms)]
        results = []
        for node in hits[:top_n]:
            neighbors = list(G.successors(node)) + list(G.predecessors(node))
            results.append({"node": node, "neighbors": neighbors[:6]})
        return {"nodes": G.number_of_nodes(), "related": results}

    def _sync_status(self) -> dict[str, Any]:
        state_path = self.data_dir / "sync_state.json"
        try:
            with open(state_path) as f:
                state = json.load(f)
            if self._compute_source_hashes() != state.get("sourceHashes", {}):
                state["status"] = "OUT_OF_SYNC"
            return state
        except Exception:
            return {"status": "UNKNOWN", "lastSync": "never"}

    def _build_markdown(self, query: str, top_n: int, compact: bool) -> str:
        sync = self._sync_status()
        chunks = self._hybrid_search(query, n=top_n)
        graph = self._query_graph(query)

        lines = ["## Auto-Retrieved Memory Context"]
        if sync["status"] != "synced":
            lines.append(f"**Sync:** {sync['status']} · Last: {sync.get('lastSync', 'never')[:19]}")
        if chunks:
            lines.append(f"\n### Hybrid Search ({len(chunks)} results — BM25 + vector + RRF)")
            for r in chunks:
                snippet = r["text"][:150] if compact else r["text"][:300]
                lines.append(f"- **[{r['section']}]** {snippet}")
        else:
            lines.append("\n_No strong matches in memory for this query._")
        if graph["related"]:
            lines.append(f"\n### Knowledge Graph ({graph['nodes']} nodes)")
            for r in graph["related"]:
                lines.append(f"- **{r['node']}** -> {', '.join(r['neighbors'][:4])}")
        if sync["status"] == "OUT_OF_SYNC":
            lines.append("\n### WARNING: MEMORY OUT OF SYNC — index may be stale")
        return "\n".join(lines)

    async def retrieve_markdown(self, query: str, top_n: int = 5, compact: bool = True) -> str:
        if not query.strip():
            return ""
        async with self.lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                functools.partial(self._build_markdown, query, top_n, compact),
            )


