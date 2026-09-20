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
import collections
import functools
import hashlib
import itertools
import json
import logging
import math
import os
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import chromadb
import networkx as nx
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from claw.channel.envelope import strip_inbound_envelope
from claw.config import MemoryRetrievalConfig

log = logging.getLogger("claw.memory")

MODEL_NAME = "all-MiniLM-L6-v2"
RRF_K = 60

# Keyword tokens: lowercase words and numbers with punctuation stripped, so
# "music." and "music" are one word; inner dots/dashes/underscores are kept so
# versions, filenames and ids ("0.34.2", "claw.yaml", "m-7f3a2c9e") survive.
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._-]*[a-z0-9]|[a-z0-9]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# Generic English words carrying no topic, dropped from every query term list
# — the keyword side of the search (_candidates, which the per-turn gate, the
# search tool and the curator all reach) and the graph (_query_graph). NOT
# from the vector side, which embeds the sentence as written.
#
# Mostly closed-class — pronouns, determiners, auxiliaries,
# wh-words, modals, prepositions — plus a few light verbs and particles
# ("go", "out") that are open-class but say nothing about what is being
# asked. Nothing here was chosen by looking at a store's contents; the test
# for adding a word is whether it would be filler in any English store.
#
# The derived common words in _keyword_index cover whatever is FREQUENT in a
# store, personal filler included — a live store derives its own project and
# operator names, the current year, "section", "file" with no list at all, and
# that half needs no configuring. What it cannot reach is a generic word that
# happens to be rare HERE: memories are terse notes, so on one live store
# "did" appeared in 6 of 437 memories and "what" in 9 — the same 9 as "room".
# No frequency rule separates those, because the distinction is word class, a
# fact about the language rather than about the store.
#
# Two reasons this is a constant and not config. It is not store-specific, so
# there is nothing for an operator to tune; and _MIN_MEMORIES_FOR_COMMON means
# a store under 20 memories derives nothing, so on a fresh workspace this list
# is the only filtering there is.
#
# English only. A non-English store loses nothing (these simply never match)
# but gains nothing either; add sets per language if that ever matters.
_QUERY_STOPWORDS = frozenset("""
    a about an and any are as at be been being but by can could did do does
    doing done for from go had has have having how i if in into is it its me
    my of on or our out so than that the their them then there these they
    this those to too was we were what when where which while who whom why
    will with would you your
""".split())

# How many recent retrievals the drift check looks at.
_DRIFT_WINDOW = 100
# Below this many memories, no word is treated as common (see _keyword_index).
_MIN_MEMORIES_FOR_COMMON = 20

EXCLUDED_SECTION_PATTERNS = re.compile(
    r"^(Auto-Retrieved Memory Context|Conversation Summary|Hybrid Search|Knowledge Graph)\b"
)

DAILY_NOTE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")

# The curator's archive sink: monthly shards under `memory/archive/YYYY-MM.md`.
# They live in a SUBDIRECTORY, so the non-recursive `memory/*.md` source glob
# never picks them up — archived memories are preserved on disk, git-diffable,
# and never indexed/retrieved. Sharding keeps every archive file small so the
# curator can grep/append cheaply instead of read-modify-writing a monolith.
# (A legacy flat `memory/archive.md` also fails DAILY_NOTE_PATTERN, so any
# pre-migration archive stays excluded from indexing too.)
ARCHIVE_DIR = "archive"

# Per-section memory marker, written as the first body line under a `##`
# heading, e.g. `<!-- mem ts=2026-06-28 id=m-7f3a2c9e status=active
# supersededBy=m-1a0b22f4 -->`. The collector seeds only `ts`; the nightly
# curator adds `id`/`status`/`supersededBy`. Markdown is the source of truth
# for all of it — these fields are parsed into DERIVED chunk metadata
# (overwritten on every upsert), never read back to persist.
_MEM_MARKER_RE = re.compile(r"^<!--\s*mem\s+(?P<fields>.*?)\s*-->$")
_MEM_FIELD_RE = re.compile(r"(\w+)=([A-Za-z0-9._:\-]+)")

# Indexing is disjoint from prompt injection (extra_paths in claw.yaml).
# Files the model already sees verbatim every turn (IDENTITY/USER/SOUL/AGENTS/
# TOOLS) shouldn't compete for retrieval slots — only large/growing content
# belongs here. MEMORY.md is the boundary case; daily notes scale.
ROOT_SOURCES = ("MEMORY.md",)


class MemoryIndex:
    """One per agent. Owns its own ChromaDB collection plus BM25/graph sidecars."""

    def __init__(
        self, agent_id: str, workspace_dir: Path,
        retrieval: MemoryRetrievalConfig = MemoryRetrievalConfig(),
    ) -> None:
        self.agent_id = agent_id
        self.workspace_dir = Path(workspace_dir)
        self.data_dir = self.workspace_dir / ".memory"
        self.retrieval = retrieval
        self.embedder: Any | None = None
        self.chroma_client: chromadb.api.ClientAPI | None = None
        # The keyword index, built once per corpus version (see _keyword_index)
        # rather than reloaded and rebuilt on every turn.
        self._kw: dict[str, Any] | None = None
        # Best vector distance of recent retrievals, for the drift warning.
        self._best_distances: collections.deque[float] = collections.deque(maxlen=_DRIFT_WINDOW)
        self._retrievals = 0
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
    def _extract_marker(body: str) -> tuple[str, dict[str, str]]:
        """Pull a leading ``<!-- mem ... -->`` marker off a section body.

        Returns ``(body_without_marker, fields)``. The marker must be the
        first non-blank line; otherwise nothing is recognized. Stripping it
        keeps the marker out of the embedded/BM25 text."""
        lines = body.split("\n")
        idx = 0
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        if idx < len(lines):
            m = _MEM_MARKER_RE.match(lines[idx].strip())
            if m:
                fields = dict(_MEM_FIELD_RE.findall(m.group("fields")))
                del lines[idx]
                return "\n".join(lines).strip(), fields
        return body, {}

    @staticmethod
    def _marker_metadata(fields: dict[str, str]) -> dict[str, Any]:
        """Map raw marker fields onto chunk metadata. ``status`` defaults to
        ``active``; optional keys are omitted (never stored as ``None``) so
        Chroma metadata stays scalar."""
        meta: dict[str, Any] = {"status": fields.get("status", "active")}
        if "ts" in fields:
            meta["ts"] = fields["ts"]
        if "id" in fields:
            meta["mem_id"] = fields["id"]
        if "supersededBy" in fields:
            meta["superseded_by"] = fields["supersededBy"]
        return meta

    @classmethod
    def _parse_markdown(cls, path: Path) -> list[dict[str, Any]]:
        content = path.read_text()
        chunks: list[dict[str, Any]] = []
        sections = re.split(r"(^##\s+.*$)", content, flags=re.MULTILINE)
        intro_body, intro_fields = cls._extract_marker(sections[0].strip())
        if intro_body:
            meta = {"section": "Intro", **cls._marker_metadata(intro_fields)}
            if meta["status"] != "archived":
                chunks.append({"content": intro_body, "metadata": meta})
        for i in range(1, len(sections), 2):
            header = sections[i].strip().lstrip("#").strip()
            raw_body = sections[i + 1].strip() if i + 1 < len(sections) else ""
            body, fields = cls._extract_marker(raw_body)
            if body and not EXCLUDED_SECTION_PATTERNS.match(header):
                meta = {"section": header, **cls._marker_metadata(fields)}
                # Defensive: archived memories should already live in
                # archive.md (not indexed). If one is still tagged archived
                # inside an indexed file, keep it out of retrieval anyway.
                if meta["status"] == "archived":
                    continue
                chunks.append({"content": body, "metadata": meta})
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
    def _build_graph(
        chunks: list[dict[str, Any]], common: Iterable[str] = (),
    ) -> nx.DiGraph:
        """Sections, the concepts they bold, and what refers to what.

        Both passes match on *token sequences* rather than raw substrings. The
        scan this replaced asked ``target in chunk_text``, so "ok" matched
        inside "cookbook" and "the" inside any name containing it; on one live
        store the highest-degree nodes came out "not" (159 edges), "one" (109)
        and "first" (38), none of which mean anything. A node whose name is
        made entirely of the store's common words is dropped as a mention
        target for the same reason — it occurs everywhere and distinguishes
        nothing. ``common`` comes from the keyword index, so a name is judged
        by the same word statistics the keyword leg uses.

        ``contains`` (this section bolded the concept) outranks ``mentions``
        (this section refers to it). Both were already built, but a bolded
        concept trivially appears in its own chunk's text, so the mentions
        pass overwrote every ``contains`` edge — all 2,395 edges in a live
        store were ``mentions`` and none were ``contains``.
        """
        common = frozenset(common)

        # Bold means two different things in these files. It marks a concept,
        # and it marks the label of a list field — "* **Event**: ..." — which
        # is formatting, not a claim about the world. A span written ONLY as a
        # label is furniture and becomes no node at all.
        #
        # The test is structural, on the markdown: a colon immediately after
        # the span, inside the bold or outside it. Frequency cannot do this —
        # the most-bolded span in a live store was a topical one appearing in
        # 10 files, while its labels appeared in 4. And the damage is done by
        # rare labels, not frequent ones: a label like "Status" is bolded once,
        # becomes a node, and then collects a `mentions` edge from every
        # section using that ordinary word in prose. Measured on that store,
        # every dominant neighbour ("Status" 16 label uses and 0 plain, "Fix"
        # 13/0, "Lesson" 7/0) was label-only, and the topical nodes were not.
        plain_use: collections.Counter[str] = collections.Counter()
        for c in chunks:
            for m in re.finditer(r"\*\*(.*?)\*\*(\s*:)?", c["content"]):
                span = m.group(1).strip()
                if not (m.group(2) or span.endswith(":")):
                    plain_use[span] += 1

        # A node is identified by its TOKEN SEQUENCE, which is exactly what the
        # keyword leg indexes — so "Ollama" and "ollama" are one node, as are
        # "Routing" and "Routing:". Keying by the raw string while *matching*
        # by tokens (the pass below) is what let one concept exist twice: both
        # spellings matched the same text, collected identical edges, and then
        # took two of the five result slots to say the same thing.
        #
        # The label is kept separate, and is NOT lowercased. The keyword leg
        # has no display surface — nobody reads its tokens — while a node name
        # is rendered into the prompt verbatim, and folding the case there
        # would show the model "mlx-lm #1061" and lowercased headings. The
        # label is the store's own commonest spelling, ties broken
        # lexicographically so a rebuild is stable.
        surface: dict[tuple[str, ...], collections.Counter[str]] = collections.defaultdict(
            collections.Counter)
        kinds: dict[tuple[str, ...], set[str]] = collections.defaultdict(set)
        for c in chunks:
            for name, kind in itertools.chain(
                [(c["metadata"]["section"], "section")],
                ((s.strip(), "concept") for s in re.findall(r"\*\*(.*?)\*\*", c["content"])),
            ):
                if kind == "concept" and not (
                    3 <= len(name) <= 50 and plain_use[name.rstrip(":")]
                ):
                    continue
                key = tuple(tokenize(name))
                if key:
                    surface[key][name] += 1
                    kinds[key].add(kind)

        label = {k: min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0] for k, c in surface.items()}
        # A heading and a bolded span that normalise alike are one node, typed
        # as the section: the heading is structural, and a caller distinguishing
        # "a section I can go read" from "a phrase" wants the stronger claim.
        node_kind = {k: ("section" if "section" in v else "concept") for k, v in kinds.items()}

        def named(raw: str) -> str | None:
            """The canonical label for a name, or None when it has no tokens."""
            return label.get(tuple(tokenize(raw)))

        G: nx.DiGraph = nx.DiGraph()
        for key, name in label.items():
            G.add_node(name, type=node_kind[key])
        for c in chunks:
            section = named(c["metadata"]["section"])
            if section is None:
                continue
            for concept in re.findall(r"\*\*(.*?)\*\*", c["content"]):
                target = named(concept.strip())
                if target is not None and target != section and target in G:
                    G.add_edge(section, target, relation="contains")

        # Mention targets bucketed by first token, so each chunk is scanned
        # once against the candidates that could start at each position —
        # the previous pass ran |chunks| x |nodes| substring searches. The key
        # IS the token sequence, so no name is tokenised twice.
        targets: dict[str, list[tuple[str, tuple[str, ...]]]] = collections.defaultdict(list)
        for key, name in label.items():
            if not all(t in common for t in key):
                targets[key[0]].append((name, key))

        for c in chunks:
            section = named(c["metadata"]["section"])
            if section is None:
                continue
            tokens = tokenize(c["content"])
            found: set[str] = set()
            for i, tok in enumerate(tokens):
                for name, name_tokens in targets.get(tok, ()):
                    if name != section and name not in found:
                        if tuple(tokens[i:i + len(name_tokens)]) == name_tokens:
                            found.add(name)
            for name in found:
                if not G.has_edge(section, name):
                    G.add_edge(section, name, relation="mentions")
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
             "section": chunks[i]["metadata"]["section"],
             # Carried so retrieval can resolve supersession chains without
             # re-reading markdown: mem_id is the durable link identity,
             # superseded_by points forward to the successor's mem_id.
             "mem_id": chunks[i]["metadata"].get("mem_id"),
             "status": chunks[i]["metadata"].get("status", "active"),
             "superseded_by": chunks[i]["metadata"].get("superseded_by")}
            for i in range(len(chunks))
        ]
        self._atomic_write_json(bm25_path, corpus)

        # Graph also rebuilt fully — cross-file "mentions" edges scan the
        # global node set (see _build_graph), so any change can ripple.
        # Built after the BM25 corpus so the keyword index rebuilds from the
        # fresh file: the graph and the keyword leg then judge "common" by
        # the same statistics, rather than drifting apart.
        kw = self._keyword_index()
        G = self._build_graph(chunks, kw["common"] if kw else ())
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

    # --- curation support ---------------------------------------------------

    def collect_sections(self) -> list[dict[str, Any]]:
        """Public view of the current parsed memory chunks (one per ``##``
        section across MEMORY.md + indexed daily notes), each as
        ``{content, metadata:{section, source, status, ts?, mem_id?,
        superseded_by?}}``. The curator briefing-builder uses this."""
        return self._collect_sources()

    def search(self, query: str, n: int = 8) -> list[dict[str, Any]]:
        """Public sync wrapper over the hybrid search — used by the curator to
        gather near-neighbours of a changed memory for dedup/supersession
        judgment. Blocking (Chroma + BM25); call from an executor."""
        return self._hybrid_search(query, n=n)

    def _curation_state_path(self) -> Path:
        return self.data_dir / "curation_state.json"

    def load_curation_state(self) -> dict[str, Any]:
        """Returns the persisted curation state (``{curatedThrough, lastCurated}``)
        or ``{}`` if missing/unreadable. ``curatedThrough`` is a ``YYYY-MM-DD``
        date cursor: the latest daily note the curator has groomed. The nightly
        selector walks strictly FORWARD from it — files at or before the cursor
        are never re-scanned (they only resurface as near-neighbours), so the
        curator is monotonic and its own edits never re-enqueue a file."""
        path = self._curation_state_path()
        if not path.exists():
            return {}
        try:
            with open(path) as f:
                state = json.load(f)
            return state if isinstance(state, dict) else {}
        except Exception:
            return {}

    def write_curation_state(self, curated_through: str) -> None:
        self._atomic_write_json(
            self._curation_state_path(),
            {
                "agent_id": self.agent_id,
                "curatedThrough": curated_through,
                "lastCurated": datetime.now(timezone.utc).isoformat(),
            },
        )

    # --- retrieval ----------------------------------------------------------

    @staticmethod
    def _rrf_fuse(bm25_ranked: list[str], vector_ranked: list[str]) -> list[str]:
        scores: dict[str, float] = {}
        for rank, doc_id in enumerate(bm25_ranked):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)
        for rank, doc_id in enumerate(vector_ranked):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)
        return sorted(scores, key=lambda x: scores[x], reverse=True)

    def _keyword_index(self) -> dict[str, Any] | None:
        """The keyword (BM25) index and the store's common words, rebuilt only
        when ``bm25_corpus.json`` changes — i.e. after a reindex wrote it.

        Common words are derived from the store itself: any token appearing in
        more than ``common_word_max_share`` of the memories. They match too
        much to carry meaning ("the", but also this store's own filler — a
        project name, a year, "section"), so they are left out of the index
        and dropped from queries. No hand-written word list.
        """
        path = self.data_dir / "bm25_corpus.json"
        try:
            mtime = path.stat().st_mtime_ns
        except FileNotFoundError:
            return None
        if self._kw is not None and self._kw["mtime"] == mtime:
            return self._kw
        try:
            with open(path) as f:
                corpus = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        if not corpus:
            return None
        docs = [tokenize(d["text"]) for d in corpus]
        df = collections.Counter(w for d in docs for w in set(d))
        cutoff = self.retrieval.common_word_max_share * len(docs)
        # Word frequencies mean nothing in a tiny store (with 3 memories every
        # word is in "more than 10%" of them), so the rule waits for enough
        # memories to count.
        common = ({w for w, c in df.items() if c > cutoff}
                  if len(docs) >= _MIN_MEMORIES_FOR_COMMON else set())
        # BM25Okapi cannot take an empty document list entry; a memory made
        # only of common words gets a placeholder that no query can match.
        bm25 = BM25Okapi([[w for w in d if w not in common] or ["\x00"] for d in docs])
        self._kw = {"mtime": mtime, "corpus": corpus, "ids": [d["id"] for d in corpus],
                    "by_id": {d["id"]: d for d in corpus}, "bm25": bm25, "common": common}
        log.info("[%s] keyword index: %d memories, %d common words excluded",
                 self.agent_id, len(corpus), len(common))
        return self._kw

    def _candidates(self, query: str, n: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """The fused (keyword + vector, RRF) top ``n``, each hit carrying its
        vector ``_distance`` and keyword ``_kw`` score, plus the per-query
        facts the relevance gate needs (``kw_best``, ``best_distance``).

        The keyword side searches the message without its envelope header and
        without words that carry no topic — the store's own common words, and
        the generic English ones no frequency rule can reach. Every keyword
        path arrives here: the per-turn gate, the explicit search tool, and the
        curator's neighbour lookup.

        The vector side embeds the **full text**, deliberately. Stripping
        function words is right for a bag-of-words score and wrong for an
        encoder, which was trained on ordinary sentences and reads "is the
        hose working" differently from "hose working"; the relevance weights
        were calibrated on those distances besides.

        No floor here: eligibility is the gate's decision, not the search's.
        """
        kw = self._keyword_index()
        if kw is None or self.chroma_client is None:
            return [], {}
        terms = [w for w in tokenize(strip_inbound_envelope(query))
                 if w not in kw["common"] and w not in _QUERY_STOPWORDS]
        scores = dict(zip(kw["ids"], kw["bm25"].get_scores(terms))) if terms else {}
        bm25_ranked = sorted((i for i, v in scores.items() if v > 0), key=lambda i: -scores[i])

        try:
            col = self.chroma_client.get_collection(
                f"memory_{self.agent_id}", embedding_function=self.embedder)
        except Exception:
            return [], {}
        emb = self.embedder([query])[0]
        total = max(col.count(), 1)
        res = col.query(query_embeddings=[emb], n_results=min(n * 2, total),
                        include=["distances"])
        dist = dict(zip(res["ids"][0], res["distances"][0]))
        vector_ranked = sorted(dist, key=dist.get)

        fused = [i for i in self._rrf_fuse(bm25_ranked, vector_ranked) if i in kw["by_id"]][:n]
        # Candidates found by keyword alone have no distance from the query;
        # compute it from their stored embeddings (squared L2 — Chroma's own
        # metric) instead of issuing a second query.
        missing = [i for i in fused if i not in dist]
        if missing:
            got = col.get(ids=missing, include=["embeddings"])
            for cid, e in zip(got["ids"], got["embeddings"]):
                dist[cid] = float(sum((a - b) ** 2 for a, b in zip(emb, e)))
        hits = [{**kw["by_id"][i], "_distance": dist.get(i), "_kw": scores.get(i, 0.0)}
                for i in fused]
        facts = {"kw_best": max(scores.values(), default=0.0),
                 "best_distance": min(dist.values(), default=None)}
        return hits, facts

    def _hybrid_search(self, query: str, n: int = 5) -> list[dict[str, Any]]:
        """Fused top ``n`` with supersession heads, ungated — the explicit
        ``memory_search`` tool and the curator's neighbour search want
        breadth, not the per-turn relevance cut."""
        hits, _ = self._candidates(query, n)
        kw = self._kw
        return self._resolve_supersession(kw["corpus"], hits) if kw and hits else hits

    def _relevance(self, hit: dict[str, Any], kw_best: float) -> float:
        r = self.retrieval.relevance
        kwv = hit["_kw"]
        share = kwv / kw_best if kw_best > 0 else 0.0
        z = r.offset + r.distance * (hit["_distance"] if hit["_distance"] is not None else 9.0) \
            + r.keyword * math.log1p(kwv) + r.keyword_share * share
        return 1.0 / (1.0 + math.exp(-z))

    def _relevant(self, query: str, top_n: int) -> list[dict[str, Any]]:
        """Per-turn retrieval: the top ``candidates`` of the search, kept only
        if their relevance score clears the threshold — up to ``top_n``, in
        search-rank order, then supersession heads. ``threshold: 0`` turns the
        gate off (the plain top_n slice)."""
        cfg = self.retrieval
        hits, facts = self._candidates(query, max(cfg.candidates, top_n))
        if not hits:
            return []
        if cfg.relevance.threshold <= 0:
            kept = hits[:top_n]
        else:
            kept = [h for h in hits
                    if self._relevance(h, facts["kw_best"]) >= cfg.relevance.threshold][:top_n]
        log.debug("[%s] retrieval: %d candidates, %d passed (best distance %.3f)",
                  self.agent_id, len(hits), len(kept), facts.get("best_distance") or -1)
        self._note_best_distance(facts.get("best_distance"))
        return self._resolve_supersession(self._kw["corpus"], kept) if kept else []

    def _note_best_distance(self, best: float | None) -> None:
        """Drift check: warn when recent messages' best vector distance has
        moved well away from where the relevance weights were calibrated."""
        if best is None:
            return
        self._best_distances.append(best)
        self._retrievals += 1
        if self._retrievals % _DRIFT_WINDOW or len(self._best_distances) < _DRIFT_WINDOW:
            return
        cal = self.retrieval.calibration
        median = statistics.median(self._best_distances)
        if abs(median - cal.best_distance) > cal.tolerance:
            log.warning(
                "[%s] memory retrieval drift: median best distance over the last %d "
                "turns is %.2f, but the relevance weights were calibrated at %.2f "
                "(tolerance %.2f). The store's shape has changed; re-check "
                "memory_retrieval.relevance (docs/operations.md, Memory retrieval).",
                self.agent_id, _DRIFT_WINDOW, median, cal.best_distance, cal.tolerance,
            )

    @staticmethod
    def _resolve_supersession(
        corpus: list[dict[str, Any]], hits: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Follow each hit's ``superseded_by`` chain old->new to the current
        head and append the heads AFTER the original hits, so current truth
        renders last (closest to the model's generation point). Guards: a
        per-walk ``visited`` set breaks cycles, a hop cap bounds runaway
        chains, and a missing target (an archived/removed successor) stops the
        walk at the last resolvable node. A head that is itself among ``hits``
        is moved to the end rather than duplicated, so current truth renders
        last whether it matched directly or only via the pointer. Heads are
        returned as flagged COPIES (``_current``): the corpus is cached across
        turns, and a flag written onto it would stick."""
        id_by_mem = {d["mem_id"]: d for d in corpus if d.get("mem_id")}
        head_ids: list[str] = []
        heads: dict[str, dict[str, Any]] = {}
        for d in hits:
            target = d.get("superseded_by")
            visited: set[str] = set()
            head: dict[str, Any] | None = None
            while (target and target in id_by_mem
                   and target not in visited and len(visited) < 8):
                visited.add(target)
                head = id_by_mem[target]
                target = head.get("superseded_by")
            if head is not None and head["id"] not in heads:
                heads[head["id"]] = {**head, "_current": True}
                head_ids.append(head["id"])
        return [d for d in hits if d["id"] not in heads] + [heads[i] for i in head_ids]

    def _query_graph(self, query: str, top_n: int = 5) -> dict[str, Any]:
        """Nodes whose names carry the query's meaningful words, best first,
        each with what it links to.

        The leg exists for traversal — reaching a section because it is linked
        to what was asked about, when that section says nothing the keyword or
        vector legs would match. That only pays off if the entry node is right,
        so hits are scored rather than taken in graph insertion order.

        Scoring is by summed IDF of the matched words, reusing the keyword
        leg's own term statistics, then by how much of the name they account
        for. Counting matched words instead is not enough: every word scores 1,
        so "is bluetooth working on the box" ranked "Working Directory:" over
        the sections about bluetooth — a tight match on a vague word beating a
        loose match on the word that carried the question. Dropping common
        words alone doesn't fix it either, since "working" is nowhere near
        common enough to be dropped; it is merely far less informative than
        "bluetooth", which is exactly what IDF measures.
        """
        graph_path = self.data_dir / "memory_graph.json"
        try:
            with open(graph_path) as f:
                G = nx.node_link_graph(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            return {"nodes": 0, "related": []}
        kw = self._keyword_index()
        common = kw["common"] if kw else frozenset()
        idf: dict[str, float] = kw["bm25"].idf if kw else {}
        terms = {
            t for t in tokenize(strip_inbound_envelope(query))
            if t not in common and t not in _QUERY_STOPWORDS
        }
        if not terms:
            return {"nodes": G.number_of_nodes(), "related": []}
        scored: list[tuple[float, float, str]] = []
        for node in G.nodes():
            name_tokens = tokenize(node)
            if not name_tokens:
                continue
            matched = terms.intersection(name_tokens)
            # A word the corpus has never seen carries no evidence either way;
            # scoring it 0 lets a name matched only on such words fall out.
            weight = sum(idf.get(t, 0.0) for t in matched)
            if weight > 0:
                scored.append((weight, len(matched) / len(name_tokens), node))
        scored.sort(key=lambda s: (-s[0], -s[1], s[2]))
        results = []
        for _, _, node in scored[:top_n]:
            # Successors first (what this section bolds or refers to), then
            # what points back at it; de-duplicated, since an edge can run
            # both ways between two sections.
            neighbors = list(dict.fromkeys(
                list(G.successors(node)) + list(G.predecessors(node))
            ))
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

    def _build_markdown(self, query: str, top_n: int, compact: bool, gate: bool = True) -> str:
        sync = self._sync_status()
        chunks = self._relevant(query, top_n) if gate else self._hybrid_search(query, n=top_n)
        graph = self._query_graph(query)

        # A per-turn retrieval that found nothing injects nothing. The caller
        # skips the history row entirely on "", which is the whole saving: the
        # row costs tokens and re-ranks every turn, so an empty one is pure
        # overhead. Only the gated path goes quiet — an explicit search still
        # answers "no matches", because someone asked — and an unhealthy index
        # still reports, since saying so is the point of the warning.
        if gate and not chunks and not graph["related"] and sync["status"] == "synced":
            return ""

        lines = ["## Auto-Retrieved Memory Context"]
        if sync["status"] != "synced":
            lines.append(f"**Sync:** {sync['status']} · Last: {sync.get('lastSync', 'never')[:19]}")
        if chunks:
            lines.append(f"\n### Hybrid Search ({len(chunks)} results — BM25 + vector + RRF)")
            for r in chunks:
                snippet = r["text"][:150] if compact else r["text"][:300]
                tag = " (current)" if r.get("_current") else ""
                lines.append(f"- **[{r['section']}]{tag}** {snippet}")
        else:
            lines.append("\n_No strong matches in memory for this query._")
        if graph["related"]:
            lines.append(f"\n### Knowledge Graph ({graph['nodes']} nodes)")
            for r in graph["related"]:
                lines.append(f"- **{r['node']}** -> {', '.join(r['neighbors'][:4])}")
        if sync["status"] == "OUT_OF_SYNC":
            lines.append("\n### WARNING: MEMORY OUT OF SYNC — index may be stale")
        return "\n".join(lines)

    async def retrieve_markdown(
        self, query: str, top_n: int = 5, compact: bool = True, gate: bool = True,
    ) -> str:
        """``gate=True`` (per-turn retrieval) applies the relevance gate;
        ``gate=False`` (an explicit search) returns the ranked top_n."""
        if not query.strip():
            return ""
        async with self.lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                functools.partial(self._build_markdown, query, top_n, compact, gate),
            )


