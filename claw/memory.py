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
import time
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
# search tool and the curator all reach). NOT
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

# Bumped whenever the INDEXED REPRESENTATION changes -- what goes into the
# BM25 corpus or the embedded document, how chunks are cut, how the graph is
# built. Staleness is otherwise judged by hashing the SOURCE FILES, which
# cannot notice that the code reading them now produces something different:
# a store then reports itself in sync forever while serving an index built by
# the previous version. That has happened, silently, for a whole release --
# a rewritten graph builder shipped and did nothing until someone deleted
# sync_state.json by hand. A version mismatch means "everything changed".
#
# 1 -> 2: section headings are indexed with their bodies.
INDEX_VERSION = 3

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


def _embed_doc(chunk: dict[str, Any]) -> str:
    """The text handed to the embedder: the heading, then the body.

    One function because there are two reindex paths — full rebuild and
    incremental — and they had drifted: the incremental one prepended the
    section and the full rebuild did not, so a chunk's vector depended on
    which path last touched it, and any INDEX_VERSION bump (which forces a
    full rebuild) silently dropped every heading back out of the vector side.
    BM25 does the same join in _keyword_index; keep the two in step.
    """
    return chunk["metadata"]["section"] + "\n" + chunk["content"]

# Annotation: a markdown heading, or an HTML comment. Used ONLY to ask whether
# a chunk has anything in it — never to alter what is indexed.
_ANNOTATION_RE = (re.compile(r"<!--.*?-->", re.S),
                  re.compile(r"^#{1,6} .*$", re.MULTILINE))
# A heading opening the pre-`##` region of a file. `_parse_markdown` splits on
# `##` only, so a `#` title or a `###` subheading there stays in the body.
_LEADING_HEADING_RE = re.compile(r"\A\s*(#{1,6})\s+(?P<title>.+?)\s*$", re.MULTILINE)
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
        # Optional System One scorer for smart retrieval. Wired by the
        # process that builds the index; None is fully supported and means
        # retrieval runs on the fitted formula alone (see _smart_select).
        self.scorer: Any | None = None
        self.embedder: Any | None = None
        self.chroma_client: chromadb.api.ClientAPI | None = None
        # The keyword index, built once per corpus version (see _keyword_index)
        # rather than reloaded and rebuilt on every turn.
        self._kw: dict[str, Any] | None = None
        self._graph_cache: tuple[int, Any] | None = None
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

    @staticmethod
    def _substantive(text: str) -> bool:
        """Is there anything here besides annotation?

        A heading and an HTML comment are both annotation: they describe or
        mark up content rather than being it. A chunk made of nothing else
        says nothing — a file's own `# 2026-04-10 (Friday)` title, or a
        leftover `<!-- placeholder for tomorrow's memory -->`.

        This decides ELIGIBILITY only. The text itself is left exactly as it
        is, because both kinds of annotation carry signal alongside real
        content: a section's heading is its most topical line (and is indexed
        with the body for that reason), and a comment is where a supersession
        records WHY a fact changed. Strip either from the index and searches
        get worse; count either as content and the index fills with titles.
        """
        body = text
        for pattern in _ANNOTATION_RE:
            body = pattern.sub(" ", body)
        return bool(body.strip(" \n\t-*_|#"))

    @classmethod
    def _parse_markdown(cls, path: Path) -> list[dict[str, Any]]:
        content = path.read_text()
        chunks: list[dict[str, Any]] = []
        sections = re.split(r"(^##\s+.*$)", content, flags=re.MULTILINE)
        intro_body, intro_fields = cls._extract_marker(sections[0].strip())
        # Lift a leading heading of ANY level into `section`, the way the `##`
        # split already does for level two. Without this the pre-`##` region of
        # every file is called "Intro": the agent is shown `**[Intro]**` for a
        # real memory, the heading is duplicated into the snippet the block
        # already prints above it, and every file's preamble collapses onto one
        # graph node. The name is the most topical line such a chunk has.
        section = "Intro"
        if (m := _LEADING_HEADING_RE.match(intro_body)):
            section = m.group("title")
            intro_body = intro_body[m.end():].strip()
        if intro_body or section != "Intro":
            meta = {"section": section, **cls._marker_metadata(intro_fields)}
            # Annotation-only chunks stay: they are real nodes and the graph
            # walks through them. They are simply never offered as results.
            meta["traversal_only"] = not cls._substantive(intro_body)
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
                meta["traversal_only"] = not cls._substantive(body)
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
        ``sourcesHash`` format, or an index built by a different
        INDEX_VERSION. Caller treats None as "everything changed" and triggers
        a full rebuild that upgrades the file."""
        state_path = self.data_dir / "sync_state.json"
        if not state_path.exists():
            return None
        try:
            with open(state_path) as f:
                state = json.load(f)
        except Exception:
            return None
        if state.get("indexVersion") != INDEX_VERSION:
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
            documents = [_embed_doc(c) for c in chunks]
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
                up_docs.append(_embed_doc(c))
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
             # Annotation-only: a real node the graph walks through, never a
             # result. Carried here so retrieval can drop it without
             # re-reading markdown.
             "traversal_only": chunks[i]["metadata"].get("traversal_only", False),
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
            "indexVersion": INDEX_VERSION,
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
        # The heading is indexed WITH the body. A section's own title is the
        # most topical line it has, and until now it reached neither leg: the
        # parser strips it into metadata (_parse_markdown) and the corpus row
        # stores only the body, so a note titled "Bluetooth pairing" could not
        # be found by searching for those words. Measured on one store, 13 of
        # 443 chunks happened to repeat their title in their body; the other
        # 430 were unsearchable by title. Display still uses `text` alone, so
        # the rendered snippet does not repeat the heading the block already
        # prints.
        docs = [tokenize(d["section"] + " " + d["text"]) for d in corpus]
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
        # Section name -> the chunks under it, keyed by TOKEN SEQUENCE because
        # that is how _build_graph keys a node (a node's label is the store's
        # commonest spelling, so an exact string compare misses). Built here
        # rather than on demand: this walk already happens, and the mtime check
        # above gives the map the same invalidation as the corpus it describes.
        # One-to-many is normal -- headings repeat across daily notes, and every
        # file with preamble yields an "Intro".
        by_section: dict[tuple[str, ...], list[str]] = collections.defaultdict(list)
        for d in corpus:
            key = tuple(tokenize(d["section"]))
            if key:
                by_section[key].append(d["id"])
        self._kw = {"mtime": mtime, "corpus": corpus, "ids": [d["id"] for d in corpus],
                    "by_section": dict(by_section),
                    "section_of": {d["id"]: tuple(tokenize(d["section"])) for d in corpus},
                    "by_id": {d["id"]: d for d in corpus}, "bm25": bm25, "common": common}
        log.info("[%s] keyword index: %d memories, %d common words excluded",
                 self.agent_id, len(corpus), len(common))
        return self._kw

    def _candidates(self, query: str, n: int,
                    expand: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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

        ``expand`` adds the graph's one-hop neighbours of the fused pool as
        further candidates (``_graph_expand``). It is off by default so the
        ungated callers -- the explicit search tool and the curator's
        neighbour lookup -- keep judging exactly what the two legs ranked.
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

        # Annotation-only chunks are dropped here, before anything ranks or
        # scores them. They stay in the corpus and in the graph — the walk
        # passes through them and a section node needs them to exist — but a
        # chunk that is nothing but a heading or a comment is not an answer to
        # anything, and letting one occupy a candidate slot costs a real one.
        eligible = lambda i: i in kw["by_id"] and not kw["by_id"][i].get("traversal_only")
        fused = [i for i in self._rrf_fuse(bm25_ranked, vector_ranked) if eligible(i)][:n]
        # Taken before any backfill: the ANN result is ordered by distance, so
        # its minimum IS the store's minimum. Reading it off `dist` after the
        # backfill gave the same answer only because backfilled ids are by
        # construction outside the vector top-2n -- an invariant that grows
        # thinner as more ids are backfilled, and one the drift check and the
        # calibration constant both rest on.
        best_distance = min(dist.values(), default=None)

        cfg = self.retrieval.graph
        extra: dict[str, float] = {}
        if expand and cfg.enabled:
            extra = self._graph_expand(fused[:cfg.seeds], fused, kw)
            if len(extra) > cfg.max_expand:
                # Insertion order, which is seed rank: neighbours of the
                # best-ranked candidates survive the cap. Sorting by id would
                # truncate alphabetically, and ids begin with their source
                # filename -- silently preferring whichever notes are dated
                # earliest, which is not a policy anyone chose.
                extra = dict(itertools.islice(extra.items(), cfg.max_expand))
        ordered = fused + [i for i in extra if eligible(i)]

        # Candidates found by keyword alone -- or reached through the graph --
        # have no distance from the query; compute it from their stored
        # embeddings (squared L2 — Chroma's own metric) instead of issuing a
        # second query.
        missing = [i for i in ordered if i not in dist]
        if missing:
            got = col.get(ids=missing, include=["embeddings"])
            for cid, e in zip(got["ids"], got["embeddings"]):
                dist[cid] = float(sum((a - b) ** 2 for a, b in zip(emb, e)))
        hits = [{**kw["by_id"][i], "_distance": dist.get(i), "_kw": scores.get(i, 0.0),
                 "_traversal": extra.get(i, 0.0)}
                for i in ordered]
        facts = {"kw_best": max(scores.values(), default=0.0),
                 "best_distance": best_distance}
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
            + r.keyword * math.log1p(kwv) + r.keyword_share * share \
            + r.traversal * hit.get("_traversal", 0.0)
        return 1.0 / (1.0 + math.exp(-z))

    def _relevant(self, query: str, top_n: int) -> list[dict[str, Any]]:
        """Per-turn retrieval: the top ``candidates`` of the search, kept only
        if their relevance score clears the threshold — up to ``top_n``, in
        search-rank order, then supersession heads. ``threshold: 0`` turns the
        gate off (the plain top_n slice)."""
        cfg = self.retrieval
        hits, facts = self._candidates(query, max(cfg.candidates, top_n), expand=True)
        if not hits:
            return []
        if cfg.relevance.threshold <= 0:
            # The gate off means "no relevance judgement", so the order has to
            # come from the search too -- the fused rank, with graph-expanded
            # candidates after it in the order they were reached.
            kept = hits[:top_n]
        else:
            # Ordered by the fitted score, not by fused rank. The number was
            # already being computed for every candidate and thrown away: it is
            # a calibrated estimate of the thing being decided, so using it to
            # admit but not to choose WHICH admitted candidates survive the
            # top_n cut was incoherent. It also has to be this way now, because
            # a graph-expanded candidate has no fused rank to be sorted by.
            # Measured on 50 real messages before graph expansion existed: this
            # changes the injected set on 17 of them and moves precision 0.584
            # -> 0.594, which at that sample size is "not detectably different"
            # rather than an improvement.
            scored = [(self._relevance(h, facts["kw_best"]), i, h) for i, h in enumerate(hits)]
            kept = [h for s_, _, h in sorted(scored, key=lambda t: (-t[0], t[1]))
                    if s_ >= cfg.relevance.threshold][:top_n]
        log.debug("[%s] retrieval: %d candidates, %d passed (best distance %.3f)",
                  self.agent_id, len(hits), len(kept), facts.get("best_distance") or -1)
        self._note_best_distance(facts.get("best_distance"))
        return self._resolve_supersession(self._kw["corpus"], kept) if kept else []

    # --- smart retrieval ----------------------------------------------------

    _SMART_TRUE = ("Yes — it bears directly on what was just said, and the reply "
                   "would be better for having it.")
    _SMART_FALSE = ("No — unrelated, or connected only loosely or by a shared word; "
                    "the reply would be no worse without it.")

    @classmethod
    def _smart_question(cls, text: str, note_chars: int) -> dict[str, Any]:
        """One candidate's yes/no question.

        The note goes in the QUESTION, not the state: every question in a
        request shares one state, so putting the note there would mean one
        prefill per candidate instead of one per turn.

        The wording asks for "relevant and immediately useful" rather than
        "related" deliberately — the floor is what needs lifting, and a scorer
        reading "related" admits anything sharing a topic.
        """
        return {
            "type": "noul",
            "instructions": (
                "A memory note has been retrieved for the assistant's next reply:\n\n"
                f"{text[:note_chars]}\n\n"
                "Is this note relevant and immediately useful for replying to the "
                "last message in the conversation?"),
            "criteria": {"true": cls._SMART_TRUE, "false": cls._SMART_FALSE},
        }

    def _borderline(self, query: str) -> tuple[list[dict[str, Any]],
                                               list[dict[str, Any]],
                                               list[dict[str, Any]],
                                               dict[str, Any]]:
        """Split the candidates into (keep, ask, drop) plus the query's facts.

        * **keep** — scored far enough ABOVE the threshold that the fitted
          formula is confident and the scorer is measurably worse than it.
          Injected without a question.
        * **ask** — the borderline band on either side of the threshold, plus
          every graph-reached candidate whatever its score, since the fitted
          weights lean on a keyword score those have no reason to have.
        * **drop** — scored far enough BELOW that nothing was ever promoted
          from there. Discarded without a question.

        Sync, and asks the scorer nothing: this only decides what is worth
        asking about.
        """
        cfg = self.retrieval
        sc = cfg.smart_retrieval
        thr = cfg.relevance.threshold
        hits, facts = self._candidates(query, max(cfg.candidates, cfg.top_n), expand=True)
        keep: list[dict[str, Any]] = []
        ask: list[dict[str, Any]] = []
        drop: list[dict[str, Any]] = []
        for h in hits:
            if h.get("_traversal", 0.0) > 0.0:
                ask.append(h)              # the formula cannot rank these at all
                continue
            score = self._relevance(h, facts.get("kw_best", 0.0))
            h["_fitted"] = score
            if score >= thr + sc.review_margin:
                keep.append(h)
            elif score >= thr - sc.promote_margin:
                ask.append(h)
            else:
                drop.append(h)
        return keep, ask, drop, facts

    async def _smart_select(
        self, keep: list[dict[str, Any]], ask: list[dict[str, Any]],
        state: list[dict[str, str]],
    ) -> list[dict[str, Any]] | None:
        """Ask the scorer about the borderline, then apply its answers.

        A candidate in ``ask`` is kept when it clears the bar its side of the
        threshold sets: ``promote_threshold`` to come in from below,
        ``retain_threshold`` to survive from above, and
        ``graph_promote_threshold`` for anything the graph reached (which
        scores low across the board — its relevance is usually indirect, and
        indirect relevance is not in the text the scorer reads).

        Returns ``None`` when the scorer could not answer — no client, a
        transport error, a timeout, a malformed reply. The caller then falls
        back to ordinary retrieval, so an outage costs precision, not memory.
        """
        cfg = self.retrieval
        sc = cfg.smart_retrieval
        thr = cfg.relevance.threshold
        if not ask:
            return list(keep)
        questions = {f"q{i}": self._smart_question(h["text"], sc.note_chars)
                     for i, h in enumerate(ask)}
        t0 = time.perf_counter()
        try:
            answers = await self.scorer.ask(state, questions,
                                             timeout_s=sc.timeout_s)
        except Exception as e:  # noqa: BLE001 — every failure is "no answer"
            log.warning("[%s] smart retrieval scorer failed, falling back to "
                        "ordinary retrieval: %r", self.agent_id, e)
            return None
        kept = list(keep)
        promoted = dropped = 0
        for i, h in enumerate(ask):
            try:
                p = float(answers[f"q{i}"]["noul"])
            except (KeyError, TypeError, ValueError):
                log.warning("[%s] smart retrieval: no usable answer for q%d",
                            self.agent_id, i)
                return None
            if h.get("_traversal", 0.0) > 0.0:
                bar, below = sc.graph_promote_threshold, True
            else:
                below = h.get("_fitted", 0.0) < thr
                bar = sc.promote_threshold if below else sc.retain_threshold
            if p >= bar:
                kept.append({**h, "_noul": p})
                promoted += below
            else:
                dropped += not below
        log.info("[%s] smart retrieval: %d kept unasked, %d asked "
                 "(%d promoted in, %d dropped out) -> %d in %.0fms",
                 self.agent_id, len(keep), len(ask), promoted, dropped,
                 len(kept), (time.perf_counter() - t0) * 1000)
        return kept

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

    def _graph(self) -> "nx.DiGraph | None":
        """The knowledge graph, cached on the file's mtime.

        The leg this replaced re-read and re-parsed the file on every call,
        which was affordable while it ran once per turn outside the gate.
        Expansion runs on the retrieval hot path, so the graph gets the same
        mtime cache the keyword index already has.
        """
        path = self.data_dir / "memory_graph.json"
        try:
            mtime = path.stat().st_mtime_ns
        except FileNotFoundError:
            return None
        if self._graph_cache is not None and self._graph_cache[0] == mtime:
            return self._graph_cache[1]
        try:
            with open(path) as f:
                G = nx.node_link_graph(json.load(f), edges="edges")
        except (json.JSONDecodeError, OSError):
            return None
        self._graph_cache = (mtime, G)
        return G

    def _graph_expand(self, seed_ids: Iterable[str], exclude: Iterable[str],
                      kw: dict[str, Any]) -> dict[str, float]:
        """Chunks one hop from the sections of ``seed_ids``, as {id: hops}.

        ``exclude`` is every id already in the pool, not merely the seeds: the
        seeds are the BEST candidates, so a neighbour of one is very often a
        candidate further down the same pool, and returning it again put a
        duplicate id into the Chroma backfill.

        The seeds are chunks the ordinary search already surfaced, NOT nodes
        matched against the query. That is the whole point: a graph that does
        its own query matching is a second, uncalibrated retrieval mechanism,
        and it behaved like one -- scoring node NAMES by summed IDF, it
        answered "how does compaction work" with a TTRPG section and two
        calendar entries, every hit matching "work". Anchoring on the fused
        pool removes the anchor-selection problem rather than tuning it.

        **Only section nodes nominate.** A concept node is a bolded phrase
        with no chunk of its own, so the only way to give it content is to
        follow it to the sections carrying it -- and since a phrase like
        "Notes" is bolded all over the store, that walk nominated a median 48
        of 443 chunks per message. Concepts stay as connective tissue and as
        waypoints the walk passes through; they never nominate. Measured
        against the same sample, sections-only yields a median of 1-6 new
        chunks depending on seed depth.
        """
        G = self._graph()
        if G is None:
            return {}
        section_of, by_section = kw["section_of"], kw["by_section"]
        labels = {tuple(tokenize(n)): n for n, d in G.nodes(data=True)
                  if d.get("type") == "section"}
        seeds = list(seed_ids)
        seen = set(exclude) | set(seeds)
        out: dict[str, float] = {}
        for sid in seeds:
            label = labels.get(section_of.get(sid, ()))
            if label is None:
                continue
            for nb in dict.fromkeys(list(G.successors(label)) + list(G.predecessors(label))):
                if G.nodes[nb].get("type") != "section":
                    continue
                for cid in by_section.get(tuple(tokenize(nb)), ()):
                    if cid not in seen:
                        out[cid] = 1.0
        return out

    def _build_markdown(self, query: str, top_n: int, compact: bool, gate: bool = True,
                        chunks: list[dict[str, Any]] | None = None) -> str:
        if chunks is None:
            chunks = self._relevant(query, top_n) if gate else self._hybrid_search(query, n=top_n)

        # A per-turn retrieval that found nothing injects nothing. The caller
        # skips the history row entirely on "", which is the whole saving: the
        # row costs tokens and re-ranks every turn, so an empty one is pure
        # overhead. Only the gated path goes quiet — an explicit search still
        # answers "no matches", because someone asked.
        if gate and not chunks:
            return ""

        # ONE list. There was a second section listing graph node names and
        # their neighbours: a heading, an arrow and another heading, with no
        # text under any of them. The model learned those sections existed and
        # nothing about what they said, so using one cost a `memory_search`
        # turn -- the thing this block exists to avoid -- and it was spent on
        # every turn the graph fired, which was 45 of 50 real messages at ~94
        # tokens, never cached because the block is rebuilt per turn.
        #
        # Graph-reached sections now arrive as ordinary candidates WITH their
        # content and compete for the same top_n slots, so the traversal costs
        # nothing extra and has to earn its place against the alternative.
        # Nothing here reports index freshness. Source hashes differ from the
        # index for the few minutes between an agent writing a memory and the
        # next periodic reindex -- so the warning fired BECAUSE the agent had
        # just saved something, told it its own memory was "stale" at the
        # moment that memory was most trustworthy, and cleared itself. The
        # agent can take no action on it either way; index health is an
        # operator's concern, and the reindex log lines are where it belongs.
        lines = ["## Auto-Retrieved Memory Context"]
        if chunks:
            lines.append(f"\n### Memory ({len(chunks)} results)")
            for r in chunks:
                snippet = r["text"][:150] if compact else r["text"][:300]
                tag = " (current)" if r.get("_current") else ""
                lines.append(f"- **[{r['section']}]{tag}** {snippet}")
        else:
            lines.append("\n_No strong matches in memory for this query._")
        return "\n".join(lines)

    async def retrieve_markdown(
        self, query: str, top_n: int = 5, compact: bool = True, gate: bool = True,
        state: list[dict[str, str]] | None = None,
    ) -> str:
        """``gate=True`` (per-turn retrieval) applies the relevance gate;
        ``gate=False`` (an explicit search) returns the ranked top_n.

        With ``smart_retrieval.enabled`` and a scorer wired, the gated path
        asks the scorer for a second opinion on the candidates near the
        threshold — promoting rejected ones it calls relevant, dropping
        retrieved ones it calls irrelevant — and injects everything that
        survives, with no ``top_n`` slice. ``state`` is the conversation the
        scorer judges against; it defaults to the query as a single user
        message.

        Only the gated path can take that route. An explicit ``memory_search``
        is someone asking to see the ranking, so it keeps returning it.
        """
        if not query.strip():
            return ""
        async with self.lock:
            loop = asyncio.get_running_loop()
            chunks = None
            if gate and self.retrieval.smart_retrieval.enabled and self.scorer is not None:
                chunks = await self._smart_retrieve(query, state, loop)
            return await loop.run_in_executor(
                None,
                functools.partial(self._build_markdown, query, top_n, compact, gate, chunks),
            )

    async def _smart_retrieve(
        self, query: str, state: list[dict[str, str]] | None, loop: Any,
    ) -> list[dict[str, Any]] | None:
        """Banding in a worker, the scorer over the wire, supersession after.

        ``None`` on any scorer failure, which sends the caller back to
        ordinary retrieval — the scorer is never required.
        """
        keep, ask, _drop, facts = await loop.run_in_executor(
            None, functools.partial(self._borderline, query))
        if not keep and not ask:
            self._note_best_distance(facts.get("best_distance"))
            return []
        kept = await self._smart_select(
            keep, ask, state or [{"role": "user", "content": query}])
        if kept is None:
            # The fallback re-runs ordinary retrieval, which notes the distance
            # itself. Noting it here too would count this turn twice in the
            # drift window.
            return None
        self._note_best_distance(facts.get("best_distance"))
        return self._resolve_supersession(self._kw["corpus"], kept) if kept else []


