"""Forgetory (memory curation) — marker parsing, supersession auto-follow,
archive exclusion, working-set selection, and a guarded SoT-rebuild e2e.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from claw.memory import MemoryIndex
from claw import memory_curate as mc


# --- marker parsing (pure) --------------------------------------------------

def test_extract_marker_strips_and_parses():
    body = (
        "<!-- mem ts=2026-06-28 id=m-7f3a2c9e status=active "
        "supersededBy=m-1a0b22f4 -->\n"
        "Chose pgbouncer in transaction mode."
    )
    clean, fields = MemoryIndex._extract_marker(body)
    assert clean == "Chose pgbouncer in transaction mode."
    assert fields == {
        "ts": "2026-06-28", "id": "m-7f3a2c9e",
        "status": "active", "supersededBy": "m-1a0b22f4",
    }


def test_extract_marker_absent_is_noop():
    body = "Just prose, no marker."
    clean, fields = MemoryIndex._extract_marker(body)
    assert clean == body
    assert fields == {}


def test_extract_marker_only_leading_line_counts():
    # A mem-looking comment that isn't the first line is left alone.
    body = "Real content.\n<!-- mem ts=2026-06-28 -->"
    clean, fields = MemoryIndex._extract_marker(body)
    assert fields == {}
    assert clean == body


def test_marker_metadata_defaults_status_and_omits_absent():
    assert MemoryIndex._marker_metadata({}) == {"status": "active"}
    meta = MemoryIndex._marker_metadata({"ts": "2026-06-28", "id": "m-abc12345"})
    assert meta == {"status": "active", "ts": "2026-06-28", "mem_id": "m-abc12345"}
    assert "superseded_by" not in meta  # never store None


def test_parse_markdown_extracts_marker_and_skips_archived(tmp_path: Path):
    p = tmp_path / "2026-06-20.md"
    p.write_text(
        "## Live fact\n"
        "<!-- mem ts=2026-06-20 id=m-aaaa1111 status=active -->\n"
        "Currently true.\n\n"
        "## Dead fact\n"
        "<!-- mem ts=2026-06-01 id=m-bbbb2222 status=archived -->\n"
        "Should be excluded from the index.\n"
    )
    chunks = MemoryIndex._parse_markdown(p)
    sections = {c["metadata"]["section"]: c for c in chunks}
    assert "Live fact" in sections
    assert "Dead fact" not in sections          # archived -> not indexed
    live = sections["Live fact"]
    assert live["content"] == "Currently true."  # marker stripped
    assert live["metadata"]["mem_id"] == "m-aaaa1111"
    assert live["metadata"]["status"] == "active"


# --- supersession auto-follow (pure) ---------------------------------------

def _doc(cid, mem_id=None, superseded_by=None, text="x", section="s"):
    return {
        "id": cid, "text": text, "section": section,
        "mem_id": mem_id, "status": "active", "superseded_by": superseded_by,
    }


def test_resolve_supersession_appends_head_last():
    corpus = [
        _doc("f:0", mem_id="m-old", superseded_by="m-new", section="Old"),
        _doc("f:1", mem_id="m-new", section="New"),
    ]
    hits = [corpus[0]]  # only the stale memory was a search hit
    out = MemoryIndex._resolve_supersession(corpus, hits)
    assert [d["id"] for d in out] == ["f:0", "f:1"]   # current head last
    assert out[-1].get("_current") is True


def test_resolve_supersession_follows_chain_to_head():
    corpus = [
        _doc("f:0", mem_id="m-a", superseded_by="m-b"),
        _doc("f:1", mem_id="m-b", superseded_by="m-c"),
        _doc("f:2", mem_id="m-c"),
    ]
    out = MemoryIndex._resolve_supersession(corpus, [corpus[0]])
    assert [d["id"] for d in out] == ["f:0", "f:2"]   # skips middle, lands on head


def test_resolve_supersession_breaks_cycle():
    corpus = [
        _doc("f:0", mem_id="m-a", superseded_by="m-b"),
        _doc("f:1", mem_id="m-b", superseded_by="m-a"),
    ]
    out = MemoryIndex._resolve_supersession(corpus, [corpus[0]])
    # Terminates safely: the chain loops a->b->a back to the origin (already a
    # hit), so nothing is appended and there's no infinite loop / duplicate.
    ids = [d["id"] for d in out]
    assert ids == ["f:0"]
    assert len(ids) == len(set(ids))


def test_resolve_supersession_dangling_target_is_safe():
    corpus = [_doc("f:0", mem_id="m-a", superseded_by="m-ghost")]
    out = MemoryIndex._resolve_supersession(corpus, [corpus[0]])
    assert [d["id"] for d in out] == ["f:0"]          # nothing appended


def test_resolve_supersession_no_duplicate_when_head_already_hit():
    corpus = [
        _doc("f:0", mem_id="m-old", superseded_by="m-new"),
        _doc("f:1", mem_id="m-new"),
    ]
    out = MemoryIndex._resolve_supersession(corpus, [corpus[0], corpus[1]])
    assert [d["id"] for d in out] == ["f:0", "f:1"]   # head not appended twice


# --- archive.md exclusion (pure) -------------------------------------------

def test_daily_notes_excludes_archive(tmp_path: Path):
    ws = tmp_path / "ws"
    (ws / "memory").mkdir(parents=True)
    (ws / "memory" / "2026-06-28.md").write_text("## a\nbody\n")
    (ws / "memory" / "archive.md").write_text("## old\nbody\n")
    (ws / "memory" / "2026-06-28-note.md").write_text("journal only\n")
    idx = MemoryIndex("example", ws)
    names = {p.name for p in idx._daily_notes()}
    assert names == {"2026-06-28.md"}                 # only date-shaped files


# --- working-set briefing selection (pure, fake index) ---------------------

class _FakeCurMemory:
    """Just enough of MemoryIndex for _build_briefing: collect_sections,
    search, _daily_notes."""

    def __init__(self, sections, daily_names):
        self._sections = sections
        self._daily = [Path(n) for n in daily_names]

    def collect_sections(self):
        return self._sections

    def search(self, text, n=8):
        return []  # neighbours irrelevant to selection logic

    def _daily_notes(self):
        return self._daily


def _section(source, section, content="content", **meta):
    return {"content": content, "metadata": {"source": source, "section": section, **meta}}


def _h(content="content"):
    import hashlib
    return hashlib.sha1(content.encode()).hexdigest()


def test_select_candidate_files_nightly_changed_and_recent():
    today = "2026-06-28"
    sections = [
        _section("memory/2026-06-28.md", "Today live"),        # excluded (today)
        _section("memory/2026-06-27.md", "Changed one"),       # changed -> in
        _section("memory/2026-06-26.md", "Unchanged recent"),  # recent -> in
        _section("MEMORY.md", "Old root unchanged"),           # unchanged, not recent -> out
    ]
    daily = ["2026-06-28.md", "2026-06-27.md", "2026-06-26.md"]
    mem = _FakeCurMemory(sections, daily)
    prev = {
        "memory/2026-06-26.md#Unchanged recent": _h(),
        "MEMORY.md#Old root unchanged": _h(),
    }
    ordered, by_file, cur = mc._select_candidate_files(
        mem, today, recent_window_days=14, full_corpus=False, prev_hashes=prev,
    )
    assert ordered == ["memory/2026-06-26.md", "memory/2026-06-27.md"]
    assert "memory/2026-06-28.md" not in ordered    # today excluded
    assert "MEMORY.md" not in ordered               # unchanged + out of window


def test_select_candidate_files_fullcorpus_skips_fully_id_marked():
    today = "2026-06-28"
    sections = [
        _section("memory/2026-06-10.md", "A", mem_id="m-1"),   # fully id'd -> skip
        _section("memory/2026-06-11.md", "B"),                 # no id -> take
    ]
    mem = _FakeCurMemory(sections, ["2026-06-10.md", "2026-06-11.md"])
    ordered, _, _ = mc._select_candidate_files(
        mem, today, recent_window_days=14, full_corpus=True, prev_hashes={},
    )
    assert ordered == ["memory/2026-06-11.md"]      # resumable bootstrap


def test_select_candidate_files_none_when_only_today():
    today = "2026-06-28"
    sections = [_section("memory/2026-06-28.md", "Only today")]
    mem = _FakeCurMemory(sections, ["2026-06-28.md"])
    ordered, _, _ = mc._select_candidate_files(
        mem, today, recent_window_days=14, full_corpus=False, prev_hashes={},
    )
    assert ordered == []


def test_build_file_briefing_names_file_and_includes_sections():
    sections = [_section("memory/2026-06-27.md", "Topic", content="alpha beta gamma")]
    mem = _FakeCurMemory(sections, ["2026-06-27.md"])
    b = mc._build_file_briefing(mem, "memory/2026-06-27.md", sections, "2026-06-28", 8)
    assert "memory/2026-06-27.md" in b
    assert "Topic" in b
    assert "alpha beta gamma" in b


# --- supersession-review briefing (whole-corpus, window-independent) --------

def test_build_supersession_briefing_empty_when_none():
    sections = [_section("memory/2026-06-01.md", "A", mem_id="m-a", ts="2026-06-01")]
    mem = _FakeCurMemory(sections, [])
    assert mc._build_supersession_briefing(mem, "2026-06-30", 30) == ""


def test_build_supersession_briefing_age_from_superseder_and_sorts_stalest_first():
    sections = [
        # Old1 -> New1 (recorded 2026-05-01 => 60 days stale on 2026-06-30)
        _section("memory/2026-03-01.md", "Old one", mem_id="m-old1", superseded_by="m-new1"),
        _section("memory/2026-05-01.md", "New one", mem_id="m-new1", ts="2026-05-01"),
        # Old2 -> New2 (recorded 2026-06-28 => 2 days stale)
        _section("memory/2026-04-01.md", "Old two", mem_id="m-old2", superseded_by="m-new2"),
        _section("memory/2026-06-28.md", "New two", mem_id="m-new2", ts="2026-06-28"),
    ]
    mem = _FakeCurMemory(sections, [])
    b = mc._build_supersession_briefing(mem, "2026-06-30", 30)
    # age measured from the *superseding* memory's ts, not the old one's
    assert "60 days" in b and "2 days" in b
    # most-stale first
    assert b.index("Old one") < b.index("Old two")
    # the archive-days threshold is surfaced to the curator
    assert "30+" in b


def test_build_supersession_briefing_missing_replacement_is_uncertain_not_archive():
    sections = [
        _section("memory/2026-04-01.md", "Orphan", mem_id="m-orph", superseded_by="m-gone"),
    ]
    mem = _FakeCurMemory(sections, [])
    b = mc._build_supersession_briefing(mem, "2026-06-30", 30)
    assert "Orphan" in b
    assert "MISSING" in b
    # a dangling pointer is the uncertain case, not an archive trigger
    assert "UNCERTAIN" in b


def test_build_file_briefing_excludes_today_neighbours():
    class _M:
        def search(self, text, n=8):
            return [
                {"id": "memory/2026-06-30.md:TodayNbr", "mem_id": None,
                 "section": "TodayNbr", "text": "today neighbour"},
                {"id": "memory/2026-06-20.md:OldNbr", "mem_id": "m-o",
                 "section": "OldNbr", "text": "old neighbour"},
            ]

    sections = [_section("memory/2026-06-27.md", "Topic", content="alpha")]
    b = mc._build_file_briefing(_M(), "memory/2026-06-27.md", sections, "2026-06-30", 8)
    assert "OldNbr" in b          # non-today neighbour kept
    assert "TodayNbr" not in b    # today's note excluded (un-editable, no id)


def test_curator_search_tool_exposes_ids_and_excludes_today():
    import asyncio
    from claw.tools.memory_search import build_curator_search_tool

    class _M:
        def __init__(self):
            self.lock = asyncio.Lock()  # tool holds it during the read

        def search(self, q, n=8):
            return [
                {"id": "memory/2026-06-01.md:Sec", "mem_id": "m-x",
                 "section": "Sec", "superseded_by": "m-y", "text": "hello world"},
                {"id": "memory/2026-06-30.md:Fresh", "mem_id": None,
                 "section": "Fresh", "text": "today flush"},
            ]

    tool = build_curator_search_tool(_M(), "2026-06-30")
    out = asyncio.run(tool.run({"query": "hello"}))
    assert "m-x" in out               # id exposed (agent-facing tool hides it)
    assert "supersededBy=m-y" in out  # existing chain visible
    assert "memory/2026-06-01.md" in out and "Sec" in out
    assert "Fresh" not in out         # today's note filtered out


def test_curator_search_tool_no_matches_when_only_today():
    import asyncio
    from claw.tools.memory_search import build_curator_search_tool

    class _M:
        def __init__(self):
            self.lock = asyncio.Lock()

        def search(self, q, n=8):
            return [{"id": "memory/2026-06-30.md:Fresh", "mem_id": None,
                     "section": "Fresh", "text": "today flush"}]

    out = asyncio.run(build_curator_search_tool(_M(), "2026-06-30").run({"query": "x"}))
    assert out == "(no matches)"


def test_curation_config_has_superseded_archive_days_default():
    from claw.config import MemoryCurationConfig
    assert MemoryCurationConfig().superseded_archive_days == 30


def test_recent_window_sources_respects_cutoff(tmp_path: Path):
    ws = tmp_path / "ws"
    (ws / "memory").mkdir(parents=True)
    for name in ("2026-06-28.md", "2026-06-20.md", "2026-05-01.md"):
        (ws / "memory" / name).write_text("## a\nbody\n")
    idx = MemoryIndex("example", ws)
    srcs = mc._recent_window_sources(idx, "2026-06-28", days=14)
    assert srcs == {"memory/2026-06-28.md", "memory/2026-06-20.md"}  # 05-01 too old


# --- record_action ledger tool (pure) --------------------------------------

def _logger_with_capture():
    import logging
    lg = logging.getLogger("test.curator.record")
    lg.handlers.clear()
    lg.setLevel(logging.INFO)
    lg.propagate = False
    records: list[str] = []

    class _H(logging.Handler):
        def emit(self, r):
            records.append(r.getMessage())

    lg.addHandler(_H())
    return lg, records


def test_record_action_requires_all_fields():
    import asyncio
    lg, records = _logger_with_capture()
    tool = mc.build_record_action_tool(lg, "quint", dry_run=False)
    out = asyncio.run(tool.run({"action": "archive", "target": "f#s"}))  # no reason
    assert out.startswith("error:")
    assert records == []  # nothing logged when the call is malformed


def test_record_action_logs_valid_decision():
    import asyncio
    lg, records = _logger_with_capture()
    tool = mc.build_record_action_tool(lg, "quint", dry_run=False)
    out = asyncio.run(tool.run({
        "action": "Archive",  # case-insensitive
        "target": "memory/2026-05-13.md#Daily Log (m-1a2b3c4d)",
        "reason": "lapsed May appointment reminders, superseded Ollama version",
    }))
    assert out == "recorded"
    assert len(records) == 1
    line = records[0]
    assert "ACTION ARCHIVE" in line
    assert "m-1a2b3c4d" in line
    assert "lapsed May appointment" in line
    assert "DRY-RUN" not in line


def test_record_action_dry_run_prefixes_but_still_logs():
    import asyncio
    lg, records = _logger_with_capture()
    tool = mc.build_record_action_tool(lg, "quint", dry_run=True)
    asyncio.run(tool.run({"action": "dedup", "target": "f#s (m-x)", "reason": "dup of m-y"}))
    assert records and records[0].startswith("[quint] DRY-RUN ACTION DEDUP")


def test_record_action_unknown_label_still_recorded_but_flagged():
    import asyncio
    lg, records = _logger_with_capture()
    tool = mc.build_record_action_tool(lg, "quint", dry_run=False)
    out = asyncio.run(tool.run({"action": "delete", "target": "f#s", "reason": "why"}))
    assert "not one of" in out          # nudged back toward the valid set
    assert len(records) == 1             # but the decision is not lost
    assert "ACTION DELETE" in records[0]


def test_curator_tools_adds_record_action_and_gates_dryrun_wrap():
    import asyncio
    from claw.tools.base import Tool

    async def _live(inp):
        return "did it"

    tools = {
        "read_file": Tool("read_file", "", {}, _live),
        "write_file": Tool("write_file", "", {}, _live),
    }
    lg, _ = _logger_with_capture()

    # Live pass: mutating tool passes through unwrapped (really executes),
    # record_action is present.
    live = mc._curator_tools(tools, lg, "quint", dry_run=False)
    assert "record_action" in live
    assert asyncio.run(live["write_file"].run({"path": "x", "content": "y"})) == "did it"

    # Dry run: mutating tool is stubbed (does NOT execute), read tool is not.
    dry = mc._curator_tools(tools, lg, "quint", dry_run=True)
    assert asyncio.run(dry["write_file"].run({"path": "x", "content": "y"})).startswith("[dry-run]")
    assert asyncio.run(dry["read_file"].run({"path": "x"})) == "did it"


# --- end-to-end: real index round-trip + SoT rebuild -----------------------

@pytest.mark.filterwarnings("ignore")
async def test_e2e_supersession_and_sot_rebuild(tmp_path: Path):
    pytest.importorskip("chromadb")
    ws = tmp_path / "ws"
    (ws / "memory").mkdir(parents=True)
    (ws / "MEMORY.md").write_text("## Root\n<!-- mem ts=2026-06-01 id=m-root status=active -->\nRoot fact.\n")
    # Old memory matches the query; the successor is phrased so it does NOT
    # match directly — it can only reach the result set via the pointer, which
    # is exactly the auto-follow append path we want to exercise.
    (ws / "memory" / "2026-06-10.md").write_text(
        "## Pooling\n"
        "<!-- mem ts=2026-06-10 id=m-old status=active supersededBy=m-new -->\n"
        "We standardized on the ALPHA connection pooler for Postgres.\n"
    )
    (ws / "memory" / "2026-06-20.md").write_text(
        "## Pooling update\n"
        "<!-- mem ts=2026-06-20 id=m-new status=active -->\n"
        "Migrated everything over to the BETA layer instead.\n"
    )

    idx = MemoryIndex("example", ws)
    await idx.warmup_async()
    await idx.reindex_if_stale(force=True)

    hits = idx._hybrid_search("ALPHA connection pooler Postgres", n=5)
    ids = [h["id"] for h in hits]
    # Old matched directly; the successor is pulled in via the pointer and
    # rendered LAST (current truth), flagged _current.
    assert "memory/2026-06-10.md:0" in ids
    assert ids[-1] == "memory/2026-06-20.md:0"
    assert hits[-1].get("_current") is True

    # SoT rebuild: nuke the derived index, rebuild from markdown, status intact.
    # Clear chromadb's per-path client cache so a second client at the same
    # path behaves like a fresh process (which is how a real rebuild happens).
    from chromadb.api.shared_system_client import SharedSystemClient
    SharedSystemClient.clear_system_cache()
    shutil.rmtree(idx.data_dir)
    idx2 = MemoryIndex("example", ws)
    await idx2.warmup_async()
    await idx2.reindex_if_stale(force=True)
    import json
    corpus = json.loads((idx2.data_dir / "bm25_corpus.json").read_text())
    by_mem = {d["mem_id"]: d for d in corpus if d.get("mem_id")}
    assert by_mem["m-old"]["superseded_by"] == "m-new"
    assert by_mem["m-new"]["status"] == "active"
