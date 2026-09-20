"""Per-turn memory retrieval: the keyword index, common words, the relevance
gate, supersession ordering, and the drift warning — against a real (tiny)
ChromaDB index. Imports chromadb — run under the gateway venv.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pytest

from claw.config import (MemoryRetrievalConfig, RelevanceConfig, RetrievalCalibrationConfig,
                         _parse_memory_retrieval)
from claw.memory import MemoryIndex, tokenize

pytestmark = pytest.mark.filterwarnings("ignore")

TOPICS = [
    "The garden hose was replaced with a longer hose.",
    "Car batteries were both replaced at the shop; reminder set for five years out.",
    "The printer on the second floor needs new toner cartridges.",
    "Weekly grocery order moved from Tuesday to Thursday.",
    "The router firmware was upgraded to version 7.2.1.",
    "Dentist appointment rescheduled to the first week of March.",
    "Book club picked a science-fiction novel for next month.",
    "The kitchen faucet drips; plumber booked for Friday.",
    "Backups now run nightly at two in the morning.",
    "The dog's vaccination is due in October.",
    "Solar panel output dropped after the storm; inspection requested.",
    "Tax documents are filed in the blue folder.",
    "Bike tire pressure should be kept at sixty psi.",
    "The thermostat schedule lowers heat overnight.",
    "Passport renewal application was mailed on Monday.",
    "The laptop's fan is noisy under load; cleaning planned.",
    "Recipe for lentil soup saved to the cookbook file.",
    "Gym membership renews automatically in January.",
    "The smoke detector battery chirps; replacement bought.",
    "Streaming subscription was cancelled to save money.",
    "Coffee grinder setting nine works best for the espresso machine.",
    "Water heater temperature set to 120 degrees.",
]


def _write_store(ws: Path, topics=TOPICS, common="user-1 noted"):
    (ws / "memory").mkdir(parents=True, exist_ok=True)
    body = "".join(f"## Note {i}\n<!-- mem ts=2026-06-{i % 28 + 1:02d} -->\n{common}: {t}\n\n"
                   for i, t in enumerate(topics))
    (ws / "memory" / "2026-06-01.md").write_text(body)


async def _index(ws: Path, cfg: MemoryRetrievalConfig = MemoryRetrievalConfig()) -> MemoryIndex:
    idx = MemoryIndex("example", ws, cfg)
    await idx.warmup_async()
    await idx.reindex_if_stale(force=True)
    return idx


# --- tokenizer and common words ------------------------------------------------

def test_tokenize_strips_punctuation_keeps_versions():
    assert tokenize("Music. Is it v0.34.2? (claw.yaml, m-7f3a2c9e)") == \
        ["music", "is", "it", "v0.34.2", "claw.yaml", "m-7f3a2c9e"]


async def test_common_words_come_from_the_store(tmp_path):
    _write_store(tmp_path)
    idx = await _index(tmp_path)
    kw = idx._keyword_index()
    # In every memory -> common; distinctive words are not.
    assert {"user-1", "noted", "the"} <= kw["common"]
    assert "hose" not in kw["common"] and "espresso" not in kw["common"]


async def test_tiny_store_treats_no_word_as_common(tmp_path):
    _write_store(tmp_path, TOPICS[:3])
    idx = await _index(tmp_path)
    assert idx._keyword_index()["common"] == set()


async def test_keyword_side_ignores_the_envelope_header(tmp_path):
    _write_store(tmp_path, TOPICS + ["Matrix homeserver migrated to a new host on Saturday."])
    idx = await _index(tmp_path)
    hits, _ = idx._candidates("[matrix user-1 +5m Sat 2026-09-19 12:03] user-1: how is the espresso", 25)
    by_text = {h["text"]: h for h in hits}           # 23 memories, n=25: every one is a candidate
    matrix = next(h for t, h in by_text.items() if "homeserver" in t)
    assert matrix["_kw"] == 0.0, "the header's 'matrix'/'sat' must not score as keywords"
    assert next(h for t, h in by_text.items() if "espresso" in t)["_kw"] > 0


async def test_keyword_index_is_cached_until_reindex(tmp_path):
    _write_store(tmp_path)
    idx = await _index(tmp_path)
    first = idx._keyword_index()
    assert idx._keyword_index() is first, "no rebuild between reindexes"
    _write_store(tmp_path, TOPICS + ["A brand-new note about kayak storage."])
    await idx.reindex_if_stale(force=True)
    second = idx._keyword_index()
    assert second is not first and any("kayak" in d["text"] for d in second["corpus"])


# --- candidates and the gate -----------------------------------------------------

async def test_every_candidate_has_a_distance_matching_chroma(tmp_path):
    _write_store(tmp_path)
    idx = await _index(tmp_path)
    q = "user-1: when is the plumber coming for the faucet"
    hits, facts = idx._candidates(q, 20)
    col = idx.chroma_client.get_collection("memory_example", embedding_function=idx.embedder)
    res = col.query(query_texts=[q], n_results=col.count(), include=["distances"])
    ref = dict(zip(res["ids"][0], res["distances"][0]))
    assert hits and all(h["_distance"] is not None for h in hits)
    for h in hits:
        assert h["_distance"] == pytest.approx(ref[h["id"]], abs=1e-4)
    assert facts["best_distance"] == pytest.approx(min(ref.values()), abs=1e-4)


async def test_threshold_zero_is_the_plain_slice(tmp_path):
    _write_store(tmp_path)
    cfg = MemoryRetrievalConfig(relevance=RelevanceConfig(threshold=0))
    idx = await _index(tmp_path, cfg)
    q = "user-1: good morning"
    assert [h["id"] for h in idx._relevant(q, 5)] == [h["id"] for h in idx._candidates(q, 20)[0][:5]]


async def test_gate_keeps_only_candidates_that_clear_the_bar(tmp_path):
    """Mechanics, with weights that make keyword overlap decisive: the one
    memory sharing a meaningful word passes; nothing else does."""
    _write_store(tmp_path)
    rel = RelevanceConfig(threshold=0.5, offset=-4.0, distance=0.0, keyword=10.0, keyword_share=0.0)
    idx = await _index(tmp_path, MemoryRetrievalConfig(relevance=rel))
    kept = idx._relevant("user-1: what grinder setting for espresso", 5)
    assert [("espresso" in h["text"]) for h in kept] == [True]
    assert idx._relevant("user-1: hello there", 5) == []


async def test_default_gate_admits_the_on_topic_memory(tmp_path):
    _write_store(tmp_path)
    idx = await _index(tmp_path)
    kept = idx._relevant("user-1: when do the car batteries need replacing again?", 5)
    assert kept and "batteries" in kept[0]["text"]
    assert len(kept) < 5, "the gate trims unrelated memories"


async def test_explicit_search_is_ungated(tmp_path):
    _write_store(tmp_path)
    rel = RelevanceConfig(threshold=1.0)             # nothing could pass the gate
    idx = await _index(tmp_path, MemoryRetrievalConfig(relevance=rel))
    assert idx._relevant("user-1: espresso", 5) == []
    ungated = await idx.retrieve_markdown("user-1: espresso", top_n=5, gate=False)
    assert "espresso" in ungated
    gated = await idx.retrieve_markdown("user-1: espresso", top_n=5)
    # A gated turn that admits nothing injects no row at all — it used to say
    # "No strong matches", which cost tokens to report an absence.
    assert gated == ""


# --- supersession ---------------------------------------------------------------

async def test_a_directly_matched_successor_still_renders_last(tmp_path):
    ws = tmp_path
    (ws / "memory").mkdir(parents=True)
    (ws / "memory" / "2026-06-10.md").write_text(
        "## Pooling\n<!-- mem ts=2026-06-10 id=m-old status=active supersededBy=m-new -->\n"
        "Connection pooling uses the ALPHA pooler.\n")
    (ws / "memory" / "2026-06-20.md").write_text(
        "## Pooling update\n<!-- mem ts=2026-06-20 id=m-new status=active -->\n"
        "Connection pooling moved to the BETA pooler.\n")
    idx = await _index(ws)
    hits = idx._hybrid_search("connection pooling pooler", n=5)
    assert hits[-1]["id"] == "memory/2026-06-20.md:0" and hits[-1]["_current"] is True
    assert sum(h["id"] == "memory/2026-06-20.md:0" for h in hits) == 1
    # The cached corpus itself carries no display flag.
    assert not any("_current" in d for d in idx._keyword_index()["corpus"])


# --- drift warning ---------------------------------------------------------------

def test_drift_warning_fires_only_out_of_band(tmp_path, caplog):
    cal = RetrievalCalibrationConfig(best_distance=0.97, tolerance=0.25)
    idx = MemoryIndex("example", tmp_path, MemoryRetrievalConfig(calibration=cal))
    with caplog.at_level(logging.WARNING, logger="claw.memory"):
        for _ in range(100):
            idx._note_best_distance(1.0)
    assert not caplog.records
    with caplog.at_level(logging.WARNING, logger="claw.memory"):
        for _ in range(100):
            idx._note_best_distance(1.45)
    assert any("drift" in r.getMessage() for r in caplog.records)


# --- config ------------------------------------------------------------------------

def test_parse_memory_retrieval_defaults_and_overrides():
    assert _parse_memory_retrieval(None) == MemoryRetrievalConfig()
    cfg = _parse_memory_retrieval({"top_n": 3, "candidates": 10, "common_word_max_share": "0.2",
                                   "relevance": {"threshold": "0.5"},
                                   "calibration": {"best_distance": 1.1}})
    assert cfg.top_n == 3 and cfg.candidates == 10 and cfg.common_word_max_share == 0.2
    assert cfg.relevance == dataclasses.replace(RelevanceConfig(), threshold=0.5)
    assert cfg.calibration.best_distance == 1.1 and cfg.calibration.tolerance == 0.25


@pytest.mark.parametrize("d, exc, fragment", [
    ({"relevance": {"threshhold": 0.5}}, TypeError, ""),
    ({"calibration": {"best": 1.0}}, TypeError, ""),
    ({"candidate": 10}, TypeError, ""),
    ({"relevance": {"threshold": 1.5}}, ValueError, "threshold"),
    ({"common_word_max_share": 0}, ValueError, "common_word_max_share"),
    ({"top_n": 8, "candidates": 5}, ValueError, "candidates"),
])
def test_parse_memory_retrieval_is_strict(d, exc, fragment):
    with pytest.raises(exc, match=fragment):
        _parse_memory_retrieval(d)


async def test_generic_words_do_not_change_the_keyword_ranking(tmp_path):
    """A question asked conversationally searches as the bare topic does: the
    keyword side scores on words that carry a topic, and "what did you do
    about" carries none. The vector side still embeds the sentence as written."""
    _write_store(tmp_path)
    idx = await _index(tmp_path)
    bare = [h["section"] for h in idx.search("espresso grinder", n=5)]
    asked = [h["section"] for h in idx.search("what did you do about the espresso grinder", n=5)]
    assert bare[0] == asked[0]
