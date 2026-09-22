"""Per-turn memory retrieval: the keyword index, common words, the relevance
gate, supersession ordering, and the drift warning — against a real (tiny)
ChromaDB index. Imports chromadb — run under the gateway venv.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import pytest

from claw.config import (MemoryRetrievalConfig, RelevanceConfig, RetrievalCalibrationConfig,
                         _parse_memory_retrieval)
import claw.memory as memory_mod
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


# --- headings are searchable ------------------------------------------------------

def _write_titled(ws: Path):
    """A store where one note's TITLE carries the topic and its body does not."""
    (ws / "memory").mkdir(parents=True, exist_ok=True)
    body = "".join(f"## Note {i}\n<!-- mem ts=2026-06-{i % 28 + 1:02d} -->\nuser-1 noted: {t}\n\n"
                   for i, t in enumerate(TOPICS))
    body += ("## Thermostat scheduling\n<!-- mem ts=2026-06-20 -->\n"
             "user-1 noted: it now drops to 61 overnight and back up by seven.\n\n")
    (ws / "memory" / "2026-06-01.md").write_text(body)


async def test_a_title_is_searchable_though_its_words_are_absent_from_the_body(tmp_path):
    _write_titled(tmp_path)
    idx = await _index(tmp_path)
    corpus = idx._keyword_index()["corpus"]
    titled = next(r for r in corpus if r["section"] == "Thermostat scheduling")
    assert "thermostat" not in titled["text"].lower(), "the body must not carry the word itself"

    hits, _ = idx._candidates("what did we decide about thermostat scheduling", 20)
    hit = next((h for h in hits if h["section"] == "Thermostat scheduling"), None)
    assert hit is not None and hit["_kw"] > 0, "a title's words must reach the keyword leg"


async def test_indexing_a_title_does_not_put_it_in_the_rendered_text(tmp_path):
    """Display reads `text`; the block already prints the section name above it."""
    _write_titled(tmp_path)
    idx = await _index(tmp_path)
    titled = next(r for r in idx._keyword_index()["corpus"]
                  if r["section"] == "Thermostat scheduling")
    assert titled["text"].startswith("user-1 noted:")


async def test_bodies_stay_searchable_when_titles_are_indexed(tmp_path):
    _write_titled(tmp_path)
    idx = await _index(tmp_path)
    hits, _ = idx._candidates("user-1: how is the espresso machine grinder", 20)
    assert any("espresso" in h["text"] for h in hits if h["_kw"] > 0)


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


async def test_a_source_written_since_the_index_is_not_reported_to_the_agent(tmp_path):
    """Writing a memory must not change what retrieval tells the agent.

    Source hashes differ from the index for the minutes between an agent
    appending to a note and the next periodic reindex. That used to surface as
    `WARNING: MEMORY OUT OF SYNC — index may be stale` inside the retrieval
    block, plus a `**Sync:**` line, plus an exemption that made a
    nothing-to-inject turn emit a block anyway. All three fired BECAUSE the
    agent had just saved something, and cleared themselves minutes later. The
    agent can act on none of it.
    """
    _write_store(tmp_path)
    rel = RelevanceConfig(threshold=1.0)             # nothing can pass the gate
    idx = await _index(tmp_path, MemoryRetrievalConfig(relevance=rel))

    # Exactly what a memory_flush does: append to an indexed source, and do not
    # reindex. The store is now genuinely out of sync -- that part was never wrong.
    note = tmp_path / "memory" / "2026-06-01.md"
    note.write_text(note.read_text() + "\n## Fresh\nJust written, not yet indexed.\n\n")
    assert idx._compute_source_hashes() != json.loads(
        (tmp_path / ".memory" / "sync_state.json").read_text())["sourceHashes"]

    assert await idx.retrieve_markdown("user-1: espresso", top_n=5) == "", \
        "a turn with nothing to inject still injects nothing"

    searched = await idx.retrieve_markdown("user-1: espresso", top_n=5, gate=False)
    assert "espresso" in searched
    for leak in ("OUT OF SYNC", "stale", "**Sync:**"):
        assert leak not in searched, f"index freshness leaked to the agent: {leak!r}"


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


# --- the index knows which code built it ------------------------------------

async def test_a_new_index_version_forces_a_rebuild(tmp_path):
    """Staleness is judged by hashing SOURCE FILES, which cannot notice that
    the code reading them now produces something different. A store then
    reports itself in sync forever while serving an index the previous version
    built -- which has happened for a whole release."""
    _write_store(tmp_path)
    idx = await _index(tmp_path)
    assert idx.needs_reindex() == ([], []), "freshly built, nothing to do"

    state_path = idx.data_dir / "sync_state.json"
    state = json.loads(state_path.read_text())
    assert state["indexVersion"] == memory_mod.INDEX_VERSION
    state["indexVersion"] = memory_mod.INDEX_VERSION - 1
    state_path.write_text(json.dumps(state))

    changed, _ = idx.needs_reindex()
    assert changed, "an index built by another version is stale, sources or not"


async def test_a_legacy_state_file_without_a_version_rebuilds(tmp_path):
    _write_store(tmp_path)
    idx = await _index(tmp_path)
    state_path = idx.data_dir / "sync_state.json"
    state = json.loads(state_path.read_text())
    del state["indexVersion"]
    state_path.write_text(json.dumps(state))
    assert idx.needs_reindex()[0], "no version recorded means unknown, so rebuild"


# --- smart retrieval -----------------------------------------------------------

class _FakeScorer:
    """Answers every noul question with a fixed probability, or raises.

    Records the state it was given, since what the scorer reads as the
    conversation is the thing smart retrieval exists to get right.
    """

    def __init__(self, noul: float = 0.9, error: Exception | None = None) -> None:
        self.noul = noul
        self.error = error
        self.calls: list[tuple[list[dict[str, str]], dict[str, dict]]] = []
        self.timeouts: list[float | None] = []

    async def ask(self, state, questions, model: str = "local",
                  timeout_s: float | None = None):
        self.calls.append((state, questions))
        self.timeouts.append(timeout_s)
        if self.error is not None:
            raise self.error
        return {q: {"type": "noul", "noul": self.noul} for q in questions}


def _smart_cfg(**kw) -> MemoryRetrievalConfig:
    """Enabled, with a WIDE band.

    This store is tiny and its scores are bimodal — one match near 0.9 and
    everything else under 0.1 — so the shipped margins (0.10 / 0.30) leave the
    band empty and nothing is ever asked about. That is correct behaviour
    (pinned by its own test below) and useless for exercising the mechanism,
    so these tests widen the band until real candidates fall inside it.
    """
    from claw.config import SmartRetrievalConfig
    kw = {"promote_margin": 0.40, "review_margin": 0.05, **kw}
    return MemoryRetrievalConfig(smart_retrieval=SmartRetrievalConfig(enabled=True, **kw))


async def test_a_confident_store_asks_nothing_and_costs_nothing(tmp_path):
    """When the formula is decisive, the band is empty and the scorer is never
    called — so the feature costs nothing on the turns it cannot help."""
    from claw.config import SmartRetrievalConfig
    ws = tmp_path / "ws"
    _write_store(ws)
    shipped = MemoryRetrievalConfig(smart_retrieval=SmartRetrievalConfig(enabled=True))
    idx = await _index(ws, shipped)
    idx.scorer = _FakeScorer(noul=0.99)
    plain = await (await _index(ws)).retrieve_markdown(QUERY, top_n=5)
    assert await idx.retrieve_markdown(QUERY, top_n=5) == plain
    assert idx.scorer.calls == []


QUERY = "user-1: espresso grinder setting"


async def test_only_the_borderline_is_asked_about(tmp_path):
    """Candidates far above the threshold are kept unasked and ones far below
    are dropped unasked. That band is what keeps the feature affordable."""
    ws = tmp_path / "ws"
    _write_store(ws)
    idx = await _index(ws, _smart_cfg())
    keep, ask, drop, _ = idx._borderline(QUERY)
    thr = idx.retrieval.relevance.threshold
    sc = idx.retrieval.smart_retrieval
    assert all(h["_fitted"] >= thr + sc.review_margin for h in keep)
    assert all(h["_fitted"] < thr - sc.promote_margin
               for h in drop if not h.get("_traversal"))
    assert ask, "nothing was borderline, so this store cannot exercise the feature"
    # Every graph-reached candidate is asked about whatever it scored: the
    # fitted weights lean on a keyword score it has no reason to have.
    assert all(h.get("_traversal", 0.0) == 0.0 for h in keep + drop)


async def test_a_rejected_candidate_the_scorer_likes_is_promoted_in(tmp_path):
    ws = tmp_path / "ws"
    _write_store(ws)
    idx = await _index(ws, _smart_cfg())
    plain = await (await _index(ws)).retrieve_markdown(QUERY, top_n=5)
    idx.scorer = _FakeScorer(noul=0.99)          # says yes to everything
    block = await idx.retrieve_markdown(QUERY, top_n=5)
    n = lambda b: len([l for l in b.splitlines() if l.startswith("- **[")])
    assert n(block) > n(plain), "a scorer saying yes to all should widen the set"


async def test_a_retrieved_candidate_the_scorer_dislikes_is_dropped(tmp_path):
    ws = tmp_path / "ws"
    _write_store(ws)
    idx = await _index(ws, _smart_cfg())
    idx.scorer = _FakeScorer(noul=0.0)           # says no to everything
    keep, _ask, _drop, _ = idx._borderline(QUERY)
    block = await idx.retrieve_markdown(QUERY, top_n=5)
    injected = [l for l in block.splitlines() if l.startswith("- **[")]
    # Only the confidently-above-threshold entries survive a scorer that
    # rejects everything it is asked about.
    assert len(injected) == len(keep)


async def test_there_is_no_top_n_slice_on_this_path(tmp_path):
    """The fixed five is gone here: a turn may inject more than top_n."""
    ws = tmp_path / "ws"
    _write_store(ws)
    idx = await _index(ws, _smart_cfg())
    idx.scorer = _FakeScorer(noul=0.99)
    keep, ask, _drop, _ = idx._borderline(QUERY)
    block = await idx.retrieve_markdown(QUERY, top_n=2)
    injected = [l for l in block.splitlines() if l.startswith("- **[")]
    assert len(injected) == len(keep) + len(ask) > 2


async def test_falls_back_to_ordinary_retrieval_when_the_scorer_fails(tmp_path):
    """A scorer outage must cost precision, never retrieval."""
    ws = tmp_path / "ws"
    _write_store(ws)
    plain = await (await _index(ws)).retrieve_markdown(QUERY, top_n=5)
    idx = await _index(ws, _smart_cfg())
    idx.scorer = _FakeScorer(error=RuntimeError("scorer down"))
    assert await idx.retrieve_markdown(QUERY, top_n=5) == plain


async def test_is_skipped_entirely_without_a_scorer(tmp_path):
    ws = tmp_path / "ws"
    _write_store(ws)
    plain = await (await _index(ws)).retrieve_markdown(QUERY, top_n=5)
    idx = await _index(ws, _smart_cfg())
    assert idx.scorer is None
    assert await idx.retrieve_markdown(QUERY, top_n=5) == plain


async def test_reads_the_conversation_state_it_is_given(tmp_path):
    ws = tmp_path / "ws"
    _write_store(ws)
    idx = await _index(ws, _smart_cfg())
    idx.scorer = _FakeScorer(noul=0.99)
    state = [{"role": "user", "content": "how do I set the grinder?"},
             {"role": "assistant", "content": "Setting nine."},
             {"role": "user", "content": QUERY}]
    _keep, ask, _drop, _ = idx._borderline(QUERY)
    await idx.retrieve_markdown(QUERY, top_n=5, state=state)
    seen, questions = idx.scorer.calls[0]
    assert seen == state
    # The note travels in the QUESTION, not the state: one prefill per turn,
    # not one per candidate. One question per asked candidate, each carrying
    # that candidate's own text.
    assert all(q["type"] == "noul" for q in questions.values())
    assert len(questions) == len(ask)
    asked_text = [h["text"][:40] for h in ask]
    for q in questions.values():
        assert any(t in q["instructions"] for t in asked_text)


async def test_the_note_is_clipped_to_note_chars(tmp_path):
    """note_chars bounds only what the SCORER reads; the stored note is
    untouched and the full text is what gets injected."""
    ws = tmp_path / "ws"
    _write_store(ws)
    idx = await _index(ws, _smart_cfg(note_chars=20))
    idx.scorer = _FakeScorer(noul=0.99)
    await idx.retrieve_markdown(QUERY, top_n=5)
    _state, questions = idx.scorer.calls[0]
    for q in questions.values():
        note = q["instructions"].split("reply:\n\n")[1].split("\n\nIs this")[0]
        assert len(note) <= 20


async def test_a_continuation_message_shortlists_nothing(tmp_path):
    """"looks good" carries no content word, so at the SHIPPED margins nothing
    lands in the band and the scorer is never consulted. Pinned as behaviour:
    the subject of such a turn lives in the conversation, not in the text being
    matched, and no second opinion can rescue an empty shortlist."""
    from claw.config import SmartRetrievalConfig
    ws = tmp_path / "ws"
    _write_store(ws)
    idx = await _index(ws, MemoryRetrievalConfig(
        smart_retrieval=SmartRetrievalConfig(enabled=True)))
    idx.scorer = _FakeScorer(noul=0.99)
    assert await idx.retrieve_markdown("looks good", top_n=5) == ""
    assert idx.scorer.calls == []


async def test_explicit_search_never_takes_the_smart_path(tmp_path):
    """`memory_search` is someone asking to see the ranking."""
    ws = tmp_path / "ws"
    _write_store(ws)
    idx = await _index(ws, _smart_cfg())
    idx.scorer = _FakeScorer(noul=0.99)
    await idx.retrieve_markdown("user-1: espresso", top_n=5, gate=False)
    assert idx.scorer.calls == []


# --- annotation-only chunks ----------------------------------------------------

def test_substantive_separates_annotation_from_content():
    """A heading and an HTML comment are both annotation. A chunk of nothing
    else says nothing; a chunk that HAS content keeps its annotation."""
    sub = MemoryIndex._substantive
    assert not sub("# 2026-04-10 (Friday)")
    assert not sub("# Agent Memory")
    assert not sub("<!-- placeholder for tomorrow's memory -->")
    assert not sub("### Heading\n<!-- a note to self -->\n")
    # Content present: still substantive, annotation and all.
    assert sub("- the hose was replaced")
    assert sub("# Title\n- the hose was replaced")
    # The case that must not regress: a supersession records WHY a fact
    # changed, in a comment, alongside the fact.
    assert sub("<!-- superseded: was 192.168.1.0/24, changed by the fibre "
               "cutover -->\n- **Network topology:** 10.10.11.0/24")


async def test_an_annotation_only_chunk_is_never_a_candidate(tmp_path):
    ws = tmp_path / "ws"
    _write_store(ws)
    # A file whose pre-heading region is nothing but its own title, and whose
    # one real section shares the query's words.
    (ws / "memory" / "2026-06-02.md").write_text(
        "# 2026-06-02 (Tuesday)\n\n## Espresso\nuser-1 noted: the espresso "
        "grinder was serviced.\n")
    idx = await _index(ws)
    kw = idx._keyword_index()
    title = next(c for c in kw["corpus"] if c["id"].endswith("2026-06-02.md:0"))
    assert title["traversal_only"] is True
    # The heading became the chunk's NAME rather than its text.
    assert title["section"] == "2026-06-02 (Tuesday)"

    hits, _ = idx._candidates("user-1: espresso grinder", 20, expand=True)
    assert title["id"] not in [h["id"] for h in hits]
    # ...while the real section from the same file is reachable.
    assert any(h["id"].endswith("2026-06-02.md:1") for h in hits)


async def test_an_annotation_only_chunk_still_exists_for_traversal(tmp_path):
    """It is a real node. The walk may pass through it; it is only barred
    from being a result."""
    ws = tmp_path / "ws"
    _write_store(ws)
    (ws / "memory" / "2026-06-03.md").write_text("# 2026-06-03\n\n## Grinder\nnotes.\n")
    idx = await _index(ws)
    kw = idx._keyword_index()
    ids = [c["id"] for c in kw["corpus"]]
    assert "memory/2026-06-03.md:0" in ids, "dropped from the corpus, not just from results"
    assert tuple(tokenize("2026-06-03")) in kw["by_section"]


async def test_a_leading_heading_becomes_the_section_name(tmp_path):
    """`_parse_markdown` splits on `##` only, so a `#` or `###` heading opening
    a file used to sit in the body while the chunk was called "Intro" — the
    agent was shown a meaningless label and the heading was repeated into the
    snippet. Lifting it applies the rule the `##` split already follows."""
    ws = tmp_path / "ws"
    _write_store(ws)
    (ws / "memory" / "2026-06-04.md").write_text(
        "### Roku play.py Workflow\n- blind navigation is the right approach.\n")
    idx = await _index(ws)
    row = next(c for c in idx._keyword_index()["corpus"]
               if c["id"].endswith("2026-06-04.md:0"))
    assert row["section"] == "Roku play.py Workflow"
    assert row["text"] == "- blind navigation is the right approach."
    assert row["traversal_only"] is False


async def test_a_region_with_no_heading_is_still_Intro(tmp_path):
    """The fence: content written with no heading at all must still be kept."""
    ws = tmp_path / "ws"
    _write_store(ws)
    (ws / "memory" / "2026-06-05.md").write_text(
        "[07:49] the subagent flow now includes a task name.\n")
    idx = await _index(ws)
    row = next(c for c in idx._keyword_index()["corpus"]
               if c["id"].endswith("2026-06-05.md:0"))
    assert row["section"] == "Intro"
    assert row["traversal_only"] is False
    assert "subagent flow" in row["text"]


async def test_the_scorer_is_asked_with_retrievals_own_deadline(tmp_path):
    """Retrieval passes `smart_retrieval.timeout_s`, not the client's.

    One SystemOneClient is shared with the decision gate, whose bound is
    deliberately short so it can fail open in front of the LLM. Retrieval
    batches every borderline candidate into ONE request and is worth waiting
    longer for, so it overrides per call. Measured: a 2s bound truncated that
    request's tail and cost ~5 points on every retrieval metric.
    """
    ws = tmp_path / "ws"
    _write_store(ws)
    idx = await _index(ws, _smart_cfg(timeout_s=9.5))
    idx.scorer = _FakeScorer()
    await idx.retrieve_markdown(QUERY, top_n=5)
    assert idx.scorer.timeouts and set(idx.scorer.timeouts) == {9.5}


async def test_a_scorer_timeout_costs_precision_not_memory(tmp_path):
    """Blowing the deadline falls back to ordinary retrieval, which still
    returns memories — an outage must never leave the turn with nothing."""
    import httpx
    ws = tmp_path / "ws"
    _write_store(ws)
    idx = await _index(ws, _smart_cfg(timeout_s=0.01))
    idx.scorer = _FakeScorer(error=httpx.ReadTimeout("too slow"))
    assert (await idx.retrieve_markdown(QUERY, top_n=5)).strip()
