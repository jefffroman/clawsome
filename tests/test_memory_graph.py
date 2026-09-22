"""The graph leg: how sections link to what they name, and how a query picks
its entry node. Builds are pure (no ChromaDB); the query tests need a real
index for its term statistics, so they run under the gateway venv.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

import dataclasses

from pathlib import Path

import pytest

from claw.config import GraphConfig, MemoryRetrievalConfig
from claw.memory import MemoryIndex

pytestmark = pytest.mark.filterwarnings("ignore")


def _chunk(section: str, content: str) -> dict:
    return {"content": content, "metadata": {"section": section, "source": "memory/n.md"}}


# --- building ------------------------------------------------------------------

def test_mentions_match_whole_tokens_not_substrings():
    """A name is a mention only when its tokens occur as tokens. The substring
    scan this replaced linked "ok" to any section containing "cookbook"."""
    chunks = [
        _chunk("Recipe box", "The **cook** was out; notes went in the box."),
        _chunk("Kitchen shelf", "The cookbook is on the shelf."),
    ]
    G = MemoryIndex._build_graph(chunks)
    assert G.has_edge("Recipe box", "cook")             # bolded in its own section
    assert not G.has_edge("Kitchen shelf", "cook")      # only inside "cookbook"


def test_a_name_is_matched_as_a_whole_phrase():
    chunks = [
        _chunk("Tool notes", "Check the **garden hose** before winter."),
        _chunk("Shed inventory", "One garden hose, one rake."),
        _chunk("Unrelated", "The garden is dry and the hose is elsewhere."),
    ]
    G = MemoryIndex._build_graph(chunks)
    assert G.has_edge("Shed inventory", "garden hose")
    # "garden" and "hose" both appear but never adjacently.
    assert not G.has_edge("Unrelated", "garden hose")


def test_contains_survives_the_mentions_pass():
    """A bolded concept trivially appears in its own chunk text, so the
    mentions pass used to overwrite every `contains` edge with `mentions`."""
    chunks = [_chunk("Printer notes", "Order more **toner cartridges** soon.")]
    G = MemoryIndex._build_graph(chunks)
    assert G["Printer notes"]["toner cartridges"]["relation"] == "contains"


def test_common_names_are_not_mention_targets():
    """A name made only of the store's common words matches everywhere and
    distinguishes nothing, so it links to nothing it did not bold itself."""
    chunks = [
        _chunk("Intro", "This **note** explains the layout."),
        _chunk("Second", "This note covers the rest."),
    ]
    linked = MemoryIndex._build_graph(chunks, common={"note"})
    assert not linked.has_edge("Second", "note")
    # ...and it is still reachable as something Intro bolded.
    assert linked["Intro"]["note"]["relation"] == "contains"
    # Without the common set it is an ordinary target.
    assert MemoryIndex._build_graph(chunks).has_edge("Second", "note")


def test_a_section_never_links_to_itself():
    chunks = [_chunk("Backup schedule", "The backup schedule runs nightly.")]
    G = MemoryIndex._build_graph(chunks)
    assert not G.has_edge("Backup schedule", "Backup schedule")


# --- querying ------------------------------------------------------------------

NOTES = [
    ("Garden hose replacement", "The **garden hose** was replaced; the old one leaked."),
    ("Working directory layout", "The **working directory** holds build output."),
    ("Working hours", "Quiet **working hours** are before noon."),
    ("Dentist appointment", "Moved to the first week of March; **reminder** set."),
    ("Printer toner", "The second-floor printer needs **toner cartridges**."),
    ("Grocery order", "Weekly order moved from Tuesday to Thursday."),
    ("Router firmware", "Upgraded to version 7.2.1 after the **outage**."),
    ("Book club", "Picked a science-fiction novel for next month."),
    ("Kitchen faucet", "The faucet drips; plumber booked for Friday."),
    ("Nightly backups", "Backups now run at two in the morning."),
    ("Vaccination due", "The dog's vaccination is due in October."),
    ("Solar output", "Output dropped after the storm; inspection requested."),
    ("Tax documents", "Filed in the blue folder."),
    ("Bike tires", "Pressure should be kept at sixty psi."),
    ("Thermostat schedule", "Lowers heat overnight."),
    ("Passport renewal", "Application was mailed on Monday."),
    ("Laptop fan", "Noisy under load; cleaning planned."),
    ("Lentil soup", "Recipe saved to the cookbook file."),
    ("Gym membership", "Renews automatically in January."),
    ("Smoke detector", "Battery chirps; replacement bought."),
    ("Coffee grinder", "Setting nine works best for the espresso machine."),
    ("Water heater", "Temperature set to 120 degrees."),
]


async def _index(ws: Path) -> MemoryIndex:
    """Writes the store AND indexes it. The two are one step on purpose: an
    empty store makes every query return nothing, which silently *passes* a
    test asserting that a query matches nothing."""
    (ws / "memory").mkdir(parents=True, exist_ok=True)
    body = "".join(f"## {h}\n<!-- mem ts=2026-06-{i % 28 + 1:02d} -->\n{b}\n\n"
                   for i, (h, b) in enumerate(NOTES))
    (ws / "memory" / "2026-06-01.md").write_text(body)
    idx = MemoryIndex("example", ws, MemoryRetrievalConfig())
    await idx.warmup_async()
    await idx.reindex_if_stale(force=True)
    assert idx._keyword_index()["corpus"], "store did not index"
    return idx







# --- the per-turn block --------------------------------------------------------

@pytest.mark.asyncio
async def test_a_turn_that_retrieves_nothing_injects_nothing(tmp_path):
    """The row is skipped on "", so a gated turn with no hits costs nothing."""
    idx = await _index(tmp_path)
    assert await idx.retrieve_markdown("thanks") == ""


@pytest.mark.asyncio
async def test_an_explicit_search_never_goes_silent(tmp_path):
    """Someone asked, so silence would be the wrong answer — the ungated path
    ranks and reports whatever is closest, even for a query like this."""
    idx = await _index(tmp_path)
    assert await idx.retrieve_markdown("thanks", gate=False) != ""


@pytest.mark.asyncio
async def test_a_turn_that_retrieves_something_still_injects(tmp_path):
    idx = await _index(tmp_path)
    assert "toner" in (await idx.retrieve_markdown("printer toner cartridges")).lower()


def test_a_span_written_only_as_a_field_label_is_not_a_concept():
    """"**Status**: ..." is list formatting, not a claim about the world."""
    chunks = [
        _chunk("Monday", "* **Status**: green.\nThe **garden hose** was replaced."),
        _chunk("Tuesday", "* **Status**: amber. The garden hose held."),
    ]
    G = MemoryIndex._build_graph(chunks)
    assert "Status" not in G
    # ...while a span that also appears plain somewhere stays a concept.
    assert G.nodes["garden hose"]["type"] == "concept"


def test_a_colon_inside_the_bold_reads_as_a_label_too():
    chunks = [_chunk("Monday", "**Lesson:** check the valve first.")]
    assert "Lesson:" not in MemoryIndex._build_graph(chunks)


def test_a_label_word_used_plainly_elsewhere_survives():
    chunks = [
        _chunk("Monday", "* **Event**: the fair."),
        _chunk("Tuesday", "The **Event** horizon was the topic."),
    ]
    assert "Event" in MemoryIndex._build_graph(chunks)


# --- node identity -------------------------------------------------------------

def test_spellings_of_one_name_are_one_node():
    """Keyed by token sequence, as the keyword leg indexes it — so case and a
    trailing colon do not split a concept into two nodes that match the same
    text, collect the same edges, and take two result slots to say one thing."""
    chunks = [
        _chunk("Monday", "The **Ollama** server restarted."),
        _chunk("Tuesday", "ollama came back on its own."),
        _chunk("Wednesday", "* **Routing**: fixed.\nThe **Routing** note is filed."),
        _chunk("Thursday", "Routing: still fine."),
    ]
    G = MemoryIndex._build_graph(chunks)
    assert [n for n in G if n.lower() == "ollama"] == ["Ollama"]
    assert [n for n in G if n.lower().rstrip(":") == "routing"] == ["Routing"]


def test_the_label_is_the_commonest_spelling():
    chunks = [
        _chunk("A", "The **ollama** runner."),
        _chunk("B", "The **ollama** runner again."),
        _chunk("C", "The **Ollama** runner once."),
    ]
    assert "ollama" in MemoryIndex._build_graph(chunks)


def test_a_tie_is_broken_the_same_way_every_build():
    chunks = [_chunk("A", "**Ollama** and **ollama** once each.")]
    first = MemoryIndex._build_graph(chunks)
    assert [n for n in first if n.lower() == "ollama"] == ["Ollama"]   # lexicographic
    assert list(MemoryIndex._build_graph(chunks)) == list(first)


def test_a_heading_outranks_a_bolded_span_of_the_same_name():
    """The heading is structural: a caller telling "a section I can go read"
    from "a phrase somebody bolded" wants the stronger claim."""
    chunks = [
        _chunk("Toner cartridges", "Ordered more."),
        _chunk("Monday", "Check the **toner cartridges** today."),
    ]
    G = MemoryIndex._build_graph(chunks)
    assert G.nodes["Toner cartridges"]["type"] == "section"


# --- graph expansion of the candidate pool --------------------------------------
#
# The graph does not match the query here. It expands outward from what the two
# ranked legs already found, which is what makes the anchor a chunk the
# calibrated path endorsed rather than a node name that shares a word.

LINKED = [
    # "Fermentation crock" links OUT to two sections whose own text shares no
    # vocabulary with it, which is the shape expansion exists for -- and two,
    # so the max_expand cap has something to cut.
    ("Fermentation crock",
     "The cabbage needs two weeks. See also Brine ratios and Lid gasket."),
    ("Brine ratios", "Three tablespoons per litre of water."),
    # Deliberately shares no word with the query that finds the crock note --
    # otherwise it ranks directly and expansion is not what reached it.
    ("Lid gasket", "Replaced in April; the old one had perished."),
    ("Bicycle service", "The rear derailleur was adjusted in March."),
]


async def _linked_index(ws: Path, cfg: MemoryRetrievalConfig | None = None) -> MemoryIndex:
    (ws / "memory").mkdir(parents=True, exist_ok=True)
    body = "".join(f"## {h}\n<!-- mem ts=2026-06-{i % 28 + 1:02d} -->\n{b}\n\n"
                   for i, (h, b) in enumerate(NOTES + LINKED))
    (ws / "memory" / "2026-06-01.md").write_text(body)
    idx = MemoryIndex("example", ws, cfg or MemoryRetrievalConfig())
    await idx.warmup_async()
    await idx.reindex_if_stale(force=True)
    return idx


def _by_section(hits):
    return {h["section"]: h for h in hits}


@pytest.mark.asyncio
async def test_expansion_reaches_a_section_only_a_link_leads_to(tmp_path):
    idx = await _linked_index(tmp_path)
    # Pool of 5, not 20: this store has fewer chunks than a full pool holds, so
    # a deep pool would already contain every neighbour and expansion would
    # have nothing left to contribute -- which is right, and untestable.
    q = "how long does the fermentation crock take"
    plain, _ = idx._candidates(q, 5)
    grown, _ = idx._candidates(q, 5, expand=True)
    assert "Brine ratios" not in _by_section(plain), "must not already be ranked in"
    reached = _by_section(grown).get("Brine ratios")
    assert reached is not None, "a linked section should be reachable"
    assert reached["_traversal"] == 1.0


@pytest.mark.asyncio
async def test_expansion_never_repeats_a_candidate(tmp_path):
    """The seeds are the BEST candidates, so their neighbours are very often
    further down the same pool. Returning one twice put a duplicate id into the
    embedding backfill, which Chroma rejects outright."""
    idx = await _linked_index(tmp_path)
    for q in ("fermentation crock brine", "bicycle derailleur", "coffee grinder espresso"):
        ids = [h["id"] for h in idx._candidates(q, 20, expand=True)[0]]
        assert len(ids) == len(set(ids)), q


@pytest.mark.asyncio
async def test_a_directly_ranked_candidate_carries_no_traversal_credit(tmp_path):
    idx = await _linked_index(tmp_path)
    hits, _ = idx._candidates("fermentation crock cabbage", 5, expand=True)
    assert _by_section(hits)["Fermentation crock"]["_traversal"] == 0.0


@pytest.mark.asyncio
async def test_expansion_is_off_for_the_ungated_search(tmp_path):
    """`memory_search` and the curator judge what the two legs ranked, not what
    the graph reached from it."""
    idx = await _linked_index(tmp_path)
    q = "how long does the fermentation crock take"
    assert all(h["_traversal"] == 0.0 for h in idx._hybrid_search(q, n=5))


@pytest.mark.asyncio
async def test_disabling_the_graph_removes_the_step(tmp_path):
    cfg = MemoryRetrievalConfig(graph=GraphConfig(enabled=False))
    idx = await _linked_index(tmp_path, cfg)
    q = "how long does the fermentation crock take"
    assert "Brine ratios" not in _by_section(idx._candidates(q, 5, expand=True)[0])


@pytest.mark.asyncio
async def test_expansion_is_capped(tmp_path):
    cfg = MemoryRetrievalConfig(graph=GraphConfig(max_expand=1))
    idx = await _linked_index(tmp_path, cfg)
    hits, _ = idx._candidates("fermentation crock brine ratios", 20, expand=True)
    assert sum(h["_traversal"] for h in hits) <= 1.0


@pytest.mark.asyncio
async def test_the_cap_keeps_neighbours_of_the_best_candidates(tmp_path):
    """The cap must bite by seed rank, not by id. Ids start with their source
    filename, so truncating a sorted list quietly prefers whichever notes are
    dated earliest."""
    idx = await _linked_index(tmp_path)
    # A SHALLOW pool, or there is nothing to expand into: on a store this size
    # a pool of 20 already holds every chunk, and expansion correctly adds
    # nothing.
    q = "how long does the fermentation crock take"
    wide, _ = idx._candidates(q, 3, expand=True)
    reached = [h["id"] for h in wide if h["_traversal"] == 1.0]
    assert len(reached) >= 2, f"fixture should reach at least two, got {reached}"

    idx.retrieval = dataclasses.replace(
        idx.retrieval, graph=GraphConfig(max_expand=1))
    capped, _ = idx._candidates(q, 3, expand=True)
    kept = [h["id"] for h in capped if h["_traversal"] == 1.0]
    assert kept == reached[:1], "the cap should keep the first reached, not the alphabetical first"
