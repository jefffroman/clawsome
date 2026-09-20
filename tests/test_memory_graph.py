"""The graph leg: how sections link to what they name, and how a query picks
its entry node. Builds are pure (no ChromaDB); the query tests need a real
index for its term statistics, so they run under the gateway venv.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from claw.config import MemoryRetrievalConfig
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
    assert idx._query_graph("toner")["nodes"] > 0, "store did not index"
    return idx


@pytest.mark.asyncio
async def test_a_query_of_only_generic_words_matches_nothing(tmp_path):
    idx = await _index(tmp_path)
    for filler in ("thanks", "ok then", "what did you do"):
        assert idx._query_graph(filler)["related"] == [], filler


@pytest.mark.asyncio
async def test_the_informative_word_wins_over_the_vague_one(tmp_path):
    """Both "hose" and "working" are uncommon enough to survive the common-word
    cut, so counting matched words ties them and name length breaks the tie the
    wrong way. Weighting by IDF is what puts the asked-about thing first."""
    idx = await _index(tmp_path)
    top = [r["node"] for r in idx._query_graph("is the garden hose working", top_n=3)["related"]]
    # Either the concept or the section it belongs to is a right answer; what
    # matters is that neither "Working ..." section takes the lead.
    assert top[0] in ("garden hose", "Garden hose replacement"), top
    assert not top[0].lower().startswith("working"), top


@pytest.mark.asyncio
async def test_an_entry_node_comes_back_with_what_it_links_to(tmp_path):
    """The leg exists for traversal, so a hit is only useful with neighbors."""
    idx = await _index(tmp_path)
    [hit] = [r for r in idx._query_graph("toner")["related"]
             if r["node"] == "Printer toner"]
    assert "toner cartridges" in hit["neighbors"]


@pytest.mark.asyncio
async def test_neighbors_are_not_repeated(tmp_path):
    idx = await _index(tmp_path)
    for hit in idx._query_graph("garden hose leaked")["related"]:
        assert len(hit["neighbors"]) == len(set(hit["neighbors"]))


@pytest.mark.asyncio
async def test_a_missing_graph_file_is_not_an_error(tmp_path):
    idx = MemoryIndex("example", tmp_path, MemoryRetrievalConfig())
    assert idx._query_graph("anything") == {"nodes": 0, "related": []}


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
