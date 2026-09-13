"""music: entities, inheritance, and the one rule — curation is preserved.

Every test here exists because a plausible future edit could break it. The
protections are structural (an UPDATE that never names a column; a NULL that
means "nobody said"), which is exactly the kind of thing someone tidies away.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from claw import music_db


def _row(path="A/B/01 - T.mp3", artist="A", album="B", **over):
    d = dict(
        path=path, artist=artist, album=album, title="T", track_no=1, disc_no=None,
        genre="Rock", year=1971, duration_s=200.0, lufs=-14.0,
        true_peak_dbfs=-0.5, lra_lu=6.0, size=1000, mtime=1,
        measured_at="2026-09-09T00:00:00+00:00", measure_version=1,
    )
    d.update(over)
    return d


@pytest.fixture
def con(tmp_path):
    c = music_db.connect(tmp_path / "library.db")
    yield c
    c.close()


def _resolved(con, path="A/B/01 - T.mp3"):
    return con.execute("SELECT * FROM catalog WHERE path = ?", (path,)).fetchone()


def _artist(con, name="A"):
    return music_db.find_artist(con, name)


def _album(con, artist="A", album="B"):
    return music_db.find_album(con, artist, album)


# --- the boundary --------------------------------------------------------

def test_nothing_measurable_is_curatable():
    # The line is "measurably verifiable from the local file". A measurement is
    # a fact about the bytes; everything else is an opinion.
    every = set(music_db.TRACK_CURATABLE) | set(music_db.ALBUM_CURATABLE) | set(
        music_db.ARTIST_CURATABLE)
    assert not every & set(music_db.MEASURED)
    for field in ("lufs", "duration_s", "true_peak_dbfs", "size"):
        assert field not in every


def test_the_conflict_clause_names_only_measured_columns():
    # The protection IS the omission, so assert on the statement itself.
    for field in ("title", "album_id") + music_db.CURATED_ONLY:
        assert f"{field} = excluded.{field}" not in music_db._UPSERT
    for field in music_db.MEASURED:
        assert f"{field} = excluded.{field}" in music_db._UPSERT


# --- ingest --------------------------------------------------------------

def test_ingest_creates_the_artist_and_album_it_needs(con):
    music_db.upsert(con, _row())
    assert [r["name"] for r in con.execute("SELECT name FROM artists")] == ["A"]
    assert [r["name"] for r in con.execute("SELECT name FROM albums")] == ["B"]
    assert _resolved(con)["artist"] == "A"


def test_ingest_refreshes_what_the_bytes_determine(con):
    music_db.upsert(con, _row())
    music_db.upsert(con, _row(lufs=-6.0, size=2000, mtime=99))
    got = _resolved(con)
    assert (got["lufs"], got["size"], got["mtime"]) == (-6.0, 2000, 99)


def test_ingest_cannot_overwrite_curation(con):
    music_db.upsert(con, _row())
    music_db.curate(con, "album", _album(con), {"genre": "Dub"}, by="z")
    music_db.curate(con, "track", "A/B/01 - T.mp3", {"mood": ["mellow"], "energy": 2}, by="z")
    music_db.upsert(con, _row(genre="Rock", lufs=-9.0))     # the file still says Rock
    got = _resolved(con)
    assert got["genre"] == "Dub"
    assert json.loads(got["r_mood"]) == ["mellow"]
    assert got["r_energy"] == 2
    assert got["lufs"] == -9.0                              # measured still refreshed


def test_ingest_never_writes_a_track_level_genre(con):
    # It cannot tell a seeded album genre from a corrected one, so writing the
    # file's value would shadow the correction. Genre is seeded onto the album;
    # a genuine per-track genre is curation and says so.
    music_db.upsert(con, _row())
    music_db.upsert(con, _row("A/B/02 - U.mp3", genre="Dub"))
    assert {r["genre"] for r in con.execute("SELECT genre FROM tracks")} == {None}
    assert con.execute("SELECT genre FROM albums").fetchone()["genre"] == "Rock"


def test_repopulate_is_the_reset_and_discards_curation(con):
    music_db.upsert(con, _row())
    music_db.curate(con, "track", "A/B/01 - T.mp3", {"mood": ["mellow"], "energy": 4}, by="z")
    music_db.repopulate(con, _row())
    got = _resolved(con)
    assert got["r_mood"] is None and got["r_energy"] is None


def test_forget_keeps_the_album_because_it_may_hold_curation(con):
    music_db.upsert(con, _row())
    music_db.curate(con, "album", _album(con), {"notes": "keep me"}, by="z")
    music_db.forget(con, ["A/B/01 - T.mp3"])
    assert con.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 0
    assert con.execute("SELECT notes FROM albums").fetchone()["notes"] == "keep me"


def test_known_paths_is_what_ingest_checks_without_opening_a_file(con):
    music_db.upsert(con, _row(size=1234, mtime=567))
    assert music_db.known_paths(con) == {"A/B/01 - T.mp3": (1234, 567, 1)}


# --- inheritance ---------------------------------------------------------

def test_an_artist_note_reaches_every_track_from_one_row(con):
    music_db.upsert(con, _row("A/B/01.mp3"))
    music_db.upsert(con, _row("A/B/02.mp3"))
    music_db.upsert(con, _row("A/C/01.mp3", album="C"))
    music_db.curate(con, "artist", _artist(con), {"notes": "one era only"}, by="z")
    assert {r["r_notes"] for r in con.execute("SELECT r_notes FROM catalog")} == {"one era only"}


def test_a_track_ingested_later_inherits_what_the_album_already_had(con):
    # The defect that motivated entities: under a flat table a new file arrived
    # with NULL curation into an album annotated last year.
    music_db.upsert(con, _row("A/B/01.mp3"))
    music_db.curate(con, "album", _album(con), {"mood": ["mellow"], "genre": "Dub"}, by="z")
    music_db.upsert(con, _row("A/B/02.mp3"))
    got = _resolved(con, "A/B/02.mp3")
    assert json.loads(got["r_mood"]) == ["mellow"]
    assert got["genre"] == "Dub"          # a SEEDED field inherits too, now


def test_precedence_is_track_then_album_then_artist(con):
    music_db.upsert(con, _row())
    music_db.curate(con, "artist", _artist(con), {"energy": 1}, by="z")
    assert _resolved(con)["r_energy"] == 1
    music_db.curate(con, "album", _album(con), {"energy": 3}, by="z")
    assert _resolved(con)["r_energy"] == 3
    music_db.curate(con, "track", "A/B/01 - T.mp3", {"energy": 5}, by="z")
    assert _resolved(con)["r_energy"] == 5


def test_the_view_says_which_level_a_value_came_from(con):
    music_db.upsert(con, _row())
    music_db.curate(con, "artist", _artist(con), {"mood": ["dark"]}, by="z")
    assert _resolved(con)["mood_level"] == "artist"
    music_db.curate(con, "album", _album(con), {"mood": ["warm"]}, by="z")
    assert _resolved(con)["mood_level"] == "album"


def test_notes_add_up_lowest_level_first(con):
    # Unlike mood or energy, a note never hides the one above it: an artist-level
    # instruction must survive an album tidbit written beneath it.
    music_db.upsert(con, _row())
    assert _resolved(con)["r_notes"] is None
    music_db.curate(con, "artist", _artist(con), {"notes": "one era only"}, by="z")
    assert _resolved(con)["r_notes"] == "one era only"
    music_db.curate(con, "track", "A/B/01 - T.mp3", {"notes": "live take"}, by="z")
    assert _resolved(con)["r_notes"] == "live take\none era only"
    music_db.curate(con, "album", _album(con), {"notes": "the remaster"}, by="z")
    got = _resolved(con)
    assert got["r_notes"] == "live take\nthe remaster\none era only"
    assert (got["track_notes"], got["album_notes"], got["artist_notes"]) == (
        "live take", "the remaster", "one era only")


def test_mood_still_takes_the_most_specific(con):
    music_db.upsert(con, _row())
    music_db.curate(con, "album", _album(con), {"mood": ["warm"]}, by="z")
    music_db.curate(con, "track", "A/B/01 - T.mp3", {"mood": ["cold"]}, by="z")
    assert json.loads(_resolved(con)["r_mood"]) == ["cold"]


def test_never_shuffle_level_tells_nobody_said_from_somebody_said_no(con):
    # r_never_shuffle reads 0 in both cases; only the level can tell them apart.
    music_db.upsert(con, _row())
    got = _resolved(con)
    assert (got["r_never_shuffle"], got["never_shuffle_level"]) == (0, None)
    music_db.curate(con, "album", _album(con), {"never_shuffle": True}, by="z")
    assert _resolved(con)["never_shuffle_level"] == "album"
    music_db.curate(con, "track", "A/B/01 - T.mp3", {"never_shuffle": False}, by="z")
    got = _resolved(con)
    assert (got["r_never_shuffle"], got["never_shuffle_level"]) == (0, "track")


def test_never_shuffle_is_nullable_so_silence_does_not_beat_the_album(con):
    # A NOT NULL DEFAULT 0 would win every inheritance it was meant to defer
    # to — which is exactly what v1 had.
    music_db.upsert(con, _row())
    assert _resolved(con)["r_never_shuffle"] == 0
    music_db.curate(con, "album", _album(con), {"never_shuffle": True}, by="z")
    assert _resolved(con)["r_never_shuffle"] == 1


def test_never_shuffle_is_not_offered_at_artist_level(con):
    # "Never shuffle this artist" is not a thing anyone means; it is a property
    # of how a record was sequenced.
    assert "never_shuffle" not in music_db.ARTIST_CURATABLE
    with pytest.raises(music_db.NotCuratable, match="artist level"):
        music_db.curate(con, "artist", _artist(con) or 1, {"never_shuffle": True}, by="z")


def test_clearing_curation_lets_the_level_above_show_through(con):
    music_db.upsert(con, _row())
    music_db.curate(con, "artist", _artist(con), {"energy": 1}, by="z")
    music_db.curate(con, "album", _album(con), {"energy": 4}, by="z")
    music_db.clear_curation(con, "album", _album(con), by="z")
    assert _resolved(con)["r_energy"] == 1


def test_clearing_a_tracks_curation_does_not_destroy_what_the_file_said(con):
    # A seeded column holds either the file's value or an override,
    # indistinguishably — so clearing opinions must not touch it. Restoring
    # those is `ingest --repopulate`, a different operation.
    music_db.upsert(con, _row())
    music_db.curate(con, "track", "A/B/01 - T.mp3", {"energy": 5, "genre": "Dub"}, by="z")
    music_db.clear_curation(con, "track", "A/B/01 - T.mp3", by="z")
    got = _resolved(con)
    assert got["r_energy"] is None and got["genre"] == "Dub"


def test_clearing_a_mood_stores_null_not_the_string_null(con):
    # json.dumps(None) == '"null"' — non-empty, so a cleared mood would read as
    # set and then decode to None on the way out.
    music_db.upsert(con, _row())
    music_db.curate(con, "track", "A/B/01 - T.mp3", {"mood": ["mellow"]}, by="z")
    music_db.clear_curation(con, "track", "A/B/01 - T.mp3", by="z")
    assert _resolved(con)["r_mood"] is None


# --- renames and moves ---------------------------------------------------

def test_renaming_an_artist_is_one_row_and_keeps_its_annotations(con):
    music_db.upsert(con, _row("A/B/01.mp3"))
    music_db.upsert(con, _row("A/C/01.mp3", album="C"))
    music_db.curate(con, "artist", _artist(con), {"notes": "keep me"}, by="z")
    music_db.curate(con, "artist", _artist(con), {"name": "A and the Bs"}, by="z")

    assert con.execute("SELECT COUNT(*) FROM artists").fetchone()[0] == 1
    for path in ("A/B/01.mp3", "A/C/01.mp3"):
        got = _resolved(con, path)
        assert got["artist"] == "A and the Bs"
        assert got["r_notes"] == "keep me"        # nothing orphaned


def test_moving_one_album_to_another_artist_leaves_the_others_alone(con):
    music_db.upsert(con, _row("A/B/01.mp3"))
    music_db.upsert(con, _row("A/C/01.mp3", album="C"))
    music_db.curate(con, "album", _album(con, "A", "B"), {"artist": "Just The Bs"}, by="z")
    got = {(r["path"], r["artist"]) for r in con.execute("SELECT path, artist FROM catalog")}
    assert got == {("A/B/01.mp3", "Just The Bs"), ("A/C/01.mp3", "A")}


def test_a_track_artist_overrides_its_albums_which_is_how_compilations_work(con):
    music_db.upsert(con, _row("V/Comp/01.mp3", artist="Various", album="Comp"))
    music_db.upsert(con, _row("V/Comp/02.mp3", artist="Various", album="Comp"))
    music_db.curate(con, "track", "V/Comp/01.mp3", {"artist": "Someone Real"}, by="z")
    got = {(r["path"], r["artist"]) for r in con.execute("SELECT path, artist FROM catalog")}
    assert got == {("V/Comp/01.mp3", "Someone Real"), ("V/Comp/02.mp3", "Various")}


def test_a_track_inherits_the_mood_of_the_artist_actually_credited_on_it(con):
    music_db.upsert(con, _row("V/Comp/01.mp3", artist="Various", album="Comp"))
    music_db.curate(con, "track", "V/Comp/01.mp3", {"artist": "Someone Real"}, by="z")
    music_db.curate(con, "artist", _artist(con, "Someone Real"), {"mood": ["warm"]}, by="z")
    assert json.loads(_resolved(con, "V/Comp/01.mp3")["r_mood"]) == ["warm"]


# --- the audit log -------------------------------------------------------

def test_every_edit_is_logged_with_who_and_what_it_displaced(con):
    music_db.upsert(con, _row())
    aid = _album(con)
    music_db.curate(con, "album", aid, {"genre": "Dub"}, by="agent-a")
    music_db.curate(con, "album", aid, {"genre": "Roots"}, by="someone")
    log = music_db.history(con, str(aid))
    assert [(e["old"], e["new"], e["by"], e["scope"]) for e in log] == [
        ("Rock", "Dub", "agent-a", "album"),
        ("Dub", "Roots", "someone", "album"),
    ]


def test_setting_a_field_to_what_it_already_is_logs_nothing(con):
    music_db.upsert(con, _row())
    assert music_db.curate(con, "album", _album(con), {"genre": "Rock"}, by="z") == {}
    assert list(con.execute("SELECT * FROM edits")) == []


def test_curating_an_unknown_entity_is_an_error_not_an_insert(con):
    with pytest.raises(KeyError):
        music_db.curate(con, "track", "nope.mp3", {"genre": "Dub"}, by="z")
    with pytest.raises(KeyError):
        music_db.curate(con, "artist", 999, {"notes": "x"}, by="z")
    assert con.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 0


def test_plays_record_which_output_and_say_they_are_queue_time(con):
    music_db.upsert(con, _row())
    music_db.record_plays(con, ["A/B/01 - T.mp3"], "room-a")
    row = music_db.recent_plays(con)[0]
    assert (row["path"], row["output"], row["source"]) == ("A/B/01 - T.mp3", "room-a", "queued")


# --- migrations ----------------------------------------------------------

_V2 = """
CREATE TABLE schema_version (version INTEGER NOT NULL);
CREATE TABLE tracks (
    path TEXT PRIMARY KEY, artist TEXT NOT NULL DEFAULT '',
    album TEXT NOT NULL DEFAULT '', title TEXT NOT NULL DEFAULT '',
    track_no INTEGER, disc_no INTEGER, genre TEXT, year INTEGER,
    duration_s REAL, lufs REAL, true_peak_dbfs REAL, lra_lu REAL,
    size INTEGER NOT NULL, mtime INTEGER NOT NULL, measured_at TEXT,
    measure_version INTEGER NOT NULL DEFAULT 1,
    mood TEXT, energy INTEGER, notes TEXT, never_shuffle INTEGER,
    first_seen TEXT NOT NULL);
CREATE TABLE album_curation (artist TEXT NOT NULL, album TEXT NOT NULL,
    mood TEXT, energy INTEGER, notes TEXT, never_shuffle INTEGER,
    PRIMARY KEY (artist, album));
CREATE TABLE artist_curation (artist TEXT PRIMARY KEY,
    mood TEXT, energy INTEGER, notes TEXT);
CREATE TABLE edits (id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL,
    scope TEXT NOT NULL, subject TEXT NOT NULL, field TEXT NOT NULL,
    old TEXT, new TEXT, by TEXT NOT NULL);
CREATE TABLE plays (id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL,
    path TEXT NOT NULL, output TEXT, source TEXT NOT NULL DEFAULT 'queued');
INSERT INTO schema_version(version) VALUES (2);
"""


def _seed_v2(p):
    c = sqlite3.connect(p)
    c.executescript(_V2)
    for path, album, title in [("X/One/01.mp3", "One", "a"), ("X/One/02.mp3", "One", "b"),
                               ("X/Two/01.mp3", "Two", "c")]:
        c.execute("INSERT INTO tracks(path,artist,album,title,genre,year,size,mtime,"
                  "first_seen,lufs) VALUES(?,'X',?,?,'Rock',1971,1,1,'now',-14.0)",
                  (path, album, title))
    c.execute("INSERT INTO artist_curation(artist,notes) VALUES('X','an era only')")
    c.execute("INSERT INTO album_curation(artist,album,energy,never_shuffle) "
              "VALUES('X','One',4,1)")
    c.execute("UPDATE tracks SET mood='[\"menacing\"]', energy=2 WHERE path='X/One/02.mp3'")
    c.commit(); c.close()


def test_v2_upgrades_into_entities_and_keeps_every_opinion(tmp_path):
    p = tmp_path / "library.db"
    _seed_v2(p)
    con = music_db.connect(p)
    try:
        assert con.execute("SELECT version FROM schema_version").fetchone()[0] == music_db.SCHEMA_VERSION
        assert con.execute("SELECT COUNT(*) FROM artists").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM albums").fetchone()[0] == 2

        # artist-level note reaches all three tracks
        assert {r["r_notes"] for r in con.execute("SELECT r_notes FROM catalog")} == {"an era only"}
        # album-level values land on the right album only
        one = _resolved(con, "X/One/01.mp3")
        two = _resolved(con, "X/Two/01.mp3")
        assert (one["r_energy"], one["r_never_shuffle"]) == (4, 1)
        assert (two["r_energy"], two["r_never_shuffle"]) == (None, 0)
        # the track-level override still wins
        over = _resolved(con, "X/One/02.mp3")
        assert over["r_energy"] == 2 and json.loads(over["r_mood"]) == ["menacing"]
        # seeded values moved up to the album and are no longer per-track
        assert (one["genre"], one["year"]) == ("Rock", 1971)
        assert {r["genre"] for r in con.execute("SELECT genre FROM tracks")} == {None}
        # the side tables are gone
        assert not con.execute(
            "SELECT name FROM sqlite_master WHERE name IN "
            "('album_curation','artist_curation')").fetchall()
    finally:
        con.close()


def test_after_upgrading_a_rename_is_one_row(tmp_path):
    # The whole point of the migration: what used to need a fan-out.
    p = tmp_path / "library.db"
    _seed_v2(p)
    con = music_db.connect(p)
    try:
        music_db.curate(con, "artist", _artist(con, "X"), {"name": "X and the Ys"}, by="z")
        assert {r["artist"] for r in con.execute("SELECT artist FROM catalog")} == {"X and the Ys"}
        assert {r["r_notes"] for r in con.execute("SELECT r_notes FROM catalog")} == {"an era only"}
    finally:
        con.close()


def test_migrating_twice_is_harmless(tmp_path):
    p = tmp_path / "library.db"
    _seed_v2(p)
    music_db.connect(p).close()
    con = music_db.connect(p)
    try:
        assert con.execute("SELECT version FROM schema_version").fetchone()[0] == music_db.SCHEMA_VERSION
        assert con.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 3
    finally:
        con.close()


def test_v3_upgrades_and_keeps_its_data(tmp_path):
    # Build a v3 catalogue: current tables, the v3 view, version 3.
    p = tmp_path / "library.db"
    con = music_db.connect(p)
    music_db.upsert(con, _row())
    music_db.curate(con, "album", _album(con), {"notes": "kept", "never_shuffle": True}, by="z")
    con.executescript("""
        DROP VIEW catalog;
        CREATE VIEW catalog AS SELECT t.path, t.notes FROM tracks t;
        UPDATE schema_version SET version = 3;
    """)
    con.commit(); con.close()

    con = music_db.connect(p)
    try:
        assert con.execute("SELECT version FROM schema_version").fetchone()[0] == music_db.SCHEMA_VERSION
        got = _resolved(con)
        assert (got["r_notes"], got["album_notes"]) == ("kept", "kept")
        assert (got["r_never_shuffle"], got["never_shuffle_level"]) == (1, "album")
    finally:
        con.close()


def test_a_newer_schema_is_refused_rather_than_guessed_at(tmp_path):
    p = tmp_path / "library.db"
    music_db.connect(p).close()
    c = sqlite3.connect(p)
    c.execute("UPDATE schema_version SET version = ?", (music_db.SCHEMA_VERSION + 1,))
    c.commit(); c.close()
    with pytest.raises(RuntimeError, match="newer version wrote it"):
        music_db.connect(p)


# --- merging duplicate artists -------------------------------------------

def test_renaming_onto_an_existing_artist_merges_rather_than_failing(con):
    # The usual reason to rename is that the same act is filed twice, so the
    # target normally already exists. A UNIQUE violation would be a useless
    # answer to what the operator plainly meant.
    music_db.upsert(con, _row("A/B/01.mp3", artist="A", album="B"))
    music_db.upsert(con, _row("A2/C/01.mp3", artist="A and the Bs", album="C"))
    music_db.curate(con, "artist", _artist(con, "A"), {"name": "A and the Bs"}, by="z")

    assert [r["name"] for r in con.execute("SELECT name FROM artists")] == ["A and the Bs"]
    assert {r["artist"] for r in con.execute("SELECT artist FROM catalog")} == {"A and the Bs"}
    assert {r["album"] for r in con.execute("SELECT album FROM catalog")} == {"B", "C"}


def test_a_merge_keeps_the_survivors_opinions(con):
    music_db.upsert(con, _row("A/B/01.mp3", artist="A"))
    music_db.upsert(con, _row("A2/C/01.mp3", artist="A and the Bs", album="C"))
    music_db.curate(con, "artist", _artist(con, "A"), {"notes": "from the old row"}, by="z")
    music_db.curate(con, "artist", _artist(con, "A and the Bs"), {"notes": "keep mine"}, by="z")
    music_db.curate(con, "artist", _artist(con, "A"), {"name": "A and the Bs"}, by="z")
    assert con.execute("SELECT notes FROM artists").fetchone()["notes"] == "keep mine"


def test_a_merge_carries_across_an_annotation_the_survivor_lacks(con):
    # Losing an annotation is the one outcome nobody wants from tidying up.
    music_db.upsert(con, _row("A/B/01.mp3", artist="A"))
    music_db.upsert(con, _row("A2/C/01.mp3", artist="A and the Bs", album="C"))
    music_db.curate(con, "artist", _artist(con, "A"), {"notes": "worth keeping"}, by="z")
    music_db.curate(con, "artist", _artist(con, "A"), {"name": "A and the Bs"}, by="z")
    assert con.execute("SELECT notes FROM artists").fetchone()["notes"] == "worth keeping"


def test_a_merge_that_would_collide_on_an_album_refuses_and_says_why(con):
    music_db.upsert(con, _row("A/B/01.mp3", artist="A", album="Same"))
    music_db.upsert(con, _row("A2/B/01.mp3", artist="A and the Bs", album="Same"))
    with pytest.raises(music_db.NotCuratable, match="both artists have an album"):
        music_db.curate(con, "artist", _artist(con, "A"), {"name": "A and the Bs"}, by="z")
    assert con.execute("SELECT COUNT(*) FROM artists").fetchone()[0] == 2


def test_a_merge_is_recorded_in_the_audit_log(con):
    music_db.upsert(con, _row("A/B/01.mp3", artist="A"))
    music_db.upsert(con, _row("A2/C/01.mp3", artist="A and the Bs", album="C"))
    target = _artist(con, "A and the Bs")
    music_db.curate(con, "artist", _artist(con, "A"), {"name": "A and the Bs"}, by="z")
    fields = [e["field"] for e in music_db.history(con, str(target))]
    assert "merged_from" in fields



# --- system events in the edit log --------------------------------------------------

def test_repopulate_logs_the_curation_it_discards(con):
    music_db.upsert(con, _row())
    music_db.curate(con, "track", "A/B/01 - T.mp3", {"energy": 5, "notes": "keep me"}, by="z")
    music_db.repopulate(con, _row(), by="op")
    ev = con.execute("SELECT * FROM edits WHERE field = 'repopulated'").fetchone()
    assert ev["by"] == "op" and json.loads(ev["old"]) == {"energy": 5, "notes": "keep me"}
    assert _resolved(con)["r_energy"] is None                   # and it really did discard it


def test_forgetting_an_uncurated_file_logs_nothing(con):
    music_db.upsert(con, _row())
    music_db.forget(con, ["A/B/01 - T.mp3"])
    assert con.execute("SELECT COUNT(*) FROM edits").fetchone()[0] == 0


def test_forgetting_a_curated_file_keeps_its_curation_in_the_log(con):
    music_db.upsert(con, _row())
    music_db.curate(con, "track", "A/B/01 - T.mp3", {"mood": ["dark"]}, by="z")
    music_db.forget(con, ["A/B/01 - T.mp3"], by="op")
    ev = con.execute("SELECT * FROM edits WHERE field = 'forgotten'").fetchone()
    assert json.loads(ev["old"]) == {"mood": '["dark"]'}


def test_edits_for_counts_the_true_total_past_the_limit(con):
    music_db.upsert(con, _row())
    for e in (1, 2, 3):
        music_db.curate(con, "track", "A/B/01 - T.mp3", {"energy": e}, by="z")
    rows, total = music_db.edits_for(con, [("track", "A/B/01 - T.mp3")], limit=2)
    assert total == 3 and [r["new"] for r in rows] == ["3", "2"]


def test_v4_notes_become_additive_without_moving_a_note(tmp_path):
    p = tmp_path / "library.db"
    con = music_db.connect(p)
    music_db.upsert(con, _row())
    music_db.curate(con, "artist", _artist(con), {"notes": "instruction"}, by="z")
    music_db.curate(con, "album", _album(con), {"notes": "tidbit"}, by="z")
    # the v4 view: most specific note wins
    con.executescript("""
        DROP VIEW catalog;
        CREATE VIEW catalog AS SELECT t.path, COALESCE(t.notes, al.notes) AS r_notes
        FROM tracks t JOIN albums al ON al.id = t.album_id;
        UPDATE schema_version SET version = 4;
    """)
    con.commit(); con.close()
    con = music_db.connect(p)
    try:
        assert _resolved(con)["r_notes"] == "tidbit\ninstruction"
        assert con.execute("SELECT notes FROM albums").fetchone()[0] == "tidbit"
    finally:
        con.close()
