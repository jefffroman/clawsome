"""The music catalogue: three entities, and one rule about who may write what.

The database is **not** derived data. The measured columns could be rebuilt from
the files in minutes, but curation — mood, energy, notes, corrections — exists
nowhere else, so this belongs wherever the deployment's backup already reaches,
and schema changes are real migrations rather than "drop it and re-ingest".

## Entities, because most facts are not per-track

A collection is artists containing albums containing tracks, and almost every
interesting fact attaches above the track: an album's genre, an artist's mood,
a note about a band. Modelling only tracks forces one of two bad answers —
duplicate the fact onto every row (and watch a track added next year miss it),
or have nowhere to put it at all.

So each level is a row, each carries the same curatable fields, and a value is
resolved **most specific first**: the track's if somebody set one, else the
album's, else the artist's. ``NULL`` means *nobody said*, which is the entire
mechanism — see the warning on :func:`migrate`.

Ids rather than names as keys. Renaming an artist is then one row, and no
annotation can be orphaned by it; an earlier draft keyed on names and needed a
fan-out across every track plus a re-keying pass, all of which this deletes.

## Curation is preserved, not overwritten

Enforced structurally rather than by a rule anyone has to remember. The
boundary is *measurably verifiable from the local file*:

- **Measured** (``lufs``, ``duration_s``, ``true_peak_dbfs``, ``lra_lu``,
  ``size``, ``mtime``) are facts about the bytes. Ingest always refreshes them.
  Not curatable: an opinion about loudness is not a thing, and if a measurement
  looks wrong then the file is wrong.
- **Seeded** (``title``, ``genre``, ``year``, and the artist and album names)
  are written when the row is first created and never named again. The upsert's
  ``ON CONFLICT`` clause simply omits them — an ``UPDATE`` that does not mention
  a column cannot clobber it, so there is no flag to get wrong.
- **Curated** (``mood``, ``energy``, ``notes``, ``never_shuffle``) are never
  written by ingest at all.

Putting a row back to what the file says is ``repopulate`` — a delete and a
re-insert, a different statement rather than a mode, so the unsafe write does
not exist in the codebase.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

log = logging.getLogger(__name__)

SCHEMA_VERSION = 5

# Facts about the bytes. Ingest refreshes them; nobody may override them.
MEASURED: tuple[str, ...] = (
    "duration_s", "lufs", "true_peak_dbfs", "lra_lu",
    "size", "mtime", "measured_at", "measure_version",
)
# Read from the file when a row is created, then owned by whoever curates it.
# A retag reaches an existing row only through `repopulate`.
# Read from the file when a row is created, then owned by whoever curates it.
# Note where each lives: an album owns genre and year, because in a
# directory tree those describe the record, not the file.
SEEDED: tuple[str, ...] = ("name", "title", "genre", "year")
# No file source at all.
CURATED_ONLY: tuple[str, ...] = ("mood", "energy", "notes", "never_shuffle")

# What may be set at each level. Uniform apart from what the level can mean:
# a track has a title where an album has a name, and "never shuffle this
# artist" is not a thing anyone means — it is a property of a record's
# sequencing.
TRACK_CURATABLE: tuple[str, ...] = ("title", "track_no", "disc_no", "artist",
                                    "genre", "year") + CURATED_ONLY
ALBUM_CURATABLE: tuple[str, ...] = ("name", "artist", "genre", "year",
                                    "mood", "energy", "notes", "never_shuffle")
ARTIST_CURATABLE: tuple[str, ...] = ("name", "mood", "energy", "notes")

SCOPES: tuple[str, ...] = ("track", "album", "artist")

_DDL = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS artists (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name   TEXT NOT NULL UNIQUE,
    mood   TEXT,
    energy INTEGER,
    notes  TEXT
);

CREATE TABLE IF NOT EXISTS albums (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    artist_id     INTEGER NOT NULL REFERENCES artists(id),
    name          TEXT NOT NULL,
    genre         TEXT,
    year          INTEGER,
    mood          TEXT,
    energy        INTEGER,
    notes         TEXT,
    never_shuffle INTEGER,
    UNIQUE (artist_id, name)
);

CREATE TABLE IF NOT EXISTS tracks (
    path            TEXT PRIMARY KEY,      -- relative to library_root
    album_id        INTEGER NOT NULL REFERENCES albums(id),
    title           TEXT NOT NULL DEFAULT '',
    track_no        INTEGER,
    disc_no         INTEGER,
    -- NULL means inherit from the album. A track artist is how a compilation
    -- says who actually played on it.
    artist_id       INTEGER REFERENCES artists(id),
    genre           TEXT,
    year            INTEGER,
    -- measured from the bytes
    duration_s      REAL,
    lufs            REAL,
    true_peak_dbfs  REAL,
    lra_lu          REAL,
    size            INTEGER NOT NULL,
    mtime           INTEGER NOT NULL,
    measured_at     TEXT,
    measure_version INTEGER NOT NULL DEFAULT 1,
    -- curated. All nullable, never_shuffle included: NULL is "nobody said".
    mood            TEXT,
    energy          INTEGER,
    notes           TEXT,
    never_shuffle   INTEGER,
    first_seen      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS tracks_album ON tracks(album_id);

-- Append-only. Replaces the per-row provenance columns an earlier draft
-- carried: this records who, when, and what the value displaced, and cannot
-- drift out of step with the data because it is never updated.
CREATE TABLE IF NOT EXISTS edits (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    at      TEXT NOT NULL,
    scope   TEXT NOT NULL,          -- track | album | artist
    subject TEXT NOT NULL,          -- track path, or the entity id
    field   TEXT NOT NULL,
    old     TEXT,
    new     TEXT,
    by      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS edits_subject ON edits(subject);

-- `at` is QUEUE time, not play time: the tool knows what it put in the
-- playlist, and observing what actually finished would need something watching
-- mpv's event stream. `source` says which, so a later real observer can write
-- 'played' rows alongside these without the two being confused.
CREATE TABLE IF NOT EXISTS plays (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    at     TEXT NOT NULL,
    path   TEXT NOT NULL,
    output TEXT,
    source TEXT NOT NULL DEFAULT 'queued'
);
CREATE INDEX IF NOT EXISTS plays_at ON plays(at);

-- Resolution, in one place. Everything that reads the catalogue reads this, so
-- no caller can implement the precedence differently. Most specific wins — for
-- everything EXCEPT notes, which ADD UP: each level carries its own facts and
-- all of them apply, lowest level first (track, album, artist), one per line.
-- Shadowing them buried an artist-level instruction under any album tidbit.
CREATE VIEW IF NOT EXISTS catalog AS
SELECT
    t.path, t.album_id, t.title, t.track_no, t.disc_no,
    t.duration_s, t.lufs, t.true_peak_dbfs, t.lra_lu,
    t.size, t.mtime, t.measured_at, t.measure_version, t.first_seen,
    COALESCE(ta.name, aa.name)               AS artist,
    al.name                                  AS album,
    COALESCE(t.genre,  al.genre)             AS genre,
    COALESCE(t.year,   al.year)              AS year,
    COALESCE(t.mood,   al.mood,   ar.mood)   AS r_mood,
    COALESCE(t.energy, al.energy, ar.energy) AS r_energy,
    NULLIF(COALESCE(t.notes, '')
        || CASE WHEN t.notes IS NOT NULL AND (al.notes IS NOT NULL OR ar.notes IS NOT NULL)
                THEN char(10) ELSE '' END
        || COALESCE(al.notes, '')
        || CASE WHEN al.notes IS NOT NULL AND ar.notes IS NOT NULL THEN char(10) ELSE '' END
        || COALESCE(ar.notes, ''), '')       AS r_notes,
    -- Each level's own note. There is no notes_level: with notes adding up,
    -- which levels contributed is just which of these are set.
    t.notes AS track_notes, al.notes AS album_notes, ar.notes AS artist_notes,
    COALESCE(t.never_shuffle, al.never_shuffle, 0) AS r_never_shuffle,
    CASE WHEN t.mood IS NOT NULL THEN 'track'
         WHEN al.mood IS NOT NULL THEN 'album'
         WHEN ar.mood IS NOT NULL THEN 'artist' END AS mood_level,
    CASE WHEN t.energy IS NOT NULL THEN 'track'
         WHEN al.energy IS NOT NULL THEN 'album'
         WHEN ar.energy IS NOT NULL THEN 'artist' END AS energy_level,
    -- Two levels, not three: never_shuffle is not curatable on an artist. NULL
    -- here while r_never_shuffle reads 0 is the "nobody said" case — the one
    -- a NOT NULL DEFAULT 0 made invisible in v1.
    CASE WHEN t.never_shuffle IS NOT NULL THEN 'track'
         WHEN al.never_shuffle IS NOT NULL THEN 'album' END AS never_shuffle_level
FROM tracks t
JOIN albums  al ON al.id = t.album_id
JOIN artists aa ON aa.id = al.artist_id
LEFT JOIN artists ta ON ta.id = t.artist_id
-- Whose mood a track inherits: the artist actually credited on it, which is
-- the track's own when one overrides the album's.
LEFT JOIN artists ar ON ar.id = COALESCE(t.artist_id, al.artist_id);
"""


class NotCuratable(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _text(v: Any) -> str | None:
    return None if v is None else str(v)


def connect(db_path: Path) -> sqlite3.Connection:
    """Open (and migrate) the catalogue.

    WAL because three users with different lifetimes share it: a gateway
    holding it open for days, a short-lived CLI, and an ingest doing long
    batches of writes. The default rollback journal makes readers and a writer
    exclude each other.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    migrate(con)
    return con


# --- migration ---------------------------------------------------------

def migrate(con: sqlite3.Connection) -> None:
    """Bring the schema to ``SCHEMA_VERSION``.

    Real migrations, because this database holds curation that no re-ingest can
    reproduce — rebuilding from the files is the one strategy guaranteed to
    lose the part worth keeping.

    Order matters and is the opposite of the obvious one: **existing tables are
    altered first, and only then is the DDL applied.** The DDL is all
    ``IF NOT EXISTS``, so it looks safe to run first — but its indexes and view
    reference columns a migration is about to create, and running it against an
    old table fails on a column that does not exist yet.
    """
    version = _current_version(con)
    if version is None:
        con.executescript(_DDL)
        con.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
        con.commit()
        return
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"catalogue is schema v{version} but this claw understands v{SCHEMA_VERSION}; "
            "a newer version wrote it — upgrade rather than downgrading the file"
        )
    if version < 2:
        _v1_to_v2(con)
    if version < 3:
        _v2_to_v3(con)
    if version < 4:
        _v3_to_v4(con)
    if version < 5:
        _v4_to_v5(con)
    con.executescript(_DDL)
    if version < SCHEMA_VERSION:
        con.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
    con.commit()


def _current_version(con: sqlite3.Connection) -> int | None:
    have = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()
    if have is None:
        return None
    row = con.execute("SELECT version FROM schema_version").fetchone()
    return row[0] if row else None


def _cols(con: sqlite3.Connection, table: str) -> dict[str, Any]:
    return {r[1]: r for r in con.execute(f"PRAGMA table_info({table})")}


def _v1_to_v2(con: sqlite3.Connection) -> None:
    """Curation gains levels. Two shape changes on existing tables.

    ⚠ The ``never_shuffle`` change is the interesting one. v1 declared it
    ``NOT NULL DEFAULT 0``, which makes "nobody said" and "somebody said no"
    indistinguishable — and under inheritance a stored 0 is not a default, it
    is an assertion that outranks the album it was meant to defer to, silently,
    because the value it returns is perfectly plausible. Every v1 zero was a
    default rather than a decision, so they all become NULL.
    """
    if _cols(con, "tracks")["never_shuffle"][3]:          # the NOT NULL flag
        con.executescript("""
            ALTER TABLE tracks RENAME COLUMN never_shuffle TO never_shuffle_v1;
            ALTER TABLE tracks ADD COLUMN never_shuffle INTEGER;
            UPDATE tracks SET never_shuffle = NULLIF(never_shuffle_v1, 0);
            ALTER TABLE tracks DROP COLUMN never_shuffle_v1;
        """)
        con.execute("DROP VIEW IF EXISTS catalog")
    if "subject" not in _cols(con, "edits"):
        con.executescript("""
            ALTER TABLE edits RENAME COLUMN path TO subject;
            ALTER TABLE edits ADD COLUMN scope TEXT NOT NULL DEFAULT 'track';
            DROP INDEX IF EXISTS edits_path;
        """)


def _v3_to_v4(con: sqlite3.Connection) -> None:
    """The view reports where notes and never_shuffle came from, as it already
    did for mood and energy.

    View-only. The DDL creates it ``IF NOT EXISTS``, so an existing catalogue
    would otherwise keep the old definition forever; dropping it lets the DDL
    that runs next recreate it. A view holds no data, so nothing is at risk.
    """
    con.execute("DROP VIEW IF EXISTS catalog")


def _v4_to_v5(con: sqlite3.Connection) -> None:
    """Notes add up across levels instead of the most specific shadowing the rest.

    View-only, like v4: drop it and let the DDL recreate it. No data moves —
    every note stays on the level it was written at; only how they resolve
    changes. ``notes_level`` (v4) goes: each level's note is its own column now.
    """
    con.execute("DROP VIEW IF EXISTS catalog")


def _v2_to_v3(con: sqlite3.Connection) -> None:
    """Artists and albums become rows with ids.

    v2 kept artist and album as strings on every track, with curation in
    side-tables keyed on those strings. That made a rename a fan-out across
    every track plus a re-keying pass, and left seeded fields unable to
    inherit — correct an album's genre and a track added later still arrived
    with whatever its own file said.

    Here the distinct strings become entities, the side-tables fold into them,
    and `tracks` is rebuilt around foreign keys. Nothing is derived from the
    files: every value moves across from what v2 already held.
    """
    con.execute("DROP VIEW IF EXISTS catalog")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS artists (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE,
            mood TEXT, energy INTEGER, notes TEXT);
        CREATE TABLE IF NOT EXISTS albums (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            artist_id INTEGER NOT NULL REFERENCES artists(id),
            name TEXT NOT NULL, genre TEXT, year INTEGER,
            mood TEXT, energy INTEGER, notes TEXT, never_shuffle INTEGER,
            UNIQUE (artist_id, name));
    """)
    con.execute("INSERT OR IGNORE INTO artists(name) SELECT DISTINCT artist FROM tracks")
    # An album's genre and year come from its first track, which is what the
    # v2 rows all agreed on in practice; MIN is just a deterministic pick.
    con.execute("""
        INSERT OR IGNORE INTO albums(artist_id, name, genre, year)
        SELECT ar.id, t.album, MIN(t.genre), MIN(t.year)
        FROM tracks t JOIN artists ar ON ar.name = t.artist
        GROUP BY ar.id, t.album
    """)
    if _cols(con, "artist_curation"):
        con.execute("""
            UPDATE artists SET
                mood   = (SELECT c.mood   FROM artist_curation c WHERE c.artist = artists.name),
                energy = (SELECT c.energy FROM artist_curation c WHERE c.artist = artists.name),
                notes  = (SELECT c.notes  FROM artist_curation c WHERE c.artist = artists.name)
            WHERE name IN (SELECT artist FROM artist_curation)
        """)
    if _cols(con, "album_curation"):
        con.execute("""
            UPDATE albums SET
                mood = (SELECT c.mood FROM album_curation c
                        JOIN artists ar ON ar.name = c.artist
                        WHERE ar.id = albums.artist_id AND c.album = albums.name),
                energy = (SELECT c.energy FROM album_curation c
                        JOIN artists ar ON ar.name = c.artist
                        WHERE ar.id = albums.artist_id AND c.album = albums.name),
                notes = (SELECT c.notes FROM album_curation c
                        JOIN artists ar ON ar.name = c.artist
                        WHERE ar.id = albums.artist_id AND c.album = albums.name),
                never_shuffle = (SELECT c.never_shuffle FROM album_curation c
                        JOIN artists ar ON ar.name = c.artist
                        WHERE ar.id = albums.artist_id AND c.album = albums.name)
        """)
    # Rebuild `tracks` around album_id. SQLite cannot add a NOT NULL foreign
    # key to a populated table, so this is copy-and-swap rather than ALTER.
    con.executescript("""
        CREATE TABLE tracks_v3 (
            path TEXT PRIMARY KEY,
            album_id INTEGER NOT NULL REFERENCES albums(id),
            title TEXT NOT NULL DEFAULT '', track_no INTEGER, disc_no INTEGER,
            artist_id INTEGER REFERENCES artists(id), genre TEXT, year INTEGER,
            duration_s REAL, lufs REAL, true_peak_dbfs REAL, lra_lu REAL,
            size INTEGER NOT NULL, mtime INTEGER NOT NULL, measured_at TEXT,
            measure_version INTEGER NOT NULL DEFAULT 1,
            mood TEXT, energy INTEGER, notes TEXT, never_shuffle INTEGER,
            first_seen TEXT NOT NULL);
        INSERT INTO tracks_v3
        SELECT t.path, al.id, t.title, t.track_no, t.disc_no,
               NULL,
               CASE WHEN t.genre IS NOT DISTINCT FROM al.genre THEN NULL ELSE t.genre END,
               CASE WHEN t.year  IS NOT DISTINCT FROM al.year  THEN NULL ELSE t.year  END,
               t.duration_s, t.lufs, t.true_peak_dbfs, t.lra_lu,
               t.size, t.mtime, t.measured_at, t.measure_version,
               t.mood, t.energy, t.notes, t.never_shuffle, t.first_seen
        FROM tracks t
        JOIN artists ar ON ar.name = t.artist
        JOIN albums  al ON al.artist_id = ar.id AND al.name = t.album;
        DROP TABLE tracks;
        ALTER TABLE tracks_v3 RENAME TO tracks;
        DROP TABLE IF EXISTS album_curation;
        DROP TABLE IF EXISTS artist_curation;
    """)


# --- ingest writes -----------------------------------------------------

_UPSERT = f"""
INSERT INTO tracks (
    path, album_id, title, track_no, disc_no,
    duration_s, lufs, true_peak_dbfs, lra_lu, size, mtime,
    measured_at, measure_version, first_seen
) VALUES (
    :path, :album_id, :title, :track_no, :disc_no,
    :duration_s, :lufs, :true_peak_dbfs, :lra_lu, :size, :mtime,
    :measured_at, :measure_version, :first_seen
)
ON CONFLICT(path) DO UPDATE SET
    {", ".join(f"{c} = excluded.{c}" for c in MEASURED)}
"""
# ^ Note what the conflict clause does NOT name: title, and every curated
#   column, and album_id. That omission is the whole protection mechanism —
#   there is no flag to get wrong because the unsafe statement is not written.


def _artist_id(con: sqlite3.Connection, name: str) -> int:
    row = con.execute("SELECT id FROM artists WHERE name = ?", (name,)).fetchone()
    if row:
        return row["id"]
    return con.execute("INSERT INTO artists(name) VALUES (?)", (name,)).lastrowid


def _album_id(con: sqlite3.Connection, artist: str, album: str,
              genre: str | None, year: int | None) -> int:
    aid = _artist_id(con, artist)
    row = con.execute(
        "SELECT id FROM albums WHERE artist_id = ? AND name = ?", (aid, album)
    ).fetchone()
    if row:
        return row["id"]
    # Genre and year are seeded here, from whichever track of the album was
    # ingested first, and never updated by a later one. They are album facts in
    # this tree; a track whose file disagrees keeps its own value below.
    return con.execute(
        "INSERT INTO albums(artist_id, name, genre, year) VALUES (?,?,?,?)",
        (aid, album, genre, year),
    ).lastrowid


def upsert(con: sqlite3.Connection, row: dict[str, Any]) -> None:
    """Insert a track, or refresh only what the bytes determine.

    *row* carries ``artist``/``album`` as names; they are resolved to entity
    ids here, creating rows as needed. Genre and year are seeded onto the
    **album**, from whichever of its tracks was ingested first.

    Ingest never writes a track-level genre or year, even when the file's tag
    disagrees with the album's. It cannot tell a seeded album genre from a
    corrected one, so writing the file's value would shadow the correction —
    and a newly-added track silently disagreeing with its album is the precise
    defect this schema exists to prevent. A genuine per-track genre (a
    compilation, a guest track) is curation, and says so.
    """
    with con:
        album_id = _album_id(con, row["artist"], row["album"], row.get("genre"), row.get("year"))
        payload = {k: v for k, v in row.items()
                   if k not in ("artist", "album", "genre", "year")}
        con.execute(_UPSERT, {"first_seen": _now(), "album_id": album_id, **payload})


# What a track row holds that no file can give back: logged when a row is reset
# or dropped, so the edit log can say what was lost and put it back. Track-level
# genre is here because ingest never writes one — a non-NULL value is a correction.
_TRACK_OPINIONS: tuple[str, ...] = CURATED_ONLY + ("genre",)


def _opinions(con: sqlite3.Connection, path: str) -> dict[str, Any]:
    row = con.execute(
        f"SELECT {', '.join(_TRACK_OPINIONS)} FROM tracks WHERE path = ?", (path,)
    ).fetchone()
    return {k: row[k] for k in _TRACK_OPINIONS if row and row[k] is not None}


def _log_event(con: sqlite3.Connection, path: str, event: str, lost: dict[str, Any], by: str) -> None:
    """A system event in the edit log — not curation, but it changes what curation
    exists, so the log would be silently incomplete without it."""
    con.execute(
        "INSERT INTO edits(at, scope, subject, field, old, new, by) VALUES (?,?,?,?,?,?,?)",
        (_now(), "track", path, event, json.dumps(lost, sort_keys=True) if lost else None,
         None, by),
    )


# Edit-log fields that record something happening TO curation rather than an opinion.
SYSTEM_EVENTS: tuple[str, ...] = ("repopulated", "forgotten")


def repopulate(con: sqlite3.Connection, row: dict[str, Any], by: str = "ingest") -> None:
    """Restore a track to exactly what its file says. **Discards curation.**

    Delete-then-insert rather than a wide UPDATE, so "reset this row" is a
    different statement from "refresh this row" and cannot be reached by
    accident from the ingest path. What it discarded goes into the edit log.
    """
    with con:
        _log_event(con, row["path"], "repopulated", _opinions(con, row["path"]), by)
        con.execute("DELETE FROM tracks WHERE path = ?", (row["path"],))
    upsert(con, row)


def forget(con: sqlite3.Connection, paths: Sequence[str], by: str = "ingest") -> int:
    """Drop rows for files that are no longer on disk.

    Albums and artists left with nothing are **kept**, not swept: they may hold
    curation, and an artist emptied by a botched rename is exactly the case
    where you want the annotations still there when the files come back.
    """
    if not paths:
        return 0
    with con:
        # A file that vanished is usually a rename, and path is the key — so the
        # curation it carried would otherwise be gone without a trace. Logged
        # only when there was some; a plain file leaving says nothing worth keeping.
        for p in paths:
            if lost := _opinions(con, p):
                _log_event(con, p, "forgotten", lost, by)
        cur = con.executemany("DELETE FROM tracks WHERE path = ?", [(p,) for p in paths])
    return cur.rowcount


# --- curation ----------------------------------------------------------

_SCOPE = {
    "track":  ("tracks",  "path", TRACK_CURATABLE),
    "album":  ("albums",  "id",   ALBUM_CURATABLE),
    "artist": ("artists", "id",   ARTIST_CURATABLE),
}


def curate(
    con: sqlite3.Connection, scope: str, key: Any, fields: dict[str, Any], by: str
) -> dict[str, tuple[Any, Any]]:
    """Set curated fields on one entity. Returns ``{field: (old, new)}``.

    *key* is a track path, or an album/artist id. Every level takes the same
    fields, so a fact goes wherever it is true and is inherited downwards —
    one row, including for a rename, which no longer has to touch a track at
    all.

    ``artist`` on a track or an album is given as a **name** and resolved to an
    id, creating the artist if it is new. That is how a compilation says who
    actually played on a track, and how an album moves between artists.
    """
    if scope not in _SCOPE:
        raise ValueError(f"scope must be one of {list(SCOPES)}")
    table, key_col, allowed = _SCOPE[scope]
    if unknown := sorted(set(fields) - set(allowed)):
        raise NotCuratable(
            f"not curatable at {scope} level: {unknown}; allowed here: {list(allowed)}. "
            "Measured values (loudness, duration) come from the file and are not "
            "matters of opinion."
        )
    before = con.execute(f"SELECT * FROM {table} WHERE {key_col} = ?", (key,)).fetchone()
    if before is None:
        raise KeyError(key)

    changed: dict[str, tuple[Any, Any]] = {}
    at = _now()
    with con:
        for field, new in fields.items():
            column, stored = field, new
            if field == "mood" and new is not None:
                stored = json.dumps(new)
            elif field == "artist":
                column, stored = "artist_id", (None if new is None else _artist_id(con, new))
            elif field == "name" and scope == "artist":
                if (target := find_artist(con, new)) is not None and target != key:
                    # Renaming onto a name that already exists is a MERGE, not
                    # a rename — and it is the common case, because the reason
                    # to rename is usually that the same act is filed twice.
                    merged = _merge_artists(con, key, target, at, by)
                    changed["name"] = (before["name"], new)
                    changed |= merged
                    break
            old = before[column]
            if old == stored:
                continue
            con.execute(f"UPDATE {table} SET {column} = ? WHERE {key_col} = ?", (stored, key))
            con.execute(
                "INSERT INTO edits(at, scope, subject, field, old, new, by) "
                "VALUES (?,?,?,?,?,?,?)",
                (at, scope, str(key), field, _text(old), _text(stored), by),
            )
            changed[field] = (old, stored)
    return changed


def _merge_artists(
    con: sqlite3.Connection, source: int, target: int, at: str, by: str
) -> dict[str, tuple[Any, Any]]:
    """Fold *source* into *target*: repoint its albums and tracks, then drop it.

    The target's own curation always wins — it is the row being kept, and a
    merge should not quietly rewrite opinions already recorded against it. A
    field the target has *not* got is carried across rather than discarded,
    since losing an annotation is the one outcome nobody wants from tidying up
    a duplicate.
    """
    src = con.execute("SELECT * FROM artists WHERE id = ?", (source,)).fetchone()
    dst = con.execute("SELECT * FROM artists WHERE id = ?", (target,)).fetchone()
    carried: dict[str, tuple[Any, Any]] = {}
    for field in ("mood", "energy", "notes"):
        if src[field] is not None and dst[field] is None:
            con.execute(f"UPDATE artists SET {field} = ? WHERE id = ?", (src[field], target))
            con.execute(
                "INSERT INTO edits(at, scope, subject, field, old, new, by) "
                "VALUES (?,?,?,?,?,?,?)",
                (at, "artist", str(target), field, None, _text(src[field]), by),
            )
            carried[field] = (None, src[field])

    # An album name that exists under both artists would collide on the unique
    # key. Leave those where they are and report it rather than guessing which
    # copy of a record the operator meant to keep.
    clashes = [r["name"] for r in con.execute(
        "SELECT name FROM albums WHERE artist_id = ? AND name IN "
        "(SELECT name FROM albums WHERE artist_id = ?)", (source, target))]
    if clashes:
        raise NotCuratable(
            f"cannot merge: both artists have an album called {clashes!r}. "
            "Rename or move one of them first — which copy to keep is not "
            "something this can decide."
        )
    con.execute("UPDATE albums SET artist_id = ? WHERE artist_id = ?", (target, source))
    con.execute("UPDATE tracks SET artist_id = ? WHERE artist_id = ?", (target, source))
    con.execute("DELETE FROM artists WHERE id = ?", (source,))
    con.execute(
        "INSERT INTO edits(at, scope, subject, field, old, new, by) VALUES (?,?,?,?,?,?,?)",
        (at, "artist", str(target), "merged_from", src["name"], dst["name"], by),
    )
    return carried


def clear_curation(con: sqlite3.Connection, scope: str, key: Any, by: str) -> int:
    """Drop the opinions at one level, letting the level above show through.

    Only :data:`CURATED_ONLY` — never a seeded column, whose value came from
    the file rather than from a judgement, and which nulling would destroy
    rather than restore. Resetting those is ``ingest --repopulate``.
    """
    allowed = [f for f in CURATED_ONLY if f in _SCOPE[scope][2]]
    return len(curate(con, scope, key, {f: None for f in allowed}, by))


def history(con: sqlite3.Connection, subject: str) -> list[sqlite3.Row]:
    return list(con.execute("SELECT * FROM edits WHERE subject = ? ORDER BY id", (subject,)))


def edits_for(
    con: sqlite3.Connection, subjects: Iterable[tuple[str, str]], limit: int
) -> tuple[list[sqlite3.Row], int]:
    """The edit log for a set of ``(scope, subject)`` keys, newest first, and
    the true total so a capped reply can say how much it left out."""
    keys = list(dict.fromkeys(subjects))
    if not keys:
        return [], 0
    values = ",".join("(?,?)" for _ in keys)
    flat = [x for k in keys for x in k]
    where = f"(scope, subject) IN (VALUES {values})"
    total = con.execute(f"SELECT COUNT(*) FROM edits WHERE {where}", flat).fetchone()[0]
    rows = list(con.execute(
        f"SELECT * FROM edits WHERE {where} ORDER BY id DESC LIMIT ?", [*flat, limit]))
    return rows, total


def recent_edits(con: sqlite3.Connection, limit: int) -> tuple[list[sqlite3.Row], int]:
    total = con.execute("SELECT COUNT(*) FROM edits").fetchone()[0]
    return list(con.execute("SELECT * FROM edits ORDER BY id DESC LIMIT ?", (limit,))), total


def own_curation(con: sqlite3.Connection, scope: str, key: Any) -> dict[str, Any]:
    """What one level itself says — not what a track resolves to.

    The detail view shows every level separately, because the most specific
    note wins everywhere else and would otherwise hide the ones above it.
    Mood comes back decoded; absent fields are omitted.
    """
    table, key_col, allowed = _SCOPE[scope]
    cols = [c for c in CURATED_ONLY if c in allowed]
    row = con.execute(f"SELECT {', '.join(cols)} FROM {table} WHERE {key_col} = ?",
                      (key,)).fetchone()
    if row is None:
        return {}
    out = {c: row[c] for c in cols if row[c] is not None}
    if "mood" in out:
        out["mood"] = json.loads(out["mood"])
    return out


def entity_names(con: sqlite3.Connection) -> tuple[dict[int, str], dict[int, tuple[str, str]]]:
    """``(artists by id, albums by id -> (name, artist))`` — for labelling log
    rows whose subject is an id, including entities no track points at any more."""
    artists = {r[0]: r[1] for r in con.execute("SELECT id, name FROM artists")}
    albums = {r[0]: (r[1], artists.get(r[2], "")) for r in
              con.execute("SELECT id, name, artist_id FROM albums")}
    return artists, albums


# --- reads -------------------------------------------------------------

def all_tracks(con: sqlite3.Connection) -> list[sqlite3.Row]:
    """Every track with its curation already resolved up the chain."""
    return list(con.execute("SELECT * FROM catalog"))


def find_album(con: sqlite3.Connection, artist: str, album: str) -> int | None:
    row = con.execute(
        "SELECT al.id FROM albums al JOIN artists ar ON ar.id = al.artist_id "
        "WHERE ar.name = ? AND al.name = ?", (artist, album),
    ).fetchone()
    return row["id"] if row else None


def find_artist(con: sqlite3.Connection, name: str) -> int | None:
    row = con.execute("SELECT id FROM artists WHERE name = ?", (name,)).fetchone()
    return row["id"] if row else None


def known_paths(con: sqlite3.Connection) -> dict[str, tuple[int, int, int]]:
    """``path -> (size, mtime, measure_version)`` — what ingest checks to decide
    whether a file needs looking at, without reading any of them."""
    return {
        r["path"]: (r["size"], r["mtime"], r["measure_version"])
        for r in con.execute("SELECT path, size, mtime, measure_version FROM tracks")
    }


def stats(con: sqlite3.Connection) -> dict[str, int]:
    q = lambda sql: con.execute(sql).fetchone()[0]
    return {
        "tracks": q("SELECT COUNT(*) FROM tracks"),
        "artists": q("SELECT COUNT(*) FROM artists"),
        "albums": q("SELECT COUNT(*) FROM albums"),
        "measured": q("SELECT COUNT(*) FROM tracks WHERE lufs IS NOT NULL"),
        # Resolved, so a track counts as annotated when its album or artist is.
        "with_mood": q("SELECT COUNT(*) FROM catalog WHERE r_mood IS NOT NULL"),
        "with_energy": q("SELECT COUNT(*) FROM catalog WHERE r_energy IS NOT NULL"),
        "curated_albums": q("SELECT COUNT(*) FROM albums WHERE mood IS NOT NULL "
                            "OR energy IS NOT NULL OR notes IS NOT NULL "
                            "OR never_shuffle IS NOT NULL"),
        "curated_artists": q("SELECT COUNT(*) FROM artists WHERE mood IS NOT NULL "
                             "OR energy IS NOT NULL OR notes IS NOT NULL"),
    }


def loudness_rows(con: sqlite3.Connection) -> list[tuple[float, float | None]]:
    """(integrated LUFS, true peak) for every measured track."""
    return [(r[0], r[1]) for r in con.execute(
        "SELECT lufs, true_peak_dbfs FROM tracks WHERE lufs IS NOT NULL")]


def record_plays(
    con: sqlite3.Connection, paths: Iterable[str], output: str | None,
    source: str = "queued",
) -> None:
    at = _now()
    with con:
        con.executemany(
            "INSERT INTO plays(at, path, output, source) VALUES (?,?,?,?)",
            [(at, p, output, source) for p in paths],
        )


def last_queued(con: sqlite3.Connection) -> dict[str, str]:
    """``path -> ISO time`` it was last queued, for every track ever queued."""
    return {r[0]: r[1] for r in con.execute("SELECT path, MAX(at) FROM plays GROUP BY path")}


def recent_plays(con: sqlite3.Connection, limit: int = 50) -> list[sqlite3.Row]:
    return list(con.execute("SELECT * FROM plays ORDER BY id DESC LIMIT ?", (limit,)))


def recent_records(con: sqlite3.Connection, limit: int = 8) -> list[sqlite3.Row]:
    """The last few *records* queued, newest first, one row each.

    Grouped by album rather than listed per track, because a set is remembered
    as "we had the Zorn on" and a raw track list would be forty lines of the
    same album. Resolved through the catalogue so a renamed artist shows under
    its current name.
    """
    return list(con.execute("""
        SELECT c.artist, c.album, MAX(p.at) AS at, COUNT(*) AS n
        FROM plays p JOIN catalog c ON c.path = p.path
        GROUP BY c.artist, c.album
        ORDER BY MAX(p.id) DESC
        LIMIT ?
    """, (limit,)))
