"""Local music playback: a library index, an output router, and an mpv client.

Three problems, kept apart because they fail for unrelated reasons.

**The library** is 4,000-odd files on disk under ``Artist/Album/NN - Title``.
Matching happens *here*, in Python, over an index — never by handing a path to a
shell. The names in a real music collection are a quoting minefield
(``Guns N' Roses``, ``Motörhead``, percent-encoded slashes from a decade-old
ripper), and a path composed by a language model onto a command line is the
worst possible place to discover that. The agent sends a query; only this module
ever sees a path.

**The output** is a named speaker, not a device string. Playback is routed with
mpv's own ``audio-device``, never by moving the machine-wide default: only
explicitly-targeted audio should ever reach a speaker someone is sitting next
to. That is a safety property, not a tidiness one — the Bluetooth path here
feeds a large amplifier, and macOS puts system alert sounds through whatever the
default output happens to be.

Which creates the one genuinely surprising obligation in this file: **macOS
makes a Bluetooth audio device the default output the moment it connects.** So
the reconnect that :func:`ensure_output` performs would itself re-point system
audio at the amplifier as a side effect. It therefore captures the default
before connecting and puts it back afterwards.

**mpv** runs as a long-lived idle process holding a unix socket, and everything
here talks to it over that socket. Starting a process per track would lose the
queue, and — more to the point — would reopen the audio device on every track,
which on a Bluetooth link is audible.

Reachability is reported, never worked around. A speaker that cannot be woken
produces a refusal that says so; audio is never silently redirected to a
different room, because music arriving somewhere nobody asked for is worse than
music not arriving at all.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import random
import re
import time
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import unquote

from claw.config import LoudnessConfig, MusicConfig, MusicOutput

log = logging.getLogger(__name__)

# Extensions worth indexing. Deliberately broader than the collection actually
# contains: a library grows by someone dropping files into it, and a format
# silently not appearing is a bad way to find out it is unsupported.
AUDIO_EXTS = frozenset(
    {".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wav", ".aiff", ".aif", ".wma"}
)

# `01 - Title`, `01. Title`, `01 Title`, and a disc-prefixed `1-01 Title`. Track
# numbers are part of the filename convention, not part of the title, and
# leaving them in makes every title in an album match a query for any other
# album's same position.
#
# The disc prefix must be followed by an explicit separator. Letting it be
# optional-and-bare makes `10 - The Chase` parse as disc 1, track 0 — the greedy
# read is always available, and it is always wrong.
_TRACK_PREFIX = re.compile(r"^\s*(?:\d{1,2}\s*[-_]\s*)?(\d{1,3})\s*[-._)]*\s+(?=\S)")

_NONWORD = re.compile(r"[^\w\s]+")

# Handles: how one tool's answer names a thing so another tool can act on
# exactly that thing, with no second fuzzy match in between. A search that
# found the record and a play that then matched a *different* one is the
# failure this exists to rule out. Albums and artists have integer ids; a track
# is keyed by its path, which is long and full of quoting hazards, so its handle
# is a short hash of it instead.
HANDLE = re.compile(r"^(t|al|ar):([a-z0-9]+)$")


def track_handle(rel_path: str) -> str:
    """``t:`` + 8 base32 chars of the relative path's SHA-1 — 40 bits.

    Stable across restarts and re-scans for as long as the file keeps its name,
    and short enough for a model to copy. A collision is detected at load and
    makes both handles unresolvable rather than ambiguous.
    """
    digest = hashlib.sha1(rel_path.encode("utf-8")).digest()
    return "t:" + base64.b32encode(digest).decode("ascii")[:8].lower()


def is_handle(text: str) -> bool:
    return bool(HANDLE.match(text.strip()))


def _fold(text: str) -> str:
    """Casefold, strip diacritics, drop punctuation, collapse whitespace.

    So ``Motörhead`` matches ``motorhead`` and ``Guns N' Roses`` matches
    ``guns n roses`` — which is how a person types them, and therefore how a
    model relaying a person types them.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(_NONWORD.sub(" ", stripped).casefold().split())


def _display(name: str) -> str:
    """Undo percent-encoding left behind by old ripping tools.

    ``AC%2fDC`` is a directory called ``AC/DC`` that could not be one. Applied to
    display and matching only — the real path on disk is never rewritten, since
    that is what has to be opened.
    """
    return unquote(name) if "%" in name else name


@dataclass(frozen=True)
class Track:
    path: Path
    artist: str
    album: str
    title: str
    number: int | None
    # Catalogue fields. All optional, because :func:`_parse_track` builds a
    # Track from a filename alone during ingest, before anything is measured.
    genre: str | None = None
    year: int | None = None
    duration_s: float | None = None
    lufs: float | None = None
    # How much gain the track can take upward before it clips. Not how loud it
    # sounds — that is `lufs`.
    true_peak_dbfs: float | None = None
    mood: tuple[str, ...] = ()
    energy: int | None = None
    never_shuffle: bool = False
    # Every level's note, joined lowest first — notes ADD UP rather than the most
    # specific winning (see the catalog view).
    notes: str | None = None
    # The same notes kept apart, lowest first: (("track", …), ("album", …), …).
    # How rows label them, and how a search reports a note once, on the thing
    # it was written about, rather than once per track that inherits it.
    notes_by_level: tuple[tuple[str, str], ...] = ()
    # Where the resolved mood was written — mood still takes the most specific.
    mood_level: str | None = None
    album_id: int | None = None
    handle: str = ""

    def label(self) -> str:
        who = f"{self.artist} — " if self.artist else ""
        return f"{who}{self.title}"

    def full_label(self) -> str:
        where = f" ({self.album})" if self.album else ""
        return f"{self.label()}{where}"


# A field matches a search at this score — containment, or a fuzzy match close
# enough to be a misspelling. Below it, but above NEAR_MISS_SCORE, a name is
# offered only when nothing matched at all, and labelled as a near miss.
SEARCH_MIN_SCORE = 0.6
NEAR_MISS_SCORE = 0.4

# A hit reported as another kind — an artist's tracks, a track's record — ranks
# below that kind's own matches: its score scaled by the 0.62 an artist's
# records take when the artist matched, so it keeps its source's order.
PROJECTED_SCORE = 0.62

# Reasons that already name the field they matched read the same on any row.
_LEVEL_NEUTRAL = frozenset({"artist", "album", "artist + album", "artist + title"})


def _via(level: str, why: str) -> str:
    """A reason seen from another kind of row: a note that matched on an album
    reads "album note" on that album's tracks."""
    return why if why in _LEVEL_NEUTRAL else f"{level} {why}"


@dataclass
class Hit:
    score: float
    why: list[str]
    tracks: list[Track]


@dataclass
class SearchResult:
    query: str
    scope: str | None
    genre: str | None
    years: tuple[int, int] | None
    searched: int
    artists: list[Hit] = field(default_factory=list)
    albums: list[Hit] = field(default_factory=list)
    tracks: list[Hit] = field(default_factory=list)
    near: list[str] = field(default_factory=list)
    # With a scope: how many artists, albums and tracks matched before being
    # reported as that kind — so a hit that cannot be one is never a silent miss.
    unscoped: tuple[int, int, int] | None = None


def _parse_track(root: Path, path: Path) -> Track:
    """Derive artist/album/title from position in the tree.

    Tags are not read. The tree is the convention that is actually reliable
    here — ID3 in a collection assembled over twenty years is not — and reading
    4,000 files' tags to answer one query would make the index expensive enough
    to need a cache with an invalidation story.
    """
    parts = path.relative_to(root).parts
    artist = _display(parts[0]) if len(parts) >= 2 else ""
    album = _display(parts[1]) if len(parts) >= 3 else ""
    stem = _display(path.stem)
    number: int | None = None
    if m := _TRACK_PREFIX.match(stem):
        number = int(m.group(1))
        stem = stem[m.end():]
    return Track(path=path, artist=artist, album=album, title=stem.strip(), number=number)


class Library:
    """A read-through view of the catalogue, cached for the process.

    The tree is no longer walked to answer a query — :mod:`claw.music_ingest`
    put everything in SQLite, including the loudness measurement that playback
    gain depends on and the mood and energy nobody can derive from an MP3.

    Rows are pulled in bulk and matched in Python rather than in SQL. Measured:
    at this collection's size a full scan with a compound predicate is ~0.2 ms
    against ~0.04 ms for indexed SQLite, and both are far below noticing inside
    a workflow whose other steps are a model turn and a four-minute song. What
    Python buys for that fifth of a millisecond is the fuzzy scoring below,
    which no amount of SQL expresses.
    """

    def __init__(self, db_path: Path, ttl_s: float = 60.0) -> None:
        self.db_path = db_path
        self.ttl_s = ttl_s
        self._tracks: list[Track] = []
        self._by_path: dict[Path, Track] = {}
        self._by_handle: dict[str, Track] = {}
        self._artist_ids: dict[str, int] = {}
        self._loaded_at = 0.0

    # -- loading --

    def _row_to_track(self, row: Any, root: Path) -> Track:
        # The r_* columns come from the `catalog` view, which has already
        # resolved each value down the track -> album -> artist chain.
        mood = tuple(json.loads(row["r_mood"])) if row["r_mood"] else ()
        return Track(
            path=root / row["path"],
            artist=row["artist"], album=row["album"], title=row["title"],
            number=row["track_no"], genre=row["genre"], year=row["year"],
            duration_s=row["duration_s"], lufs=row["lufs"],
            true_peak_dbfs=row["true_peak_dbfs"],
            mood=mood, energy=row["r_energy"],
            never_shuffle=bool(row["r_never_shuffle"]),
            notes=row["r_notes"], mood_level=row["mood_level"],
            notes_by_level=tuple((lvl, row[f"{lvl}_notes"]) for lvl in ("track", "album", "artist")
                                 if row[f"{lvl}_notes"]),
            album_id=row["album_id"],
            handle=track_handle(row["path"]),
        )

    def reload(self, root: Path) -> None:
        from claw import music_db

        try:
            con = music_db.connect(self.db_path)
        except Exception:
            log.exception("music: cannot open catalogue at %s", self.db_path)
            self._tracks, self._by_path, self._loaded_at = [], {}, time.monotonic()
            return
        try:
            self._tracks = [self._row_to_track(r, root) for r in music_db.all_tracks(con)]
            self._artist_ids = {r["name"]: r["id"]
                                for r in con.execute("SELECT id, name FROM artists")}
        finally:
            con.close()
        self._by_path = {t.path: t for t in self._tracks}
        self._by_handle = {}
        clashes = set()
        for t in self._tracks:
            if t.handle in self._by_handle:
                clashes.add(t.handle)
            self._by_handle[t.handle] = t
        for h in clashes:
            # Refuse rather than guess: a handle that could mean two files
            # would play the wrong one with total confidence.
            log.error("music: track handle collision on %s — both unresolvable", h)
            del self._by_handle[h]
        self._loaded_at = time.monotonic()
        log.info("music: loaded %d tracks from %s", len(self._tracks), self.db_path)

    def tracks(self, root: Path) -> list[Track]:
        if not self._tracks or (time.monotonic() - self._loaded_at) > self.ttl_s:
            self.reload(root)
        return self._tracks

    # --- matching ------------------------------------------------------

    @staticmethod
    def _score(query: str, candidate: str) -> float:
        """0.0-1.0. Substring containment beats fuzzy similarity.

        Someone asking for ``exodus`` means the album called *Exodus*, not the
        nearest string by edit distance - so a clean containment is scored above
        anything ``SequenceMatcher`` can produce, and word-boundary containment
        above containment anywhere.
        """
        if not query or not candidate:
            return 0.0
        if query == candidate:
            return 1.0
        if re.search(rf"\b{re.escape(query)}\b", candidate):
            return 0.95 - 0.15 * (1 - len(query) / len(candidate))
        if query in candidate:
            return 0.80 - 0.15 * (1 - len(query) / len(candidate))
        return 0.7 * SequenceMatcher(None, query, candidate).ratio()

    def resolve(
        self, root: Path, query: str, *, album: bool = False
    ) -> tuple[list[Track], str, str]:
        """Turn a query into a playlist.

        Returns ``(tracks, kind, label)`` where *kind* is ``album``, ``artist``,
        ``track`` or ``none``. The kind is the caller's cue for ordering: an
        album was sequenced deliberately and plays in order, while an artist or
        a loose match is a pile of songs and shuffles.

        Albums and artists are tried before individual titles regardless of
        score, because a request that names one is almost never a request for a
        single track that happens to share the word.
        """
        if is_handle(query):
            return self.by_handle(root, query)
        q = _fold(query)
        library = self.tracks(root)
        if not q or not library:
            return [], "none", ""

        best_album = self._best_group(q, library, lambda t: (t.artist, t.album), index=1)
        if best_album:
            score, key, group = best_album
            if score >= 0.6:
                ordered = sorted(group, key=lambda t: (t.number is None, t.number or 0, t.title))
                return ordered, "album", f"{key[1]} - {key[0]}" if key[0] else key[1]
        if album:
            # An explicit album request that matched nothing is a miss, not an
            # invitation to play a same-named song instead.
            return [], "none", ""

        best_artist = self._best_group(q, library, lambda t: (t.artist,), index=0)
        if best_artist:
            score, key, group = best_artist
            if score >= 0.7:
                return list(group), "artist", key[0]

        scored = [(self._score(q, _fold(t.title)), t) for t in library if t.title]
        hits = sorted((st for st in scored if st[0] >= 0.6), key=lambda st: -st[0])
        if hits:
            return [t for _, t in hits[:200]], "track", hits[0][1].full_label()
        return [], "none", ""

    def select(
        self,
        root: Path,
        *,
        genre: str | None = None,
        mood: str | None = None,
        energy: tuple[int, int] | None = None,
        years: tuple[int, int] | None = None,
        limit: int = 200,
    ) -> list[Track]:
        """Filter rather than match - how a mood request becomes a set.

        Genre and mood match loosely (substring, folded) because the tags say
        ``Progressive Rock`` where someone asks for ``prog``, and a mood is a
        word in a list rather than a category.
        """
        g = _fold(genre) if genre else None
        m = _fold(mood) if mood else None
        out = []
        for t in self.tracks(root):
            if g and not (t.genre and g in _fold(t.genre)):
                continue
            if m and not any(m in _fold(w) for w in t.mood):
                continue
            if energy and (t.energy is None or not energy[0] <= t.energy <= energy[1]):
                continue
            if years and (t.year is None or not years[0] <= t.year <= years[1]):
                continue
            out.append(t)
        return out[:limit]

    @staticmethod
    def _best_group(
        q: str,
        library: Sequence[Track],
        key_of,
        *,
        index: int,
    ) -> tuple[float, tuple[str, ...], list[Track]] | None:
        groups: dict[tuple[str, ...], list[Track]] = {}
        for t in library:
            key = key_of(t)
            if not key[index]:
                continue
            groups.setdefault(key, []).append(t)
        best: tuple[float, tuple[str, ...], list[Track]] | None = None
        for key, group in groups.items():
            score = Library._score(q, _fold(key[index]))
            if best is None or score > best[0]:
                best = (score, key, group)
        return best

    # --- handles ------------------------------------------------------

    def artist_handle(self, root: Path, name: str) -> str:
        self.tracks(root)
        aid = self._artist_ids.get(name)
        return f"ar:{aid}" if aid is not None else ""

    def by_handle(self, root: Path, handle: str) -> tuple[list[Track], str, str]:
        """Exact resolution — the whole point of a handle. An unknown handle
        resolves to nothing, never to the nearest thing."""
        m = HANDLE.match(handle.strip())
        if not m:
            return [], "none", ""
        kind, key = m.groups()
        library = self.tracks(root)
        if kind == "t":
            t = self._by_handle.get(handle.strip())
            return ([t], "track", t.full_label()) if t else ([], "none", "")
        if not key.isdigit():
            return [], "none", ""
        if kind == "al":
            got = sorted((t for t in library if t.album_id == int(key)),
                         key=lambda t: (t.number is None, t.number or 0, t.title))
            if not got:
                return [], "none", ""
            head = got[0]
            return got, "album", f"{head.album} - {head.artist}" if head.artist else head.album
        name = next((n for n, i in self._artist_ids.items() if i == int(key)), None)
        got = [t for t in library if name is not None and t.artist == name]
        return (got, "artist", name) if got else ([], "none", "")

    def resolve_artist(self, root: Path, name: str) -> tuple[list[Track], str, str]:
        """An artist by name — the same threshold resolve() uses, but never an
        album or a title, because the caller asked for an artist."""
        q = _fold(name)
        best = self._best_group(q, self.tracks(root), lambda t: (t.artist,), index=0)
        if best and best[0] >= 0.7:
            return list(best[2]), "artist", best[1][0]
        return [], "none", ""

    # --- search ---------------------------------------------------------

    def search(
        self,
        root: Path,
        query: str,
        *,
        scope: str | None = None,
        genre: str | None = None,
        years: tuple[int, int] | None = None,
    ) -> "SearchResult":
        """Everything that matches — no sampling, no quiet exclusions.

        Unlike :meth:`resolve`, which picks the one best thing to *play*, this
        reports every artist, album and track the query touches, so that "is it
        here?" has an honest answer. The only narrowing is what the caller
        passes explicitly, and the result carries it back so the reply can say
        so.

        Matching is on artist, album, title, genre, notes and mood words. Each
        query word must appear somewhere in a track's fields (so ``marley
        exodus`` finds the album by that artist), with a fuzzy fallback on each
        field for misspellings. A note or mood is attributed to the level it was
        written at, so an artist-level note is one artist row, not one row per
        track that inherits it. (Notes add up across levels, so each level's
        own note is matched on its own row.)

        ``scope`` changes how hits are reported, never what matches: every hit
        is re-expressed as that kind. An artist that matched comes back as
        their records or their tracks, a track as its record or its artist.
        """
        q = _fold(query)
        words = q.split()
        g = _fold(genre) if genre else None
        pool = [t for t in self.tracks(root)
                if (not g or (t.genre and g in _fold(t.genre)))
                and (not years or (t.year is not None and years[0] <= t.year <= years[1]))]
        res = SearchResult(query=query, scope=scope, genre=genre, years=years,
                           searched=len(pool))
        if not words:
            return res

        def hit(text: str | None) -> float:
            if not text:
                return 0.0
            f = _fold(text)
            s = self._score(q, f)
            if s >= SEARCH_MIN_SCORE:
                return s
            return 0.75 if all(w in f for w in words) else 0.0

        artists: dict[str, list[Track]] = {}
        albums: dict[int, list[Track]] = {}
        for t in pool:
            artists.setdefault(t.artist, []).append(t)
            if t.album_id is not None:
                albums.setdefault(t.album_id, []).append(t)

        for name, ts in artists.items():
            why = []
            if (s := hit(name)):
                why.append(("artist", s))
            if any(hit(dict(t.notes_by_level).get("artist")) for t in ts):
                why.append(("note", 0.7))
            if any(t.mood_level == "artist" and hit(" ".join(t.mood)) for t in ts):
                why.append(("mood", 0.7))
            if why:
                res.artists.append(Hit(max(s for _, s in why), [w for w, _ in why], ts))
        by_artist = {h.tracks[0].artist for h in res.artists if "artist" in h.why}

        def ordered(ts: list[Track]) -> list[Track]:
            return sorted(ts, key=lambda t: (t.number is None, t.number or 0, t.title))

        for aid, ts in albums.items():
            head = ts[0]
            why = []
            if (s := hit(head.album)):
                why.append(("album", s))
            elif len(words) > 1 and all(w in _fold(f"{head.artist} {head.album}") for w in words):
                why.append(("artist + album", 0.8))
            elif head.artist in by_artist:
                # An artist match lists their records too — "what do we have
                # by X" is half of what a search is for. Ranked just below a
                # record whose own name matched.
                why.append(("artist", 0.62))
            genres = {t.genre for t in ts if t.genre}
            if any(hit(x) for x in genres):
                why.append(("genre", 0.65))
            if any(hit(dict(t.notes_by_level).get("album")) for t in ts):
                why.append(("note", 0.7))
            if any(t.mood_level == "album" and hit(" ".join(t.mood)) for t in ts):
                why.append(("mood", 0.7))
            if why:
                res.albums.append(Hit(max(s for _, s in why), [w for w, _ in why], ordered(ts)))

        for t in pool:
            why = []
            if (s := hit(t.title)):
                why.append(("title", s))
            elif len(words) > 1 and all(
                    w in _fold(f"{t.artist} {t.album} {t.title}") for w in words) \
                    and not all(w in _fold(f"{t.artist} {t.album}") for w in words):
                why.append(("artist + title", 0.8))
            if hit(dict(t.notes_by_level).get("track")):
                why.append(("note", 0.7))
            if t.mood_level == "track" and hit(" ".join(t.mood)):
                why.append(("mood", 0.7))
            if why:
                res.tracks.append(Hit(max(s for _, s in why), [w for w, _ in why], [t]))

        found = bool(res.artists or res.albums or res.tracks)
        if scope:
            res.unscoped = (len(res.artists), len(res.albums), len(res.tracks))
            merged: dict[Any, Hit] = {}

            def keep(key: Any, score: float, why: list[str], tracks: list[Track]) -> None:
                if (h := merged.get(key)) is None:
                    merged[key] = Hit(score, list(why), tracks)
                else:
                    h.score = max(h.score, score)
                    h.why += [w for w in why if w not in h.why]

            # The scope's own hits first, so their reasons lead.
            levels = sorted((("artist", res.artists), ("album", res.albums),
                             ("track", res.tracks)), key=lambda lv: lv[0] != scope)
            for level, hits in levels:
                for h in hits:
                    score = h.score if level == scope else h.score * PROJECTED_SCORE
                    why = h.why if level == scope else [_via(level, w) for w in h.why]
                    if scope == "track":
                        for t in h.tracks:
                            keep(t.path, score, why, [t])
                    elif scope == "album":
                        for aid in {t.album_id for t in h.tracks if t.album_id is not None}:
                            keep(aid, score, why, ordered(albums[aid]))
                    else:
                        for name in {t.artist for t in h.tracks if t.artist}:
                            keep(name, score, why, artists[name])
            kept = list(merged.values())
            res.artists = kept if scope == "artist" else []
            res.albums = kept if scope == "album" else []
            res.tracks = kept if scope == "track" else []
        for hits in (res.artists, res.albums, res.tracks):
            hits.sort(key=lambda h: (-h.score, _fold(h.tracks[0].artist), _fold(h.tracks[0].album),
                                     h.tracks[0].number is None, h.tracks[0].number or 0,
                                     _fold(h.tracks[0].title)))

        if not found:
            near: dict[str, float] = {}
            for t in pool:
                for name, label in ((t.artist, t.artist),
                                    (t.album, f"{t.album} — {t.artist}"),
                                    (t.title, t.full_label())):
                    if name and (sc := self._score(q, _fold(name))) >= NEAR_MISS_SCORE:
                        near[label] = max(sc, near.get(label, 0.0))
            res.near = [k for k, _ in sorted(near.items(), key=lambda kv: -kv[1])[:3]]
        return res

    def album_means(self, root: Path) -> dict[tuple[str, str], float]:
        """Mean loudness per record, for spotting album furniture."""
        sums: dict[tuple[str, str], list[float]] = {}
        for t in self.tracks(root):
            if t.lufs is not None:
                sums.setdefault((t.artist, t.album), []).append(t.lufs)
        return {k: sum(v) / len(v) for k, v in sums.items()}

    def drop_furniture(
        self, root: Path, tracks: Sequence[Track], loudness: LoudnessConfig
    ) -> list[Track]:
        """Remove codas, segues and skits — for a shuffle only.

        Never applied to a record played in its own order, where these are the
        joins that hold it together.
        """
        means = self.album_means(root)
        kept = [t for t in tracks
                if not is_furniture(t, means.get((t.artist, t.album)), loudness)]
        return kept or list(tracks)      # never return nothing

    def stats(self, root: Path) -> tuple[int, int, int]:
        library = self.tracks(root)
        return (
            len(library),
            len({t.artist for t in library if t.artist}),
            len({(t.artist, t.album) for t in library if t.album}),
        )

    def by_path(self, root: Path, path: str) -> Track | None:
        self.tracks(root)
        return self._by_path.get(Path(path))


# --- loudness policy ---------------------------------------------------
#
# LUFS is how loud a track sounds; true peak is how much gain it can take. The
# policy lives in LoudnessConfig — see its docstring for why it aims at the
# collection's mode and boosts only within measured headroom.

# Floor, so a pathological measurement cannot mute a track outright. A safety
# rail rather than a property of any collection, hence not in config.
MIN_GAIN_DB = -24.0


def gain_db(lufs: float | None, true_peak: float | None, ld: LoudnessConfig) -> float:
    """Gain in dB that brings a track toward ``ld.target_lufs``.

    **Down is free; up is bounded by the track's own true peak.** A louder track
    is cut all the way to the target. A quieter one is boosted toward it only
    until its peak would reach ``ld.boost_ceiling_dbtp`` — in a collection that
    mostly peaks near full scale that is often not far, and going further would
    need a limiter, which changes the music rather than its level.

    An unmeasured track is assumed to be ``ld.assumed_lufs`` loud and is never
    boosted: without a peak there is nothing to bound the boost by.
    """
    measured = ld.assumed_lufs if lufs is None else lufs
    want = ld.target_lufs - measured
    if want <= 0:
        return max(MIN_GAIN_DB, want)
    if lufs is None or true_peak is None:
        return 0.0
    return max(0.0, min(want, ld.boost_ceiling_dbtp - true_peak))


def album_gain_db(tracks: Sequence[Track], ld: LoudnessConfig) -> float:
    """One gain for a whole record, so its own quiet/loud relationships survive.

    Loudness is duration-weighted in the **energy** domain rather than a mean of
    dB values, which is what R128 album gain means: a loud eight-minute track
    counts for more than a loud two-minute one. Headroom is the record's
    *loudest* peak, since one figure has to be safe for every track it moves —
    so one unmeasured track means no boost for the record.
    """
    num = den = 0.0
    for t in tracks:
        seconds = t.duration_s or 1.0
        num += seconds * 10 ** ((ld.assumed_lufs if t.lufs is None else t.lufs) / 10)
        den += seconds
    if den == 0:
        return gain_db(None, None, ld)
    peaks = [t.true_peak_dbfs for t in tracks]
    peak = None if any(p is None for p in peaks) or any(t.lufs is None for t in tracks) \
        else max(peaks)
    return gain_db(10 * math.log10(num / den), peak, ld)


def loudness_mode(values: Iterable[float], bandwidth: float = 1.0) -> float | None:
    """The most common loudness in a collection — where ``target_lufs`` belongs.

    A Gaussian kernel density over 0.1 LU bins, rather than the tallest bin of a
    histogram, which jumps between neighbours on a handful of tracks. Readings
    at the -70 LUFS gate floor are silence (a rip artifact, not a record) and
    are left out.
    """
    step = 0.1
    bins: dict[int, int] = {}
    for v in values:
        if v > -69.0:
            k = round(v / step)
            bins[k] = bins.get(k, 0) + 1
    if not bins:
        return None
    reach = int(4 * bandwidth / step)
    best_k, best_d = None, -1.0
    for k in range(min(bins) - reach, max(bins) + reach + 1):
        d = sum(n * math.exp(-0.5 * ((k - j) * step / bandwidth) ** 2)
                for j in range(k - reach, k + reach + 1) if (n := bins.get(j)))
        if d > best_d:
            best_k, best_d = k, d
    return round(best_k * step, 1)


# --- album furniture ---------------------------------------------------


def is_furniture(track: Track, album_mean_lufs: float | None, ld: LoudnessConfig) -> bool:
    """Whether a track only makes sense in its album's running order.

    A track much quieter than its own record, and short: codas, segues, spoken
    intros, skits. Used to keep it out of a **shuffle**, never to change its
    level: these are quiet because someone decided they should be, and lifting
    "Goodbye Cruel World" to match the record around it would be vandalism. In
    album order they are essential; dropped into a shuffled set they are a dead
    half-minute that makes the whole set feel broken.
    """
    if album_mean_lufs is None or track.lufs is None:
        return False
    if (track.duration_s or 0) > ld.furniture_max_s:
        return False
    return track.lufs < album_mean_lufs - ld.furniture_below_album_lu


# --- mpv IPC -----------------------------------------------------------


class MpvUnavailable(RuntimeError):
    """mpv is not holding the socket — the daemon is down, or never started."""


async def _exec(program: str | Path, args: Iterable[str], timeout: float) -> tuple[int, str, str]:
    """Run a program with argv (never a shell) under a hard wall-clock cap."""
    try:
        proc = await asyncio.create_subprocess_exec(
            str(program), *[str(a) for a in args],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, PermissionError) as exc:
        return 127, "", f"{program}: {exc}"
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", f"timed out after {timeout:g}s"
    return (
        proc.returncode or 0,
        out.decode(errors="replace").strip(),
        err.decode(errors="replace").strip(),
    )


async def ipc(socket_path: Path, commands: Sequence[Sequence[Any]], timeout: float = 10.0) -> list[dict]:
    """Send commands to mpv over its JSON IPC socket and collect the replies.

    A fresh connection per exchange, deliberately. mpv multiplexes replies with
    an unsolicited event stream on the same socket, so a long-lived connection
    would need a reader task draining events forever just to keep the buffer
    from filling — state whose only purpose is to avoid a connect that costs
    microseconds on a unix socket.
    """
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(socket_path)), timeout
        )
    except (FileNotFoundError, ConnectionRefusedError, asyncio.TimeoutError, OSError) as exc:
        raise MpvUnavailable(f"cannot reach mpv on {socket_path}: {exc}") from exc
    try:
        for i, cmd in enumerate(commands, 1):
            writer.write(json.dumps({"command": list(cmd), "request_id": i}).encode() + b"\n")
        await writer.drain()
        replies: dict[int, dict] = {}
        while len(replies) < len(commands):
            line = await asyncio.wait_for(reader.readline(), timeout)
            if not line:
                raise MpvUnavailable("mpv closed the IPC connection")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Events carry no request_id; they are someone else's business.
            if (rid := msg.get("request_id")) is not None:
                replies[rid] = msg
        return [replies[i] for i in range(1, len(commands) + 1)]
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass


def _data(reply: dict, fallback: Any = None) -> Any:
    """mpv answers a property that is simply not set right now with an error
    rather than a null — an idle player has no ``path`` — so a failed read is
    routine and returns the fallback instead of raising."""
    return reply.get("data") if reply.get("error") == "success" else fallback


# --- output routing ----------------------------------------------------


@dataclass(frozen=True)
class OutputState:
    output: MusicOutput
    connected: bool | None   # None = not a Bluetooth output, so not applicable
    present: bool            # visible to mpv as an audio device right now


class Player:
    """Everything the tools do, with no tool-shaped strings in it.

    Kept free of presentation so the same operations back both the agent tools
    and the operator's shell wrapper, and so the interesting parts — matching,
    the reconnect dance — are testable without going through a Tool.
    """

    def __init__(self, cfg: MusicConfig, blueutil: Path) -> None:
        self.cfg = cfg
        # Borrowed from the bluetooth block rather than declared again here:
        # two paths to the same binary is two things to keep in step, and the
        # one that is wrong is always the one nobody is looking at.
        self.blueutil = blueutil
        self.library = Library(cfg.db_path)

    # -- devices --

    async def device_ids(self) -> set[str]:
        (reply,) = await ipc(self.cfg.mpv_socket, [["get_property", "audio-device-list"]])
        return {d.get("name", "") for d in (_data(reply) or [])}

    async def _bt(self, args: Sequence[str], timeout: float) -> tuple[int, str, str]:
        """Run blueutil through the console GUI session.

        Not in the gateway's own daemon context: on macOS some Bluetooth verbs
        are per-user operations that answer *plausibly and wrongly* outside a
        console session rather than failing. Routing every call the one way that
        is known to be correct avoids encoding a belief about which verbs are
        safe — a belief nothing at runtime could check.
        """
        from claw.tools.bluetooth import run_in_console_session

        return await run_in_console_session(self.blueutil, args, timeout)

    async def _default_output_name(self) -> str | None:
        rc, out, _ = await _exec(self.cfg.switchaudio_binary, ["-c", "-t", "output"], 15)
        return out or None if rc == 0 else None

    async def _set_default_output(self, name: str) -> None:
        rc, _, err = await _exec(self.cfg.switchaudio_binary, ["-s", name, "-t", "output"], 15)
        if rc != 0:
            log.warning("music: could not restore default output to %r: %s", name, err)

    async def state_of(self, output: MusicOutput) -> OutputState:
        present = output.mpv_device in await self.device_ids()
        connected: bool | None = None
        if output.bluetooth_address:
            rc, out, _ = await self._bt(["--is-connected", output.bluetooth_address], 30)
            connected = (rc == 0 and out.strip() == "1")
        return OutputState(output=output, connected=connected, present=present)

    async def ensure_output(self, output: MusicOutput) -> str | None:
        """Make *output* the one mpv plays through. Returns a reason on refusal.

        The restore-the-default step is not defensive tidiness. macOS promotes a
        Bluetooth audio device to system default the instant it connects, so
        without it every reconnect performed here would quietly re-point system
        audio — including alert sounds, which have their own volume — at
        whatever this speaker is plugged into.
        """
        if output.bluetooth_address:
            rc, out, _ = await self._bt(["--is-connected", output.bluetooth_address], 30)
            if not (rc == 0 and out.strip() == "1"):
                previous = await self._default_output_name()
                rc, out, err = await self._bt(
                    ["--connect", output.bluetooth_address], self.cfg.connect_timeout_s + 20
                )
                if rc != 0:
                    return (
                        f"{output.name} did not answer: {err or out or 'connect failed'}. "
                        "That normally means it is switched off or out of range — or "
                        "that another machine currently holds it, since these receivers "
                        "accept one link at a time."
                    )
                if not await self._await_device(output.mpv_device):
                    return (
                        f"{output.name} connected but never appeared as an audio device "
                        f"within {self.cfg.connect_timeout_s:g}s. Nothing was played and "
                        "no other speaker was substituted."
                    )
                if previous and previous != output.coreaudio_name:
                    await self._set_default_output(previous)
        elif output.mpv_device not in await self.device_ids():
            return f"{output.name} is not present as an audio device on this machine."

        (reply,) = await ipc(
            self.cfg.mpv_socket, [["set_property", "audio-device", output.mpv_device]]
        )
        if reply.get("error") != "success":
            return f"mpv refused to route to {output.name}: {reply.get('error')}"
        return None

    async def _await_device(self, mpv_device: str) -> bool:
        deadline = asyncio.get_running_loop().time() + self.cfg.connect_timeout_s
        while asyncio.get_running_loop().time() < deadline:
            if mpv_device in await self.device_ids():
                return True
            await asyncio.sleep(0.5)
        return False

    # -- playback --

    def is_clip(self, path: str | Path | None) -> bool:
        """Whether a playlist entry is a spoken DJ link rather than a track.

        By path — the render dir is the whole mechanism (see claw.dj). Imported
        lazily: dj imports this module.
        """
        from claw.dj import Announcer
        return Announcer(self.cfg.dj).is_clip(path)

    async def play(
        self,
        tracks: Sequence[Any],
        *,
        append: bool = False,
        album_gain: bool = False,
        gains: Sequence[float | None] | None = None,
    ) -> None:
        """Queue tracks with their loudness gain baked into each entry.

        The gain rides on ``loadfile``'s per-file options rather than being set
        on the player, which is what makes per-track normalisation possible with
        no event loop: mpv applies the option for that entry only and drops it
        when the entry ends. Setting the volume property instead would need
        something watching for track changes and racing them.

        ``album_gain`` computes one figure for the whole queue, so a record's
        own quiet-track/loud-track relationships survive. Correct for an album
        played in order, wrong for a shuffle — where the point is that
        consecutive tracks from different decades should match.

        ``gains`` is for a queue that mixes the two — a DJ set with a whole
        record inside it: one dB figure per entry, aligned with ``tracks``, and
        ``None`` for "the usual rule". Links stay at unity whatever it says.

        ⚠ The gain goes through ``volume-gain``, which is in dB. mpv's
        ``volume`` looks like a percentage but is **cubic** (50 is -18 dB, not
        -6), and converting dB to it as if it were linear once made every cut
        three times deeper than intended. Nor does ``--volume-gain-max`` cap a
        per-file option, so the never-past-the-peak rule is enforced in
        :func:`gain_db`, not by mpv.
        """
        ld = self.cfg.loudness
        music = [t for t in tracks if isinstance(t, Track)]
        one = album_gain_db(music, ld) if (album_gain and ld.normalize and music) else None
        cmds: list[list[Any]] = []
        if append:
            cmds.extend(await self._drop_trailing_signoff())
        for i, track in enumerate(tracks):
            mode = "append-play" if (append or i) else "replace"
            if not isinstance(track, Track):
                # A spoken link. Unity, deliberately: its level is set when it
                # is rendered (DjConfig.target_lufs), where the finished clip is
                # measured and peak-checked. A gain here would bypass both.
                gain = 0.0
            elif gains is not None and gains[i] is not None:
                gain = gains[i]
            elif one is not None:
                gain = one
            elif ld.normalize:
                gain = gain_db(track.lufs, track.true_peak_dbfs, ld)
            else:
                gain = 0.0
            cmds.append(["loadfile", str(track.path), mode, 0, f"volume-gain={gain:.2f}"])
        if not append:
            cmds.append(["set_property", "pause", False])
        # mpv applies commands in order on one connection, so the whole
        # playlist swap is a single exchange and never half-applied.
        await ipc(self.cfg.mpv_socket, cmds, timeout=max(10.0, len(cmds) * 0.05))

    async def _drop_trailing_signoff(self) -> list[list[Any]]:
        """If the queue ends in a spoken link, remove it before appending.

        A set that ended "that's me for tonight" and then gets more music
        appended would otherwise say goodbye in the middle of the evening. Left
        alone when that link is the one being spoken right now — cutting a
        sentence off is worse than a goodbye that turns out not to be one.
        """
        try:
            (reply,) = await ipc(self.cfg.mpv_socket, [["get_property", "playlist"]])
        except MpvUnavailable:
            return []
        entries = _data(reply, []) or []
        if not entries:
            return []
        last = entries[-1]
        if self.is_clip(last.get("filename")) and not last.get("current"):
            return [["playlist-remove", len(entries) - 1]]
        return []

    async def control(self, action: str) -> str:
        if action == "pause":
            await ipc(self.cfg.mpv_socket, [["set_property", "pause", True]])
            return "paused"
        if action == "resume":
            await ipc(self.cfg.mpv_socket, [["set_property", "pause", False]])
            return "resumed"
        if action == "next":
            return render_skip(await self.skip())
        if action == "stop":
            await ipc(self.cfg.mpv_socket, [["stop"], ["playlist-clear"]])
            return "stopped"
        raise ValueError(action)

    def entry(self, filename: str | None) -> "Track | str | None":
        """A queue entry as a caller should see it: a Track, "a DJ link", or
        None for something the catalogue does not know."""
        if not filename:
            return None
        if self.is_clip(filename):
            return "a DJ link"
        return self.library.by_path(self.cfg.library_root, filename)

    async def skip(self) -> dict[str, Any]:
        """Skip to the next queue entry, reporting what that means.

        The queue is read BEFORE skipping, so the result says what comes next
        rather than guessing from what plays afterwards (mpv takes a moment to
        open the next file). ``result`` is:

        * ``skipped`` — ``next`` is the Track or "a DJ link" now starting, or
          None when the catalogue does not know the file;
        * ``end_of_queue`` — that was the last entry: the skip went through
          and nothing is playing now. (``playlist-next force`` succeeds on the
          last entry and stops playback; it never reports "nothing further".)
        * ``idle`` — nothing was playing, so nothing was done.
        """
        playlist_r, idle_r = await ipc(
            self.cfg.mpv_socket,
            [["get_property", "playlist"], ["get_property", "idle-active"]],
        )
        playlist = _data(playlist_r) or []
        cur = next((i for i, e in enumerate(playlist) if e.get("current")), None)
        if _data(idle_r) or cur is None:
            return {"result": "idle"}
        nxt = playlist[cur + 1].get("filename") if cur + 1 < len(playlist) else None
        await ipc(self.cfg.mpv_socket, [["playlist-next", "force"]])
        if nxt is None:
            return {"result": "end_of_queue"}
        return {"result": "skipped", "next": self.entry(nxt)}

    async def status(self) -> dict[str, Any]:
        props = [
            "path", "media-title", "pause", "idle-active",
            "playlist-pos-1", "playlist-count",
            "time-pos", "duration", "audio-device", "playlist",
        ]
        replies = await ipc(
            self.cfg.mpv_socket, [["get_property", p] for p in props]
        )
        got = {p: _data(r) for p, r in zip(props, replies)}
        entry = self.entry

        playlist = got.get("playlist") or []
        cur = next((i for i, e in enumerate(playlist) if e.get("current")), None)
        nxt = playlist[cur + 1].get("filename") if cur is not None and cur + 1 < len(playlist) else None
        tail = playlist[-1].get("filename") if playlist else None
        now = entry(got.get("path"))
        output = next(
            (o for o in self.cfg.outputs if o.mpv_device == got.get("audio-device")), None
        )
        return {
            **got,
            "track": now if isinstance(now, Track) else None,
            "clip": now == "a DJ link",
            "next": entry(nxt),
            "tail": entry(tail) if tail != nxt else None,
            "output": output,
        }


def render_skip(res: dict[str, Any]) -> str:
    """Player.skip()'s result as the text the model (and the CLI) reads."""
    if res["result"] == "idle":
        return "nothing is playing — there was nothing to skip"
    if res["result"] == "end_of_queue":
        return "skipped past the end of the queue — that was the last entry; nothing is playing now"
    nxt = res.get("next")
    if isinstance(nxt, Track):
        return f"skipped to {nxt.full_label()}"
    if nxt == "a DJ link":
        return "skipped to a DJ link (the agent talking between songs)"
    return "skipped"


# --- set candidates ------------------------------------------------------


def interleave(
    tracks: Sequence[Track],
    last_queued: dict[Path, str],
    rng: random.Random | None = None,
) -> list[Track]:
    """Order tracks so that every prefix is as varied as it can be.

    Round-robin across artists; within an artist, round-robin across their
    records; within a record, never-queued tracks first, then least recently
    queued, ties broken at random. Taking the first N of the result therefore
    gives one track from each of N artists when there are N, spreads a single
    artist across their records when that is all there is, and only repeats an
    artist or a record once variety is exhausted — no per-artist or per-album
    cap to tune.

    Artist order is random, so asking twice offers a different pool.
    """
    rng = rng or random.Random()
    by_artist: dict[str, dict[Any, list[Track]]] = {}
    for t in tracks:
        by_artist.setdefault(t.artist, {}).setdefault(t.album_id or t.album, []).append(t)

    def fresh_first(ts: list[Track]) -> list[Track]:
        keyed = [(last_queued.get(t.path, ""), rng.random(), t) for t in ts]
        return [t for *_, t in sorted(keyed, key=lambda k: (k[0], k[1]))]

    queues: list[list[Track]] = []
    for albums in by_artist.values():
        per_album = [fresh_first(ts) for ts in albums.values()]
        per_album.sort(key=lambda ts: (last_queued.get(ts[0].path, ""), rng.random()))
        merged: list[Track] = []
        while any(per_album):
            for ts in per_album:
                if ts:
                    merged.append(ts.pop(0))
        queues.append(merged)
    rng.shuffle(queues)

    out: list[Track] = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def shuffled(tracks: Sequence[Track]) -> list[Track]:
    out = list(tracks)
    random.shuffle(out)
    return out
