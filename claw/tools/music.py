"""The ``music_*`` tool family: play the local library on a named speaker.

One idea across the family — **the agent names a thing, never a device**. It asks for
``exodus`` on ``room-a``; it never sees a file path, a CoreAudio device
string, or a MAC address. All three exist, and all three are things a model
would otherwise have to guess at, mistype, or quote correctly on a shell line.

The tool descriptions are generated from config, so the ids they list are the
ids that actually exist. That is the half of the surface that must not drift,
which is why the companion skill is a guide to *taste* — when to play an album
in order, when to build a set — and deliberately restates none of it.

Refusals are load-bearing. When a speaker cannot be woken the tool says which
one and why, plays nothing, and switches nothing: quietly falling back to
another room turns "the speaker is off" into "why is the kitchen shouting".
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from claw.config import MusicConfig, MusicOutput
from claw import music_db
import json
import random
from datetime import datetime, timedelta, timezone

from claw.music import (
    Hit, MpvUnavailable, Player, SearchResult, Track, _fold, album_gain_db, interleave,
    is_furniture, is_handle, render_skip, shuffled,
)
from claw.tools.base import Tool

log = logging.getLogger(__name__)

ACTIONS: tuple[str, ...] = ("pause", "resume", "next", "stop")

# What the model is told it may set, per level. Reads from the schema so the
# message cannot drift from what curate() will actually accept.
_SCOPE_FIELDS = {
    "track": music_db.TRACK_CURATABLE,
    "album": music_db.ALBUM_CURATABLE,
    "artist": music_db.ARTIST_CURATABLE,
}

# A queue is a listening session, not the whole collection. Loading thousands of
# entries costs mpv nothing to hold but makes every status read and every
# "what's on" answer useless, and nobody has ever wanted track 900.
MAX_QUEUE = 200

_DOWN = (
    "error: the music player is not running — it is a separate service "
    "holding mpv open, and nothing can be played until it is back. Report that rather "
    "than retrying."
)


def _hms(seconds: Any) -> str:
    try:
        total = int(float(seconds))
    except (TypeError, ValueError):
        return "?"
    if total >= 3600:
        return f"{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}"
    return f"{total // 60}:{total % 60:02d}"


# How many rows of each kind a search shows. A display limit, not a filter —
# whenever one is hit the reply states the true total.
SEARCH_SHOW = {"artists": 20, "albums": 30, "tracks": 40}

# Candidate pools are drawn at random so asking twice offers a different one.
# Module-level so a test can seed it.
_RNG = random.Random()

# How many log entries music_history shows by default, and at most.
HISTORY_DEFAULT = 20
HISTORY_MAX = 200


def _zone(name: str | None):
    from zoneinfo import ZoneInfo
    try:
        return ZoneInfo(name) if name else timezone.utc
    except Exception:
        return timezone.utc


# How far a DJ set's music may run from the `minutes` it was asked for before it
# is sent back. Checked after resolving and before rendering, so a miss costs
# milliseconds rather than a wasted synthesis.
DJ_MINUTES_TOLERANCE = 0.10

_BAD_HANDLE = (
    "error: no such handle {h}. Handles come from music_search and "
    "music_candidates, and change only if a file is renamed — search again."
)


def _ago(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        then = datetime.fromisoformat(iso)
    except ValueError:
        return ""
    secs = (datetime.now(timezone.utc) - then).total_seconds()
    if secs < 3600:
        return f"{max(1, int(secs // 60))}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    return f"{int(secs // 86400)}d ago"


# A list row shows each level's note cut to this; the full text is one call
# away — music_search with the handle.
NOTE_ROW_CHARS = 90


def _note(t: Track, levels: tuple[str, ...] = ("track", "album", "artist")) -> str:
    """Every applicable note, lowest level first, each labelled and cut short.

    Notes add up across levels, so a row shows all of them rather than one.
    """
    out = ""
    for lvl, text in t.notes_by_level:
        if lvl in levels:
            short = text if len(text) <= NOTE_ROW_CHARS else text[:NOTE_ROW_CHARS - 3] + "..."
            out += f" · note ({lvl}): {short}"
    return out


def _filter_words(f: dict[str, Any]) -> str:
    bits = []
    if f.get("genre"): bits.append(f"genre {f['genre']}")
    if f.get("mood"): bits.append(f"mood {f['mood']}")
    if f.get("energy"): bits.append("energy %d-%d" % f["energy"])
    return " + ".join(bits) or "no filter"


def _describe(outputs: Sequence[MusicOutput]) -> str:
    return ", ".join(
        f"'{o.id}' ({o.name}){' — the default' if o.default else ''}" for o in outputs
    ) or "none configured"


def build_music_tools(
    cfg: MusicConfig, blueutil: Path, agent_id: str, tz: str | None = None
) -> dict[str, Tool]:
    """Build the ``music_*`` family for one agent.

    ``agent_id`` is recorded as the author of every curation edit, so the
    audit log says who formed the opinion rather than just that someone did.
    ``tz`` is the deployment's display timezone (``Config.tz``); the log itself
    stays in UTC.
    """
    player = Player(cfg, blueutil)
    output_list = _describe(cfg.outputs)

    async def _resolve_output(requested: str | None) -> MusicOutput | str:
        """Pick the speaker. Returns an error string rather than raising.

        With none named, prefer whatever is already playing over the configured
        default: a follow-up request during a listening session means *more of
        this, here*, and jumping rooms mid-session because an argument was
        omitted is the kind of surprise that makes an agent feel unsafe to use.
        """
        if requested:
            if (found := cfg.by_id(requested.strip())) is None:
                return f"error: no output called {requested!r}. Configured: {output_list}"
            return found
        try:
            current = (await player.status()).get("output")
        except MpvUnavailable:
            current = None
        if current is not None:
            return current
        if (fallback := cfg.default_output) is None:
            return f"error: no default output is configured. Name one: {output_list}"
        return fallback

    def _filters(args: dict[str, Any]) -> dict[str, Any]:
        lo, hi = args.get("energy_min"), args.get("energy_max")
        return {
            "genre": (args.get("genre") or "").strip() or None,
            "mood": (args.get("mood") or "").strip() or None,
            "energy": (int(lo or 1), int(hi or 5)) if (lo or hi) else None,
        }

    async def _play(args: dict[str, Any]) -> str:
        if args.get("handles"):
            return await _play_handles(args)
        query = (args.get("query") or "").strip()
        filters = _filters(args)
        if not query and not any(filters.values()):
            return (
                "error: say what to play — a query (artist, album or song), or "
                "at least one of genre / mood / energy_min / energy_max"
            )
        want_album = bool(args.get("album"))
        append = bool(args.get("append"))
        root = cfg.library_root

        chosen = await _resolve_output(args.get("output"))
        if isinstance(chosen, str):
            return chosen

        if query:
            tracks, kind, label = player.library.resolve(root, query, album=want_album)
            if tracks and any(filters.values()):
                keep = set(player.library.select(root, limit=10**6, **filters))
                narrowed = [t for t in tracks if t in keep]
                if not narrowed:
                    return (
                        f"{label or query} is in the library, but nothing in it matches "
                        f"{_filter_words(filters)}. Play it unfiltered, or drop the filter — "
                        "do not quietly substitute a different artist."
                    )
                tracks, kind = narrowed, "selection" if len(narrowed) < len(tracks) else kind
        else:
            tracks = player.library.select(root, **filters)
            kind, label = "selection", _filter_words(filters)

        if not tracks and is_handle(query):
            return _BAD_HANDLE.format(h=query)
        if not tracks:
            total, artists, albums = player.library.stats(root)
            return (
                f"nothing in the library matches {query or _filter_words(filters)}. "
                f"It holds {total} tracks by {artists} artists across {albums} albums — "
                "it is one person's collection, not a streaming catalogue, so a "
                "miss usually means it genuinely is not here. Say so rather than "
                "guessing at a near-miss."
            )

        # An album was sequenced on purpose; anything else is a pile of songs.
        # An explicit shuffle argument overrides either way — except where the
        # record itself is marked never_shuffle, which is curation saying so.
        shuffle = args.get("shuffle")
        if shuffle is None:
            shuffle = kind != "album"
        pinned = kind == "album" and any(t.never_shuffle for t in tracks)
        if pinned:
            shuffle = False
        if shuffle:
            # Codas, segues and spoken intros belong to a running order. In a
            # shuffle they are a dead half-minute; in album order they are the
            # joins, so this only ever applies here.
            tracks = player.library.drop_furniture(root, tracks, cfg.loudness)
        queue = (shuffled(tracks) if shuffle else list(tracks))[:MAX_QUEUE]

        try:
            if not append and (refusal := await player.ensure_output(chosen)) is not None:
                return f"refused: {refusal}"
            await player.play(queue, append=append, album_gain=(kind == "album" and not shuffle))
        except MpvUnavailable:
            return _DOWN
        _remember(queue, chosen)

        verb = "queued" if append else "playing"
        how = "shuffled" if shuffle else "in order"
        head = queue[0].full_label()
        what = (f"the album {label}" if kind == "album"
                else label if kind in ("artist", "selection") else head)
        note = " (marked never-shuffle, so in order)" if pinned else ""
        return (
            f"{verb} {what} on {chosen.name} — {len(queue)} track"
            f"{'s' if len(queue) != 1 else ''} {how}{note}"
            + ("" if append else f", starting with {head}")
        )

    async def _play_handles(args: dict[str, Any]) -> str:
        """A set chosen by the caller: exactly these, in exactly this order.

        All or nothing. One bad handle refuses the whole set, because the
        caller chose an order and a set with a hole in it is a different set.
        Nothing is dropped as an interlude either — every entry was picked.
        """
        handles = [str(h).strip() for h in args["handles"] if str(h).strip()]
        if (args.get("query") or "").strip() or any(_filters(args).values()):
            return "error: pass handles OR a query/filters, not both"
        root = cfg.library_root
        queue: list[Track] = []
        bad, whole_artist = [], []
        for h in handles:
            if h.startswith("ar:"):
                whole_artist.append(h)
                continue
            got, _, _ = player.library.by_handle(root, h)
            if not got:
                bad.append(h)
            queue.extend(got)
        if whole_artist:
            return (f"error: {', '.join(whole_artist)} is a whole artist, not a pick — "
                    "play it as the query instead. Nothing was queued.")
        if bad:
            return (f"error: no such handle(s): {', '.join(bad)}. Handles come from "
                    "music_search and music_candidates. Nothing was queued.")
        if args.get("shuffle"):
            queue = shuffled(queue)
        queue = queue[:MAX_QUEUE]
        append = bool(args.get("append"))
        chosen = await _resolve_output(args.get("output"))
        if isinstance(chosen, str):
            return chosen
        try:
            if not append and (refusal := await player.ensure_output(chosen)) is not None:
                return f"refused: {refusal}"
            await player.play(queue, append=append)
        except MpvUnavailable:
            return _DOWN
        _remember(queue, chosen)
        secs = sum(t.duration_s or 0 for t in queue)
        verb = "queued" if append else "playing"
        how = "shuffled" if args.get("shuffle") else "in your order"
        return (f"{verb} {len(queue)} track{'s' if len(queue) != 1 else ''} on {chosen.name}, "
                f"{how}, {_hms(secs)}" + ("" if append else f", starting with {queue[0].full_label()}"))

    async def _candidates(args: dict[str, Any]) -> str:
        c = cfg.candidates
        root = cfg.library_root
        library = player.library.tracks(root)

        def words(key: str) -> list[str]:
            return [w.strip() for w in str(args.get(key) or "").split(",") if w.strip()]

        genres = [_fold(g) for g in words("genre")]
        artists: set[str] = set()
        for a in words("artist"):
            got, kind, name = (player.library.by_handle(root, a) if is_handle(a)
                               else player.library.resolve_artist(root, a))
            if kind != "artist":
                return (f"error: no artist matching {a!r} in the library — music_search "
                        "shows what is there. Nothing was filtered on a guess.")
            artists.add(name)
        lo, hi = args.get("year_min"), args.get("year_max")
        years = (int(lo or 0), int(hi or 9999)) if (lo or hi) else None
        f = _filters(args)
        mood = _fold(f["mood"]) if f["mood"] else None

        matched = [
            t for t in library
            if (not genres or (t.genre and any(g in _fold(t.genre) for g in genres)))
            and (not artists or t.artist in artists)
            and (not years or (t.year is not None and years[0] <= t.year <= years[1]))
            and (not mood or any(mood in _fold(w) for w in t.mood))
            and (not f["energy"] or (t.energy is not None
                                     and f["energy"][0] <= t.energy <= f["energy"][1]))
        ]
        asked = []
        if genres: asked.append("genre " + " or ".join(repr(g) for g in words("genre")))
        if artists: asked.append("artist " + " or ".join(sorted(artists)))
        if years: asked.append(f"years {years[0]}-{years[1]}")
        if mood: asked.append(f"mood {f['mood']!r}")
        if f["energy"]: asked.append("energy %d-%d" % f["energy"])
        source = " + ".join(asked) or "the whole library"
        if not matched:
            return (f"nothing matches {source} — music_search shows what the library "
                    "holds. (Mood and energy only match what somebody has curated.)")

        # The exclusions — each one config, each one counted in the reply.
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=c.fresh_hours)).isoformat()
        queued = {root / k: v for k, v in _last_queued().items()}
        skip_genres = {_fold(g) for g in c.exclude_genres
                       if not any(q in _fold(g) for q in genres)}   # asking for one overrides
        means = player.library.album_means(root)
        counts = {"recent": 0, "interlude": 0, "genre": 0, "no duration": 0}
        eligible = []
        for t in matched:
            if queued.get(t.path, "") > cutoff:
                counts["recent"] += 1
            elif is_furniture(t, means.get((t.artist, t.album)), cfg.loudness):
                counts["interlude"] += 1
            elif t.genre and _fold(t.genre) in skip_genres:
                counts["genre"] += 1
            elif not t.duration_s:
                counts["no duration"] += 1
            else:
                eligible.append(t)

        limit = args.get("limit")
        minutes = float(args.get("minutes") or c.default_minutes)
        want_s = None if limit else minutes * 60 * c.pool_factor
        cap = min(int(limit), c.max_tracks) if limit else c.max_tracks
        pool: list[Track] = []
        secs = 0.0
        for t in interleave(eligible, queued, _RNG):
            if len(pool) >= cap or (want_s is not None and secs >= want_s):
                break
            pool.append(t)
            secs += t.duration_s or 0

        dropped = [f"{n} {what}" for what, n in (
            ("queued in the last %gh" % c.fresh_hours, counts["recent"]),
            ("interlude" if counts["interlude"] == 1 else "interludes", counts["interlude"]),
            (f"in excluded genres ({', '.join(c.exclude_genres)})", counts["genre"]),
            ("with no duration", counts["no duration"]),
        ) if n]
        head = (f"candidates from {source}: {len(matched)} tracks matched"
                + (f"; left out {', '.join(dropped)}" if dropped else "; nothing left out")
                + ".")
        if not pool:
            return head + " Nothing is left to offer — music_search shows everything, unfiltered."
        n_artists = len({t.artist for t in pool})
        n_albums = len({t.album_id for t in pool})
        size = (f"pool of {len(pool)} tracks, {_hms(secs)}, from {n_artists} artist"
                f"{'s' if n_artists != 1 else ''} and {n_albums} record{'s' if n_albums != 1 else ''}")
        if want_s is not None:
            short = secs < minutes * 60
            size += (f" — for a set of about {minutes:g} min"
                     + (f"; that is ALL that is eligible, less than the {minutes:g} min asked"
                        if short else ""))
        lines = [head, size + ".",
                 "Choose by taste, in your order, and play them with music_play(handles=[...]). "
                 "For whether something specific is in the library, use music_search."]
        for t in sorted(pool, key=lambda t: (_fold(t.artist), _fold(t.album), t.number or 0)):
            where = ", ".join(str(x) for x in (t.album, t.year) if x)
            extra = ""
            if t.mood:
                extra += f" · mood: {', '.join(t.mood)}"
            if t.energy is not None:
                extra += f" · energy {t.energy}"
            lines.append(f"  {t.handle}  {t.artist} — {t.title}{f' ({where})' if where else ''} "
                         f"{_hms(t.duration_s)}{extra}{_note(t)}")
        return "\n".join(lines)

    async def _dj(args: dict[str, Any]) -> str:
        """A run sheet: links and tracks, rendered and queued together or not at all."""
        from claw.dj import Announcer, RenderFailed

        sheet = args.get("set")
        if not isinstance(sheet, list) or not sheet:
            return "error: set is required — a list of entries, each with say, play, or both"
        root = cfg.library_root
        # (say, tracks, record label or None). A record is one entry holding its
        # whole running order; a track entry holds one track.
        plan: list[tuple[str | None, list[Track], str | None]] = []
        problems: list[str] = []
        for i, raw in enumerate(sheet, 1):
            if not isinstance(raw, dict):
                problems.append(f"entry {i} is not an object")
                continue
            say = str(raw.get("say") or "").strip() or None
            handle = str(raw.get("play") or "").strip() or None
            if not say and not handle:
                problems.append(f"entry {i} has neither say nor play")
                continue
            got: list[Track] = []
            record = None
            if handle:
                if not handle.startswith(("t:", "al:")):
                    problems.append(
                        f"entry {i}: {handle!r} is not a track or record handle — a DJ "
                        "entry plays one track (t:…) or one whole record in its own "
                        "order (al:…), from music_candidates or music_search."
                    )
                    continue
                got, kind, label = player.library.by_handle(root, handle)
                if not got:
                    problems.append(f"entry {i}: no such handle {handle}")
                    continue
                record = label if kind == "album" else None
            plan.append((say, got, record))
        if problems:
            return "error: nothing was queued — " + "; ".join(problems)

        # Two links with no music between them is one long link, badly split.
        for i in range(1, len(plan)):
            if plan[i][0] and not plan[i - 1][1]:
                return (f"error: entries {i} and {i + 1} are two links with no track "
                        "between them — merge them into one. Nothing was queued.")
        tracks = [t for _, got, _ in plan for t in got]
        append = bool(args.get("append"))
        if not tracks and not append:
            return ("error: a set with no tracks — to add a link to what is already "
                    "playing, pass append. Nothing was queued.")

        music_s = sum(t.duration_s or 0 for t in tracks)
        if (minutes := args.get("minutes")) is not None:
            want = float(minutes) * 60
            if abs(music_s - want) > want * DJ_MINUTES_TOLERANCE:
                return (f"error: the set runs {_hms(music_s)} of music against the "
                        f"{float(minutes):g} min asked — {'add' if music_s < want else 'drop'} "
                        f"about {abs(music_s - want) / 60:.0f} min and send it again. "
                        "Nothing was rendered or queued.")

        chosen = await _resolve_output(args.get("output"))
        if isinstance(chosen, str):
            return chosen
        announcer = Announcer(cfg.dj)
        announcer.reap()
        try:
            clips = await announcer.render([say for say, _, _ in plan if say])
        except RenderFailed as exc:
            return f"error: {exc}. Nothing was queued."
        it = iter(clips)
        queue: list[Any] = []
        # Per entry: a record plays at ONE gain for the whole running order, the
        # same figure music_play gives it, so its own quiet and loud tracks keep
        # their relationship; everything else takes the usual rule (None).
        gains: list[float | None] = []
        for say, got, record in plan:
            if say:
                queue.append(next(it))
                gains.append(None)
            one = (album_gain_db(got, cfg.loudness)
                   if record is not None and cfg.loudness.normalize else None)
            queue.extend(got)
            gains.extend([one] * len(got))
        try:
            if not append and (refusal := await player.ensure_output(chosen)) is not None:
                return f"refused: {refusal}"
            await player.play(queue[:MAX_QUEUE], append=append, gains=gains[:MAX_QUEUE])
        except MpvUnavailable:
            return _DOWN
        if tracks:
            _remember(tracks, chosen)
        speech = sum(c.duration_s or 0 for c in clips)
        verb = "queued" if append else "playing"
        records = [r for _, _, r in plan if r is not None]
        return (f"{verb} a DJ set on {chosen.name}: {len(tracks)} track"
                f"{'s' if len(tracks) != 1 else ''} and {len(clips)} link"
                f"{'s' if len(clips) != 1 else ''}, {_hms(music_s + speech)} "
                f"({_hms(music_s)} of music, {speech:.0f} s of talk)"
                + (f", with {', '.join(records)} whole and in order" if records else "")
                + (f", starting with {queue[0].full_label() if isinstance(queue[0], Track) else 'your opening'}"
                   if not append else ""))

    def _remember(queue: Sequence[Track], output: MusicOutput) -> None:
        """Record the queue. Best-effort — losing history must not fail a play."""
        try:
            con = music_db.connect(cfg.db_path)
        except Exception:
            log.warning("music: could not open the catalogue to record plays", exc_info=True)
            return
        try:
            music_db.record_plays(
                con, [str(t.path.relative_to(cfg.library_root)) for t in queue], output.id
            )
        except Exception:
            log.warning("music: could not record plays", exc_info=True)
        finally:
            con.close()

    def _fields_for(scope: str, args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Map the tool's arguments onto the columns of the chosen level.

        The same word means different things at different levels: `artist` at
        artist scope is a **rename**, while at album or track scope it is a
        **move** to a (possibly new) artist. `album` at album scope renames the
        record. Both land on the entity's own `name`, which is why the mapping
        lives here rather than in the schema.
        """
        allowed = {
            "track": music_db.TRACK_CURATABLE,
            "album": music_db.ALBUM_CURATABLE,
            "artist": music_db.ARTIST_CURATABLE,
        }[scope]
        rename_arg = {"artist": "artist", "album": "album"}.get(scope)
        out, rejected = {}, []
        for arg, value in args.items():
            if value is None or arg in ("query", "scope"):
                continue
            field = "name" if arg == rename_arg else arg
            if field in allowed:
                out[field] = value
            elif arg in set(music_db.TRACK_CURATABLE) | {"album", "artist"}:
                rejected.append(arg)
        return out, rejected

    async def _curate(args: dict[str, Any]) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return "error: query is required — name the track, album or artist to annotate"
        scope = (args.get("scope") or "").strip().lower() or None
        if scope and scope not in music_db.SCOPES:
            return f"error: scope must be one of: {', '.join(music_db.SCOPES)}"

        root = cfg.library_root
        tracks, kind, label = player.library.resolve(root, query)
        if not tracks and is_handle(query):
            return _BAD_HANDLE.format(h=query)
        if not tracks:
            return f"nothing in the library matches {query!r}"
        # Default to the level the query itself named. Someone who says an
        # artist means the artist; someone who names a record means the record.
        if scope is None:
            scope = {"album": "album", "artist": "artist"}.get(kind, "track")

        fields, rejected = _fields_for(scope, args)
        if not fields:
            hint = (f" ({', '.join(rejected)} cannot be set at {scope} level)"
                    if rejected else "")
            return (
                f"error: nothing to set at {scope} level{hint}. Curatable there: "
                + ", ".join(_SCOPE_FIELDS[scope])
                + ". Loudness and duration are measured from the file and are "
                  "never curatable — if one is wrong, the file is wrong."
            )
        if isinstance(fields.get("mood"), str):
            fields["mood"] = [w.strip() for w in fields["mood"].split(",") if w.strip()]
        if "energy" in fields and not 1 <= int(fields["energy"]) <= 5:
            return "error: energy runs 1 (still) to 5 (relentless)"

        head = tracks[0]
        try:
            con = music_db.connect(cfg.db_path)
        except Exception:
            return "error: the catalogue is unreachable; nothing was changed"
        try:
            if scope == "track":
                key = str(head.path.relative_to(root))
            elif scope == "album":
                key = music_db.find_album(con, head.artist, head.album)
            else:
                key = music_db.find_artist(con, head.artist)
            if key is None:
                return f"error: no {scope} row for {head.artist} — try ingesting first"
            changed = music_db.curate(con, scope, key, fields, by=agent_id)
        except KeyError:
            return f"error: {key} is not in the catalogue"
        except music_db.NotCuratable as exc:
            return f"error: {exc}"
        finally:
            con.close()
        player.library.reload(root)      # so the next query sees the change

        subject = {"track": head.full_label(),
                   "album": f"{head.album} — {head.artist}",
                   "artist": head.artist}[scope]
        what = ", ".join(f"{k}={v}" for k, v in fields.items())
        if not changed:
            return f"no change — {subject} already had {what}"
        reach = len([t for t in player.library.tracks(root)
                     if (scope == "artist" and t.artist in (head.artist, fields.get("name")))
                     or (scope == "album" and t.album == head.album
                         and t.artist in (head.artist, fields.get("artist", head.artist)))
                     or (scope == "track" and t.path == head.path)])
        return (
            f"set {what} on {subject} ({scope}) — one row, inherited by "
            f"{reach} track{'s' if reach != 1 else ''}"
        )

    def _last_queued() -> dict[str, str]:
        try:
            con = music_db.connect(cfg.db_path)
        except Exception:
            return {}
        try:
            return music_db.last_queued(con)
        except Exception:
            return {}
        finally:
            con.close()

    async def _search(args: dict[str, Any]) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return "error: query is required — an artist, album, song, genre or word from a note"
        scope = (args.get("scope") or "").strip().lower() or None
        if scope and scope not in ("artist", "album", "track"):
            return "error: scope must be one of: artist, album, track"
        lo, hi = args.get("year_min"), args.get("year_max")
        years = (int(lo or 0), int(hi or 9999)) if (lo or hi) else None
        genre = (args.get("genre") or "").strip() or None
        root = cfg.library_root
        if is_handle(query):
            return _detail(query)
        res = player.library.search(root, query, scope=scope, genre=genre, years=years)
        return _format_search(res, root)

    def _detail(handle: str) -> str:
        """Everything about exactly one thing, in full.

        Lists (search rows, crates) cut notes short to stay readable at 30 rows;
        this is the other half — every level's own curation, untruncated and
        labelled, including notes a more specific one hides elsewhere. For an
        album, its tracks with handles; for an artist, their records.
        """
        root = cfg.library_root
        lib = player.library
        tracks, kind, label = lib.by_handle(root, handle)
        if not tracks:
            return _BAD_HANDLE.format(h=handle)
        queued = _last_queued()
        means = lib.album_means(root)
        head = tracks[0]
        artist_id = lib.artist_handle(root, head.artist).removeprefix("ar:")
        try:
            con = music_db.connect(cfg.db_path)
        except Exception:
            return "error: the catalogue is unreachable"
        try:
            levels: list[tuple[str, dict]] = []
            if kind == "track":
                levels.append(("track", music_db.own_curation(
                    con, "track", str(head.path.relative_to(root)))))
            if kind in ("track", "album"):
                levels.append(("album", music_db.own_curation(con, "album", head.album_id)))
            if artist_id:
                levels.append(("artist", music_db.own_curation(con, "artist", int(artist_id))))
            per_album = ({a: music_db.own_curation(con, "album", a)
                          for a in {t.album_id for t in tracks}} if kind == "artist" else {})
            per_track = ({str(t.path.relative_to(root)): music_db.own_curation(
                              con, "track", str(t.path.relative_to(root))) for t in tracks}
                         if kind == "album" else {})
            subjects, _ = _log_scope(tracks, kind, label)
            _, n_edits = music_db.edits_for(con, subjects, 1)
        finally:
            con.close()

        def said(c: dict) -> str:
            bits = []
            if c.get("mood"):
                bits.append("mood: " + ", ".join(c["mood"]))
            if c.get("energy") is not None:
                bits.append(f"energy {c['energy']}")
            if c.get("never_shuffle") is not None:
                bits.append("never shuffle" if c["never_shuffle"] else "shuffle allowed")
            if c.get("notes"):
                bits.append(f"notes: {c['notes']}")
            return " · ".join(bits)

        def when(ts: Sequence[Track]) -> str:
            last = max((queued.get(str(t.path.relative_to(root)), "") for t in ts), default="")
            return f"last queued {_ago(last)}" if last else "never queued"

        secs = sum(t.duration_s or 0 for t in tracks)
        meta = ", ".join(str(x) for x in (head.year, head.genre) if x)
        if kind == "track":
            lines = [f"{head.handle}  {head.artist} — {head.title}",
                     f"  track {head.number or '?'} of al:{head.album_id} {head.album}"
                     f"{f' ({meta})' if meta else ''} · by ar:{artist_id} {head.artist} · "
                     f"{_hms(head.duration_s)} · {when(tracks)}"
                     + (" · interlude" if is_furniture(head, means.get((head.artist, head.album)),
                                                       cfg.loudness) else "")]
        elif kind == "album":
            lines = [f"al:{head.album_id}  {head.artist} — {head.album}{f' ({meta})' if meta else ''}",
                     f"  by ar:{artist_id} {head.artist} · {len(tracks)} tracks, {_hms(secs)} · "
                     f"{when(tracks)}"]
        else:
            n_albums = len({t.album_id for t in tracks})
            lines = [f"ar:{artist_id}  {label}",
                     f"  {n_albums} record{'s' if n_albums != 1 else ''}, {len(tracks)} tracks, "
                     f"{_hms(secs)} · {when(tracks)}"]

        lines.append("curation — each level's own. Notes add up, lowest level first; "
                     "mood and energy take the most specific:")
        if any(c for _, c in levels):
            lines.extend(f"  {lvl:<6} {said(c)}" for lvl, c in levels if c)
        else:
            lines.append("  (nothing curated at any level)")

        if kind == "album":
            lines.append("tracks:")
            for t in tracks:
                own = said(per_track.get(str(t.path.relative_to(root)), {}))
                flag = (" · interlude" if is_furniture(t, means.get((t.artist, t.album)),
                                                       cfg.loudness) else "")
                lines.append(f"  {t.number or '?':>2} {t.handle}  {t.title} {_hms(t.duration_s)}"
                             f"{flag}{f' · {own}' if own else ''}")
        elif kind == "artist":
            lines.append("records:")
            albums: dict[int, list[Track]] = {}
            for t in tracks:
                albums.setdefault(t.album_id, []).append(t)
            for aid, ts in sorted(albums.items(), key=lambda kv: (kv[1][0].year or 0, kv[1][0].album)):
                t0 = ts[0]
                m = ", ".join(str(x) for x in (t0.year, t0.genre) if x)
                own = said(per_album.get(aid, {}))
                lines.append(f"  al:{aid}  {t0.album}{f' ({m})' if m else ''} · {len(ts)} tracks, "
                             f"{_hms(sum(x.duration_s or 0 for x in ts))}{f' · {own}' if own else ''}")
        lines.append(f"history: {n_edits} edit{'s' if n_edits != 1 else ''} — music_history {handle}")
        return "\n".join(lines)

    def _format_search(res: SearchResult, root: Path) -> str:
        # Genre and years narrow what is searched. Scope does not — it is only
        # which kind the hits are reported as, so it is said separately.
        narrowed = []
        if res.genre:
            narrowed.append(f"genre {res.genre!r}")
        if res.years:
            narrowed.append(f"years {res.years[0]}-{res.years[1]}")
        within = f" within {', '.join(narrowed)}" if narrowed else ""
        if res.searched == 0:
            return f"the narrowing ({', '.join(narrowed)}) leaves nothing to search"
        if not (res.artists or res.albums or res.tracks):
            if res.unscoped and any(res.unscoped):
                a, al, t = res.unscoped
                return (f'nothing matching "{res.query}"{within} can be shown as '
                        f"{'a' if res.scope == 'track' else 'an'} {res.scope} — it matched "
                        f"{a} artist(s), {al} album(s), {t} track(s). Search without scope "
                        "to see them.")
            out = f'nothing matches "{res.query}"{within} in {res.searched} tracks.'
            if res.near:
                out += " Closest names — NOT matches: " + "; ".join(res.near) + "."
            return out

        queued = _last_queued()
        means = player.library.album_means(root)

        def rel(t: Track) -> str:
            return str(t.path.relative_to(root))

        def when(ts: Sequence[Track]) -> str:
            last = max((queued.get(rel(t), "") for t in ts), default="")
            return f" · last queued {_ago(last)}" if last else ""

        counts = {"artist": f"{len(res.artists)} artist(s)",
                  "album": f"{len(res.albums)} album(s)",
                  "track": f"{len(res.tracks)} track(s)"}
        lines = [
            f'search "{res.query}"{within}{f" as {res.scope}s" if res.scope else ""} — '
            f"{res.searched} tracks searched: "
            + (counts[res.scope] if res.scope else ", ".join(counts.values()))
        ]

        def section(title: str, key: str, hits: list[Hit], row) -> None:
            if not hits:
                return
            cap = SEARCH_SHOW[key]
            shown = f" — showing {cap} of {len(hits)}; narrow to see the rest" if len(hits) > cap else ""
            lines.append(f"{title}{shown}:")
            lines.extend("  " + row(h) for h in hits[:cap])

        def artist_row(h: Hit) -> str:
            t0 = h.tracks[0]
            n_albums = len({t.album_id for t in h.tracks})
            note = next((n for t in h.tracks if (n := _note(t, ("artist",)))), "")
            return (f"{player.library.artist_handle(root, t0.artist)}  {t0.artist} · "
                    f"{n_albums} album{'s' if n_albums != 1 else ''}, {len(h.tracks)} tracks"
                    f"{note}{when(h.tracks)} · matched: {', '.join(h.why)}")

        def album_row(h: Hit) -> str:
            ts = h.tracks
            t0 = ts[0]
            who = t0.artist if len({t.artist for t in ts}) == 1 else "Various"
            meta = ", ".join(str(x) for x in (t0.year, t0.genre) if x)
            secs = sum(t.duration_s or 0 for t in ts)
            flags = " · never shuffle" if any(t.never_shuffle for t in ts) else ""
            note = _note(t0, ("album", "artist"))
            return (f"al:{t0.album_id}  {who} — {t0.album}{f' ({meta})' if meta else ''} · "
                    f"{len(ts)} tracks, {_hms(secs)}{flags}{note}{when(ts)}"
                    f" · matched: {', '.join(h.why)}")

        def track_row(h: Hit) -> str:
            t = h.tracks[0]
            where = ", ".join(str(x) for x in (t.album, t.year) if x)
            flags = (" · interlude" if is_furniture(t, means.get((t.artist, t.album)), cfg.loudness)
                     else "")
            return (f"{t.handle}  {t.artist} — {t.title}{f' ({where})' if where else ''} "
                    f"{_hms(t.duration_s)}{flags}{_note(t)}{when([t])}"
                    f" · matched: {', '.join(h.why)}")

        section("artists", "artists", res.artists, artist_row)
        section("albums", "albums", res.albums, album_row)
        section("tracks", "tracks", res.tracks, track_row)
        return "\n".join(lines)

    async def _history(args: dict[str, Any]) -> str:
        """The curation log, read-only: for one thing (and what is inside it, and
        the levels above it), or the most recent edits across the library."""
        query = (args.get("query") or "").strip()
        try:
            limit = max(1, min(int(args.get("limit") or HISTORY_DEFAULT), HISTORY_MAX))
        except (TypeError, ValueError):
            return "error: limit must be a number"
        root = cfg.library_root
        subjects: list[tuple[str, str]] = []
        head = "the most recent edits across the library"
        if query:
            tracks, kind, label = player.library.resolve(root, query)
            if not tracks and is_handle(query):
                return _BAD_HANDLE.format(h=query)
            if not tracks:
                return (f"nothing in the library matches {query!r} — music_search shows "
                        "what is there")
            subjects, head = _log_scope(tracks, kind, label)
        try:
            con = music_db.connect(cfg.db_path)
        except Exception:
            return "error: the catalogue is unreachable"
        try:
            rows, total = (music_db.edits_for(con, subjects, limit) if query
                           else music_db.recent_edits(con, limit))
            artists, albums = music_db.entity_names(con)
        finally:
            con.close()
        if not rows:
            return f"no edits recorded for {head}"
        zone = _zone(tz)
        lines = [f"history of {head} — {total} edit{'s' if total != 1 else ''}, newest first"
                 + (f" (showing {len(rows)}; raise limit for more)" if total > len(rows) else "")
                 + ":"]
        for r in rows:
            lines.append("  " + _edit_line(r, zone, root, artists, albums))
        return "\n".join(lines)

    def _log_scope(tracks: list[Track], kind: str, label: str) -> tuple[list[tuple[str, str]], str]:
        """Which edit-log subjects belong to a thing: a track with its record and
        artist; an album with its tracks and artist; an artist with their records
        and tracks. Shared by music_history and the search detail view, so the
        count one shows is the log the other lists."""
        root = cfg.library_root
        lib = player.library

        def track_key(t: Track) -> tuple[str, str]:
            return ("track", str(t.path.relative_to(root)))

        def artist_key(name: str) -> tuple[str, str]:
            return ("artist", lib.artist_handle(root, name).removeprefix("ar:"))

        if kind == "track":
            t = tracks[0]
            return ([track_key(t), ("album", str(t.album_id)), artist_key(t.artist)],
                    f"the track {t.full_label()} ({t.handle}), with its record and artist")
        if kind == "album":
            t = tracks[0]
            return ([("album", str(t.album_id))] + [track_key(x) for x in tracks]
                    + [artist_key(n) for n in {x.artist for x in tracks}],
                    f"the album {label} (al:{t.album_id}), its tracks and its artist")
        return ([artist_key(label)] + [("album", str(a)) for a in {x.album_id for x in tracks}]
                + [track_key(x) for x in tracks],
                f"the artist {label} ({lib.artist_handle(root, label)}), their records and tracks")

    def _edit_line(r: Any, zone, root: Path, artists: dict, albums: dict) -> str:
        when = datetime.fromisoformat(r["at"]).astimezone(zone)
        subject = r["subject"]
        if r["scope"] == "track":
            t = player.library.by_path(root, str(root / subject))
            what = t.full_label() if t else f"{subject} (no longer in the library)"
        elif subject.isdigit() and r["scope"] == "album":
            name, who = albums.get(int(subject), (f"album {subject}", ""))
            what = f"{name} — {who}" if who else name
        elif subject.isdigit():
            what = artists.get(int(subject), f"artist {subject}")
        else:
            what = f"{subject} (an early entry keyed by name)"

        def val(v: Any, field: str) -> str:
            if v is None:
                return "(none)"
            if field == "artist" and str(v).isdigit():
                return artists.get(int(v), f"artist #{v} (since merged or removed)")
            if field == "mood":
                try:
                    return ", ".join(json.loads(v))
                except (ValueError, TypeError):
                    pass
            text = str(v)
            return text if len(text) <= 80 else text[:77] + "..."

        field = r["field"]
        if field in music_db.SYSTEM_EVENTS:
            lost = json.loads(r["old"]) if r["old"] else {}
            detail = (f"{field} — its curation went with it: "
                      + "; ".join(f"{k}={val(v, k)}" for k, v in lost.items())
                      if lost else f"{field}")
            change = detail
        else:
            change = f"{field}: {val(r['old'], field)} → {val(r['new'], field)}"
        return (f"{when:%Y-%m-%d %H:%M %Z} ({_ago(r['at'])})  {r['by']:<8} "
                f"{r['scope']:<6} {what} — {change}")

    def _track_fields(e: Any) -> dict[str, Any] | None:
        if isinstance(e, Track):
            return {"title": e.title, "artist": e.artist, "album": e.album}
        if e == "a DJ link":
            return {"link": True}
        return None

    async def _do_control(args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Perform ONE transport action; return (text for the model, fields
        for code). Both faces of music_control come from this single call —
        an action must never run twice to be described twice."""
        action = str(args.get("action", "")).strip().lower()
        if action not in ACTIONS:
            return (f"error: action must be one of: {', '.join(ACTIONS)}",
                    {"result": "error"})
        try:
            if action == "next":
                res = await player.skip()
                return (render_skip(res),
                        {"result": res["result"], "next": _track_fields(res.get("next"))})
            text = await player.control(action)
            return text, {"result": text}
        except MpvUnavailable:
            return _DOWN, {"result": "down"}

    async def _control(args: dict[str, Any]) -> str:
        return (await _do_control(args))[0]

    async def _control_data(args: dict[str, Any]) -> dict[str, Any]:
        """``result``: paused | resumed | stopped | skipped | end_of_queue |
        idle | down | error; after a skip, ``next`` is the entry now starting
        ({title, artist, album}, {link: True}, or None if uncatalogued)."""
        return (await _do_control(args))[1]

    def _recently() -> str:
        """A compact tail of what has been queued, for building a set on.

        Queue time, not play time — the caveat lives in the schema, and the
        wording here says "queued" rather than "played" so the tool does not
        claim more than it knows.
        """
        try:
            con = music_db.connect(cfg.db_path)
        except Exception:
            return ""
        try:
            rows = music_db.recent_records(con, limit=8)
        except Exception:
            return ""
        finally:
            con.close()
        if not rows:
            return ""
        return "recently queued: " + "; ".join(
            f"{r['album'] or 'loose tracks'} — {r['artist']}" for r in rows
        )

    async def _status_data(_args: dict[str, Any]) -> dict[str, Any]:
        """What is on now, as fields: ``state`` is playing | paused | idle |
        loading | link (a spoken DJ link between songs) | down; ``title``/``artist``/
        ``album`` are set only for a catalogued track."""
        try:
            st = await player.status()
        except MpvUnavailable:
            return {"state": "down"}
        if not st.get("path"):
            # idle-active explicitly False with no path: an entry is current
            # but its file is still opening — the few milliseconds after a
            # skip. Not "nothing is playing". Anything else is idle.
            return {"state": "loading" if st.get("idle-active") is False else "idle"}
        if st.get("clip"):
            return {"state": "link"}
        track: Track | None = st.get("track")
        return {
            "state": "paused" if st.get("pause") else "playing",
            "title": track.title if track else (st.get("media-title") or None),
            "artist": track.artist if track else None,
            "album": track.album if track else None,
        }

    async def _status(_args: dict[str, Any]) -> str:
        try:
            st = await player.status()
        except MpvUnavailable:
            return _DOWN
        recent = _recently()
        if not st.get("path"):
            if st.get("idle-active") is False:
                return "the next entry is loading — ask again in a moment"
            return "nothing is playing" + (f". {recent}" if recent else "")
        track: Track | None = st.get("track")
        if st.get("clip"):
            title = "a DJ link (the agent talking between songs)"
        else:
            title = track.full_label() if track else (st.get("media-title") or st["path"])
        output: MusicOutput | None = st.get("output")
        where = output.name if output else (st.get("audio-device") or "an unconfigured device")
        pos, count = st.get("playlist-pos-1"), st.get("playlist-count")
        bits = [
            f"{'paused at' if st.get('pause') else 'playing'} {title}",
            f"on {where}",
            f"{_hms(st.get('time-pos'))} of {_hms(st.get('duration'))}",
        ]
        if pos and count:
            bits.append(f"entry {pos} of {count} in the queue")

        def label(e: Any) -> str:
            return e.full_label() if isinstance(e, Track) else str(e)

        if st.get("next") is not None:
            bits.append(f"next: {label(st['next'])}")
        if st.get("tail") is not None:
            # What a link appended after this set would follow — the seam.
            bits.append(f"the queue ends with {label(st['tail'])}")
        line = ", ".join(bits)
        return f"{line}. {recent}" if recent else line

    async def _outputs(_args: dict[str, Any]) -> str:
        lines = []
        for out in cfg.outputs:
            try:
                state = await player.state_of(out)
                if state.connected is False:
                    how = "paired but not connected — will be woken on play"
                elif state.present:
                    how = "ready"
                else:
                    how = "not currently visible as an audio device"
            except MpvUnavailable:
                how = "unknown (the player is not running)"
            lines.append(
                f"{out.id} — {out.name}{' [default]' if out.default else ''}: {how}"
            )
        return "\n".join(lines) or "no outputs are configured"

    tools = {
        "music_play": Tool(
            name="music_play",
            description=(
                "Play music from the local library on a named speaker. The query "
                "is an artist, an album, or a song title — matched inside the "
                "tool, so approximate spelling and missing accents are fine. "
                "A matched album plays in its own order; an artist or a loose "
                "match is shuffled. Use append to add to what is already "
                "playing instead of replacing it, which is how a set is built "
                f"without a gap. Outputs: {output_list}. Leave output unset to "
                "stay wherever music is already playing."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Artist, album, or song title — or a handle (t:…, al:…, "
                            "ar:…) from music_search, which plays exactly that thing."
                        ),
                    },
                    "output": {
                        "type": "string",
                        "description": f"Which speaker. One of: {', '.join(o.id for o in cfg.outputs)}.",
                    },
                    "album": {
                        "type": "boolean",
                        "description": (
                            "Treat the query as an album name only. Use when the "
                            "request clearly names a record, so a song of the same "
                            "name cannot win instead."
                        ),
                    },
                    "shuffle": {
                        "type": "boolean",
                        "description": (
                            "Override the ordering. Omit unless the request asks "
                            "for it — the default already plays albums in order "
                            "and everything else shuffled."
                        ),
                    },
                    "append": {
                        "type": "boolean",
                        "description": "Add to the end of the queue instead of replacing it.",
                    },
                    "handles": {
                        "type": "array", "items": {"type": "string"},
                        "description": (
                            "A set you chose: track (t:…) or album (al:…) handles from "
                            "music_candidates or music_search, played exactly and in "
                            "this order. All or nothing — one unknown handle queues "
                            "nothing. Replaces query and filters."
                        ),
                    },
                    "genre": {
                        "type": "string",
                        "description": (
                            "Narrow by genre, matched loosely — 'prog' finds "
                            "'Progressive Rock'. Usable on its own, without a query."
                        ),
                    },
                    "mood": {
                        "type": "string",
                        "description": (
                            "Narrow by a curated mood word. Only tracks somebody "
                            "has annotated will match, so this finds less than the "
                            "library holds. Usable on its own."
                        ),
                    },
                    "energy_min": {"type": "integer", "description": "1 (still) to 5 (relentless)."},
                    "energy_max": {"type": "integer", "description": "1 (still) to 5 (relentless)."},
                },
            },
            run=_play,
        ),
        "music_curate": Tool(
            name="music_curate",
            description=(
                "Record what you know about music in the library that the files "
                "do not say: mood, energy, a corrected genre or title, a note, or "
                "that a record must never be shuffled. Annotations attach to a "
                "track, an album or an artist and are inherited downwards, so one "
                "row about an artist covers everything by them including records "
                "not in the library yet. They persist across re-scans, and are "
                "what makes mood requests work later — so annotate as you go "
                "rather than in a batch. Loudness and duration are measured from "
                "the file and cannot be set here."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Which track, album or artist — a name, or a handle from music_search.",
                    },
                    "scope": {
                        "type": "string", "enum": ["track", "album", "artist"],
                        "description": (
                            "Which level the fact belongs to. Defaults to whatever "
                            "the query named. Put it as high as it is true: an "
                            "artist-level note reaches records not bought yet, "
                            "where the same words on an album reach only that one."
                        ),
                    },
                    "mood": {
                        "type": "string",
                        "description": "Comma-separated words, e.g. 'mellow, warm'. Replaces any existing mood.",
                    },
                    "energy": {"type": "integer", "description": "1 (still) to 5 (relentless)."},
                    "genre": {"type": "string"},
                    "title": {"type": "string"},
                    "artist": {"type": "string"},
                    "album": {"type": "string"},
                    "year": {"type": "integer"},
                    "notes": {"type": "string"},
                    "never_shuffle": {
                        "type": "boolean",
                        "description": "Mark a record that must always play in its own order.",
                    },
                },
                "required": ["query"],
            },
            run=_curate,
        ),
        "music_search": Tool(
            name="music_search",
            description=(
                "Look up what the library holds. Read-only, and complete: nothing "
                "is sampled, skipped or quietly filtered — interludes, spoken-word "
                "tracks, recently played records and annotated ones all appear, "
                "labelled. Matches artist, album, song title, genre, and words in "
                "notes or moods. Results come back as artists, albums and tracks, "
                "each with a handle (ar:…, al:…, t:…) that music_play and "
                "music_curate accept to act on exactly that thing. Use it to check "
                "whether something is here before promising it, and to see what "
                "an artist's records are. Pass a handle as the query for the full "
                "detail of exactly that thing: every level's notes, untruncated, and "
                "for an album its tracks, for an artist their records. It is one person's collection: a miss "
                "here means it is not in the library. The only narrowing is what "
                "you pass, and the reply restates it; when a display cap is hit the "
                "reply gives the true total."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Words to look for — an artist, album, song, genre, or note.",
                    },
                    "scope": {
                        "type": "string", "enum": ["artist", "album", "track"],
                        "description": "Return only this kind of result. Omit to see all three.",
                    },
                    "genre": {"type": "string", "description": "Only search tracks whose genre contains this."},
                    "year_min": {"type": "integer"},
                    "year_max": {"type": "integer"},
                },
                "required": ["query"],
            },
            run=_search,
        ),
        "music_candidates": Tool(
            name="music_candidates",
            description=(
                "Offer a pool of tracks to build a set from — about "
                f"{cfg.candidates.pool_factor:g}x the time asked for (default "
                f"{cfg.candidates.default_minutes:g} min), spread across as many "
                "artists and records as the request allows, each with its duration "
                "and a handle. It FILTERS, on purpose, and counts what it left out: "
                f"tracks queued in the last {cfg.candidates.fresh_hours:g}h, album "
                "interludes"
                + (f", and the genres {', '.join(cfg.candidates.exclude_genres)}"
                   if cfg.candidates.exclude_genres else "")
                + ". Choose from the pool by taste — about the length asked for, in "
                "your order — then play it with music_play(handles=[...]). It is not "
                "a search: to find out whether something is in the library, use "
                "music_search, which hides nothing. Mood and energy only match what "
                "has been curated, which is little so far; genre, artist and years "
                "are reliable."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "genre": {"type": "string", "description": "One or more, comma-separated; matched loosely ('prog' finds 'Progressive Rock')."},
                    "artist": {"type": "string", "description": "One or more names or ar:… handles, comma-separated."},
                    "year_min": {"type": "integer"},
                    "year_max": {"type": "integer"},
                    "mood": {"type": "string", "description": "A curated mood word."},
                    "energy_min": {"type": "integer", "description": "1 (still) to 5 (relentless)."},
                    "energy_max": {"type": "integer", "description": "1 (still) to 5 (relentless)."},
                    "minutes": {"type": "number", "description": "How long the set will be. The pool is larger, to choose from."},
                    "limit": {"type": "integer", "description": "A track count instead of minutes."},
                },
            },
            run=_candidates,
        ),
        "music_history": Tool(
            name="music_history",
            description=(
                "Read the curation log: every annotation, correction, rename and "
                "merge — who made it, when, and what it replaced — plus files whose "
                "curation was lost to a rename or a reset. Given an artist, album or "
                "track (a name or a handle), shows that thing, what is inside it, and "
                "the levels above it; with no query, the most recent edits across the "
                "library. Read-only. Useful before annotating (has someone already "
                "described this record?), for seeing what was curated lately, and "
                "for putting back curation a rename dropped."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "An artist, album or track — a name or a handle. Omit for recent edits everywhere."},
                    "limit": {"type": "integer", "description": f"How many entries, newest first (default {HISTORY_DEFAULT}, at most {HISTORY_MAX})."},
                },
            },
            run=_history,
        ),
        "music_control": Tool(
            name="music_control",
            description=(
                "Transport control for what is already playing: pause, resume, "
                "skip to the next track, or stop and clear the queue. Stop is "
                "not a pause — the queue is gone afterwards."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": list(ACTIONS)},
                },
                "required": ["action"],
            },
            run=_control,
            data=_control_data,
        ),
        "music_status": Tool(
            name="music_status",
            description=(
                "What is playing right now, where, how far in, how much of the "
                "queue is left, and which records were queued recently. Worth "
                "checking before acting on a vague request — so a session "
                "already under way is not restarted, and so a set does not "
                "repeat what was just on."
            ),
            input_schema={"type": "object", "properties": {}},
            run=_status,
            data=_status_data,
        ),
        "music_outputs": Tool(
            name="music_outputs",
            description=(
                "List the speakers and whether each can be reached right now. "
                "A Bluetooth speaker that is merely disconnected is woken "
                "automatically on play; one that stays unreachable is switched "
                "off or out of range."
            ),
            input_schema={"type": "object", "properties": {}},
            run=_outputs,
        ),
    }
    if cfg.dj.enabled:
        tools["music_dj"] = Tool(
            name="music_dj",
            description=(
                "Play a set you have chosen, speaking between the songs. The set is "
                "a run sheet — a list of entries, each with `say` (a short spoken "
                "link), `play` (a handle from music_candidates or music_search), or "
                "both; `say` is spoken before what it plays. `play` is one track "
                "(t:…) or one whole record (al:…), which plays complete and in its "
                "own order at one gain, exactly as music_play plays an album — so "
                "`say` on it introduces the record. A `say` with no `play` at the "
                "start is an opening, at the end a sign-off. A link is its own "
                "entry in the queue, never spoken over the music. All or nothing: "
                "one bad handle, or speech that will not render, and nothing is "
                "queued. Pass `minutes` and the set's length is checked before "
                "anything is rendered. Talk sparingly; the music skill says how."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "set": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "say": {"type": "string", "description": "Spoken before what this entry plays — one or two sentences, plain words, no stage directions."},
                                "play": {"type": "string", "description": "A track handle (t:…), or a record handle (al:…) to play that whole album in order."},
                            },
                        },
                    },
                    "output": {
                        "type": "string",
                        "description": f"Which speaker. One of: {', '.join(o.id for o in cfg.outputs)}.",
                    },
                    "append": {
                        "type": "boolean",
                        "description": (
                            "Add to the end of what is playing. The last thing "
                            "queued is unknown to you unless music_status said so, "
                            "so open with an intro rather than a back-announce."
                        ),
                    },
                    "minutes": {"type": "number", "description": "The length you intend; checked before rendering."},
                },
                "required": ["set"],
            },
            run=_dj,
        )
    return tools
