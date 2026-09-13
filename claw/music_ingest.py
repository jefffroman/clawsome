"""Ingest: turn one MP3 into one catalogue row.

The unit is **a single file**. Collection ingest is a loop over it, not a
separate code path — so adding one album and rebuilding everything exercise the
same logic, and a bug cannot hide in the batch case.

Two operations, and the difference is the whole safety model:

- ``ingest_file`` — insert if new, otherwise refresh only what the bytes
  determine. It cannot touch curation, because :mod:`claw.music_db`'s conflict
  clause does not name those columns.
- ``ingest_file(..., repopulate=True)`` — delete and re-insert. Puts the row
  back to exactly what the file says and **discards curation deliberately**.
  This is "restore defaults for this track", and it is the only way a retag
  reaches an existing row.

Loudness is measured with ffmpeg's ``ebur128`` filter (EBU R128 integrated
loudness, in LUFS) rather than read from a ReplayGain tag, because the library
has essentially none — 1 file in 25 sampled — and because a measurement we made
is one we can re-make. Nothing is ever written back into the MP3s.
"""

from __future__ import annotations

import asyncio
import concurrent.futures as cf
import json
import logging
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from claw import music_db
from claw.music import AUDIO_EXTS, _display, _parse_track

log = logging.getLogger(__name__)

# Bump when what we measure changes; rows with an older value are re-measured
# on the next ingest even though their bytes have not moved.
MEASURE_VERSION = 1

FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"

_RE_I = re.compile(r"Integrated loudness:\s+I:\s+(-?[\d.]+) LUFS")
_RE_TP = re.compile(r"True peak:\s+Peak:\s+(-?[\d.]+) dBFS")
_RE_LRA = re.compile(r"Loudness range:\s+LRA:\s+(-?[\d.]+) LU")

# Things that live in a music tree and are not music. Reported by a sweep, never
# deleted — it is not our library.
CRUFT_NAMES = {".DS_Store", "Folder.jpg", "AlbumArtSmall.jpg", "desktop.ini"}
CRUFT_SUFFIXES = {".amz", ".sh", ".m3u", ".pls", ".log", ".cue", ".nfo"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def measure(path: Path, timeout: float = 300.0) -> dict[str, Any]:
    """Integrated loudness, true peak and loudness range for one file.

    Decodes the whole file — about 250x realtime on this hardware, so ~1s for a
    four-minute track. Returns ``None`` values rather than raising when ffmpeg
    cannot read the file, so one corrupt MP3 does not stop a sweep.

    ``errors="replace"`` is load-bearing, not defensive: ffmpeg echoes the file's
    ID3 text into its stderr, and a 2008-era rip carries latin-1 bytes there. A
    strict decode raises inside ``communicate`` — before any of our own error
    handling — and took out 13 of 4,532 files on the first real sweep.
    """
    try:
        proc = subprocess.run(
            [FFMPEG, "-nostats", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"],
            capture_output=True, text=True, errors="replace", timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        log.warning("music: could not measure %s: %s", path, exc)
        return {"lufs": None, "true_peak_dbfs": None, "lra_lu": None}
    # The summary is the last few hundred bytes; the rest is a per-frame log
    # that also contains "I: ... LUFS" lines, which is why this reads the tail.
    tail = proc.stderr[-2000:]
    grab = lambda rx: (float(m.group(1)) if (m := rx.search(tail)) else None)
    return {
        "lufs": grab(_RE_I),
        "true_peak_dbfs": grab(_RE_TP),
        "lra_lu": grab(_RE_LRA),
    }


def probe(path: Path, timeout: float = 60.0) -> dict[str, Any]:
    """Duration and ID3 tags, via ffprobe."""
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "quiet", "-show_entries", "format=duration:format_tags",
             "-of", "json", str(path)],
            capture_output=True, text=True, errors="replace", timeout=timeout,
        ).stdout
        fmt = json.loads(out or "{}").get("format", {})
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError, json.JSONDecodeError):
        return {"duration_s": None, "tags": {}}
    return {
        "duration_s": float(fmt["duration"]) if fmt.get("duration") else None,
        "tags": {k.lower(): v for k, v in (fmt.get("tags") or {}).items()},
    }


def _int(value: Any) -> int | None:
    """ID3 numbers arrive as '3', '3/12', '1999-04-01' or nonsense."""
    if value is None:
        return None
    m = re.match(r"\s*(\d{1,4})", str(value))
    return int(m.group(1)) if m else None


def build_row(root: Path, path: Path) -> dict[str, Any]:
    """Everything a catalogue row needs, read from one file.

    Artist and album come from the **tree**, not from ID3: the directory layout
    is the convention that is actually reliable here (album_artist is present on
    14% of this library), and it is where the percent-encoded names get decoded.
    Title prefers the tag and falls back to the filename; the track number
    prefers the filename, because ID3 renders it as '3/12' as often as '3'.
    """
    st = path.stat()
    from_path = _parse_track(root, path)
    pr = probe(path)
    tags = pr["tags"]
    return {
        "path": str(path.relative_to(root)),
        "artist": from_path.artist,
        "album": from_path.album,
        "title": _display(tags.get("title") or "").strip() or from_path.title,
        "track_no": from_path.number if from_path.number is not None else _int(tags.get("track")),
        "disc_no": _int(tags.get("disc")),
        "genre": (tags.get("genre") or "").strip() or None,
        "year": _int(tags.get("date") or tags.get("year")),
        "duration_s": pr["duration_s"],
        **measure(path),
        "size": st.st_size,
        "mtime": int(st.st_mtime),
        "measured_at": _now(),
        "measure_version": MEASURE_VERSION,
    }


def ingest_file(
    con,
    root: Path,
    path: Path,
    *,
    repopulate: bool = False,
    known: dict[str, tuple[int, int, int]] | None = None,
    by: str = "ingest",
) -> str:
    """Ingest one file. Returns what happened.

    ``new`` | ``remeasured`` | ``repopulated`` | ``unchanged`` | ``skipped``.

    A file whose size, mtime and measure version all match what is stored is not
    opened at all — that is what makes a sweep over 4,500 files cheap once the
    first one has run.
    """
    if path.suffix.lower() not in AUDIO_EXTS or not path.is_file():
        return "skipped"
    rel = str(path.relative_to(root))
    if known is None:
        known = music_db.known_paths(con)
    prior = known.get(rel)
    if prior is not None and not repopulate:
        st = path.stat()
        if prior == (st.st_size, int(st.st_mtime), MEASURE_VERSION):
            return "unchanged"

    row = build_row(root, path)
    if repopulate and prior is not None:
        music_db.repopulate(con, row, by=by)
        return "repopulated"
    music_db.upsert(con, row)
    return "new" if prior is None else "remeasured"


def sweep_cruft(root: Path) -> dict[str, list[str]]:
    """What is in the tree and is not music. Reported, never touched."""
    found: dict[str, list[str]] = {"percent_encoded": [], "junk": []}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        if "%" in rel and p.suffix.lower() in AUDIO_EXTS:
            found["percent_encoded"].append(rel)
        elif p.name in CRUFT_NAMES or p.suffix.lower() in CRUFT_SUFFIXES:
            found["junk"].append(rel)
    return found


def ingest_tree(
    con,
    root: Path,
    *,
    repopulate: bool = False,
    workers: int = 8,
    prune: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    by: str = "ingest",
) -> dict[str, int]:
    """Loop ``ingest_file`` over the tree. Nothing here that one file does not do.

    Measurement runs on a thread pool because it is entirely ffmpeg wall-clock;
    the writes stay on this thread, since SQLite would rather have one writer
    than a lock contest with seven of its own.
    """
    known = music_db.known_paths(con)
    files = [p for p in sorted(root.rglob("*"))
             if p.suffix.lower() in AUDIO_EXTS and p.is_file()]
    counts = {"new": 0, "remeasured": 0, "repopulated": 0, "unchanged": 0, "failed": 0}

    # Cheap pass first: anything whose bytes have not moved never reaches ffmpeg.
    todo: list[Path] = []
    for p in files:
        rel = str(p.relative_to(root))
        prior = known.get(rel)
        if prior is not None and not repopulate:
            st = p.stat()
            if prior == (st.st_size, int(st.st_mtime), MEASURE_VERSION):
                counts["unchanged"] += 1
                continue
        todo.append(p)

    done = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        for path, result in zip(todo, ex.map(lambda p: _safe_row(root, p), todo)):
            done += 1
            if result is None:
                counts["failed"] += 1
                continue
            rel = result["path"]
            if repopulate and rel in known:
                music_db.repopulate(con, result, by=by)
                counts["repopulated"] += 1
            else:
                music_db.upsert(con, result)
                counts["new" if rel not in known else "remeasured"] += 1
            if on_progress and done % 50 == 0:
                on_progress(done, len(todo))

    if prune:
        live = {str(p.relative_to(root)) for p in files}
        gone = [p for p in known if p not in live]
        if gone:
            music_db.forget(con, gone, by=by)
        counts["forgotten"] = len(gone)
    return counts


def _safe_row(root: Path, path: Path) -> dict[str, Any] | None:
    try:
        return build_row(root, path)
    except Exception:                       # one bad file must not stop a sweep
        log.exception("music: ingest failed for %s", path)
        return None
