"""The music CLI — the operator's front end to the same tools the agent gets.

Deliberately built on :func:`claw.tools.music.build_music_tools` rather than on
:class:`claw.music.Player` directly. A CLI that reimplemented the tool layer
would drift from it, and the drift would only ever be discovered by a human
being told something different from what the agent was told about the same
speaker. Here the operator sees the tool's own words — including its refusals,
which is the case worth being able to reproduce by hand.

Playback verbs are separate from the service verbs (``start``/``stop``/
``restart``/``status``), which a service wrapper around this CLI keeps for
itself: a service helper conventionally answers ``status`` with service health,
and monitoring depends on that. So stopping playback is ``quiet`` and asking
what is on is ``now``.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import sys
from pathlib import Path

from claw import config as config_mod, music_db, music_ingest
from claw.tools.music import ACTIONS, build_music_tools


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="claw-music", description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, type=Path, help="path to claw.yaml")
    sub = p.add_subparsers(dest="verb", required=True)

    play = sub.add_parser("play", help="play an artist, album, or song")
    play.add_argument("query", nargs="+")
    play.add_argument("-o", "--output", help="which speaker")
    play.add_argument("--album", action="store_true", help="treat the query as an album name only")
    order = play.add_mutually_exclusive_group()
    order.add_argument("--shuffle", dest="shuffle", action="store_true", default=None)
    order.add_argument("--in-order", dest="shuffle", action="store_false", default=None)
    play.add_argument("--append", action="store_true", help="add to the queue instead of replacing it")

    cur = sub.add_parser("curate", help="annotate a track or album")
    cur.add_argument("query", nargs="+")
    cur.add_argument("--scope", choices=["track", "album", "artist"],
                 help="which level the fact belongs to; defaults to what the query named")
    cur.add_argument("--mood", help="comma-separated words, e.g. 'mellow, warm'")
    cur.add_argument("--energy", type=int, help="1 (still) to 5 (relentless)")
    cur.add_argument("--genre")
    cur.add_argument("--title")
    cur.add_argument("--artist")
    cur.add_argument("--album-name", dest="album")
    cur.add_argument("--year", type=int)
    cur.add_argument("--notes")
    cur.add_argument("--never-shuffle", dest="never_shuffle", action="store_true", default=None)

    srch = sub.add_parser("search", help="look up what the library holds — read-only, nothing hidden")
    srch.add_argument("query", nargs="+")
    srch.add_argument("--scope", choices=["artist", "album", "track"],
                      help="report every result as this kind — never changes what matches")
    srch.add_argument("--genre")
    srch.add_argument("--years", help="a range, e.g. 1970-1979, or one year")

    hist = sub.add_parser("history", help="the curation log — for one thing, or recent edits everywhere")
    hist.add_argument("query", nargs="*")
    hist.add_argument("--limit", type=int)

    cand = sub.add_parser("candidates", help="a pool to build a set from — filtered, and says what it left out")
    cand.add_argument("--genre", help="one or more, comma-separated")
    cand.add_argument("--artist", help="one or more, comma-separated")
    cand.add_argument("--years", help="a range, e.g. 1970-1979, or one year")
    cand.add_argument("--minutes", type=float)
    cand.add_argument("--limit", type=int)

    ing = sub.add_parser("ingest", help="scan files into the catalogue")
    ing.add_argument("paths", nargs="*", type=Path,
                     help="files to ingest; omit to sweep the whole library")
    ing.add_argument("--repopulate", action="store_true",
                     help="DISCARD curation for these tracks and reread the files")
    ing.add_argument("--workers", type=int, default=8)

    for verb, helptext in (
        ("stats", "what the catalogue holds"),
        ("pause", "pause playback"),
        ("resume", "resume playback"),
        ("next", "skip to the next track"),
        ("quiet", "stop playback and clear the queue"),
        ("now", "what is playing, where, and how far in"),
        ("outputs", "list the speakers and whether each can be reached"),
    ):
        sub.add_parser(verb, help=helptext)
    return p


# The curate parser's own arguments, as opposed to the annotation fields it
# collects. Anything else is offered to the tool, which knows what each scope
# accepts and names what it rejected.
_NOT_A_FIELD = frozenset({"verb", "config", "query", "scope"})


async def _run(args: argparse.Namespace) -> str:
    cfg = config_mod.load(args.config)
    if not cfg.music.enabled:
        return "error: music is not enabled in this claw.yaml"
    tools = build_music_tools(cfg.music, cfg.bluetooth.binary, _who(), getattr(cfg, "tz", None))
    if args.verb == "play":
        return await tools["music_play"].run({
            "query": " ".join(args.query),
            "output": args.output,
            "album": args.album,
            "shuffle": args.shuffle,
            "append": args.append,
        })
    if args.verb == "history":
        return await tools["music_history"].run(
            {"query": " ".join(args.query) or None, "limit": args.limit})
    if args.verb in ("search", "candidates"):
        lo = hi = None
        if args.years:
            lo, _, hi = args.years.partition("-")
            hi = hi or lo
        years = {"year_min": int(lo) if lo else None, "year_max": int(hi) if hi else None}
        if args.verb == "search":
            return await tools["music_search"].run({
                "query": " ".join(args.query), "scope": args.scope, "genre": args.genre,
                **years,
            })
        return await tools["music_candidates"].run({
            "genre": args.genre, "artist": args.artist, "minutes": args.minutes,
            "limit": args.limit, **years,
        })
    if args.verb == "curate":
        # Pass everything the user set; the tool decides what the chosen scope
        # accepts and says so, rather than the CLI silently dropping it.
        #
        # Expressed as an EXCLUDE list on purpose. It was an include list
        # against `music_db.CURATABLE`, and when curation split into per-scope
        # tuples that name stopped existing — so every `curate` from the CLI
        # raised AttributeError, while the agent's tool went on working. An
        # include list has to be updated in step with the schema and silently
        # was not; excluding the four arguments that are *not* fields cannot
        # drift, because they are this parser's own.
        fields = {k: v for k, v in vars(args).items()
                  if k not in _NOT_A_FIELD and v is not None}
        return await tools["music_curate"].run(
            {"query": " ".join(args.query), "scope": args.scope, **fields}
        )
    if args.verb == "ingest":
        return _ingest(cfg, args)
    if args.verb == "stats":
        con = music_db.connect(cfg.music.db_path)
        try:
            st = music_db.stats(con)
            measured = music_db.loudness_rows(con)
        finally:
            con.close()
        width = max(len(k) for k in st) + 2
        counts = "\n".join(f"{k:<{width}}{v}" for k, v in st.items())
        return f"{counts}\n\n{loudness_report(measured, cfg.music.loudness)}"
    if args.verb == "now":
        return await tools["music_status"].run({})
    if args.verb == "outputs":
        return await tools["music_outputs"].run({})
    action = "stop" if args.verb == "quiet" else args.verb
    assert action in ACTIONS
    return await tools["music_control"].run({"action": action})


def loudness_report(rows: list[tuple[float, float | None]], ld) -> str:
    """The collection's measured loudness beside what config says to do with it.

    This is where ``music.loudness.target_lufs`` is re-derived: the mode is
    printed next to the configured target, so a collection that has drifted
    shows it. The spread is the 5th-95th percentile of what a shuffle would
    actually play — at unity, and with the configured policy applied per track.
    """
    from claw.music import gain_db, loudness_mode

    rows = [(lu, tp) for lu, tp in rows if lu > -69.0]    # -70 is the gate floor: silence
    if not rows:
        return "loudness      nothing measured yet — run the CLI's `ingest`"

    def pct(xs: list[float], p: float) -> float:
        xs = sorted(xs)
        return xs[int(p * (len(xs) - 1))]

    lufs = [lu for lu, _ in rows]
    peaks = [tp for _, tp in rows if tp is not None]
    played = [lu + gain_db(lu, tp, ld) for lu, tp in rows]
    mode = loudness_mode(lufs)
    drift = "" if abs(mode - ld.target_lufs) < 1.0 else "   <- differs from the mode"
    over = sum(p > 0 for p in peaks) / len(peaks) if peaks else 0.0
    return "\n".join([
        f"loudness      {len(lufs)} tracks measured",
        f"  mode        {mode:.1f} LUFS   (where target_lufs belongs)",
        f"  median      {pct(lufs, .5):.1f} LUFS",
        f"  p05..p95    {pct(lufs, .05):.1f} .. {pct(lufs, .95):.1f} LUFS",
        f"  peak > 0    {over:.0%} of tracks already exceed full scale",
        f"configured    normalize {'on' if ld.normalize else 'off'}, "
        f"target {ld.target_lufs:.1f} LUFS{drift}, boost ceiling {ld.boost_ceiling_dbtp:.1f} dBTP",
        f"  shuffle spread (p05..p95)  unity {pct(lufs, .95) - pct(lufs, .05):.1f} LU"
        f" -> normalised {pct(played, .95) - pct(played, .05):.1f} LU",
    ])


def _who() -> str:
    """Whoever actually typed the command, not the account it runs under.

    The wrapper reaches the socket via ``sudo -u`` the service user, so the
    process identity is always the same and would make every hand edit look
    like the daemon's.
    """
    return os.environ.get("SUDO_USER") or getpass.getuser()


def _ingest(cfg, args) -> str:
    """Scan files into the catalogue. One file, or a sweep of the whole tree."""
    root = cfg.music.library_root
    con = music_db.connect(cfg.music.db_path)
    try:
        if args.paths:
            out = []
            for p in args.paths:
                p = p if p.is_absolute() else (root / p)
                out.append(f"{music_ingest.ingest_file(con, root, p, repopulate=args.repopulate, by=_who()):<12}{p.name}")
            return "\n".join(out)

        seen = {"n": 0}
        def progress(done, total):
            if done != seen["n"]:
                seen["n"] = done
                print(f"  {done}/{total}", end="\r", file=sys.stderr, flush=True)

        counts = music_ingest.ingest_tree(
            con, root, repopulate=args.repopulate, workers=args.workers,
            on_progress=progress, by=_who(),
        )
        print(" " * 30, end="\r", file=sys.stderr)
        lines = ["  ".join(f"{k} {v}" for k, v in counts.items() if v)]
        cruft = music_ingest.sweep_cruft(root)
        for label, items in cruft.items():
            if items:
                lines.append(f"\nnoticed, not touched — {label} ({len(items)}):")
                lines += [f"  {i}" for i in items[:8]]
                if len(items) > 8:
                    lines.append(f"  ... and {len(items) - 8} more")
        return "\n".join(lines)
    finally:
        con.close()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out = asyncio.run(_run(args))
    print(out)
    # Refusals and errors are ordinary outcomes with a message, not tracebacks —
    # but a script calling this needs to be able to tell them apart.
    return 1 if out.startswith(("error:", "refused:")) else 0


if __name__ == "__main__":
    sys.exit(main())
