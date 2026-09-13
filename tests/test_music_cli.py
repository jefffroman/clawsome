"""The music CLI's dispatch.

There was no coverage here, and the gap shipped a CLI that raised
AttributeError on every `curate` while the agent's identical tool worked
fine — the include list naming the curatable columns was not updated when
curation split into per-scope tuples.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from claw import music_cli
from claw.config import BluetoothConfig, MusicConfig, MusicOutput


@dataclass
class _Stub:
    music: MusicConfig
    bluetooth: BluetoothConfig


class _Recorder:
    def __init__(self) -> None:
        self.got: dict | None = None

    async def run(self, args: dict) -> str:
        self.got = args
        return "ok"


@pytest.fixture
def curate(monkeypatch):
    """Run a CLI `curate ...` command line; return what the tool got."""
    rec = _Recorder()
    cfg = _Stub(
        music=MusicConfig(
            enabled=True,
            outputs=(MusicOutput(id="room", name="Room", mpv_device="dev"),),
        ),
        bluetooth=BluetoothConfig(),
    )
    monkeypatch.setattr(music_cli.config_mod, "load", lambda p: cfg)
    monkeypatch.setattr(music_cli, "build_music_tools",
                        lambda *a, **k: {"music_curate": rec})

    def run(argv: list[str]) -> dict:
        args = music_cli._parser().parse_args(["--config", "/nowhere.yaml", *argv])
        assert asyncio.run(music_cli._run(args)) == "ok"
        return rec.got

    return run


def test_curate_reaches_the_tool_at_all(curate):
    # The regression: this raised AttributeError before the tool was ever called.
    got = curate(["curate", "some", "artist", "--scope", "artist",
                  "--notes", "a note"])
    assert got["query"] == "some artist"
    assert got["scope"] == "artist"
    assert got["notes"] == "a note"


def test_every_annotation_field_is_forwarded(curate):
    got = curate([
        "curate", "a record", "--scope", "album",
        "--mood", "mellow, warm", "--energy", "3", "--genre", "Dub",
        "--year", "1977", "--notes", "n", "--title", "t",
        "--artist", "someone", "--album-name", "renamed", "--never-shuffle",
    ])
    assert got["mood"] == "mellow, warm"
    assert got["energy"] == 3
    assert got["genre"] == "Dub"
    assert got["year"] == 1977
    assert got["artist"] == "someone"
    assert got["album"] == "renamed"
    assert got["never_shuffle"] is True


def test_unset_fields_are_not_forwarded(curate):
    # The tool refuses when nothing was set, and reports what a scope accepts.
    # A CLI that forwarded None for every flag would defeat that.
    got = curate(["curate", "a track", "--mood", "warm"])
    assert set(got) == {"query", "scope", "mood"}
    assert got["scope"] is None


def test_the_parsers_own_arguments_are_not_mistaken_for_fields(curate):
    got = curate(["curate", "a track", "--energy", "5"])
    assert "verb" not in got and "config" not in got


# --- stats: where target_lufs is re-derived -----------------------------

def _rows():
    # A hump at -13, a quiet dynamic tail peaking near full scale, a loud tail,
    # and one silent rip at the gate floor that must not count.
    return ([(-13.0, -0.5)] * 40 + [(-12.5, 0.4)] * 20 + [(-13.5, -0.2)] * 20
            + [(-21.0, -3.0)] * 10 + [(-5.0, 1.5)] * 10 + [(-70.0, -60.0)])


def test_stats_reports_the_mode_beside_the_configured_target():
    from claw.config import LoudnessConfig
    out = music_cli.loudness_report(_rows(), LoudnessConfig(target_lufs=-13.0))
    assert "100 tracks measured" in out                     # the silent rip is excluded
    assert "mode        -13.0 LUFS" in out
    assert "differs from the mode" not in out


def test_stats_flags_a_target_that_has_drifted_from_the_mode():
    from claw.config import LoudnessConfig
    out = music_cli.loudness_report(_rows(), LoudnessConfig(target_lufs=-18.0))
    assert "differs from the mode" in out


def test_stats_shows_what_the_policy_would_do_to_a_shuffle():
    from claw.config import LoudnessConfig
    out = music_cli.loudness_report(_rows(), LoudnessConfig(target_lufs=-13.0))
    line = next(l for l in out.splitlines() if "shuffle spread" in l)
    unity, normalised = (float(x.split()[0]) for x in line.split("unity ")[1].split("-> normalised "))
    assert normalised < unity


def test_stats_with_nothing_measured_says_so():
    from claw.config import LoudnessConfig
    assert "nothing measured" in music_cli.loudness_report([], LoudnessConfig())


# --- search / candidates dispatch ---------------------------------------

@pytest.fixture
def verb(monkeypatch):
    """Run any CLI `<verb> ...` line; return (tool name, args it got)."""
    recs: dict[str, _Recorder] = {}

    class _Tools(dict):
        def __getitem__(self, name):
            return recs.setdefault(name, _Recorder())

    cfg = _Stub(music=MusicConfig(enabled=True, outputs=(MusicOutput(id="room", name="Room",
                                                                        mpv_device="dev"),)),
                bluetooth=BluetoothConfig())
    monkeypatch.setattr(music_cli.config_mod, "load", lambda p: cfg)
    monkeypatch.setattr(music_cli, "build_music_tools", lambda *a, **k: _Tools())

    def run(argv: list[str]) -> tuple[str, dict]:
        args = music_cli._parser().parse_args(["--config", "/nowhere.yaml", *argv])
        assert asyncio.run(music_cli._run(args)) == "ok"
        (name, rec), = recs.items()
        return name, rec.got

    return run


def test_search_reaches_music_search_with_its_narrowing(verb):
    name, got = verb(["search", "burning", "spear", "--scope", "album", "--years", "1975-1985"])
    assert name == "music_search"
    assert got == {"query": "burning spear", "scope": "album", "genre": None,
                   "year_min": 1975, "year_max": 1985}


def test_a_single_year_is_a_one_year_range(verb):
    _, got = verb(["search", "x", "--years", "1977"])
    assert (got["year_min"], got["year_max"]) == (1977, 1977)


def test_candidates_reaches_music_candidates(verb):
    name, got = verb(["candidates", "--genre", "reggae,dub", "--minutes", "90"])
    assert name == "music_candidates"
    assert got == {"genre": "reggae,dub", "artist": None, "minutes": 90.0, "limit": None,
                   "year_min": None, "year_max": None}


def test_history_reaches_music_history(verb):
    name, got = verb(["history", "burning", "spear", "--limit", "5"])
    assert name == "music_history" and got == {"query": "burning spear", "limit": 5}
