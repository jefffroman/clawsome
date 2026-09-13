"""music: library matching, the reconnect dance, and the tool surface.

PUBLIC MIRROR: neutral placeholders only — no real device, address, path, or
deployment agent name.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from claw import music_db
from dataclasses import replace

from claw.config import LoudnessConfig, MusicConfig, MusicOutput
from claw.music import (
    Library, MpvUnavailable, Player, Track, _fold, _parse_track,
    album_gain_db, gain_db, loudness_mode,
)
import claw.music as music_mod
import claw.tools.music as tools_mod
from claw.tools.music import build_music_tools

BT = MusicOutput(
    id="room-a", name="Room A", mpv_device="coreaudio/AA-BB:output",
    coreaudio_name="Room A Adapter", bluetooth_address="aa-bb-cc-dd-ee-ff",
    default=True,
)
WIRED = MusicOutput(
    id="room-b", name="Room B", mpv_device="coreaudio/Internal",
    coreaudio_name="Internal Speakers",
)


TREE = [
    # (relative path, genre, year, lufs)
    ("Motörhead/Ace of Spades/01 - Ace of Spades.mp3", "Metal", 1980, -12.0),
    ("Motörhead/Ace of Spades/02 - Love Me Like a Reptile.mp3", "Metal", 1980, -12.5),
    ("Motörhead/Ace of Spades/10 - The Chase Is Better Than the Catch.mp3", "Metal", 1980, -11.0),
    ("Motörhead/Overkill/01 - Overkill.mp3", "Metal", 1979, -13.0),
    ("Guns N' Roses/Appetite for Destruction/01 - Welcome to the Jungle.mp3", "Rock", 1987, -10.0),
    ("AC%2fDC/Back in Black/01 - Hells Bells.mp3", "Rock", 1980, -9.0),
    ("AC%2fDC/Back in Black/02 - Shoot to Thrill.mp3", "Rock", 1980, -8.5),
    ("loose-single.mp3", None, None, -20.0),
]


@pytest.fixture
def library_root(tmp_path: Path) -> Path:
    """The files on disk. Only their paths matter here — nothing decodes them."""
    root = tmp_path / "audio"
    for rel, *_ in TREE:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    (root / "Motörhead/Ace of Spades/cover.jpg").write_bytes(b"")
    return root


@pytest.fixture
def catalog(tmp_path: Path, library_root: Path) -> Path:
    """A populated catalogue, written the way ingest would write it.

    Rows go in through ``music_db.upsert`` rather than through ingest itself, so
    these tests exercise the query and playback layers without decoding audio.
    Ingest has its own tests.
    """
    db = tmp_path / "library.db"
    con = music_db.connect(db)
    for rel, genre, year, lufs in TREE:
        parsed = _parse_track(library_root, library_root / rel)
        music_db.upsert(con, {
            "path": rel, "artist": parsed.artist, "album": parsed.album,
            "title": parsed.title, "track_no": parsed.number, "disc_no": None,
            "genre": genre, "year": year, "duration_s": 240.0, "lufs": lufs,
            "true_peak_dbfs": -1.0, "lra_lu": 5.0, "size": 1, "mtime": 1,
            "measured_at": "2026-09-09T00:00:00+00:00", "measure_version": 1,
        })
    con.close()
    return db


@pytest.fixture
def lib(catalog: Path) -> Library:
    return Library(catalog)


# --- folding and parsing -------------------------------------------------

@pytest.mark.parametrize("raw,folded", [
    ("Motörhead", "motorhead"),
    ("Guns N' Roses", "guns n roses"),
    ("  AC/DC  ", "ac dc"),
])
def test_folding_matches_how_a_person_types_a_name(raw, folded):
    assert _fold(raw) == folded


def test_track_number_prefix_is_not_part_of_the_title(tmp_path):
    root = tmp_path
    t = _parse_track(root, root / "Artist" / "Album" / "07 - Some Song.mp3")
    assert (t.artist, t.album, t.title, t.number) == ("Artist", "Album", "Some Song", 7)


def test_percent_encoding_is_undone_for_display_but_not_on_disk(tmp_path):
    path = tmp_path / "AC%2fDC" / "Back in Black" / "01 - Hells Bells.mp3"
    t = _parse_track(tmp_path, path)
    assert t.artist == "AC/DC"
    assert t.path == path          # the real name is what has to be opened


def test_a_file_at_the_root_still_indexes_as_a_track(tmp_path):
    t = _parse_track(tmp_path, tmp_path / "loose-single.mp3")
    assert (t.artist, t.album, t.title) == ("", "", "loose-single")


def test_non_audio_files_are_not_indexed(lib, library_root):
    # The catalogue holds what ingest put there; a cover image never becomes a row.
    titles = {t.title for t in lib.tracks(library_root)}
    assert "cover" not in titles


# --- matching ------------------------------------------------------------

def test_an_album_matches_as_an_album_and_keeps_its_own_order(lib, library_root):
    tracks, kind, label = lib.resolve(library_root, "ace of spades")
    assert kind == "album"
    assert [t.number for t in tracks] == [1, 2, 10]     # not lexical: 10 last
    assert "Ace of Spades" in label


def test_an_artist_matches_across_albums(lib, library_root):
    tracks, kind, label = lib.resolve(library_root, "motorhead")
    assert kind == "artist"
    assert label == "Motörhead"
    assert len({t.album for t in tracks}) == 2


def test_an_accented_or_punctuated_name_is_found_as_typed(lib, library_root):
    _, kind, label = lib.resolve(library_root, "guns n roses")
    assert (kind, label) == ("artist", "Guns N' Roses")


def test_a_percent_encoded_directory_is_findable_by_its_real_name(lib, library_root):
    _, kind, label = lib.resolve(library_root, "ac/dc")
    assert (kind, label) == ("artist", "AC/DC")


def test_a_song_title_matches_when_nothing_larger_does(lib, library_root):
    tracks, kind, _ = lib.resolve(library_root, "welcome to the jungle")
    assert kind == "track"
    assert tracks[0].title == "Welcome to the Jungle"


def test_an_explicit_album_request_that_misses_does_not_fall_back_to_a_song(lib, library_root):
    # "play the album Overkill" must not quietly play the song Overkill from
    # some other record.
    tracks, kind, _ = lib.resolve(library_root, "welcome to the jungle", album=True)
    assert (tracks, kind) == ([], "none")


def test_nothing_matches_nothing(lib, library_root):
    assert lib.resolve(library_root, "zzzz nonexistent zzzz")[1] == "none"


def test_stats_describe_the_collection(lib, library_root):
    total, artists, albums = lib.stats(library_root)
    assert (total, artists, albums) == (8, 3, 4)


# --- the reconnect dance -------------------------------------------------

class FakePlayer(Player):
    """Player with the two things that touch the world replaced."""

    def __init__(self, cfg, *, devices, bt_table, default_name="Internal Speakers"):
        super().__init__(cfg, Path("/nonexistent/blueutil"))
        self.devices = set(devices)
        self.bt_table = bt_table
        self.bt_calls: list[list[str]] = []
        self.default_name = default_name
        self.default_sets: list[str] = []

    async def device_ids(self):
        return set(self.devices)

    async def _bt(self, args, timeout):
        self.bt_calls.append(list(args))
        return self.bt_table[args[0]]

    async def _default_output_name(self):
        return self.default_name

    async def _set_default_output(self, name):
        self.default_sets.append(name)


@pytest.fixture
def cfg(tmp_path, catalog):
    return MusicConfig(
        enabled=True, exposed_to=("example",), library_root=tmp_path / "audio",
        mpv_socket=tmp_path / "mpv.sock", db_path=catalog, connect_timeout_s=1.0,
        outputs=(BT, WIRED),
    )


@pytest.fixture
def ipc_calls(monkeypatch):
    calls: list[list] = []

    async def fake_ipc(sock, commands, timeout=10.0):
        calls.extend([list(c) for c in commands])
        return [{"error": "success", "data": None} for _ in commands]

    monkeypatch.setattr(music_mod, "ipc", fake_ipc)
    return calls


async def test_a_connected_bluetooth_output_is_not_reconnected(cfg, ipc_calls):
    p = FakePlayer(cfg, devices={BT.mpv_device}, bt_table={"--is-connected": (0, "1", "")})
    assert await p.ensure_output(BT) is None
    assert p.bt_calls == [["--is-connected", BT.bluetooth_address]]
    assert p.default_sets == []          # nothing was disturbed
    assert ["set_property", "audio-device", BT.mpv_device] in ipc_calls


async def test_connecting_restores_the_default_output_it_stole(cfg, ipc_calls):
    # macOS promotes a Bluetooth device to system default the moment it
    # connects. Without the restore, every reconnect silently re-points system
    # audio — alert sounds included — at this speaker.
    p = FakePlayer(cfg, devices={BT.mpv_device}, bt_table={
        "--is-connected": (0, "0", ""), "--connect": (0, "", ""),
    })
    assert await p.ensure_output(BT) is None
    assert p.default_sets == ["Internal Speakers"]


async def test_the_default_is_left_alone_when_it_was_already_this_speaker(cfg, ipc_calls):
    p = FakePlayer(cfg, devices={BT.mpv_device}, default_name=BT.coreaudio_name, bt_table={
        "--is-connected": (0, "0", ""), "--connect": (0, "", ""),
    })
    assert await p.ensure_output(BT) is None
    assert p.default_sets == []


async def test_a_speaker_that_will_not_answer_is_refused_by_name(cfg, ipc_calls):
    p = FakePlayer(cfg, devices=set(), bt_table={
        "--is-connected": (0, "0", ""), "--connect": (1, "", "no connection"),
    })
    refusal = await p.ensure_output(BT)
    assert refusal is not None and "Room A" in refusal
    # Nothing was routed: a refusal must not leave mpv pointed somewhere new.
    assert ["set_property", "audio-device", BT.mpv_device] not in ipc_calls


async def test_a_connect_that_never_becomes_an_audio_device_is_refused(cfg, ipc_calls):
    p = FakePlayer(cfg, devices=set(), bt_table={
        "--is-connected": (0, "0", ""), "--connect": (0, "", ""),
    })
    refusal = await p.ensure_output(BT)
    assert refusal is not None and "never appeared" in refusal


async def test_a_wired_output_needs_no_radio_at_all(cfg, ipc_calls):
    p = FakePlayer(cfg, devices={WIRED.mpv_device}, bt_table={})
    assert await p.ensure_output(WIRED) is None
    assert p.bt_calls == []


async def test_a_missing_wired_output_is_reported_not_guessed(cfg, ipc_calls):
    p = FakePlayer(cfg, devices=set(), bt_table={})
    refusal = await p.ensure_output(WIRED)
    assert refusal is not None and "Room B" in refusal


# --- tools ---------------------------------------------------------------

@pytest.fixture
def tools(cfg, library_root, monkeypatch, ipc_calls):
    cfg = MusicConfig(**{**cfg.__dict__, "library_root": library_root})
    return build_music_tools(cfg, Path("/nonexistent/blueutil"), "example"), cfg


async def test_the_family_is_exactly_these_tools(tools):
    built, _ = tools
    assert set(built) == {
        "music_play", "music_control", "music_status", "music_outputs", "music_curate",
        "music_search", "music_candidates", "music_history",
    }


async def test_tool_descriptions_name_the_configured_outputs(tools):
    built, _ = tools
    desc = built["music_play"].description
    assert "'room-a'" in desc and "'room-b'" in desc


async def test_an_unknown_output_is_refused_with_the_real_list(tools):
    built, _ = tools
    out = await built["music_play"].run({"query": "overkill", "output": "kitchen"})
    assert out.startswith("error:") and "room-a" in out


async def test_an_empty_query_is_refused(tools):
    built, _ = tools
    assert (await built["music_play"].run({"query": "  "})).startswith("error:")


async def test_a_miss_is_reported_as_a_miss_not_a_near_guess(tools):
    built, _ = tools
    out = await built["music_play"].run({"query": "zzzz nothing zzzz", "output": "room-b"})
    assert "nothing in the library matches" in out
    assert "8 tracks by 3 artists" in out


async def test_control_rejects_an_unknown_action(tools):
    built, _ = tools
    assert (await built["music_control"].run({"action": "rewind"})).startswith("error:")


async def test_a_player_that_is_down_says_so_rather_than_failing_the_turn(tools, monkeypatch):
    built, _ = tools

    async def dead(*a, **kw):
        raise MpvUnavailable("no socket")

    monkeypatch.setattr(music_mod, "ipc", dead)
    assert "not running" in await built["music_status"].run({})
    assert "not running" in await built["music_control"].run({"action": "pause"})


@pytest.mark.parametrize("stem,number,title", [
    ("01 - Ace of Spades", 1, "Ace of Spades"),
    ("10 - The Chase", 10, "The Chase"),          # not disc 1, track 0
    ("02. Love Me", 2, "Love Me"),
    ("1-05 Some Song", 5, "Some Song"),
    ("Untitled", None, "Untitled"),
])
def test_track_numbers_parse_the_unglamorous_way(tmp_path, stem, number, title):
    t = _parse_track(tmp_path, tmp_path / "A" / "B" / f"{stem}.mp3")
    assert (t.number, t.title) == (number, title)


@pytest.fixture
def playing(cfg, library_root, monkeypatch, ipc_calls):
    """Tools built around a Player whose world is faked."""
    cfg = MusicConfig(**{**cfg.__dict__, "library_root": library_root})
    made: dict = {}

    def factory(c, blueutil):
        made["player"] = FakePlayer(
            c, devices={BT.mpv_device, WIRED.mpv_device},
            bt_table={"--is-connected": (0, "1", "")},
        )
        return made["player"]

    monkeypatch.setattr("claw.tools.music.Player", factory)
    return build_music_tools(cfg, Path("/nonexistent/blueutil"), "example"), made, ipc_calls


async def test_an_album_plays_in_order_and_replaces_the_queue(playing):
    built, _, calls = playing
    out = await built["music_play"].run({"query": "ace of spades", "output": "room-a"})
    loads = [c for c in calls if c[0] == "loadfile"]
    assert [c[2] for c in loads] == ["replace", "append-play", "append-play"]
    assert "Ace of Spades" in loads[0][1] and loads[0][1].endswith("01 - Ace of Spades.mp3")
    assert "in order" in out and "Room A" in out


async def test_an_artist_shuffles_unless_told_otherwise(playing):
    built, _, calls = playing
    out = await built["music_play"].run(
        {"query": "motorhead", "output": "room-a", "shuffle": False}
    )
    assert "in order" in out
    calls.clear()
    out = await built["music_play"].run({"query": "motorhead", "output": "room-a"})
    assert "shuffled" in out


async def test_append_adds_without_replacing_or_re_routing(playing):
    built, made, calls = playing
    await built["music_play"].run({"query": "overkill", "output": "room-a", "append": True})
    loads = [c for c in calls if c[0] == "loadfile"]
    assert loads and all(c[2] == "append-play" for c in loads)
    # A set is built onto a speaker that is already playing; re-running the
    # reconnect would risk a refusal in the middle of a working session.
    assert made["player"].bt_calls == []


async def test_a_refused_output_plays_nothing(cfg, library_root, monkeypatch, ipc_calls):
    cfg = MusicConfig(**{**cfg.__dict__, "library_root": library_root})
    monkeypatch.setattr(
        "claw.tools.music.Player",
        lambda c, b: FakePlayer(c, devices=set(), bt_table={
            "--is-connected": (0, "0", ""), "--connect": (1, "", "no connection"),
        }),
    )
    built = build_music_tools(cfg, Path("/nonexistent/blueutil"), "example")
    out = await built["music_play"].run({"query": "ace of spades", "output": "room-a"})
    assert out.startswith("refused:") and "Room A" in out
    assert not [c for c in ipc_calls if c[0] == "loadfile"]


# --- loudness policy -----------------------------------------------------

# A collection whose mode is -13 LUFS, boosting up to a 0 dBTP ceiling.
LD = LoudnessConfig(normalize=True, target_lufs=-13.0, boost_ceiling_dbtp=0.0)


def _t(lufs, peak, seconds=240.0):
    return Track(Path("x"), "", "", "", None, lufs=lufs, true_peak_dbfs=peak,
                 duration_s=seconds)


def test_playback_is_unity_unless_normalisation_is_switched_on():
    assert MusicConfig().loudness.normalize is False


def test_a_loud_track_is_cut_all_the_way_to_the_target():
    # Down is free: the peak is irrelevant when the gain is negative.
    assert gain_db(-4.2, 2.4, LD) == pytest.approx(-8.8)


def test_a_track_at_the_mode_is_untouched():
    # The whole reason to aim at the mode: switching normalisation on does not
    # move the typical record, so the room does not get quieter.
    assert gain_db(-13.0, -0.5, LD) == 0.0


def test_a_quiet_track_is_boosted_only_as_far_as_its_peak_allows():
    # Wants +8.3 to reach the target; its peak at -3.4 leaves room for +3.4.
    assert gain_db(-21.3, -3.4, LD) == pytest.approx(3.4)


def test_a_quiet_track_with_the_headroom_reaches_the_target():
    assert gain_db(-15.0, -6.0, LD) == pytest.approx(2.0)


def test_a_quiet_track_that_already_peaks_over_the_ceiling_is_left_alone():
    # Quiet by dynamics, not by level: a dynamic master already touching full
    # scale. Boosting it would clip; cutting it would be wrong too.
    assert gain_db(-16.4, 0.8, LD) == 0.0


def test_the_ceiling_is_a_config_value():
    assert gain_db(-21.3, -3.4, replace(LD, boost_ceiling_dbtp=-1.0)) == pytest.approx(2.4)


def test_the_target_is_a_config_value():
    assert gain_db(-8.0, 1.0, replace(LD, target_lufs=-18.0)) == pytest.approx(-10.0)


def test_gain_has_a_floor_so_a_bad_measurement_cannot_mute_a_track():
    assert gain_db(20.0, 0.0, LD) == -24.0


def test_an_unmeasured_track_is_assumed_loud_and_never_boosted():
    # The failure mode of a file nobody has ingested must be "cut too far",
    # never "far too loud in a room".
    assert gain_db(None, None, LD) == pytest.approx(LD.target_lufs - LD.assumed_lufs)
    assert gain_db(None, None, replace(LD, assumed_lufs=-30.0)) == 0.0


def test_no_peak_means_no_boost():
    assert gain_db(-20.0, None, LD) == 0.0


def test_album_gain_is_duration_weighted_in_the_energy_domain():
    # A loud eight-minute track must count for more than a loud two-minute one.
    long_loud = [_t(-6.0, 1.0, 480.0), _t(-20.0, -1.0, 120.0)]
    short_loud = [_t(-6.0, 1.0, 120.0), _t(-20.0, -1.0, 480.0)]
    assert album_gain_db(long_loud, LD) < album_gain_db(short_loud, LD)


def test_album_boost_is_bounded_by_the_records_loudest_peak():
    # One figure moves every track, so it must be safe for the hottest one.
    record = [_t(-20.0, -6.0), _t(-20.0, -1.0)]
    assert album_gain_db(record, LD) == pytest.approx(1.0)


def test_one_unmeasured_track_means_no_boost_for_the_record():
    record = [_t(-20.0, -6.0), Track(Path("y"), "", "", "", None, duration_s=240.0)]
    assert album_gain_db(record, replace(LD, assumed_lufs=-30.0)) == 0.0


def test_album_gain_is_one_figure_so_a_records_own_dynamics_survive(lib, library_root):
    tracks, kind, _ = lib.resolve(library_root, "ace of spades")
    assert kind == "album"
    one = album_gain_db(tracks, LD)
    # not simply the gain of the first track
    assert one != gain_db(tracks[0].lufs, tracks[0].true_peak_dbfs, LD)


def test_the_mode_is_the_hump_not_the_mean():
    # A skewed collection: most records near -13, a long quiet tail. The mean
    # is dragged toward the tail; the mode stays where the records are.
    values = [-13.0] * 40 + [-12.5] * 25 + [-13.5] * 25 + [-22.0] * 15 + [-30.0] * 10
    assert loudness_mode(values) == pytest.approx(-13.0, abs=0.3)
    assert sum(values) / len(values) < -15.0


def test_the_mode_ignores_silence_at_the_gate_floor():
    assert loudness_mode([-70.0] * 50 + [-12.0] * 3) == pytest.approx(-12.0, abs=0.1)
    assert loudness_mode([]) is None


# --- filters -------------------------------------------------------------

def test_genre_matches_loosely(lib, library_root):
    assert len(lib.select(library_root, genre="metal")) == 4
    assert len(lib.select(library_root, genre="rock")) == 3


def test_a_year_range_filters(lib, library_root):
    assert {t.year for t in lib.select(library_root, years=(1979, 1980))} == {1979, 1980}


def test_mood_finds_nothing_until_something_is_curated(lib, library_root):
    assert lib.select(library_root, mood="mellow") == []


# --- curation through the tool ------------------------------------------

async def test_curate_annotates_a_whole_album_and_the_next_query_sees_it(playing):
    built, _, _ = playing
    out = await built["music_curate"].run(
        {"query": "ace of spades", "mood": "heavy, driving", "energy": 5}
    )
    assert "3 tracks" in out
    got = await built["music_play"].run({"mood": "driving", "output": "room-b"})
    assert "3 tracks" in got


async def test_curate_refuses_a_measured_field(playing):
    built, _, _ = playing
    # Not in the schema, so this is the belt-and-braces path.
    out = await built["music_curate"].run({"query": "overkill", "lufs": -3})
    assert out.startswith("error:") and "nothing to set" in out


async def test_curate_is_a_no_op_when_nothing_actually_changes(playing):
    built, _, _ = playing
    await built["music_curate"].run({"query": "overkill", "scope": "track", "energy": 4})
    again = await built["music_curate"].run({"query": "overkill", "scope": "track", "energy": 4})
    assert "no change" in again


async def test_never_shuffle_overrides_an_explicit_shuffle(playing):
    built, _, calls = playing
    await built["music_curate"].run({"query": "ace of spades", "never_shuffle": True})
    calls.clear()
    out = await built["music_play"].run(
        {"query": "ace of spades", "output": "room-a", "shuffle": True}
    )
    assert "never-shuffle" in out and "in order" in out
    loads = [c for c in calls if c[0] == "loadfile"]
    assert loads[0][1].endswith("01 - Ace of Spades.mp3")


async def test_a_filter_that_empties_a_real_match_says_so_rather_than_substituting(playing):
    built, _, calls = playing
    out = await built["music_play"].run(
        {"query": "motorhead", "mood": "mellow", "output": "room-b"}
    )
    assert "is in the library, but nothing in it matches" in out
    assert not [c for c in calls if c[0] == "loadfile"]


async def test_play_needs_either_a_query_or_a_filter(playing):
    built, _, _ = playing
    assert (await built["music_play"].run({})).startswith("error:")


async def test_tracks_play_at_unity_by_default(playing):
    built, _, calls = playing
    await built["music_play"].run({"query": "overkill", "output": "room-a"})
    loads = [c for c in calls if c[0] == "loadfile"]
    # loadfile <path> <mode> <index> <per-file options>
    assert all(len(c) == 5 and c[4] == "volume-gain=0.00" for c in loads)


def _normalising_tools(cfg, library_root, monkeypatch, ld=LD):
    cfg = replace(cfg, library_root=library_root, loudness=ld)
    monkeypatch.setattr("claw.tools.music.Player",
                        lambda c, b: FakePlayer(c, devices={BT.mpv_device, WIRED.mpv_device},
                                                bt_table={"--is-connected": (0, "1", "")}))
    return build_music_tools(cfg, Path("/nonexistent/blueutil"), "example")


def _gains(calls):
    return [float(c[4].removeprefix("volume-gain=")) for c in calls if c[0] == "loadfile"]


async def test_normalisation_rides_on_each_entry_in_db_not_mpv_volume(
    cfg, library_root, monkeypatch, ipc_calls
):
    # mpv's `volume` is cubic, so a percentage computed from dB triples the cut.
    # `volume-gain` is dB-native; nothing here may use `volume=` at all.
    built = _normalising_tools(cfg, library_root, monkeypatch)
    await built["music_play"].run({"query": "ace of spades", "output": "room-a"})
    loads = [c for c in ipc_calls if c[0] == "loadfile"]
    assert loads and all(c[4].startswith("volume-gain=") for c in loads)
    # the album is louder than the -13 target, so one negative figure for all of it
    gains = _gains(ipc_calls)
    assert len(set(gains)) == 1 and gains[0] < 0


async def test_a_quiet_track_is_boosted_up_to_its_peak_on_the_way_out(
    cfg, library_root, monkeypatch, ipc_calls
):
    # loose-single: -20 LUFS peaking at -1.0 dBFS, against a -13 target and a
    # 0 dBTP ceiling — it wants +7 and its peak allows +1.
    built = _normalising_tools(cfg, library_root, monkeypatch)
    await built["music_play"].run({"query": "loose single", "output": "room-a"})
    assert _gains(ipc_calls) == [pytest.approx(1.0)]


# --- album furniture -----------------------------------------------------

def test_furniture_is_a_short_track_far_quieter_than_its_own_record():
    from claw.music import is_furniture
    coda = Track(Path("a"), "", "", "Outside The Wall", None, lufs=-34.0, duration_s=104.0)
    song = Track(Path("b"), "", "", "Comfortably Numb", None, lufs=-19.0, duration_s=380.0)
    quiet_but_long = Track(Path("c"), "", "", "A Long Quiet Piece", None,
                           lufs=-34.0, duration_s=600.0)
    assert is_furniture(coda, -19.3, LD)
    assert not is_furniture(song, -19.3, LD)
    assert not is_furniture(quiet_but_long, -19.3, LD)   # length says it is the point
    assert not is_furniture(coda, None, LD)              # unknown album: leave it alone


def test_the_furniture_thresholds_are_config_values():
    from claw.music import is_furniture
    segue = Track(Path("a"), "", "", "Segue", None, lufs=-25.0, duration_s=150.0)
    assert not is_furniture(segue, -19.0, LD)            # 150 s is past the default 120
    assert is_furniture(segue, -19.0, replace(LD, furniture_max_s=180.0,
                                              furniture_below_album_lu=5.0))


async def test_a_shuffle_drops_furniture_but_album_order_keeps_it(playing, cfg, library_root):
    built, _, calls = playing
    con = music_db.connect(cfg.db_path)
    try:                                    # make one track look like a coda
        con.execute("UPDATE tracks SET lufs=-30.0, duration_s=40.0 "
                    "WHERE path LIKE '%Ace of Spades/10%'")
        con.commit()
    finally:
        con.close()

    calls.clear()
    await built["music_play"].run({"query": "ace of spades", "output": "room-a"})
    in_order = [c[1] for c in calls if c[0] == "loadfile"]
    assert any("10 - The Chase" in p for p in in_order)      # kept in sequence

    calls.clear()
    await built["music_play"].run(
        {"query": "ace of spades", "output": "room-a", "shuffle": True})
    shuffled_ = [c[1] for c in calls if c[0] == "loadfile"]
    assert not any("10 - The Chase" in p for p in shuffled_)  # dropped from a shuffle


async def test_dropping_furniture_never_leaves_an_empty_queue(lib, library_root):
    # An album that is *all* interstitials must still play.
    only = [Track(Path("x"), "A", "B", "seg", 1, lufs=-40.0, duration_s=10.0)]
    assert lib.drop_furniture(library_root, only, LD) == only


async def test_queued_tracks_are_recorded_as_history(playing, cfg):
    built, _, _ = playing
    await built["music_play"].run({"query": "ace of spades", "output": "room-a"})
    con = music_db.connect(cfg.db_path)
    try:
        rows = music_db.recent_plays(con)
    finally:
        con.close()
    assert len(rows) == 3 and rows[0]["output"] == "room-a"


async def test_status_reports_what_was_recently_queued(playing):
    # music_play stays dumb about repeats; the skill decides, using this.
    built, _, _ = playing
    await built["music_play"].run({"query": "ace of spades", "output": "room-a"})
    out = await built["music_status"].run({})
    assert "recently queued" in out and "Ace of Spades" in out


async def test_recent_records_are_grouped_by_album_not_listed_per_track(playing, cfg):
    built, _, _ = playing
    await built["music_play"].run({"query": "ace of spades", "output": "room-a"})
    con = music_db.connect(cfg.db_path)
    try:
        rows = music_db.recent_records(con)
    finally:
        con.close()
    assert len(rows) == 1 and rows[0]["n"] == 3     # one record, three tracks


async def test_status_still_reports_history_when_nothing_is_playing(playing, monkeypatch):
    built, _, _ = playing
    await built["music_play"].run({"query": "overkill", "output": "room-a"})

    async def idle(sock, commands, timeout=10.0):
        return [{"error": "success", "data": True if c[1] == "idle-active" else None}
                for c in commands]

    monkeypatch.setattr(music_mod, "ipc", idle)
    out = await built["music_status"].run({})
    assert out.startswith("nothing is playing") and "recently queued" in out


async def test_a_renamed_artist_shows_under_its_current_name_in_history(playing, cfg):
    built, _, _ = playing
    await built["music_play"].run({"query": "overkill", "output": "room-a"})
    con = music_db.connect(cfg.db_path)
    try:
        music_db.curate(con, "artist", music_db.find_artist(con, "Motörhead"),
                        {"name": "Motorhead (renamed)"}, by="z")
        rows = music_db.recent_records(con)
    finally:
        con.close()
    assert rows[0]["artist"] == "Motorhead (renamed)"


# --- handles ---------------------------------------------------------------

def test_a_track_handle_is_short_stable_and_path_derived():
    from claw.music import track_handle
    h = track_handle("Motörhead/Overkill/01 - Overkill.mp3")
    assert h == track_handle("Motörhead/Overkill/01 - Overkill.mp3")
    assert h.startswith("t:") and len(h) == 10 and h[2:].isalnum()
    assert h != track_handle("Motörhead/Overkill/02 - Stay Clean.mp3")


def test_handles_resolve_exactly(lib, library_root):
    t = next(t for t in lib.tracks(library_root) if t.title == "Overkill")
    assert lib.resolve(library_root, t.handle) == ([t], "track", t.full_label())

    album, kind, _ = lib.resolve(library_root, f"al:{t.album_id}")
    assert kind == "album" and [x.title for x in album] == ["Overkill"]

    artist, kind, label = lib.resolve(library_root, lib.artist_handle(library_root, "Motörhead"))
    assert (kind, label, len(artist)) == ("artist", "Motörhead", 4)


def test_an_unknown_handle_resolves_to_nothing_not_the_nearest_thing(lib, library_root):
    for h in ("t:zzzzzzzz", "al:99999", "ar:99999"):
        assert lib.resolve(library_root, h) == ([], "none", "")


async def test_play_says_a_bad_handle_is_a_bad_handle(playing):
    built, _, calls = playing
    out = await built["music_play"].run({"query": "t:zzzzzzzz", "output": "room-a"})
    assert out.startswith("error: no such handle") and "music_search" in out
    assert not [c for c in calls if c[0] == "loadfile"]


async def test_an_album_handle_plays_that_record_in_order(playing, lib, library_root):
    built, _, calls = playing
    aid = next(t.album_id for t in lib.tracks(library_root) if t.album == "Ace of Spades")
    out = await built["music_play"].run({"query": f"al:{aid}", "output": "room-a"})
    assert "in order" in out
    loads = [c[1] for c in calls if c[0] == "loadfile"]
    assert [Path(p).name[:2] for p in loads] == ["01", "02", "10"]


async def test_curate_through_an_artist_handle_lands_at_artist_level(playing, lib, library_root):
    built, _, _ = playing
    out = await built["music_curate"].run(
        {"query": lib.artist_handle(library_root, "Motörhead"), "notes": "loud on purpose"})
    assert "(artist)" in out and "4 tracks" in out


# --- search ------------------------------------------------------------------

async def _search(built, **args):
    return await built["music_search"].run(args)


async def test_search_is_read_only(playing):
    built, _, calls = playing
    await _search(built, query="motorhead")
    assert calls == []


async def test_an_artist_match_lists_their_records(playing):
    built, _, _ = playing
    out = await _search(built, query="motorhead")
    assert "Motörhead · 2 albums, 4 tracks" in out
    assert "Motörhead — Ace of Spades" in out and "Motörhead — Overkill" in out


async def test_words_can_span_fields(playing):
    built, _, _ = playing
    out = await _search(built, query="motorhead ace")
    assert "Motörhead — Ace of Spades" in out and "matched: artist + album" in out


async def test_a_title_match_is_a_track_row_with_a_handle(playing, lib, library_root):
    built, _, _ = playing
    out = await _search(built, query="hells bells")
    t = next(t for t in lib.tracks(library_root) if t.title == "Hells Bells")
    assert f"{t.handle}  AC/DC — Hells Bells" in out


async def test_a_misspelling_still_finds_the_artist(playing):
    built, _, _ = playing
    assert "Motörhead · 2 albums" in await _search(built, query="motorhed")


async def test_a_miss_says_so_and_offers_only_labelled_near_misses(playing):
    built, _, _ = playing
    out = await _search(built, query="zzqx")
    assert out.startswith('nothing matches "zzqx"') and "in 8 tracks" in out
    near = await _search(built, query="overkil zz")
    assert near.startswith("nothing matches") and "NOT matches" in near


async def test_a_note_is_one_row_on_the_thing_it_is_about(playing, lib, library_root):
    built, _, _ = playing
    await built["music_curate"].run(
        {"query": lib.artist_handle(library_root, "Motörhead"), "notes": "played loud"})
    out = await _search(built, query="played loud", scope="artist")
    assert "matched: note" in out and out.count("Motörhead ·") == 1
    unscoped = await _search(built, query="played loud")
    assert "0 track(s)" in unscoped                  # inherited, not re-reported per track
    # ...unless tracks are what was asked for: then the artist is its tracks.
    tracks = await _search(built, query="played loud", scope="track")
    assert "4 track(s)" in tracks and "matched: artist note" in tracks


async def test_scope_reports_an_artist_match_as_their_tracks(playing):
    built, _, _ = playing
    out = await _search(built, query="motorhead", scope="track")
    assert out.startswith('search "motorhead" as tracks') and "4 track(s)" in out
    assert "NOT matches" not in out and "matched: artist" in out
    rows = [l for l in out.splitlines() if l.startswith("  t:")]
    assert [r.split(" — ")[1].split(" (")[0] for r in rows] == [
        "Ace of Spades", "Love Me Like a Reptile", "The Chase Is Better Than the Catch",
        "Overkill"]                                   # record by record, in running order


async def test_scope_ranks_the_kinds_own_matches_first(playing):
    built, _, _ = playing
    out = await _search(built, query="ace of spades", scope="track")
    rows = [l for l in out.splitlines() if l.startswith("  t:")]
    assert len(rows) == 3
    assert "— Ace of Spades (" in rows[0] and "matched: title, album" in rows[0]
    assert all(r.endswith("matched: album") for r in rows[1:])


async def test_scope_reports_a_track_match_as_its_record_and_artist(playing):
    built, _, _ = playing
    album = await _search(built, query="hells bells", scope="album")
    assert "AC/DC — Back in Black" in album and "matched: track title" in album
    artist = await _search(built, query="appetite", scope="artist")
    assert "Guns N' Roses ·" in artist and "matched: album" in artist


async def test_a_hit_that_cannot_be_the_scope_is_not_a_silent_miss(playing, lib, library_root):
    built, _, _ = playing
    loose = next(t for t in lib.tracks(library_root) if not t.artist)
    out = await _search(built, query=loose.title, scope="artist")
    assert "can be shown as a artist" not in out
    assert "can be shown as an artist" in out and "1 track(s)" in out
    assert "NOT matches" not in out


async def test_nothing_is_quietly_filtered_interludes_appear_labelled(playing, cfg):
    built, _, _ = playing
    con = music_db.connect(cfg.db_path)
    try:                                    # make one track look like a coda
        con.execute("UPDATE tracks SET lufs=-30.0, duration_s=40.0 "
                    "WHERE path LIKE '%Ace of Spades/10%'")
        con.commit()
    finally:
        con.close()
    out = await _search(built, query="the chase")
    assert "The Chase Is Better Than the Catch" in out and "interlude" in out


async def test_recently_queued_things_appear_with_when(playing):
    built, _, _ = playing
    await built["music_play"].run({"query": "overkill", "output": "room-a"})
    out = await _search(built, query="overkill")
    assert "last queued" in out


async def test_a_display_cap_is_announced_with_the_true_total(playing, monkeypatch):
    built, _, _ = playing
    monkeypatch.setitem(tools_mod.SEARCH_SHOW, "albums", 1)
    out = await _search(built, query="rock", scope="album")
    assert "showing 1 of 2" in out


async def test_narrowing_is_restated_and_applied(playing):
    built, _, _ = playing
    out = await _search(built, query="motorhead", genre="rock")
    assert out.startswith('nothing matches "motorhead" within genre')
    out = await _search(built, query="motorhead", year_min=1980, year_max=1980)
    assert "within years 1980-1980" in out and "Overkill" not in out



# --- set candidates -----------------------------------------------------------

def _tr(artist, album, n, album_id, queued=""):
    return Track(Path(f"/lib/{artist}/{album}/{n:02d}.mp3"), artist, album, f"{album} {n}", n,
                 album_id=album_id, duration_s=240.0)


def test_every_prefix_of_the_interleave_is_as_varied_as_it_can_be():
    import random
    from claw.music import interleave
    a = [_tr("A", f"A{al}", n, al) for al in (1, 2, 3) for n in (1, 2)]
    b = [_tr("B", "B1", n, 10) for n in (1, 2)]
    c = [_tr("C", "C1", 1, 20)]
    order = interleave(a + b + c, {}, random.Random(7))
    assert {t.artist for t in order[:3]} == {"A", "B", "C"}          # one each first
    a_picks = [t for t in order if t.artist == "A"]
    assert len({t.album_id for t in a_picks[:3]}) == 3               # A spread over records
    assert sorted(order, key=id) == sorted(a + b + c, key=id)        # nothing lost


def test_within_a_record_never_queued_comes_first():
    import random
    from claw.music import interleave
    ts = [_tr("A", "A1", n, 1) for n in (1, 2, 3)]
    queued = {ts[0].path: "2026-09-01T00:00:00+00:00", ts[1].path: "2026-09-05T00:00:00+00:00"}
    for seed in range(5):
        assert interleave(ts, queued, random.Random(seed))[0] is ts[2]


async def _cands(built, **args):
    return await built["music_candidates"].run(args)


async def test_the_pool_is_sized_to_the_set_times_the_factor(playing):
    built, _, _ = playing
    # 240 s tracks; an 8 minute set at factor 2 wants 16 minutes = 4 tracks.
    out = await _cands(built, minutes=8)
    assert "pool of 4 tracks, 16:00" in out and out.count("  t:") == 4


async def test_candidates_are_filtered_and_say_so(playing, cfg):
    built, _, _ = playing
    con = music_db.connect(cfg.db_path)
    try:                                    # one interlude
        con.execute("UPDATE tracks SET lufs=-30.0, duration_s=40.0 "
                    "WHERE path LIKE '%Ace of Spades/10%'")
        con.commit()
    finally:
        con.close()
    await built["music_play"].run({"query": "overkill", "output": "room-a"})   # one recent
    out = await _cands(built, artist="motorhead")
    assert "4 tracks matched" in out
    assert "1 queued in the last 24h" in out and "1 interlude." in out
    assert "Overkill" not in out.split("\n", 3)[3]      # not offered in the pool rows


async def test_an_excluded_genre_is_counted_and_asking_for_it_overrides(
    cfg, library_root, monkeypatch, ipc_calls
):
    from claw.config import CandidatesConfig
    built = _normalising_tools(replace(cfg, candidates=CandidatesConfig(exclude_genres=("Metal",))),
                               library_root, monkeypatch)
    out = await _cands(built)
    assert "4 in excluded genres (Metal)" in out
    asked = await _cands(built, genre="metal")
    assert "nothing left out" in asked


async def test_an_unknown_artist_is_refused_not_guessed(playing):
    built, _, _ = playing
    out = await _cands(built, artist="nobody at all")
    assert out.startswith("error: no artist matching") and "music_search" in out


async def test_a_short_pool_says_it_is_everything(playing):
    built, _, _ = playing
    out = await _cands(built, artist="ac/dc", minutes=60)
    assert "ALL that is eligible" in out


async def test_candidates_point_at_search_for_existence(playing):
    built, _, _ = playing
    assert "music_search" in await _cands(built, genre="rock")
    assert "music_search" in built["music_candidates"].description


# --- playing a chosen set by handle ---------------------------------------------

async def test_handles_play_exactly_in_the_order_given(playing, lib, library_root):
    built, _, calls = playing
    picks = [t for t in lib.tracks(library_root) if t.title in ("Overkill", "Hells Bells")]
    order = [picks[1].handle, picks[0].handle]
    out = await built["music_play"].run({"handles": order, "output": "room-a"})
    assert "in your order" in out and "8:00" in out
    loads = [c[1] for c in calls if c[0] == "loadfile"]
    assert loads == [str(picks[1].path), str(picks[0].path)]


async def test_one_bad_handle_queues_nothing(playing, lib, library_root):
    built, _, calls = playing
    good = lib.tracks(library_root)[0].handle
    out = await built["music_play"].run({"handles": [good, "t:zzzzzzzz"], "output": "room-a"})
    assert out.startswith("error: no such handle") and "Nothing was queued" in out
    assert not [c for c in calls if c[0] == "loadfile"]


async def test_an_artist_handle_is_not_a_pick(playing, lib, library_root):
    built, _, _ = playing
    out = await built["music_play"].run(
        {"handles": [lib.artist_handle(library_root, "Motörhead")], "output": "room-a"})
    assert out.startswith("error:") and "whole artist" in out


async def test_handles_and_a_query_together_are_refused(playing, lib, library_root):
    built, _, _ = playing
    out = await built["music_play"].run(
        {"handles": [lib.tracks(library_root)[0].handle], "query": "overkill"})
    assert out.startswith("error: pass handles OR")


# --- music_dj -------------------------------------------------------------------

@pytest.fixture
def dj(cfg, library_root, tmp_path, monkeypatch, ipc_calls):
    """Tools with the DJ enabled; rendering faked, recorded in `rendered`."""
    from claw import dj as dj_mod
    from claw.config import DjConfig
    rendered: list[list[str]] = []

    async def fake_render(self, lines):
        rendered.append(list(lines))
        session = self.cfg.render_dir / f"s{len(rendered)}"
        session.mkdir(parents=True, exist_ok=True)
        clips = []
        for i, text in enumerate(lines, 1):
            p = session / f"{i:02d}.wav"
            p.write_bytes(b"")
            clips.append(dj_mod.Clip(path=p, text=text, duration_s=5.0))
        return clips

    monkeypatch.setattr(dj_mod.Announcer, "render", fake_render)
    c = replace(cfg, library_root=library_root,
                dj=DjConfig(enabled=True, render_dir=tmp_path / "dj"))
    monkeypatch.setattr("claw.tools.music.Player",
                        lambda cc, b: FakePlayer(cc, devices={BT.mpv_device, WIRED.mpv_device},
                                                 bt_table={"--is-connected": (0, "1", "")}))
    return build_music_tools(c, Path("/nonexistent/blueutil"), "example"), rendered, ipc_calls, c


def _h(lib, root, title):
    return next(t.handle for t in lib.tracks(root) if t.title == title)


async def test_music_dj_exists_only_when_the_dj_is_enabled(tools, dj):
    built, _ = tools
    assert "music_dj" not in built
    assert "music_dj" in dj[0]


async def test_a_run_sheet_queues_each_link_before_its_track(dj, lib, library_root):
    built, rendered, calls, c = dj
    out = await built["music_dj"].run({"output": "room-a", "set": [
        {"say": "Evening."},
        {"play": _h(lib, library_root, "Overkill")},
        {"say": "That was Motörhead. Here's AC/DC.", "play": _h(lib, library_root, "Hells Bells")},
        {"say": "That's me for now."},
    ]})
    assert rendered == [["Evening.", "That was Motörhead. Here's AC/DC.", "That's me for now."]]
    loads = [c_[1] for c_ in calls if c_[0] == "loadfile"]
    kinds = ["link" if "/dj/" in p else Path(p).stem for p in loads]
    assert kinds == ["link", "01 - Overkill", "link", "01 - Hells Bells", "link"]
    assert "2 tracks and 3 links" in out and "8:15" in out     # 8:00 of music + 15 s of talk


async def test_links_play_at_unity_even_when_normalising(dj, lib, library_root):
    _, _, calls, c = dj
    built = build_music_tools(replace(c, loudness=LD), Path("/nonexistent/blueutil"), "example")
    await built["music_dj"].run({"output": "room-a", "set": [
        {"say": "Here we go.", "play": _h(lib, library_root, "Hells Bells")}]})
    link = next(x for x in calls if x[0] == "loadfile" and "/dj/" in x[1])
    track = next(x for x in calls if x[0] == "loadfile" and "/dj/" not in x[1])
    assert link[4] == "volume-gain=0.00"
    assert track[4] == "volume-gain=-4.00"       # -9 LUFS, normalised to the -13 target


async def test_one_bad_handle_renders_and_queues_nothing(dj, lib, library_root):
    built, rendered, calls, _ = dj
    out = await built["music_dj"].run({"output": "room-a", "set": [
        {"say": "First.", "play": _h(lib, library_root, "Overkill")},
        {"say": "Second.", "play": "t:zzzzzzzz"},
    ]})
    assert out.startswith("error: nothing was queued") and "t:zzzzzzzz" in out
    assert rendered == [] and not [x for x in calls if x[0] == "loadfile"]


@pytest.mark.parametrize("play", ["ar:1", "overkill"])
async def test_only_track_and_record_handles_can_be_djd(dj, play):
    built, rendered, _, _ = dj
    out = await built["music_dj"].run({"set": [{"say": "Hi.", "play": play}]})
    assert "not a track or record handle" in out and rendered == []


async def test_a_record_plays_whole_in_order_at_one_gain(dj, lib, library_root):
    # A link introduces the record; the record then plays complete, in its own
    # order, at the single figure music_play would give it — so its own quiet
    # and loud tracks keep their relationship — while a lone track beside it
    # still gets its own.
    _, rendered, calls, c = dj
    built = build_music_tools(replace(c, loudness=LD), Path("/nonexistent/blueutil"), "example")
    record = [t for t in lib.tracks(library_root) if t.album == "Ace of Spades"]
    record.sort(key=lambda t: t.number)
    out = await built["music_dj"].run({"output": "room-a", "set": [
        {"say": "Here's a whole record.", "play": f"al:{record[0].album_id}"},
        {"say": "And one more.", "play": _h(lib, library_root, "Hells Bells")},
    ]})
    loads = [x for x in calls if x[0] == "loadfile"]
    names = ["link" if "/dj/" in x[1] else Path(x[1]).stem for x in loads]
    assert names == ["link", *[Path(t.path).stem for t in record], "link", "01 - Hells Bells"]
    one = f"volume-gain={album_gain_db(record, LD):.2f}"
    assert [x[4] for x in loads[1:1 + len(record)]] == [one] * len(record)
    assert loads[-1][4] == "volume-gain=-4.00"          # the lone track, its own gain
    assert "Ace of Spades" in out and "whole and in order" in out


async def test_an_unknown_record_handle_queues_nothing(dj):
    built, rendered, calls, _ = dj
    out = await built["music_dj"].run({"set": [{"say": "Hi.", "play": "al:99999"}]})
    assert "no such handle al:99999" in out
    assert rendered == [] and not [x for x in calls if x[0] == "loadfile"]


async def test_two_links_with_no_track_between_are_refused(dj, lib, library_root):
    built, rendered, _, _ = dj
    out = await built["music_dj"].run({"set": [
        {"say": "One."}, {"say": "Two.", "play": _h(lib, library_root, "Overkill")}]})
    assert "two links with no track between" in out and rendered == []


async def test_a_set_of_only_talk_needs_append(dj):
    built, rendered, calls, _ = dj
    refused = await built["music_dj"].run({"set": [{"say": "Goodnight."}]})
    assert refused.startswith("error: a set with no tracks") and rendered == []
    ok = await built["music_dj"].run({"set": [{"say": "Goodnight."}], "append": True,
                                      "output": "room-a"})
    assert ok.startswith("queued a DJ set") and len(rendered) == 1


async def test_minutes_are_checked_before_anything_is_rendered(dj, lib, library_root):
    built, rendered, calls, _ = dj
    two = [{"play": _h(lib, library_root, "Overkill")},
           {"play": _h(lib, library_root, "Hells Bells")}]            # 8 min of music
    out = await built["music_dj"].run({"set": two, "minutes": 20, "output": "room-a"})
    assert "runs 8:00 of music against the 20 min" in out and "add about 12 min" in out
    assert rendered == [] and not [x for x in calls if x[0] == "loadfile"]
    ok = await built["music_dj"].run({"set": two, "minutes": 8.5, "output": "room-a"})
    assert ok.startswith("playing a DJ set")


async def test_a_render_failure_queues_nothing(dj, lib, library_root, monkeypatch):
    from claw import dj as dj_mod
    built, _, calls, _ = dj

    async def broken(self, lines):
        raise dj_mod.RenderFailed("piper is missing")

    monkeypatch.setattr(dj_mod.Announcer, "render", broken)
    out = await built["music_dj"].run({"output": "room-a", "set": [
        {"say": "Hi.", "play": _h(lib, library_root, "Overkill")}]})
    assert "piper is missing" in out and "Nothing was queued" in out
    assert not [x for x in calls if x[0] == "loadfile"]


# --- the seam: a trailing sign-off, and a link in status ---------------------------

@pytest.fixture
def mpv_state(monkeypatch):
    """Program what mpv reports for get_property; record every command."""
    state: dict[str, object] = {}
    calls: list[list] = []

    async def fake_ipc(sock, commands, timeout=10.0):
        calls.extend([list(c) for c in commands])
        return [{"error": "success",
                 "data": state.get(c[1]) if c[0] == "get_property" else None}
                for c in commands]

    monkeypatch.setattr(music_mod, "ipc", fake_ipc)
    return state, calls


def _signoff(c):
    p = c.dj.render_dir / "old" / "03.wav"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return str(p)


async def test_appending_drops_a_trailing_signoff(dj, mpv_state, lib, library_root):
    built, _, _, c = dj
    state, calls = mpv_state
    state["playlist"] = [{"filename": "/lib/a.mp3", "current": True}, {"filename": _signoff(c)}]
    await built["music_play"].run({"handles": [_h(lib, library_root, "Overkill")], "append": True})
    cmds = [x for x in calls if x[0] in ("playlist-remove", "loadfile")]
    assert cmds[0] == ["playlist-remove", 1] and cmds[1][0] == "loadfile"


async def test_a_signoff_being_spoken_is_not_cut_off(dj, mpv_state, lib, library_root):
    built, _, _, c = dj
    state, calls = mpv_state
    state["playlist"] = [{"filename": "/lib/a.mp3"}, {"filename": _signoff(c), "current": True}]
    await built["music_play"].run({"handles": [_h(lib, library_root, "Overkill")], "append": True})
    assert not [x for x in calls if x[0] == "playlist-remove"]


async def test_status_says_the_dj_is_talking_and_what_comes_next(dj, mpv_state, lib, library_root):
    built, _, _, c = dj
    state, _ = mpv_state
    link = _signoff(c)
    overkill = next(t for t in lib.tracks(library_root) if t.title == "Overkill")
    hells = next(t for t in lib.tracks(library_root) if t.title == "Hells Bells")
    state.update({
        "path": link, "idle-active": False, "pause": False, "playlist-pos-1": 1,
        "playlist-count": 3, "time-pos": 2, "duration": 5, "audio-device": BT.mpv_device,
        "playlist": [{"filename": link, "current": True}, {"filename": str(overkill.path)},
                     {"filename": str(hells.path)}],
    })
    out = await built["music_status"].run({})
    assert "a DJ link" in out and "next: Motörhead — Overkill" in out
    assert "the queue ends with AC/DC — Hells Bells" in out



# --- music_history ------------------------------------------------------------------

async def test_history_of_a_track_includes_its_record_and_artist(playing, lib, library_root):
    built, _, _ = playing
    await built["music_curate"].run({"query": lib.artist_handle(library_root, "Motörhead"),
                                     "notes": "loud on purpose"})
    await built["music_curate"].run({"query": _h(lib, library_root, "Overkill"), "energy": 5})
    out = await built["music_history"].run({"query": _h(lib, library_root, "Overkill")})
    assert "2 edits" in out
    assert "energy: (none) → 5" in out and "notes: (none) → loud on purpose" in out
    assert out.index("energy") < out.index("notes")               # newest first


async def test_history_of_an_artist_includes_their_records_and_tracks(playing, lib, library_root):
    built, _, _ = playing
    aid = next(t.album_id for t in lib.tracks(library_root) if t.album == "Ace of Spades")
    await built["music_curate"].run({"query": f"al:{aid}", "mood": "fast"})
    await built["music_curate"].run({"query": "hells bells", "scope": "track", "energy": 3})
    out = await built["music_history"].run({"query": lib.artist_handle(library_root, "Motörhead")})
    assert "Ace of Spades" in out and "mood: (none) → fast" in out
    assert "Hells Bells" not in out                              # someone else's track


async def test_history_with_no_query_is_the_whole_library_newest_first(playing, lib, library_root):
    built, _, _ = playing
    await built["music_curate"].run({"query": "hells bells", "scope": "track", "energy": 3})
    await built["music_curate"].run({"query": "overkill", "scope": "track", "energy": 4})
    out = await built["music_history"].run({"limit": 1})
    assert "2 edits" in out and "showing 1" in out and "Overkill" in out


async def test_history_is_read_only(playing):
    built, _, calls = playing
    await built["music_history"].run({})
    assert calls == []


async def test_nothing_recorded_says_so(playing):
    built, _, _ = playing
    assert (await built["music_history"].run({"query": "overkill"})).startswith("no edits recorded")


async def test_a_bad_handle_in_history_is_a_bad_handle(playing):
    built, _, _ = playing
    assert (await built["music_history"].run({"query": "t:zzzzzzzz"})).startswith("error: no such handle")


async def test_curation_lost_to_a_vanished_file_is_in_the_log(playing, cfg, library_root):
    built, made, _ = playing
    await built["music_curate"].run({"query": "overkill", "scope": "track",
                                     "mood": "fast", "notes": "the opener"})
    con = music_db.connect(cfg.db_path)
    try:
        music_db.forget(con, ["Motörhead/Overkill/01 - Overkill.mp3"], by="tester")
    finally:
        con.close()
    made["player"].library.reload(library_root)     # as the next cache refresh would
    out = await built["music_history"].run({})
    assert "(no longer in the library)" in out and "forgotten" in out
    assert "notes=the opener" in out and "tester" in out



# --- notes add up; the detail view shows everything ---------------------------------

async def test_a_row_shows_every_levels_note_lowest_first(playing, lib, library_root):
    built, _, _ = playing
    await built["music_curate"].run({"query": lib.artist_handle(library_root, "Motörhead"),
                                     "notes": "loud on purpose"})
    await built["music_curate"].run({"query": _h(lib, library_root, "Overkill"),
                                     "notes": "the opener"})
    out = await built["music_search"].run({"query": "overkill", "scope": "track"})
    row = next(l for l in out.splitlines() if "Overkill (Overkill" in l)
    assert row.index("note (track): the opener") < row.index("note (artist): loud on purpose")


async def test_a_row_cuts_a_long_note_and_the_detail_view_does_not(playing, lib, library_root):
    built, _, _ = playing
    long = "A " + "very " * 40 + "long story about the record."
    h = _h(lib, library_root, "Overkill")
    await built["music_curate"].run({"query": h, "notes": long})
    row = await built["music_search"].run({"query": "overkill", "scope": "track"})
    assert long not in row and "..." in row
    detail = await built["music_search"].run({"query": h})
    assert long in detail


async def test_track_detail_shows_each_level_and_points_at_history(playing, lib, library_root):
    built, _, _ = playing
    await built["music_curate"].run({"query": lib.artist_handle(library_root, "Motörhead"),
                                     "notes": "loud on purpose"})
    h = _h(lib, library_root, "Overkill")
    await built["music_curate"].run({"query": h, "energy": 5})
    out = await built["music_search"].run({"query": h})
    assert out.startswith(f"{h}  Motörhead — Overkill")
    assert "track  energy 5" in out and "artist notes: loud on purpose" in out
    assert f"history: 2 edits — music_history {h}" in out


async def test_album_detail_lists_its_tracks_with_handles(playing, lib, library_root):
    built, _, _ = playing
    aid = next(t.album_id for t in lib.tracks(library_root) if t.album == "Ace of Spades")
    out = await built["music_search"].run({"query": f"al:{aid}"})
    assert "tracks:" in out and _h(lib, library_root, "Love Me Like a Reptile") in out
    assert "(nothing curated at any level)" in out


async def test_artist_detail_lists_their_records(playing, lib, library_root):
    built, _, _ = playing
    out = await built["music_search"].run({"query": lib.artist_handle(library_root, "Motörhead")})
    assert "2 records, 4 tracks" in out and "Ace of Spades" in out and "Overkill" in out


async def test_a_bad_handle_to_search_is_a_bad_handle(playing):
    built, _, _ = playing
    assert (await built["music_search"].run({"query": "al:99999"})).startswith("error: no such handle")
