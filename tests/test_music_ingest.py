"""music: ingest is per-file, and collection ingest is a loop over it.

Audio is never decoded here — ``measure`` and ``probe`` are the boundary to
ffmpeg and are replaced, so these tests are about the *bookkeeping*: what gets
opened, what gets refreshed, and what survives.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from claw import music_db, music_ingest


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "audio"
    for rel in ("A/B/01 - One.mp3", "A/B/02 - Two.mp3", "A/C/01 - Three.mp3"):
        p = r / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 100)
    (r / "A/B/.DS_Store").write_bytes(b"")
    (r / "A/B/lamer.sh").write_bytes(b"")
    (r / "A/B/cover.jpg").write_bytes(b"")
    return r


@pytest.fixture
def con(tmp_path):
    c = music_db.connect(tmp_path / "library.db")
    yield c
    c.close()


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    """Stand in for the two subprocess boundaries, and count the openings."""
    opened: list[str] = []

    def measure(path, timeout=300.0):
        opened.append(str(path))
        return {"lufs": -14.0, "true_peak_dbfs": -0.5, "lra_lu": 6.0}

    def probe(path, timeout=60.0):
        return {"duration_s": 240.0, "tags": {"title": "Tagged Title",
                                              "genre": "Rock", "date": "1971"}}

    monkeypatch.setattr(music_ingest, "measure", measure)
    monkeypatch.setattr(music_ingest, "probe", probe)
    return opened


def test_a_new_file_is_inserted(con, root, fake_ffmpeg):
    assert music_ingest.ingest_file(con, root, root / "A/B/01 - One.mp3") == "new"
    row = con.execute("SELECT * FROM catalog").fetchone()
    assert (row["artist"], row["album"], row["track_no"]) == ("A", "B", 1)
    assert row["lufs"] == -14.0


def test_the_tag_supplies_the_title_and_the_filename_the_number(con, root, fake_ffmpeg):
    # ID3 renders a track number as '3/12' as often as '3'; the filename does not.
    music_ingest.ingest_file(con, root, root / "A/B/02 - Two.mp3")
    row = con.execute("SELECT * FROM tracks").fetchone()
    assert (row["title"], row["track_no"]) == ("Tagged Title", 2)


def test_an_unchanged_file_is_never_opened_again(con, root, fake_ffmpeg):
    f = root / "A/B/01 - One.mp3"
    music_ingest.ingest_file(con, root, f)
    fake_ffmpeg.clear()
    assert music_ingest.ingest_file(con, root, f) == "unchanged"
    assert fake_ffmpeg == []          # the cheap stat pass short-circuited it


def test_moved_bytes_are_remeasured(con, root, fake_ffmpeg):
    f = root / "A/B/01 - One.mp3"
    music_ingest.ingest_file(con, root, f)
    f.write_bytes(b"y" * 200)
    assert music_ingest.ingest_file(con, root, f) == "remeasured"


def test_a_remeasure_does_not_disturb_curation(con, root, fake_ffmpeg):
    f = root / "A/B/01 - One.mp3"
    music_ingest.ingest_file(con, root, f)
    music_db.curate(con, "track", "A/B/01 - One.mp3", {"genre": "Dub", "mood": ["mellow"]}, by="a")
    f.write_bytes(b"y" * 200)
    music_ingest.ingest_file(con, root, f)
    row = con.execute("SELECT * FROM catalog").fetchone()
    assert row["genre"] == "Dub"
    assert json.loads(row["r_mood"]) == ["mellow"]


def test_repopulate_puts_the_row_back_to_what_the_file_says(con, root, fake_ffmpeg):
    f = root / "A/B/01 - One.mp3"
    music_ingest.ingest_file(con, root, f)
    music_db.curate(con, "track", "A/B/01 - One.mp3", {"genre": "Dub", "mood": ["mellow"]}, by="a")
    assert music_ingest.ingest_file(con, root, f, repopulate=True) == "repopulated"
    row = con.execute("SELECT * FROM catalog").fetchone()
    # The album still says Rock, and the track no longer overrides it.
    assert row["genre"] == "Rock" and row["r_mood"] is None


def test_a_bumped_measure_version_reopens_everything(con, root, fake_ffmpeg, monkeypatch):
    f = root / "A/B/01 - One.mp3"
    music_ingest.ingest_file(con, root, f)
    fake_ffmpeg.clear()
    monkeypatch.setattr(music_ingest, "MEASURE_VERSION", music_ingest.MEASURE_VERSION + 1)
    assert music_ingest.ingest_file(con, root, f) == "remeasured"
    assert fake_ffmpeg != []


def test_non_audio_is_skipped(con, root, fake_ffmpeg):
    assert music_ingest.ingest_file(con, root, root / "A/B/cover.jpg") == "skipped"
    assert con.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 0


def test_collection_ingest_is_just_the_loop(con, root, fake_ffmpeg):
    counts = music_ingest.ingest_tree(con, root, workers=2)
    assert counts["new"] == 3
    assert counts["unchanged"] == 0
    again = music_ingest.ingest_tree(con, root, workers=2)
    assert again == {**again, "new": 0, "unchanged": 3}


def test_a_deleted_file_is_forgotten(con, root, fake_ffmpeg):
    music_ingest.ingest_tree(con, root, workers=2)
    (root / "A/C/01 - Three.mp3").unlink()
    counts = music_ingest.ingest_tree(con, root, workers=2)
    assert counts["forgotten"] == 1
    assert con.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 2


def test_one_unreadable_file_does_not_stop_a_sweep(con, root, monkeypatch):
    bad = str(root / "A/B/02 - Two.mp3")

    def measure(path, timeout=300.0):
        if str(path) == bad:
            raise OSError("boom")
        return {"lufs": -14.0, "true_peak_dbfs": -0.5, "lra_lu": 6.0}

    monkeypatch.setattr(music_ingest, "measure", measure)
    monkeypatch.setattr(music_ingest, "probe",
                        lambda p, timeout=60.0: {"duration_s": 1.0, "tags": {}})
    counts = music_ingest.ingest_tree(con, root, workers=2)
    assert counts["new"] == 2 and counts["failed"] == 1


def test_cruft_is_reported_not_touched(root):
    found = music_ingest.sweep_cruft(root)
    assert sorted(Path(p).name for p in found["junk"]) == [".DS_Store", "lamer.sh"]
    assert (root / "A/B/.DS_Store").exists()


def test_percent_encoded_names_are_reported_separately(tmp_path, fake_ffmpeg):
    r = tmp_path / "audio"
    p = r / "AC%2fDC/Back in Black/01 - Hells Bells.mp3"
    p.parent.mkdir(parents=True); p.write_bytes(b"x")
    assert music_ingest.sweep_cruft(r)["percent_encoded"] == [
        "AC%2fDC/Back in Black/01 - Hells Bells.mp3"
    ]


def test_a_percent_encoded_artist_lands_decoded_in_the_catalogue(tmp_path, fake_ffmpeg):
    r = tmp_path / "audio"
    p = r / "AC%2fDC/Back in Black/01 - Hells Bells.mp3"
    p.parent.mkdir(parents=True); p.write_bytes(b"x")
    con = music_db.connect(tmp_path / "library.db")
    try:
        music_ingest.ingest_file(con, r, p)
        row = con.execute("SELECT * FROM catalog").fetchone()
        assert row["artist"] == "AC/DC"          # decoded for use
        assert row["path"].startswith("AC%2fDC/")  # but the real path is preserved
    finally:
        con.close()


@pytest.mark.parametrize("raw,expect", [("3", 3), ("3/12", 3), ("1999-04-01", 1999), (None, None), ("", None)])
def test_id3_numbers_survive_their_many_renderings(raw, expect):
    assert music_ingest._int(raw) == expect
