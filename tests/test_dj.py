"""claw.dj — rendering the links a DJ speaks between songs.

The subprocesses are faked. What is worth testing here is not that Piper works
but that this module never mis-pairs words with seams, never leaves a half
rendered set behind, and never deletes clips a live playlist is still holding.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from claw import dj
from claw.config import DjConfig


def _cfg(tmp_path, **over) -> DjConfig:
    """A config whose binaries exist, so the missing-binary guard is not what
    every other test ends up exercising."""
    piper = tmp_path / "piper"
    piper.write_text("#!/bin/sh\n")
    model = tmp_path / "voice.onnx"
    model.write_text("not really a model")
    return DjConfig(
        enabled=True,
        piper_binary=piper,
        voice_model=model,
        render_dir=tmp_path / "dj",
        **over,
    )


def _fake_synth(counter_start: int = 100, count: int | None = None):
    """Stand in for Piper: writes one file per line, named with an ascending
    counter exactly as ``-d --output-dir-naming timestamp`` does."""
    async def synth(self, texts, session, raw):
        raw.mkdir(parents=True, exist_ok=True)
        n = len(texts) if count is None else count
        for i in range(n):
            (raw / f"{counter_start + i}.wav").write_bytes(b"RIFF")
    return synth


@pytest.fixture
def rendered(monkeypatch):
    """Fake both subprocess stages; ``_finish`` just copies the source path."""
    monkeypatch.setattr(dj.Announcer, "_synthesize", _fake_synth())

    async def finish(self, src, dst, text):
        dst.write_bytes(src.read_bytes())
        return dj.Clip(path=dst, text=text, duration_s=1.0, lufs=-14.0)

    monkeypatch.setattr(dj.Announcer, "_finish", finish)


# --- the newline hazard ------------------------------------------------

@pytest.mark.parametrize("raw,flat", [
    ("that was one.\nhere is the next", "that was one. here is the next"),
    ("  padded  ", "padded"),
    ("tabs\tand\r\nnewlines", "tabs and newlines"),
    ("already one line", "already one line"),
])
def test_links_are_flattened_to_one_line(raw, flat):
    # Piper renders one clip per input LINE, so an embedded newline would split
    # a link in two and shift every later link onto the wrong seam.
    assert dj._one_line(raw) == flat


def test_flattening_survives_the_render(tmp_path, rendered):
    a = dj.Announcer(_cfg(tmp_path))
    clips = asyncio.run(a.render(["first\nlink", "second"]))
    assert [c.text for c in clips] == ["first link", "second"]


def test_synthesize_writes_one_sheet_line_per_link(tmp_path, monkeypatch):
    """The real ``_synthesize``, against a faked subprocess.

    Worth testing directly rather than through a fake: the sheet is what Piper
    splits on, so "one line per link" is the invariant the whole ordering
    contract rests on — and the sheet staying on disk beside the clips is how
    anyone answers "what was it actually asked to say".
    """
    seen = {}

    async def fake_exec(program, args, timeout):
        seen["program"], seen["args"] = program, [str(a) for a in args]
        return 0, "", ""

    monkeypatch.setattr(dj, "_exec", fake_exec)
    cfg = _cfg(tmp_path, length_scale=0.9, sentence_silence=0.25)
    session = cfg.render_dir / "session"
    session.mkdir(parents=True)

    asyncio.run(dj.Announcer(cfg)._synthesize(
        ["that was one. here is two", "and three"], session, session / "raw"
    ))

    sheet = session / "lines.txt"
    assert sheet.read_text().splitlines() == [
        "that was one. here is two", "and three",
    ]
    assert seen["program"] == cfg.piper_binary
    args = seen["args"]
    assert args[args.index("-i") + 1] == str(sheet)
    assert args[args.index("-m") + 1] == str(cfg.voice_model)
    assert args[args.index("-d") + 1] == str(session / "raw")
    # Per-render, and not what the streaming daemon was started with.
    assert args[args.index("--length-scale") + 1] == "0.9"
    assert args[args.index("--sentence-silence") + 1] == "0.25"


def test_synthesize_surfaces_piper_stderr(tmp_path, monkeypatch):
    async def fake_exec(program, args, timeout):
        return 1, "", "no such voice"

    monkeypatch.setattr(dj, "_exec", fake_exec)
    cfg = _cfg(tmp_path)
    session = cfg.render_dir / "session"
    session.mkdir(parents=True)
    with pytest.raises(dj.RenderFailed, match="no such voice"):
        asyncio.run(dj.Announcer(cfg)._synthesize(["hello"], session, session / "raw"))


# --- ordering ----------------------------------------------------------

def test_clips_sort_numerically_not_lexicographically(tmp_path):
    # The counter gains a digit eventually; a lexicographic sort would put
    # 1000 before 999 and silently reorder the whole set on that day.
    paths = [tmp_path / f"{n}.wav" for n in (999, 1000, 1001, 99)]
    assert [p.stem for p in sorted(paths, key=dj._stamp_key)] == \
        ["99", "999", "1000", "1001"]


def test_unexpected_names_sort_last_without_raising(tmp_path):
    paths = [tmp_path / "zz.wav", tmp_path / "7.wav"]
    assert [p.stem for p in sorted(paths, key=dj._stamp_key)] == ["7", "zz"]


def test_render_preserves_run_sheet_order(tmp_path, rendered):
    a = dj.Announcer(_cfg(tmp_path))
    lines = [f"link {i}" for i in range(1, 6)]
    clips = asyncio.run(a.render(lines))
    assert [c.text for c in clips] == lines
    assert [c.path.name for c in clips] == \
        ["01.wav", "02.wav", "03.wav", "04.wav", "05.wav"]


# --- all of them, or none ---------------------------------------------

def test_short_render_is_refused_rather_than_paired_up(tmp_path, monkeypatch):
    # Pairing three clips onto four links would put the right words in the
    # wrong seams, which sounds deliberate and is the worst outcome available.
    monkeypatch.setattr(dj.Announcer, "_synthesize", _fake_synth(count=3))
    a = dj.Announcer(_cfg(tmp_path))
    with pytest.raises(dj.RenderFailed, match="3 clip"):
        asyncio.run(a.render([f"link {i}" for i in range(4)]))


def test_a_failed_render_leaves_nothing_behind(tmp_path, monkeypatch):
    monkeypatch.setattr(dj.Announcer, "_synthesize", _fake_synth(count=1))
    cfg = _cfg(tmp_path)
    a = dj.Announcer(cfg)
    with pytest.raises(dj.RenderFailed):
        asyncio.run(a.render(["one", "two"]))
    assert list(cfg.render_dir.iterdir()) == []


def test_an_unexpected_error_is_wrapped_and_cleaned_up(tmp_path, monkeypatch):
    async def boom(self, texts, session, raw):
        raise OSError("disk went away")
    monkeypatch.setattr(dj.Announcer, "_synthesize", boom)
    cfg = _cfg(tmp_path)
    with pytest.raises(dj.RenderFailed, match="disk went away"):
        asyncio.run(dj.Announcer(cfg).render(["one"]))
    assert list(cfg.render_dir.iterdir()) == []


# --- refusals ----------------------------------------------------------

def test_empty_run_sheet_renders_nothing(tmp_path, rendered):
    assert asyncio.run(dj.Announcer(_cfg(tmp_path)).render([])) == []


def test_a_blank_link_is_refused(tmp_path, rendered):
    a = dj.Announcer(_cfg(tmp_path))
    with pytest.raises(dj.RenderFailed, match="link 2 is empty"):
        asyncio.run(a.render(["something", "   "]))


def test_too_many_links_is_refused(tmp_path, rendered):
    a = dj.Announcer(_cfg(tmp_path, max_links=3))
    with pytest.raises(dj.RenderFailed, match="more than the 3"):
        asyncio.run(a.render([f"link {i}" for i in range(4)]))


def test_missing_binary_names_the_path(tmp_path, rendered):
    cfg = _cfg(tmp_path)
    cfg.piper_binary.unlink()
    with pytest.raises(dj.RenderFailed, match=str(cfg.piper_binary)):
        asyncio.run(dj.Announcer(cfg).render(["hello"]))


def test_missing_voice_model_names_the_path(tmp_path, rendered):
    cfg = _cfg(tmp_path)
    cfg.voice_model.unlink()
    with pytest.raises(dj.RenderFailed, match="voice model"):
        asyncio.run(dj.Announcer(cfg).render(["hello"]))


# --- telling a clip from a track --------------------------------------

def test_is_clip_recognises_our_own_renders(tmp_path, rendered):
    cfg = _cfg(tmp_path)
    a = dj.Announcer(cfg)
    (clip,) = asyncio.run(a.render(["hello"]))
    assert a.is_clip(clip.path)
    assert a.is_clip(str(clip.path))


def test_is_clip_rejects_tracks_and_nothing(tmp_path):
    a = dj.Announcer(_cfg(tmp_path))
    assert not a.is_clip(tmp_path / "music" / "Artist" / "Album" / "01 - Title.mp3")
    assert not a.is_clip(None)
    assert not a.is_clip("")


def test_is_clip_works_before_anything_is_rendered(tmp_path):
    # The render dir does not exist yet on a fresh install; status still has to
    # be able to ask the question.
    a = dj.Announcer(_cfg(tmp_path))
    assert not a.is_clip(tmp_path / "elsewhere.wav")


# --- reaping -----------------------------------------------------------

def test_reap_takes_old_sessions_and_leaves_fresh_ones(tmp_path):
    cfg = _cfg(tmp_path, keep_hours=24.0)
    old = cfg.render_dir / "20200101-000000-aaaaaa"
    new = cfg.render_dir / "20990101-000000-bbbbbb"
    for d in (old, new):
        d.mkdir(parents=True)
        (d / "01.wav").write_bytes(b"RIFF")
    stale = time.time() - 25 * 3600
    import os
    os.utime(old, (stale, stale))

    assert dj.Announcer(cfg).reap() == 1
    assert not old.exists()
    assert new.exists()


def test_reap_keeps_a_recent_session_even_though_it_looks_idle(tmp_path):
    # A queue that is still playing holds paths into a session that is
    # indistinguishable from an abandoned one from out here. Age is the only
    # signal that cannot delete links out from under a live playlist.
    cfg = _cfg(tmp_path)
    session = cfg.render_dir / "20990101-000000-cccccc"
    session.mkdir(parents=True)
    assert dj.Announcer(cfg).reap() == 0
    assert session.exists()


def test_reap_on_a_missing_render_dir_is_quiet(tmp_path):
    assert dj.Announcer(_cfg(tmp_path)).reap() == 0


def test_sessions_do_not_collide_within_one_second(tmp_path):
    a = dj.Announcer(_cfg(tmp_path))
    assert len({a._session_dir() for _ in range(5)}) == 5


# --- the loudness assertion, and the one it deliberately is not ---------

def test_a_hot_clip_warns_but_still_plays(tmp_path, caplog):
    # Peak, not loudness. Warns rather than refuses: losing a set over a
    # decibel would be worse than the fault being reported.
    a = dj.Announcer(_cfg(tmp_path, max_true_peak_dbfs=-1.0))
    with caplog.at_level("WARNING"):
        a._check_peak(dj.Clip(tmp_path / "01.wav", "hi", 1.0, -9.0, 0.4))
    assert "above the -1.0 ceiling" in caplog.text


def test_a_quiet_clip_is_not_a_fault(tmp_path, caplog):
    # A low LUFS figure says the material is dynamic, not that it is quiet, so
    # there is nothing here to warn about. Asserting on it is the trap.
    a = dj.Announcer(_cfg(tmp_path))
    with caplog.at_level("WARNING"):
        a._check_peak(dj.Clip(tmp_path / "01.wav", "hi", 1.0, -21.7, -3.1))
    assert caplog.text == ""


def test_an_unmeasurable_clip_says_so(tmp_path, caplog):
    a = dj.Announcer(_cfg(tmp_path))
    with caplog.at_level("WARNING"):
        a._check_peak(dj.Clip(tmp_path / "01.wav", "hi", 1.0, None, None))
    assert "unverified" in caplog.text


def test_saturation_is_off_by_default():
    # Off until somebody listens: the level argument for raising it does not
    # survive how this collection measures, and what remains is an ear question.
    assert DjConfig().drive == 1.0


def test_saturation_is_absent_from_the_chain_when_off(tmp_path, monkeypatch):
    # ffmpeg's asoftclip at param=1.0 still applies the tanh curve (-0.4 LU,
    # peaks 2.4 dB down), so "off" has to mean absent, not param=1.0.
    seen = {}

    async def fake_exec(program, args, timeout):
        seen["args"] = [str(a) for a in args]
        open(seen["args"][-1], "wb").write(b"RIFF")   # the output is the last arg
        return 0, "", ""

    monkeypatch.setattr(dj, "_exec", fake_exec)
    monkeypatch.setattr(dj.music_ingest, "measure",
                        lambda p, **k: {"lufs": -21.7, "true_peak_dbfs": -3.0})
    monkeypatch.setattr(dj.music_ingest, "probe", lambda p, **k: {"duration_s": 1.0})

    src = tmp_path / "in.wav"; src.write_bytes(b"RIFF")
    a = dj.Announcer(_cfg(tmp_path, drive=1.0, pad_ms=0))
    asyncio.run(a._finish(src, tmp_path / "out.wav", "hi"))
    assert "-af" not in seen["args"]

    a = dj.Announcer(_cfg(tmp_path, drive=1.75, pad_ms=0))
    asyncio.run(a._finish(src, tmp_path / "out.wav", "hi"))
    assert "asoftclip=type=tanh:param=1.75" in seen["args"]


# --- drive shapes the voice, never its level ---------------------------

def _level_rig(tmp_path, monkeypatch, *, flat, saturated, peak=-6.0):
    """Fake ffmpeg (every call recorded) and a measure that tells the flat
    source apart from the saturated probe by path."""
    calls = []

    async def fake_exec(program, args, timeout):
        args = [str(a) for a in args]
        calls.append(args)
        open(args[-1], "wb").write(b"RIFF")          # the output is the last arg
        return 0, "", ""

    def measure(p, **k):
        lufs = saturated if ".level-probe" in str(p) else flat
        return {"lufs": lufs, "true_peak_dbfs": peak}

    monkeypatch.setattr(dj, "_exec", fake_exec)
    monkeypatch.setattr(dj.music_ingest, "measure", measure)
    monkeypatch.setattr(dj.music_ingest, "probe", lambda p, **k: {"duration_s": 1.0})
    src = tmp_path / "in.wav"
    src.write_bytes(b"RIFF")
    return calls, src


def _final_filters(calls):
    final = calls[-1]
    return final[final.index("-af") + 1].split(",")


def test_saturation_is_cut_back_to_the_flat_level(tmp_path, monkeypatch):
    calls, src = _level_rig(tmp_path, monkeypatch, flat=-20.1, saturated=-18.8)
    a = dj.Announcer(_cfg(tmp_path, drive=1.2, pad_ms=0))
    asyncio.run(a._finish(src, tmp_path / "out.wav", "hi"))
    # Saturate first, then take the added loudness back off.
    assert _final_filters(calls) == ["asoftclip=type=tanh:param=1.2", "volume=-1.30dB"]


def test_level_match_never_boosts(tmp_path, monkeypatch):
    # If saturation ever measured quieter, lifting it would spend headroom the
    # flat render never had. Leave it be.
    calls, src = _level_rig(tmp_path, monkeypatch, flat=-20.0, saturated=-20.4)
    a = dj.Announcer(_cfg(tmp_path, drive=1.2, pad_ms=0))
    asyncio.run(a._finish(src, tmp_path / "out.wav", "hi"))
    assert not any(f.startswith("volume=") for f in _final_filters(calls))


def test_unmeasurable_level_match_says_so_and_still_renders(tmp_path, monkeypatch, caplog):
    calls, src = _level_rig(tmp_path, monkeypatch, flat=None, saturated=-18.8)
    a = dj.Announcer(_cfg(tmp_path, drive=1.2, pad_ms=0))
    clip = asyncio.run(a._finish(src, tmp_path / "out.wav", "hi"))
    assert clip.path.is_file()
    assert "un-matched" in caplog.text
    assert not any(f.startswith("volume=") for f in _final_filters(calls))


def test_level_probe_is_not_left_beside_the_clip(tmp_path, monkeypatch):
    # The render dir is how a playlist entry is recognised as a link, so a
    # stray probe there would be one more "clip" nobody queued.
    calls, src = _level_rig(tmp_path, monkeypatch, flat=-20.1, saturated=-18.8)
    a = dj.Announcer(_cfg(tmp_path, drive=1.2, pad_ms=0))
    asyncio.run(a._finish(src, tmp_path / "out.wav", "hi"))
    assert not list(tmp_path.glob("*.level-probe.wav"))


def test_flat_renders_skip_the_level_match_entirely(tmp_path, monkeypatch):
    calls, src = _level_rig(tmp_path, monkeypatch, flat=-20.1, saturated=-18.8)
    a = dj.Announcer(_cfg(tmp_path, drive=1.0, pad_ms=0))
    asyncio.run(a._finish(src, tmp_path / "out.wav", "hi"))
    assert len(calls) == 1                          # no probe render


# --- target_lufs sets the level --------------------------------------------

def test_there_is_no_target_by_default():
    # Unset, a link keeps the flat render's level — the pre-target behaviour.
    assert DjConfig().target_lufs is None


def test_a_quiet_link_is_raised_to_the_target(tmp_path, monkeypatch):
    calls, src = _level_rig(tmp_path, monkeypatch, flat=None, saturated=-20.2, peak=-5.6)
    a = dj.Announcer(_cfg(tmp_path, drive=1.2, target_lufs=-17.0, pad_ms=0))
    asyncio.run(a._finish(src, tmp_path / "out.wav", "hi"))
    # Timbre first, then the level. The target replaces the flat level-match:
    # one probe render, not two measurements of it.
    assert _final_filters(calls) == ["asoftclip=type=tanh:param=1.2", "volume=3.20dB"]
    assert len(calls) == 2


def test_the_raise_stops_at_the_peak_ceiling(tmp_path, monkeypatch):
    # -22.4 wants +5.4, but a -5.7 peak has only 4.7 dB to the -1 ceiling. No
    # limiter: it lands short of the target instead.
    calls, src = _level_rig(tmp_path, monkeypatch, flat=None, saturated=-22.4, peak=-5.7)
    a = dj.Announcer(_cfg(tmp_path, target_lufs=-17.0, max_true_peak_dbfs=-1.0, pad_ms=0))
    asyncio.run(a._finish(src, tmp_path / "out.wav", "hi"))
    assert _final_filters(calls) == ["volume=4.70dB"]


def test_a_loud_link_is_cut_to_the_target_whatever_its_peak(tmp_path, monkeypatch):
    calls, src = _level_rig(tmp_path, monkeypatch, flat=None, saturated=-15.0, peak=0.5)
    a = dj.Announcer(_cfg(tmp_path, target_lufs=-17.0, pad_ms=0))
    asyncio.run(a._finish(src, tmp_path / "out.wav", "hi"))
    assert _final_filters(calls) == ["volume=-2.00dB"]


def test_the_target_is_measured_in_the_final_format(tmp_path, monkeypatch):
    # An absolute reading: Piper's mono upmixed to stereo can move it, so the
    # probe carries the library's rate and channels, not the source's.
    calls, src = _level_rig(tmp_path, monkeypatch, flat=None, saturated=-20.0)
    a = dj.Announcer(_cfg(tmp_path, target_lufs=-17.0, pad_ms=0))
    asyncio.run(a._finish(src, tmp_path / "out.wav", "hi"))
    probe = calls[0]
    assert probe[-1].endswith(".level-probe.wav")
    assert probe[probe.index("-ar") + 1] == "44100" and probe[probe.index("-ac") + 1] == "2"
    assert not list(tmp_path.glob("*.level-probe.wav"))


def test_an_unmeasurable_target_probe_says_so_and_still_renders(tmp_path, monkeypatch, caplog):
    calls, src = _level_rig(tmp_path, monkeypatch, flat=None, saturated=None)
    a = dj.Announcer(_cfg(tmp_path, target_lufs=-17.0, pad_ms=0))
    clip = asyncio.run(a._finish(src, tmp_path / "out.wav", "hi"))
    assert clip.path.is_file() and "render level" in caplog.text
    assert "-af" not in calls[-1]


def test_target_lufs_parses_from_config():
    from claw.config import _parse_dj
    assert _parse_dj({"target_lufs": "-17"}).target_lufs == -17.0
    assert _parse_dj({"target_lufs": None}).target_lufs is None
