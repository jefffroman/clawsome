"""Speech between the songs — rendering a DJ's links to WAV files.

A **link** sits between two tracks, so it is always both a back-announce and an
intro; only the head and the tail of a set are one-sided. It is queued as a
playlist entry of its own rather than mixed over the music, which is what makes
the whole-song rule structural: an entry cannot overlap another entry, so there
is nothing to enforce and no ducking to get wrong.

Three things happen here and nothing else. Piper turns text into a WAV, ffmpeg
sets its level and matches the library's format, and the result is measured to
confirm it landed. No mpv, no playlist, and no
tool-shaped strings: what a refusal reads like is the caller's business, the
same way :class:`claw.music.Player` leaves that to :mod:`claw.tools.music`.

**The Piper CLI, not wyoming-piper.** ``length_scale`` is a daemon *CLI flag*,
so a streaming voice path has one global speaking rate shared by every device,
and giving one device its own rate means running a second piper daemon on its
own port. That constraint belongs to the streaming path alone: a file render
never touches the daemon, so the DJ gets its own voice, its own rate and its
own loudness, with no contention against the live TTS and nothing new to run.

**Loudness: two axes, and only one of them is ours to assert.** A ``tanh``
soft-clip is available (``drive``, the same curve as the voice-gateway's
``loudness.apply_drive``, reached through ffmpeg so claw needs no numpy and no
second copy of it).

**``drive`` shapes the voice; it never moves its level.** Saturation makes
speech louder as well as denser, and those are two different decisions. The
level was chosen by ear with saturation off — links spliced between a dynamic
record, a mid one and a brickwalled one, and the voice was never buried — so a
saturated link is measured against the same text rendered flat, and the
difference is cut back off (:meth:`Announcer._level_match_db`). Measured on
the real chain — the same Piper render finished at each drive, 1.2 to 2.0 — a
matched link lands within 0.2 LU of flat, with 2.5-5.4 dB *more* peak headroom.
It has to be the same render: Piper samples noise, so two renders of one line
differ by up to ~0.7 LU on their own, and comparing across them measures Piper,
not the match. Because the level is pinned, a ladder of drives compares only
what saturation does to the voice, which is the only question it should answer.

A flat Piper render measures ~-19 to -22 LUFS — quieter than a typical record,
since LUFS is how loud something sounds — and speech (~1.2 LU of loudness
range) is already flatter than any music it sits beside. An earlier version of
this note argued that a LUFS gap "is not level" in this collection. That was
wrong — see the collection's loudness note — and the level stands on the
listening, not on that argument.

**``target_lufs`` is the level lever**, and the only one: a link is cut or
raised to it on the finished clip, the raise bounded by the peak ceiling, never
limited. Unset, a link keeps the flat render's own level. Set, it replaces the
level-match — the target pins the level whatever the drive — and it evens out
the render-to-render spread as well.

What is asserted automatically is the axis entirely in our gift: a rendered
clip must not clip.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from claw import music_ingest
from claw.config import DjConfig

# Borrowed rather than declared again: two paths to the same subprocess helper
# is two things to keep in step, and the one that is wrong is always the one
# nobody is looking at. (Player borrows blueutil from the bluetooth block for
# the same reason.)
from claw.music import _exec

log = logging.getLogger(__name__)

# Piper renders **one clip per input line**, so a newline inside a link would
# silently become two clips — and every clip after it would then carry the
# words meant for the seam before it. That is the one input in this module that
# can produce a plausible-sounding set saying entirely the wrong things, so it
# is flattened rather than rejected.
_WHITESPACE = re.compile(r"\s+")

# Piper names files in `-d` mode with a monotonic counter, which is what makes
# input order recoverable. Sorted **numerically**, not lexicographically: the
# counter gains a digit eventually, and a lexicographic sort silently reorders
# the set on the day it does.
_STAMP = re.compile(r"^\d+$")


class RenderFailed(RuntimeError):
    """A link could not be rendered. The message is safe to show a human."""


@dataclass(frozen=True)
class Clip:
    """One rendered link, ready to be queued as a playlist entry."""
    path: Path
    text: str
    duration_s: float | None = None
    # Recorded, not asserted on: how loud a link should sit against music is an
    # ear question. Kept because comparing a ladder of drives, or noticing that
    # a voice has moved, needs a number rather than a memory of last time.
    lufs: float | None = None
    true_peak_dbfs: float | None = None


def _one_line(text: str) -> str:
    """Collapse a link to a single line. See ``_WHITESPACE`` for why."""
    return _WHITESPACE.sub(" ", text).strip()


def _stamp_key(path: Path) -> tuple[int, int | str]:
    """Sort key preserving Piper's emission order.

    Anything that is not a bare counter sorts after everything that is, by name,
    rather than raising — the count check in :meth:`Announcer.render` is the
    real guard, and it gives a far better message than a ``ValueError`` from a
    sort key.
    """
    return (0, int(path.stem)) if _STAMP.match(path.stem) else (1, path.stem)


class Announcer:
    """Renders links. Holds no state beyond its config; safe to build per call."""

    def __init__(self, cfg: DjConfig) -> None:
        self.cfg = cfg

    # --- identity -------------------------------------------------------

    def is_clip(self, path: str | Path | None) -> bool:
        """Whether *path* is one of our clips rather than a track.

        Path is the whole mechanism, deliberately — no metadata, no registry,
        nothing to keep in step. It is what lets the player recognise a link in
        a playlist it did not build: to report "the DJ is talking" instead of an
        unrecognised file, and to drop a trailing sign-off when a set is
        appended to rather than leaving a goodbye in the middle of an evening.
        """
        if not path:
            return False
        try:
            candidate = Path(path).resolve()
            root = self.cfg.render_dir.resolve()
        except OSError:
            return False
        return candidate.is_relative_to(root)

    # --- rendering ------------------------------------------------------

    async def render(self, lines: Sequence[str]) -> list[Clip]:
        """Render every link in a run sheet. All of them, or none.

        One Piper process for the whole sheet: measured at 0.845 s for four
        links against 2.54 s for four separate invocations, because ~0.42 s of
        each invocation is ONNX model load. Everything after that is per-clip
        and costs ~0.09 s.

        Raises :class:`RenderFailed` rather than returning a partial set. A set
        is refused whole for the same reason a missed query refuses it: the
        words assert what plays next, so half a set is a DJ announcing records
        that never arrive.
        """
        texts = [_one_line(t) for t in lines]
        if not texts:
            return []
        if blank := [i for i, t in enumerate(texts) if not t]:
            raise RenderFailed(
                f"link {blank[0] + 1} is empty — a link that says nothing should "
                "be left out of the run sheet, not rendered as silence"
            )
        if len(texts) > self.cfg.max_links:
            raise RenderFailed(
                f"{len(texts)} links is more than the {self.cfg.max_links} allowed. "
                "A set that talks this often is a worse set; say less, less often."
            )
        for what, binary in (("piper", self.cfg.piper_binary),
                             ("the voice model", self.cfg.voice_model)):
            if not binary.is_file():
                raise RenderFailed(
                    f"cannot render: {what} is missing at {binary}. It belongs to "
                    "the voice stack; nothing was queued."
                )

        session = self._session_dir()
        raw = session / "raw"
        try:
            await self._synthesize(texts, session, raw)
            wavs = sorted(raw.glob("*.wav"), key=_stamp_key)
            if len(wavs) != len(texts):
                # Never pair up whatever arrived. A short render would shift
                # every link onto the wrong seam, which sounds deliberate.
                raise RenderFailed(
                    f"piper produced {len(wavs)} clip(s) for {len(texts)} link(s); "
                    "refusing to guess which words belong to which seam"
                )
            clips = await asyncio.gather(*(
                self._finish(src, session / f"{i:02d}.wav", text)
                for i, (src, text) in enumerate(zip(wavs, texts), 1)
            ))
        except RenderFailed:
            shutil.rmtree(session, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(session, ignore_errors=True)
            raise RenderFailed(f"could not render the links: {exc}") from exc
        shutil.rmtree(raw, ignore_errors=True)
        return list(clips)

    def _session_dir(self) -> Path:
        """A directory per set. The suffix is not decoration — two sets can
        start inside the same second, and a collision would have one overwrite
        the other's clips while both were queued."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = self.cfg.render_dir / f"{stamp}-{uuid4().hex[:6]}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def _synthesize(self, texts: Sequence[str], session: Path, raw: Path) -> None:
        """One Piper invocation for the whole run sheet.

        Fed from a file rather than stdin so the sheet stays on disk beside the
        clips: when a set sounds wrong, the first question is always what it was
        actually asked to say.
        """
        sheet = session / "lines.txt"
        sheet.write_text("\n".join(texts) + "\n", encoding="utf-8")
        rc, out, err = await _exec(
            self.cfg.piper_binary,
            [
                "-m", self.cfg.voice_model,
                "-i", sheet,
                "-d", raw,
                "--output-dir-naming", "timestamp",
                "--length-scale", self.cfg.length_scale,
                "--sentence-silence", self.cfg.sentence_silence,
            ],
            # Renders run ~21x realtime and the model load is ~0.4 s, so this is
            # a backstop against a wedged process, not a budget.
            timeout=30.0 + 10.0 * len(texts),
        )
        if rc != 0:
            raise RenderFailed(f"piper failed ({rc}): {err or out or 'no output'}")

    async def _finish(self, src: Path, dst: Path, text: str) -> Clip:
        """Saturate, set the level, pad, match the library's format, then measure."""
        filters = []
        gain = 0.0
        if self.cfg.drive > 1.0:
            # Only when actually asked for. ffmpeg's asoftclip at param=1.0 is
            # NOT a passthrough — it still applies the tanh curve, measured at
            # -0.4 LU and 2.4 dB off the peaks. The voice-gateway's apply_drive
            # returns early below unity; this filter has no such courtesy, so
            # "off" has to mean "not in the chain".
            filters.append(f"asoftclip=type=tanh:param={self.cfg.drive:g}")
        if self.cfg.target_lufs is not None:
            # A target pins the level outright, so there is nothing for the
            # flat level-match to protect.
            gain = await self._target_gain_db(src, dst, filters)
        elif filters:
            gain = await self._level_match_db(src, dst, filters[0])
        if round(gain, 2):
            filters.append(f"volume={gain:.2f}dB")
        if self.cfg.pad_ms:
            # Breathing room on either side of the cut into music. NOT ducking,
            # which was considered and rejected: this moves nothing's level.
            filters.append(f"adelay=all=1:delays={self.cfg.pad_ms}")
            filters.append(f"apad=pad_dur={self.cfg.pad_ms / 1000:g}")
        argv = ["-y", "-loglevel", "error", "-i", src]
        if filters:
            argv += ["-af", ",".join(filters)]
        argv += ["-ar", self.cfg.sample_rate, "-ac", self.cfg.channels, dst]
        rc, out, err = await _exec(music_ingest.FFMPEG, argv, timeout=60.0)
        if rc != 0 or not dst.is_file():
            raise RenderFailed(f"ffmpeg failed ({rc}): {err or out or 'no output'}")

        # Both are pure ffmpeg wall-clock, so they belong on a thread — the same
        # reason the collection sweep measures on a thread pool.
        measured, probed = await asyncio.gather(
            asyncio.to_thread(music_ingest.measure, dst),
            asyncio.to_thread(music_ingest.probe, dst),
        )
        clip = Clip(
            path=dst, text=text, duration_s=probed.get("duration_s"),
            lufs=measured.get("lufs"), true_peak_dbfs=measured.get("true_peak_dbfs"),
        )
        self._check_peak(clip)
        return clip

    async def _level_match_db(self, src: Path, dst: Path, soften: str) -> float:
        """The cut that puts a saturated link back at its flat loudness.

        Saturation changes two things at once — how the voice sounds and how
        loud it is — and only the first is what ``drive`` is for. The level was
        chosen with it off; raising it would quietly move the voice against the
        music. So the same text is measured flat and saturated, and the
        difference comes back off. Saturation only ever adds loudness, so this
        is always a cut and can never clip; it lowers the peaks further still.

        Measured in the source's own format: channel count and rate change a
        LUFS reading, but by the same amount for both, so the difference holds
        once the final chain upmixes and resamples. Unmeasurable returns 0 and
        says so — the link then plays saturated at the louder level, which is
        worth a warning and not worth losing the set over.
        """
        probe = dst.with_name(dst.stem + ".level-probe.wav")
        try:
            rc, out, err = await _exec(
                music_ingest.FFMPEG,
                ["-y", "-loglevel", "error", "-i", src, "-af", soften, probe],
                timeout=60.0,
            )
            if rc != 0 or not probe.is_file():
                raise RenderFailed(f"ffmpeg failed ({rc}): {err or out or 'no output'}")
            flat, saturated = await asyncio.gather(
                asyncio.to_thread(music_ingest.measure, src),
                asyncio.to_thread(music_ingest.measure, probe),
            )
        finally:
            probe.unlink(missing_ok=True)
        if flat.get("lufs") is None or saturated.get("lufs") is None:
            log.warning("dj: could not measure %s flat and saturated; queuing it "
                        "un-matched, so it will play louder than the flat level", dst.name)
            return 0.0
        return min(0.0, flat["lufs"] - saturated["lufs"])

    async def _target_gain_db(self, src: Path, dst: Path, timbre: list[str]) -> float:
        """The gain that puts a link at ``target_lufs``, bounded by its peak.

        The music's rule (:func:`claw.music.gain_db`): a louder link is cut to
        the target, a quieter one raised toward it only until its true peak
        reaches ``max_true_peak_dbfs``. Going further would need a limiter,
        which changes the voice rather than its level.

        Measured on a probe in the FINAL format, not the source's: this is an
        absolute reading, and upmixing Piper's mono to the library's stereo
        can move it — unlike the level-match, which compares two readings in
        the same format and so is immune. Everything after the probe is linear
        (gain, silence), so the clip lands where the probe says. Unmeasurable
        returns 0 and says so: the link plays at its render level.
        """
        probe = dst.with_name(dst.stem + ".level-probe.wav")
        try:
            argv = ["-y", "-loglevel", "error", "-i", src]
            if timbre:
                argv += ["-af", ",".join(timbre)]
            argv += ["-ar", self.cfg.sample_rate, "-ac", self.cfg.channels, probe]
            rc, out, err = await _exec(music_ingest.FFMPEG, argv, timeout=60.0)
            if rc != 0 or not probe.is_file():
                raise RenderFailed(f"ffmpeg failed ({rc}): {err or out or 'no output'}")
            got = await asyncio.to_thread(music_ingest.measure, probe)
        finally:
            probe.unlink(missing_ok=True)
        lufs, peak = got.get("lufs"), got.get("true_peak_dbfs")
        if lufs is None or peak is None:
            log.warning("dj: could not measure %s; queuing it at its render level, "
                        "not at %g LUFS", dst.name, self.cfg.target_lufs)
            return 0.0
        want = self.cfg.target_lufs - lufs
        if want <= 0:
            return want
        return max(0.0, min(want, self.cfg.max_true_peak_dbfs - peak))

    def _check_peak(self, clip: Clip) -> None:
        """Warn when a clip is hotter than it should be. Peak, not loudness.

        The only loudness assertion made here, deliberately. Where a link
        should sit against a record is decided by ear; a true peak above the
        ceiling is unambiguous, and the plausible way to cause it is raising
        ``drive`` and not listening for what it cost.

        Warns rather than refuses: a hot clip is still a clip, and losing a set
        over a decibel would be a worse failure than the one being reported.
        """
        if clip.true_peak_dbfs is None:
            log.warning("dj: could not measure %s; queuing it unverified", clip.path.name)
            return
        if clip.true_peak_dbfs > self.cfg.max_true_peak_dbfs:
            log.warning(
                "dj: %s peaks at %+.1f dBFS, above the %+.1f ceiling, at drive %g "
                "(%.1f LUFS) — back the drive off rather than living with it",
                clip.path.name, clip.true_peak_dbfs, self.cfg.max_true_peak_dbfs,
                self.cfg.drive, clip.lufs if clip.lufs is not None else float("nan"),
            )

    # --- housekeeping ---------------------------------------------------

    def reap(self) -> int:
        """Delete abandoned sessions. Returns how many went.

        **By age only.** A queue that was replaced leaves its clips behind and
        nothing will ever come for them, but a queue that is still *playing*
        holds paths into a session that looks equally idle from here — so
        clearing "everything but the current set" would delete links out from
        under a live playlist. Age is the one signal that cannot get that wrong:
        the queue is capped at 200 tracks, around thirteen hours of music,
        comfortably inside the default day.
        """
        cutoff = time.time() - self.cfg.keep_hours * 3600
        gone = 0
        try:
            sessions = list(self.cfg.render_dir.iterdir())
        except OSError:
            return 0
        for session in sessions:
            try:
                if not session.is_dir() or session.stat().st_mtime >= cutoff:
                    continue
                shutil.rmtree(session)
                gone += 1
            except OSError as exc:
                # Housekeeping must never be the reason a set does not play.
                log.warning("dj: could not reap %s: %s", session, exc)
        if gone:
            log.info("dj: reaped %d abandoned session(s) from %s", gone, self.cfg.render_dir)
        return gone
