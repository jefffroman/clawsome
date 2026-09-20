# Audio — Bluetooth and music

Reference for the two audio families: what the tools promise, and the
reasoning behind the parts that are easy to get wrong. Their configuration
keys live in [Configuration](configuration.md#bluetooth) — this is the
behaviour those keys select.

## Bluetooth host requirements

**A logged-in console session.** Every call runs inside the console GUI session
as a one-shot launchd job, not in the gateway's own daemon context, because the
two are not equivalent — see [Extending](extending.md#tools-that-need-a-gui-session)
for the measurement and the reasoning. If no console session exists, the tools
say so rather than reporting a Bluetooth fault.

**A TCC grant.** On modern macOS, reaching IOBluetooth at all requires
`kTCCServiceBluetoothAlways`. Without it `blueutil` does not error — it reports
the controller as powered off and lists no devices, which is indistinguishable
from a radio that really is off. The tools disambiguate by cross-checking
`system_profiler`, which reads the IORegistry and needs neither a session nor a
grant, and say "grant, not radio" when the two disagree. Note that such a grant
is typically pinned to the binary's code signature, so upgrading `blueutil`
can silently revoke it.


## Music

The tools are built from the `music:` block; these are the behaviours a
caller has to know about, and the reasoning behind the ones that are easy
to get wrong.

### `music_dj` — speech between the songs

The agent picks a set (usually from `music_candidates`) and says something
between the records. The call is a **run sheet**: entries with `say`, `play` (a
`t:` track handle, or an `al:` record handle) or both, `say` spoken before what
it plays; a `say` alone at the start is an opening, at the end a sign-off.

- **Insert, never duck.** A link is rendered to a WAV and queued as its own
  playlist entry, so it *cannot* overlap music — the whole-song rule is
  structural.
- **A record is one entry.** An `al:` entry plays the whole album, in its own
  order, codas kept, at one album gain — exactly as `music_play` plays it —
  while single tracks in the same set keep per-track gain. So a `say` on it
  introduces the record, and nothing can be queued inside its running order.
  Where links belong around a record is taste, left to the agent's skill.
  Entries take handles, never free text, so an entry cannot mis-resolve.
- **All or nothing.** Every handle resolves and every clip renders before the
  first entry is queued; one failure queues nothing, because the words assert
  what plays next.
- **`minutes` is checked before rendering**, within 10%, so a set of the wrong
  length costs milliseconds rather than a synthesis.
- **Appending drops a trailing sign-off** — unless it is being spoken — so a
  goodbye never lands mid-evening. `music_status` reports what is next and what
  the queue ends with, and recognises a link as a link.
- **`target_lufs` sets the voice level; links then play at unity.** Measured on
  the finished clip in the library's format: a louder link is cut to it, a
  quieter one raised toward it only as far as `max_true_peak_dbfs` allows — the
  music's own rule, no limiter. It also evens out Piper's render-to-render
  spread. Unset, a link keeps the flat render's level (~-20 LUFS). Independent
  of `loudness.target_lufs` and of `normalize` — choose it against where your
  music actually plays, which with normalisation on is near
  `loudness.target_lufs`.
- **`drive` shapes the voice, never its level.** Saturation adds loudness as
  well as density. With `target_lufs` set, the target pins the level anyway;
  without it, each saturated link is measured against the same Piper render
  flat and the difference is cut back off (never boosted): within 0.2 LU of
  flat, with more peak headroom. A ladder of drives therefore compares only
  timbre. Piper samples noise — two renders of one line differ by up to
  ~0.7 LU — so compare on one render.

`dj:` keys: `enabled`, `piper_binary`, `voice_model`, `render_dir`,
`length_scale`, `sentence_silence`, `drive` (1.0 = off; timbre only),
`target_lufs` (null = the render's own level), `max_true_peak_dbfs`,
`pad_ms`, `sample_rate`, `channels`, `max_links`, `keep_hours`. Rendering uses
the Piper CLI, so the DJ's voice and speaking rate are independent of any live
TTS daemon; one process per run sheet (model load dominates).

### Search and candidates — opposite contracts on purpose

**`music_search` looks and never acts, and hides nothing.** It matches artist,
album, title, genre, and words in notes and moods, and returns artists, albums
and tracks. Interludes, spoken word, recently played and annotated records all
appear, labelled rather than removed; the only narrowing is what the caller
passes, and the reply restates it. A display cap, when hit, states the true
total. A miss is a miss — near misses appear only when nothing matched, marked
as not being matches. Notes and moods are attributed to the level they were
written at, so an artist-level note is one artist row, not one per track.
**`scope` returns only one kind, and never finds less:** every hit is reported
as that kind — an artist that matches comes back as their tracks or records, a
track as its record or artist — with the kind's own matches ranked first. It is
not a narrowing (genre and years are), so it cannot turn a match into a miss.

**Notes add up across levels; everything else takes the most specific.** A
track's mood overrides its album's, but a track's note is *added to* its
album's and its artist's — each level carries its own facts, and all of them
apply, lowest level first. List rows show each level's note cut to 90
characters; **passing a handle as the query** returns the detail view — every
level's own curation in full, and for an album its tracks, for an artist their
records.

**`music_candidates` filters on purpose, and counts what it filtered.** It
offers a pool about `pool_factor`× the requested length, leaving out recent
plays, interludes and `exclude_genres`, and its reply says how many of each.
The pool is taken from the top of an **interleave** — round-robin across
artists, then across each artist's records, never-queued tracks first — so any
prefix is as varied as the request allows: a broad genre yields one track each
from many artists, a single artist is spread across their records, and no
per-artist cap needs tuning. Choosing from the pool is left to the agent: the
catalogue knows what exists, how long it is, and when it last played, but not
what suits an evening.

If the set-builder were the only lookup, "not in the pool" would read as "not in
the library". Keeping the two apart — and having each say which it is — is the
point.

**`music_history`** reads the curation log — every annotation, correction,
rename and merge, with who, when and what it replaced — for one artist, album or
track (with what is inside it and the levels above it), or the most recent edits
everywhere. The log also records the operations that *discard* curation (a
repopulate, or a file gone from disk), with what was lost, so it can be put
back. There is no stored "last curated" field; the log is the record.

**Handles** carry an answer from one tool into the next exactly: `ar:<id>`,
`al:<id>`, `t:<hash of the path>`. `music_play` and `music_curate` resolve a
handle exactly and refuse an unknown one rather than matching the nearest thing;
`music_play(handles=[...])` plays a chosen set in the given order, all or
nothing.

**Outputs are a named list, and that is the point.** The agent asks for
`room-a`; it never handles a CoreAudio device string, a MAC address, or a
filesystem path. Matching happens inside the tool, over the catalogue — a real
music collection is full of apostrophes, accents and percent-encoded slashes,
and a path composed by a language model onto a command line is the worst place
to discover that.

An output with **no** `bluetooth_address` is a wired or built-in device with no
connect step. The key being absent is the signal; it is never null-for-none.

### Playback routing never moves the system default

mpv is told `audio-device` per instance, so playback cannot disturb anything
else on the host. That began as isolation and became a safety property: only
explicitly-targeted audio should reach a speaker somebody is sitting next to.

Which creates one non-obvious obligation. **macOS makes a Bluetooth audio
device the system default output the moment it connects** — so the reconnect
that `ensure_output` performs would itself re-point system audio, including
alert sounds with their own volume, at that speaker. The reconnect therefore
captures the current default before connecting and restores it afterwards.

A speaker that cannot be woken produces a refusal naming it and why. Playback
never falls back to a different output: audio arriving in a room nobody asked
about is worse than audio not arriving.

### Loudness — aim at the mode, cut freely, boost only into headroom

Three measurements, each answering one question, and it pays to keep them
apart:

| Metric | Tells you | Does not tell you |
|---|---|---|
| Integrated LUFS | how loud a track **sounds** | how much gain it can take |
| True peak | **headroom** — gain available before it clips | how loud it sounds |
| LRA | dynamics | either of the above |

In a collection that spans the loudness war the first two come apart. Measured
across one real collection of 4,500 tracks: integrated loudness spans 31.9 dB,
yet **93% of tracks peak within 3 dB of full scale** and 57% already peak above
it. The loud records are loud *by compression*. Two traps follow, and both are
easy to walk into:

- **A LUFS gap is not gain you can apply.** Gain moves loudness and peak
  together, so closing a gap *upward* needs headroom the quiet, dynamic
  records mostly do not have. Upward means a limiter, which changes the music
  rather than its level. **Downward is free.**
- **Nor does matched peak mean matched loudness.** Peak-matched masters that are
  more compressed sound louder — that is the whole mechanism of the loudness
  war. A LUFS gap between records is a level difference a listener hears.

So, with `normalize` on:

- **The target is the collection's mode**, not its mean or a broadcast
  standard. At the mode the typical record is untouched, so switching
  normalisation on does not make the room quieter. The music CLI's `stats` prints
  the measured mode beside the configured target and flags drift.
- **Louder tracks are cut to the target.**
- **Quieter tracks are boosted toward it only as far as their own true peak
  allows**, up to `boost_ceiling_dbtp`. In a collection like the one above that
  is often not far — the median boost was +0.5 dB — but together with the cuts
  it took the 5th–95th percentile spread of a shuffle from 12.8 LU to 4.6.
- **A shuffle gets per-track gain; an album in order gets one figure** for the
  whole record — energy-weighted loudness, headroom from its loudest peak — so
  its own quiet/loud relationships survive.
- An **unmeasured** track is assumed loud (`assumed_lufs`) and never boosted.

Why the ceiling defaults to 0 dBTP rather than the textbook −1 (the margin for
lossy codecs): the ceiling bounds *our boost only*. Where most of a collection
already peaks above 0 at unity, −1 protects only the boosted tracks — which
would then be the cleanest ones playing — while roughly halving how many get
any boost at all.

⚠ **mpv's `volume` is cubic, not a percentage of amplitude** — 50 is −18 dB,
not −6. Gain is therefore sent as the per-entry `volume-gain` option, which is
in dB. And mpv's `--volume-gain-max` does **not** cap a per-file option, so the
never-past-the-peak rule is enforced in claw, not by mpv. A dB-to-`volume`
conversion done as if it were linear makes every cut three times deeper than
intended; that bug once produced a listening verdict against normalisation
that had to be withdrawn.

There is still **no per-output volume**. A standing attenuation happens before
a lossy encoder and the amplifier then raises music and codec noise together.
Normalisation is not that: it brings a loud record down to where the typical
one already enters the encoder.

ReplayGain tags are the portable alternative and are not used: they must be
written into the files, and a catalogue can hold the same measurement without
modifying anything the operator owns.

**Album furniture** — a track far quieter than its own record and short
(`furniture_below_album_lu`, `furniture_max_s`). Codas, segues, spoken intros:
under 1% of a collection, quiet on purpose. Dropped from a *shuffle*, where they
are a dead half-minute; always kept in album order, where they are the joins.
Never lifted on their own.

### The catalogue

`db_path` is SQLite (WAL). Artists contain albums contain tracks, each a row
with an id, each carrying the same curatable fields. It holds the loudness
measurement, what the tags said, and **curation** — mood, energy, notes,
corrections — resolved most specific first: the track's value, else the
album's, else the artist's. A track can override its album's artist, which is
how a compilation credits who actually played.

> ⚠ **This file is state, not cache. Put it where your backups reach.**
> Everything measured could be rebuilt from the files in minutes, but curation
> exists nowhere else. Do not put it beside the IPC socket in a runtime
> directory; that placement quietly says "disposable", and it is not.

Ingest is per file; a collection sweep is a loop over the same function. A file
whose size, mtime and measure version all match is never opened, so a re-sweep
of a large library costs seconds. See
[Extending](extending.md#curation-must-outlive-what-produced-it) for the write
rules, which are the part worth copying.
