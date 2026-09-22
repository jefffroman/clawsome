from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    default_compaction_model: str
    # Per-inbound tool-use loop ceiling. The model can request tools, get
    # results back, and repeat — this caps the round-trips before claw
    # forces a stub final reply. 50 covers cron-driven research (web_search × N
    # + reads + write_file + summarize) with headroom; the ceiling is felt as a
    # turn that stops mid-investigation and answers with what it has, which is
    # quiet, so err high. Raise per deployment if needed.
    max_tool_turns: int = 50
    # Per-/api/chat-call generation cap (Ollama options.num_predict). When
    # the model hits this, claw inspects the partial: dangerous truncation
    # (unclosed code fence) → discard partial + return structured error;
    # otherwise → re-feed partial + recovery system note and call once
    # more so the model can wrap up, restart tighter, or bail. None or -1
    # disables (unbounded — original behavior, not recommended; runaway
    # reasoning traces can hang the call until request_timeout_s fires).
    num_predict: int | None = 8192
    # httpx read timeout per /api/chat call. Should comfortably exceed the
    # worst-case wall time for one num_predict-bounded generation on the
    # slowest agent's model. Bumped 2026-05-03 from 900 to 1800 to give
    # 16K-cap turns on qwen3.5:122b (persona-1) headroom; recovery branch
    # double-budgets a length-truncated turn (initial + recovery), each
    # call independently bounded by this timeout.
    request_timeout_s: float = 1800.0


@dataclass(frozen=True)
class RelevanceConfig:
    """The per-turn relevance gate over retrieval candidates.

    Each candidate gets ``sigmoid(offset + distance*d + keyword*log(1+kw) +
    keyword_share*kw/kw_best + traversal*g)``, where ``d`` is its squared-L2
    distance to the (full) message, ``kw`` its keyword (BM25) score with the
    store's common words excluded, ``kw_best`` the message's best keyword score
    over the whole store, and ``g`` 1.0 when the graph reached this candidate
    by expansion rather than either leg ranking it. Candidates at or above ``threshold`` are injected, in
    search-rank order. ``threshold: 0`` disables the gate (a plain top_n slice).

    The defaults were fitted on one real store (443 chunks, MiniLM embeddings,
    1,713 graded message-memory pairs). Headings were in the keyword index at
    the time but not yet in every vector, so ``distance`` was fitted against a
    slightly weaker vector leg than the one it now scores; re-measuring after
    that was corrected moved the median best distance by 0.008 and did not
    justify a re-fit. They are a calibrated starting point, not universal
    constants — see docs/operations.md, *Memory retrieval*, for
    re-calibrating them on another store.
    """
    threshold: float = 0.45
    offset: float = 2.40
    distance: float = -2.99
    keyword: float = 0.64
    keyword_share: float = 0.24
    # Weight on `traversal`: 1.0 for a candidate the graph reached from the
    # fused pool, 0.0 for everything the two legs ranked directly. It defaults
    # to 0.0 -- being reachable is not by itself evidence of relevance, so a
    # graph-reached chunk must stand on its content like any other until a fit
    # against graded data earns it a positive weight.
    traversal: float = 0.0


@dataclass(frozen=True)
class RetrievalCalibrationConfig:
    """Where the relevance weights were calibrated, for the drift warning.

    ``best_distance`` is the median best vector distance per message when the
    weights were fitted. Retrieval tracks the same median over recent turns and
    logs a warning when it strays past ``tolerance`` — the store's shape has
    moved and the weights deserve a re-check. (A stale floor constant once went
    unnoticed for months exactly this way: topical distances fell from ~1.42 to
    ~0.97 and every memory passed.)
    """
    best_distance: float = 0.973
    tolerance: float = 0.25


@dataclass(frozen=True)
class GraphConfig:
    """Graph expansion of the candidate pool.

    The knowledge graph does NOT match the query. It expands outward from the
    chunks the two ranked legs already found, one hop, and only through
    section nodes -- a concept node is a bolded phrase with no chunk of its
    own, and walking through concepts nominated a median 48 of 443 chunks per
    message on one store. What comes back is judged by the ordinary relevance
    gate on its content, so a bad expansion costs candidate slots rather than
    prompt tokens.

    ``enabled: false`` removes the step entirely.
    """
    enabled: bool = True
    # How many of the fused candidates to expand from, best-ranked first.
    seeds: int = 10
    # Hard cap on chunks added per query, whatever the graph's shape.
    max_expand: int = 20


@dataclass(frozen=True)
class SmartRetrievalConfig:
    """Smart retrieval: a second opinion on the candidates near the line.

    Ordinary retrieval scores every candidate with the fitted relevance
    formula and keeps those over ``relevance.threshold``. That formula is
    calibrated and cheap, but it is a bag of four numbers, and near its own
    threshold it is guessing. Smart retrieval sends just those borderline
    candidates to the System One scorer for a second opinion, and acts on it:

    * a REJECTED candidate the scorer calls relevant is **promoted** in;
    * a RETRIEVED candidate the scorer calls irrelevant is **dropped**.

    Everything surviving is injected. There is no ``top_n`` slice on this
    path -- a turn may inject nothing, or everything that passed.

    **Only the borderline is asked about**, which is what keeps this
    affordable. Measured on one store: of 170 rejected candidates scoring
    more than ``promote_margin`` below the threshold, the scorer promoted
    exactly **zero**, so asking about them buys nothing. And candidates more
    than ``review_margin`` above it are kept unasked, because there the
    fitted score is confident and the scorer is measurably worse than it --
    it wanted to drop 17 such entries and 13 of them were useful.

    Graph-reached candidates are asked about **whatever their fitted score**.
    The fitted weights lean on a keyword score a graph-expanded chunk has no
    reason to have, so the formula's opinion of one is not evidence. Those
    candidates are the reason this feature can reach anything the ordinary
    path structurally cannot.

    Absent scorer, a scorer error, or a timeout ⇒ the whole block is skipped
    and retrieval behaves exactly as it does without it. Nothing here may
    become required.
    """
    enabled: bool = False

    # --- which candidates are worth asking about ---------------------------
    #
    # Both are margins AROUND `relevance.threshold`, never absolute scores.
    # The band exists to police that boundary, so if the threshold is tuned
    # the band has to follow it; absolute values would silently drift off the
    # line they were chosen for.
    #
    # How far BELOW the threshold to look for candidates to promote.
    promote_margin: float = 0.10
    # How far ABOVE the threshold to keep reviewing candidates for dropping.
    # Beyond this, entries are kept without asking. Raising it means a
    # tighter, smaller set; lowering it means more material at a slightly
    # lower hit rate. Measured on one store, 0.30 gave 2.54 memories a turn
    # against 2.14 for no ceiling at all -- 17% more context for 1.4 points
    # of precision.
    review_margin: float = 0.30

    # --- what the scorer's answer has to be --------------------------------
    #
    # P(yes) from the scorer. These sit far lower than a "0 to 1" reading
    # suggests: on this scorer a note squarely on topic measures ~0.8, an
    # irrelevant one ~0.06, and the useful separation is near the bottom of
    # the range. They are also tied to `note_chars` -- see below.
    promote_threshold: float = 0.905
    retain_threshold: float = 0.107
    # Graph-reached candidates are judged on a much lower bar, because they
    # score low across the board: their relevance is usually indirect (the
    # note that matched REFERS to this one), which is not visible in the text
    # the scorer reads.
    graph_promote_threshold: float = 0.10

    # --- cost --------------------------------------------------------------
    #
    # How much of a note the scorer reads. Cost is roughly
    # `fixed + questions x per-question`, and the per-question part scales
    # with this: measured 59 ms at 400 characters, 42 ms at 200, 35 ms at 100.
    #
    # ⚠ `note_chars` and the two thresholds above are ONE calibration. Clipping
    # moves the whole score distribution, so the same thresholds cut at a
    # different place: between 400 and 200 characters the retain threshold had
    # to move 0.149 -> 0.107 to hold the same operating point. Changing this
    # number alone is a silent regression, not a tuning. 200 was measured to
    # be indistinguishable from 400 once the thresholds were re-fitted, and
    # 100 was not -- it returned measurably worse material.
    note_chars: int = 200
    # Retrieval's own scorer deadline, independent of ``systemone.timeout_s``.
    # The two callers of one shared client want different bounds: the gate
    # sits in front of the LLM and must fail open fast, while this asks about
    # every borderline candidate in ONE batched request whose answer decides
    # what the turn is told. Measured over 50 real turns, that request runs a
    # median 1.4s and at most ~4.4s, so a 2s bound truncated the tail and cost
    # ~5 points on every retrieval metric — the turns it cut fell back to
    # ordinary retrieval. A failure here is never lost memory, only lost
    # precision, which is why it can afford to wait.
    timeout_s: float = 5.0

    # Prior user/assistant messages the scorer reads as conversation state,
    # in addition to the current one.
    #
    # Worth having: many turns ("looks good", "try it again") carry their
    # subject in the conversation and not in the text being matched, and no
    # amount of tuning on a bare message reaches those. Measured on a graded
    # set collected WITH the same context in front of the referee -- grading
    # against the bare message while the scorer reads the thread asks a
    # different question and marks down exactly the turns context rescues --
    # it made the filter sharper rather than merely stricter: relevance
    # 54.4% -> 59.9% with recall of clearly-relevant material UNCHANGED at
    # 73.4%. What it drops is "somewhat relevant" padding.
    #
    # Cost is flat per turn, not per candidate: the state is prefilled once
    # and shared by every question, so 6 turns adds ~190 ms whether 4
    # candidates are asked about or 22.
    #
    # Default matches `gate.context_turns`, which is the same question asked
    # of the same conversation.
    context_turns: int = 6


@dataclass(frozen=True)
class MemoryRetrievalConfig:
    top_n: int = 5
    compact: bool = True
    # Search results (fused vector + keyword ranking) that get the relevance
    # check; up to top_n of those that pass are injected.
    candidates: int = 20
    # A word appearing in more than this share of the store's memories is
    # ignored by keyword search: it matches too much to mean anything. Derived
    # from the store itself on every reindex — no hand-written word list.
    common_word_max_share: float = 0.10
    relevance: RelevanceConfig = RelevanceConfig()
    calibration: RetrievalCalibrationConfig = RetrievalCalibrationConfig()
    graph: GraphConfig = GraphConfig()
    smart_retrieval: SmartRetrievalConfig = SmartRetrievalConfig()


@dataclass(frozen=True)
class SearxngConfig:
    base_url: str


@dataclass(frozen=True)
class CronConfig:
    enabled: bool
    jobs_file: Path
    max_instances_per_job: int = 1
    # Agent ids that get cron_add / cron_list / cron_remove. Empty = no agent
    # has scheduling tools (cron jobs in jobs.json still fire, but no agent
    # can mutate them via tool calls). Per-deployment policy: gate by
    # responsibility, not name in code.
    exposed_to: tuple[str, ...] = ()
    # Fallback delivery target when cron_add is called without an explicit
    # deliver_to. Empty/None = require agent to specify per call. Used to
    # hardcode a deployment-specific phone MXID; pulled into config so the
    # source stays deployment-neutral.
    default_deliver_to: str | None = None
    # Grace (seconds) for one-shot `at` jobs missed while the daemon was down
    # (e.g. across a restart). Detected at the next boot. A reminder missed by
    # <= this fires late then (delivered + cleaned via the normal executed
    # path); one missed by more is collected and surfaced to its owning agent
    # as a review turn (deliver-late-or-discard) rather than silently dropped
    # or blindly re-fired. Applied to `at` jobs only — recurring `cron` jobs
    # keep APScheduler's default to avoid a downtime burst.
    misfire_grace_time: int = 3600


@dataclass(frozen=True)
class BluetoothConfig:
    """Agent-facing control of the host's Bluetooth radio (macOS, via blueutil).

    Absent block = disabled, so a deployment that never thought about Bluetooth
    gets no tools. Two gates, answering different questions: ``exposed_to`` says
    *who* may drive the radio, ``verbs`` says *what* may be done to it. They are
    independent because the risky part is the verb, not the caller — pairing a
    speaker and forgetting a mouse are not the same act, and one agent usually
    wants the first without the second.

    ``verbs`` defaults to the observing pair. Anything that mutates radio state
    is opt-in per deployment: on a headless machine there is no GUI to undo a
    controller power-off or a stray unpair with.
    """
    enabled: bool = False
    binary: Path = Path("/opt/homebrew/bin/blueutil")
    exposed_to: tuple[str, ...] = ()
    verbs: tuple[str, ...] = ("list", "scan")
    # An inquiry occupies the radio for its whole duration, so an unbounded
    # value lets a single tool call wedge every other Bluetooth consumer.
    # Classic BR/EDR inquiry works in 1.28s slots and wants ~10.24s to find a
    # device reliably; a cap below that makes scanning quietly unreliable.
    max_scan_seconds: int = 20
    # Wall-clock ceiling on one blueutil invocation. Its own internal waits are
    # ~10s (power, connect), so this is a backstop against a wedged radio, not
    # a tuning knob.
    timeout_s: float = 90.0


@dataclass(frozen=True)
class SystemOneConfig:
    """An optional System One decision scorer (typed decisions, no generated
    text — see claw/systemone.py).

    Absent block = this claw has no scorer, and that is a fully supported way
    to run: nothing may REQUIRE one. Every feature that asks the scorer
    anything — the gate's handlers today, internal decisions later — must
    check that a client exists and, when it does not (or when a call fails),
    behave exactly as it would have without the feature. One client is built
    from this block and shared by every user.
    """
    base_url: str = "http://127.0.0.1:11502"
    # No timeout here, deliberately. Each caller states its own deadline at
    # the call site (gate.timeout_s, memory_retrieval.smart_retrieval
    # .timeout_s), because they sit on different paths and want different
    # bounds. A client-level default could not be reached in any case — a
    # per-request timeout overrides it — so it would only serve to let the
    # next caller inherit someone else's policy by accident.


@dataclass(frozen=True)
class GateReplyConfig:
    """How a handler answers — exactly one of:

    * ``reply``: say one of these (chosen at random when there are several);
    * ``reply_fn``: say what this named function (claw/gate_replies.py)
      returns, or one of ``fallback`` when it returns nothing;
    * ``relay``: say the tool's own text output.
    """
    reply: tuple[str, ...] = ()
    reply_fn: str | None = None
    fallback: tuple[str, ...] = ()
    relay: bool = False


@dataclass(frozen=True)
class GateHandlerConfig:
    """One direct handler: a turn the gate may answer without the LLM.

    Declarative on purpose: ``description`` is what the scorer reads, ``tool``
    names a tool already in the agent's own registry, ``args`` are fixed — a
    handler picks, it never extracts from the message. Two shapes:

    * **text** (``expect`` + one reply form): the tool's text output must be
      exactly ``expect`` (or, with no ``expect``, anything not starting
      ``error:``/``refused:``), and the handler answers with ``reply``.
    * **outcomes**: the tool's structured result (its ``data`` twin) carries a
      ``result``; each listed result has its own reply form. For tools whose
      answer matters beyond success, e.g. a skip that reports what is next or
      that the queue has ended.

    Anything not accepted declines the turn, which then goes to the LLM: a
    failure never gets a cheerful fixed reply.
    """
    id: str
    description: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    expect: str | None = None
    reply: GateReplyConfig | None = None
    outcomes: dict[str, GateReplyConfig] = field(default_factory=dict)
    # Overrides the gate's min_confidence for this handler (e.g. higher for a
    # destructive action).
    min_confidence: float | None = None


@dataclass(frozen=True)
class GateConfig:
    """The decision gate in front of every human turn.

    Before a matrix or voice turn reaches the LLM, the gate may hand it to a
    direct handler instead: the scorer picks one handler's description or
    "none of these". Anything the scorer is not sure of, and every failure of
    the scorer or the handler, goes to the LLM as before — the gate fails open.

    With no handlers the gate never calls the scorer and needs none: every
    eligible turn passes through unchanged, and says so in the log. Handlers
    require a ``systemone`` block.

    Absent block = disabled. ``exposed_to`` says which agents are gated.
    """
    enabled: bool = False
    exposed_to: tuple[str, ...] = ()
    # A handler runs only when the scorer picks it with at least this
    # confidence (1 - normalised entropy), unless the handler sets its own.
    # Below it, the LLM answers: a needless LLM turn costs seconds, a wrong
    # handler does the wrong thing.
    min_confidence: float = 0.8
    # How many prior user/assistant messages (tool traffic and synthetic
    # notes excluded) the scorer sees with the new message, so a short
    # follow-up ("skip this one") can be read in context.
    context_turns: int = 6
    # The gate's own fail-open bound. It runs before every eligible turn and
    # a timeout means "let the LLM answer", so it is a latency guarantee to
    # the user, not a transport setting — which is why it lives here beside
    # the gate's other policy rather than under `systemone`.
    timeout_s: float = 2.0
    handlers: tuple[GateHandlerConfig, ...] = ()


@dataclass(frozen=True)
class MusicOutput:
    """One named speaker the agent can address by id.

    The agent says ``room-a``; it never handles a CoreAudio device string,
    a MAC, or a filesystem path. That is not only ergonomics — a raw device
    string on a model-composed command line is a quoting hazard, and a MAC is a
    thing a model will cheerfully invent.

    ``bluetooth_address`` being **absent** means a wired or built-in output with
    no connect step. It is never null-for-none: an output that is reachable
    unconditionally and one that must be woken are different kinds of thing, and
    the difference is whether the key is there.
    """
    id: str
    name: str
    mpv_device: str
    # Diagnostics and the SwitchAudioSource default-output dance. Never matched
    # against; mpv_device is the identity.
    coreaudio_name: str = ""
    bluetooth_address: str | None = None
    default: bool = False


@dataclass(frozen=True)
class DjConfig:
    """Speech between the songs: what the DJ sounds like, and how loud.

    Rendered by the **Piper CLI**, not by wyoming-piper. That is the whole
    reason these settings can exist at all: ``length_scale`` is a daemon CLI
    flag, so a streaming voice path has one global rate shared by every device,
    and a per-device rate would need a second piper daemon on its own port. A
    file render goes nowhere near the daemon, so every knob below is
    an argument to a subprocess and independent of how an agent sounds when it
    answers out loud.

    Independent on purpose: a later change to the desktop voice should not drag
    the DJ along with it, and vice versa.
    """
    enabled: bool = False
    # The Piper CLI and the voice it speaks in. Point both at an existing Piper
    # install — typically the same one a voice stack already uses, declared once
    # in the deployment so the two cannot drift.
    piper_binary: Path = Path("/opt/homebrew/var/claw/voice/venv/bin/piper")
    voice_model: Path = Path(
        "/opt/homebrew/var/claw/voice/models/piper/en_US-joe-medium.onnx"
    )
    # Regenerable cache, so the runtime dir with the socket — NOT ~claw/.claw,
    # where library.db lives because it cannot be rebuilt from the files.
    render_dir: Path = Path("/opt/homebrew/var/claw/music/dj")

    # -- synthesis (piper flags) --
    # Phoneme duration: 1.0 native, <1 faster. The daemon's global --length-scale
    # does not apply here.
    length_scale: float = 1.0
    # Silence Piper leaves after each sentence, in seconds.
    sentence_silence: float = 0.3

    # -- loudness --
    # Soft-clip drive (ffmpeg asoftclip=type=tanh:param=N), the same curve as
    # the voice-gateway's loudness.apply_drive. 1.0 = off — and off means the
    # filter is left out of the chain entirely, because asoftclip at param=1.0
    # is not a passthrough (measured: -0.4 LU, peaks 2.4 dB down).
    #
    # **Timbre only — it never moves the level.** A saturated link is measured
    # against the same text rendered flat and the difference is cut back off,
    # so the voice sits against the music exactly where it does with this off
    # (within 0.2 LU, measured on the same render). A taste setting, found by ear: small values
    # add density and presence; push it and aspirations and sibilance start
    # to grit on a real speaker, well before anything clips.
    drive: float = 1.0
    # Where a link sits against the music, in LUFS, measured on the finished
    # clip. None = the flat Piper render's own level (~-20 LUFS), untouched. A
    # louder render is cut to it; a quieter one is raised toward it only as far
    # as max_true_peak_dbfs allows — the music's rule, with no limiter. Also
    # evens out Piper's line-to-line spread (~3 LU). Set, it pins the level,
    # so drive stays timbre only without being cut back to flat first.
    target_lufs: float | None = None
    # Ceiling on the rendered clip's true peak, in dBFS. This is the one
    # loudness assertion worth making automatically, because it is on the axis
    # that matters and it is entirely in our gift: much of a real collection
    # already clips, but there is no reason to ship a clipped clip. Raising the
    # drive without checking this is how you would.
    max_true_peak_dbfs: float = -1.0
    # Silence padded onto each end. Breathing room around the cut into music —
    # NOT ducking, which was considered and rejected. 0 disables.
    pad_ms: int = 250
    # Clips are resampled to match the library so the gapless path does not have
    # to reopen the audio device between an entry and a track, which on the
    # Bluetooth link is audible. Piper renders 22050 mono; the collection is
    # overwhelmingly 44100 stereo.
    sample_rate: int = 44100
    channels: int = 2

    # -- housekeeping --
    # A run sheet longer than this is a mistake, not a set: cost scales with
    # speech seconds, and a model that wants twenty links has misread the room.
    max_links: int = 12
    # How long an abandoned session's clips survive. Matches the tool-results
    # spool sweep, for the same reason: a queue that got replaced leaves its
    # clips behind and nothing else will ever come for them.
    keep_hours: float = 24.0


@dataclass(frozen=True)
class LoudnessConfig:
    """How playback level is set — and every number that depends on the collection.

    Three measurements, each answering one question. **Integrated LUFS** is how
    loud a track *sounds*: a LUFS gap between two records is a level difference
    a listener hears. **True peak** is headroom — how much gain a track can take
    before it clips — and says nothing about how loud it sounds. LRA is dynamics.
    In a collection spanning the loudness war the first two come apart: nearly
    every track peaks near full scale, and the loud ones are loud *by
    compression*. So a loudness gap is free to close downward and mostly cannot
    be closed upward without a limiter, which would change the music rather
    than its level.

    That is the policy. Aim at the collection's **mode**, so a typical record
    plays exactly as it does at unity; cut what is louder; boost what is quieter
    only as far as its own true peak allows. Everything here is a property of a
    particular collection, so it lives in config — the music CLI's ``stats``
    reports the measured mode next to the configured target, which is how you
    re-derive ``target_lufs`` as the collection grows.
    """
    # Off = every entry plays at unity, which is also what a collection nobody
    # has measured gets.
    normalize: bool = False
    # Where gain aims: the collection's MODE, not its mean or a broadcast
    # standard. At the mode the typical record is untouched, so turning this on
    # does not make the room quieter. -18 is the ReplayGain 2.0 reference, and
    # only a placeholder until the collection has been measured.
    target_lufs: float = -18.0
    # Boost stops where the track's true peak would reach this. It is a ceiling
    # on OUR boost only — a master that already peaks above it is left as it
    # is. 0.0 rather than the textbook -1.0 (the lossy-codec margin): where
    # most of a collection already peaks above 0 at unity, -1 protects the
    # boosted tracks alone, which would then be the cleanest ones playing,
    # while roughly halving how many tracks get any boost at all.
    boost_ceiling_dbtp: float = 0.0
    # A track nobody has measured is assumed to be this loud, so the failure
    # mode of an un-ingested file is "cut too far" rather than "far too loud
    # in a room". Put it near the collection's loud end. Never boosted either:
    # there is no peak to bound the boost by.
    assumed_lufs: float = -8.0
    # Album furniture — codas, segues, spoken intros: a track this far below its
    # own record's mean and no longer than furniture_max_s. Dropped from a
    # shuffle, always kept in album order, and never lifted on its own. Both are
    # calibrated against what the collection's interstitials measure.
    furniture_below_album_lu: float = 8.0
    furniture_max_s: float = 120.0


@dataclass(frozen=True)
class CandidatesConfig:
    """``music_candidates`` — the set-builder's pool, and what it leaves out.

    Unlike ``music_search`` this filters on purpose, so every exclusion is
    config and every one is counted in the tool's reply. Both lists below are
    properties of a household and a collection, not of the code.
    """
    # A track queued this recently is left out of a pool. Beyond the window
    # nothing is excluded, but never-queued and least-recently-queued tracks
    # are still offered first.
    fresh_hours: float = 24.0
    # Genres that are not music for a set — spoken word, comedy. Matched whole
    # and case-insensitively, so "Speech" does not also remove "Speeches of
    # Malcolm X" filed as Hip-Hop.
    exclude_genres: tuple[str, ...] = ()
    # Minutes of candidates offered per minute asked for: room to choose by
    # taste rather than a list to accept.
    pool_factor: float = 2.0
    # What "a set" means when no length is given.
    default_minutes: float = 60.0
    # Hard ceiling on a pool, so a long request cannot flood the prompt.
    max_tracks: int = 60


@dataclass(frozen=True)
class MusicConfig:
    """Local music playback through a long-lived mpv held open on an IPC socket.

    Absent block = disabled, so a deployment with no speakers gets no tools.
    Gated by ``exposed_to`` the same way cron and bluetooth are: playback is a
    thing that happens in a room someone is sitting in, so who may do it is
    per-deployment policy rather than a property of the code.

    There is deliberately **no per-output volume**. Loudness is set once,
    downstream, on the amplifier: a standing attenuation in software happens
    *before* a lossy encoder, so the amp then raises music and codec noise
    together. Per-track normalisation (see :class:`LoudnessConfig`) is not
    that — it brings a loud record down to where the typical one already
    enters the encoder, and never boosts anything past its own true peak.
    """
    enabled: bool = False
    exposed_to: tuple[str, ...] = ()
    library_root: Path = Path("/Users/Shared/media/audio")
    mpv_binary: Path = Path("/opt/homebrew/bin/mpv")
    mpv_socket: Path = Path("/opt/homebrew/var/claw/music/mpv.sock")
    # The catalogue: measured loudness plus the curation nobody can derive from
    # an MP3. Deliberately NOT under the runtime dir with the socket — this is
    # the one part of the music stack that cannot be rebuilt from the files, so
    # it belongs wherever the deployment's backup already reaches.
    db_path: Path = Path("/opt/homebrew/var/claw/music/library.db")
    # Used only to read and restore the machine-wide default output around a
    # Bluetooth connect (see claw.music.ensure_output). Playback itself never
    # touches the default; mpv is told its device directly.
    switchaudio_binary: Path = Path("/opt/homebrew/bin/SwitchAudioSource")
    # How long to wait for a Bluetooth output to come back after asking for it,
    # measured from the connect to the device appearing in mpv's device list.
    # Past this the tool refuses and says why; it never falls back to another
    # speaker, because audio arriving from the wrong room is worse than silence.
    connect_timeout_s: float = 15.0
    loudness: LoudnessConfig = field(default_factory=LoudnessConfig)
    candidates: CandidatesConfig = field(default_factory=CandidatesConfig)
    outputs: tuple[MusicOutput, ...] = ()
    # Absent block = no DJ tool, the same way an absent music block means no
    # music tools at all.
    dj: DjConfig = field(default_factory=DjConfig)

    def by_id(self, output_id: str) -> "MusicOutput | None":
        return next((o for o in self.outputs if o.id == output_id), None)

    @property
    def default_output(self) -> "MusicOutput | None":
        return next((o for o in self.outputs if o.default), None)


@dataclass(frozen=True)
class PersonaConfig:
    model: str
    role: str
    # Max chain depth this persona is willing to root. 0 = leaf (cannot spawn);
    # 1 = can spawn one level under itself; etc. Effective budget when this
    # persona is forked is min(parent_remaining - 1, this value).
    max_spawn_depth: int = 0
    # Persona names this persona is allowed to spawn. None = no restriction
    # (subject to max_spawn_depth). Empty tuple = explicitly nothing (also
    # subject to max_spawn_depth). Validated at config load against the
    # persona registry.
    can_spawn: tuple[str, ...] | None = None
    # Per-persona override of OllamaConfig.num_predict. None inherits global.
    num_predict: int | None = None
    # Per-persona override of OllamaConfig.max_tool_turns. None inherits global.
    # Tune by role: researchers (web_search + reads) need headroom; grunts
    # should stay tight; coders sit in the middle.
    max_tool_turns: int | None = None


@dataclass(frozen=True)
class SubagentsConfig:
    max_concurrent: int
    max_children_per_agent: int
    default_model: str
    personas: dict[str, PersonaConfig]
    # Ceiling on one spawned task's whole life, including any time it spends
    # waiting on its own children. A subagent's session is scratch and is only
    # reachable through its handle, so a task that never finishes would pin
    # both until the gateway restarts; this bounds that. Generous by design —
    # a researcher persona legitimately runs for many minutes.
    task_timeout_seconds: int = 3600


@dataclass(frozen=True)
class CompactionConfig:
    idle_recap_seconds: int = 3600
    # Floor for the idle recap. A session smaller than this is left verbatim
    # rather than summarized: below it the recap is no smaller than the rows
    # it replaces, so compressing is pure loss. Guards the observed case of a
    # 2-turn session being recapped at boot.
    idle_recap_min_tokens: int = 4000
    # Mid-session compaction fires when estimated transcript tokens exceed
    # this. Defaults are tuned for a 192K context window — set so compaction
    # triggers around 50% of context, leaving generation headroom and a
    # comfortable reserve.
    mid_session_token_threshold: int = 96000
    # When compaction fires, the newest ``reserve_tokens`` worth of rows are
    # preserved verbatim; the older portion is summarized into a single
    # recap turn. The split walks newest-first to accumulate this budget
    # then advances to the next ``user`` boundary so it doesn't slice mid
    # tool-call sequence. Default = 1/4 of a 192K window.
    reserve_tokens: int = 48000


@dataclass(frozen=True)
class MemoryFlushConfig:
    enabled: bool = True
    # The growth gate: a flush fires at turn-end whenever a session's
    # transcript has grown by this many tokens since its last flush. Evaluated
    # alongside the compaction gate in one place (Agent.
    # _spawn_bg_maintenance_if_needed) rather than on a timer — a timer could
    # not see anything extra (growth only comes from turns) and firing on one
    # risks starting GPU work alongside a live reply. Lets long-running sessions capture durable info
    # regularly instead of waiting for compaction to be imminent. With a
    # 96K compaction trigger, 4K of growth ≈ 4% — a comfortable cadence.
    periodic_growth_threshold: int = 4000
    # Per-flush deadline. The flush runs as a background task off the user's
    # critical path, but a wedged Ollama generation can otherwise hold it for
    # the full httpx 900s × tool-loop iterations. On timeout we drop the flush
    # and let compaction proceed anyway — losing one flush is cheaper than
    # delaying compaction.
    turn_timeout_s: float = 300.0


@dataclass(frozen=True)
class MemoryCurationConfig:
    """Nightly "forgetory" curator — grooms each agent's markdown memory.

    A separate, heavier pass than the frequent collector (memory_flush): once
    per night a larger model dedups near-identical memories (recurring-cron
    churn), marks superseded long-term facts with a forward ``[SUPERSEDED BY ->
    <id>]`` pointer (retrieval auto-follows old->new), and archives lapsed
    ephemera out of the indexed daily notes into ``memory/archive/YYYY-MM.md``. All
    state lives in the markdown (the source of truth); ChromaDB/BM25 are
    re-derived, so ``rm -rf .memory/`` rebuilds everything intact.

    Coupled to the collector: only runs for agents when ``memory_flush`` is
    enabled (no memory collected -> nothing to curate).
    """
    enabled: bool = False
    # Local-tz hour (0-23) for the nightly pass. A quiet hour both keeps the
    # 122b off the user-reply path and sidesteps the collector/curator file
    # race (the curator also skips today's daily note).
    hour: int = 4
    # Bigger model than the collector — supersession/dedup is relational
    # judgment worth the cost. Per-deployment so source stays neutral.
    model: str = "qwen3.5:122b"
    # Near-neighbours fetched per changed memory (via the existing hybrid
    # search) so the curator can judge dedup/supersession against them.
    near_neighbor_k: int = 8
    # The curator processes ONE daily-note file per turn (a whole-corpus
    # briefing wouldn't fit context). This caps candidate files per
    # invocation (0 = unlimited); lets a large first bootstrap be chunked
    # across runs. Each file is a separate bounded turn regardless.
    max_files_per_run: int = 0
    # Investigative latitude — the curator reads old notes / files before
    # judging, so it needs more tool round-trips than a collector flush.
    max_tool_turns: int = 80
    # Per-generation token cap for the curator turn. Doubled vs the global
    # default (OllamaConfig.num_predict, 8192): a single write_file that
    # rewrites a whole daily note can be large, and truncating it mid-file
    # would corrupt memory. None inherits the global.
    num_predict: int | None = 16384
    # Per-curation deadline. Generous: a full-corpus-ish nightly groom over a
    # 122b legitimately takes a while. On timeout the pass is abandoned and
    # retried next night (markdown is untouched-or-partially-edited but valid).
    turn_timeout_s: float = 1800.0
    # A superseded memory is kept searchable as a visible timeline, but once
    # it's been superseded this many days the old version is usually dead
    # weight. The nightly supersession-review turn is handed the COMPLETE
    # superseded list (any age, whole corpus — not window-limited) with each
    # one's age, and archives the stale ones case-by-case. Age is measured
    # from the *superseding* memory's ts (the marker records when a memory was
    # written, never when it was superseded; the replacement's ts is when the
    # old fact went stale).
    superseded_archive_days: int = 30


@dataclass(frozen=True)
class LifecycleConfig:
    # Hour (0-23, local time) at which to wipe in-flight session transcripts
    # for every agent and start fresh. None disables. Memory_flush runs once
    # per session before the wipe so durable knowledge lands in
    # memory/YYYY-MM-DD.md first; the JSONL is then archived with a `.reset-*`
    # suffix and the next inbound message starts a clean session. Useful as
    # a daily reset so transcripts don't grow indefinitely.
    daily_session_rotate_hour: int | None = None


@dataclass(frozen=True)
class CommandsConfig:
    # In-band admin commands parsed out of Matrix message bodies before they
    # reach the LLM. A message is only treated as a command when enabled, the
    # channel is matrix, the sender is in `allow`, and the body starts with
    # `prefix`. Anything else (incl. an unauthorized sender's prefixed text)
    # flows to the LLM as ordinary text — no command, no reply, no indication.
    # Defaults true, but `allow` is still fail-closed (empty = nobody), so a
    # deployment that never sets `allow` has no usable commands regardless.
    enabled: bool = True
    # Sigil that prefixes a command word. "%" is safe: Matrix clients don't
    # intercept it (unlike "/"), it isn't Markdown, and it's improbable as a
    # natural first character. Configurable per deployment.
    prefix: str = "%"
    # MXIDs allowed to run commands. Empty tuple = nobody (fail closed).
    # Deliberately separate from per-agent matrix.allow_from so a sender who
    # may DM an agent does not automatically gain control-plane access.
    allow: tuple[str, ...] = ()


@dataclass(frozen=True)
class VoiceServiceConfig:
    """Claw-side voice config — the *modality* of a voice interaction.

    Slim by design: the audio pipeline (mic capture, wake, STT, TTS) lives in an
    external voice stack, not in claw. All claw needs to know is how to *steer an
    agent's reply* when the turn is voice — which applies to **any** voice
    client, including one that does its own STT/TTS and reaches claw over the
    HTTP turn endpoint. So this holds only the modality hint; reply rendering
    (voice selection) is the external voice stack's concern.
    """
    # System-prompt note injected when a turn's ``modality == "voice"`` (see
    # Agent._voice_modality_hint — gated on modality, never on channel). Steers
    # replies to stay brief and speakable and warns the input is an STT
    # transcript prone to homophones. Empty string disables the hint entirely.
    modality_hint: str = (
        "You are speaking over a voice channel: your reply is read aloud by "
        "text-to-speech and the user's message is a speech-to-text transcript. "
        "Keep replies brief and speakable — one or two short sentences, no "
        "markdown, tables, code blocks, or bullet lists; spell out URLs, "
        "symbols, and long numbers as words. The transcript may contain "
        "mishearings (their/there, to/two, aria/area, digits, proper nouns); "
        "when a load-bearing word is ambiguous, ask the user to confirm rather "
        "than act on a possibly-misheard token."
    )


@dataclass(frozen=True)
class HttpApiConfig:
    """claw's HTTP turn endpoint (``claw.voice_http``).

    A transport-agnostic inbound: a caller POSTs ``{device_id, endpoint_id,
    text}`` and gets ``{reply, agent_id}`` back. Used both by an external voice
    stack (fronting a thin client that can't do STT/TTS) and directly by a
    client that does its own STT/TTS. Decoupled from voice: it only needs
    ``devices:`` for routing/identity; the voice modality hint applies when the
    resolved endpoint's ``type`` is ``"voice"``.
    """
    enabled: bool = False
    # Binds ``0.0.0.0`` so a remote client can reach it over the network, not
    # just loopback — consistent with how matrix and the other services bind.
    # Auth for direct callers is still TODO; the closed LAN/tailnet is the
    # boundary for now.
    bind_host: str = "0.0.0.0"
    bind_port: int = 11501


@dataclass(frozen=True)
class EndpointConfig:
    """One logical source within a device (``id : name`` ≈ MXID : displayname,
    scoped inside the device). ``sender_id = f"{device_id}/{id}"``;
    ``sender_name = name``. A Box has one implicit ``voice`` endpoint."""
    id: str
    type: str          # modality: voice | event | state
    name: str          # human label the agent hears as sender_name


@dataclass(frozen=True)
class DeviceConfig:
    """A physical device. ``Hello.device_id`` resolves to one of these; the
    bound ``agent`` answers, and the matching endpoint supplies identity."""
    device_id: str
    name: str
    agent: str
    endpoints: tuple[EndpointConfig, ...]


@dataclass(frozen=True)
class MatrixAccountConfig:
    user_id: str
    homeserver: str
    access_token_file: Path
    device_id: str
    device_name: str
    store_path: Path
    encryption: bool = True
    auto_join: Literal["always", "never"] = "always"
    allow_bots: Literal["mentions", "all", "none"] = "mentions"
    allow_from: tuple[str, ...] = ()
    # Password file is needed to satisfy the UIA challenge on
    # /keys/device_signing/upload during cross-signing bootstrap. If unset,
    # cross-signing is skipped — the bot still works but appears as
    # "user verification unavailable" in Element. Mode 0600.
    password_file: Path | None = None
    # One-shot migration flag: replace any existing cross-signing keys on
    # the homeserver with freshly-generated ones. Use this when migrating
    # an account that was previously bootstrapped by another client and we
    # don't have access to the original SSK private key. *Destructive* —
    # invalidates prior device signatures and forces every other user to
    # re-verify this account in their Element client. Set true once for
    # the migration boot, then remove from config (or set false) on
    # subsequent runs.
    force_cross_signing_replace: bool = False


@dataclass(frozen=True)
class AgentConfig:
    id: str
    workspace: Path
    primary_model: str
    matrix: MatrixAccountConfig
    compaction_model: str | None = None
    extra_paths: tuple[str, ...] = ()
    # Max chain depth this top-level agent is willing to root. 0 = no spawning.
    # Subagent forks inherit a budget capped by this value (and further by the
    # persona's own max_spawn_depth).
    max_spawn_depth: int = 1
    # Persona allowlist. None = any persona (the default for top-level agents).
    # Tuple = restricted set. Same semantics as PersonaConfig.can_spawn.
    can_spawn: tuple[str, ...] | None = None
    # Per-agent override of OllamaConfig.num_predict. None inherits global.
    num_predict: int | None = None
    # Per-agent override of OllamaConfig.max_tool_turns. None inherits global.
    max_tool_turns: int | None = None


@dataclass(frozen=True)
class Config:
    verbose: bool
    ollama: OllamaConfig
    memory_retrieval: MemoryRetrievalConfig
    searxng: SearxngConfig
    cron: CronConfig
    bluetooth: BluetoothConfig
    music: MusicConfig
    subagents: SubagentsConfig
    compaction: CompactionConfig
    memory_flush: MemoryFlushConfig
    memory_curation: MemoryCurationConfig
    lifecycle: LifecycleConfig
    commands: CommandsConfig
    agents: tuple[AgentConfig, ...]
    # IANA timezone name (e.g. "America/New_York", "Europe/Berlin") used for
    # every operator-facing timestamp claw renders: inbound-message envelope,
    # daily-note filename `memory/YYYY-MM-DD.md`, daily-session-rotate hour.
    # ``None`` falls back to UTC. Internal/persisted timestamps (transcripts,
    # memory sync state, compaction bookkeeping) stay in UTC regardless —
    # this knob only affects what humans (and the model) see.
    tz: str | None = None
    # Language every agent replies in, as a human-readable id with locale, e.g.
    # "English (en_US)". Injected into the system prompt for ALL turns/channels so
    # a bilingual model (e.g. qwen) doesn't code-switch — critical for the voice
    # path, where TTS can't read non-Latin output and renders it as garbage.
    # ``None`` = no language steering.
    language: str | None = None
    # Claw-side voice config (modality hint). ``None`` = no voice steering; a
    # voice turn still works, it just gets no speakable-reply hint.
    voice: VoiceServiceConfig | None = None
    # claw's HTTP turn endpoint (for external voice stacks + direct clients).
    # ``None`` / disabled = not stood up.
    http_api: HttpApiConfig | None = None
    # Physical devices keyed (at use-site) by device_id. A turn (HTTP or, in the
    # gateway, a box Hello) resolves to one of these → its ``agent`` answers,
    # and the matching endpoint supplies identity + modality.
    devices: tuple[DeviceConfig, ...] = ()
    # Optional decision scorer. ``None`` = none configured; every user of it
    # must then fall back to its no-scorer behaviour (see SystemOneConfig).
    systemone: SystemOneConfig | None = None
    gate: GateConfig = GateConfig()


def load(path: Path | str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = _parse(raw)
    _validate_can_spawn(cfg)
    _validate_voice(cfg)
    _validate_music(cfg)
    _validate_smart_retrieval(cfg)
    _validate_gate(cfg)
    return cfg


def _parse_can_spawn(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(s.lower() for s in value)


def _parse_persona(spec: dict[str, Any]) -> "PersonaConfig":
    return PersonaConfig(
        model=spec["model"],
        role=spec["role"],
        max_spawn_depth=spec.get("max_spawn_depth", 0),
        can_spawn=_parse_can_spawn(spec.get("can_spawn")),
        num_predict=spec.get("num_predict"),
        max_tool_turns=spec.get("max_tool_turns"),
    )


def _validate_can_spawn(cfg: "Config") -> None:
    """Each can_spawn entry must reference an actual persona."""
    persona_names = set(cfg.subagents.personas.keys())
    for ac in cfg.agents:
        if ac.can_spawn is not None:
            unknown = set(ac.can_spawn) - persona_names
            if unknown:
                raise ValueError(
                    f"agent {ac.id!r} can_spawn references unknown persona(s): "
                    f"{sorted(unknown)}; available: {sorted(persona_names)}"
                )
    for name, pc in cfg.subagents.personas.items():
        if pc.can_spawn is not None:
            unknown = set(pc.can_spawn) - persona_names
            if unknown:
                raise ValueError(
                    f"persona {name!r} can_spawn references unknown persona(s): "
                    f"{sorted(unknown)}; available: {sorted(persona_names)}"
                )


def _validate_voice(cfg: "Config") -> None:
    """Device routing sanity: every device must bind to a real agent and carry
    at least one endpoint (so identity/modality resolution works). The audio
    pipeline and per-agent reply voice are the external voice stack's concern,
    not validated here."""
    agent_ids = {ac.id for ac in cfg.agents}
    for dev in cfg.devices:
        if dev.agent not in agent_ids:
            raise ValueError(
                f"device {dev.device_id!r} binds to unknown agent "
                f"{dev.agent!r}; known agents: {sorted(agent_ids)}"
            )
        if not dev.endpoints:
            raise ValueError(
                f"device {dev.device_id!r} has no endpoints"
            )


def _parse_voice_service(d: dict[str, Any] | None) -> VoiceServiceConfig | None:
    if not d:
        return None
    return VoiceServiceConfig(
        **({"modality_hint": d["modality_hint"]} if "modality_hint" in d else {}),
    )


def _parse_http_api(d: dict[str, Any] | None) -> HttpApiConfig | None:
    if not d:
        return None
    return HttpApiConfig(
        enabled=bool(d.get("enabled", False)),
        bind_host=d.get("bind_host", "0.0.0.0"),
        bind_port=int(d.get("bind_port", 11501)),
    )


def _parse_devices(d: list[dict[str, Any]] | None) -> tuple[DeviceConfig, ...]:
    if not d:
        return ()
    out: list[DeviceConfig] = []
    for spec in d:
        endpoints = tuple(
            EndpointConfig(id=e["id"], type=e["type"], name=e["name"])
            for e in spec.get("endpoints", ())
        )
        out.append(DeviceConfig(
            device_id=spec["device_id"],
            name=spec["name"],
            agent=spec["agent"],
            endpoints=endpoints,
        ))
    return tuple(out)


def _parse_commands(d: dict[str, Any]) -> CommandsConfig:
    return CommandsConfig(
        enabled=d.get("enabled", True),
        prefix=d.get("prefix", "%"),
        allow=tuple(d.get("allow", [])),
    )


def _parse(d: dict[str, Any]) -> Config:
    return Config(
        verbose=d.get("verbose", False),
        ollama=OllamaConfig(**d["ollama"]),
        memory_retrieval=_parse_memory_retrieval(d.get("memory_retrieval")),
        searxng=SearxngConfig(**d["searxng"]),
        cron=CronConfig(
            enabled=d["cron"]["enabled"],
            jobs_file=Path(d["cron"]["jobs_file"]),
            max_instances_per_job=d["cron"].get("max_instances_per_job", 1),
            exposed_to=tuple(d["cron"].get("exposed_to", ())),
            default_deliver_to=d["cron"].get("default_deliver_to"),
        ),
        bluetooth=_parse_bluetooth(d.get("bluetooth")),
        music=_parse_music(d.get("music")),
        subagents=SubagentsConfig(
            max_concurrent=d["subagents"]["max_concurrent"],
            max_children_per_agent=d["subagents"]["max_children_per_agent"],
            default_model=d["subagents"]["default_model"],
            task_timeout_seconds=d["subagents"].get(
                "task_timeout_seconds", 3600,
            ),
            personas={
                name.lower(): _parse_persona(spec)
                for name, spec in d["subagents"]["personas"].items()
            },
        ),
        compaction=CompactionConfig(**(d.get("compaction") or {})),
        memory_flush=MemoryFlushConfig(**(d.get("memory_flush") or {})),
        memory_curation=MemoryCurationConfig(**(d.get("memory_curation") or {})),
        lifecycle=LifecycleConfig(**(d.get("lifecycle") or {})),
        commands=_parse_commands(d.get("commands") or {}),
        agents=tuple(_parse_agent(a) for a in d["agents"]),
        tz=d.get("tz"),
        language=d.get("language"),
        voice=_parse_voice_service(d.get("voice")),
        http_api=_parse_http_api(d.get("http_api")),
        devices=_parse_devices(d.get("devices")),
        systemone=_parse_systemone(d.get("systemone")),
        gate=_parse_gate(d.get("gate")),
    )


def _parse_memory_retrieval(d: dict[str, Any] | None) -> MemoryRetrievalConfig:
    """Parse ``memory_retrieval:``. Strict, nested blocks included: an unknown
    key raises rather than silently keeping a default."""
    if not d:
        return MemoryRetrievalConfig()
    scalars = {k: v for k, v in d.items()
               if k not in ("relevance", "calibration", "graph", "smart_retrieval")}
    for k in ("candidates", "top_n"):
        if k in scalars:
            scalars[k] = int(scalars[k])
    if "common_word_max_share" in scalars:
        scalars["common_word_max_share"] = float(scalars["common_word_max_share"])
    rel = {k: float(v) for k, v in (d.get("relevance") or {}).items()}
    cal = {k: float(v) for k, v in (d.get("calibration") or {}).items()}
    # Not blanket-floated like the other two: `enabled` is a bool and the rest
    # are counts.
    graph = dict(d.get("graph") or {})
    if "enabled" in graph:
        graph["enabled"] = bool(graph["enabled"])
    for k in ("seeds", "max_expand"):
        if k in graph:
            graph[k] = int(graph[k])
    sr = dict(d.get("smart_retrieval") or {})
    if "enabled" in sr:
        sr["enabled"] = bool(sr["enabled"])
    for k in ("note_chars", "context_turns"):
        if k in sr:
            sr[k] = int(sr[k])
    for k in ("promote_margin", "review_margin", "promote_threshold",
              "retain_threshold", "graph_promote_threshold", "timeout_s"):
        if k in sr:
            sr[k] = float(sr[k])
    cfg = MemoryRetrievalConfig(
        relevance=RelevanceConfig(**rel), calibration=RetrievalCalibrationConfig(**cal),
        graph=GraphConfig(**graph), smart_retrieval=SmartRetrievalConfig(**sr),
        **scalars,
    )
    if not 0.0 <= cfg.relevance.threshold <= 1.0:
        raise ValueError(f"memory_retrieval.relevance.threshold must be within [0, 1], got {cfg.relevance.threshold}")
    if not 0.0 < cfg.common_word_max_share <= 1.0:
        raise ValueError(f"memory_retrieval.common_word_max_share must be within (0, 1], got {cfg.common_word_max_share}")
    if cfg.graph.seeds < 1 or cfg.graph.max_expand < 0:
        raise ValueError("memory_retrieval.graph.seeds must be >= 1 and max_expand >= 0")
    sc = cfg.smart_retrieval
    for name in ("promote_margin", "review_margin", "promote_threshold",
                 "retain_threshold", "graph_promote_threshold"):
        v = getattr(sc, name)
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"memory_retrieval.smart_retrieval.{name} must be within [0, 1], got {v}")
    if sc.retain_threshold > sc.promote_threshold:
        # Not a range check but a coherence one: a retrieved candidate would
        # need a HIGHER score to stay than a rejected one needs to get in,
        # which is not a policy anyone means to write.
        raise ValueError(
            "memory_retrieval.smart_retrieval.retain_threshold must not exceed "
            f"promote_threshold ({sc.retain_threshold} > {sc.promote_threshold})")
    if sc.context_turns < 0:
        raise ValueError("memory_retrieval.smart_retrieval.context_turns must be >= 0")
    if sc.note_chars < 1:
        raise ValueError("memory_retrieval.smart_retrieval.note_chars must be >= 1")
    if sc.timeout_s <= 0:
        raise ValueError(
            "memory_retrieval.smart_retrieval.timeout_s must be > 0, got "
            f"{sc.timeout_s}")
    if cfg.candidates < cfg.top_n:
        raise ValueError(f"memory_retrieval.candidates ({cfg.candidates}) must be >= top_n ({cfg.top_n})")
    return cfg


def _parse_systemone(d: dict[str, Any] | None) -> SystemOneConfig | None:
    """Parse the optional ``systemone:`` block. Strict; absent = no scorer."""
    if d is None:
        return None
    return SystemOneConfig(**dict(d))


def _parse_gate(d: dict[str, Any] | None) -> GateConfig:
    """Parse the optional ``gate:`` block. Strict: an unknown key raises, so a
    misspelled threshold fails the load instead of silently keeping the
    default. Handler structure is checked here too (see _parse_gate_handler)."""
    if not d:
        return GateConfig()
    scalars = {k: v for k, v in d.items() if k != "handlers"}
    if "exposed_to" in scalars:
        scalars["exposed_to"] = tuple(scalars["exposed_to"] or ())
    if "enabled" in scalars:
        scalars["enabled"] = bool(scalars["enabled"])
    if "min_confidence" in scalars:
        scalars["min_confidence"] = float(scalars["min_confidence"])
    if "context_turns" in scalars:
        scalars["context_turns"] = int(scalars["context_turns"])
    if "timeout_s" in scalars:
        scalars["timeout_s"] = float(scalars["timeout_s"])
    handlers = tuple(_parse_gate_handler(h) for h in d.get("handlers") or ())
    return GateConfig(handlers=handlers, **scalars)


_REPLY_KEYS = ("reply", "reply_fn", "fallback", "relay")


def _parse_reply(d: dict[str, Any], where: str, *, structured: bool) -> GateReplyConfig:
    """One reply form, strictly: exactly one of reply / reply_fn / relay;
    ``fallback`` only beside reply_fn. A structured outcome has no text to
    relay, and its reply_fn needs a fallback — the action has already run, so
    there must always be something to say."""
    from claw.gate_replies import REPLY_FUNCTIONS

    if unknown := sorted(set(d) - set(_REPLY_KEYS)):
        raise ValueError(f"{where}: unknown key(s) {unknown}")

    def texts(v: Any) -> tuple[str, ...]:
        return (v,) if isinstance(v, str) else tuple(str(x) for x in v or ())

    rc = GateReplyConfig(
        reply=texts(d.get("reply")), reply_fn=d.get("reply_fn"),
        fallback=texts(d.get("fallback")), relay=bool(d.get("relay", False)),
    )
    forms = [bool(rc.reply), rc.reply_fn is not None, rc.relay]
    if sum(forms) != 1:
        raise ValueError(f"{where}: needs exactly one of reply, reply_fn, relay: true")
    if rc.fallback and rc.reply_fn is None:
        raise ValueError(f"{where}: fallback only accompanies a reply_fn")
    if rc.reply_fn is not None and rc.reply_fn not in REPLY_FUNCTIONS:
        raise ValueError(
            f"{where}: unknown reply_fn {rc.reply_fn!r}; known: {sorted(REPLY_FUNCTIONS)}"
        )
    if structured and rc.relay:
        raise ValueError(f"{where}: an outcome has no text output to relay")
    if structured and rc.reply_fn is not None and not rc.fallback:
        raise ValueError(f"{where}: an outcome's reply_fn needs a fallback")
    return rc


def _parse_gate_handler(d: dict[str, Any]) -> GateHandlerConfig:
    """One handler, strictly — either the text shape (expect + a reply form)
    or the outcomes shape, never both."""
    h = dict(d)
    hid = h.get("id")
    where = f"gate.handlers[{hid!r}]"
    reply_part = {k: h.pop(k) for k in _REPLY_KEYS if k in h}
    outcomes_raw = h.pop("outcomes", None)
    if outcomes_raw is not None:
        if reply_part or "expect" in h:
            raise ValueError(f"{where}: outcomes replaces expect and the reply keys")
        if not outcomes_raw:
            raise ValueError(f"{where}: outcomes is empty")
        h["outcomes"] = {
            str(k): _parse_reply(dict(v or {}), f"{where}.outcomes[{k!r}]", structured=True)
            for k, v in outcomes_raw.items()
        }
    else:
        h["reply"] = _parse_reply(reply_part, where, structured=False)
    if "min_confidence" in h and h["min_confidence"] is not None:
        h["min_confidence"] = float(h["min_confidence"])
        if not 0.0 <= h["min_confidence"] <= 1.0:
            raise ValueError(f"{where}: min_confidence must be within [0, 1]")
    h["args"] = dict(h.get("args") or {})
    return GateHandlerConfig(**h)


def _parse_bluetooth(d: dict[str, Any] | None) -> BluetoothConfig:
    """Parse the optional ``bluetooth:`` block.

    Verbs are validated against the tool module's own list rather than a copy
    kept here, so a typo in claw.yaml fails the load instead of silently
    dropping a capability the operator believed they had granted.
    """
    if not d:
        return BluetoothConfig()
    from claw.tools.bluetooth import ALL_VERBS

    verbs = tuple(d.get("verbs", BluetoothConfig.verbs))
    if unknown := [v for v in verbs if v not in ALL_VERBS]:
        raise ValueError(
            f"bluetooth.verbs: unknown verb(s) {unknown}; known: {list(ALL_VERBS)}"
        )
    return BluetoothConfig(
        enabled=bool(d.get("enabled", False)),
        binary=Path(d.get("binary", BluetoothConfig.binary)),
        exposed_to=tuple(d.get("exposed_to", ())),
        verbs=verbs,
        max_scan_seconds=int(d.get("max_scan_seconds", BluetoothConfig.max_scan_seconds)),
        timeout_s=float(d.get("timeout_s", BluetoothConfig.timeout_s)),
    )


def _parse_music(d: dict[str, Any] | None) -> MusicConfig:
    """Parse the optional ``music:`` block.

    Both halves splat strictly, so an unknown or misspelled key raises rather
    than being dropped. That matters more here than usual: this config is
    rendered from a single ansible variable into two files, and a key that is
    silently ignored on one side is exactly how the two drift apart.
    """
    if not d:
        return MusicConfig()
    scalars = {k: v for k, v in d.items()
               if k not in ("outputs", "dj", "loudness", "candidates")}
    for key in ("library_root", "mpv_binary", "mpv_socket", "switchaudio_binary", "db_path"):
        if key in scalars:
            scalars[key] = Path(scalars[key])
    if "exposed_to" in scalars:
        scalars["exposed_to"] = tuple(scalars["exposed_to"])
    if "connect_timeout_s" in scalars:
        scalars["connect_timeout_s"] = float(scalars["connect_timeout_s"])
    if "enabled" in scalars:
        scalars["enabled"] = bool(scalars["enabled"])
    return MusicConfig(
        outputs=tuple(MusicOutput(**o) for o in d.get("outputs", ())),
        dj=_parse_dj(d.get("dj")),
        loudness=_parse_loudness(d.get("loudness")),
        candidates=_parse_candidates(d.get("candidates")),
        **scalars,
    )


def _parse_candidates(d: dict[str, Any] | None) -> CandidatesConfig:
    """Parse the optional ``music.candidates:`` block. Strict, like its parent."""
    if not d:
        return CandidatesConfig()
    scalars = dict(d)
    for key in ("fresh_hours", "pool_factor", "default_minutes"):
        if key in scalars:
            scalars[key] = float(scalars[key])
    if "max_tracks" in scalars:
        scalars["max_tracks"] = int(scalars["max_tracks"])
    if "exclude_genres" in scalars:
        scalars["exclude_genres"] = tuple(str(g) for g in scalars["exclude_genres"] or ())
    return CandidatesConfig(**scalars)


def _parse_loudness(d: dict[str, Any] | None) -> LoudnessConfig:
    """Parse the optional ``music.loudness:`` block.

    Strict, like the rest of ``music:``. The keys that used to sit directly
    under ``music:`` (``normalize``, ``target_lufs``) are rejected there rather
    than quietly ignored, so a stale render fails the load instead of playing
    at a level nobody chose.
    """
    if not d:
        return LoudnessConfig()
    scalars = dict(d)
    for key in ("target_lufs", "boost_ceiling_dbtp", "assumed_lufs",
                "furniture_below_album_lu", "furniture_max_s"):
        if key in scalars:
            scalars[key] = float(scalars[key])
    if "normalize" in scalars:
        scalars["normalize"] = bool(scalars["normalize"])
    return LoudnessConfig(**scalars)


def _parse_dj(d: dict[str, Any] | None) -> DjConfig:
    """Parse the optional ``music.dj:`` block.

    Splats strictly, like its parent and for the same reason: this is rendered
    from ansible, and a key silently ignored on one side is how two renders of
    one declaration drift apart.
    """
    if not d:
        return DjConfig()
    scalars = dict(d)
    for key in ("piper_binary", "voice_model", "render_dir"):
        if key in scalars:
            scalars[key] = Path(scalars[key])
    for key in ("length_scale", "sentence_silence", "drive",
                "max_true_peak_dbfs", "keep_hours"):
        if key in scalars:
            scalars[key] = float(scalars[key])
    if scalars.get("target_lufs") is not None:
        scalars["target_lufs"] = float(scalars["target_lufs"])
    for key in ("pad_ms", "sample_rate", "channels", "max_links"):
        if key in scalars:
            scalars[key] = int(scalars[key])
    if "enabled" in scalars:
        scalars["enabled"] = bool(scalars["enabled"])
    return DjConfig(**scalars)


def _validate_gate(cfg: "Config") -> None:
    """An exposed_to naming no real agent would gate nobody, silently; and
    handlers are the one part of the gate that needs a scorer."""
    if not cfg.gate.enabled:
        return
    agent_ids = {ac.id for ac in cfg.agents}
    if unknown := sorted(set(cfg.gate.exposed_to) - agent_ids):
        raise ValueError(
            f"gate.exposed_to references unknown agent(s): {unknown}; "
            f"known agents: {sorted(agent_ids)}"
        )
    if not 0.0 <= cfg.gate.min_confidence <= 1.0:
        raise ValueError(f"gate.min_confidence must be within [0, 1], got {cfg.gate.min_confidence}")
    if cfg.gate.context_turns < 0:
        raise ValueError(f"gate.context_turns must be >= 0, got {cfg.gate.context_turns}")
    if cfg.gate.timeout_s <= 0:
        raise ValueError(f"gate.timeout_s must be > 0, got {cfg.gate.timeout_s}")
    ids = [h.id for h in cfg.gate.handlers]
    if "none" in ids:
        raise ValueError("gate.handlers: id 'none' is reserved (it is the LLM's option)")
    if dupes := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"gate.handlers: duplicate id(s): {dupes}")
    if cfg.gate.handlers and cfg.systemone is None:
        raise ValueError(
            "gate.handlers need a scorer: add a systemone block, or remove the "
            "handlers (a gate with none passes every turn through)"
        )


def _validate_smart_retrieval(cfg: "Config") -> None:
    """Smart retrieval needs a scorer, and says so at load rather than at turn 1.

    The runtime is deliberately forgiving -- a scorer that errors or times out
    mid-turn drops back to ordinary retrieval, because an outage must cost
    precision and never retrieval. But a config that asks for the feature with
    no scorer configured is not an outage, it is a mistake, and it would
    otherwise present as the feature silently never running. Same reasoning as
    gate.handlers, which refuses for the same reason.
    """
    if cfg.memory_retrieval.smart_retrieval.enabled and cfg.systemone is None:
        raise ValueError(
            "memory_retrieval.smart_retrieval.enabled is true but no scorer is "
            "configured: add a systemone block, or set it to false (ordinary "
            "retrieval is the supported default and needs no scorer)"
        )


def _validate_music(cfg: "Config") -> None:
    """Reject a music block that cannot mean what it says.

    Every check here is for a mistake that would otherwise surface as a puzzling
    runtime refusal rather than as a bad config: a tool exposed to an agent that
    does not exist is simply never built, and two outputs sharing an id makes
    which speaker answers depend on list order.
    """
    if not cfg.music.enabled:
        return
    agent_ids = {ac.id for ac in cfg.agents}
    if unknown := sorted(set(cfg.music.exposed_to) - agent_ids):
        raise ValueError(
            f"music.exposed_to references unknown agent(s): {unknown}; "
            f"known agents: {sorted(agent_ids)}"
        )
    if not cfg.music.outputs:
        raise ValueError("music.enabled is true but no outputs are configured")
    ids = [o.id for o in cfg.music.outputs]
    if dupes := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"music.outputs: duplicate output id(s): {dupes}")
    if len(defaults := [o.id for o in cfg.music.outputs if o.default]) > 1:
        raise ValueError(
            f"music.outputs: more than one output marked default: {sorted(defaults)}"
        )
    _validate_loudness(cfg.music.loudness)
    _validate_candidates(cfg.music.candidates)
    _validate_dj(cfg.music.dj)


def _validate_candidates(c: CandidatesConfig) -> None:
    if c.fresh_hours < 0:
        raise ValueError(f"music.candidates.fresh_hours must be >= 0 (got {c.fresh_hours})")
    if c.pool_factor < 1:
        raise ValueError(
            "music.candidates.pool_factor must be >= 1 — below it the pool is smaller "
            f"than the set asked for (got {c.pool_factor})"
        )
    if c.default_minutes <= 0:
        raise ValueError(f"music.candidates.default_minutes must be > 0 (got {c.default_minutes})")
    if c.max_tracks < 1:
        raise ValueError(f"music.candidates.max_tracks must be >= 1 (got {c.max_tracks})")


def _validate_loudness(ld: LoudnessConfig) -> None:
    """Reject loudness settings that would do something nobody could intend.

    Checked whether or not ``normalize`` is on: the furniture thresholds apply
    to every shuffle, and a bad target discovered only on the day someone
    switches normalisation on is a worse time to find out.
    """
    if ld.boost_ceiling_dbtp > 0:
        raise ValueError(
            "music.loudness.boost_ceiling_dbtp must be <= 0 — above it the boost "
            f"itself would clip (got {ld.boost_ceiling_dbtp})"
        )
    for key in ("target_lufs", "assumed_lufs"):
        if (v := getattr(ld, key)) >= 0:
            raise ValueError(f"music.loudness.{key} is in LUFS and must be < 0 (got {v})")
    for key in ("furniture_below_album_lu", "furniture_max_s"):
        if (v := getattr(ld, key)) <= 0:
            raise ValueError(f"music.loudness.{key} must be > 0 (got {v})")


def _validate_dj(dj: DjConfig) -> None:
    """Reject DJ settings that cannot do what they say.

    Deliberately does NOT check that the piper binary or the voice model exist.
    They live in the voice venv, which is a separate role's business and may be
    mid-install; a missing one surfaces at render time as a refusal naming the
    path, which is clearer than refusing to boot the gateway. What is checked
    here is the set of values that would otherwise produce audio nobody
    intended and no error at all.
    """
    if not dj.enabled:
        return
    if dj.drive < 1.0:
        # Below unity asoftclip attenuates. The whole point of the saturator is
        # that a flat render sits ~8 dB under the music at an identical peak;
        # a drive under 1.0 makes that worse while looking like a tuning.
        raise ValueError(f"music.dj.drive must be >= 1.0 (got {dj.drive})")
    if dj.length_scale <= 0:
        raise ValueError(f"music.dj.length_scale must be > 0 (got {dj.length_scale})")
    if dj.channels not in (1, 2):
        raise ValueError(f"music.dj.channels must be 1 or 2 (got {dj.channels})")
    if dj.sample_rate <= 0:
        raise ValueError(f"music.dj.sample_rate must be > 0 (got {dj.sample_rate})")
    if dj.pad_ms < 0:
        raise ValueError(f"music.dj.pad_ms must be >= 0 (got {dj.pad_ms})")
    if dj.max_links < 1:
        raise ValueError(f"music.dj.max_links must be >= 1 (got {dj.max_links})")
    if dj.max_true_peak_dbfs > 0:
        raise ValueError(
            "music.dj.max_true_peak_dbfs must be <= 0 — it is headroom below full "
            f"scale, not a target (got {dj.max_true_peak_dbfs})"
        )


def _parse_agent(d: dict[str, Any]) -> AgentConfig:
    m = d["matrix"]
    return AgentConfig(
        id=d["id"],
        workspace=Path(d["workspace"]),
        primary_model=d["primary_model"],
        compaction_model=d.get("compaction_model"),
        extra_paths=tuple(d.get("extra_paths", [])),
        max_spawn_depth=d.get("max_spawn_depth", 1),
        can_spawn=_parse_can_spawn(d.get("can_spawn")),
        num_predict=d.get("num_predict"),
        max_tool_turns=d.get("max_tool_turns"),
        matrix=MatrixAccountConfig(
            user_id=m["user_id"],
            homeserver=m["homeserver"],
            access_token_file=Path(m["access_token_file"]),
            device_id=m["device_id"],
            device_name=m["device_name"],
            store_path=Path(m["store_path"]),
            encryption=m.get("encryption", True),
            auto_join=m.get("auto_join", "always"),
            allow_bots=m.get("allow_bots", "mentions"),
            allow_from=tuple(m.get("allow_from", [])),
            password_file=Path(m["password_file"]) if m.get("password_file") else None,
            force_cross_signing_replace=bool(m.get("force_cross_signing_replace", False)),
        ),
    )
