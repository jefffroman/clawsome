"""Measure what per-turn memory retrieval actually returns.

``docs/operations.md`` ("Tuning the gate") describes re-calibrating the
relevance weights as four steps. This is those steps as a tool, in three
commands that write to one directory:

    python -m claw.evaluate collect --workspace DIR --agent ID --out DIR
    python -m claw.evaluate grade   --out DIR --model <referee>
    python -m claw.evaluate score   --out DIR --workspace DIR --agent ID

``collect`` freezes a sample of real messages with the ungated candidates for
each. ``grade`` asks a referee model for a 0/1/2 relevance label per
(message, memory) pair. ``score`` runs the CURRENT code over the frozen
queries and reports what it retrieves against those labels.

The grades are the expensive artifact and the reason to keep the directory:

* They are a **frozen baseline**. A referee is not deterministic, so regrading
  produces different labels and every earlier measurement stops being
  comparable. Grade once; keep it.
* They are **mechanism-independent**. A label says whether this memory helps
  answer this message — which does not change when scoring, term selection or
  fusion changes. So any retrieval change can be scored against them with no
  model calls at all, which is the difference between validating a change and
  asserting it.

Tests cannot see this. A change to the retrieval path can leave every test
green while moving what the agent is told, and the drift WARNING watches
vector distance only, so a keyword-side change does not raise it.

The sample holds real messages and real memory text, so the output directory
is private data — keep it where backups reach and out of version control.
Nothing here writes into the workspace it reads.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import math
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any

from claw.channel.envelope import strip_inbound_envelope
from claw.gate import build_state
from claw.config import MemoryRetrievalConfig, SmartRetrievalConfig
from claw.memory import MemoryIndex
from claw.systemone import SystemOneClient

# Rows the agent did not receive from a person: recaps, system notes, and the
# in-band admin commands, none of which are turns retrieval runs for.
_SYNTHETIC = ("## Pre-compaction Recap", "⚙️", "[SYSTEM (out-of-band")


def _messages(workspace: Path, n: int, seed: int,
              context_turns: int = 0) -> list[tuple[str, list[dict[str, str]]]]:
    """``n`` distinct user messages, each with the turns that preceded it.

    ``context_turns`` freezes that many prior user/assistant messages beside
    each sampled one, built exactly the way the decision gate builds its state
    — real turns only, envelopes stripped, synthetic rows dropped.

    Freezing it is the whole point. A referee that grades a memory against the
    bare message, while the thing being evaluated reads the conversation, is
    judging a different question and will mark down exactly the improvement
    being tested. Whatever context the consumer sees, the referee must see.
    """
    seen: set[str] = set()
    out: list[tuple[str, list[dict[str, str]]]] = []
    for path in sorted(workspace.glob("transcripts/*.jsonl")):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        prev_rows: list[dict[str, Any]] = []
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            prev_rows.append(row)
            text = row.get("content")
            if row.get("role") != "user" or not isinstance(text, str):
                continue
            body = strip_inbound_envelope(text).strip()
            key = body.lower()
            if (not 12 < len(body) < 400 or key in seen
                    or body.startswith("%") or text.lstrip().startswith(_SYNTHETIC)):
                continue
            seen.add(key)
            prior = (build_state(prev_rows[:-1], body, context_turns)[:-1]
                     if context_turns else [])
            out.append((text, prior))
    random.Random(seed).shuffle(out)
    return out[:n]


def _index(workspace: Path, agent: str,
           cfg: MemoryRetrievalConfig | None = None) -> MemoryIndex:
    idx = MemoryIndex(agent, workspace, cfg or MemoryRetrievalConfig())
    idx.warmup()
    return idx


def collect(args: argparse.Namespace) -> None:
    """Freeze the queries and the candidates a turn could draw on.

    ``--extend`` keeps an existing file's MESSAGES and only widens each one's
    candidate list. That is the difference between adding to the labelled set
    and replacing it: the messages are the sample, and re-drawing them (a new
    seed, a longer transcript, a different ``--n``) would strand every grade
    already paid for. Widening is additive -- `grade` fills the gaps and the
    earlier labels still describe the same pairs.

    Candidates include graph expansion, matching what the per-turn path
    actually considers. Collect WIDER than production injects: the referee
    costs one call per message whatever the list length, so a pool that
    already covers the next experiment is far cheaper than a second run.
    """
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = _index(Path(args.workspace), args.agent)
    path = out / "queries.json"

    previous: list[dict[str, Any]] = []
    if args.extend:
        if not path.exists():
            raise SystemExit(f"--extend needs an existing {path}")
        previous = json.loads(path.read_text())
    if args.extend:
        messages = [(q["message"], q.get("context", [])) for q in previous]
    else:
        messages = _messages(Path(args.workspace), args.n, args.seed,
                             args.context_turns)

    live = idx._keyword_index()["by_id"] if idx._keyword_index() else {}
    queries, added, drifted = [], 0, []
    for n, (text, context) in enumerate(messages):
        hits, _ = idx._candidates(text, args.candidates, expand=True)
        prior = previous[n] if args.extend else {"candidates": [], "texts": {}}
        # A chunk id is positional ({source}:{index}), so an edit to a note
        # between runs re-points an id at different content and would silently
        # mislabel it. `texts` was frozen for exactly this: compare, don't
        # trust. Checked over the whole frozen set rather than only the ids
        # still ranking, because a label outliving its content is wrong
        # whether or not that chunk happens to be a candidate today -- and an
        # id that fell out of the pool is precisely the one nobody re-reads.
        for cid, was in prior["texts"].items():
            now = live.get(cid)
            if now is not None and now["text"] != was:
                drifted.append(cid)
        merged = list(dict.fromkeys(list(prior["candidates"]) + [h["id"] for h in hits]))
        added += len(merged) - len(prior["candidates"])
        queries.append({
            "message": text,
            # Frozen with the message, so every later pass — the referee, the
            # scorer, any arm — reads the same conversation. [] means the set
            # was collected without context and can only judge the bare
            # message; the two are different sets and must not be mixed.
            "context": context,
            "candidates": merged,
            "texts": {**prior["texts"], **{h["id"]: h["text"] for h in hits}},
        })
    if drifted and not args.drop_drifted:
        raise SystemExit(
            f"{len(set(drifted))} frozen candidate(s) now hold different text "
            f"(e.g. {drifted[0]}). A chunk id is positional, so an edited or "
            f"re-sectioned note re-points one at content nobody graded. Pass "
            f"--drop-drifted to discard just those labels and re-grade them, "
            f"or collect a fresh set.")
    if drifted:
        # The labels describe text that is no longer at these ids, so they are
        # not labels any more. Dropping the PAIRS (not the messages) costs no
        # extra referee calls: per-pair resume re-grades them inside the run
        # that is already visiting those messages for their new candidates.
        gpath = out / "grades.json"
        if gpath.exists():
            grades = json.loads(gpath.read_text())
            stale = {cid for cid in drifted}
            dropped = 0
            for key, per in grades.items():
                for cid in list(per):
                    if cid in stale:
                        del per[cid]
                        dropped += 1
            gpath.write_text(json.dumps(grades, indent=1))
            print(f"dropped {dropped} stale label(s) across "
                  f"{len(stale)} id(s): {', '.join(sorted(stale)[:4])}"
                  + (" ..." if len(stale) > 4 else ""))
    path.write_text(json.dumps(queries, indent=1))
    print(f"{len(queries)} queries -> {path}"
          + (f" (+{added} new candidates)" if args.extend else ""))


_RUBRIC = (
    "You grade memory retrieval for a personal assistant. The assistant is about "
    "to reply to the user's message; before it does, stored memory notes are put "
    "in its prompt. Grade how useful each numbered memory would be for replying "
    "to THIS message:\n"
    "2 = clearly relevant: the same specific subject, task, person or fact, and "
    "it would inform the reply\n"
    "1 = somewhat relevant: related context that could plausibly help\n"
    "0 = not relevant: unrelated, or shares only a word or a broad topic\n"
    "Many messages (greetings, acknowledgements) need no memory at all — then "
    'grade everything 0. Answer with JSON only: {"grades": {"<number>": <0|1|2>}} '
    "covering every number."
)


async def _grade(args: argparse.Namespace) -> None:
    import httpx

    out = Path(args.out)
    queries = json.loads((out / "queries.json").read_text())
    path = out / "grades.json"
    grades: dict[str, dict[str, int]] = {}
    if path.exists():                      # resumable: a referee run is slow
        grades = json.loads(path.read_text())
    # Resume is per (message, memory) PAIR. It was per message -- `if str(i) in
    # grades: continue` -- which is correct only while the candidate lists never
    # change. Widening a list (a new candidate source, a deeper pool) then made
    # a re-run skip every message that already had ANY grade, label nothing,
    # print the full count and exit 0. A referee run is the expensive artifact
    # here, and that failed silently in the direction nobody checks.
    async with httpx.AsyncClient(base_url=args.ollama, timeout=args.timeout) as client:
        for i, q in enumerate(queries):
            done = grades.get(str(i), {})
            ids = [c for c in q["candidates"] if c not in done]
            if not ids:
                continue
            random.Random(i).shuffle(ids)
            listing = "\n\n".join(
                f"[{n + 1}] {q['texts'][cid][:1500]}" for n, cid in enumerate(ids))
            # The referee reads whatever conversation was frozen with this
            # message. Grading against the bare message while the consumer
            # reads the thread asks a different question, and marks down the
            # continuation turns that context exists to rescue.
            ctx = "".join(f"{r['role']}: {r['content']}\n" for r in q.get("context", ()))
            head = (f"CONVERSATION SO FAR:\n{ctx}\n" if ctx else "")
            reply = await client.post("/api/chat", json={
                "model": args.model, "stream": False, "format": "json",
                "messages": [
                    {"role": "system", "content": _RUBRIC},
                    {"role": "user",
                     "content": f"{head}USER MESSAGE:\n{q['message']}\n\n"
                                f"MEMORIES:\n{listing}"},
                ]})
            reply.raise_for_status()
            try:
                scored = json.loads(reply.json()["message"]["content"])["grades"]
            except (KeyError, ValueError) as e:
                print(f"  query {i}: unusable reply ({e}); skipped")
                continue
            # Merged, not assigned: `ids` is only the UNGRADED remainder, so
            # assigning would drop every pair a previous run had labelled.
            grades[str(i)] = {**done, **{
                ids[int(k) - 1]: int(v) for k, v in scored.items()
                if str(k).isdigit() and 1 <= int(k) <= len(ids)
            }}
            path.write_text(json.dumps(grades, indent=1))
            print(f"  graded {i + 1}/{len(queries)}", flush=True)
    print(f"{len(grades)} graded -> {path}")


def report(injected: list[list[str]], grades: dict[str, dict[str, int]]) -> dict[str, float]:
    """What an arm retrieved, against the labels.

    ``precision`` counts injected memories the referee called relevant, and
    ``recall`` only the clearly-relevant ones a turn could have had. The last
    two are the ones that matter in use and are easy to forget: whether a turn
    that HAD something relevant got any of it, and whether a turn with nothing
    relevant was correctly left alone. An arm can raise precision simply by
    injecting less, so read them together.

    ``unlabeled`` is the share of injected memories carrying no grade at all.
    Every rate here treats one as irrelevant, so a non-zero value makes
    ``precision`` a FLOOR rather than a measurement -- and that is exactly
    what a change widening the candidate pool past the graded set produces,
    which is when the number is most likely to be misread as a regression.
    """
    n = shown = rel = clear = missing = 0
    recall: list[float] = []
    hit: list[bool] = []
    quiet: list[bool] = []
    for i, ids in enumerate(injected):
        g = grades.get(str(i))
        if g is None:
            continue
        relevant = {k for k, v in g.items() if v >= 1}
        clearly = {k for k, v in g.items() if v == 2}
        n += 1
        shown += len(ids)
        rel += sum(g.get(x, 0) >= 1 for x in ids)
        clear += sum(g.get(x, 0) == 2 for x in ids)
        missing += sum(x not in g for x in ids)
        if clearly:
            recall.append(len(clearly & set(ids)) / len(clearly))
        if relevant:
            hit.append(any(x in relevant for x in ids))
        else:
            quiet.append(not ids)
    return {
        "queries": n,
        "per_query": shown / n if n else 0.0,
        "precision": rel / shown if shown else 0.0,
        "precision_clear": clear / shown if shown else 0.0,
        "recall_clear": statistics.mean(recall) if recall else 0.0,
        "hit_when_relevant": statistics.mean(hit) if hit else 0.0,
        "quiet_when_none": statistics.mean(quiet) if quiet else 0.0,
        "unlabeled": missing / shown if shown else 0.0,
    }


def score(args: argparse.Namespace) -> None:
    """Run the current code over the frozen queries and report."""
    out = Path(args.out)
    queries = json.loads((out / "queries.json").read_text())
    grades = json.loads((out / "grades.json").read_text())
    idx = _index(Path(args.workspace), args.agent)
    injected = []
    for q in queries:
        idx._kw = None
        injected.append([h["id"] for h in idx._relevant(q["message"], args.top_n)])
    r = report(injected, grades)
    print(f"{r['queries']} graded queries, {r['per_query']:.2f} memories per turn\n")
    for label, key in (
        ("relevant (grade >= 1)", "precision"),
        ("clearly relevant (= 2)", "precision_clear"),
        ("recall of clearly relevant", "recall_clear"),
        ("got something, when there was something", "hit_when_relevant"),
        ("stayed quiet, when there was nothing", "quiet_when_none"),
        ("UNLABELED share of what was injected", "unlabeled"),
    ):
        print(f"  {label:42} {r[key]:.2f}")
    if r["unlabeled"] > 0.05:
        print(f"\n  ! {r['unlabeled']:.0%} of injected memories have no grade, and every rate\n"
              f"    above counts them as irrelevant. Re-run `grade` before reading these\n"
              f"    as a comparison against an earlier measurement.")


def _features(idx: MemoryIndex, queries: list[dict[str, Any]], candidates: int,
              ) -> list[tuple[str, str, list[float]]]:
    """(message key, chunk id, feature row) for every current candidate.

    The row is exactly what ``_relevance`` consumes, in its order, so a fitted
    coefficient can be pasted into config without a mapping step. Frozen
    candidates the search no longer returns are skipped: they carry no
    features today and cannot be selected today either.
    """
    rows: list[tuple[str, str, list[float]]] = []
    for i, q in enumerate(queries):
        idx._kw = None
        hits, facts = idx._candidates(q["message"], candidates, expand=True)
        kw_best = facts.get("kw_best", 0.0)
        for h in hits:
            kwv = h["_kw"]
            rows.append((str(i), h["id"], [
                h["_distance"] if h["_distance"] is not None else 9.0,
                math.log1p(kwv),
                kwv / kw_best if kw_best > 0 else 0.0,
                h.get("_traversal", 0.0),
            ]))
    return rows


def _irls(X, y, ridge, iters=60):
    """Ridge-penalised logistic regression by IRLS.

    Hand-rolled on numpy rather than pulled from scipy or scikit-learn,
    neither of which this package depends on. The penalty is not decoration:
    one feature here is zero for most rows and non-zero for a clustered
    minority, which is the classic separation setup -- an unpenalised fit then
    runs a coefficient off toward infinity and reports a model that looks
    superb and predicts nothing. The intercept is left unpenalised.
    """
    import numpy as np
    beta = np.zeros(X.shape[1])
    penalty = np.eye(X.shape[1]) * ridge
    penalty[0, 0] = 0.0
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        pr = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(pr * (1 - pr), 1e-6, None)
        z = eta + (y - pr) / w
        XtW = X.T * w
        try:
            step = np.linalg.solve(XtW @ X + penalty, XtW @ z)
        except Exception:
            break
        if not np.all(np.isfinite(step)):
            break
        done = float(np.max(np.abs(step - beta))) < 1e-8
        beta = step
        if done:
            break
    return beta


def fit(args: argparse.Namespace) -> None:
    """Fit the relevance weights against the labels, and sweep the threshold.

    Step 3 of the procedure in docs/operations.md, which was the only step
    with no tool behind it -- the original fit lived in a scratch script that
    no longer exists, so the shipped weights could be changed by anyone and
    re-derived by nobody.

    Folds are grouped BY MESSAGE. A random split would put candidates from one
    message on both sides, and since they share ``kw_best`` and much of their
    subject matter, that leaks and flatters the result.
    """
    import numpy as np
    out = Path(args.out)
    queries = json.loads((out / "queries.json").read_text())
    grades = json.loads((out / "grades.json").read_text())
    idx = _index(Path(args.workspace), args.agent)

    rows = [(k, cid, f) for k, cid, f in _features(idx, queries, args.candidates)
            if cid in grades.get(k, {})]
    if not rows:
        raise SystemExit("no graded candidates -- run `collect` and `grade` first")
    keys = sorted({k for k, _, _ in rows}, key=int)
    X = np.array([[1.0, *f] for _, _, f in rows])
    y = np.array([1.0 if grades[k][cid] >= 1 else 0.0 for k, cid, _ in rows])
    print(f"{len(rows)} graded candidates over {len(keys)} messages "
          f"({y.mean():.0%} relevant); {int((X[:, 4] > 0).sum())} reached by the graph")

    beta = _irls(X, y, args.ridge)

    folds = [keys[i::args.folds] for i in range(args.folds)]
    aucs = []
    for held in folds:
        tr = [j for j, (k, _, _) in enumerate(rows) if k not in held]
        te = [j for j, (k, _, _) in enumerate(rows) if k in held]
        if not te or not tr or len(set(y[tr].tolist())) < 2:
            continue
        b = _irls(X[tr], y[tr], args.ridge)
        pr = 1.0 / (1.0 + np.exp(-np.clip(X[te] @ b, -30, 30)))
        pos, neg = pr[y[te] == 1], pr[y[te] == 0]
        if len(pos) and len(neg):
            aucs.append(float((pos[:, None] > neg[None, :]).mean()))
    print("grouped %d-fold AUC: %s" % (
        args.folds, f"{statistics.mean(aucs):.3f}" if aucs else "n/a"))

    print("\nfitted weights (memory_retrieval.relevance):")
    for name, b in zip(("offset", "distance", "keyword", "keyword_share", "traversal"), beta):
        print(f"  {name + ':':16} {b: .2f}")

    print("\nthreshold sweep (what the gate would inject):")
    print(f"  {'thr':>5} {'per turn':>9} {'precis.':>8} {'recall':>7} "
          f"{'hit|rel':>8} {'quiet|none':>11} {'unlab.':>7}")
    by_key = collections.defaultdict(list)
    pr_all = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
    for (k, cid, _), pr in zip(rows, pr_all):
        by_key[k].append((float(pr), cid))
    for thr in args.sweep:
        injected = []
        for k in keys:
            ranked = sorted(by_key[k], key=lambda t: -t[0])
            injected.append([cid for pr, cid in ranked if pr >= thr][:args.top_n])
        r = report(injected, {str(n): grades[k] for n, k in enumerate(keys)})
        print(f"  {thr:5.2f} {r['per_query']:9.2f} {r['precision']:8.2f} "
              f"{r['recall_clear']:7.2f} {r['hit_when_relevant']:8.2f} "
              f"{r['quiet_when_none']:11.2f} {r['unlabeled']:7.2f}")
    print("\nThe sweep scores only GRADED candidates, so it is an upper bound on what\n"
          "the live path does. Set the weights, then confirm with `score`.")


def smart(args: argparse.Namespace) -> None:
    """Score SMART RETRIEVAL against the labels, through the shipped code path.

    This runs ``MemoryIndex`` itself with ``smart_retrieval`` enabled and a
    real scorer, so what is measured is what ships -- not a reimplementation
    of it here that can drift. Ordinary retrieval is reported beside it,
    against the same labels and the same messages, because the only question
    worth answering is whether the feature beats not having it.

    Experiments that are still being argued about do NOT belong here; keep
    those in the deployment's own scratch bench, outside this package.
    """
    from claw.systemone import SystemOneClient

    out = Path(args.out)
    queries = json.loads((out / "queries.json").read_text())
    grades = json.loads((out / "grades.json").read_text())
    ws, agent = Path(args.workspace), args.agent

    plain = _index(ws, agent)
    base = [[h["id"] for h in plain._relevant(q["message"], args.top_n)] for q in queries]

    cfg = MemoryRetrievalConfig(smart_retrieval=SmartRetrievalConfig(
        enabled=True, note_chars=args.note_chars, timeout_s=args.timeout,
        promote_margin=args.promote_margin, review_margin=args.review_margin,
        graph_promote_threshold=args.graph_promote_threshold))
    idx = _index(ws, agent, cfg)
    # The client holds no deadline; retrieval states its own per request, so
    # --timeout has to go into the config to mean anything. Its default is the
    # shipped one, because a bench that waits longer than the deployment does
    # measures a system nobody runs -- which is exactly how this was first
    # mismeasured.
    idx.scorer = SystemOneClient(args.jev)

    async def run() -> tuple[list[list[str]], list[float]]:
        loop = asyncio.get_running_loop()
        rows, took = [], []
        try:
            for n, q in enumerate(queries):
                t0 = time.perf_counter()
                # The frozen context, when the set has one — the same
                # conversation the referee graded against. A set collected
                # without context yields [], i.e. the bare message.
                ctx = list(q.get("context", ()))
                state = (ctx + [{"role": "user", "content": q["message"]}]) or None
                got = await idx._smart_retrieve(q["message"], state, loop)
                took.append((time.perf_counter() - t0) * 1000)
                if got is None:                      # scorer failed: fell back
                    got = idx._relevant(q["message"], args.top_n)
                rows.append([h["id"] for h in got])
                print(f"  {n + 1}/{len(queries)}", end="\r", flush=True)
        finally:
            await idx.scorer.aclose()
        return rows, took

    rows, took = asyncio.run(run())
    print(" " * 20, end="\r")
    for label, injected in (("ordinary retrieval", base), ("smart retrieval", rows)):
        r = report(injected, grades)
        print(f"\n{label}: {r['per_query']:.2f} memories per turn")
        for name, key in (
            ("relevant (grade >= 1)", "precision"),
            ("clearly relevant (= 2)", "precision_clear"),
            ("recall of clearly relevant", "recall_clear"),
            ("got something, when there was something", "hit_when_relevant"),
            ("stayed quiet, when there was nothing", "quiet_when_none"),
            ("UNLABELED share of what was injected", "unlabeled"),
        ):
            print(f"  {name:42} {r[key]:.2f}")
    print(f"\nscorer: {statistics.mean(took):.0f} ms per turn "
          f"(median {statistics.median(took):.0f}, max {max(took):.0f})")
    print("A turn whose candidates all sit outside the band asks nothing and "
          "costs nothing,\nso the mean is over a mix of asked and unasked turns.")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="claw.evaluate", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser, workspace: bool = True) -> None:
        sp.add_argument("--out", required=True, help="directory for the frozen set")
        if workspace:
            sp.add_argument("--workspace", required=True, help="the agent's workspace")
            sp.add_argument("--agent", required=True, help="the agent id")

    c = sub.add_parser("collect", help="freeze queries and their ungated candidates")
    common(c)
    c.add_argument("--n", type=int, default=50)
    c.add_argument("--candidates", type=int, default=20)
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--context-turns", type=int, default=0, dest="context_turns",
                   help="freeze this many prior turns beside each message, so "
                        "the referee and the scorer judge the same conversation")
    c.add_argument("--extend", action="store_true",
                   help="keep an existing file's messages; only widen candidate lists")
    c.add_argument("--drop-drifted", action="store_true",
                   help="discard labels whose chunk now holds different text, so they re-grade")
    c.set_defaults(fn=collect)

    g = sub.add_parser("grade", help="label each pair with a referee model")
    common(g, workspace=False)
    g.add_argument("--model", required=True, help="referee model; use a strong one")
    g.add_argument("--ollama", default="http://127.0.0.1:11434")
    g.add_argument("--timeout", type=float, default=900.0)
    g.set_defaults(fn=lambda a: asyncio.run(_grade(a)))

    f = sub.add_parser("fit", help="fit the relevance weights against the labels")
    common(f)
    f.add_argument("--candidates", type=int, default=20)
    f.add_argument("--top-n", type=int, default=5)
    f.add_argument("--ridge", type=float, default=1.0)
    f.add_argument("--folds", type=int, default=5)
    f.add_argument("--sweep", type=float, nargs="+",
                   default=[0.3, 0.35, 0.4, 0.45, 0.5, 0.6])
    f.set_defaults(fn=fit)

    a = sub.add_parser("smart", help="score smart retrieval against the labels")
    common(a)
    a.add_argument("--top-n", type=int, default=5, help="ordinary retrieval's cap")
    a.add_argument("--note-chars", type=int, default=200)
    a.add_argument("--promote-margin", type=float, default=0.10)
    a.add_argument("--review-margin", type=float, default=0.30)
    a.add_argument("--graph-promote-threshold", type=float, default=0.10,
                   dest="graph_promote_threshold")
    a.add_argument("--jev", default="http://127.0.0.1:11502")
    a.add_argument("--timeout", type=float, default=SmartRetrievalConfig().timeout_s,
                   help="retrieval's scorer deadline; defaults to the shipped one")
    a.set_defaults(fn=smart)

    s = sub.add_parser("score", help="score the current code against the labels")
    common(s)
    s.add_argument("--top-n", type=int, default=5)
    s.set_defaults(fn=score)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
