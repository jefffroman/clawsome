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
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any

from claw.channel.envelope import strip_inbound_envelope
from claw.config import MemoryRetrievalConfig
from claw.memory import MemoryIndex

# Rows the agent did not receive from a person: recaps, system notes, and the
# in-band admin commands, none of which are turns retrieval runs for.
_SYNTHETIC = ("## Pre-compaction Recap", "⚙️", "[SYSTEM (out-of-band")


def _messages(workspace: Path, n: int, seed: int) -> list[str]:
    """``n`` distinct user messages from the workspace's transcripts."""
    seen: set[str] = set()
    out: list[str] = []
    for path in sorted(workspace.glob("transcripts/*.jsonl")):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            text = row.get("content")
            if row.get("role") != "user" or not isinstance(text, str):
                continue
            body = strip_inbound_envelope(text).strip()
            key = body.lower()
            if (not 12 < len(body) < 400 or key in seen
                    or body.startswith("%") or text.lstrip().startswith(_SYNTHETIC)):
                continue
            seen.add(key)
            out.append(text)
    random.Random(seed).shuffle(out)
    return out[:n]


def _index(workspace: Path, agent: str) -> MemoryIndex:
    idx = MemoryIndex(agent, workspace, MemoryRetrievalConfig())
    idx.warmup()
    return idx


def collect(args: argparse.Namespace) -> None:
    """Freeze the queries and their ungated candidates."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = _index(Path(args.workspace), args.agent)
    queries = []
    for text in _messages(Path(args.workspace), args.n, args.seed):
        hits, _ = idx._candidates(text, args.candidates)
        queries.append({
            "message": text,
            "candidates": [h["id"] for h in hits],
            "texts": {h["id"]: h["text"] for h in hits},
        })
    (out / "queries.json").write_text(json.dumps(queries, indent=1))
    print(f"{len(queries)} queries -> {out / 'queries.json'}")


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
    async with httpx.AsyncClient(base_url=args.ollama, timeout=args.timeout) as client:
        for i, q in enumerate(queries):
            if str(i) in grades:
                continue
            ids = list(q["candidates"])
            random.Random(i).shuffle(ids)
            listing = "\n\n".join(
                f"[{n + 1}] {q['texts'][cid][:1500]}" for n, cid in enumerate(ids))
            reply = await client.post("/api/chat", json={
                "model": args.model, "stream": False, "format": "json",
                "messages": [
                    {"role": "system", "content": _RUBRIC},
                    {"role": "user",
                     "content": f"USER MESSAGE:\n{q['message']}\n\nMEMORIES:\n{listing}"},
                ]})
            reply.raise_for_status()
            try:
                scored = json.loads(reply.json()["message"]["content"])["grades"]
            except (KeyError, ValueError) as e:
                print(f"  query {i}: unusable reply ({e}); skipped")
                continue
            grades[str(i)] = {
                ids[int(k) - 1]: int(v) for k, v in scored.items()
                if str(k).isdigit() and 1 <= int(k) <= len(ids)
            }
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
    """
    n = shown = rel = clear = 0
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
    ):
        print(f"  {label:42} {r[key]:.2f}")


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
    c.set_defaults(fn=collect)

    g = sub.add_parser("grade", help="label each pair with a referee model")
    common(g, workspace=False)
    g.add_argument("--model", required=True, help="referee model; use a strong one")
    g.add_argument("--ollama", default="http://127.0.0.1:11434")
    g.add_argument("--timeout", type=float, default=900.0)
    g.set_defaults(fn=lambda a: asyncio.run(_grade(a)))

    s = sub.add_parser("score", help="score the current code against the labels")
    common(s)
    s.add_argument("--top-n", type=int, default=5)
    s.set_defaults(fn=score)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
