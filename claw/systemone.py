"""Client for a System One decision endpoint.

System One is a request/response shape for typed decisions rather than
generated text: a ``state`` plus a map of named questions — ``choice`` (pick
one option), ``noul`` (probability of yes), ``score`` (rate against ordered
levels) — returns one answer per question, each carrying the full probability
distribution. Nothing is generated, so an answer is always well-formed; that
is a guarantee about shape, not about correctness.

This client speaks the wire format and nothing else. It is deliberately thin:
callers decide what to do with probabilities, and callers decide what a
failure means (the decision gate treats every error as "let the LLM answer").
"""

from __future__ import annotations

from typing import Any

import httpx


class SystemOneError(RuntimeError):
    """The endpoint answered, but not with a usable result."""


class SystemOneClient:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def ask(
        self, state: Any, questions: dict[str, dict[str, Any]],
        model: str = "local",
    ) -> dict[str, dict[str, Any]]:
        """POST one request; return its ``answers`` map (question id -> answer).

        Raises ``httpx.HTTPError`` on transport failure or timeout, and
        ``SystemOneError`` on a non-2xx status or a response missing any
        requested answer.
        """
        resp = await self._client.post(
            "/v1/systemone",
            json={"state": state, "model": model, "questions": questions},
        )
        if resp.status_code != 200:
            raise SystemOneError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        answers = resp.json().get("answers")
        if not isinstance(answers, dict) or set(questions) - set(answers):
            raise SystemOneError(f"response lacks answers for {sorted(questions)}")
        return answers
