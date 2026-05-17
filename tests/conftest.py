"""Shared fakes + frozen-config factories for the clawsome suite.

PUBLIC MIRROR: this directory is synced to the public repo. Use ONLY
neutral placeholders — @bot:example.org / @user-1:example.org,
http://ollama.invalid, http://searx.invalid, tmp_path. Never a real
hostname, IP, username, path, or deployment agent/persona name.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import pytest

from claw.config import (
    AgentConfig,
    CommandsConfig,
    CompactionConfig,
    Config,
    CronConfig,
    LifecycleConfig,
    MatrixAccountConfig,
    MemoryFlushConfig,
    MemoryRetrievalConfig,
    OllamaConfig,
    SearxngConfig,
    SubagentsConfig,
)
from claw.transcript import TranscriptStore


# --- Fakes (duck-typed; Agent injects these deps by reference) -------------

class FakeChannel:
    """Implements the claw.channel.base.Channel Protocol.

    Records every send as (peer_id, text) in ``.sent`` for assertions.
    ``typing`` is a no-op async context manager (mirrors base.no_typing).
    """

    name = "matrix"

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.started = False
        self.shutdown_called = False
        self.cleared_typing: list[str] = []
        self.on_message = None

    async def start(self, on_message) -> None:
        self.started = True
        self.on_message = on_message

    async def send(self, peer_id: str, text: str) -> None:
        self.sent.append((peer_id, text))

    async def shutdown(self) -> None:
        self.shutdown_called = True

    @contextlib.asynccontextmanager
    async def typing(self, peer_id: str):
        yield

    async def clear_typing(self, peer_id: str) -> None:
        self.cleared_typing.append(peer_id)


class FakeOllama:
    """OllamaClient-like. ``run_turn`` returns a fixed
    (new_messages, final_text, final_thinking); calls are recorded.
    """

    def __init__(
        self,
        final_text: str = "hello back",
        final_thinking: str = "",
        new_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self._final_text = final_text
        self._final_thinking = final_thinking
        self._new_messages = new_messages
        self.turns: list[dict[str, Any]] = []
        self.summaries: list[dict[str, Any]] = []

    async def run_turn(self, **kw) -> tuple[list[dict[str, Any]], str, str]:
        self.turns.append(kw)
        new = self._new_messages
        if new is None:
            new = [{"role": "assistant", "content": self._final_text}]
        return list(new), self._final_text, self._final_thinking

    async def chat_once(self, **kw) -> str:
        return self._final_text

    async def summarize(self, *args, **kw) -> str:
        self.summaries.append({"args": args, "kw": kw})
        return "RECAP SUMMARY"


class FakeMemory:
    """MemoryIndex-like. ``retrieve_markdown`` returns a fixed string;
    queries are recorded.
    """

    def __init__(self, markdown: str = "") -> None:
        self._markdown = markdown
        self.queries: list[str] = []
        self.warmed = False

    async def warmup_async(self) -> None:
        self.warmed = True

    async def retrieve_markdown(
        self, query: str, *, top_n: int = 5, compact: bool = True
    ) -> str:
        self.queries.append(query)
        return self._markdown

    async def reindex_if_stale(self) -> dict[str, Any]:
        return {}


# --- Fixtures --------------------------------------------------------------

@pytest.fixture
def fake_channel() -> FakeChannel:
    return FakeChannel()


@pytest.fixture
def fake_ollama() -> FakeOllama:
    return FakeOllama()


@pytest.fixture
def fake_memory() -> FakeMemory:
    return FakeMemory()


@pytest.fixture
def transcripts(tmp_path: Path) -> TranscriptStore:
    return TranscriptStore(tmp_path / "transcripts")


@pytest.fixture
def make_matrix_account():
    """Frozen MatrixAccountConfig factory — neutral placeholders only."""

    def _make(tmp_path: Path, **over) -> MatrixAccountConfig:
        d: dict[str, Any] = dict(
            user_id="@bot:example.org",
            homeserver="https://example.org",
            access_token_file=tmp_path / "bot.token",
            device_id="CLAW_TEST",
            device_name="Claw Test",
            store_path=tmp_path / "store",
        )
        d.update(over)
        return MatrixAccountConfig(**d)

    return _make


@pytest.fixture
def make_agent_cfg(make_matrix_account):
    """Frozen AgentConfig factory. Required: id, workspace, primary_model,
    matrix. Everything else defaulted by the dataclass.
    """

    def _make(tmp_path: Path, **over) -> AgentConfig:
        d: dict[str, Any] = dict(
            id="example",
            workspace=tmp_path / "ws",
            primary_model="test-model",
            matrix=make_matrix_account(tmp_path),
        )
        d.update(over)
        return AgentConfig(**d)

    return _make


@pytest.fixture
def make_cfg(make_agent_cfg):
    """Frozen Config factory — builds the full required object graph by
    hand (no from_dict). Override any top-level field via **over.
    """

    def _make(tmp_path: Path, **over) -> Config:
        d: dict[str, Any] = dict(
            verbose=False,
            ollama=OllamaConfig(
                base_url="http://ollama.invalid",
                default_compaction_model="test-compact-model",
            ),
            memory_retrieval=MemoryRetrievalConfig(),
            searxng=SearxngConfig(base_url="http://searx.invalid"),
            cron=CronConfig(enabled=False, jobs_file=tmp_path / "jobs.json"),
            subagents=SubagentsConfig(
                max_concurrent=1,
                max_children_per_agent=1,
                default_model="test-model",
                personas={},
            ),
            compaction=CompactionConfig(),
            memory_flush=MemoryFlushConfig(),
            lifecycle=LifecycleConfig(),
            commands=CommandsConfig(allow=("@user-1:example.org",)),
            agents=(make_agent_cfg(tmp_path),),
            tz=None,
        )
        d.update(over)
        return Config(**d)

    return _make
