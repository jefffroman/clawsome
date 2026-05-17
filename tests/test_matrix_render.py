"""channel/matrix.py — markdown render + chunking.

Imports claw.channel.matrix (markdown-it / linkify-it) — run under the
gateway venv interpreter.

PUBLIC MIRROR: neutral placeholders only.
"""
from __future__ import annotations

from claw.channel.matrix import MATRIX_CHUNK_MAX, _chunk, _to_html


def test_chunk_max_constant():
    assert MATRIX_CHUNK_MAX == 16000


def test_chunk_small_and_empty():
    assert _chunk("") == []
    assert _chunk("short") == ["short"]


def test_chunk_oversize_splits_on_paragraph_boundary():
    chunks = _chunk("a" * 5 + "\n\n" + "b" * 5, limit=6)
    assert chunks == ["aaaaa", "bbbbb"]
    assert all(0 < len(c) <= 6 for c in chunks)


def test_to_html_markdown_and_raw_html_passthrough():
    assert "<strong>b</strong>" in _to_html("**b**")
    assert "<br" in _to_html("a\nb")                 # breaks=True
    assert '<a href="http://example.org"' in _to_html("see http://example.org")
    # The channel uses the "commonmark" preset, which sets html=True — raw
    # HTML the model emits passes through *verbatim* (it is NOT sanitized
    # here; the matrix layer/client is responsible). Pin this so a preset
    # change that silently starts escaping (or keeps passing) is caught.
    assert "<b>x</b>" in _to_html("keep <b>x</b>")
    # Bare special chars are still entity-escaped.
    assert "&amp;" in _to_html("a & b")
