"""Regression tests for issue #23 — synto crashed with UnicodeEncodeError when
the console stream could not encode the ✓/✗ status glyphs (Windows cp1252, or an
ascii/POSIX locale). `_ensure_utf8_streams` reconfigures such streams to UTF-8
before any output, so the glyphs are emittable instead of crashing."""

from __future__ import annotations

import io

from synto.cli import _ensure_utf8_streams


def test_cp1252_stream_is_switched_to_utf8_and_glyph_survives(monkeypatch):
    # A real cp1252-backed text stream is the exact condition that crashed #23:
    # writing "✓" through it raises UnicodeEncodeError before the fix.
    buf = io.BytesIO()
    stream = io.TextIOWrapper(buf, encoding="cp1252")
    monkeypatch.setattr("sys.stdout", stream)
    monkeypatch.setattr("sys.stderr", stream)

    _ensure_utf8_streams()

    assert stream.encoding == "utf-8"
    stream.write("✓")  # would raise on a cp1252 stream
    stream.flush()
    assert "✓".encode() in buf.getvalue()


def test_utf8_stream_is_left_untouched(monkeypatch):
    # Healthy terminals must not be disturbed: no reconfigure on the fast path.
    reconfigured: list[str] = []

    class FakeStream:
        encoding = "utf-8"

        def reconfigure(self, *, encoding):
            reconfigured.append(encoding)

    monkeypatch.setattr("sys.stdout", FakeStream())
    monkeypatch.setattr("sys.stderr", FakeStream())

    _ensure_utf8_streams()

    assert reconfigured == []


def test_reconfigure_failure_is_swallowed_and_none_stream_skipped(monkeypatch):
    # Startup must never crash on an exotic/captured stream or a missing one.
    class RaisingStream:
        encoding = "cp1252"

        def reconfigure(self, *, encoding):
            raise OSError("not reconfigurable")

    monkeypatch.setattr("sys.stdout", RaisingStream())
    monkeypatch.setattr("sys.stderr", None)

    _ensure_utf8_streams()  # must not raise
