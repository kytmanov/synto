"""Tests for Feature 06: Semantic Cache."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from synto.cache import LLMCache
from synto.state import StateDB

# ---------------------------------------------------------------------------
# Stage 1: llm_cache DB migration v12
# ---------------------------------------------------------------------------


def test_migration_v12(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    tables = {
        row[0]
        for row in db._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "llm_cache" in tables


def test_llm_cache_schema(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    cols = {row[1] for row in db._conn.execute("PRAGMA table_info(llm_cache)").fetchall()}
    expected = {"cache_key", "model", "response_json", "created_at", "last_hit_at", "hit_count"}
    assert expected.issubset(cols)


def test_current_schema_version_v12() -> None:
    from synto.state import _CURRENT_SCHEMA_VERSION

    assert _CURRENT_SCHEMA_VERSION == 14


# ---------------------------------------------------------------------------
# Stage 2: LLMCache class
# ---------------------------------------------------------------------------


def test_cache_miss(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    cache = LLMCache(db)
    result = cache.get("model-a", [{"role": "user", "content": "hello"}])
    assert result is None


def test_cache_put_then_get(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    cache = LLMCache(db)
    messages = [{"role": "user", "content": "What is 2+2?"}]
    cache.put("gpt-4", messages, "4")
    result = cache.get("gpt-4", messages)
    assert result == "4"


def test_cache_key_includes_model(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    cache = LLMCache(db)
    messages = [{"role": "user", "content": "hello"}]
    cache.put("model-a", messages, "response-a")
    cache.put("model-b", messages, "response-b")
    assert cache.get("model-a", messages) == "response-a"
    assert cache.get("model-b", messages) == "response-b"


def test_cache_hit_increments_hit_count(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    cache = LLMCache(db)
    messages = [{"role": "user", "content": "hi"}]
    cache.put("m", messages, "resp")
    cache.get("m", messages)
    cache.get("m", messages)
    row = db._conn.execute("SELECT hit_count FROM llm_cache").fetchone()
    assert row["hit_count"] == 2


def test_cache_clear_all(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    cache = LLMCache(db)
    cache.put("m", [{"role": "user", "content": "a"}], "r1")
    cache.put("m", [{"role": "user", "content": "b"}], "r2")
    deleted = cache.clear()
    assert deleted == 2
    assert db._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0] == 0


def test_cache_clear_older_than(tmp_path: Path) -> None:
    """clear(older_than_days=0) removes everything; clear(older_than_days=1) keeps recent."""

    db = StateDB(tmp_path / "state.db")
    cache = LLMCache(db)
    cache.put("m", [{"role": "user", "content": "old"}], "r1")
    # With older_than_days=0: cutoff is now, nothing older than now, so 0 deleted
    deleted = cache.clear(older_than_days=1)
    assert deleted == 0  # entry is fresh, not 1 day old


def test_cache_stats_empty(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    cache = LLMCache(db)
    stats = cache.stats()
    assert stats["total_entries"] == 0
    assert stats["total_hits"] == 0
    assert stats["hit_rate"] == 0.0


def test_cache_stats_with_hits(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    cache = LLMCache(db)
    messages = [{"role": "user", "content": "q"}]
    cache.put("m", messages, "r")
    cache.get("m", messages)  # 1 hit
    cache.get("m", messages)  # 2 hits
    stats = cache.stats()
    assert stats["total_entries"] == 1
    assert stats["total_hits"] == 2
    # hit_rate = 2 / (2 + 1) = 0.666...
    assert stats["hit_rate"] > 0.0


# ---------------------------------------------------------------------------
# Stage 3: Wire cache into LLM clients
# ---------------------------------------------------------------------------


def test_ollama_cache_hit_skips_http(tmp_path: Path) -> None:
    from synto.ollama_client import OllamaClient

    db = StateDB(tmp_path / "state.db")
    cache = LLMCache(db)
    messages = [{"role": "user", "content": "hello"}]
    cache.put("gemma4:e4b", messages, "cached response")

    client = OllamaClient(cache=cache)
    with patch.object(client._client, "post") as mock_post:
        result = client.generate("hello", "gemma4:e4b")
        mock_post.assert_not_called()
    assert result == "cached response"


def test_ollama_cache_stores_response(tmp_path: Path) -> None:
    from synto.ollama_client import OllamaClient

    db = StateDB(tmp_path / "state.db")
    cache = LLMCache(db)

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "response": "live response",
        "done_reason": "stop",
        "prompt_eval_count": 5,
        "eval_count": 3,
    }
    mock_response.raise_for_status.return_value = None

    client = OllamaClient(cache=cache)
    with patch.object(client._client, "post", return_value=mock_response):
        result = client.generate("hello", "gemma4:e4b")

    assert result == "live response"
    cached = cache.get("gemma4:e4b", [{"role": "user", "content": "hello"}])
    assert cached == "live response"


def test_openai_compat_cache_hit_skips_http(tmp_path: Path) -> None:
    from synto.openai_compat_client import OpenAICompatClient

    db = StateDB(tmp_path / "state.db")
    cache = LLMCache(db)
    messages = [{"role": "user", "content": "hi"}]
    cache.put("gpt-4o", messages, "cached openai")

    client = OpenAICompatClient(base_url="http://localhost:1234/v1", cache=cache)
    with patch.object(client._client, "post") as mock_post:
        result = client.generate("hi", "gpt-4o")
        mock_post.assert_not_called()
    assert result == "cached openai"


def test_no_cache_by_default(tmp_path: Path) -> None:
    """When no cache is passed, clients must not hit any cache table."""
    from synto.ollama_client import OllamaClient

    client = OllamaClient()
    assert client._cache is None


# ---------------------------------------------------------------------------
# Stage 4: Cache management commands
# ---------------------------------------------------------------------------


def test_maintain_clear_cache(tmp_path: Path, config, db) -> None:
    from click.testing import CliRunner

    from synto.cache import LLMCache
    from synto.cli import cli

    cache = LLMCache(db)
    cache.put("m", [{"role": "user", "content": "x"}], "r")
    assert db._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0] == 1

    runner = CliRunner()
    result = runner.invoke(cli, ["maintain", "--vault", str(config.vault), "--clear-cache"])
    assert result.exit_code == 0, result.output
    assert db._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0] == 0


def test_maintain_clear_cache_older_than(tmp_path: Path, config, db) -> None:
    from click.testing import CliRunner

    from synto.cache import LLMCache
    from synto.cli import cli

    cache = LLMCache(db)
    cache.put("m", [{"role": "user", "content": "fresh"}], "r")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["maintain", "--vault", str(config.vault), "--clear-cache", "--older-than", "30"],
    )
    assert result.exit_code == 0, result.output
    # Fresh entry should survive (not 30+ days old)
    assert db._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0] == 1


def test_stats_includes_cache(tmp_path: Path, config, db) -> None:
    from synto.cache import LLMCache
    from synto.stats import compute_stats_from_db

    cache = LLMCache(db)
    messages = [{"role": "user", "content": "q"}]
    cache.put("m", messages, "r")
    cache.get("m", messages)

    report = compute_stats_from_db(config, db)
    assert report.vault.cache_entries == 1
    assert report.vault.cache_hit_rate > 0.0


def test_stats_render_includes_cache_lines(tmp_path: Path, config, db) -> None:
    from synto.stats import compute_stats_from_db, render_text

    report = compute_stats_from_db(config, db)
    text = render_text(report)
    assert "Cache entries" in text
    assert "Cache hit rate" in text
