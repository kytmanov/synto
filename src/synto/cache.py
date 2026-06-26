"""LLM response cache backed by the SQLite state DB."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta

from .state import StateDB


class LLMCache:
    def __init__(self, db: StateDB) -> None:
        self._db = db

    def _key(self, model: str, messages: list[dict], namespace: str = "") -> str:
        # namespace (the client's base_url) keeps the same model name on different
        # endpoints/accounts from colliding — otherwise two providers serving an
        # identically-named model would return each other's cached responses.
        data = namespace + "\x00" + model + json.dumps(messages, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def get(self, model: str, messages: list[dict], namespace: str = "") -> str | None:
        # Route through StateDB._tx() rather than committing on the shared connection directly:
        # under parallel ingest these run on worker threads, and a bare commit() would land
        # inside the main thread's open transaction and corrupt it (#75). _tx() serializes via
        # the StateDB lock and keeps the SELECT + hit-count UPDATE in one transaction.
        key = self._key(model, messages, namespace)
        with self._db._tx() as conn:
            row = conn.execute(
                "SELECT response_json FROM llm_cache WHERE cache_key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE llm_cache SET hit_count = hit_count + 1, last_hit_at = ? "
                "WHERE cache_key = ?",
                (datetime.now().isoformat(), key),
            )
            return row["response_json"]

    def put(self, model: str, messages: list[dict], response: str, namespace: str = "") -> None:
        key = self._key(model, messages, namespace)
        with self._db._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (cache_key, model, response_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (key, model, response, datetime.now().isoformat()),
            )

    def clear(self, older_than_days: int | None = None) -> int:
        with self._db._tx() as conn:
            if older_than_days is None:
                cursor = conn.execute("DELETE FROM llm_cache")
            else:
                cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat()
                cursor = conn.execute("DELETE FROM llm_cache WHERE created_at < ?", (cutoff,))
            return cursor.rowcount

    def stats(self) -> dict:
        # Read under the StateDB lock for consistency with get/put/clear: a single shared
        # connection is mutated by other threads under parallel ingest, so even this read goes
        # through the serialized read path rather than touching _conn directly (#75).
        with self._db._read() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as total_entries, COALESCE(SUM(hit_count), 0) as total_hits "
                "FROM llm_cache"
            ).fetchone()
        total_entries = row["total_entries"]
        total_hits = row["total_hits"]
        # Each entry represents one cache miss (initial put); hits are subsequent reuses.
        total_requests = total_entries + total_hits
        hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
        return {
            "total_entries": total_entries,
            "total_hits": total_hits,
            "hit_rate": hit_rate,
        }
