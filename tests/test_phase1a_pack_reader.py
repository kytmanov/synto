from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from synto.readers import ArticleNotFound, MalformedPackError, PackReader


def _write_pack(root: Path) -> None:
    (root / "agent").mkdir(parents=True)
    (root / "index").mkdir(parents=True)
    (root / "articles").mkdir(parents=True)
    (root / "articles" / "nested").mkdir(parents=True)
    (root / "pack.toml").write_text('[pack]\nid = "sample"\n', encoding="utf-8")
    (root / "agent" / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pack": {
                    "id": "sample",
                    "version": "1.0.0",
                    "capabilities": ["articles", "concepts"],
                },
                "redistribution": "unknown",
            }
        ),
        encoding="utf-8",
    )
    (root / "agent" / "concepts.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "concepts": [
                    {
                        "name": "Vector Clocks",
                        "aliases": ["VC"],
                        "canonical_article_id": "01VCLOCKS0000000000000000",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "index" / "INDEX.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "articles": [
                    {
                        "id": "01VCLOCKS0000000000000000",
                        "name": "Vector Clocks",
                        "path": "articles/Vector-Clocks.md",
                        "summary": "Clocks",
                        "tags": ["distributed"],
                        "confidence": "high",
                    },
                    {
                        "id": "01IDEMPOTENCY0000000000000",
                        "name": "Idempotency",
                        "path": "articles/nested/Idempotency.md",
                        "summary": "Idempotent ops",
                        "tags": [],
                        "confidence": "high",
                    },
                ],
                "sources": [{"id": "src-1", "title": "Source", "source_type": "unknown_text"}],
                "papers": [],
            }
        ),
        encoding="utf-8",
    )
    (root / "articles" / "Vector-Clocks.md").write_text(
        "---\ntitle: Vector Clocks\n---\nBody", encoding="utf-8"
    )
    (root / "articles" / "nested" / "Idempotency.md").write_text(
        "---\ntitle: Idempotency\n---\nBody", encoding="utf-8"
    )


def test_pack_reader_reads_manifest_index_and_article(tmp_path: Path) -> None:
    _write_pack(tmp_path)

    reader = PackReader(tmp_path)

    assert reader.manifest.pack_id == "sample"
    assert len(reader.list_articles()) == 2
    assert reader.read_article("Vector Clocks").body == "Body"
    assert reader.read_article("01VCLOCKS0000000000000000").name == "Vector Clocks"
    assert reader.read_article("Idempotency").path == "articles/nested/Idempotency.md"


def test_pack_reader_find_concept_uses_aliases(tmp_path: Path) -> None:
    _write_pack(tmp_path)
    reader = PackReader(tmp_path)

    concept = reader.find_concept("VC")
    assert concept is not None
    assert concept.name == "Vector Clocks"


def test_pack_reader_concept_lookup_is_cached(tmp_path: Path) -> None:
    _write_pack(tmp_path)
    reader = PackReader(tmp_path)

    with patch.object(reader, "_load_concepts_json", wraps=reader._load_concepts_json) as mocked:
        for _ in range(25):
            assert reader.find_concept("Vector Clocks") is not None
            assert reader.find_concept("VC") is not None
        assert reader.find_concept("missing") is None

    assert mocked.call_count == 1


def test_pack_reader_missing_article_raises(tmp_path: Path) -> None:
    _write_pack(tmp_path)
    reader = PackReader(tmp_path)
    with pytest.raises(ArticleNotFound):
        reader.read_article("missing")


def test_pack_reader_rejects_missing_manifest(tmp_path: Path) -> None:
    (tmp_path / "pack.toml").write_text("[pack]\nid='x'\n", encoding="utf-8")
    reader = PackReader(tmp_path)
    with pytest.raises(MalformedPackError):
        _ = reader.manifest


def test_pack_reader_rejects_duplicate_ids(tmp_path: Path) -> None:
    _write_pack(tmp_path)
    index_path = tmp_path / "index" / "INDEX.json"
    data = json.loads(index_path.read_text(encoding="utf-8"))
    data["articles"][1]["id"] = data["articles"][0]["id"]
    index_path.write_text(json.dumps(data), encoding="utf-8")

    reader = PackReader(tmp_path)
    with pytest.raises(MalformedPackError):
        _ = reader.index


def test_pack_reader_rejects_path_traversal(tmp_path: Path) -> None:
    _write_pack(tmp_path)
    index_path = tmp_path / "index" / "INDEX.json"
    data = json.loads(index_path.read_text(encoding="utf-8"))
    data["articles"][0]["path"] = "../outside.md"
    index_path.write_text(json.dumps(data), encoding="utf-8")

    reader = PackReader(tmp_path)
    with pytest.raises(MalformedPackError):
        _ = reader.index


def test_pack_reader_rejects_absolute_paths(tmp_path: Path) -> None:
    _write_pack(tmp_path)
    index_path = tmp_path / "index" / "INDEX.json"
    data = json.loads(index_path.read_text(encoding="utf-8"))
    data["articles"][0]["path"] = "/tmp/evil.md"
    index_path.write_text(json.dumps(data), encoding="utf-8")

    reader = PackReader(tmp_path)
    with pytest.raises(MalformedPackError):
        _ = reader.index


def test_pack_reader_rejects_missing_article_file(tmp_path: Path) -> None:
    _write_pack(tmp_path)
    (tmp_path / "articles" / "Vector-Clocks.md").unlink()

    reader = PackReader(tmp_path)
    with pytest.raises(MalformedPackError):
        reader.read_article("Vector Clocks")


def test_pack_reader_rejects_duplicate_paths(tmp_path: Path) -> None:
    _write_pack(tmp_path)
    index_path = tmp_path / "index" / "INDEX.json"
    data = json.loads(index_path.read_text(encoding="utf-8"))
    data["articles"][1]["path"] = data["articles"][0]["path"]
    index_path.write_text(json.dumps(data), encoding="utf-8")

    reader = PackReader(tmp_path)
    with pytest.raises(MalformedPackError):
        _ = reader.index


def test_pack_reader_rejects_invalid_index_json(tmp_path: Path) -> None:
    _write_pack(tmp_path)
    (tmp_path / "index" / "INDEX.json").write_text("{broken", encoding="utf-8")

    reader = PackReader(tmp_path)
    with pytest.raises(MalformedPackError):
        _ = reader.index


def test_pack_reader_rejects_symlink_escape(tmp_path: Path) -> None:
    _write_pack(tmp_path)
    target = tmp_path.parent / "outside.md"
    target.write_text("---\ntitle: Outside\n---\nNope", encoding="utf-8")
    article_path = tmp_path / "articles" / "Vector-Clocks.md"
    article_path.unlink()
    article_path.symlink_to(target)

    reader = PackReader(tmp_path)
    with pytest.raises(MalformedPackError):
        reader.read_article("Vector Clocks")
