"""Tests for the Reader protocol and concrete implementations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synto.readers import (
    ArticleFilter,
    ArticleRef,
    PackIndex,
    PackManifest,
    PackReader,
    Reader,
    VaultReader,
)


def _make_minimal_pack(root: Path) -> None:
    (root / "agent").mkdir(parents=True)
    (root / "index").mkdir(parents=True)
    (root / "pack.toml").write_text('[pack]\nid = "sample"\n', encoding="utf-8")
    (root / "agent" / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pack": {"id": "sample", "version": "1.0.0", "capabilities": ["articles"]},
            }
        ),
        encoding="utf-8",
    )
    (root / "index" / "INDEX.json").write_text(
        json.dumps({"schema_version": 1, "articles": [], "terms": [], "papers": [], "sources": []}),
        encoding="utf-8",
    )


def test_pack_reader_init(tmp_path: Path) -> None:
    _make_minimal_pack(tmp_path)
    reader = PackReader(tmp_path)
    assert reader.pack_root == tmp_path


def test_vault_reader_init(tmp_path: Path) -> None:
    reader = VaultReader(tmp_path)
    assert reader.vault_root == tmp_path


def test_pack_reader_requires_pack_toml(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        PackReader(tmp_path)


def test_vault_reader_methods_are_safe_when_db_missing(tmp_path: Path) -> None:
    reader = VaultReader(tmp_path)
    assert reader.manifest.pack_id == tmp_path.name
    assert reader.index.articles == ()
    assert reader.capabilities == frozenset({"articles", "concepts"})
    assert reader.list_articles() == []
    assert reader.find_concept("x") is None
    assert reader.list_terms() == []
    assert reader.find_term("x") is None
    assert reader.list_sources() == []
    assert reader.list_segments() == []
    assert reader.has_capability("articles") is True


def test_article_ref_is_immutable() -> None:
    ref = ArticleRef(id="01HXX", name="Vector Clocks", path="articles/Vector-Clocks.md")
    with pytest.raises((AttributeError, TypeError)):
        ref.name = "Different"  # type: ignore[misc]


def test_article_filter_optional_fields() -> None:
    article_filter = ArticleFilter()
    assert article_filter.tag is None
    assert article_filter.min_confidence is None
    article_filter = ArticleFilter(tag="distributed-systems", min_confidence="high")
    assert article_filter.tag == "distributed-systems"


def test_pack_manifest_construction() -> None:
    manifest = PackManifest(
        schema_version=1,
        pack_id="ostep",
        version="1.0.0",
        capabilities=frozenset({"articles", "concepts"}),
    )
    assert "articles" in manifest.capabilities
    assert manifest.redistribution == "unknown"


def test_pack_index_construction() -> None:
    index = PackIndex(schema_version=1, articles=())
    assert index.articles == ()


def test_protocol_recognized_structurally(tmp_path: Path) -> None:
    """Reader is a Protocol; PackReader and VaultReader satisfy it structurally."""

    def takes_reader(_reader: Reader) -> None:
        return None

    _make_minimal_pack(tmp_path)
    takes_reader(PackReader(tmp_path))
    takes_reader(VaultReader(Path(".")))
    assert hasattr(PackReader, "list_articles")
    assert hasattr(VaultReader, "list_articles")
