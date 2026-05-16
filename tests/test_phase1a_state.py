from __future__ import annotations

from synto.state import StateDB


def test_find_concept_by_name_or_alias_treats_percent_literally(tmp_path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["A%B", "AXXB"])

    result = db.find_concept_by_name_or_alias("A%B")

    assert result is not None
    assert result[0] == "A%B"


def test_find_concept_by_name_or_alias_treats_underscore_literally(tmp_path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["foo_bar", "fooXbar"])

    result = db.find_concept_by_name_or_alias("foo_bar")

    assert result is not None
    assert result[0] == "foo_bar"


def test_find_concept_by_name_or_alias_handles_backslash(tmp_path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", [r"back\slash"])

    result = db.find_concept_by_name_or_alias(r"back\slash")

    assert result is not None
    assert result[0] == r"back\slash"
