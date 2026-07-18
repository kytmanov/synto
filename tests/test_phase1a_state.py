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


# SQLite's built-in lower() only folds ASCII, so any-script case-insensitivity must come
# from Python-side casefold — these pin it for the two name-matching entry points.


def test_find_concept_substring_fallback_folds_nonascii_case(tmp_path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Привет Мир"])

    # Substring (not exact), differently cased — must still hit the fallback match.
    result = db.find_concept_by_name_or_alias("привет ми")

    assert result is not None
    assert result[0] == "Привет Мир"


def test_concept_name_exists_exact_folds_nonascii_case(tmp_path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Привет"])

    assert db.concept_name_exists_exact("привет") is True
    assert db.concept_name_exists_exact("ПРИВЕТ") is True
    assert db.concept_name_exists_exact("здравствуй") is False


def test_concept_name_exists_exact_folds_nonascii_knowledge_item(tmp_path) -> None:
    # knowledge_items have no entity-label backstop, so the name compare itself must fold.
    from synto.models import KnowledgeItemRecord

    db = StateDB(tmp_path / "state.db")
    db.upsert_item(KnowledgeItemRecord(name="Правило Байеса"))

    assert db.concept_name_exists_exact("правило байеса") is True


def test_concept_name_exists_exact_nontrivial_fold(tmp_path) -> None:
    # casefold-only equivalence (lower() would not fold the dotted capital İ).
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["İstanbul"])

    assert db.concept_name_exists_exact("i̇stanbul") is True
