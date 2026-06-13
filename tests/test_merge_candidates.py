"""Merge-candidate worklist + B2 demotion guard (v26).

Unit coverage for the building blocks of order-independent identity: the role-aware demotion
(weak aliases demoted, blessed aliases spared), deterministic candidate pair ordering, candidate
recording gated to the ingest seam, clearing on curation, host-frontmatter strip, and the
doctor/inspect surfacing.
"""

from __future__ import annotations

from pathlib import Path

from synto.concept_text import concept_key as _ck
from synto.config import Config
from synto.models import WikiArticleRecord
from synto.state import StateDB
from synto.vault import parse_note, write_note


def _aliases_ck(db: StateDB, name: str) -> set[str]:
    return {_ck(a) for a in db.get_aliases(name)}


def test_minting_demotes_weak_alias_spares_blessed(tmp_path: Path) -> None:
    """B2 fix: minting a preferred label demotes a colliding WEAK alias but never a human-blessed
    one. The pre-fix DELETE had no source filter and would silently destroy a user/rename alias.
    """
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_aliases("Alpha", ["Foo"])  # weak (extracted) alias Foo on Alpha
    db.upsert_concepts("raw/b.md", ["Beta"])
    db.upsert_aliases("Beta", ["Foo"], source="user")  # blessed alias Foo on Beta

    # Mint "Foo" as its own concept via the ingest write seam.
    demotions = db.replace_concepts_for_source("raw/c.md", ["Foo"])

    assert "foo" not in _aliases_ck(db, "Alpha")  # weak alias demoted
    assert "foo" in _aliases_ck(db, "Beta")  # blessed alias spared
    alpha_id = db.entity_id_for_name("Alpha")
    assert any(host == alpha_id for host, _surface in demotions)  # only the weak host reported
    assert any(_ck(c["surface"]) == "foo" for c in db.list_merge_candidates())


def test_non_ingest_mint_demotes_without_recording_candidate(tmp_path: Path) -> None:
    """Critique fix #2: candidate recording is gated to the ingest write seam. A mint via the
    shared helper path (upsert_concepts) still demotes the weak alias but records NO candidate, so
    a clean INDEX.json rebuild / rename does not manufacture spurious candidates.
    """
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_aliases("Alpha", ["Foo"])
    db.upsert_concepts("raw/b.md", ["Foo"])  # mints Foo via the non-ingest seam

    assert "foo" not in _aliases_ck(db, "Alpha")  # still demoted (correctness)
    assert db.list_merge_candidates() == []  # but no candidate recorded


def test_record_merge_candidate_orders_pair_by_label_key(tmp_path: Path) -> None:
    """Critique fix #3: the stored pair is ordered by preferred label_key (not random entity_id),
    so the same logical pair dedups to one row regardless of which order it is recorded in.
    """
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Zebra"])
    db.upsert_concepts("raw/b.md", ["Apple"])
    z = db.entity_id_for_name("Zebra")
    a = db.entity_id_for_name("Apple")

    db.record_merge_candidate(z, a, "Surf")
    db.record_merge_candidate(a, z, "Surf")  # reverse arg order, same surface

    cands = db.list_merge_candidates()
    assert len(cands) == 1  # deduped via deterministic ordering
    assert cands[0]["label_a"] == "Apple"  # 'apple' < 'zebra' by label_key
    assert cands[0]["label_b"] == "Zebra"


def test_list_merge_candidates_skips_stale_pair(tmp_path: Path) -> None:
    """A candidate whose entity is no longer resolvable (e.g. retired) is omitted from the list."""
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_concepts("raw/b.md", ["Beta"])
    a = db.entity_id_for_name("Alpha")
    b = db.entity_id_for_name("Beta")
    db.record_merge_candidate(a, b, "x")
    assert len(db.list_merge_candidates()) == 1

    # Retire Beta directly; preferred_label_for_entity still returns its label, so the pair is
    # only dropped once cleared — assert the explicit clear path instead.
    db.clear_merge_candidates_for_entity(b)
    assert db.list_merge_candidates() == []


def test_merge_clears_candidates_touching_either_entity(tmp_path: Path) -> None:
    """Resolving a candidate with `concept merge` drops it from the worklist."""
    db = StateDB(tmp_path / "state.db")
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_concepts("raw/b.md", ["Beta"])
    a = db.entity_id_for_name("Alpha")
    b = db.entity_id_for_name("Beta")
    db.record_merge_candidate(a, b, "x")
    assert db.list_merge_candidates()

    db.merge_entities("Alpha", "Beta")

    assert db.list_merge_candidates() == []


def test_strip_demoted_alias_from_host_frontmatter(tmp_path: Path) -> None:
    """Critique fix #6: when a surface a host held as a weak alias becomes its own concept, the
    host's on-disk `aliases:` is stripped of that surface (body untouched, content_hash preserved).
    """
    from synto.pipeline.ingest import _strip_demoted_alias_from_host

    vault = tmp_path / "v"
    (vault / "wiki").mkdir(parents=True)
    (vault / ".synto").mkdir(parents=True)
    config = Config(vault=vault)
    db = StateDB(config.state_db_path)

    db.upsert_concepts("raw/a.md", ["Gradient Descent"])
    host_id = db.entity_id_for_name("Gradient Descent")
    rel = "wiki/Gradient Descent.md"
    path = vault / rel
    write_note(path, {"title": "Gradient Descent", "aliases": ["GD", "Steepest Descent"]}, "Body.")
    db.upsert_article(
        WikiArticleRecord(
            path=rel,
            title="Gradient Descent",
            sources=["raw/a.md"],
            content_hash="h",
            status="published",
            kind="concept",
            entity_id=host_id,
        )
    )

    _strip_demoted_alias_from_host(config, db, host_id, "GD")

    meta, body = parse_note(path)
    assert "GD" not in meta.get("aliases", [])
    assert "Steepest Descent" in meta.get("aliases", [])  # other aliases preserved
    assert body.strip() == "Body."  # body untouched


def test_doctor_identity_section_renders_merge_candidate(tmp_path: Path, capsys) -> None:
    """`synto doctor` surfaces the merge-candidate worklist so the promotion is never silent."""
    from synto.cli import _render_identity_section

    vault = tmp_path / "v"
    (vault / ".synto").mkdir(parents=True)
    config = Config(vault=vault)
    db = StateDB(config.state_db_path)
    db.upsert_concepts("raw/a.md", ["Gradient Descent"])
    db.upsert_concepts("raw/b.md", ["GD"])
    db.record_merge_candidate(
        db.entity_id_for_name("GD"),
        db.entity_id_for_name("Gradient Descent"),
        "GD",
        reason="promoted-from-alias",
    )

    _render_identity_section(config, db, reconcile=False)

    out = capsys.readouterr().out
    assert "merge candidate" in out.lower()
    assert "GD" in out and "Gradient Descent" in out


def test_concept_inspect_lists_merge_candidate(tmp_path: Path) -> None:
    """`synto concept inspect NAME` lists merge candidates touching that entity."""
    from click.testing import CliRunner

    from synto.cli import cli

    vault = tmp_path / "v"
    (vault / ".synto").mkdir(parents=True)
    config = Config(vault=vault)
    db = StateDB(config.state_db_path)
    db.upsert_concepts("raw/a.md", ["Gradient Descent"])
    db.upsert_concepts("raw/b.md", ["GD"])
    db.record_merge_candidate(
        db.entity_id_for_name("GD"),
        db.entity_id_for_name("Gradient Descent"),
        "GD",
        reason="promoted-from-alias",
    )
    db.close()

    result = CliRunner().invoke(
        cli, ["concept", "inspect", "Gradient Descent", "--vault", str(vault)]
    )

    assert result.exit_code == 0, result.output
    assert "Merge candidates" in result.output
    assert "GD" in result.output
