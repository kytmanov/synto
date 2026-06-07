from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from synto.config import Config
from synto.indexer import generate_index
from synto.models import RawNoteRecord
from synto.pack_export import _export_source_refs
from synto.pipeline.compile import _build_source_refs
from synto.readers import ArticleRef
from synto.state import StateDB
from synto.vault import write_note


def _check(condition: bool, name: str) -> dict[str, object]:
    return {"name": name, "passed": bool(condition)}


def main() -> int:
    checks: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="synto-portability-smoke-") as tmp:
        vault = Path(tmp)
        (vault / "raw").mkdir()
        (vault / "wiki").mkdir()
        (vault / "wiki" / ".drafts").mkdir()
        (vault / ".synto").mkdir()

        config = Config(vault=vault)
        db = StateDB(config.state_db_path)

        try:
            write_note(vault / "raw" / "note.md", {"title": "Raw Note"}, "Body\n")
            write_note(
                config.sources_dir / "Canonical Source.md",
                {"title": "Canonical Source", "source_file": r"raw\note.md"},
                "Summary\n",
            )
            write_note(
                config.wiki_dir / "Article.md",
                {"title": "Article", "sources": ["raw/note.md"]},
                "Body [[sources/note|legacy]]\n",
            )
            db.upsert_raw(RawNoteRecord(path="raw/note.md", content_hash="hash", status="ingested"))
            db.upsert_concepts("raw/note.md", ["Portable Concept"])

            source_refs = _build_source_refs(["raw/note.md"], vault)
            checks.append(
                _check(
                    len(source_refs) == 1
                    and source_refs[0].title == "Canonical Source"
                    and source_refs[0].wiki_target == "sources/Canonical Source",
                    "compile matches legacy source summary metadata",
                )
            )

            refs = _export_source_refs(
                config,
                db,
                [ArticleRef(id="1", name="Article", path="wiki/Article.md")],
            )
            checks.append(
                _check(
                    len(refs) == 1 and refs[0]["raw_path"] == "raw/note.md",
                    "export normalizes legacy source_file",
                )
            )
            checks.append(
                _check(
                    len(refs) == 1 and refs[0]["referenced_by_articles"] == ["articles/Article.md"],
                    "export preserves article references",
                )
            )

            index_text = generate_index(config, db).read_text(encoding="utf-8")
            checks.append(
                _check(
                    "\\" not in index_text and "raw/note.md" in index_text,
                    "index renders normalized source hint",
                )
            )
        finally:
            db.close()

    passed = sum(1 for c in checks if c["passed"])
    failed = len(checks) - passed
    report = {"passed": passed, "failed": failed, "checks": checks}
    report_file = os.environ.get("REPORT_FILE")
    if report_file:
        Path(report_file).write_text(json.dumps(report) + "\n", encoding="utf-8")
    if failed:
        for check in checks:
            if not check["passed"]:
                print(f"FAIL: {check['name']}")
        return 1
    print("Portability smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
