"""Tests for Feature 05: synto add command."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest
from click.testing import CliRunner

from synto.cli import cli
from synto.config import Config
from synto.state import StateDB

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_png(width: int = 4, height: int = 4) -> bytes:
    def _chunk(ctype: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(ctype + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    row = b"\x00" + b"\xff\x00\x00" * width
    idat = _chunk(b"IDAT", zlib.compress(row * height))
    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    doc = fitz.open()
    for title in ("Introduction", "Methods"):
        page = doc.new_page()
        page.insert_text((72, 72), f"# {title}\nContent for {title}.")
    out = tmp_path / "sample.pdf"
    doc.save(str(out))
    doc.close()
    return out


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    p = tmp_path / "notes.txt"
    p.write_text("Some raw notes content.")
    return p


@pytest.fixture
def sample_md(tmp_path: Path) -> Path:
    p = tmp_path / "clip.md"
    p.write_text(
        "---\n"
        "title: Imported Clip\n"
        "source: https://example.com/post\n"
        "url: https://example.com/post\n"
        "tags:\n"
        "  - clip\n"
        "---\n\n"
        "Imported markdown body.\n"
    )
    return p


@pytest.fixture
def renamed_sample_txt(tmp_path: Path, sample_txt: Path) -> Path:
    p = tmp_path / "renamed-notes.txt"
    p.write_text(sample_txt.read_text(encoding="utf-8"), encoding="utf-8")
    return p


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# Stage 1: Basic import — file copy + DB row
# ---------------------------------------------------------------------------


def test_add_txt_creates_db_row(
    config: Config, db: StateDB, sample_txt: Path, runner: CliRunner
) -> None:
    result = runner.invoke(cli, ["add", str(sample_txt), "--vault", str(config.vault)])
    assert result.exit_code == 0, result.output
    rows = db.list_source_documents()
    assert len(rows) == 1
    assert rows[0][2] == "notes"  # source_type


def test_add_copies_original_to_app_dir(
    config: Config, db: StateDB, sample_txt: Path, runner: CliRunner
) -> None:
    result = runner.invoke(cli, ["add", str(sample_txt), "--vault", str(config.vault)])
    assert result.exit_code == 0, result.output
    source_dir = config.app_dir / "sources"
    copies = list(source_dir.glob("*/original.txt"))
    assert len(copies) == 1
    assert copies[0].read_text(encoding="utf-8") == sample_txt.read_text(encoding="utf-8")


def test_add_source_id_is_stable(
    config: Config, db: StateDB, sample_txt: Path, runner: CliRunner
) -> None:
    """Two imports of the same file produce the same source_id."""
    runner.invoke(cli, ["add", str(sample_txt), "--vault", str(config.vault)])
    rows1 = db.list_source_documents()
    # Re-import with --force
    runner.invoke(cli, ["add", "--force", str(sample_txt), "--vault", str(config.vault)])
    rows2 = db.list_source_documents()
    assert rows1[0][0] == rows2[0][0]


def test_add_source_id_format(
    config: Config, db: StateDB, sample_txt: Path, runner: CliRunner
) -> None:
    runner.invoke(cli, ["add", str(sample_txt), "--vault", str(config.vault)])
    source_id = db.list_source_documents()[0][0]
    # slug-hash8 format
    assert "-" in source_id
    parts = source_id.rsplit("-", 1)
    assert len(parts[1]) == 8
    int(parts[1], 16)  # must be valid hex


# ---------------------------------------------------------------------------
# Stage 2: PDF extraction
# ---------------------------------------------------------------------------


def test_add_pdf_extracts_segments(
    config: Config, db: StateDB, sample_pdf: Path, runner: CliRunner
) -> None:
    result = runner.invoke(cli, ["add", str(sample_pdf), "--vault", str(config.vault)])
    assert result.exit_code == 0, result.output
    segs = db.list_source_segments_brief()
    assert len(segs) >= 1


def test_add_pdf_infers_type_paper(
    config: Config, db: StateDB, sample_pdf: Path, runner: CliRunner
) -> None:
    runner.invoke(cli, ["add", str(sample_pdf), "--vault", str(config.vault)])
    rows = db.list_source_documents()
    assert rows[0][2] == "paper"


def test_add_pdf_explicit_type(
    config: Config, db: StateDB, sample_pdf: Path, runner: CliRunner
) -> None:
    runner.invoke(cli, ["add", "--type", "textbook", str(sample_pdf), "--vault", str(config.vault)])
    rows = db.list_source_documents()
    assert rows[0][2] == "textbook"


def test_add_pdf_segments_linked_to_source(
    config: Config, db: StateDB, sample_pdf: Path, runner: CliRunner
) -> None:
    runner.invoke(cli, ["add", str(sample_pdf), "--vault", str(config.vault)])
    source_id = db.list_source_documents()[0][0]
    segs = db.list_source_segments_brief()
    assert all(s[2] == source_id for s in segs)


# ---------------------------------------------------------------------------
# Stage 3: --extend-pack
# ---------------------------------------------------------------------------


def test_extend_pack_does_not_mutate_config(
    config: Config, db: StateDB, sample_txt: Path, runner: CliRunner
) -> None:
    result = runner.invoke(
        cli,
        ["add", "--extend-pack", "my-pack", str(sample_txt), "--vault", str(config.vault)],
    )
    assert result.exit_code == 0, result.output
    toml_path = config.vault / "synto.toml"
    assert not toml_path.exists()
    assert "not implemented" in result.output


def test_extend_pack_preserves_existing_toml(
    config: Config, db: StateDB, sample_txt: Path, runner: CliRunner
) -> None:
    toml_path = config.vault / "synto.toml"
    toml_path.write_text('[models]\nfast = "gemma4:e4b"\n')
    result = runner.invoke(
        cli,
        ["add", "--extend-pack", "my-pack", str(sample_txt), "--vault", str(config.vault)],
    )
    assert result.exit_code == 0, result.output
    text = toml_path.read_text()
    assert "[models]" in text  # original content preserved
    assert "[[pack.sources]]" not in text


# ---------------------------------------------------------------------------
# Stage 4: Duplicate detection
# ---------------------------------------------------------------------------


def test_duplicate_import_blocked(
    config: Config, db: StateDB, sample_txt: Path, runner: CliRunner
) -> None:
    runner.invoke(cli, ["add", str(sample_txt), "--vault", str(config.vault)])
    result = runner.invoke(cli, ["add", str(sample_txt), "--vault", str(config.vault)])
    assert result.exit_code != 0
    assert len(db.list_source_documents()) == 1


def test_duplicate_import_allowed_with_force(
    config: Config, db: StateDB, sample_txt: Path, runner: CliRunner
) -> None:
    runner.invoke(cli, ["add", str(sample_txt), "--vault", str(config.vault)])
    result = runner.invoke(cli, ["add", "--force", str(sample_txt), "--vault", str(config.vault)])
    assert result.exit_code == 0


def test_duplicate_import_blocked_by_content_hash_for_renamed_file(
    config: Config, db: StateDB, sample_txt: Path, renamed_sample_txt: Path, runner: CliRunner
) -> None:
    first = runner.invoke(cli, ["add", str(sample_txt), "--vault", str(config.vault)])
    second = runner.invoke(cli, ["add", str(renamed_sample_txt), "--vault", str(config.vault)])

    assert first.exit_code == 0, first.output
    assert second.exit_code != 0
    assert len(db.list_source_documents()) == 1


def test_force_reimport_replaces_prior_raw_and_assets(
    config: Config, db: StateDB, sample_pdf: Path, runner: CliRunner
) -> None:
    from types import SimpleNamespace
    from unittest.mock import patch

    first_segs = [
        SimpleNamespace(
            text="First import text.",
            structural_locator="section:first",
            image_refs=["assets/sample-src/img-old.png"],
        )
    ]
    second_segs = [
        SimpleNamespace(
            text="Second import text.",
            structural_locator="section:second",
            image_refs=["assets/sample-src/img-new.png"],
        )
    ]

    def fake_extract(*_args, **kwargs):
        vault_root = kwargs["vault_root"]
        source_id = _args[0]
        assets_dir = vault_root / "assets" / source_id
        assets_dir.mkdir(parents=True, exist_ok=True)
        filename = "img-old.png" if fake_extract.calls == 0 else "img-new.png"
        (assets_dir / filename).write_bytes(b"img")
        result = first_segs if fake_extract.calls == 0 else second_segs
        fake_extract.calls += 1
        return result

    fake_extract.calls = 0

    with patch("synto.extractors.pdf.extract_pdf", side_effect=fake_extract):
        first = runner.invoke(cli, ["add", str(sample_pdf), "--vault", str(config.vault)])
        second = runner.invoke(
            cli, ["add", "--force", str(sample_pdf), "--vault", str(config.vault)]
        )

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    raw_files = list((config.vault / "raw").glob("*.md"))
    assert len(raw_files) == 1
    content = raw_files[0].read_text()
    assert "Second import text." in content
    assert "First import text." not in content

    assets = config.vault / "assets"
    assert list(assets.glob("**/img-old.png")) == []
    assert list(assets.glob("**/img-new.png"))


def test_get_source_document_method(config: Config, db: StateDB) -> None:
    """get_source_document returns None for unknown IDs."""
    assert db.get_source_document("nonexistent") is None


# ---------------------------------------------------------------------------
# Pipeline wiring: write_source_content_md / raw note creation
# ---------------------------------------------------------------------------


def test_add_txt_writes_raw_note(
    config: Config, db: StateDB, sample_txt: Path, runner: CliRunner
) -> None:
    result = runner.invoke(cli, ["add", str(sample_txt), "--vault", str(config.vault)])
    assert result.exit_code == 0
    raw_files = list((config.vault / "raw").glob("*.md"))
    assert len(raw_files) == 1
    content = raw_files[0].read_text()
    assert "Some raw notes content." in content


def test_add_txt_raw_note_has_source_type_frontmatter(
    config: Config, db: StateDB, sample_txt: Path, runner: CliRunner
) -> None:
    runner.invoke(cli, ["add", str(sample_txt), "--vault", str(config.vault)])
    raw_files = list((config.vault / "raw").glob("*.md"))
    assert len(raw_files) == 1
    content = raw_files[0].read_text()
    assert "source_type:" in content


def test_add_md_preserves_existing_frontmatter(
    config: Config, db: StateDB, sample_md: Path, runner: CliRunner
) -> None:
    result = runner.invoke(cli, ["add", str(sample_md), "--vault", str(config.vault)])
    assert result.exit_code == 0, result.output
    raw_files = list((config.vault / "raw").glob("*.md"))
    assert len(raw_files) == 1
    content = raw_files[0].read_text()
    assert "title: Imported Clip" in content
    assert "source: https://example.com/post" in content
    assert "url: https://example.com/post" in content
    assert "source_type: notes" in content
    assert "Imported markdown body." in content


def test_add_pdf_writes_raw_note_with_segments(
    config: Config, db: StateDB, sample_pdf: Path, runner: CliRunner
) -> None:
    result = runner.invoke(
        cli, ["add", str(sample_pdf), "--type", "paper", "--vault", str(config.vault)]
    )
    assert result.exit_code == 0
    raw_files = list((config.vault / "raw").glob("*.md"))
    assert len(raw_files) == 1
    content = raw_files[0].read_text()
    assert "source_type: paper" in content


def test_add_pdf_surfaces_bibliographic_metadata_in_raw_frontmatter(
    config: Config, db: StateDB, tmp_path: Path, runner: CliRunner
) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "DOI: 10.9999/example.2021 Published 2021.")
    doc.set_metadata({"title": "Metadata Title", "author": "Alice; Bob"})
    pdf_path = tmp_path / "with_meta.pdf"
    doc.save(str(pdf_path))
    doc.close()

    result = runner.invoke(cli, ["add", str(pdf_path), "--vault", str(config.vault)])

    assert result.exit_code == 0, result.output
    raw_files = list((config.vault / "raw").glob("*.md"))
    assert len(raw_files) == 1
    content = raw_files[0].read_text()
    assert "source_title: Metadata Title" in content
    assert "doi: 10.9999/example.2021" in content
    assert "year: 2021" in content


def test_add_pdf_raw_note_preserves_image_refs(
    config: Config, db: StateDB, sample_pdf: Path, runner: CliRunner
) -> None:
    from types import SimpleNamespace
    from unittest.mock import patch

    segs = [
        SimpleNamespace(
            text="Segment with image.",
            structural_locator="section:intro",
            image_refs=["assets/src-1/img-0-0.png"],
        )
    ]
    with patch("synto.extractors.pdf.extract_pdf", return_value=segs):
        result = runner.invoke(cli, ["add", str(sample_pdf), "--vault", str(config.vault)])

    assert result.exit_code == 0, result.output
    raw_files = list((config.vault / "raw").glob("*.md"))
    assert len(raw_files) == 1
    content = raw_files[0].read_text()
    assert "### Media" in content
    assert "![[assets/src-1/img-0-0.png]]" in content


def test_add_pdf_writes_raw_note_from_extracted_segments_directly(
    config: Config, sample_pdf: Path, runner: CliRunner
) -> None:
    from types import SimpleNamespace
    from unittest.mock import patch

    segs = [
        SimpleNamespace(text="Segment A", structural_locator="section:intro"),
        SimpleNamespace(text="Segment B", structural_locator="section:methods"),
    ]

    with patch("synto.extractors.pdf.extract_pdf", return_value=segs):
        result = runner.invoke(cli, ["add", str(sample_pdf), "--vault", str(config.vault)])

    assert result.exit_code == 0, result.output
    raw_files = list((config.vault / "raw").glob("*.md"))
    assert len(raw_files) == 1
    content = raw_files[0].read_text()
    assert "Segment A" in content
    assert "Segment B" in content


def test_add_cleans_up_partial_import_when_raw_note_write_fails(
    config: Config, db: StateDB, sample_txt: Path, runner: CliRunner
) -> None:
    with patch("synto.pipeline.ingest.write_note", side_effect=RuntimeError("boom")):
        result = runner.invoke(cli, ["add", str(sample_txt), "--vault", str(config.vault)])

    assert result.exit_code != 0
    assert db.list_source_documents() == []
    assert list((config.vault / "raw").glob("*.md")) == []
    assert list((config.app_dir / "sources").glob("*/original.txt")) == []


# ---------------------------------------------------------------------------
# Semantic cache wired into _load_deps
# ---------------------------------------------------------------------------


def test_load_deps_passes_cache_to_build_router(config: Config) -> None:
    """_load_deps must pass a non-None LLMCache to build_router."""
    from synto.cli import _load_deps

    mock_router = MagicMock()
    mock_router.require_healthy.return_value = None

    # Patch at the source module since _load_deps imports build_router locally
    with patch("synto.client_factory.build_router", return_value=mock_router) as mock_build:
        _load_deps(config)

    _args, kwargs = mock_build.call_args
    assert "cache" in kwargs, "build_router must be called with cache= keyword"
    assert kwargs["cache"] is not None, "cache must be a live LLMCache, not None"


# ---------------------------------------------------------------------------
# #91: import decoding must not depend on silently mis-applying the host locale
# ---------------------------------------------------------------------------


def test_add_txt_legacy_locale_encoding_falls_back(
    config: Config, db: StateDB, tmp_path: Path, runner: CliRunner, monkeypatch
) -> None:
    """A legacy cp1251 note (not valid UTF-8) must import readably on a cp1251 host:
    UTF-8 is tried first, then the locale codec — not blind replacement."""
    p = tmp_path / "legacy.txt"
    p.write_bytes("Заметки о квантовой запутанности".encode("cp1251"))
    monkeypatch.setattr("locale.getpreferredencoding", lambda do_setlocale=True: "cp1251")

    result = runner.invoke(cli, ["add", str(p), "--vault", str(config.vault)])
    assert result.exit_code == 0, result.output

    raw_files = list(config.raw_dir.glob("*.md"))
    assert len(raw_files) == 1
    body = raw_files[0].read_text(encoding="utf-8")
    assert "квантовой запутанности" in body
    assert "�" not in body


def test_add_txt_utf8_content_imports_verbatim(
    config: Config, db: StateDB, tmp_path: Path, runner: CliRunner
) -> None:
    p = tmp_path / "utf8.txt"
    p.write_text("Заметки — с тире и юникодом", encoding="utf-8")

    result = runner.invoke(cli, ["add", str(p), "--vault", str(config.vault)])
    assert result.exit_code == 0, result.output

    raw_files = list(config.raw_dir.glob("*.md"))
    assert len(raw_files) == 1
    assert "Заметки — с тире и юникодом" in raw_files[0].read_text(encoding="utf-8")
