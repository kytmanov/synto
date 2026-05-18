"""Tests for Feature 05: synto add command."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

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
    assert copies[0].read_text() == sample_txt.read_text()


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


def test_extend_pack_creates_toml_entry(
    config: Config, db: StateDB, sample_txt: Path, runner: CliRunner
) -> None:
    runner.invoke(
        cli,
        ["add", "--extend-pack", "my-pack", str(sample_txt), "--vault", str(config.vault)],
    )
    toml_path = config.vault / "synto.toml"
    assert toml_path.exists()
    text = toml_path.read_text()
    assert "[[pack.sources]]" in text
    assert "my-pack" in text or "id" in text


def test_extend_pack_appends_to_existing_toml(
    config: Config, db: StateDB, sample_txt: Path, runner: CliRunner
) -> None:
    toml_path = config.vault / "synto.toml"
    toml_path.write_text('[models]\nfast = "gemma4:e4b"\n')
    runner.invoke(
        cli,
        ["add", "--extend-pack", "my-pack", str(sample_txt), "--vault", str(config.vault)],
    )
    text = toml_path.read_text()
    assert "[models]" in text  # original content preserved
    assert "[[pack.sources]]" in text


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


def test_get_source_document_method(config: Config, db: StateDB) -> None:
    """get_source_document returns None for unknown IDs."""
    assert db.get_source_document("nonexistent") is None
