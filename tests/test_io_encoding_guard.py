"""Encoding guards for #91: generated files must be UTF-8 regardless of host locale.

Windows defaults text-mode I/O to the locale codepage (e.g. cp1251), which corrupted
`synto init`-generated configs: the em dash in TOML comments became byte 0x97 and the
strict-UTF-8 tomllib read side then failed every command. Ruff's PLW1514 flags bare
`open()`; this guard covers what ruff can't: `Path.read_text`/`write_text` and
subprocess text decoding.
"""

from __future__ import annotations

import ast
from pathlib import Path

from click.testing import CliRunner

from synto.cli import cli

SRC = Path(__file__).resolve().parents[1] / "src" / "synto"


def _encoding_violations() -> list[str]:
    violations: list[str] = []
    for py in sorted(SRC.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            kwargs = {kw.arg for kw in node.keywords}
            where = f"{py.relative_to(SRC)}:{node.lineno}"
            if node.func.attr in ("read_text", "write_text") and "encoding" not in kwargs:
                violations.append(f"{where} {node.func.attr}() without encoding=")
            if (
                node.func.attr == "run"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"
                and ({"text", "universal_newlines"} & kwargs)
                and "encoding" not in kwargs
            ):
                violations.append(f"{where} subprocess.run(text=...) without encoding=")
    return violations


def test_all_text_io_pins_encoding():
    violations = _encoding_violations()
    assert not violations, (
        "Text-mode I/O without explicit encoding= decodes/encodes with the Windows "
        "locale codepage and corrupts generated files (#91):\n" + "\n".join(violations)
    )


def test_init_writes_utf8_config_regardless_of_locale(tmp_path, monkeypatch):
    """True regression test on the windows-latest CI job: its cp1252 locale also encodes
    the em dash (as 0x97), so an unpinned write_text produces invalid UTF-8 there."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    vault = tmp_path / "vault"
    result = CliRunner().invoke(cli, ["init", str(vault)])
    assert result.exit_code == 0, result.output

    raw = (vault / "synto.toml").read_bytes()
    text = raw.decode("utf-8")  # raises UnicodeDecodeError if written in a locale codepage
    assert "—" in text  # the em dash that broke cp1251/cp1252 vaults must round-trip
