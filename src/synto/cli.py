"""
Synto CLI

Commands:
  init     — create vault structure (or adopt existing)
  ingest   — analyze raw notes
  compile  — synthesize notes into wiki articles (writes to .drafts/)
  approve  — publish drafts to wiki/
  reject   — discard a draft
  status   — show vault health and pending drafts
  report   — vault analytics and metrics (report clear to reset)
  undo     — revert last N synto auto-commits
  query    — RAG-powered Q&A (Phase 2)
  watch    — file watcher daemon (Phase 3)
"""

from __future__ import annotations

import re
import sys
import tomllib
from contextlib import contextmanager
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Prompt
from rich.table import Table

from .paths import (
    APP_DIR_NAME,
    CLI_NAME,
    CONFIG_FILE_NAME,
    LEGACY_APP_DIR_NAME,
    LEGACY_CONFIG_FILE_NAME,
    VAULT_ENV_VAR,
    config_path,
    is_legacy_vault,
    legacy_config_path,
    migration_message,
)


def _ensure_utf8_streams() -> None:
    """Reconfigure stdout/stderr to UTF-8 when the console can't encode the
    status glyphs (✓, ✗) we print — see issue #23. No-op when already UTF-8.

    The classic case is a Windows legacy console (cp1252), but an ascii/POSIX
    locale on Linux/CI hits the same UnicodeEncodeError. Must run before the
    Console objects below are built, since rich binds the stream at construction.
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is None:  # e.g. pythonw / detached GUI process
            continue
        encoding = (getattr(stream, "encoding", None) or "").lower()
        if encoding.startswith("utf"):
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # non-TextIOWrapper (pytest capture, pipe wrapper)
            continue
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass


_ensure_utf8_streams()

console = Console()
err_console = Console(stderr=True, style="bold red")

PROJECT_REPO_URL = "https://github.com/kytmanov/synto"
PROJECT_ISSUES_URL = f"{PROJECT_REPO_URL}/issues"
PROJECT_DISCUSSIONS_URL = f"{PROJECT_REPO_URL}/discussions"

_EXPERIMENTAL_CITATIONS_COPY = (
    "Links each generated claim back to its source page.\n"
    "    [dim]Note:[/dim] small models may omit citations or add noisy markers.\n"
    f"    [dim]Change later:[/dim]\n"
    f"    [bold]{CLI_NAME} config inline-source-citations on|off --vault <path>[/bold]"
)


def _format_optional_bool(value: bool | None) -> str:
    if value is None:
        return "[dim]not set[/dim]"
    return "on" if value else "off"


class InlineSourceCitationsConfigError(Exception):
    """Raised when inline citation config cannot be read safely."""


def _read_inline_source_citations_setting(toml_path: Path, *, strict: bool = False) -> bool | None:

    if not toml_path.exists():
        return None
    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        if strict:
            raise InlineSourceCitationsConfigError(f"Invalid TOML in {toml_path}: {exc}") from exc
        return None
    except OSError as exc:
        if strict:
            raise InlineSourceCitationsConfigError(f"Could not read {toml_path}: {exc}") from exc
        return None
    pipeline = data.get("pipeline", {})
    value = pipeline.get("inline_source_citations") if isinstance(pipeline, dict) else None
    if value is not None and not isinstance(value, bool):
        if strict:
            raise InlineSourceCitationsConfigError(
                "Invalid pipeline.inline_source_citations in "
                f"{toml_path}: expected boolean true/false, got {type(value).__name__}"
            )
        return None
    return value if isinstance(value, bool) else None


def _set_inline_source_citations(toml_path: Path, enabled: bool) -> None:
    """Patch one pipeline key while preserving unrelated vault config content."""
    from .vault import atomic_write

    if not toml_path.exists():
        raise FileNotFoundError(toml_path)

    text = toml_path.read_text(encoding="utf-8")
    line = f"inline_source_citations = {'true' if enabled else 'false'}"
    section_match = re.search(r"(?m)^\[pipeline\]\s*$", text)

    if section_match is None:
        separator = "" if text.endswith("\n") or not text else "\n"
        atomic_write(toml_path, f"{text}{separator}\n[pipeline]\n{line}\n")
        return

    section_start = section_match.end()
    next_section = re.search(r"(?m)^\[[^\]]+\]\s*$", text[section_start:])
    section_end = section_start + next_section.start() if next_section else len(text)
    section = text[section_start:section_end]
    key_re = re.compile(r"(?m)^(\s*)#?\s*inline_source_citations\s*=.*$")

    if key_re.search(section):
        new_section = key_re.sub(rf"\1{line}", section, count=1)
    else:
        insertion = ("" if section.endswith("\n") or not section else "\n") + line + "\n"
        new_section = section + insertion

    atomic_write(toml_path, text[:section_start] + new_section + text[section_end:])


def _normalize_migrated_legacy_config(toml_path: Path) -> None:
    from .vault import atomic_write

    text = toml_path.read_text(encoding="utf-8")
    normalized = re.sub(r"(?m)^\[telemetry\]\s*$", "[metrics]", text)
    if normalized != text:
        atomic_write(toml_path, normalized)


# ── Context helpers ───────────────────────────────────────────────────────────


def _resolve_vault_path(vault_str: str | None) -> Path:
    import os

    from .global_config import load_global_config

    if vault_str is None:
        vault_str = os.environ.get(VAULT_ENV_VAR)

    if vault_str is None:
        gcfg = load_global_config()
        vault_str = gcfg.vault if gcfg and gcfg.vault else None

    if not vault_str:
        cwd = Path.cwd()
        for parent in [cwd, *cwd.parents]:
            if config_path(parent).exists() or legacy_config_path(parent).exists():
                vault_str = str(parent)
                break

    if not vault_str:
        click.echo(
            f"Error: no vault specified. Use --vault, set {VAULT_ENV_VAR}, run `{CLI_NAME} setup`, "
            "or cd into a vault directory.",
            err=True,
        )
        sys.exit(1)
    vault_path = Path(vault_str).expanduser().resolve()
    if not vault_path.exists():
        click.echo(
            f"Error: vault path does not exist: {vault_path}\n"
            f"Run `{CLI_NAME} init {vault_path}` to create it, or re-run `{CLI_NAME} setup` "
            f"to update the default vault.",
            err=True,
        )
        sys.exit(1)
    if not vault_path.is_dir():
        click.echo(
            f"Error: vault path is not a directory: {vault_path}\n"
            f"A vault is a directory containing {CONFIG_FILE_NAME}. "
            f"Point --vault / {VAULT_ENV_VAR} at the parent directory instead.",
            err=True,
        )
        sys.exit(1)
    if is_legacy_vault(vault_path):
        click.echo(migration_message(vault_path), err=True)
        sys.exit(1)
    return vault_path


def _load_config(vault_str: str | None, **kwargs):
    from .config import Config

    return Config.from_vault(_resolve_vault_path(vault_str), **kwargs)


def _load_db(config):
    from .state import StateDB

    return StateDB(config.state_db_path)


@contextmanager
def _metrics_context(config, db=None):
    from .metrics import PersistentMetricsSink, persistent_metrics_sink

    owns_db = db is None
    if db is None:
        db = _load_db(config)
    try:
        with persistent_metrics_sink(PersistentMetricsSink(db, config.metrics, config.vault)):
            yield db
    finally:
        if owns_db:
            db.close()


def _model_override_options(f):
    """Shared decorator adding --fast-model/--heavy-model/--provider/--provider-url."""
    f = click.option(
        "--fast-model",
        "fast_model",
        default=None,
        help="Override fast model for this invocation",
    )(f)
    f = click.option(
        "--heavy-model",
        "heavy_model",
        default=None,
        help="Override heavy model for this invocation",
    )(f)
    f = click.option(
        "--provider",
        "provider_name",
        default=None,
        help="Override provider name (ollama, groq, openai, azure, ...)",
    )(f)
    f = click.option(
        "--provider-url",
        "provider_url",
        default=None,
        help="Override provider base URL (e.g. https://api.groq.com/openai/v1)",
    )(f)
    return f


def _model_override_kwargs(
    fast_model: str | None,
    heavy_model: str | None,
    provider_name: str | None,
    provider_url: str | None,
) -> dict:
    """Pack CLI model-override flags into kwargs for Config.from_vault."""
    kwargs: dict = {}
    models: dict = {}
    if fast_model:
        models["fast"] = fast_model
    if heavy_model:
        models["heavy"] = heavy_model
    if models:
        kwargs["models"] = models
    provider: dict = {}
    if provider_name:
        provider["name"] = provider_name
    if provider_url:
        provider["url"] = provider_url
    if provider:
        kwargs["provider"] = provider
    return kwargs


def _resolve_draft_arg(config, raw_path: str | Path) -> Path:
    """Resolve a CLI draft argument relative to wiki/.drafts/ when appropriate."""
    path = Path(raw_path).expanduser()
    candidates: list[Path]
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [config.drafts_dir / path, config.vault / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _load_deps(config):
    from .cache import LLMCache
    from .client_factory import LLMError, build_client

    db = _load_db(config)
    cache = LLMCache(db)
    client = build_client(config, cache=cache)
    try:
        client.require_healthy()
    except LLMError as e:
        err_console.print(str(e))
        db.close()
        sys.exit(1)
    ctx = click.get_current_context(silent=True)
    if ctx is not None:
        metrics_cm = _metrics_context(config, db)
        metrics_cm.__enter__()
        ctx.call_on_close(lambda cm=metrics_cm: cm.__exit__(None, None, None))
        ctx.call_on_close(db.close)
        ctx.call_on_close(client.close)
    return client, db


# ── CLI root ──────────────────────────────────────────────────────────────────


@click.group()
@click.version_option(package_name="synto")
def cli():
    """Synto — local knowledge packs and synthesized wiki pipeline.

    Run `synto setup` for interactive configuration.
    Run `synto support` for bug reports, suggestions, and feedback links.
    """
    import logging

    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, show_time=False)],
    )
    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


# ── pack ──────────────────────────────────────────────────────────────────────


@cli.group()
def pack():
    """Pack export commands."""


@pack.command("export")
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@click.option(
    "--target",
    type=click.Choice(["agents"]),
    required=True,
    help="Export target. Phase 1A supports only 'agents'.",
)
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory (default: .synto/exports/<target> inside the vault).",
)
def pack_export(vault_str: str | None, target: str, out: Path | None) -> None:
    """Export the working vault as a portable pack."""
    from .pack_export import export_pack

    config = _load_config(vault_str)
    result = export_pack(config, target=target, out=out)
    console.print(f"[green]Exported {result.n_articles} articles to {result.out_dir}[/green]")
    console.print(f"Capabilities: {', '.join(sorted(result.capabilities))}")


@cli.group(invoke_without_command=True)
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@click.option(
    "--since",
    default=None,
    help="Filter metrics to an ISO date (YYYY-MM-DD) or Nd like 7d.",
)
@click.option("--json", "json_out", is_flag=True, help="Emit machine-readable JSON")
@click.pass_context
def report(ctx, vault_str, since, json_out):
    """Show vault analytics and runtime metrics report."""
    if ctx.invoked_subcommand is None:
        from .stats import compute_stats, render_json, render_text

        config = _load_config(vault_str)
        try:
            r = compute_stats(config, since=since)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="--since") from exc
        click.echo(render_json(r) if json_out else render_text(r))


@report.command("clear")
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@click.option("--yes", is_flag=True, default=False, help="Skip confirmation prompt.")
def report_clear(vault_str: str | None, yes: bool) -> None:
    """Delete all stored runtime metrics events and daily rollups."""
    from .state import StateDB

    config = _load_config(vault_str)
    if not config.state_db_path.exists():
        click.echo("No metrics data found (state.db does not exist).")
        return
    if not yes:
        click.confirm("This will permanently delete all metrics data. Continue?", abort=True)
    db = StateDB(config.state_db_path)
    try:
        deleted = db.clear_metrics()
        click.echo(f"Metrics cleared ({deleted} rows deleted).")
    finally:
        db.close()


# ── init ──────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("vault_path", type=click.Path())
@click.option("--existing", is_flag=True, help="Adopt an existing Obsidian vault")
@click.option("--non-interactive", is_flag=True)
@click.option(
    "--default",
    "set_default",
    is_flag=True,
    help="Set this vault as the default (no --vault flag needed for future commands)",
)
def init(vault_path: str, existing: bool, non_interactive: bool, set_default: bool):
    """Create vault structure and initialise Synto."""
    from .git_ops import git_init

    vault = Path(vault_path).expanduser().resolve()
    vault.mkdir(parents=True, exist_ok=True)

    if existing:
        _init_existing(vault, non_interactive)
    else:
        _init_fresh(vault)

    # Write or sync synto.toml from global config
    toml_path = vault / CONFIG_FILE_NAME
    from .config import default_wiki_toml
    from .global_config import GlobalConfig, load_global_config, save_global_config

    gcfg = load_global_config()
    provider_name = gcfg.provider_name if gcfg and gcfg.provider_name else "ollama"
    # Only fall back to Ollama-specific model names when using Ollama; cloud providers
    # must have been configured explicitly via `synto setup`.
    _ollama = provider_name == "ollama"
    fast = gcfg.fast_model if gcfg and gcfg.fast_model else ("gemma4:e4b" if _ollama else "")
    heavy = gcfg.heavy_model if gcfg and gcfg.heavy_model else ("qwen2.5:14b" if _ollama else "")
    provider_url = gcfg.provider_url if gcfg and gcfg.provider_url else None
    ollama_url = gcfg.ollama_url if gcfg and gcfg.ollama_url else "http://localhost:11434"
    effective_url = provider_url or ollama_url
    azure_api_version = gcfg.azure_api_version if gcfg and gcfg.azure_api_version else None

    if not toml_path.exists():
        from .providers import get_provider

        prov_info = get_provider(provider_name)
        timeout = prov_info.default_timeout if prov_info else 600.0
        toml_path.write_text(
            default_wiki_toml(
                fast,
                heavy,
                ollama_url=ollama_url,
                provider_name=provider_name,
                provider_url=effective_url if provider_name != "ollama" else None,
                provider_timeout=timeout,
                azure_api_version=azure_api_version,
                inline_source_citations=(
                    bool(gcfg.experimental_inline_source_citations) if gcfg else False
                ),
            )
        )
    else:
        # Existing vault: patch model/URL fields from global config so that
        # synto setup changes are reflected without overwriting pipeline settings.
        _sync_wiki_toml_models(
            toml_path,
            fast,
            heavy,
            effective_url,
            provider_name=provider_name if provider_name != "ollama" else None,
        )

    # Init git
    git_init(vault)

    # Create .gitignore
    gi = vault / ".gitignore"
    if not gi.exists():
        gi.write_text(
            ".DS_Store\n"
            ".synto/chroma/\n"
            ".synto/state.db\n"
            ".synto/compare/\n"
            ".synto/pipeline.lock\n"
            ".synto/exports/\n"
            ".obsidian/workspace.json\n"
            "*.log\n"
        )

    if set_default:
        try:
            _gcfg = gcfg if gcfg is not None else GlobalConfig()
            _gcfg.vault = str(vault)
            save_global_config(_gcfg)
        except Exception:
            console.print("[yellow]⚠ Could not save default vault to global config.[/yellow]")

    console.print(f"[green]Vault initialised:[/green] {vault}")
    if set_default:
        console.print("[dim]Set as default vault — no --vault flag needed.[/dim]")
    console.print("Next steps:")
    console.print("  1. Drop .md notes into [bold]raw/[/bold]")
    if set_default:
        console.print(f"  2. Run [bold]{CLI_NAME} run[/bold]")
        console.print(f"  3. Review drafts: [bold]{CLI_NAME} review[/bold]")
        console.print(f"  4. Publish all drafts: [bold]{CLI_NAME} approve --all[/bold]")
    else:
        console.print(f"  2. Run [bold]{CLI_NAME} run --vault {vault}[/bold]")
        console.print(f"  3. Review drafts: [bold]{CLI_NAME} review --vault {vault}[/bold]")
        console.print(
            f"  4. Publish all drafts: [bold]{CLI_NAME} approve --all --vault {vault}[/bold]"
        )


def _sync_wiki_toml_models(
    toml_path: Path,
    fast: str,
    heavy: str,
    ollama_url: str,
    provider_name: str | None = None,
) -> None:
    """Patch fast/heavy model, URL, and optionally provider name in an existing synto.toml.

    Preserves all other settings (pipeline, rag, etc.) so user customisations
    are not lost. Only updates fields that come from global config.

    URL is only updated within the [ollama] or [provider] section, never globally,
    so switching providers cannot overwrite unrelated url= fields.
    """
    import re

    text = toml_path.read_text(encoding="utf-8")
    original = text

    def _replace_value(t: str, key: str, value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return re.sub(
            rf'^({re.escape(key)}\s*=\s*)".+"',
            rf'\g<1>"{escaped}"',
            t,
            flags=re.MULTILINE,
        )

    def _replace_in_section(t: str, section: str, key: str, value: str) -> str:
        """Replace key=value only within the named TOML section."""
        escaped_val = value.replace("\\", "\\\\").replace('"', '\\"')
        # Match the section header then capture everything until the next section or EOF
        pattern = rf'(\[{re.escape(section)}\][^\[]*?)^({re.escape(key)}\s*=\s*)".+"'
        replacement = rf'\1\2"{escaped_val}"'
        return re.sub(pattern, replacement, t, flags=re.MULTILINE | re.DOTALL)

    text = _replace_value(text, "fast", fast)
    text = _replace_value(text, "heavy", heavy)
    # Update URL only in [ollama] or [provider] sections to avoid clobbering other urls
    for section in ("ollama", "provider"):
        text = _replace_in_section(text, section, "url", ollama_url)
    if provider_name is not None:
        if "[provider]" not in text:
            console.print(
                f"  [yellow]Warning:[/yellow] {CONFIG_FILE_NAME} has no [provider] section — "
                f"provider '{provider_name}' not applied. "
                f"Delete {CONFIG_FILE_NAME} and re-run [bold]synto init[/bold] to regenerate it."
            )
        else:
            text = _replace_in_section(text, "provider", "name", provider_name)

    if text != original:
        toml_path.write_text(text, encoding="utf-8")
        console.print(
            f"[dim]{CONFIG_FILE_NAME} updated: fast={fast}, heavy={heavy}, url={ollama_url}[/dim]"
        )


def _init_fresh(vault: Path) -> None:
    for d in ["raw", "wiki", "wiki/.drafts", "wiki/sources", ".synto", ".synto/chroma"]:
        (vault / d).mkdir(parents=True, exist_ok=True)
    _write_vault_schema(vault)
    _write_index(vault)
    console.print("[dim]Created fresh vault structure[/dim]")


def _init_existing(vault: Path, non_interactive: bool) -> None:
    note_count = sum(1 for _ in vault.rglob("*.md"))
    console.print(f"Found [bold]{note_count}[/bold] existing .md files in {vault}")

    for d in ["raw", "wiki", "wiki/.drafts", "wiki/sources", ".synto", ".synto/chroma"]:
        (vault / d).mkdir(parents=True, exist_ok=True)

    if not non_interactive and note_count > 0:
        if click.confirm(f"Treat existing notes as raw source material? ({note_count} files)"):
            console.print("[dim]Existing notes will be ingested as raw material.[/dim]")
            console.print("[dim]Run [bold]synto ingest --all[/bold] to process them.[/dim]")

    _write_vault_schema(vault)
    _write_index(vault)
    _cleanup_legacy_index(vault)


def _cleanup_legacy_index(vault: Path) -> None:
    """Remove wiki/INDEX.md if it's the bootstrap stub and is distinct from wiki/index.md."""
    old = vault / "wiki" / "INDEX.md"
    new = vault / "wiki" / "index.md"
    if not old.exists():
        return
    # On case-insensitive FS old and new are the same file — don't delete
    if new.exists():
        try:
            if old.samefile(new):
                return
        except OSError:
            return
    try:
        content = old.read_text(encoding="utf-8")
        if content == _INDEX_STUB:
            old.unlink()
    except Exception:
        pass


def _write_vault_schema(vault: Path) -> None:
    schema_path = vault / "vault-schema.md"
    if not schema_path.exists():
        schema_path.write_text(
            "# Vault Schema\n\n"
            "## Folder Structure\n"
            "- `raw/` — input notes (immutable, never edited by synto)\n"
            "- `wiki/` — AI-synthesised articles (managed by synto)\n"
            "- `wiki/.drafts/` — pending human review\n\n"
            "## Note Format\n"
            "Every wiki note has YAML frontmatter with: title, tags, sources, "
            "confidence, status, created, updated.\n\n"
            "## Links\n"
            "Use `[[Article Title]]` wikilinks between notes.\n"
        )


_INDEX_STUB = (
    "---\ntitle: Index\ntags: [index]\nstatus: published\n---\n\n"
    "# Wiki Index\n\n_Updated automatically by synto._\n"
)


def _write_index(vault: Path) -> None:
    index = vault / "wiki" / "index.md"
    if not index.exists():
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(_INDEX_STUB)


@cli.command("migrate-olw")
@click.option("--vault", "vault_str", type=click.Path(path_type=Path), required=True)
def migrate_olw(vault_str: Path) -> None:
    """Copy an old olw vault layout into the Synto layout."""
    import shutil

    from .state import StateDB

    vault = Path(vault_str).expanduser().resolve()
    old_cfg = legacy_config_path(vault)
    new_cfg = config_path(vault)
    old_dir = vault / LEGACY_APP_DIR_NAME
    new_dir = vault / APP_DIR_NAME

    if not old_cfg.exists() and not old_dir.exists():
        raise click.ClickException("No old olw vault layout found to migrate.")
    if new_cfg.exists() or new_dir.exists():
        raise click.ClickException(
            f"Refusing to overwrite existing {CONFIG_FILE_NAME} or {APP_DIR_NAME}."
        )

    if old_cfg.exists():
        shutil.copy2(old_cfg, new_cfg)
        _normalize_migrated_legacy_config(new_cfg)
    if old_dir.exists():
        shutil.copytree(old_dir, new_dir)
        db = StateDB(new_dir / "state.db")
        db.close()

    gitignore = vault / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text(encoding="utf-8")
    else:
        content = ""
    for line in [
        ".synto/chroma/",
        ".synto/state.db",
        ".synto/compare/",
        ".synto/pipeline.lock",
        ".synto/exports/",
    ]:
        if line not in content:
            content += ("" if not content or content.endswith("\n") else "\n") + line + "\n"
    gitignore.write_text(content, encoding="utf-8")

    console.print(f"[green]Migrated vault:[/green] {vault}")
    if old_cfg.exists():
        console.print(f"  copied {LEGACY_CONFIG_FILE_NAME} -> {CONFIG_FILE_NAME}")
    if old_dir.exists():
        console.print(f"  copied {LEGACY_APP_DIR_NAME}/ -> {APP_DIR_NAME}/")


# ── setup ─────────────────────────────────────────────────────────────────────


def _pick_model(
    console: Console,
    client,
    step_label: str,
    description: str,
    default_fallback: str,
    connected: bool,
) -> str:
    """Interactive model selector — shows table if models available, else free-text."""
    console.print()
    console.print(f"  [bold]{step_label}[/bold]  {description}")

    models: list[dict] = []
    if connected:
        models = client.list_models_detailed()

    if models:
        table = Table(show_header=True, box=None, padding=(0, 2))
        table.add_column("#", style="dim", width=3)
        table.add_column("Model")
        table.add_column("Size", style="dim")
        for i, m in enumerate(models, 1):
            table.add_row(str(i), m["name"], m["size_gb"])
        console.print(table)
        console.print()
        raw = Prompt.ask("    Select (number or name)", default="1", console=console).strip()
        if not raw:
            return default_fallback
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(models):
                return models[idx]["name"]
            console.print(f"    [yellow]Invalid number, using {default_fallback}[/yellow]")
            return default_fallback
        return raw
    else:
        if connected:
            console.print(
                "    [yellow]No models found.[/yellow] "
                "Pull one first: [bold]ollama pull gemma4:e4b[/bold]"
            )
        console.print("    (e.g. gemma4:e4b, llama3.2:3b, qwen2.5:14b)")
        raw = Prompt.ask("    Model name", default=default_fallback, console=console).strip()
        return raw if raw else default_fallback


@cli.command()
@click.option("--non-interactive", is_flag=True, help="Print current config and exit")
@click.option("--reset", is_flag=True, help="Clear saved config and re-run wizard")
@click.option(
    "--provider",
    "provider_preset",
    default=None,
    help="Skip provider selection (e.g. groq, lm_studio)",
)
def setup(non_interactive: bool, reset: bool, provider_preset: str | None):
    """Interactive wizard: configure provider, models, and default vault."""
    from .global_config import GlobalConfig, load_global_config, save_global_config
    from .providers import PROVIDER_REGISTRY, get_provider, list_all_providers

    # ── non-interactive: show current config ──────────────────────────────────
    if non_interactive:
        gcfg = load_global_config()
        if not gcfg:
            console.print(
                "[dim]No global config found. Run [bold]synto setup[/bold] to configure.[/dim]"
            )
            return
        table = Table(title="Global config", show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold")
        table.add_column("Value")
        prov_display = gcfg.provider_name or (gcfg.ollama_url and "ollama") or "[dim]not set[/dim]"
        table.add_row("Provider", prov_display)
        table.add_row("URL", gcfg.provider_url or gcfg.ollama_url or "[dim]not set[/dim]")
        table.add_row("API key", "***" if gcfg.api_key else "[dim]not set[/dim]")
        table.add_row("Fast model", gcfg.fast_model or "[dim]not set[/dim]")
        table.add_row("Heavy model", gcfg.heavy_model or "[dim]not set[/dim]")
        table.add_row("Default vault", gcfg.vault or "[dim]not set[/dim]")
        table.add_row(
            "Inline source citations for new vaults",
            _format_optional_bool(gcfg.experimental_inline_source_citations),
        )
        console.print(table)
        return

    # ── reset: wipe config before wizard ─────────────────────────────────────
    if reset:
        save_global_config(GlobalConfig())
        console.print("[dim]Config cleared.[/dim]")

    try:
        # ── Header ───────────────────────────────────────────────────────────
        console.print()
        from importlib.metadata import version as _pkg_version

        try:
            _ver = _pkg_version("synto")
        except Exception:
            _ver = "unknown"
        from rich import box as _box
        from rich.markup import escape as _re

        _ascii_lines = [
            ("   _____             __      ", "bold bright_blue"),
            ("  / ___/__  ______  / /_____", "bold bright_blue"),
            ("  \\__ \\/ / / / __ \\/ __/ __ \\", "bold cyan"),
            (" ___/ / /_/ / / / / /_/ /_/ /", "bold cyan"),
            ("/____/\\__, /_/ /_/\\__/\\____/ ", "bold bright_cyan"),
            ("     /____/                  ", "bold bright_cyan"),
        ]
        _ascii = "\n".join(f"[{color}]{_re(line)}[/{color}]" for line, color in _ascii_lines)
        _subtitle = (
            f"\n  [dim]Knowledge compiler[/dim]"
            f"   [bold white]v{_ver}[/bold white]"
            f"  [dim]·[/dim]  [bold cyan]setup wizard[/bold cyan]"
        )
        console.print(
            Panel(
                _ascii + _subtitle,
                expand=False,
                border_style="bright_blue",
                padding=(0, 2),
                box=_box.ROUNDED,
            )
        )
        console.print()

        all_providers = list_all_providers()

        # ── Step 1 — Provider selection ───────────────────────────────────────
        if provider_preset:
            chosen_prov = get_provider(provider_preset)
            if chosen_prov is None:
                console.print(
                    f"    [yellow]Unknown provider '{provider_preset}', using Ollama.[/yellow]"
                )
                chosen_prov = PROVIDER_REGISTRY["ollama"]
            chosen_name = chosen_prov.name
        else:
            console.print("  [bold]Step 1[/bold]  Provider\n")

            # Build numbered list
            local_provs = [p for p in all_providers if p.is_local]
            cloud_provs = [p for p in all_providers if not p.is_local and p.name != "custom"]

            console.print("    [bold]Local[/bold] (no API key needed):")
            idx_map: dict[int, str] = {}
            counter = 1
            for p in local_provs:
                marker = "  [default]" if p.name == "ollama" else ""
                console.print(f"      {counter:2}. {p.display_name:<14} {p.default_url}{marker}")
                idx_map[counter] = p.name
                counter += 1

            console.print()
            console.print("    [bold]Cloud[/bold] (API key required):")
            for p in cloud_provs:
                url_hint = p.default_url if p.default_url else "(enter URL manually)"
                console.print(f"      {counter:2}. {p.display_name:<14} {url_hint}")
                idx_map[counter] = p.name
                counter += 1

            console.print()
            console.print(f"      {counter:2}. Custom         (enter URL manually)")
            idx_map[counter] = "custom"

            console.print()
            raw = Prompt.ask(
                "    Select provider (number or name)", default="1", console=console
            ).strip()

            if raw.isdigit():
                num = int(raw)
                chosen_name = idx_map.get(num, "ollama")
            elif raw in PROVIDER_REGISTRY:
                chosen_name = raw
            else:
                console.print(f"    [yellow]Unknown '{raw}', defaulting to Ollama.[/yellow]")
                chosen_name = "ollama"

            chosen_prov = PROVIDER_REGISTRY[chosen_name]

        # ── Step 2 — URL ──────────────────────────────────────────────────────
        console.print()
        console.print("  [bold]Step 2[/bold]  URL")
        default_url = chosen_prov.default_url or ""
        if chosen_name == "azure":
            console.print(
                "    Azure format: https://{resource}.openai.azure.com/openai/deployments/{model}"
            )
        provider_url = Prompt.ask("    Base URL", default=default_url, console=console).strip()
        if not provider_url:
            provider_url = default_url
        if not provider_url and chosen_name in ("custom", "azure"):
            console.print(
                "    [red]URL is required for this provider. "
                "Run [bold]synto setup[/bold] again and enter a valid URL.[/red]"
            )
            sys.exit(1)

        # ── Step 3 — API key (all non-Ollama providers) ──────────────────────
        # Local providers (vLLM, LM Studio, etc.) default to no-auth but can
        # require a key in enterprise deployments, so we always offer the prompt.
        import os

        needs_key_prompt = chosen_name != "ollama"
        api_key: str | None = None
        if needs_key_prompt:
            console.print()
            console.print("  [bold]Step 3[/bold]  API key")
            if chosen_prov.env_var:
                env_hint = f"  [dim](or set {chosen_prov.env_var} env var)[/dim]"
            else:
                env_hint = "  [dim](optional — press Enter to skip)[/dim]"
            console.print(f"    API key{env_hint}")
            try:
                raw_key = Prompt.ask("    Key", default="", password=True, console=console).strip()
            except Exception:
                console.print(
                    "    [dim]Note: terminal does not support hidden input "
                    "— key will be visible[/dim]"
                )
                raw_key = Prompt.ask("    Key", default="", console=console).strip()
            api_key = raw_key if raw_key else None

        # ── Build a temp client to probe for model list ───────────────────────
        if chosen_name == "ollama":
            from .ollama_client import OllamaClient

            temp_client = OllamaClient(base_url=provider_url, timeout=5)
        else:
            from .openai_compat_client import OpenAICompatClient

            resolved_key = api_key
            if not resolved_key and chosen_prov.env_var:
                resolved_key = os.environ.get(chosen_prov.env_var)
            if not resolved_key:
                resolved_key = os.environ.get("SYNTO_API_KEY")
            temp_client = OpenAICompatClient(
                base_url=provider_url,
                provider_name=chosen_name,
                api_key=resolved_key,
                timeout=5,
                supports_json_mode=chosen_prov.supports_json_mode,
                supports_embeddings=chosen_prov.supports_embeddings,
                azure=chosen_prov.azure,
            )
        connected = temp_client.healthcheck()
        if connected:
            console.print("    [green]✓ connected[/green]")
        else:
            console.print(
                f"    [yellow]Warning:[/yellow] Cannot reach {provider_url} — continuing anyway."
            )

        # ── Default model names per provider ──────────────────────────────────
        # For non-Ollama providers, leave defaults empty — model names are
        # provider-specific and must be entered by the user.
        default_fast = "gemma4:e4b" if chosen_name == "ollama" else ""
        default_heavy = "qwen2.5:14b" if chosen_name == "ollama" else ""
        if chosen_name != "ollama" and not connected:
            console.print(
                "    [dim]Tip: enter the model name exactly as the provider lists it "
                "(e.g. llama-3.1-70b-versatile for Groq).[/dim]"
            )

        step_offset = 1 if needs_key_prompt else 0

        # ── Step 4 — Fast model ───────────────────────────────────────────────
        fast_model = _pick_model(
            console=console,
            client=temp_client,
            step_label=f"Step {3 + step_offset}",
            description="Fast model  [dim](analysis & routing · 3–8B recommended)[/dim]",
            default_fallback=default_fast,
            connected=connected,
        )

        # ── Step 5 — Heavy model ──────────────────────────────────────────────
        heavy_model = _pick_model(
            console=console,
            client=temp_client,
            step_label=f"Step {4 + step_offset}",
            description="Heavy model  [dim](article writing · 7–14B recommended)[/dim]",
            default_fallback=default_heavy,
            connected=connected,
        )

        temp_client.close()

        # ── Final step — Default vault ────────────────────────────────────────
        console.print()
        step_label = f"Step {5 + step_offset}"
        console.print(
            f"  [bold]{step_label}[/bold]  Default vault path  [dim](press Enter to skip)[/dim]"
        )
        vault_input = Prompt.ask("    Vault path", default="", console=console)
        vault_path: str | None = None
        if vault_input.strip():
            vault_path = str(Path(vault_input).expanduser().resolve())

        # ── Experimental features ─────────────────────────────────────────────
        console.print()
        step_label = f"Step {6 + step_offset}"
        console.print(f"  [bold]{step_label}[/bold]  Experimental features (optional)\n")
        console.print("    [bold]Inline source citations[/bold]")
        console.print(f"    {_EXPERIMENTAL_CITATIONS_COPY}")
        console.print()
        raw_citations = (
            Prompt.ask(
                "    Enable inline source citations for new vaults?",
                choices=["y", "n"],
                default="n",
                show_choices=False,
                console=console,
            )
            .strip()
            .lower()
        )
        experimental_inline_source_citations = raw_citations == "y"

        applied_to_existing_vault = False
        current_vault_setting: bool | None = None
        current_toml_exists = False
        if vault_path:
            current_toml = Path(vault_path) / CONFIG_FILE_NAME
            if not current_toml.exists():
                current_toml = Path(vault_path) / LEGACY_CONFIG_FILE_NAME
            current_toml_exists = current_toml.exists()
            current_vault_setting = _read_inline_source_citations_setting(current_toml)
            if current_toml_exists:
                apply_now = (
                    Prompt.ask(
                        f"    Apply this setting to {current_toml} now?",
                        choices=["y", "n"],
                        default="n",
                        show_choices=False,
                        console=console,
                    )
                    .strip()
                    .lower()
                )
                if apply_now == "y":
                    _set_inline_source_citations(current_toml, experimental_inline_source_citations)
                    current_vault_setting = experimental_inline_source_citations
                    applied_to_existing_vault = True

        # ── Save ──────────────────────────────────────────────────────────────
        # Preserve existing azure_api_version so re-running setup doesn't reset it.
        existing_cfg = load_global_config()
        if chosen_name == "azure":
            azure_api_ver = (
                existing_cfg.azure_api_version
                if existing_cfg and existing_cfg.azure_api_version
                else "2024-02-15-preview"
            )
        else:
            azure_api_ver = None

        # Keep ollama_url for backward compat when Ollama is selected
        cfg = GlobalConfig(
            vault=vault_path,
            ollama_url=provider_url if chosen_name == "ollama" else None,
            fast_model=fast_model if fast_model else None,
            heavy_model=heavy_model if heavy_model else None,
            provider_name=chosen_name,
            provider_url=provider_url,
            api_key=api_key,
            azure_api_version=azure_api_ver,
            experimental_inline_source_citations=experimental_inline_source_citations,
        )
        save_global_config(cfg)

        # ── Summary panel ─────────────────────────────────────────────────────
        init_target = vault_path or "~/my-wiki"
        summary_lines = [
            "[green]✓[/green]  Setup complete\n",
            f"  Provider:     [bold]{chosen_prov.display_name}[/bold]",
            f"  URL:          {provider_url}",
        ]
        if api_key:
            summary_lines.append("  API key:      ***")
        if fast_model:
            summary_lines.append(f"  Fast model:   [bold]{fast_model}[/bold]")
        if heavy_model:
            summary_lines.append(f"  Heavy model:  [bold]{heavy_model}[/bold]")
        if vault_path:
            summary_lines.append(f"  Vault:        {vault_path}")
        summary_lines.append(
            "  Inline source citations: "
            f"{'on' if experimental_inline_source_citations else 'off'} for new vaults"
        )
        if vault_path:
            if current_toml_exists:
                current_display = (
                    _format_optional_bool(current_vault_setting)
                    if current_vault_setting is not None
                    else "[dim]not set (default: off)[/dim]"
                )
                suffix = " [dim](updated)[/dim]" if applied_to_existing_vault else ""
            else:
                current_display = (
                    f"[dim]not initialized yet; will be "
                    f"{'on' if experimental_inline_source_citations else 'off'} after init[/dim]"
                )
                suffix = ""
            summary_lines.append(f"  Current vault: {current_display}{suffix}")
        summary_lines += [
            "",
            "  Next steps:",
            f"    [bold]synto init {init_target}[/bold]",
            "    [bold]synto run[/bold]  (or: synto ingest --all && synto compile)",
            "",
            "  Feedback:",
            "    [bold]synto support[/bold]",
            (
                "    [dim]synto stores local runtime and cost metrics by default; "
                "bug reports, suggestions, and experience notes are still the "
                "main way this project improves.[/dim]"
            ),
        ]
        console.print()
        console.print(
            Panel("\n".join(summary_lines), border_style="green", expand=False, padding=(0, 2))
        )

    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Setup interrupted.[/yellow]")
        sys.exit(1)


# ── config ───────────────────────────────────────────────────────────────────


@cli.group(name="config")
def config_cmd():
    """Inspect or update vault-local configuration."""


@config_cmd.command(name="inline-source-citations")
@click.argument("action", type=click.Choice(["on", "off", "status"]))
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
def config_inline_source_citations(action: str, vault_str: str | None):
    """Enable, disable, or inspect inline source citations for one vault."""
    vault_path = _resolve_vault_path(vault_str)
    toml_path = vault_path / CONFIG_FILE_NAME
    if not toml_path.exists():
        click.echo(
            f"Error: {toml_path} not found. Run `synto init {vault_path}` first.",
            err=True,
        )
        sys.exit(1)

    if action == "status":
        try:
            setting = _read_inline_source_citations_setting(toml_path, strict=True)
        except InlineSourceCitationsConfigError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        if setting is None:
            status = "not set (default: disabled)"
        else:
            status = "enabled" if setting else "disabled"
        console.print(f"inline_source_citations: {status} in {toml_path}")
        return

    enabled = action == "on"
    _set_inline_source_citations(toml_path, enabled)
    console.print(f"inline_source_citations = {'true' if enabled else 'false'} in {toml_path}")
    if enabled:
        console.print(
            "[dim]Turn off later with `synto config inline-source-citations off --vault "
            f"{vault_path}`.[/dim]"
        )


# ── ingest ────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@click.option("--all", "ingest_all", is_flag=True, help="Ingest all files in raw/")
@click.option("--force", is_flag=True, help="Re-ingest already-processed notes")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@_model_override_options
def ingest(
    vault_str,
    ingest_all,
    force,
    paths,
    fast_model,
    heavy_model,
    provider_name,
    provider_url,
):
    """Analyze raw notes: extract concepts, quality, suggested topics."""
    from .pipeline.ingest import ingest_all as _ingest_all

    overrides = _model_override_kwargs(fast_model, heavy_model, provider_name, provider_url)
    config = _load_config(vault_str, **overrides)
    client, db = _load_deps(config)

    all_raw = bool(ingest_all)
    if all_raw:
        target_paths = [
            p
            for p in config.raw_dir.rglob("*.md")
            if "processed" not in p.parts and not p.name.startswith(".")
        ]
    elif paths:
        target_paths = [Path(p).resolve() for p in paths]
    else:
        click.echo("Specify --all or provide file paths.", err=True)
        sys.exit(1)

    if not target_paths:
        console.print("[yellow]No notes found in raw/[/yellow]")
        return

    if all_raw:
        ingested = failed = skipped = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Ingesting...", total=len(target_paths))

            def _update_ingest_progress(done: int, total: int, current_note_path: str) -> None:
                progress.update(
                    task,
                    total=total,
                    completed=done,
                    description=f"[dim]{Path(current_note_path).name}[/dim]",
                )

            results = _ingest_all(
                config=config,
                client=client,
                db=db,
                force=force,
                on_progress=_update_ingest_progress,
            )

        for path, result in results:
            if result is None:
                rel = str(path.relative_to(config.vault))
                rec = db.get_raw(rel)
                if rec and rec.status == "failed":
                    failed += 1
                else:
                    skipped += 1
            else:
                ingested += 1
    else:
        skipped = ingested = failed = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Ingesting...", total=len(target_paths))

            for path in target_paths:
                progress.update(task, description=f"[dim]{path.name}[/dim]")
                from .pipeline.ingest import ingest_note as _ingest_note

                result = _ingest_note(
                    path=path,
                    config=config,
                    client=client,
                    db=db,
                    force=force,
                )
                if result is None:
                    # Distinguish skip vs failure by checking DB status
                    rel = str(path.relative_to(config.vault))
                    rec = db.get_raw(rel)
                    if rec and rec.status == "failed":
                        failed += 1
                    else:
                        skipped += 1
                else:
                    ingested += 1
                progress.advance(task)

    console.print(
        f"[green]Done.[/green] Ingested: {ingested}  Skipped: {skipped}  Failed: {failed}"
    )

    # Update index and log
    from .indexer import append_log, generate_index

    generate_index(config, db)
    if ingested:
        append_log(config, f"ingest | {ingested} notes ingested")

    if ingested and config.pipeline.auto_commit:
        from .git_ops import git_commit

        outcome = git_commit(
            config.vault,
            f"ingest: {ingested} notes",
            paths=["raw/", "wiki/sources/", "wiki/index.md", "wiki/log.md", "vault-schema.md"],
        )
        if outcome == "committed":
            console.print("[dim]Git commit created.[/dim]")
        elif outcome == "failed":
            console.print("[yellow]⚠ Git commit failed — run 'git status' in your vault.[/yellow]")
        elif outcome == "blocked":
            console.print(
                "[yellow]⚠ Auto-commit skipped — you have staged changes. "
                "Commit or stash them first.[/yellow]"
            )


# ── compile ───────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@click.option("--dry-run", is_flag=True, help="Show plan, write nothing")
@click.option("--auto-approve", is_flag=True, help="Publish immediately (skip draft review)")
@click.option("--force", is_flag=True, help="Recompile even manually-edited articles")
@click.option("--legacy", is_flag=True, help="Use legacy LLM-planning compile (CompilePlan)")
@click.option(
    "--concept",
    "concepts",
    multiple=True,
    help="Compile specific concept(s), even if not currently pending",
)
@click.option(
    "--retry-failed",
    "retry_failed",
    is_flag=True,
    help="Re-ingest failed raw notes and retry failed concept compiles",
)
@_model_override_options
def compile(
    vault_str,
    dry_run,
    auto_approve,
    force,
    legacy,
    concepts,
    retry_failed,
    fast_model,
    heavy_model,
    provider_name,
    provider_url,
):
    """Synthesize ingested notes into wiki article drafts."""
    from .git_ops import git_commit
    from .pipeline.compile import approve_drafts, compile_concepts, compile_notes

    overrides = _model_override_kwargs(fast_model, heavy_model, provider_name, provider_url)
    config = _load_config(vault_str, **overrides)
    client, db = _load_deps(config)

    explicit_concepts: list[str] | None = None
    if concepts:
        known_concepts = {name.casefold(): name for name in db.list_all_concept_names()}
        known_stubs = {name.casefold(): name for name in db.get_stubs()}
        resolved = []
        unresolved = []
        seen = set()
        for concept in concepts:
            canonical = db.resolve_alias(concept) or concept
            canonical_lookup = known_concepts.get(canonical.casefold()) or known_stubs.get(
                canonical.casefold()
            )
            if canonical_lookup is None:
                unresolved.append(concept)
                continue
            if canonical_lookup.casefold() not in seen:
                seen.add(canonical_lookup.casefold())
                resolved.append(canonical_lookup)
        for concept in unresolved:
            console.print(f"[yellow]Unknown concept, skipping:[/yellow] {concept}")
        if not resolved:
            console.print("[red]No valid concepts to compile.[/red]")
            sys.exit(1)
        explicit_concepts = resolved

    # Re-ingest previously failed notes before compiling
    if retry_failed:
        failed_recs = db.list_raw(status="failed")
        if not failed_recs:
            console.print("[dim]No failed notes to retry.[/dim]")
        else:
            console.print(f"[yellow]Retrying {len(failed_recs)} failed note(s)...[/yellow]")
            from .pipeline.ingest import ingest_note as _ingest_note

            retried = 0
            for rec in failed_recs:
                p = config.vault / rec.path
                if not p.exists():
                    console.print(f"  [red]Not found, skipping:[/red] {rec.path}")
                    continue
                db.mark_raw_status(rec.path, "new")
                result = _ingest_note(path=p, config=config, client=client, db=db, force=True)
                if result is not None:
                    retried += 1
            console.print(f"[green]Re-ingested {retried}/{len(failed_recs)} note(s).[/green]")

        failed_concepts = db.list_failed_concepts()
        if failed_concepts:
            console.print(f"[yellow]Retrying {len(failed_concepts)} failed concept(s)...[/yellow]")
            if explicit_concepts is None:
                explicit_concepts = failed_concepts
            else:
                for concept in failed_concepts:
                    if concept.casefold() not in {name.casefold() for name in explicit_concepts}:
                        explicit_concepts.append(concept)
        else:
            console.print("[dim]No failed concepts to retry.[/dim]")

    if dry_run:
        console.print("[dim]Dry run — no files will be written.[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        if legacy:
            task = progress.add_task("Planning compilation (legacy)...", total=None)
            draft_paths, failed = compile_notes(
                config=config,
                client=client,
                db=db,
                dry_run=dry_run,
            )
        else:
            task = progress.add_task("Compiling concepts...", total=1)

            def _on_progress(idx: int, total: int, name: str) -> None:
                progress.update(
                    task,
                    total=total,
                    completed=idx - 1,
                    description=f"[dim]{name}[/dim]",
                )

            draft_paths, failed, _ = compile_concepts(
                config=config,
                client=client,
                db=db,
                force=force,
                dry_run=dry_run,
                on_progress=_on_progress,
                concepts=explicit_concepts,
            )
            progress.update(task, completed=progress.tasks[task].total or 1)

    if dry_run:
        return

    if draft_paths:
        console.print(f"\n[green]{len(draft_paths)} draft(s) written:[/green]")
        for p in draft_paths:
            console.print(f"  {p.relative_to(config.vault)}")

    if failed:
        console.print(f"[yellow]{len(failed)} article(s) failed:[/yellow] {', '.join(failed)}")

    # Update index and log
    from .indexer import append_log, generate_index

    if draft_paths:
        generate_index(config, db)
        append_log(config, f"compile | {len(draft_paths)} drafts written")

    if auto_approve and draft_paths:
        published = approve_drafts(config, db, draft_paths)
        generate_index(config, db)
        append_log(config, f"approve | {len(published)} articles published")
        if config.pipeline.auto_commit:
            outcome = git_commit(
                config.vault, f"compile: {len(published)} articles", paths=["wiki/", ".synto/"]
            )
            if outcome == "failed":
                console.print("[yellow]⚠ Git commit failed — run 'git status' in vault.[/yellow]")
            elif outcome == "blocked":
                console.print(
                    "[yellow]⚠ Auto-commit skipped — you have staged changes. "
                    "Commit or stash them first.[/yellow]"
                )
        console.print(f"[green]Published {len(published)} articles.[/green]")
    elif draft_paths:
        console.print("\nReview drafts in [bold]wiki/.drafts/[/bold], then run:")
        console.print("  [bold]synto review[/bold]")
        console.print("  [bold]synto approve --all[/bold]")


# ── approve ───────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@click.option("--all", "approve_all", is_flag=True)
@click.option(
    "--min-confidence",
    type=float,
    default=0.0,
    help="Skip drafts below this confidence score (0–1).",
)
@click.argument("files", nargs=-1, type=click.Path())
def approve(vault_str, approve_all, min_confidence, files):
    """Publish draft(s) from wiki/.drafts/ to wiki/."""
    from .git_ops import git_commit
    from .pipeline.compile import approve_drafts

    config = _load_config(vault_str)
    db = _load_db(config)

    if approve_all:
        paths = None  # approve_drafts handles all
    elif files:
        paths = [_resolve_draft_arg(config, f) for f in files]
    else:
        click.echo("Specify --all or file paths.", err=True)
        sys.exit(1)

    all_paths = list((config.drafts_dir).glob("*.md")) if paths is None else paths
    affected = approve_drafts(config, db, paths, min_confidence=min_confidence)
    if not affected:
        console.print("[yellow]No drafts to publish.[/yellow]")
        return

    held_back = len(all_paths) - len(affected)
    if held_back > 0:
        console.print(
            f"[yellow]Held back {held_back} draft(s) below confidence "
            f"{min_confidence:.2f}.[/yellow]"
        )

    console.print(f"[green]Published {len(affected)} article(s).[/green]")

    # Update index and log
    from .indexer import append_log, generate_index

    generate_index(config, db)
    append_log(config, f"approve | {len(affected)} articles published")

    if config.pipeline.auto_commit:
        outcome = git_commit(
            config.vault,
            f"approve: {len(affected)} articles published",
            paths=["wiki/", ".synto/"],
        )
        if outcome == "committed":
            console.print("[dim]Git commit created.[/dim]")
        elif outcome == "failed":
            console.print("[yellow]⚠ Git commit failed — run 'git status' in your vault.[/yellow]")
        elif outcome == "blocked":
            console.print(
                "[yellow]⚠ Auto-commit skipped — you have staged changes. "
                "Commit or stash them first.[/yellow]"
            )


@cli.command()
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@click.option("--all", "verify_all", is_flag=True)
@click.option(
    "--min-confidence",
    type=float,
    default=0.0,
    help="Skip drafts below this confidence score (0–1).",
)
@click.argument("files", nargs=-1, type=click.Path())
def verify(vault_str, verify_all, min_confidence, files):
    """Mark draft(s) verified in place without publishing."""
    from .git_ops import git_commit
    from .pipeline.compile import verify_drafts

    config = _load_config(vault_str)
    db = _load_db(config)

    if verify_all:
        paths = None
    elif files:
        paths = [_resolve_draft_arg(config, f) for f in files]
    else:
        click.echo("Specify --all or file paths.", err=True)
        sys.exit(1)

    all_paths = list(config.drafts_dir.glob("*.md")) if paths is None else paths
    affected = verify_drafts(config, db, paths, min_confidence=min_confidence)
    if not affected:
        console.print("[yellow]No drafts to verify.[/yellow]")
        return

    held_back = len(all_paths) - len(affected)
    if held_back > 0:
        console.print(
            f"[yellow]Held back {held_back} draft(s) below confidence "
            f"{min_confidence:.2f}.[/yellow]"
        )
    console.print(f"[green]Verified {len(affected)} article(s).[/green]")
    console.print("Run [bold]synto approve --all[/bold] to publish.")

    from .indexer import append_log, generate_index

    generate_index(config, db)
    append_log(config, f"verify | {len(affected)} articles verified")

    if config.pipeline.auto_commit:
        outcome = git_commit(
            config.vault,
            f"verify: {len(affected)} articles verified",
            paths=["wiki/", ".synto/"],
        )
        if outcome == "committed":
            console.print("[dim]Git commit created.[/dim]")
        elif outcome == "failed":
            console.print("[yellow]⚠ Git commit failed — run 'git status' in your vault.[/yellow]")
        elif outcome == "blocked":
            console.print(
                "[yellow]⚠ Auto-commit skipped — you have staged changes. "
                "Commit or stash them first.[/yellow]"
            )


# ── reject ────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@click.option("--all", "reject_all", is_flag=True, help="Reject all drafts in wiki/.drafts/")
@click.option("--feedback", default="", help="Reason for rejection")
@click.argument("files", nargs=-1, type=click.Path())
def reject(vault_str, reject_all, feedback, files):
    """Discard draft article(s) and store rejection feedback for future recompiles."""
    from .pipeline.compile import reject_draft

    config = _load_config(vault_str)
    db = _load_db(config)

    if reject_all:
        draft_paths = list(config.drafts_dir.rglob("*.md")) if config.drafts_dir.exists() else []
        if not draft_paths:
            console.print("[yellow]No drafts to reject.[/yellow]")
            return
        if not feedback:
            feedback = click.prompt("Reason for rejecting all drafts?", default="")
    elif files:
        draft_paths = [_resolve_draft_arg(config, f) for f in files]
        for p in draft_paths:
            if not p.exists():
                click.echo(f"File not found: {p}", err=True)
                sys.exit(1)
        if not feedback:
            feedback = click.prompt("Reason for rejection?", default="")
    else:
        click.echo("Specify --all or provide file paths.", err=True)
        sys.exit(1)

    from .vault import parse_note as _parse

    for draft_path in draft_paths:
        title = draft_path.stem
        try:
            meta, _ = _parse(draft_path)
            title = meta.get("title", draft_path.stem)
        except Exception:
            pass

        reject_draft(draft_path, config, db, feedback=feedback)
        console.print(f"[yellow]Draft rejected:[/yellow] {draft_path.name}")

        if feedback:
            count = db.rejection_count(title)
            if db.is_concept_blocked(title):
                console.print(
                    f"[red]⚠ '{title}' blocked after {count} rejections. "
                    f'Use [bold]synto unblock "{title}"[/bold] to re-enable.[/red]'
                )
            else:
                console.print(
                    f"[dim]Feedback saved. Next compile of '{title}' will address it. "
                    f"({count}/{db._REJECTION_CAP} rejections)[/dim]"
                )

    if len(draft_paths) > 1:
        console.print(f"[green]Rejected {len(draft_paths)} draft(s).[/green]")


# ── status ────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@click.option("--failed", "show_failed", is_flag=True, help="List failed notes with error messages")
def status(vault_str, show_failed):
    """Show vault health, pending drafts, and pipeline stats."""
    from .models import WikiArticleRecord

    config = _load_config(vault_str)
    db = _load_db(config)

    stats = db.stats(config.vault)
    raw = stats.get("raw", {})

    table = Table(title="Vault Status", show_header=True)
    table.add_column("Category")
    table.add_column("Count", justify="right")

    table.add_row("Raw: new", str(raw.get("new", 0)))
    table.add_row("Raw: ingested", str(raw.get("ingested", 0)))
    table.add_row("Raw: compiled", str(raw.get("compiled", 0)))
    table.add_row("Raw: failed", str(raw.get("failed", 0)))
    table.add_row("Drafts pending", str(stats["drafts"]))
    table.add_row("Verified pending", str(stats.get("verified", 0)))
    table.add_row("Published articles", str(stats["published"]))

    console.print(table)

    # List pending drafts
    drafts = db.list_articles(drafts_only=True)
    known_draft_paths = {article.path for article in drafts}
    verified = [article for article in db.list_articles() if article.is_verified]
    known_verified_paths = {article.path for article in verified}
    if config.drafts_dir.exists():
        from .vault import list_draft_articles

        for title, path, sources in list_draft_articles(config.drafts_dir):
            rel_path = str(path.relative_to(config.vault))
            if rel_path in known_draft_paths or rel_path in known_verified_paths:
                continue
            drafts.append(
                WikiArticleRecord(
                    path=rel_path,
                    title=title,
                    sources=sources,
                    content_hash="",
                    status="draft",
                )
            )
    if drafts:
        console.print(f"\n[bold]{len(drafts)} draft(s) pending review:[/bold]")
        for article in drafts:
            sources_str = ", ".join(Path(s).name for s in article.sources)
            console.print(f"  [dim]{article.path}[/dim]  (from: {sources_str})")
        console.print("\nRun [bold]synto verify --all[/bold] to mark reviewed.")
    if verified:
        console.print(f"\n[bold]{len(verified)} verified article(s) pending publish:[/bold]")
        for article in verified:
            sources_str = ", ".join(Path(s).name for s in article.sources)
            console.print(f"  [dim]{article.path}[/dim]  (from: {sources_str})")
        console.print("\nRun [bold]synto approve --all[/bold] to publish.")

    # List failed notes if requested (or if there are any)
    if show_failed or raw.get("failed", 0):
        failed_recs = db.list_raw(status="failed")
        if failed_recs:
            console.print(f"\n[red][bold]{len(failed_recs)} failed note(s):[/bold][/red]")
            for rec in failed_recs:
                err = rec.error or "unknown error"
                console.print(f"  [dim]{rec.path}[/dim]")
                console.print(f"    [red]{err}[/red]")
            console.print("\nRun [bold]synto compile --retry-failed[/bold] to re-attempt.")

    # Show blocked concepts
    blocked = db.list_blocked_concepts()
    if blocked:
        console.print(f"\n[red][bold]{len(blocked)} blocked concept(s):[/bold][/red]")
        for concept in blocked:
            count = db.rejection_count(concept)
            console.print(f"  {concept} [dim]({count} rejections)[/dim]")
        console.print('[dim]Run [bold]synto unblock "Concept"[/bold] to re-enable.[/dim]')

    # Show pipeline lock status
    from .pipeline.lock import has_invalid_lock_file, lock_holder_pid

    pid = lock_holder_pid(config.vault)
    if pid is not None:
        console.print(f"\n[yellow]⚠ Pipeline lock held by PID {pid}[/yellow]")
    elif has_invalid_lock_file(config.vault):
        console.print("\n[dim]Lock file present but invalid; no live process holds it[/dim]")


@cli.command(name="eval")
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@click.option(
    "--queries",
    "queries_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to TOML query fixtures. Defaults to tests/eval/queries_default.toml.",
)
@click.option("--live", is_flag=True, help="Reserved for later live-agent eval. Not in Phase 1A.")
@click.option("--json", "json_out", is_flag=True, help="Emit machine-readable JSON")
def eval_cmd(vault_str, queries_path, live, json_out):
    """Run the offline structural eval harness."""
    from .pipeline.eval import render_json, render_text, run_offline

    config = _load_config(vault_str)
    if live:
        click.echo(
            "synto eval --live is not implemented in Phase 1A. Use offline mode without --live.",
            err=True,
        )
        raise SystemExit(2)

    result = run_offline(config, queries_path)
    click.echo(render_json(result) if json_out else render_text(result))


# ── undo ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@click.option("--steps", default=1, show_default=True)
def undo(vault_str, steps):
    """Revert last N synto/legacy auto-commits (uses git revert — safe)."""
    from .git_ops import git_undo

    config = _load_config(vault_str)
    try:
        reverted = git_undo(config.vault, steps=steps)
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e
    if reverted:
        console.print(f"[green]Reverted {len(reverted)} commit(s):[/green]")
        for msg in reverted:
            console.print(f"  {msg}")
    else:
        console.print("[yellow]No synto or legacy auto-commits found to revert.[/yellow]")


# ── clean ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
def clean(vault_str, yes):
    """Clear state DB, wiki/, and drafts — raw/ notes are kept.

    Use this to start fresh without deleting your source material.
    """
    import shutil

    config = _load_config(vault_str)

    targets = [
        ("state DB", config.state_db_path),
        ("wiki/", config.wiki_dir),
    ]

    console.print("[bold yellow]This will delete:[/bold yellow]")
    for label, path in targets:
        if path.exists():
            console.print(f"  {label}: {path}")
    console.print("Raw notes in [bold]raw/[/bold] are NOT touched.")

    if not yes:
        click.confirm("Proceed?", abort=True)

    if config.state_db_path.exists():
        from .indexer import generate_index_json
        from .state import StateDB

        db = StateDB.open_readonly(config.state_db_path)
        try:
            seed_path = generate_index_json(config, db)
        finally:
            db.close()
        console.print(f"  [dim]Preserved rebuild seed: {seed_path}[/dim]")

    for label, path in targets:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            console.print(f"  [dim]Deleted {label}[/dim]")

    # Re-create empty wiki/ structure
    config.wiki_dir.mkdir(parents=True, exist_ok=True)
    config.drafts_dir.mkdir(parents=True, exist_ok=True)
    config.sources_dir.mkdir(parents=True, exist_ok=True)

    console.print("[green]Clean complete.[/green] Run [bold]synto ingest --all[/bold] to restart.")


# ── doctor ───────────────────────────────────────────────────────────────────


@cli.command()
def support():
    """Show bug-report, suggestion, and feedback links."""

    console.print("[bold]synto support[/bold]\n")
    console.print("Synto stores local runtime and cost metrics in the vault state database.")
    console.print("Default: aggregate rollups only. Detailed per-call rows are opt-in.")
    console.print("If something was confusing, useful, annoying, or missing, please tell us.\n")
    console.print("Bug reports:")
    console.print(f"  {PROJECT_ISSUES_URL}\n")
    console.print("Suggestions and experience reports:")
    console.print(f"  {PROJECT_DISCUSSIONS_URL}\n")
    console.print("Source code:")
    console.print(f"  {PROJECT_REPO_URL}\n")
    console.print("When reporting a bug, include:")
    console.print("  - `synto --version`")
    console.print("  - your OS")
    console.print("  - how you installed synto")
    console.print("  - the command you ran")
    console.print("  - the error message or unexpected behavior")


def _is_hash8(value: str | None) -> bool:
    # An 8-char SHA256 prefix (the default-audit form). Distinguishes hashed
    # labels from raw query text so the report can flag degraded mode.
    return bool(value) and len(value) == 8 and all(c in "0123456789abcdef" for c in value)


def _render_mcp_backlog(db, since: str) -> None:
    """Render the MCP demand-vs-coverage backlog section of `synto doctor`.

    Informational only — never affects exit status. Reads existing audit rows
    in `metric_events`; degrades gracefully when audit_detailed is off (labels
    show as <hash:8>).
    """
    from datetime import UTC, datetime, timedelta

    # since_ts: ISO-8601 lower bound; "" means all-time (matches every row lexically).
    if since == "all":
        since_ts = ""
        window_label = "all time"
        no_activity_label = "(no MCP activity recorded)"
    else:
        days = {"1d": 1, "7d": 7, "30d": 30}[since]
        since_ts = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        window_label = f"last {since}"
        no_activity_label = f"(no MCP activity in the {window_label})"

    console.print(f"\n[bold]MCP demand-vs-coverage ({window_label})[/bold]\n")

    if db.count_mcp_audit_rows_since(since_ts) == 0:
        console.print(f"  {no_activity_label}")
        return

    degraded = False

    def label_str(value: str | None) -> str:
        nonlocal degraded
        if value is None:
            return "<none>"
        if _is_hash8(value):
            degraded = True
            return f"<{value}>"
        return f'"{value}"'

    # 1) Zero-result queries
    console.print("  Zero-result queries (top 20 by frequency)")
    zero = db.zero_result_query_counts(since_ts, top_n=20)
    if zero:
        console.print(f"    [dim]{'Tool':<26}{'Query':<26}{'Hits':>5}[/dim]")
        for tool, label, count in zero:
            console.print(f"    {tool:<26}{label_str(label):<26}{count:>5}")
    else:
        console.print("    [dim](none)[/dim]")

    # 2) Single-source concepts in active demand
    console.print("\n  Single-source concepts in active demand")
    single = db.single_source_concepts_in_demand(since_ts)
    single_names = {name for name, _occ, _hits in single}
    if single:
        header = f"{'Concept':<26}{'Occurrences':>12}{'Resolved-label hits':>22}"
        console.print(f"    [dim]{header}[/dim]")
        for name, occ, hits in single:
            console.print(f"    {name:<26}{occ:>12}{hits:>22}")
    else:
        console.print("    [dim](none)[/dim]")

    # 3) Repeat weak queries against single-source concepts
    console.print("\n  Repeat weak queries (≥2 hits against single-source concepts)")
    repeats = db.repeat_weak_queries(since_ts, single_names, min_hits=2)
    if repeats:
        console.print(f"    [dim]{'Query':<26}{'Hits':>5}   {'Target concept'}[/dim]")
        for label, hits, target in repeats:
            console.print(f"    {label_str(label):<26}{hits:>5}   {target}")
    else:
        console.print("    [dim](none)[/dim]")

    # 4) Tool-mix per session
    console.print("\n  Tool-mix per session (≥5 calls, 30-min idle gap)")
    sessions = db.tool_mix_sessions(since_ts)
    if sessions:
        total = sum(s[1] for s in sessions)
        verbatim = sum(s[2] for s in sessions)
        answers = sum(s[3] for s in sessions)
        other = sum(s[4] for s in sessions)
        if total > 0:
            vpct, apct, opct = (
                round(100 * verbatim / total),
                round(100 * answers / total),
                round(100 * other / total),
            )
        else:
            vpct = apct = opct = 0
        console.print(
            f"    Sessions: {len(sessions)}   Verbatim: {vpct}%   "
            f"answer_question: {apct}%   Other: {opct}%"
        )
    else:
        console.print("    [dim](no sessions with ≥5 calls)[/dim]")

    if degraded:
        console.print(
            "\n  [dim](audit_detailed=false → some labels shown as <hash:8>;\n"
            "        set [mcp] audit_detailed=true in synto.toml to see raw queries)[/dim]"
        )


@cli.command()
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@click.option(
    "--backlog", is_flag=True, default=False, help="Show MCP demand-vs-coverage backlog (opt-in)."
)
@click.option(
    "--since",
    type=click.Choice(["1d", "7d", "30d", "all"]),
    default="7d",
    help="Lookback window for --backlog.",
)
def doctor(vault_str, backlog, since):
    """Check LLM provider connection, model availability, and vault health."""
    from .client_factory import LLMError, build_client

    config = _load_config(vault_str)
    db = _load_db(config)
    ok = True
    prov = config.effective_provider

    console.print("[bold]synto doctor[/bold]\n")

    # ── Vault structure ──────────────────────────────────────────────────────
    console.print("[bold]Vault structure[/bold]")

    legacy_toml = config.vault / LEGACY_CONFIG_FILE_NAME
    legacy_app = config.vault / LEGACY_APP_DIR_NAME
    if (legacy_toml.exists() or legacy_app.exists()) and not (
        config.vault / CONFIG_FILE_NAME
    ).exists():
        console.print(
            f"  [yellow]![/yellow] Legacy vault layout detected "
            f"({'wiki.toml' if legacy_toml.exists() else ''}"
            f"{', ' if legacy_toml.exists() and legacy_app.exists() else ''}"
            f"{'.olw/' if legacy_app.exists() else ''}).\n"
            f"    Run: [bold]synto migrate-olw --vault {config.vault}[/bold]"
        )

    toml_path = config.vault / CONFIG_FILE_NAME
    if not toml_path.exists():
        console.print(
            f"  [red]✗[/red] {CONFIG_FILE_NAME} missing — vault not initialised.\n"
            f"    Run: [bold]synto init {config.vault}[/bold]"
        )
        console.print("\n[red][bold]Vault not initialised. Remaining checks skipped.[/bold][/red]")
        sys.exit(1)

    checks = {
        "raw/": config.raw_dir,
        "wiki/": config.wiki_dir,
        "wiki/.drafts/": config.drafts_dir,
        "wiki/sources/": config.sources_dir,
        ".synto/": config.app_dir,
        CONFIG_FILE_NAME: toml_path,
    }
    for name, path in checks.items():
        if path.exists():
            console.print(f"  [green]✓[/green] {name}")
        else:
            console.print(f"  [yellow]![/yellow] {name} missing (run [bold]synto init[/bold])")

    # ── Provider connection ───────────────────────────────────────────────────
    console.print(f"\n[bold]{prov.name}[/bold]")
    client = build_client(config)
    try:
        client.require_healthy()
        console.print(f"  [green]✓[/green] Reachable at {prov.url}")
    except LLMError as e:
        console.print(f"  [red]✗[/red] {e}")
        ok = False

    # ── Model availability ────────────────────────────────────────────────────
    console.print("\n[bold]Models[/bold]")
    try:
        available_models = client.list_models()
    except Exception:
        available_models = []

    for label, model_name in [("fast", config.models.fast), ("heavy", config.models.heavy)]:
        if any(model_name in a for a in available_models):
            console.print(f"  [green]✓[/green] {label}: {model_name}")
        else:
            pull_hint = (
                f"run: [bold]ollama pull {model_name}[/bold]"
                if prov.name == "ollama"
                else "check provider model list"
            )
            console.print(f"  [yellow]![/yellow] {label}: {model_name} not found — {pull_hint}")
            ok = False

    # ── Vault stats ───────────────────────────────────────────────────────────
    console.print("\n[bold]Vault stats[/bold]")
    stats = db.stats(config.vault)
    raw = stats.get("raw", {})
    console.print(f"  Raw notes:         {sum(raw.values())}")
    console.print(f"  Ingested:          {raw.get('ingested', 0) + raw.get('compiled', 0)}")
    console.print(f"  Drafts pending:    {stats['drafts']}")
    console.print(f"  Published:         {stats['published']}")

    # ── Verbatim source index ─────────────────────────────────────────────────
    console.print("\n[bold]Verbatim source index[/bold]")
    try:
        fts_exists, fts_count, seg_count = db.source_segments_fts_status()
        if not fts_exists and not db.fts5_available():
            # v16 migration skipped the index because this SQLite build lacks FTS5.
            # search_source_segments is disabled; the other verbatim tools still work.
            console.print(
                "  [yellow]![/yellow] FTS5 not available in this SQLite build —"
                " full-text search disabled (other verbatim tools work)"
            )
        elif not fts_exists:
            console.print(
                "  [yellow]![/yellow] source_segments_fts not present"
                " (vault below v16 — run any synto command to migrate)"
            )
        elif fts_count == seg_count:
            console.print(f"  [green]✓[/green] {seg_count} segments indexed (FTS5 in sync)")
        else:
            console.print(
                f"  [red]✗[/red] FTS index drift: {fts_count} indexed vs {seg_count} segments"
            )
            ok = False
    except Exception as exc:  # pragma: no cover — defensive
        console.print(f"  [yellow]![/yellow] could not read FTS index status: {exc}")

    # Concept→segment links back get_source_passages. They are populated at ingest;
    # vaults ingested before this feature have segments but zero links until re-ingested.
    try:
        seg_total = db.count_source_segments()
        link_count = db.concept_occurrence_count()
        if seg_total > 0 and link_count == 0:
            console.print(
                "  [yellow]![/yellow] 0 concept→segment links — get_source_passages will be"
                " empty. Run [bold]synto ingest --force[/bold] to backfill (analysis only;"
                " published articles are untouched)."
            )
        elif link_count > 0:
            console.print(
                f"  [green]✓[/green] {link_count} concept→segment links (get_source_passages ready)"
            )
    except Exception as exc:  # pragma: no cover — defensive
        console.print(f"  [yellow]![/yellow] could not read concept-link status: {exc}")

    # ── MCP source-access posture ─────────────────────────────────────────────
    # Surfaces the effective privacy gate so the legacy-vault grandfather (which
    # exposes all raw source text over MCP) is never silent. See serve.py.
    sa = config.mcp.source_access
    if sa.mode == "permissive_only" and not db.any_source_license_declared():
        console.print(
            '  [yellow]![/yellow] source-access mode: "permissive_only" configured,'
            ' but no source declares a license → effective "all".'
        )
        console.print(
            "      [dim]All raw source text is readable by MCP clients. Declare licenses on"
            # Escape the literal [mcp.source_access] so Rich doesn't parse it as a markup tag.
            r" sources, or set \[mcp.source_access] mode explicitly in synto.toml.[/dim]"
        )
    else:
        console.print(f'  source-access mode: "{sa.mode}"')

    # ── MCP demand-vs-coverage backlog (opt-in via --backlog) ────────────────
    if backlog:
        _render_mcp_backlog(db, since)

    draft_graph_filter = [
        "-path:raw",
        "-path:wiki/sources",
        "-path:_resources",
        "-file:Welcome",
    ]
    published_graph_filter = [
        "-path:raw",
        "-path:wiki/sources",
        "-path:wiki/.drafts",
        "-path:_resources",
        "-file:Welcome",
    ]
    graph_notes: list[str] = []
    if (config.vault / "Welcome.md").exists():
        graph_notes.append("Welcome.md is present and can create starter graph noise")
    if config.raw_dir.exists() and any(config.raw_dir.rglob("*.md")):
        graph_notes.append("raw/ notes are visible unless filtered")
    if config.sources_dir.exists() and any(config.sources_dir.glob("*.md")):
        graph_notes.append("wiki/sources/ pages can dominate graph when citations are enabled")
    if config.drafts_dir.exists() and any(config.drafts_dir.rglob("*.md")):
        graph_notes.append("wiki/.drafts/ pages are review artifacts, not published wiki")

    console.print("\n[bold]Graph view[/bold]")
    if graph_notes:
        for note in graph_notes:
            console.print(f"  [yellow]![/yellow] {note}")
    else:
        console.print("  [green]✓[/green] No obvious graph-noise layers detected")
    console.print("  Draft review graph filter:")
    console.print(f"  [dim]{' '.join(draft_graph_filter)}[/dim]")
    console.print("  Published-only graph filter:")
    console.print(f"  [dim]{' '.join(published_graph_filter)}[/dim]")

    console.print()
    if ok:
        console.print("[green][bold]All checks passed.[/bold][/green]")
    else:
        console.print("[yellow][bold]Some checks need attention (see above).[/bold][/yellow]")


# ── query ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@click.option("--save", is_flag=True, help="Save answer to wiki/queries/")
@click.option("--synthesize", is_flag=True, help="Save answer to wiki/synthesis/")
@click.argument("question")
def query(vault_str, question, save, synthesize):
    """Answer a question using your wiki as context (no embeddings needed)."""
    from rich.markdown import Markdown

    from .pipeline.query import SynthesisSaveError, find_existing_synthesis, run_query

    config = _load_config(vault_str)
    client, db = _load_deps(config)
    duplicate_strategy = "keep_existing"
    if (
        synthesize
        and sys.stdin.isatty()
        and sys.stdout.isatty()
        and find_existing_synthesis(db, question) is not None
    ):
        raw_choice = (
            click.prompt(
                "Duplicate synthesis exists - keep / suffix / update?",
                type=click.Choice(["keep", "suffix", "update"], case_sensitive=False),
                default="keep",
                show_choices=False,
            )
            .strip()
            .lower()
        )
        duplicate_strategy = {
            "keep": "keep_existing",
            "suffix": "save_with_suffix",
            "update": "update_in_place",
        }[raw_choice]

    with console.status("[bold]Searching wiki index…"):
        try:
            result = run_query(
                config,
                client,
                db,
                question,
                save=save,
                synthesize=synthesize,
                duplicate_strategy=duplicate_strategy,
            )
        except SynthesisSaveError as exc:
            if synthesize:
                click.echo(str(exc), err=True)
                raise SystemExit(1) from exc
            raise

    if result.selected_pages:
        console.print(f"[dim]Sources: {', '.join(result.selected_pages)}[/dim]")
    console.print()
    console.print(Markdown(result.answer))
    if result.query_save is not None:
        console.print("\n[green]Answer saved to wiki/queries/[/green]")
    if result.synthesis is not None:
        if result.synthesis.resolution == "kept_existing":
            console.print(f"\n[yellow]Existing synthesis kept at {result.synthesis.path}[/yellow]")
        else:
            console.print(f"\n[green]Synthesis saved to {result.synthesis.path}[/green]")


# ── watch ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@click.option(
    "--auto-approve", is_flag=True, help="Publish drafts immediately without manual review"
)
def watch(vault_str, auto_approve):
    """Watch raw/ for new/changed notes → auto-ingest + compile."""
    from .pipeline.lock import pipeline_lock
    from .pipeline.orchestrator import PipelineOrchestrator
    from .watcher import watch as _watch

    config = _load_config(vault_str)
    client, db = _load_deps(config)
    orchestrator = PipelineOrchestrator(config, client, db)

    debounce = config.pipeline.watch_debounce
    console.print(f"[bold]Watching[/bold] {config.raw_dir}  (debounce={debounce:.0f}s)")
    console.print("[dim]Ctrl+C to stop.[/dim]\n")

    def _on_event(paths: list[str]) -> None:
        md_paths = [p for p in paths if p.endswith(".md")]
        if not md_paths:
            return

        console.rule(f"[dim]{len(md_paths)} file(s) changed[/dim]")

        with pipeline_lock(config.vault) as acquired:
            if not acquired:
                console.print("[yellow]⚠ compile skipped — pipeline already running[/yellow]")
                return
            try:
                report = orchestrator.run(
                    paths=md_paths,
                    auto_approve=auto_approve or config.pipeline.auto_approve,
                    fix=config.pipeline.auto_maintain,
                )
            except Exception as exc:
                console.print(f"[red]Pipeline error:[/red] {exc}")
                return

        if report.ingested:
            console.print(f"  [green]✓[/green] ingested {report.ingested} note(s)")
        if report.compiled:
            console.print(f"  [green]✓[/green] {report.compiled} draft(s) compiled")
        if report.failed:
            failed_str = ", ".join(report.failed_names)
            console.print(
                f"  [yellow]![/yellow] {len(report.failed)} concept(s) failed: {failed_str}"
            )
        if report.published:
            console.print(f"  [green]✓[/green] {report.published} article(s) published")
        elif report.compiled:
            console.print("  [dim]Run [bold]synto approve --all[/bold] to publish drafts.[/dim]")
        if report.stubs_created:
            console.print(f"  [dim]Created {report.stubs_created} stub(s) for broken links[/dim]")

    _watch(config=config, client=client, db=db, on_event=_on_event)


# ── serve ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@click.option(
    "--transport",
    type=click.Choice(["stdio"]),
    default="stdio",
    show_default=True,
    help="Phase 1A supports only stdio transport.",
)
def serve(vault_str, transport):
    """Run the read-only MCP server.

    Exposes three tools only: `list_articles`, `read_article`, `find_concept`.
    Visibility filtering is controlled by `[mcp]` settings in `synto.toml`.
    Requires the optional MCP dependency: `pip install synto[mcp]`.
    """
    from .serve import run_server

    config = _load_config(vault_str)
    run_server(config.vault, transport=transport)


# ── run ───────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@click.option("--auto-approve", is_flag=True, help="Publish drafts immediately")
@click.option("--fix", is_flag=True, help="Create stubs for broken wikilinks")
@click.option("--max-rounds", default=2, show_default=True, help="Max compile rounds")
@click.option("--dry-run", is_flag=True, help="Report what would happen, make no changes")
@click.option(
    "--min-confidence",
    type=float,
    default=0.0,
    help="With --auto-approve, skip drafts below this confidence score (0–1).",
)
@_model_override_options
def run(
    vault_str,
    auto_approve,
    fix,
    max_rounds,
    dry_run,
    min_confidence,
    fast_model,
    heavy_model,
    provider_name,
    provider_url,
):
    """Run full pipeline: ingest → compile → lint → [approve]."""
    from .pipeline.lock import pipeline_lock
    from .pipeline.orchestrator import PipelineOrchestrator

    overrides = _model_override_kwargs(fast_model, heavy_model, provider_name, provider_url)
    config = _load_config(vault_str, **overrides)
    client, db = _load_deps(config)

    if dry_run:
        console.print("[dim]Dry run — no changes will be made.[/dim]\n")

    with pipeline_lock(config.vault) as acquired:
        if not acquired:
            err_console.print("Pipeline already running — lock held. Check `synto status`.")
            sys.exit(1)
        orchestrator = PipelineOrchestrator(config, client, db)
        report = orchestrator.run(
            auto_approve=auto_approve,
            fix=fix,
            max_rounds=max_rounds,
            dry_run=dry_run,
            min_confidence=min_confidence,
        )

    table = Table(title="Pipeline Report", show_header=True)
    table.add_column("Step")
    table.add_column("Count", justify="right")
    table.add_column("Time", justify="right")

    table.add_row("Ingested", str(report.ingested), f"{report.timings.get('ingest', 0):.1f}s")
    table.add_row(
        "Compiled",
        str(report.compiled),
        f"{report.timings.get('compile_r1', 0) + report.timings.get('compile_r2', 0):.1f}s",
    )
    table.add_row("Published", str(report.published), "")
    if report.held_back > 0:
        table.add_row("Held back", str(report.held_back), "")
    table.add_row("Lint issues", str(report.lint_issues), "")
    table.add_row("Stubs created", str(report.stubs_created), "")
    if report.rounds > 1:
        table.add_row("Compile rounds", str(report.rounds), "")
    console.print(table)

    if report.failed:
        console.print(f"\n[yellow]{len(report.failed)} concept(s) failed:[/yellow]")
        for f in report.failed:
            console.print(f"  [dim]{f.concept}[/dim] ({f.reason.value})")
            if f.error_msg:
                console.print(f"    [dim]{f.error_msg}[/dim]")

    if not dry_run:
        tips: list[str] = []
        if report.compiled > 0 and not auto_approve:
            tips.append(
                f"Review drafts:  [bold]{CLI_NAME} review[/bold]"
                f"  or approve all:  [bold]{CLI_NAME} approve --all[/bold]"
            )
        if report.held_back > 0:
            tips.append(
                f"Low-confidence: [bold]{CLI_NAME} approve --all --min-confidence 0[/bold]"
                " to publish held-back drafts after review"
            )
        if report.published > 0:
            tips.append(f"Export pack:    [bold]{CLI_NAME} pack export --target agents[/bold]")
            tips.append(f'Query wiki:     [bold]{CLI_NAME} query "..."[/bold]')
        if report.lint_issues > 0:
            tips.append(f"Fix issues:     [bold]{CLI_NAME} maintain --fix[/bold]")
        if not tips and report.ingested == 0 and report.compiled == 0:
            tips.append("Add notes to [bold]raw/[/bold] and run again.")
        if tips:
            console.print("\n[dim]Next:[/dim]")
            for tip in tips:
                console.print(f"  {tip}")


# ── review ────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
def review(vault_str):
    """Interactive draft review: approve, reject, edit, or diff drafts."""

    from rich.markup import escape

    from .pipeline.compile import approve_drafts, reject_draft, verify_drafts
    from .pipeline.review import (
        compute_diff,
        compute_rejection_diff,
        list_drafts,
        load_draft_content,
    )

    config = _load_config(vault_str)
    db = _load_db(config)

    while True:
        summaries = list_drafts(config, db)
        if not summaries:
            console.print("[dim]No drafts pending review.[/dim]")
            return

        # Build menu table
        table = Table(title="Drafts Pending Review", show_header=True, show_lines=False)
        table.add_column("#", justify="right", style="dim")
        table.add_column("Title")
        table.add_column("Status")
        table.add_column("Conf", justify="right")
        table.add_column("Sources", justify="right")
        table.add_column("Rejections", justify="right")
        table.add_column("Flags", justify="left")

        for i, s in enumerate(summaries, 1):
            flags = ""
            if s.has_annotations:
                flags += "⚠ annotations  "
            if s.rejection_count > 0:
                flags += f"{'🔴' if s.rejection_count >= 3 else '🟡'} rejected"
            conf_color = (
                "green" if s.confidence >= 0.6 else "yellow" if s.confidence >= 0.4 else "red"
            )
            table.add_row(
                str(i),
                escape(s.title),
                s.status,
                f"[{conf_color}]{s.confidence:.2f}[/{conf_color}]",
                str(s.source_count),
                str(s.rejection_count),
                flags.strip(),
            )

        console.print(table)
        console.print(
            "\n[dim]  Type: number=open draft, a=approve all, v=verify all, "
            "p=publish verified, x=reject all, q=quit[/dim]"
        )
        choice = click.prompt("\nChoice", prompt_suffix=" > ").strip().lower()

        if choice == "q":
            return
        elif choice == "a":
            all_paths = [s.path for s in summaries]
            published = approve_drafts(config, db, all_paths)
            console.print(f"[green]Published {len(published)} article(s).[/green]")
            from .indexer import append_log, generate_index

            generate_index(config, db)
            append_log(config, f"review | approved {len(published)} articles")
            return
        elif choice == "v":
            all_paths = [s.path for s in summaries]
            verified = verify_drafts(config, db, all_paths)
            console.print(f"[green]Verified {len(verified)} article(s).[/green]")
            from .indexer import append_log, generate_index

            generate_index(config, db)
            append_log(config, f"review | verified {len(verified)} articles")
            return
        elif choice == "p":
            verified_paths = [s.path for s in summaries if s.status == "verified"]
            if not verified_paths:
                console.print("[yellow]No verified drafts ready to publish.[/yellow]")
                continue
            published = approve_drafts(config, db, verified_paths)
            console.print(f"[green]Published {len(published)} verified article(s).[/green]")
            from .indexer import append_log, generate_index

            generate_index(config, db)
            append_log(config, f"review | published {len(published)} verified articles")
            return
        elif choice == "x":
            reason = click.prompt("Reason for rejecting all", default="")
            for s in summaries:
                reject_draft(s.path, config, db, feedback=reason)
            console.print(f"[yellow]Rejected {len(summaries)} draft(s).[/yellow]")
            return
        elif choice.isdigit():
            idx = int(choice) - 1
            if idx < 0 or idx >= len(summaries):
                console.print("[red]Invalid selection.[/red]")
                continue
            _review_single(
                summaries[idx],
                config,
                db,
                approve_drafts,
                verify_drafts,
                reject_draft,
                compute_diff,
                compute_rejection_diff,
                load_draft_content,
            )
        else:
            console.print("[red]Unknown command.[/red]")


def _review_single(
    summary,
    config,
    db,
    approve_drafts,
    verify_drafts,
    reject_draft,
    compute_diff,
    compute_rejection_diff,
    load_draft_content,
):
    """Handle single-draft review loop."""
    from rich.markup import escape
    from rich.panel import Panel

    from .vault import sanitize_filename

    while True:
        if not summary.path.exists():
            console.print("[yellow]Draft no longer exists.[/yellow]")
            return

        try:
            meta, body = load_draft_content(summary.path)
        except Exception as e:
            console.print(f"[red]Could not read draft: {e}[/red]")
            return

        # Show rejection history
        rejections = db.get_rejections(summary.title, limit=3)
        if rejections:
            console.print(
                Panel(
                    "\n".join(f"• {escape(r['feedback'])}" for r in rejections),
                    title=f"[red]Previous rejections ({len(rejections)})[/red]",
                    border_style="red",
                )
            )

        # Show metadata
        console.print(
            f"[bold]{escape(summary.title)}[/bold]  "
            f"conf={meta.get('confidence', 0):.2f}  "
            f"sources={summary.source_count}  "
            f"rejections={summary.rejection_count}"
        )

        # Show body
        body_display = body[:3000] + ("…" if len(body) > 3000 else "")
        console.print(Panel(escape(body_display), title="Draft"))

        console.print(
            "\n[dim]Type: a=approve, p=publish, v=verify, r=reject, e=edit, "
            "d=diff vs published, x=rejection diff, s=skip[/dim]"
        )
        raw_action = click.prompt("\nAction", prompt_suffix=" > ").strip()
        action = raw_action.lower()

        if action == "s":
            return
        elif action == "a":
            if not summary.path.exists():
                console.print("[yellow]Draft disappeared.[/yellow]")
                return
            published = approve_drafts(config, db, [summary.path])
            console.print(f"[green]Published:[/green] {published[0].name if published else '?'}")
            from .indexer import append_log, generate_index

            generate_index(config, db)
            append_log(config, f"review | approved {summary.title}")
            return
        elif action == "p":
            if not summary.path.exists():
                console.print("[yellow]Draft disappeared.[/yellow]")
                return
            published = approve_drafts(config, db, [summary.path])
            console.print(f"[green]Published:[/green] {published[0].name if published else '?'}")
            from .indexer import append_log, generate_index

            generate_index(config, db)
            append_log(config, f"review | approved {summary.title}")
            return
        elif action == "v":
            if not summary.path.exists():
                console.print("[yellow]Draft disappeared.[/yellow]")
                return
            verified = verify_drafts(config, db, [summary.path])
            verified_name = verified[0].name if verified else summary.path.name
            console.print(f"[green]Verified:[/green] {verified_name}")
            from .indexer import append_log, generate_index

            generate_index(config, db)
            append_log(config, f"review | verified {summary.title}")
            return
        elif action == "r":
            reason = click.prompt("Reason?", default="")
            if not summary.path.exists():
                console.print("[yellow]Draft disappeared.[/yellow]")
                return
            reject_draft(summary.path, config, db, feedback=reason)
            console.print("[yellow]Rejected.[/yellow]")
            if reason:
                count = db.rejection_count(summary.title)
                if db.is_concept_blocked(summary.title):
                    console.print(f"[red]⚠ '{escape(summary.title)}' is now blocked.[/red]")
                else:
                    console.print(f"[dim]({count}/{db._REJECTION_CAP} rejections)[/dim]")
            return
        elif action == "e":
            import os
            import subprocess

            editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
            subprocess.call([editor, str(summary.path)])
        elif action == "d":
            safe_name = sanitize_filename(summary.title)
            wiki_path = config.wiki_dir / f"{safe_name}.md"
            diff = compute_diff(summary.path, wiki_path)
            if diff is None:
                console.print("[dim]No published version — this is a new article.[/dim]")
            else:
                console.print(diff, markup=False)
        elif action == "x":
            diff = compute_rejection_diff(summary.path, db, summary.title)
            if diff is None:
                console.print("[dim]No rejected body stored for this concept.[/dim]")
            else:
                console.print(diff, markup=False)
        else:
            console.print("[red]Unknown action.[/red]")


# ── maintain ──────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@click.option(
    "--fix", is_flag=True, help="Auto-fix missing frontmatter, invalid tags, create stubs"
)  # noqa: E501
@click.option("--stubs-only", is_flag=True, help="Only create stub articles")
@click.option("--dry-run", is_flag=True, help="Report issues without making changes")
@click.option("--clear-cache", is_flag=True, help="Delete all LLM cache entries")
@click.option(
    "--older-than",
    "older_than_days",
    type=int,
    default=None,
    help="With --clear-cache: delete only entries older than N days",
)
def maintain(vault_str, fix, stubs_only, dry_run, clear_cache, older_than_days):
    """Wiki maintenance: health check, stub creation, orphan suggestions, and merge hints.

    Use --dry-run for a read-only health check.
    """
    from .cache import LLMCache
    from .pipeline.lint import run_lint
    from .pipeline.lock import pipeline_lock
    from .pipeline.maintain import (
        create_stubs,
        fix_broken_links,
        normalize_published_alias_links,
        suggest_concept_merges,
        suggest_orphan_links,
    )

    config = _load_config(vault_str)
    db = _load_db(config)

    if clear_cache:
        cache = LLMCache(db)
        deleted = cache.clear(older_than_days=older_than_days)
        if older_than_days is not None:
            console.print(f"Cleared {deleted} LLM cache entries older than {older_than_days} days.")
        else:
            console.print(f"Cleared {deleted} LLM cache entries.")
        return

    if dry_run:
        console.print("[dim]Dry run — no changes will be made.[/dim]\n")

    with pipeline_lock(config.vault) as acquired:
        if not acquired:
            err_console.print("Pipeline already running — lock held.")
            sys.exit(1)

        # Quality warning
        quality = db.quality_stats()
        total_sources = sum(quality.values())
        if total_sources > 0:
            low_pct = round(100 * quality["low"] / total_sources)
            if low_pct > 60:
                console.print(
                    f"[yellow]⚠ {low_pct}% of sources are low quality — "
                    f"articles will have low confidence.[/yellow]"
                )

        # Blocked concepts
        blocked = db.list_blocked_concepts()
        if blocked:
            console.print(f"\n[red]{len(blocked)} blocked concept(s):[/red]")
            for concept in blocked:
                count = db.rejection_count(concept)
                console.print(f"  {concept} ({count} rejections)")
            console.print('[dim]Use [bold]synto unblock "Concept"[/bold] to re-enable.[/dim]')

        if stubs_only:
            if not dry_run:
                created = create_stubs(config, db)
                console.print(f"[green]Created {len(created)} stub(s).[/green]")
            else:
                result = run_lint(config, db)
                broken = [i for i in result.issues if i.issue_type == "broken_link"]
                console.print(f"[dim]Would create up to {min(len(broken), 5)} stub(s).[/dim]")
            return

        # Full lint
        result = run_lint(config, db, fix=fix and not dry_run)
        score = result.health_score
        colour = "green" if score >= 80 else "yellow" if score >= 50 else "red"
        headline = f"[bold {colour}]Structural health: {score}/100[/bold {colour}]"
        if result.advisory_issue_count:
            headline += f"  [dim]({result.advisory_issue_count} advisory issue(s))[/dim]"
        console.print(f"\n{headline}  {result.summary}")

        if result.issues:
            console.print()
            _TYPE_ICON = {
                "orphan": "○",
                "broken_link": "⛓",
                "missing_frontmatter": "⚙",
                "stale": "✎",
                "low_confidence": "↓",
                "config_outdated": "⚠",
            }
            from rich.markup import escape

            for iss in result.issues:
                icon = _TYPE_ICON.get(iss.issue_type, "!")
                fix_tag = " [dim][auto-fixable][/dim]" if iss.auto_fixable else ""
                console.print(
                    f"  {icon} [bold]{iss.issue_type}[/bold]{fix_tag}  {escape(iss.path)}"
                )
                console.print(f"     {escape(iss.description)}")
                console.print(f"     [dim]→ {escape(iss.suggestion)}[/dim]")

        # Alias link normalization in published articles (fix [[Alias]] → [[Canonical|Alias]])
        # Runs independently of broken-link detection: lint resolves aliases so they never
        # appear as broken, but published articles may still have raw alias-form links.
        if fix and not stubs_only:
            alias_normalized = normalize_published_alias_links(config, db, dry_run=dry_run)
            if alias_normalized:
                console.print(
                    f"\n[green]Normalized alias links in {alias_normalized} article(s).[/green]"
                )

        # Broken link repair + stub creation
        broken = [i for i in result.issues if i.issue_type == "broken_link"]
        if broken:
            if fix and not stubs_only:
                repair = fix_broken_links(config, db, broken, dry_run=dry_run)
                if repair.repaired:
                    console.print(f"\n[green]Repaired {repair.repaired} broken link(s).[/green]")
                remaining = repair.still_broken
                if remaining and not dry_run:
                    created = create_stubs(config, db, broken_link_issues=remaining)
                    if created:
                        console.print(f"[green]Created {len(created)} stub(s).[/green]")
                elif remaining:
                    console.print(
                        f"[dim]{len(remaining)} link(s) unresolvable"
                        f" — stubs would be created.[/dim]"
                    )
            elif fix:
                created = create_stubs(config, db, broken_link_issues=broken)
                if created:
                    console.print(f"\n[green]Created {len(created)} stub(s).[/green]")
            else:
                console.print(
                    f"\n[dim]{len(broken)} broken link(s) — "
                    f"run [bold]synto maintain --fix[/bold] to repair or create stubs.[/dim]"
                )

        # Orphan suggestions
        orphan_suggestions = suggest_orphan_links(config, db)
        if orphan_suggestions:
            console.print(f"\n[bold]Orphan link suggestions ({len(orphan_suggestions)}):[/bold]")
            for title, mentioners in orphan_suggestions[:5]:
                console.print(f"  {title} — mentioned in:")
                for m in mentioners[:3]:
                    console.print(f"    [dim]{m}[/dim]")

        # Concept merge suggestions
        merges = suggest_concept_merges(config, db)
        if merges:
            console.print(f"\n[bold]Possible concept duplicates ({len(merges)}):[/bold]")
            for a, b, score in merges[:5]:
                console.print(f"  '{a}' ≈ '{b}'  [dim](similarity={score:.0%})[/dim]")


# ── unblock ───────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@click.argument("concept")
def unblock(vault_str, concept):
    """Re-enable a concept that was blocked after too many rejections."""
    config = _load_config(vault_str)
    db = _load_db(config)

    if not db.is_concept_blocked(concept):
        console.print(f"[yellow]'{concept}' is not blocked.[/yellow]")
        return

    db.unblock_concept(concept)
    count = db.rejection_count(concept)
    console.print(f"[green]'{concept}' unblocked.[/green]")
    console.print(
        f"[dim]{count} rejection(s) remain on record. Next compile will include this concept.[/dim]"
    )


# ── items ─────────────────────────────────────────────────────────────────────


@cli.group()
def items():
    """Audit preserved non-concept knowledge item candidates."""


@items.command("audit")
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@click.option("--limit", default=30, show_default=True, help="Maximum items to show")
def items_audit(vault_str, limit):
    """Show ambiguous/entity candidates preserved during ingest."""
    config = _load_config(vault_str)
    db = _load_db(config)
    candidates = [item for item in db.list_items(status="candidate") if item.kind != "concept"]
    if not candidates:
        console.print("[green]No candidate knowledge items found.[/green]")
        return

    console.print(f"[bold]{len(candidates)} candidate knowledge item(s)[/bold]\n")
    for item in candidates[:limit]:
        mentions = db.get_item_mentions(item.name)
        console.print(
            f"[yellow]{item.name}[/yellow]  "
            f"kind={item.kind} subtype={item.subtype or 'unknown'} "
            f"confidence={item.confidence:.2f} mentions={len(mentions)}"
        )
        for mention in mentions[:3]:
            console.print(
                f"  - {mention.evidence_level}: {mention.source_path} ({mention.mention_text})"
            )
        console.print("  suggested: classify / ignore / keep candidate\n")


@items.command("show")
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@click.argument("name")
def items_show(vault_str, name):
    """Show one preserved knowledge item and its mentions."""
    config = _load_config(vault_str)
    db = _load_db(config)
    item = db.get_item(name)
    if item is None:
        console.print(f"[red]Item not found:[/red] {name}")
        raise SystemExit(1)
    console.print(f"[bold]{item.name}[/bold]")
    console.print(f"  kind: {item.kind}")
    console.print(f"  subtype: {item.subtype or 'unknown'}")
    console.print(f"  status: {item.status}")
    console.print(f"  confidence: {item.confidence:.2f}")
    mentions = db.get_item_mentions(item.name)
    console.print(f"\n[bold]Mentions ({len(mentions)})[/bold]")
    for mention in mentions:
        console.print(f"- {mention.evidence_level}: {mention.source_path}")
        if mention.context:
            console.print(f"  {mention.context}")


# ── compare ───────────────────────────────────────────────────────────────────


def _is_cloud_provider(provider_name: str | None) -> bool:
    from .providers import get_provider

    pname = provider_name or "ollama"
    info = get_provider(pname)
    if info is None:
        return pname != "ollama"
    return info is not None and not info.is_local


def _validate_compare_out_dir(out: Path, config) -> Path:
    out = out.expanduser().resolve()
    raw_dir = config.raw_dir.resolve()
    wiki_dir = config.wiki_dir.resolve()
    app_dir = config.app_dir.resolve()
    compare_root = (config.app_dir / "compare").resolve()

    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    if _is_within(out, raw_dir) or _is_within(out, wiki_dir):
        raise click.BadParameter("--out must not be inside raw/ or wiki/")
    if _is_within(out, app_dir) and not _is_within(out, compare_root):
        raise click.BadParameter("--out under .synto/ is only allowed inside .synto/compare/")
    return out


def _validate_compare_inputs(config, queries_path: str | None) -> None:
    from .compare.runner import _collect_raw_notes, _validate_queries_path

    try:
        _collect_raw_notes(config.raw_dir)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e
    if queries_path:
        try:
            _validate_queries_path(Path(queries_path))
        except ValueError as e:
            raise click.BadParameter(str(e)) from e


def _validate_compare_sample_n(_ctx, _param, value: int | None) -> int | None:
    if value is None or value >= 1:
        return value
    raise click.BadParameter("must be at least 1")


@cli.command(name="compare")
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@_model_override_options
@click.option(
    "--queries",
    "queries_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Optional compare queries.toml.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False),
    default=None,
    show_default=True,
    help="Output directory (default: .synto/compare in the active vault).",
)
@click.option("--keep-artifacts", is_flag=True, help="Do not delete ephemeral vaults.")
@click.option(
    "--allow-cloud-upload",
    is_flag=True,
    help="Required when the challenger uses a cloud provider.",
)
@click.option(
    "--format",
    "report_format",
    type=click.Choice(["md", "json", "both"]),
    default="both",
    show_default=True,
    help="Report output format.",
)
@click.option(
    "--sample-n",
    "sample_n",
    type=int,
    default=None,
    callback=_validate_compare_sample_n,
    help="Limit compare to first N raw notes (useful for a quick spot-check on large vaults).",
)
def compare(
    vault_str,
    fast_model,
    heavy_model,
    provider_name,
    provider_url,
    queries_path,
    out_dir,
    keep_artifacts,
    allow_cloud_upload,
    report_format,
    sample_n,
):
    """Preview whether switching LLM config would improve your vault."""
    from .compare.runner import run_compare

    config = _load_config(vault_str)
    challenger_kwargs = _model_override_kwargs(fast_model, heavy_model, provider_name, provider_url)
    if not challenger_kwargs:
        err_console.print("Provide at least one challenger override, e.g. --heavy-model.")
        sys.exit(1)
    challenger_config = _load_config(vault_str, **challenger_kwargs)

    current_summary = (
        config.models.fast,
        config.models.heavy,
        config.effective_provider.name,
        config.effective_provider.url,
    )
    challenger_summary = (
        challenger_config.models.fast,
        challenger_config.models.heavy,
        challenger_config.effective_provider.name,
        challenger_config.effective_provider.url,
    )
    if challenger_summary == current_summary:
        err_console.print("Challenger config is identical to current config.")
        sys.exit(1)

    _validate_compare_inputs(config, queries_path)

    if _is_cloud_provider(challenger_config.effective_provider.name) and not allow_cloud_upload:
        err_console.print(
            "Cloud challenger requires --allow-cloud-upload "
            "(your raw notes will be sent to the provider)."
        )
        sys.exit(1)

    out = (
        _validate_compare_out_dir(Path(out_dir), config)
        if out_dir
        else (config.app_dir / "compare").resolve()
    )
    out.mkdir(parents=True, exist_ok=True)

    sample_label = f"first {sample_n} notes" if sample_n is not None else "all notes"
    console.print(
        f"[bold]synto compare[/bold] — active vault preview\n"
        f"  vault={config.vault}\n"
        f"  current: fast={config.models.fast} heavy={config.models.heavy} "
        f"provider={config.effective_provider.name}\n"
        f"  challenger: fast={challenger_config.models.fast} "
        f"heavy={challenger_config.models.heavy} "
        f"provider={challenger_config.effective_provider.name}\n"
        f"  queries={'enabled' if queries_path else 'disabled'}\n"
        f"  scope={sample_label}\n"
        f"  Active vault will not be modified."
    )

    report = run_compare(
        current_config=config,
        challenger_config=challenger_config,
        out_dir=out,
        queries_path=Path(queries_path) if queries_path else None,
        keep_artifacts=keep_artifacts,
        sample_n=sample_n,
    )

    from .compare.report import (
        render_json,
        render_markdown,
        render_summary_json,
        render_switch_config_toml,
        resolve,
    )

    resolve(report)

    run_dir = out / report.run_id / "results"
    if report_format in ("md", "both"):
        (run_dir / "report.md").write_text(render_markdown(report))
    if report_format in ("json", "both"):
        (run_dir / "report.json").write_text(render_json(report))
    (run_dir / "summary.json").write_text(render_summary_json(report))

    from .compare.models import AdvisorVerdict

    console.print()
    console.print(f"[green]Run complete:[/green] {report.run_id}")
    console.print(f"Artifacts: {out / report.run_id}")
    console.print(f"[bold]Verdict:[/bold] {report.verdict.value}")
    for reason in report.reasons:
        console.print(f"  · {reason}")
    if report.verdict == AdvisorVerdict.SWITCH:
        console.print(f"\n[bold]Next step:[/bold] edit {CONFIG_FILE_NAME} and set:")
        console.print(
            render_switch_config_toml(
                fast_model=challenger_config.models.fast,
                heavy_model=challenger_config.models.heavy,
                provider_name=challenger_config.effective_provider.name,
                provider_url=challenger_config.effective_provider.url,
            ),
            markup=False,
        )


# ── Trace commands ────────────────────────────────────────────────────────────


@cli.group()
def trace():
    """Trace compile history for articles and concepts."""


@trace.command("article")
@click.argument("name")
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
def trace_article(name: str, vault_str: str | None) -> None:
    """Print compile history for article NAME."""
    from .vault import parse_note, sanitize_filename

    config = _load_config(vault_str)
    db = _load_db(config)

    safe = sanitize_filename(name)
    candidates = [
        config.wiki_dir / f"{safe}.md",
        config.drafts_dir / f"{safe}.md",
    ]
    article_path = next((p for p in candidates if p.exists()), None)
    if article_path is None:
        console.print(f"[red]Article not found:[/red] {name}")
        raise SystemExit(1)

    meta, _ = parse_note(article_path)
    lineage = meta.get("lineage", [])
    if not lineage:
        console.print(f"[yellow]No lineage recorded for:[/yellow] {name}")
        return

    table = Table(title=f"Compile history: {name}", show_header=True, header_style="bold")
    table.add_column("Compile run", style="cyan", no_wrap=True)
    table.add_column("Fast model")
    table.add_column("Heavy model")
    table.add_column("Timestamp")

    for entry in lineage:
        pipeline = entry.get("pipeline", {})
        run_id = str(entry.get("compile_run", ""))
        # Also look up the full row from DB if available
        row = db.get_compile_run(run_id) if run_id else None
        fast = row["fast_model"] if row else pipeline.get("fast_model", "—")
        heavy = row["heavy_model"] if row else pipeline.get("heavy_model", "—")
        ts = str(entry.get("timestamp", "—"))[:19]
        table.add_row(run_id[:16] or "—", fast, heavy, ts)

    console.print(table)


# ── synto add ─────────────────────────────────────────────────────────────────

_SOURCE_TYPES = [
    "notes",
    "textbook",
    "paper",
    "spec",
    "api_docs",
    "web_article",
    "corp_docs",
    "transcript",
    "unknown_text",
]


def _source_slug(text: str) -> str:
    """Lowercase alphanumeric slug, max 30 chars."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:30]


def _make_source_id(path: Path) -> str:
    """Stable source_id: slugified stem + first 8 hex chars of SHA-256 of file bytes."""
    import hashlib

    raw = path.read_bytes()
    content_hash = hashlib.sha256(raw).hexdigest()
    return f"{_source_slug(path.stem)}-{content_hash[:8]}"


@cli.command("add")
@click.argument("source", type=click.Path(exists=True))
@click.option(
    "--type",
    "source_type",
    default=None,
    type=click.Choice(_SOURCE_TYPES),
    help="Override source type (default: inferred from extension).",
)
@click.option("--vault", "vault_str", envvar=VAULT_ENV_VAR, default=None)
@click.option(
    "--extend-pack",
    "extend_pack",
    default=None,
    metavar="PACK_NAME",
    help="Reserved for future pack integration; currently a no-op.",
)
@click.option("--force", is_flag=True, help="Re-import even if source already exists.")
def add(
    source: str,
    source_type: str | None,
    vault_str: str | None,
    extend_pack: str | None,
    force: bool,
) -> None:
    """Import a source document into the vault.

    SOURCE is a file path (PDF, markdown, text).  The original is copied to
    .synto/sources/<source_id>/ and recorded in the state database.  For PDF
    files, segments are extracted immediately.
    """
    import hashlib
    import shutil
    from datetime import UTC, datetime
    from types import SimpleNamespace as _NS

    from .models import SourceDocument
    from .paths import effective_app_dir
    from .pipeline.ingest import write_source_content_md as _write_cm
    from .vault import parse_note

    config = _load_config(vault_str)
    db = _load_db(config)

    src_path = Path(source).expanduser().resolve()
    ext = src_path.suffix.lower()

    # Infer source_type from extension when not provided
    if source_type is None:
        source_type = "paper" if ext == ".pdf" else "notes"

    # Compute content hash and stable source_id
    raw_bytes = src_path.read_bytes()
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    source_id = _make_source_id(src_path)

    # --- Duplicate detection ---
    existing_by_hash = db.get_source_document_by_raw_hash(raw_hash)
    if existing_by_hash is not None:
        source_id = str(existing_by_hash["id"])

    existing = db.get_source_document(source_id)
    if existing and not force:
        console.print(
            f"[yellow]Already imported:[/yellow] {source_id}\n"
            f"Use [bold]--force[/bold] to re-import."
        )
        raise SystemExit(1)

    app_dir = effective_app_dir(config.vault)
    dest_dir = app_dir / "sources" / source_id
    dest_path = dest_dir / f"original{ext}"
    raw_path = config.vault / "raw" / f"{source_id}.md"
    segment_count = 0
    pdf_segs = []
    assets_dir = config.vault / "assets" / source_id

    try:
        if force:
            db.delete_source_import_data(source_id)
            if raw_path.exists():
                raw_path.unlink()
            if assets_dir.exists():
                shutil.rmtree(assets_dir)
            if dest_path.exists():
                dest_path.unlink()
            if dest_dir.exists() and not any(dest_dir.iterdir()):
                dest_dir.rmdir()

        # --- Copy original to .synto/sources/<source_id>/ ---
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)

        # --- Record in source_documents ---
        doc = SourceDocument(
            id=source_id,
            source_type=source_type,
            origin_uri=src_path.as_uri(),
            title=src_path.stem,
            imported_at=datetime.now(UTC),
            raw_hash=raw_hash,
            redistribution="unknown",
        )
        db.upsert_source_document(doc)

        # --- Extract segments (PDF only) ---
        note_meta: dict[str, object] | None = None
        if ext == ".pdf":
            from .extractors.pdf import extract_bibliographic_metadata, extract_pdf

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task(f"Extracting segments from {src_path.name}…", total=None)
                pdf_segs = extract_pdf(source_id, dest_path, db, vault_root=config.vault)
            segment_count = len(pdf_segs)
            if pdf_segs:
                biblio = extract_bibliographic_metadata(dest_path, pdf_segs[0].text)
                note_meta = {
                    "authors": biblio.authors,
                    "doi": biblio.doi,
                    "year": biblio.year,
                }
                if biblio.title and biblio.title != src_path.stem:
                    note_meta["source_title"] = biblio.title

        # --- --extend-pack: reserved for future pack integration ---
        if extend_pack is not None:
            console.print(
                f"  Note: pack extension for '{extend_pack}' is not implemented; "
                "exports remain vault-wide."
            )

        # Write assembled content to raw/ so ingest_all picks it up on next run
        if pdf_segs:
            raw_path = _write_cm(
                source_id,
                source_type,
                src_path.stem,
                pdf_segs,
                config.vault,
                metadata=note_meta,
            )
        else:
            note_title = src_path.stem
            note_body = dest_path.read_text(errors="replace")
            if ext == ".md":
                note_meta, note_body = parse_note(dest_path)
                raw_title = note_meta.get("title")
                if isinstance(raw_title, str) and raw_title.strip():
                    note_title = raw_title.strip()
            raw_path = _write_cm(
                source_id,
                source_type,
                note_title,
                [_NS(text=note_body, structural_locator=None, image_refs=[])],
                config.vault,
                metadata=note_meta,
            )
    except Exception:
        db.delete_source_import_data(source_id)
        for path in (raw_path, dest_path):
            if path.exists():
                path.unlink()
        if assets_dir.exists():
            shutil.rmtree(assets_dir)
        if dest_dir.exists() and not any(dest_dir.iterdir()):
            dest_dir.rmdir()
        raise

    # --- Summary ---
    console.print(f"[green]Imported:[/green] {source_id}")
    console.print(f"  Type:    {source_type}")
    console.print(f"  Stored:  {dest_path.relative_to(config.vault)}")
    if segment_count:
        console.print(f"  Segments extracted: {segment_count}")
    console.print(f"  Raw note: {raw_path.relative_to(config.vault)}")
