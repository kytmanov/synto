from __future__ import annotations

from pathlib import Path

APP_NAME = "synto"
APP_DISPLAY_NAME = "Synto"
CLI_NAME = "synto"
PACKAGE_NAME = "synto"

CONFIG_FILE_NAME = "synto.toml"
LEGACY_CONFIG_FILE_NAME = "wiki.toml"

APP_DIR_NAME = ".synto"
LEGACY_APP_DIR_NAME = ".olw"

VAULT_ENV_VAR = "SYNTO_VAULT"

API_KEY_ENV_VAR = "SYNTO_API_KEY"

AUTO_COMMIT_PREFIX = "[synto]"
LEGACY_AUTO_COMMIT_PREFIX = "[olw]"


def config_path(vault: Path) -> Path:
    return Path(vault) / CONFIG_FILE_NAME


def legacy_config_path(vault: Path) -> Path:
    return Path(vault) / LEGACY_CONFIG_FILE_NAME


def app_dir(vault: Path) -> Path:
    return Path(vault) / APP_DIR_NAME


def legacy_app_dir(vault: Path) -> Path:
    return Path(vault) / LEGACY_APP_DIR_NAME


def effective_config_path(vault: Path) -> Path:
    vault = Path(vault)
    path = config_path(vault)
    return path if path.exists() else legacy_config_path(vault)


def effective_app_dir(vault: Path) -> Path:
    vault = Path(vault)
    path = app_dir(vault)
    return path if path.exists() else legacy_app_dir(vault)


def is_legacy_vault(vault: Path) -> bool:
    vault = Path(vault)
    return legacy_config_path(vault).exists() and not config_path(vault).exists()


def migration_message(vault: Path) -> str:
    return (
        f"This looks like an obsidian-llm-wiki vault: {Path(vault).resolve()}\n"
        f"Run `{CLI_NAME} migrate-olw --vault {Path(vault).resolve()}` first."
    )


def is_within(path: Path, root: Path) -> bool:
    """True if `path` is the same as, or nested under, `root`.

    Both sides are resolved first, so `..` and symlinks can't be used to escape the
    root — important for the output-dir containment checks that guard against writing
    into raw/ or wiki/.
    """
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False
