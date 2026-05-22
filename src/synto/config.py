from __future__ import annotations

import logging
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from .paths import APP_DIR_NAME, LEGACY_CONFIG_FILE_NAME, effective_config_path


def _toml_quote(value: str) -> str:
    """Return a safely quoted TOML basic string, escaping backslashes, quotes, and control chars."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def default_wiki_toml(
    fast_model: str = "gemma4:e4b",
    heavy_model: str = "qwen2.5:14b",
    ollama_url: str = "http://localhost:11434",
    provider_name: str = "ollama",
    provider_url: str | None = None,
    provider_timeout: float = 600.0,
    azure_api_version: str | None = None,
    inline_source_citations: bool = False,
) -> str:
    """Generate Synto vault config content, optionally pre-filled from global config.

    When provider_name is "ollama" (default), emits the legacy [ollama] section
    so existing vaults keep working unchanged. Non-Ollama providers emit [provider].
    """
    if provider_name == "ollama":
        url = provider_url or ollama_url
        provider_section = (
            f"[ollama]\n"
            f"url = {_toml_quote(url)}\n"
            f"timeout = 600\n"
            f"fast_ctx = 16384                  # context window for fast model (tokens)\n"
            f"heavy_ctx = 32768                 # context window for heavy model (tokens)\n"
        )
    else:
        url = provider_url or ""
        timeout_int = int(provider_timeout)
        provider_section = (
            f"[provider]\n"
            f"name = {_toml_quote(provider_name)}\n"
            f"url = {_toml_quote(url)}\n"
            f"timeout = {timeout_int}\n"
            f"fast_ctx = 8192                   # context window hint (tokens)\n"
            f"heavy_ctx = 32768                 # context window hint (tokens)\n"
        )
        if provider_name == "azure":
            api_ver = azure_api_version or "2024-02-15-preview"
            provider_section += f"azure_api_version = {_toml_quote(api_ver)}\n"
    citation_line = (
        "inline_source_citations = true  # Experimental: add inline source links\n"
        if inline_source_citations
        else "# inline_source_citations = false  # Experimental: add inline source links\n"
    )
    return (
        f"[models]\n"
        f"fast = {_toml_quote(fast_model)}\n"
        f"heavy = {_toml_quote(heavy_model)}\n"
        f"# Optional: set heavy = fast to use a single model for everything\n\n"
        f"{provider_section}\n"
        f"[pipeline]\n"
        f"auto_approve = false\n"
        f"auto_commit = true\n"
        f"auto_maintain = false\n"
        f"watch_debounce = 3.0\n"
        f"max_concepts_per_source = 8\n"
        f"ingest_parallel = false   # true = parallel chunks\n"
        f"article_max_tokens = 16384 # soft cap on generated tokens per article; "
        f"auto-reduced to fit context\n"
        f"concept_draft_soft_cap = 2400 # concept-driven compile only; set to "
        f'"article_max_tokens" to disable extra capping (no effect on --legacy)\n'
        f"{citation_line}"
        f'# source_citation_style = "legend-only"  # legend-only | inline-wikilink\n'
        f'# draft_media = "reference"  # reference | embed | omit\n'
        f"graph_quality_checks = true\n"
        f'# language = "en"  # ISO 639-1 output language; autodetects from notes if unset\n'
        f"#\n"
        f"# Per-source-type ingest overrides. Raises the concept-extraction ceiling for\n"
        f"# long-form sources. Quality-based reduction (medium -> 4, low -> 2) still\n"
        f"# applies within this ceiling, so the override only lifts the high-quality cap.\n"
        f"# Editing this only affects newly-ingested sources; run `synto ingest --force`\n"
        f"# to re-apply it to sources already ingested.\n"
        f"#\n"
        f"# [pipeline.source_overrides.textbook]\n"
        f"# max_concepts_per_source = 25  # default: 8\n"
        f"#\n"
        f"# [pipeline.source_overrides.paper]\n"
        f"# max_concepts_per_source = 15  # default: 8\n"
    )


class ModelsConfig(BaseModel):
    fast: str = "gemma4:e4b"
    heavy: str = "qwen2.5:14b"
    embed: str = "nomic-embed-text"  # used only when RAG optional dependency is installed


class OllamaConfig(BaseModel):
    url: str = "http://localhost:11434"
    timeout: float = 600.0  # seconds; 14B models over network need >5min
    fast_ctx: int = 16384
    heavy_ctx: int = 32768


class ProviderConfig(BaseModel):
    """New per-vault provider config. Supersedes [ollama] when present."""

    name: str = "ollama"
    url: str = "http://localhost:11434"
    timeout: float = 600.0
    fast_ctx: int = 16384
    heavy_ctx: int = 32768
    azure_api_version: str = "2024-02-15-preview"


class SourceTypeOverride(BaseModel):
    max_concepts_per_source: int | None = Field(default=None, ge=1)


class PipelineConfig(BaseModel):
    auto_approve: bool = False
    auto_commit: bool = True
    watch_debounce: float = 3.0
    max_concepts_per_source: int = 8
    source_overrides: dict[str, SourceTypeOverride] = Field(default_factory=dict)
    auto_maintain: bool = False
    ingest_parallel: bool = False  # parallel chunk analysis (needs OLLAMA_NUM_PARALLEL≥4)
    article_max_tokens: int = 16384
    concept_draft_soft_cap: int | str = 2400
    inline_source_citations: bool = False
    source_citation_style: str = "legend-only"
    draft_media: str = "reference"
    graph_quality_checks: bool = True
    language: str | None = None  # ISO 639-1 output language; autodetects from notes if unset

    @field_validator("article_max_tokens")
    @classmethod
    def validate_article_max_tokens(cls, value: int) -> int:
        if value < 512:
            raise ValueError(
                f"article_max_tokens must be >= 512 (got {value}); "
                "values below this disable structured generation reliability."
            )
        return value

    @field_validator("concept_draft_soft_cap")
    @classmethod
    def validate_concept_draft_soft_cap(cls, value: int | str) -> int | str:
        if isinstance(value, str):
            if value != "article_max_tokens":
                raise ValueError(
                    'concept_draft_soft_cap must be an integer >= 512 or "article_max_tokens"'
                )
            return value
        if value < 512:
            raise ValueError(
                f"concept_draft_soft_cap must be >= 512 (got {value}) when set numerically"
            )
        return value

    @field_validator("source_citation_style")
    @classmethod
    def validate_source_citation_style(cls, value: str) -> str:
        allowed = {"legend-only", "inline-wikilink"}
        if value not in allowed:
            raise ValueError(f"source_citation_style must be one of {sorted(allowed)}")
        return value

    @field_validator("draft_media")
    @classmethod
    def validate_draft_media(cls, value: str) -> str:
        allowed = {"reference", "embed", "omit"}
        if value not in allowed:
            raise ValueError(f"draft_media must be one of {sorted(allowed)}")
        return value

    @field_validator("source_overrides")
    @classmethod
    def warn_unknown_source_types(
        cls, value: dict[str, SourceTypeOverride]
    ) -> dict[str, SourceTypeOverride]:
        known = {
            "notes",
            "textbook",
            "paper",
            "spec",
            "api_docs",
            "web_article",
            "corp_docs",
            "transcript",
            "unknown_text",
        }
        for key in value:
            if key not in known:
                logging.getLogger(__name__).warning(
                    "source_overrides: unknown source type %r — override will not apply", key
                )
        return value

    def effective_max_concepts(self, source_type: str) -> int:
        """Return max_concepts_per_source for source_type, or the global default.

        Quality-based reduction still applies *after* this returns: high keeps the
        ceiling, medium clamps to min(ceiling, 4), low to 2. The override only lifts
        the high-quality ceiling -- a medium-quality textbook with an override of 25
        still caps at 4. Per-quality-tier overrides are out of scope.
        """
        override = self.source_overrides.get(source_type)
        if override is not None and override.max_concepts_per_source is not None:
            return override.max_concepts_per_source
        return self.max_concepts_per_source


class RagConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 50
    similarity_threshold: float = 0.7


class MetricsConfig(BaseModel):
    persist: bool = True
    detailed: bool = False
    retention_days: int = 90
    max_size_mb: int = 100
    hash_source_ids: bool = True


class McpConfig(BaseModel):
    default_visibility: str = "public"
    exclude_tags: list[str] = Field(default_factory=list)
    audit: bool = False

    @field_validator("default_visibility")
    @classmethod
    def validate_default_visibility(cls, value: str) -> str:
        allowed = {"public", "private"}
        if value not in allowed:
            raise ValueError(f"default_visibility must be one of {sorted(allowed)}")
        return value


class CacheConfig(BaseModel):
    enabled: bool = False


class Config(BaseModel):
    vault: Path
    models: ModelsConfig = ModelsConfig()
    ollama: OllamaConfig = OllamaConfig()
    provider: ProviderConfig | None = None  # supersedes [ollama] when present
    pipeline: PipelineConfig = PipelineConfig()
    rag: RagConfig = RagConfig()
    metrics: MetricsConfig = MetricsConfig()
    mcp: McpConfig = McpConfig()
    cache: CacheConfig = CacheConfig()

    @field_validator("vault", mode="before")
    @classmethod
    def resolve_vault(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()

    @property
    def effective_provider(self) -> ProviderConfig:
        """Return provider config, silently migrating [ollama] section if needed."""
        if self.provider is not None:
            return self.provider
        return ProviderConfig(
            name="ollama",
            url=self.ollama.url,
            timeout=self.ollama.timeout,
            fast_ctx=self.ollama.fast_ctx,
            heavy_ctx=self.ollama.heavy_ctx,
        )

    @property
    def raw_dir(self) -> Path:
        return self.vault / "raw"

    @property
    def wiki_dir(self) -> Path:
        return self.vault / "wiki"

    @property
    def drafts_dir(self) -> Path:
        return self.vault / "wiki" / ".drafts"

    @property
    def app_dir(self) -> Path:
        return self.vault / APP_DIR_NAME

    @property
    def synto_dir(self) -> Path:
        return self.app_dir

    @property
    def state_db_path(self) -> Path:
        return self.app_dir / "state.db"

    @property
    def chroma_dir(self) -> Path:
        return self.app_dir / "chroma"

    @property
    def sources_dir(self) -> Path:
        return self.vault / "wiki" / "sources"

    @property
    def queries_dir(self) -> Path:
        return self.vault / "wiki" / "queries"

    @property
    def synthesis_dir(self) -> Path:
        return self.vault / "wiki" / "synthesis"

    @property
    def schema_path(self) -> Path:
        return self.vault / "vault-schema.md"

    @classmethod
    def from_vault(cls, vault_path: Path, **overrides) -> Config:
        vault = Path(vault_path).expanduser().resolve()
        config_file = effective_config_path(vault)
        if not config_file.exists() and (vault / LEGACY_CONFIG_FILE_NAME).exists():
            legacy_path = vault / LEGACY_CONFIG_FILE_NAME
            raise FileNotFoundError(
                f"Legacy vault config found at {legacy_path}; "
                f"run `synto migrate-olw --vault {vault}` first."
            )
        file_config: dict = {}
        if config_file.exists():
            with open(config_file, "rb") as f:
                file_config = tomllib.load(f)
        if "telemetry" in file_config:
            raise ValueError(
                f"Legacy [telemetry] config found in {config_file}; "
                "rename it to [metrics] or run `synto migrate-olw --vault <vault>` first."
            )
        # Merge overrides. Dict values merge key-by-key so a partial
        # override (e.g. {"models": {"fast": "X"}}) doesn't clobber
        # sibling keys; scalars replace as before.
        for key, val in overrides.items():
            if val is None:
                continue
            if isinstance(val, dict) and isinstance(file_config.get(key), dict):
                file_config[key] = {**file_config[key], **val}
            else:
                file_config[key] = val
        return cls(vault=vault, **file_config)
