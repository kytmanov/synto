from __future__ import annotations

import hashlib
import json
import logging
import tomllib
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .paths import APP_DIR_NAME, LEGACY_CONFIG_FILE_NAME, effective_config_path
from .providers import get_provider


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


def _toml_value(value: Any) -> str:
    """Serialize a TOML value for options/headers passthrough.

    Handles scalars (bool/int/float/str) plus nested dict (inline table) and list (array) —
    provider-native options can be nested, e.g. {"thinking": {"budget": 1}}. bool is checked
    before int because bool is an int subclass.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return _toml_quote(value)
    if isinstance(value, dict):
        return _toml_inline_table(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise ValueError(f"unsupported TOML value type for passthrough: {type(value).__name__}")


def _toml_inline_table(table: dict[str, Any]) -> str:
    """Serialize a dict to a TOML inline table: { "k" = v, ... } (values may nest). Keys quoted."""
    if not table:
        return "{}"
    items = ", ".join(f"{_toml_quote(str(k))} = {_toml_value(v)}" for k, v in table.items())
    return f"{{ {items} }}"


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

    Emits the named-provider-block format: one [providers.default] connection that both
    model roles reference. Legacy [ollama]/[provider] vaults keep working (the config
    resolver migrates them), but new vaults use this format so per-role providers (#24)
    are an obvious hand-edit.
    """
    if provider_name == "ollama":
        url = provider_url or ollama_url
        fast_ctx, heavy_ctx = 16384, 32768
        timeout_int = 600
    else:
        url = provider_url or ""
        fast_ctx, heavy_ctx = 8192, 32768
        timeout_int = int(provider_timeout)

    provider_lines = [
        "[providers.default]",
        f"name = {_toml_quote(provider_name)}",
        f"url = {_toml_quote(url)}",
        f"timeout = {timeout_int}",
    ]
    if provider_name == "azure":
        api_ver = azure_api_version or "2024-02-15-preview"
        provider_lines.append(f"azure_api_version = {_toml_quote(api_ver)}")
    if provider_name != "ollama":
        prov_info = get_provider(provider_name)
        env_hint = prov_info.env_var if prov_info and prov_info.env_var else "PROVIDER_API_KEY"
        provider_lines.append(
            f'# api_key_env = "{env_hint}"  '
            f"# or set that env var / store the key in ~/.config/synto/config.toml"
        )
    provider_section = "\n".join(provider_lines) + "\n"

    models_section = (
        f"[models.fast]\n"
        f'provider = "default"\n'
        f"model = {_toml_quote(fast_model)}\n"
        f"ctx = {fast_ctx}                  # context window for fast model (tokens)\n\n"
        f"[models.heavy]\n"
        f'provider = "default"\n'
        f"model = {_toml_quote(heavy_model)}\n"
        f"ctx = {heavy_ctx}                 # context window for heavy model (tokens)\n"
        f"# Advanced (optional, hand-edit): temperature, think, "
        f"[models.<role>.options] — see README\n"
        f"# Split providers: add another [providers.<alias>] block and set this role's "
        f'provider = "<alias>"\n'
    )
    return f"{provider_section}\n{models_section}\n{_vault_toml_tail(inline_source_citations)}"


def _vault_toml_tail(inline_source_citations: bool) -> str:
    """The [pipeline] + ingest-overrides section shared by all vault config writers."""
    citation_line = (
        "inline_source_citations = true  # Experimental: add inline source links\n"
        if inline_source_citations
        else "# inline_source_citations = false  # Experimental: add inline source links\n"
    )
    return (
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


def _provider_models_head(providers: list[dict], models: dict[str, dict]) -> str:
    """Build just the [providers.*] + [models.<role>] sections (no [pipeline] tail).

    providers: list of {alias, name, url, timeout?, api_key_env?, azure_api_version?, headers?}.
    Include one named "default" so string-form / embed roles resolve to it.
    models: {role: {provider, model, ctx, think?, temperature?, options?}} for "fast"/"heavy".
    The optional per-model (think/temperature/options) and per-provider (headers) keys are
    emitted only when present, so callers that omit them get byte-identical output.
    Secrets are never written — only api_key_env (an env var name).
    """
    blocks: list[str] = []
    for p in providers:
        lines = [
            f"[providers.{p['alias']}]",
            f"name = {_toml_quote(p['name'])}",
            f"url = {_toml_quote(p['url'])}",
            f"timeout = {int(p.get('timeout') or 600)}",
        ]
        if p.get("api_key_env"):
            lines.append(f"api_key_env = {_toml_quote(p['api_key_env'])}")
        if p.get("name") == "azure" and p.get("azure_api_version"):
            lines.append(f"azure_api_version = {_toml_quote(p['azure_api_version'])}")
        if p.get("headers"):
            lines.append(f"headers = {_toml_inline_table(p['headers'])}")
        blocks.append("\n".join(lines))
    provider_section = "\n\n".join(blocks) + "\n"

    model_blocks: list[str] = []
    for role in ("fast", "heavy"):
        m = models[role]
        mlines = [
            f"[models.{role}]",
            f"provider = {_toml_quote(m['provider'])}",
            f"model = {_toml_quote(m['model'])}",
            f"ctx = {int(m['ctx'])}",
        ]
        if m.get("think") is not None:
            mlines.append(f"think = {'true' if m['think'] else 'false'}")
        if m.get("temperature") is not None:
            mlines.append(f"temperature = {_toml_value(m['temperature'])}")
        if m.get("options"):
            mlines.append(f"options = {_toml_inline_table(m['options'])}")
        model_blocks.append("\n".join(mlines))
    models_section = "\n\n".join(model_blocks) + "\n"
    return f"{provider_section}\n{models_section}"


def dedup_role_connections(
    roles: dict[str, dict],
) -> tuple[list[dict], dict[str, str]]:
    """Collapse per-role connection specs into de-duplicated [providers.*] blocks.

    roles: ordered {role: spec}; spec carries name/url/timeout/api_key_env/azure_api_version/
    headers. Roles sharing a connection share one alias; the first distinct connection becomes
    "default" so string-form/embed roles resolve to it. Connections that differ by api_key_env
    (a different account) or headers stay distinct. Returns (providers_list, {role: alias}).
    Shared by `synto setup` and the `synto compare` contestant materializer.
    """

    def _key(spec: dict) -> tuple:
        return (
            spec.get("name"),
            spec.get("url"),
            spec.get("timeout"),
            spec.get("api_key_env"),
            spec.get("azure_api_version"),
            tuple(sorted((spec.get("headers") or {}).items())),
        )

    providers: list[dict] = []
    role_alias: dict[str, str] = {}
    for role, spec in roles.items():
        key = _key(spec)
        match = next((p for p in providers if _key(p) == key), None)
        if match is None:
            alias = "default" if not providers else spec["name"]
            base, n = alias, 2
            while any(p["alias"] == alias for p in providers):
                alias = f"{base}{n}"
                n += 1
            match = {
                "alias": alias,
                "name": spec["name"],
                "url": spec["url"],
                "timeout": spec.get("timeout"),
                "api_key_env": spec.get("api_key_env"),
                "azure_api_version": spec.get("azure_api_version"),
                "headers": spec.get("headers") or {},
            }
            providers.append(match)
        role_alias[role] = match["alias"]
    return providers, role_alias


def role_providers_head(config: Config) -> str:
    """Render [providers.*] + [models.fast/heavy] reproducing config's *resolved* roles.

    Single source of truth for both the `synto compare` contestant vault and the SWITCH
    "apply this config" snippet: the fast and heavy roles can resolve to different
    providers/accounts and carry distinct per-role params, all preserved here. Only api_key_env
    (the env-var name) is emitted — never the secret.
    """

    def _spec(r: ResolvedModel) -> dict:
        return {
            "name": r.provider_kind,
            "url": r.url,
            "timeout": int(r.timeout),
            "api_key_env": r.api_key_env,
            "azure_api_version": r.azure_api_version if r.provider_kind == "azure" else None,
            "headers": r.headers,
            "model": r.model,
            "ctx": r.ctx,
            "think": r.think,
            "temperature": r.temperature,
            "options": r.options,
        }

    role_specs = {
        "fast": _spec(config.resolve_role("fast")),
        "heavy": _spec(config.resolve_role("heavy")),
    }
    providers, role_alias = dedup_role_connections(role_specs)
    models = {
        role: {
            "provider": role_alias[role],
            "model": s["model"],
            "ctx": s["ctx"],
            "think": s["think"],
            "temperature": s["temperature"],
            "options": s["options"],
        }
        for role, s in role_specs.items()
    }
    return _provider_models_head(providers, models)


def multi_provider_vault_toml(
    providers: list[dict],
    models: dict[str, dict],
    inline_source_citations: bool = False,
) -> str:
    """Full vault synto.toml with several [providers.*] blocks + per-role refs."""
    head = _provider_models_head(providers, models)
    return f"{head}\n{_vault_toml_tail(inline_source_citations)}"


def strip_provider_model_sections(text: str) -> str:
    """Drop [models]/[models.*], [ollama], [provider], [providers.*] sections.

    Every other section ([pipeline], [mcp], ...) and any leading content is preserved
    verbatim, so re-applying a provider choice to an existing vault keeps user settings.
    """
    import re

    header_re = re.compile(r"^\s*\[([^\]]+)\]")
    drop = {"models", "ollama", "provider", "providers"}
    out: list[str] = []
    skipping = False
    for line in text.splitlines(keepends=True):
        m = header_re.match(line)
        if m:
            top = m.group(1).strip().split(".", 1)[0]
            skipping = top in drop
        if not skipping:
            out.append(line)
    return "".join(out)


def apply_providers_to_existing_toml(
    existing_text: str, providers: list[dict], models: dict[str, dict]
) -> str:
    """Replace the provider/model sections of an existing synto.toml, keeping the rest."""
    head = _provider_models_head(providers, models)
    remainder = strip_provider_model_sections(existing_text).lstrip("\n")
    return f"{head}\n{remainder}" if remainder.strip() else f"{head}\n"


class ProviderBlock(BaseModel):
    """A named provider connection (= one account). Roles reference it by alias.

    Lives under [providers.<alias>]. `name` is the registry kind (ollama, groq, kimi,
    custom, ...). Secrets are never stored here — only `api_key_env` (an env var name).
    `options`/`headers` are passthrough escape hatches for provider-native params.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "ollama"
    url: str | None = None  # None -> registry default_url for `name`
    timeout: float | None = None  # None -> registry default_timeout
    api_key_env: str | None = None
    azure_api_version: str = "2024-02-15-preview"
    options: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)


class ModelProfile(BaseModel):
    """Per-role model config (table form of [models.<role>]).

    First-class fields get resolver logic (role-aware `think`, `ctx` budget math,
    `temperature` override). Any other provider-native param goes in `options`.
    extra="forbid" turns a misspelled top-level key into a loud load-time error.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    provider: str | None = None  # alias into [providers.*]; None -> default/legacy
    ctx: int | None = None
    think: bool | None = None
    temperature: float | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class ModelsConfig(BaseModel):
    # str form (legacy/simple) uses the default/global provider; table form is a ModelProfile.
    fast: str | ModelProfile = "gemma4:e4b"
    heavy: str | ModelProfile = "qwen2.5:14b"
    embed: str | ModelProfile = "nomic-embed-text"  # used only when RAG optional dep installed


class OllamaConfig(BaseModel):
    url: str = "http://localhost:11434"
    timeout: float = 600.0  # seconds; 14B models over network need >5min
    fast_ctx: int = 16384
    heavy_ctx: int = 32768


class ProviderConfig(BaseModel):
    """Legacy per-vault provider config. Supersedes [ollama] when present."""

    name: str = "ollama"
    url: str = "http://localhost:11434"
    timeout: float = 600.0
    fast_ctx: int = 16384
    heavy_ctx: int = 32768
    azure_api_version: str = "2024-02-15-preview"


@dataclass
class ResolvedModel:
    """Everything a client needs for one role, after folding config + registry defaults."""

    provider_kind: str
    url: str
    api_key: str | None
    # The env-var NAME the key came from (block api_key_env), not the secret. Carried so the
    # compare materializer can reproduce the contestant's api_key_env without copying secrets.
    api_key_env: str | None
    timeout: float
    model: str
    ctx: int
    think: bool | None
    temperature: float | None
    supports_json_mode: bool
    supports_embeddings: bool
    azure: bool
    azure_api_version: str
    anthropic_compat: bool
    options: dict[str, Any] = dataclass_field(default_factory=dict)
    headers: dict[str, str] = dataclass_field(default_factory=dict)

    @property
    def connection_key(self) -> tuple:
        """Identity for client de-duplication across roles."""
        return (
            self.provider_kind,
            self.url,
            self.api_key,
            self.timeout,
            self.azure,
            self.azure_api_version,
            tuple(sorted(self.headers.items())),
        )

    @property
    def cache_namespace(self) -> str:
        """Account-aware cache namespace: the connection_key fields minus timeout (which doesn't
        change a response). The secret is hashed (sha256), never stored plaintext. Mirrors client
        identity so cache-sharing == client-sharing — two accounts on one URL never collide."""
        ident = json.dumps(
            [
                self.provider_kind,
                self.url,
                self.api_key or "",
                self.azure_api_version,
                sorted(self.headers.items()),
            ],
            sort_keys=True,
        )
        return hashlib.sha256(ident.encode()).hexdigest()


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


class McpSourceAccessConfig(BaseModel):
    mode: Literal["permissive_only", "all", "deny"] = "permissive_only"
    permissive_licenses: list[str] = Field(
        default_factory=lambda: [
            "CC-BY",
            "CC-BY-SA",
            "MIT",
            "Apache-2.0",
            "BSD-3-Clause",
            "public-domain",
        ]
    )


class McpConfig(BaseModel):
    default_visibility: str = "public"
    exclude_tags: list[str] = Field(default_factory=list)
    audit: bool = False
    # When True, MCP audit rows store raw stringified arg values and resolved
    # labels instead of 8-char SHA256 hashes. Default-off preserves the v0.4.0
    # privacy posture; opt-in to see raw query text in `synto doctor --backlog`.
    audit_detailed: bool = False
    source_access: McpSourceAccessConfig = Field(default_factory=McpSourceAccessConfig)

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
    providers: dict[str, ProviderBlock] = Field(default_factory=dict)  # named connections
    # Per-invocation CLI override (--provider / --provider-url): when set, supersedes the
    # configured provider connection for ALL roles this run (each role keeps its own model /
    # ctx / think / temperature). Set programmatically from CLI flags, not from synto.toml.
    provider_override: str | None = None
    provider_override_url: str | None = None
    pipeline: PipelineConfig = PipelineConfig()
    rag: RagConfig = RagConfig()
    metrics: MetricsConfig = MetricsConfig()
    mcp: McpConfig = McpConfig()
    cache: CacheConfig = CacheConfig()

    @field_validator("vault", mode="before")
    @classmethod
    def resolve_vault(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()

    @model_validator(mode="after")
    def _validate_provider_refs(self) -> Config:
        """Fail loud at load: bad alias refs and unknown provider kinds without a url."""
        for role in ("fast", "heavy", "embed"):
            prof = getattr(self.models, role)
            if isinstance(prof, ModelProfile) and prof.provider is not None:
                if prof.provider not in self.providers:
                    raise ValueError(
                        f"[models.{role}] references provider '{prof.provider}', "
                        f"which is not defined under [providers.*]"
                    )
        for alias, block in self.providers.items():
            if get_provider(block.name) is None and not block.url:
                raise ValueError(
                    f"[providers.{alias}] has unknown provider name '{block.name}' "
                    f"and no url; set a known name or provide a url (custom provider)"
                )
        return self

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

    def resolve_role(
        self, role: Literal["fast", "heavy", "embed"], *, api_key_env: str | None = None
    ) -> ResolvedModel:
        """Fold config + provider registry into a single per-role connection + params spec."""
        from .api_keys import resolve_api_key

        prof = getattr(self.models, role)
        profile = prof if isinstance(prof, ModelProfile) else ModelProfile(model=prof)

        # Pick the configured connection: explicit alias > "default" block > legacy [provider]/
        # [ollama]. CLI --provider/--provider-url overrides are applied as a post-step below.
        alias: str | None
        if profile.provider is not None:
            alias = profile.provider
            block = self.providers[alias]
        elif "default" in self.providers:
            alias = "default"
            block = self.providers["default"]
        else:
            alias = None
            block = None

        if block is not None:
            kind = block.name
            url = block.url
            timeout = block.timeout
            block_api_key_env = block.api_key_env
            azure_api_version = block.azure_api_version
            options = {**block.options, **profile.options}
            headers = dict(block.headers)
        else:
            legacy = self.effective_provider
            kind = legacy.name
            url = legacy.url
            timeout = legacy.timeout
            block_api_key_env = None
            azure_api_version = legacy.azure_api_version
            options = dict(profile.options)
            headers = {}

        # Per-invocation CLI overrides (this run only). --provider replaces the provider kind
        # (and endpoint, dropping the configured key/headers — a different account); --provider-url
        # alone keeps the resolved kind + its api_key_env/headers/options, redirecting only the URL.
        if self.provider_override:
            alias = None
            kind = self.provider_override
            url = self.provider_override_url  # None -> registry default below
            timeout = None  # -> registry default below
            block_api_key_env = None
            azure_api_version = "2024-02-15-preview"
            options = dict(profile.options)
            headers = {}
        elif self.provider_override_url:
            url = self.provider_override_url

        prov_info = get_provider(kind)
        if not url:
            url = prov_info.default_url if prov_info else ""
        if timeout is None:
            timeout = prov_info.default_timeout if prov_info else 600.0

        ctx = profile.ctx if profile.ctx is not None else self._legacy_role_ctx(role)

        # Role-aware think default: fast extraction off; heavy/embed leave the model's default.
        think = profile.think
        if think is None and role == "fast":
            think = False

        api_key = resolve_api_key(
            kind, alias=alias, block_api_key_env=block_api_key_env, api_key_env_override=api_key_env
        )

        return ResolvedModel(
            provider_kind=kind,
            url=url,
            api_key=api_key,
            api_key_env=block_api_key_env,
            timeout=timeout,
            model=profile.model,
            ctx=ctx,
            think=think,
            temperature=profile.temperature,
            supports_json_mode=prov_info.supports_json_mode if prov_info else True,
            supports_embeddings=prov_info.supports_embeddings if prov_info else False,
            azure=prov_info.azure if prov_info else False,
            azure_api_version=azure_api_version,
            anthropic_compat=prov_info.anthropic_compat if prov_info else False,
            options=options,
            headers=headers,
        )

    def _legacy_role_ctx(self, role: str) -> int:
        prov = self.effective_provider
        return prov.fast_ctx if role in ("fast", "embed") else prov.heavy_ctx

    def model_name(self, role: Literal["fast", "heavy", "embed"]) -> str:
        """Return just the model id for a role (no provider/key resolution, no I/O).

        For metrics, checkpoint hashing, pipeline version and display — anywhere the old
        `config.models.<role>` string was used outside an actual LLM call.
        """
        prof = getattr(self.models, role)
        return prof.model if isinstance(prof, ModelProfile) else prof

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
            if (
                key == "models"
                and isinstance(val, dict)
                and isinstance(file_config.get("models"), dict)
            ):
                merged = dict(file_config["models"])
                for role, oval in val.items():
                    existing = merged.get(role)
                    # A bare model-string override (--fast-model/--heavy-model) must keep the
                    # role's configured provider/ctx/params and only swap the model id — otherwise
                    # it drops the provider binding and silently falls back to default/legacy.
                    if isinstance(oval, str) and isinstance(existing, dict):
                        merged[role] = {**existing, "model": oval}
                    else:
                        merged[role] = oval
                file_config["models"] = merged
            elif isinstance(val, dict) and isinstance(file_config.get(key), dict):
                file_config[key] = {**file_config[key], **val}
            else:
                file_config[key] = val
        return cls(vault=vault, **file_config)
