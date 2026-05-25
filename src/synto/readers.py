"""Reader protocol and skeleton implementations (V6 §7).

A Reader provides read-only access to either a working vault (VaultReader)
or an exported pack (PackReader). Readers do not call LLMs.

Engines (engines.py) compose Readers with LLM clients to produce queries,
searches, and answers.

Phase 0 ships skeletons only. Real implementations land in Phase 1A.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Protocol

import frontmatter

from .paths import effective_app_dir
from .state import StateDB
from .vault import is_concept_article_path, is_synthesis_article_path, parse_note


def _confidence_category(value: object) -> str:
    if isinstance(value, str) and value in ("low", "medium", "high"):
        return value
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "high"
    if f < 0.4:
        return "low"
    if f < 0.7:
        return "medium"
    return "high"


# ── Lightweight value types used by Reader ─────────────────────────────────


_WIKILINK_RE = re.compile(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]")
_CITATION_RE = re.compile(r"\[S\d+\]\(#[^)]+\)")
_MD_MARKUP_RE = re.compile(r"[*_]{1,2}([^*_]+)[*_]{1,2}")
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+")
_SENTENCE_END_RE = re.compile(r"[.!?](?:\s|$)")


def _extract_first_paragraph(body: str) -> str | None:
    """Return the first real paragraph from a markdown body, cleaned for use as a summary."""
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or _MD_HEADING_RE.match(line):
            continue
        # Strip wikilinks, citations, markdown bold/italic
        line = _WIKILINK_RE.sub(r"\1", line)
        line = _CITATION_RE.sub("", line)
        line = _MD_MARKUP_RE.sub(r"\1", line)
        line = line.strip()
        if not line:
            continue
        # Step A: normalize whitespace artifacts left by citation removal
        line = re.sub(r"\s+([.,;:])", r"\1", line)
        line = re.sub(r"\s{2,}", " ", line).strip()
        if not line:
            continue
        # Step B: sentence-boundary truncation
        if len(line) <= 200:
            return line
        m = _SENTENCE_END_RE.search(line, 0, 250)
        if m and m.start() >= 20:
            return line[: m.end()].strip()
        truncated = line[:200].rsplit(" ", 1)[0].rstrip(".,;:")
        return (truncated + "…") if truncated else None
    return None


_STATUS_RANK = {"draft": 0, "verified": 1, "published": 2}


def _status_rank(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    return _STATUS_RANK.get(value)


@dataclass(frozen=True)
class ArticleRef:
    id: str
    name: str
    path: str
    summary: str | None = None
    tags: tuple[str, ...] = ()
    confidence: str | None = None
    confidence_score: float | None = None
    source_count: int | None = None
    single_source: bool | None = None
    source_quality: str | None = None
    status: str | None = None
    kind: str = "concept"


@dataclass(frozen=True)
class ConceptRef:
    name: str
    canonical_article_id: str | None = None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class TermRef:
    name: str
    definition: str
    article_id: str | None = None
    provenance: str = "extracted"


@dataclass(frozen=True)
class SegmentRef:
    id: str
    identity: str
    source_id: str
    content_hash: str
    article_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceRef:
    id: str
    title: str | None = None
    source_type: str = "unknown_text"


@dataclass
class Article:
    id: str
    name: str
    path: str
    body: str
    frontmatter: dict[str, object] = field(default_factory=dict)


@dataclass
class Provenance:
    article_id: str
    segment_ids: tuple[str, ...]
    extracted: int = 0
    inferred: int = 0
    ambiguous: int = 0


@dataclass
class PackManifest:
    schema_version: int
    pack_id: str
    version: str
    capabilities: frozenset[str]
    redistribution: str = "unknown"


@dataclass
class PackIndex:
    schema_version: int
    articles: tuple[ArticleRef, ...]
    terms: tuple[TermRef, ...] = ()
    sources: tuple[SourceRef, ...] = ()


class PackError(Exception):
    """Base for all pack-related errors."""


class MalformedPackError(PackError):
    """Pack files exist but are malformed or inconsistent."""


class ArticleNotFound(PackError, KeyError):
    """Requested article is not available."""


class CapabilityNotAvailable(PackError):
    """Requested capability is not present in this reader."""


# ── ArticleFilter ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArticleFilter:
    """Optional filter passed to Reader.list_articles()."""

    tag: str | None = None
    min_confidence: str | None = None
    contains: str | None = None
    min_status: str | None = None
    kind: str | None = None
    exclude_single_source: bool = False


def _apply_filter(refs: list[ArticleRef], filt: ArticleFilter) -> list[ArticleRef]:
    """Apply ArticleFilter to a ref list. Shared by VaultReader and PackReader.

    Legacy-safe: refs without `status`/`single_source` are treated as
    `published` / corroborated so a min_status='published' filter does not
    silently empty a pre-v15 vault.
    """
    if filt.contains is not None:
        query = filt.contains.casefold()
        refs = [
            ref
            for ref in refs
            if query in ref.name.casefold() or query in (ref.summary or "").casefold()
        ]
    if filt.tag is not None:
        refs = [ref for ref in refs if filt.tag in ref.tags]
    if filt.kind is not None:
        refs = [ref for ref in refs if ref.kind == filt.kind]
    if filt.min_status is not None:
        threshold = _status_rank(filt.min_status)
        if threshold is None and filt.min_status != "":
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "ArticleFilter.min_status=%r is not one of %s; filter ignored",
                filt.min_status,
                sorted(_STATUS_RANK),
            )
        if threshold is not None:
            refs = [
                ref
                for ref in refs
                if (
                    _status_rank(ref.status)
                    if ref.status is not None
                    else _STATUS_RANK["published"]
                )
                >= threshold
            ]
    if filt.min_confidence is not None:
        try:
            min_score = float(filt.min_confidence)
        except (TypeError, ValueError):
            min_score = None
        if min_score is not None:
            refs = [
                ref
                for ref in refs
                if ref.confidence_score is None or ref.confidence_score >= min_score
            ]
    if filt.exclude_single_source:
        refs = [ref for ref in refs if ref.single_source is not True]
    return refs


# ── Reader protocol ────────────────────────────────────────────────────────


class Reader(Protocol):
    """Read-only access to a pack or working vault. No LLM calls."""

    @property
    def manifest(self) -> PackManifest: ...

    @property
    def index(self) -> PackIndex: ...

    @property
    def capabilities(self) -> frozenset[str]: ...

    def list_articles(self, filter: ArticleFilter | None = None) -> list[ArticleRef]: ...

    def read_article(self, name_or_id: str) -> Article: ...

    def find_concept(self, query: str) -> ConceptRef | None: ...

    def list_terms(self) -> list[TermRef]: ...

    def find_term(self, query: str) -> TermRef | None: ...

    def get_provenance(self, article_id: str) -> Provenance | None: ...

    def list_sources(self) -> list[SourceRef]: ...

    def list_segments(self) -> list[SegmentRef]: ...

    def has_capability(self, name: str) -> bool: ...


# ── Skeleton implementations (Phase 0: NotImplementedError) ────────────────


class PackReader:
    """Read-only access to an exported pack on disk.

    Phase 0 skeleton. Real implementation lands in Phase 1A.
    """

    def __init__(self, pack_root: Path) -> None:
        self.pack_root = Path(pack_root)
        self._article_lookup: dict[str, ArticleRef] | None = None

        pack_toml = self.pack_root / "pack.toml"
        if not pack_toml.exists():
            raise PackError(f"Not a pack: {self.pack_root} (missing pack.toml)")

    @cached_property
    def manifest(self) -> PackManifest:
        manifest_path = self.pack_root / "agent" / "manifest.json"
        if not manifest_path.exists():
            raise MalformedPackError(f"Missing manifest: {manifest_path}")
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            pack = data["pack"]
            capabilities = frozenset(str(item) for item in pack.get("capabilities", []))
            return PackManifest(
                schema_version=int(data["schema_version"]),
                pack_id=str(pack["id"]),
                version=str(pack["version"]),
                capabilities=capabilities,
                redistribution=str(data.get("redistribution", "unknown")),
            )
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise MalformedPackError(f"Invalid manifest: {manifest_path}") from exc

    @cached_property
    def index(self) -> PackIndex:
        index_path = self.pack_root / "index" / "INDEX.json"
        if not index_path.exists():
            raise MalformedPackError(f"Missing index: {index_path}")
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MalformedPackError(f"Invalid JSON in {index_path}") from exc

        try:
            articles = tuple(
                ArticleRef(
                    id=str(item["id"]),
                    name=str(item["name"]),
                    path=str(item["path"]),
                    summary=item.get("summary"),
                    tags=tuple(str(tag) for tag in item.get("tags", [])),
                    confidence=_confidence_category(item.get("confidence")),
                    confidence_score=(
                        float(item["confidence"])
                        if isinstance(item.get("confidence"), (int, float))
                        else None
                    ),
                )
                for item in data.get("articles", [])
            )
            sources = tuple(
                SourceRef(
                    id=str(item["id"]),
                    title=item.get("title"),
                    source_type=str(item.get("source_type", "unknown_text")),
                )
                for item in data.get("sources", [])
            )
            index = PackIndex(
                schema_version=int(data["schema_version"]),
                articles=articles,
                sources=sources,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedPackError(f"Invalid index payload: {index_path}") from exc

        self._build_article_lookup(index)
        return index

    @property
    def capabilities(self) -> frozenset[str]:
        return self.manifest.capabilities

    def list_articles(self, filter: ArticleFilter | None = None) -> list[ArticleRef]:
        refs = list(self.index.articles)
        if filter is None:
            return refs
        return _apply_filter(refs, filter)

    def read_article(self, name_or_id: str) -> Article:
        ref = self._lookup_article(name_or_id)
        path = self._safe_article_path(ref.path)
        if not path.exists():
            raise MalformedPackError(f"Index references missing article file: {ref.path}")
        post = frontmatter.load(str(path))
        return Article(
            id=ref.id,
            name=ref.name,
            path=ref.path,
            body=post.content,
            frontmatter=dict(post.metadata),
        )

    def find_concept(self, query: str) -> ConceptRef | None:
        entry = self._concepts_lookup.get(query.casefold())
        if entry is None:
            return None
        return self._concept_ref_from_entry(entry)

    def list_terms(self) -> list[TermRef]:
        return list(self.index.terms)

    def find_term(self, query: str) -> TermRef | None:
        q = query.casefold()
        for term in self.list_terms():
            if term.name.casefold() == q:
                return term
        return None

    def get_provenance(self, article_id: str) -> Provenance | None:
        return Provenance(article_id=article_id, segment_ids=())

    def list_sources(self) -> list[SourceRef]:
        return list(self.index.sources)

    def list_segments(self) -> list[SegmentRef]:
        if not self.has_capability("segments"):
            return []

        segments_path = self.pack_root / "agent" / "segments.json"
        if not segments_path.exists():
            return []
        try:
            data = json.loads(segments_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MalformedPackError(f"Invalid JSON in {segments_path}") from exc

        entries = data.get("segments", {})
        segments: list[SegmentRef] = []
        for key, value in entries.items():
            segments.append(
                SegmentRef(
                    id=str(value.get("version_id", key)),
                    identity=str(value["identity"]),
                    source_id=str(value.get("source_id", "")),
                    content_hash=str(value["content_hash"]),
                    article_ids=tuple(str(item) for item in value.get("articles", [])),
                )
            )
        return segments

    def has_capability(self, name: str) -> bool:
        return name in self.capabilities

    def _build_article_lookup(self, index: PackIndex) -> None:
        lookup: dict[str, ArticleRef] = {}
        seen_ids: set[str] = set()
        seen_paths: set[str] = set()
        for ref in index.articles:
            normalized_path = self._normalize_pack_article_path(ref.path)
            if normalized_path in seen_paths:
                raise MalformedPackError(f"Duplicate article path in index: {ref.path}")
            seen_paths.add(normalized_path)

            if ref.id in seen_ids:
                raise MalformedPackError(f"Duplicate article id in index: {ref.id}")
            seen_ids.add(ref.id)

            lookup[ref.id] = ref
            lookup[ref.name.casefold()] = ref
            lookup[normalized_path.casefold()] = ref
            lookup[Path(normalized_path).stem.casefold()] = ref
        self._article_lookup = lookup

    def _load_concepts_json(self) -> dict[str, object]:
        concepts_path = self.pack_root / "agent" / "concepts.json"
        if not concepts_path.exists():
            return {"concepts": []}
        try:
            return json.loads(concepts_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MalformedPackError(f"Invalid JSON in {concepts_path}") from exc

    @cached_property
    def _concepts_lookup(self) -> dict[str, dict[str, object]]:
        concepts = self._load_concepts_json().get("concepts", [])
        lookup: dict[str, dict[str, object]] = {}
        if not isinstance(concepts, list):
            raise MalformedPackError("Invalid concepts payload: concepts must be a list")
        for concept in concepts:
            if not isinstance(concept, dict):
                raise MalformedPackError(
                    "Invalid concepts payload: concept entry must be an object"
                )
            name = concept.get("name")
            if not isinstance(name, str):
                raise MalformedPackError("Invalid concepts payload: concept name must be a string")
            lookup.setdefault(name.casefold(), concept)
            for alias in concept.get("aliases", []):
                if alias:
                    lookup.setdefault(str(alias).casefold(), concept)
        return lookup

    def _concept_ref_from_entry(self, entry: dict[str, object]) -> ConceptRef:
        aliases = tuple(str(alias) for alias in entry.get("aliases", []))
        return ConceptRef(
            name=str(entry["name"]),
            canonical_article_id=(
                str(entry["canonical_article_id"])
                if entry.get("canonical_article_id") is not None
                else None
            ),
            aliases=aliases,
        )

    def _lookup_article(self, name_or_id: str) -> ArticleRef:
        if self._article_lookup is None:
            _ = self.index
        assert self._article_lookup is not None

        key = name_or_id if name_or_id in self._article_lookup else name_or_id.casefold()
        ref = self._article_lookup.get(key)
        if ref is None:
            raise ArticleNotFound(name_or_id)
        return ref

    def _normalize_pack_article_path(self, relative_path: str) -> str:
        raw = Path(relative_path)
        if raw.is_absolute():
            raise MalformedPackError(f"Absolute article path not allowed: {relative_path}")
        parts = raw.parts
        if any(part == ".." for part in parts):
            raise MalformedPackError(f"Path traversal not allowed: {relative_path}")
        return raw.as_posix()

    def _safe_article_path(self, relative_path: str) -> Path:
        normalized = self._normalize_pack_article_path(relative_path)
        candidate = (self.pack_root / normalized).resolve()
        root = self.pack_root.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise MalformedPackError(f"Pack path escapes root: {relative_path}") from exc
        if candidate.is_symlink():
            target = candidate.resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise MalformedPackError(f"Pack symlink escapes root: {relative_path}") from exc
        return candidate


class VaultReader:
    """Read-only access to a working vault (state.db + wiki/).

    Phase 0 skeleton. Real implementation lands in Phase 1A by wrapping
    existing vault.py / state.py code paths.
    """

    def __init__(self, vault_root: Path) -> None:
        self.vault_root = Path(vault_root)
        self._db: StateDB | None = None
        self._cached_records = None
        self._by_article_id: dict[str, object] = {}
        self._by_title_cf: dict[str, object] = {}
        self._by_path_cf: dict[str, object] = {}
        self._by_stem_cf: dict[str, object] = {}

    def _state(self) -> StateDB | None:
        if self._db is not None:
            return self._db

        db_path = effective_app_dir(self.vault_root) / "state.db"
        if not db_path.exists():
            return None
        self._db = StateDB.open_readonly(db_path)
        return self._db

    @cached_property
    def manifest(self) -> PackManifest:
        return PackManifest(
            schema_version=1,
            pack_id=self.vault_root.name,
            version="0.0.0-vault",
            capabilities=self.capabilities,
            redistribution="unknown",
        )

    @cached_property
    def index(self) -> PackIndex:
        return PackIndex(schema_version=1, articles=tuple(self.list_articles()))

    @cached_property
    def capabilities(self) -> frozenset[str]:
        caps = {"articles", "concepts"}
        db = self._state()
        if db is not None and db.count_source_segments() > 0:
            caps.update({"segments", "lifecycle"})
        return frozenset(caps)

    def _ensure_article_cache(self) -> None:
        if self._cached_records is not None:
            return
        db = self._state()
        records = (
            [
                record
                for record in db.list_articles()
                if record.is_published
                and (is_concept_article_path(record.path) or is_synthesis_article_path(record.path))
            ]
            if db is not None
            else []
        )
        self._cached_records = records
        for record in records:
            if record.article_id:
                self._by_article_id.setdefault(record.article_id, record)
            self._by_title_cf.setdefault(record.title.casefold(), record)
            self._by_path_cf.setdefault(record.path.casefold(), record)
            self._by_stem_cf.setdefault(Path(record.path).stem.casefold(), record)

    def list_articles(self, filter: ArticleFilter | None = None) -> list[ArticleRef]:
        refs: list[ArticleRef] = []
        self._ensure_article_cache()
        for record in self._cached_records or []:
            file_path = self.vault_root / record.path
            if not file_path.exists():
                continue
            metadata, body = parse_note(file_path)
            tags = metadata.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            raw_conf = metadata.get("confidence")
            try:
                confidence_score: float | None = float(raw_conf) if raw_conf is not None else None
            except (TypeError, ValueError):
                confidence_score = None

            raw_sc = metadata.get("source_count")
            try:
                source_count: int | None = int(raw_sc) if raw_sc is not None else None
            except (TypeError, ValueError):
                source_count = None

            raw_ss = metadata.get("single_source")
            single_source: bool | None = raw_ss if isinstance(raw_ss, bool) else None

            raw_sq = metadata.get("source_quality")
            source_quality: str | None = (
                raw_sq if isinstance(raw_sq, str) and raw_sq in {"high", "medium", "low"} else None
            )

            raw_status = metadata.get("status")
            status: str | None = (
                raw_status if isinstance(raw_status, str) and raw_status in _STATUS_RANK else None
            )

            kind = "synthesis" if is_synthesis_article_path(record.path) else "concept"

            refs.append(
                ArticleRef(
                    id=record.article_id or record.path,
                    name=record.title,
                    path=record.path,
                    summary=_extract_first_paragraph(body),
                    tags=tuple(str(tag) for tag in tags),
                    confidence=_confidence_category(raw_conf),
                    confidence_score=confidence_score,
                    source_count=source_count,
                    single_source=single_source,
                    source_quality=source_quality,
                    status=status,
                    kind=kind,
                )
            )

        if filter is None:
            return refs
        return _apply_filter(refs, filter)

    def read_article(self, name_or_id: str) -> Article:
        self._ensure_article_cache()
        name_or_id_cf = name_or_id.casefold()
        target = self._by_article_id.get(name_or_id)
        if target is None:
            target = self._by_title_cf.get(name_or_id_cf)
        if target is None:
            target = self._by_path_cf.get(name_or_id_cf)
        if target is None:
            target = self._by_stem_cf.get(name_or_id_cf)
        if target is None:
            raise ArticleNotFound(name_or_id)

        file_path = self.vault_root / target.path
        if not file_path.exists():
            raise ArticleNotFound(name_or_id)
        metadata, body = parse_note(file_path)
        return Article(
            id=target.article_id or target.path,
            name=target.title,
            path=target.path,
            body=body,
            frontmatter=metadata,
        )

    def find_concept(self, query: str) -> ConceptRef | None:
        db = self._state()
        if db is None:
            return None
        result = db.find_concept_by_name_or_alias(query)
        if result is None:
            return None
        name, aliases = result
        article_id = None
        for candidate in db.find_article_candidates(name):
            if candidate.is_published and is_concept_article_path(candidate.path):
                article_id = candidate.article_id
                break
        return ConceptRef(name=name, canonical_article_id=article_id, aliases=tuple(aliases))

    def list_terms(self) -> list[TermRef]:
        state = self._state()
        if state is None:
            return []
        if not state._has_table("concept_occurrences"):
            return []
        rows = state._conn.execute(
            "SELECT concept_name, MAX(confidence) as confidence "
            "FROM concept_occurrences GROUP BY concept_name ORDER BY concept_name"
        ).fetchall()
        return [
            TermRef(name=row["concept_name"], definition="", provenance="extracted") for row in rows
        ]

    def find_term(self, query: str) -> TermRef | None:
        return None

    def get_provenance(self, article_id: str) -> Provenance | None:
        return Provenance(article_id=article_id, segment_ids=())

    def list_sources(self) -> list[SourceRef]:
        db = self._state()
        if db is None:
            return []
        return [
            SourceRef(id=source_id, title=title, source_type=source_type)
            for source_id, title, source_type in db.list_source_documents()
        ]

    def list_segments(self) -> list[SegmentRef]:
        db = self._state()
        if db is None or not self.has_capability("segments"):
            return []
        return [
            SegmentRef(
                id=segment_id,
                identity=identity,
                source_id=source_id,
                content_hash=content_hash,
            )
            for segment_id, identity, source_id, content_hash in db.list_source_segments_brief()
        ]

    def has_capability(self, name: str) -> bool:
        return name in self.capabilities
