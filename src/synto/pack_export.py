from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import Config
from .readers import ArticleFilter, ArticleRef, VaultReader
from .state import StateDB
from .vault import atomic_write, is_concept_article_path, parse_note

ExportTarget = Literal["agents"]

_GENERATED_BLOCK_RE = re.compile(
    r'<!--\s*(?:synto|olw):generated:start\s+name="(?P<name>[^"]+)"\s*-->'
    r"(?P<body>.*?)"
    r"<!--\s*(?:synto|olw):generated:end\s*-->",
    re.DOTALL,
)
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(#[^\]|]*)?(?:\|([^\]]*))?\]\]")
_RAW_MD_GLOB = "*.md"


@dataclass(frozen=True)
class ExportResult:
    target: ExportTarget
    out_dir: Path
    n_articles: int
    n_assets: int
    capabilities: frozenset[str]


def export_pack(config: Config, target: ExportTarget, out: Path | None = None) -> ExportResult:
    if target != "agents":
        raise NotImplementedError("Phase 1A supports only target='agents'")
    if not config.state_db_path.exists():
        raise FileNotFoundError(f"State database not found: {config.state_db_path}")

    out_dir = (out or (config.app_dir / "exports" / target)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _reset_managed_dirs(out_dir)

    db = StateDB.open_readonly(config.state_db_path)
    try:
        reader = VaultReader(config.vault)
        # Pack exports keep concept articles in articles/ and synthesis articles
        # under synthesis/ via _copy_rewritten_markdown_tree below. Including
        # synthesis in this ref list would double-copy them and inflate
        # n_articles, so we filter here even though VaultReader can return both.
        article_refs = sorted(
            reader.list_articles(filter=ArticleFilter(kind="concept")),
            key=lambda ref: (ref.id, ref.path.casefold()),
        )
        source_refs = _export_source_refs(config, db, article_refs)

        _write_pack_toml(out_dir / "pack.toml", reader)
        _write_json(out_dir / "agent" / "manifest.json", _manifest_payload(reader, source_refs))
        _write_json(out_dir / "agent" / "concepts.json", _concepts_payload(db))
        _write_json(out_dir / "agent" / "sources.json", _sources_payload(source_refs))
        _write_json(out_dir / "agent" / "routes.json", _routes_payload(db, source_refs))

        if reader.has_capability("segments"):
            _write_json(out_dir / "agent" / "segments.json", _segments_payload(db))

        _write_json(
            out_dir / "index" / "INDEX.json",
            _pack_index_payload(config, db, reader, article_refs, source_refs),
        )
        _write_agents_files(out_dir, reader, db, source_refs)

        _copy_articles(config, out_dir, article_refs, source_refs)
        _copy_optional_tree(config.drafts_dir, out_dir / "drafts")
        _copy_rewritten_markdown_tree(
            config.queries_dir, out_dir / "queries", source_refs, "queries"
        )
        _copy_rewritten_markdown_tree(
            config.synthesis_dir, out_dir / "synthesis", source_refs, "synthesis"
        )
        _copy_rewritten_markdown_tree(
            config.sources_dir, out_dir / "sources", source_refs, "sources"
        )
        _copy_raw_notes(config.raw_dir, out_dir / "raw")
        _copy_resources(config.vault / "_resources", out_dir / "_resources")

        return ExportResult(
            target=target,
            out_dir=out_dir,
            n_articles=len(article_refs),
            n_assets=0,
            capabilities=reader.capabilities,
        )
    finally:
        db.close()


def _reset_managed_dirs(out_dir: Path) -> None:
    for name in [
        "agent",
        "articles",
        "drafts",
        "index",
        "queries",
        "raw",
        "_resources",
        "source-notes",
        "sources",
        "synthesis",
    ]:
        shutil.rmtree(out_dir / name, ignore_errors=True)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _manifest_payload(
    reader: VaultReader, source_refs: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "pack": {
            "id": reader.manifest.pack_id,
            "name": reader.manifest.pack_id,
            "version": reader.manifest.version,
            "capabilities": sorted(reader.capabilities),
        },
        "redistribution": reader.manifest.redistribution,
        "raw_included": True,
        "source_count": len(source_refs),
    }


def _filter_export_aliases(aliases: list[str], frequent: frozenset[str] = frozenset()) -> list[str]:
    """Drop path-like, over-long, and ambiguous (multi-concept) aliases from export."""
    return [
        a for a in aliases if "/" not in a and len(a.split()) <= 4 and a.casefold() not in frequent
    ]


def _find_cross_language_related(raw_concepts: list[dict]) -> dict[str, list[str]]:
    """Return {concept_name: [related_name, ...]} for alias↔canonical-name collisions."""
    all_names_cf = {c["name"].casefold(): c["name"] for c in raw_concepts}
    related: dict[str, list[str]] = {c["name"]: [] for c in raw_concepts}

    def _link(a_name: str, b_name: str) -> None:
        if b_name not in related[a_name]:
            related[a_name].append(b_name)
        if a_name not in related[b_name]:
            related[b_name].append(a_name)

    for c in raw_concepts:
        for alias in c["aliases"]:
            alias_cf = alias.casefold()
            # Rule 1: exact alias == canonical name (casefold)
            if match := all_names_cf.get(alias_cf):
                if match != c["name"]:
                    _link(c["name"], match)
                    continue
            # Rule 2: first word of a multi-word alias matches a canonical name.
            # Works for any space-delimited language; no-op for Chinese/Japanese (no spaces).
            words = alias_cf.split()
            if len(words) > 1:
                first_word = words[0]
                if match := all_names_cf.get(first_word):
                    if match != c["name"]:
                        _link(c["name"], match)

    return related


def _concepts_payload(db: StateDB) -> dict[str, object]:
    frequent = frozenset(db.list_frequent_aliases())
    concepts = []
    for name in db.list_all_concept_names():
        canonical_article_id = None
        article_path = None
        for article in db.find_article_candidates(name):
            if not article.is_published or not is_concept_article_path(article.path):
                continue
            canonical_article_id = article.article_id
            article_path = _pack_article_path(article.path)
            break
        concepts.append(
            {
                "name": name,
                "aliases": _filter_export_aliases(db.aliases_for_concept(name), frequent),
                "canonical_article_id": canonical_article_id,
                "article_path": article_path,
            }
        )
    related = _find_cross_language_related(concepts)
    for c in concepts:
        c["related_names"] = related[c["name"]]
    return {"schema_version": 1, "concepts": concepts}


def _sources_payload(source_refs: list[dict[str, object]]) -> dict[str, object]:
    return {"schema_version": 1, "sources": source_refs}


def _routes_payload(db: StateDB, source_refs: list[dict[str, object]]) -> dict[str, object]:
    frequent = frozenset(db.list_frequent_aliases())
    routes: list[dict[str, object]] = []
    for name in db.list_all_concept_names():
        article_path = None
        for article in db.find_article_candidates(name):
            if not article.is_published or not is_concept_article_path(article.path):
                continue
            article_path = _pack_article_path(article.path)
            break
        if article_path is None:
            continue
        routes.append({"surface": name, "kind": "article", "path": article_path, "canonical": name})
        for alias in _filter_export_aliases(db.aliases_for_concept(name), frequent):
            routes.append(
                {
                    "surface": alias,
                    "kind": "article",
                    "path": article_path,
                    "canonical": name,
                }
            )

    for source in source_refs:
        canonical = source["title"]
        routes.append(
            {
                "surface": canonical,
                "kind": "source",
                "path": source["path"],
                "canonical": canonical,
                "raw_path": source["raw_path"],
            }
        )
        routes.append(
            {
                "surface": source["raw_path"],
                "kind": "raw",
                "path": source["raw_path"],
                "canonical": canonical,
            }
        )
        raw_name = Path(str(source["raw_path"])).name
        if raw_name != source["raw_path"]:
            routes.append(
                {
                    "surface": raw_name,
                    "kind": "raw",
                    "path": source["raw_path"],
                    "canonical": canonical,
                }
            )

    unique: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for route in routes:
        key = (str(route["kind"]), str(route["surface"]).casefold(), str(route["path"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(route)
    return {"schema_version": 1, "routes": unique}


def _segments_payload(db: StateDB) -> dict[str, object]:
    segments: dict[str, object] = {}
    for segment_id, identity, source_id, content_hash in db.list_source_segments_brief():
        segments[segment_id] = {
            "version_id": segment_id,
            "identity": identity,
            "source_id": source_id,
            "content_hash": content_hash,
            "articles": [],
        }
    return {"schema_version": 1, "segments": segments}


def _pack_index_payload(
    config: Config,
    db: StateDB,
    reader: VaultReader,
    article_refs: list[ArticleRef],
    source_refs: list[dict[str, object]],
) -> dict[str, object]:
    frequent = frozenset(db.list_frequent_aliases())
    return {
        "schema_version": 1,
        "pack": {
            "id": reader.manifest.pack_id,
            "name": config.vault.name,
            "version": reader.manifest.version,
            "language": _detect_languages(db),
            "capabilities": sorted(reader.capabilities),
        },
        "articles": [
            {
                "id": ref.id,
                "name": ref.name,
                "path": _pack_article_path(ref.path),
                "summary": ref.summary,
                "tags": list(ref.tags),
                "aliases": _filter_export_aliases(db.aliases_for_concept(ref.name), frequent),
                "confidence": (
                    ref.confidence_score if ref.confidence_score is not None else ref.confidence
                ),
            }
            for ref in article_refs
        ],
        "terms": [],
        "papers": [],
        "sources": [
            {
                "id": str(source["id"]),
                "title": source["title"],
                "source_type": str(source["source_type"]),
            }
            for source in source_refs
        ],
        "source_concepts": [
            {
                "source_path": source_path,
                "content_hash": content_hash,
                "concepts": concepts,
            }
            for source_path, content_hash, concepts in db.list_source_concept_seeds()
        ],
        "synthesis": [
            {"path": _rewrite_tree_path(path, "wiki/synthesis", "synthesis"), "title": title}
            for path, title in db.list_synthesis_articles_brief()
        ],
        "stats": {
            "article_count": len(article_refs),
            "draft_count": sum(1 for article in db.list_articles() if article.is_draft),
            "concept_count": db.count_concepts(),
            "alias_count": db.count_aliases(),
            "knowledge_item_count": db.count_knowledge_items(),
            "source_count": len(source_refs),
            "source_segment_count": db.count_source_segments(),
            "failed_note_count": db.count_failed_notes(),
            "failed_concept_count": db.count_failed_concepts(),
        },
    }


def _detect_languages(db: StateDB) -> list[str]:
    languages = db.list_note_languages()
    return languages or ["en"]


def _pack_article_path(vault_article_path: str) -> str:
    path = Path(vault_article_path)
    parts = path.parts
    if not parts or parts[0] != "wiki":
        raise ValueError(f"Unexpected vault article path: {vault_article_path}")
    return Path("articles", *parts[1:]).as_posix()


def _rewrite_tree_path(relative_path: str, source_root: str, dest_root: str) -> str:
    path = Path(relative_path).as_posix()
    prefix = f"{source_root}/"
    if path.startswith(prefix):
        return f"{dest_root}/{path[len(prefix) :]}"
    return path


def _copy_articles(
    config: Config,
    out_dir: Path,
    article_refs: list[ArticleRef],
    source_refs: list[dict[str, object]],
) -> None:
    source_index = _build_source_lookup(source_refs)
    for ref in article_refs:
        source = config.vault / ref.path
        target = out_dir / _pack_article_path(ref.path)
        _copy_rewritten_markdown_file(source, target, source_index, path_kind="articles")


def _copy_optional_tree(source_root: Path, dest_root: Path) -> None:
    if not source_root.exists():
        return
    for path in sorted(source_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Refusing to export symlinked path: {path}")
        if not path.is_file():
            continue
        _copy_file(path, dest_root / path.relative_to(source_root))


def _copy_rewritten_markdown_tree(
    source_root: Path,
    dest_root: Path,
    source_refs: list[dict[str, object]],
    path_kind: str,
) -> None:
    if not source_root.exists():
        return
    source_index = _build_source_lookup(source_refs)
    for path in sorted(source_root.rglob("*.md")):
        if path.is_symlink():
            raise ValueError(f"Refusing to export symlinked path: {path}")
        _copy_rewritten_markdown_file(
            path, dest_root / path.relative_to(source_root), source_index, path_kind=path_kind
        )


def _copy_rewritten_markdown_file(
    source: Path,
    target: Path,
    source_index: dict[str, dict[str, object]],
    *,
    path_kind: str,
) -> None:
    text = source.read_text(encoding="utf-8")
    text = _rewrite_export_wikilinks(text, source_index)
    text = _rewrite_media_refs(text, path_kind)
    atomic_write(target, text)


def _copy_raw_notes(source_root: Path, dest_root: Path) -> None:
    if not source_root.exists():
        return
    for path in sorted(source_root.rglob(_RAW_MD_GLOB)):
        if path.is_symlink():
            raise ValueError(f"Refusing to export symlinked path: {path}")
        if path.name.startswith(".") or "processed" in path.parts:
            continue
        _copy_rewritten_markdown_file(
            path,
            dest_root / path.relative_to(source_root),
            {},
            path_kind="raw",
        )


def _copy_resources(source_root: Path, dest_root: Path) -> None:
    _copy_optional_tree(source_root, dest_root)


def _source_id_from_raw_path(raw_path: str) -> str:
    return raw_path


def _export_source_refs(
    config: Config, db: StateDB, article_refs: list[ArticleRef]
) -> list[dict[str, object]]:
    referenced_by_raw: dict[str, list[str]] = {}
    for ref in article_refs:
        article_path = config.vault / ref.path
        if not article_path.exists():
            continue
        try:
            meta, _ = parse_note(article_path)
        except Exception:
            meta = {}
        sources = meta.get("sources", [])
        if isinstance(sources, str):
            sources = [sources]
        elif not isinstance(sources, list):
            sources = []
        for raw_path in sources:
            referenced_by_raw.setdefault(str(raw_path), []).append(_pack_article_path(ref.path))

    concepts_by_raw = {
        source_path: concepts for source_path, _hash, concepts in db.list_source_concept_seeds()
    }
    raw_hashes = {record.path: record.content_hash for record in db.list_raw()}
    refs: list[dict[str, object]] = []
    if not config.sources_dir.exists():
        return refs
    for page in sorted(config.sources_dir.rglob("*.md")):
        try:
            meta, body = parse_note(page)
        except Exception:
            continue
        title = meta.get("title", page.stem)
        raw_path = meta.get("source_file")
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        concepts = concepts_by_raw.get(raw_path, _concepts_from_source_body(body))
        refs.append(
            {
                "id": _source_id_from_raw_path(raw_path),
                "title": title,
                "path": Path("sources", page.name).as_posix(),
                "raw_path": raw_path,
                "quality": meta.get("quality") if isinstance(meta, dict) else None,
                "concepts": concepts,
                "referenced_by_articles": sorted(set(referenced_by_raw.get(raw_path, []))),
                "raw_included": True,
                "raw_content_hash": raw_hashes.get(raw_path),
                "source_type": "source_summary",
            }
        )
    return refs


def _concepts_from_source_body(body: str) -> list[str]:
    concepts: list[str] = []
    in_concepts = False
    for line in body.splitlines():
        stripped = line.strip()
        if stripped == "## Concepts":
            in_concepts = True
            continue
        if in_concepts and stripped.startswith("## "):
            break
        if not in_concepts or not stripped.startswith("- "):
            continue
        match = _WIKILINK_RE.search(stripped)
        if match is None:
            continue
        target = (match.group(3) or match.group(1)).strip()
        if target:
            concepts.append(target)
    return concepts


def _build_source_lookup(source_refs: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    lookup: dict[str, dict[str, object]] = {}
    for source in source_refs:
        path = Path(str(source["path"]))
        title = str(source["title"])
        raw_path = str(source["raw_path"])
        for key in {
            path.stem.casefold(),
            title.casefold(),
            sanitize_source_target(title).casefold(),
            Path(raw_path).stem.casefold(),
            sanitize_source_target(Path(raw_path).stem).casefold(),
            path.as_posix().casefold(),
        }:
            lookup.setdefault(key, source)
    return lookup


def sanitize_source_target(value: str) -> str:
    return re.sub(r"[\\/:*?\"<>|#\[\]^]", "", value).strip()


def _rewrite_export_wikilinks(text: str, source_index: dict[str, dict[str, object]]) -> str:
    def replace(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        fragment = match.group(2) or ""
        display = match.group(3)
        raw_target = target.split("/", 1)[1] if target.casefold().startswith("sources/") else target
        resolved = source_index.get(raw_target.casefold())
        if resolved is None:
            return match.group(0)
        source_path = str(resolved["path"])
        canonical_target = source_path[:-3] if source_path.endswith(".md") else source_path
        display_text = display if display is not None else str(resolved["title"])
        return f"[[{canonical_target}{fragment}|{display_text}]]"

    return _WIKILINK_RE.sub(replace, text)


def _rewrite_media_refs(text: str, path_kind: str) -> str:
    if path_kind not in {"articles", "queries", "raw", "sources", "synthesis"}:
        return text
    return text.replace("./_resources/", "../_resources/")


def _copy_file(source: Path, target: Path) -> None:
    if source.is_symlink():
        raise ValueError(f"Refusing to export symlinked file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _write_pack_toml(path: Path, reader: VaultReader) -> None:
    capabilities = ", ".join(f'"{capability}"' for capability in sorted(reader.capabilities))
    atomic_write(
        path,
        "\n".join(
            [
                "[pack]",
                f'id = "{reader.manifest.pack_id}"',
                f'name = "{reader.manifest.pack_id}"',
                f'version = "{reader.manifest.version}"',
                f"capabilities = [{capabilities}]",
                "",
                "[pack.license]",
                f'redistribution = "{reader.manifest.redistribution}"',
                "",
            ]
        ),
    )


_CONTENTS_CAP = 30


def _render_generated_blocks(
    reader: VaultReader, db: StateDB, source_refs: list[dict[str, object]]
) -> dict[str, str]:
    articles = reader.list_articles(filter=ArticleFilter(kind="concept"))
    article_count = len(articles)
    capabilities = ", ".join(sorted(reader.capabilities))
    detected_langs = _detect_languages(db)
    languages = ", ".join(detected_langs) if len(detected_langs) > 1 else ""

    # Build header block
    header_lines = [
        "\n## Pack Summary\n",
        f"- Articles: {article_count}",
        f"- Capabilities: {capabilities or 'none'}",
        f"- Raw notes: {sum(1 for source in source_refs if source.get('raw_included'))}",
        f"- Sources: {len(source_refs)}",
    ]
    if languages:
        header_lines.append(f"- Languages: {languages}")
    header_lines.append("- Machine-readable index: `index/INDEX.json`")
    header_lines.append("")

    # Contents table (capped)
    displayed = articles[:_CONTENTS_CAP]
    header_lines.append("## Contents\n")
    header_lines.append("| Article | Confidence | Summary |")
    header_lines.append("|---------|------------|---------|")
    for ref in displayed:
        summary_cell = (ref.summary or "").replace("|", "\\|")
        name_cell = ref.name.replace("|", "\\|")
        header_lines.append(f"| {name_cell} | {ref.confidence} | {summary_cell} |")
    if article_count > _CONTENTS_CAP:
        header_lines.append(f"\n*({article_count - _CONTENTS_CAP} more — see `index/INDEX.json`)*")
    header_lines.append("")

    header_block = "\n".join(header_lines)

    usage_block = (
        "\n## How to Use This Pack\n\n"
        "**Concept lookup:**\n"
        "1. Read `agent/concepts.json` — maps every concept name and alias to"
        " `canonical_article_id` and `article_path`.\n"
        "2. Search by name or alias (case-insensitive)."
        " `related_names` lists same-topic articles in other languages.\n"
        "3. Use `article_path` to open the article file directly.\n\n"
        "**Answering questions:**\n"
        "1. Scan `index/INDEX.json` → `articles[].summary` — often answers simple questions"
        " without opening files.\n"
        "2. For detail, open the article file."
        " Citations ([S1], [S2]) link to files in `sources/`.\n\n"
        "**Evidence:**\n"
        "- Raw notes are included in `raw/` by default. Use them for exact quotes"
        " and claim verification.\n"
        "- `sources/` contains generated source summaries; use `agent/sources.json` to map"
        " summaries back to raw notes.\n\n"
        "**Provenance:**\n"
        "- `index/INDEX.json` → `source_concepts[]` maps each raw note"
        " to the concepts it produced.\n\n"
        "**Rules:**\n"
        "- Do not invent content not present in this pack.\n"
        "- Treat pack files as read-only unless explicitly asked to edit them.\n"
    )

    return {"header": header_block, "usage": usage_block}


def _render_block(name: str, content: str) -> str:
    return (
        f'<!-- synto:generated:start name="{name}" -->\n'
        f"{content.rstrip()}\n"
        "<!-- synto:generated:end -->"
    )


def _fresh_agents_text(existing: str | None, blocks: dict[str, str]) -> str:
    parts: list[str] = []
    if existing:
        parts.append(existing.rstrip())
    if not existing or not existing.lstrip().startswith("#"):
        parts.append("# Knowledge Pack")
    parts.extend(_render_block(name, blocks[name]) for name in blocks)
    return "\n\n".join(part for part in parts if part).rstrip() + "\n"


def _merge_generated_blocks(existing: str | None, blocks: dict[str, str]) -> str:
    if existing is None or (
        "synto:generated:start" not in existing and "olw:generated:start" not in existing
    ):
        return _fresh_agents_text(existing, blocks)

    seen: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        seen.add(name)
        if name not in blocks:
            return match.group(0)
        return _render_block(name, blocks[name])

    merged = _GENERATED_BLOCK_RE.sub(replace, existing)
    missing = [name for name in blocks if name not in seen]
    if missing:
        suffix = "\n\n" if merged.rstrip() else ""
        rendered_missing = "\n\n".join(_render_block(name, blocks[name]) for name in missing)
        merged = merged.rstrip() + suffix + rendered_missing
    return merged.rstrip() + "\n"


def _write_agents_files(
    out_dir: Path, reader: VaultReader, db: StateDB, source_refs: list[dict[str, object]]
) -> None:
    blocks = _render_generated_blocks(reader, db, source_refs)
    agents_path = out_dir / "AGENTS.md"
    existing = agents_path.read_text(encoding="utf-8") if agents_path.exists() else None
    merged = _merge_generated_blocks(existing, blocks)
    atomic_write(agents_path, merged)
    atomic_write(out_dir / "CLAUDE.md", merged)
