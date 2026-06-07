"""
Compile pipeline: raw notes → wiki articles.

Two compile modes:
  compile_concepts (default, v0.3.0): concept-driven — one article per concept
    extracted during ingest. Incremental: only compiles concepts with new sources.
    Manual-edit protection via content_hash comparison.

  compile_notes (legacy, --legacy flag): two-step LLM planning (CompilePlan →
    SingleArticle). Kept as fallback.

Articles are written to wiki/.drafts/ for human review before publishing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import frontmatter as fm_lib

from ..config import Config
from ..markdown_math import mask_markdown_regions, restore_markdown_regions, sanitize_obsidian_math
from ..models import ArticlePlan, CompilePlan, PipelineVersion, SingleArticle, WikiArticleRecord
from ..openai_compat_client import LLMBadRequestError, LLMTruncatedError
from ..paths import rel_posix, to_posix
from ..sanitize import sanitize_tags
from ..state import StateDB
from ..structured_output import StructuredOutputError, request_structured
from ..vault import (
    _mask_code_blocks,
    _restore_code_blocks,
    atomic_write,
    build_wiki_frontmatter,
    ensure_wikilinks,
    extract_wikilinks,
    list_draft_articles,
    list_wiki_articles,
    normalize_wikilinks,
    parse_note,
    sanitize_filename,
    write_note,
)

if TYPE_CHECKING:
    from ..client_factory import ModelRouter

log = logging.getLogger(__name__)

# Annotation thresholds — applied to drafts, stripped on approve
_ANNOTATION_CONFIDENCE_THRESHOLD = 0.4
_ANNOTATION_MIN_SOURCES = 2

_STUB_WRITE_SYSTEM = (
    "You are a wiki editor. Write a brief stub article for a wiki concept that was referenced "
    "by other articles but has no source material yet. Keep it under 150 words. Be factual. "
    "Write in the same language as the surrounding wiki content."
)

_PLAN_SYSTEM = (
    "You are a wiki architect. Given source notes, decide what wiki articles to create or update. "
    "Keep article scope atomic (one concept per article). Plan only — no content yet."
)

_WRITE_SYSTEM = (
    "You are a wiki editor. Write a single wiki article from the provided source material. "
    "Be accurate, cite sources via [[wikilinks]] in body text, use ## section headings, "
    "write in evergreen style. Put [[wikilinks]] inline in prose — do not save them for later."
)

_WRITE_SYSTEM_WITH_CITATIONS = (
    "You are a wiki editor. Write a single wiki article from the provided source material. "
    "Be accurate, cite factual claims with provided [S1] style source ids, use ## section "
    "headings, write in evergreen style. Use [[wikilinks]] inline for related concepts. "
    "Use Obsidian math syntax: inline $...$ and display $$...$$. Do not use \\[...\\]."
)


@dataclass(frozen=True)
class SourceRef:
    id: str
    raw_path: str
    title: str
    safe_title: str
    wiki_target: str


def _source_summary_lookup(vault: Path) -> dict[str, tuple[str, str]]:
    """Return {raw/source path -> (display title, source page stem)} for source summaries."""
    lookup: dict[str, tuple[str, str]] = {}
    sources_dir = vault / "wiki" / "sources"
    if not sources_dir.exists():
        return lookup

    for page in sources_dir.rglob("*.md"):
        try:
            meta, _ = parse_note(page)
        except Exception:
            continue
        source_file = meta.get("source_file")
        if not isinstance(source_file, str) or not source_file.strip():
            continue
        title = meta.get("title", page.stem)
        display_title = title.strip() if isinstance(title, str) and title.strip() else page.stem
        lookup[source_file] = (display_title, page.stem)

    return lookup


def _load_vault_schema(config: Config) -> str:
    """Read vault-schema.md if it exists (injected into write prompts for context)."""
    if config.schema_path.exists():
        try:
            return config.schema_path.read_text(encoding="utf-8")[:1500]
        except Exception:
            pass
    return ""


def _resolve_language(sources: list[str], db: StateDB, config: Config) -> str | None:
    """Return output language: config wins; else use detected if all sources agree."""
    if config.pipeline.language:
        return config.pipeline.language
    langs = {db.get_note_language(s) for s in sources} - {None}
    return langs.pop() if len(langs) == 1 else None


_QUALITY_BONUS = {"high": 0.25, "medium": 0.1, "low": 0.0}

# Stubs are short by design (≤150 words). Hardcoded cap is intentional.
_MAX_STUB_PREDICT = 512

# Math floor: structured generation needs enough headroom for title + content + tags.
# Below this, JSON schema can't reliably complete.
_MIN_ARTICLE_PREDICT = 512

_STRUCTURED_ERROR_VERSION = 1


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _categorize_failure(exc: Exception) -> str:
    """Map an exception type to a stable category string used in compile summaries."""
    if isinstance(exc, LLMTruncatedError):
        return "truncated"
    if isinstance(exc, LLMBadRequestError):
        return "bad_request"
    if isinstance(exc, StructuredOutputError):
        return "structured_output"
    return "other"


def _structured_compile_error(reason: str, message: str) -> str:
    return json.dumps(
        {"version": _STRUCTURED_ERROR_VERSION, "reason": reason, "message": message},
        ensure_ascii=True,
    )


def _concept_draft_num_predict(
    config: Config, prompt: str, system: str, heavy_ctx: int | None = None
) -> int:
    computed = _article_num_predict(config, prompt, system, heavy_ctx)
    soft_cap = config.pipeline.concept_draft_soft_cap
    if soft_cap == "article_max_tokens":
        return computed
    capped = min(soft_cap, computed)
    if computed > capped:
        log.info(
            "Capping concept draft output budget %d -> %d (pipeline.concept_draft_soft_cap)",
            computed,
            capped,
        )
    return capped


def _article_num_predict(
    config: Config, prompt: str, system: str, heavy_ctx: int | None = None
) -> int:
    """Return num_predict capped by both user config and remaining context budget.

    heavy_ctx defaults to the legacy global heavy ctx; callers with a resolved heavy-role
    endpoint pass that role's ctx so per-role provider context windows are honored.

    Raises ValueError if the available output budget is below the floor needed for
    reliable structured generation — caller should treat this as a per-article
    failure (sources too large for context), not a global crash.
    """
    if heavy_ctx is None:
        heavy_ctx = config.effective_provider.heavy_ctx
    estimated_prompt_tokens = max(1, len(system + prompt) // 4)
    available_output = heavy_ctx - estimated_prompt_tokens - 256
    if available_output < _MIN_ARTICLE_PREDICT:
        raise ValueError(
            f"Source content too large for heavy_ctx={heavy_ctx}: "
            f"prompt ~{estimated_prompt_tokens} tokens leaves only {available_output} "
            f"for output (need >= {_MIN_ARTICLE_PREDICT}). Reduce sources or raise heavy_ctx."
        )

    return max(_MIN_ARTICLE_PREDICT, min(config.pipeline.article_max_tokens, available_output))


_DRAFT_ANNOTATION_PREFIX = "synto-auto"
_LEGACY_DRAFT_ANNOTATION_PREFIX = "olw-auto"


def _build_draft_annotations(
    confidence: float,
    source_paths: list[str],
    db: StateDB,
    prompt_degraded: bool = False,
) -> list[str]:
    """Return HTML comment annotations for low-quality drafts. Empty list = no annotations."""
    annotations = []
    if confidence < _ANNOTATION_CONFIDENCE_THRESHOLD:
        annotations.append(
            "<!-- "
            f"{_DRAFT_ANNOTATION_PREFIX}: low-confidence ({confidence:.2f}) "
            "— verify before publishing -->"
        )
    if len(source_paths) < _ANNOTATION_MIN_SOURCES:
        annotations.append(
            f"<!-- {_DRAFT_ANNOTATION_PREFIX}: single-source — cross-reference recommended -->"
        )
    if source_paths:
        qualities = []
        for sp in source_paths:
            rec = db.get_raw(sp)
            if rec and rec.quality:
                qualities.append(rec.quality)
        if qualities and all(q == "low" for q in qualities):
            annotations.append(
                f"<!-- {_DRAFT_ANNOTATION_PREFIX}: all sources low-quality — add better sources -->"
            )
    if prompt_degraded:
        annotations.append(
            f"<!-- {_DRAFT_ANNOTATION_PREFIX}: prompt degraded to fit provider context "
            "— some source or "
            "existing article context was trimmed; review carefully and consider increasing "
            "the loaded model context or lowering heavy_ctx -->"
        )
    return annotations


def _strip_draft_annotations(body: str) -> str:
    """Remove all synto/legacy HTML draft annotations from article body."""
    return re.sub(
        rf"<!--\s*(?:{_DRAFT_ANNOTATION_PREFIX}|{_LEGACY_DRAFT_ANNOTATION_PREFIX}):.*?-->\n?",
        "",
        body,
        flags=re.DOTALL,
    )


def _truncate_to_budget(text: str, max_chars: int) -> str:
    """Rough character-based truncation (≈4 chars per token)."""
    limit = max_chars * 4
    if len(text) > limit:
        return text[:limit] + "\n\n[...truncated...]"
    return text


def _is_prompt_context_overflow(error: Exception) -> bool:
    """Detect provider errors where the prompt itself exceeds the loaded context."""
    if not isinstance(error, LLMBadRequestError):
        return False
    message = str(error).lower()
    return (
        "tokens to keep" in message
        or "n_keep" in message
        or "context length" in message
        or "context too long" in message
    )


def _resolve_source_path(source_path: str, vault: Path) -> tuple[Path, str] | None:
    """Resolve a model/DB source path to an on-disk path and vault-relative path."""
    candidates = [
        vault / source_path,
        vault / "raw" / source_path,
        vault / "raw" / Path(source_path).name,
    ]
    path = next((c for c in candidates if c.exists()), None)
    if path is None:
        return None
    try:
        # POSIX rel path: `rel` is a key matched against POSIX, DB-derived values (dedup sets,
        # summary_lookup, SourceRef.raw_path). str(relative_to) is backslash-separated on
        # Windows, which would miss those lookups; rel_posix keeps it portable.
        rel = rel_posix(path, vault)
    except ValueError:
        rel = to_posix(source_path)
    return path, rel


def _source_title(path: Path, fallback: str) -> str:
    try:
        meta, _ = parse_note(path)
        title = meta.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()
    except Exception:
        pass
    return Path(fallback).stem.replace("-", " ").title()


def _build_source_refs(source_paths: list[str], vault: Path) -> list[SourceRef]:
    refs: list[SourceRef] = []
    seen: set[str] = set()
    summary_lookup = _source_summary_lookup(vault)
    for sp in source_paths:
        resolved = _resolve_source_path(sp, vault)
        if resolved is None:
            continue
        path, rel = resolved
        if rel in seen:
            continue
        seen.add(rel)
        mapped = summary_lookup.get(rel)
        if mapped is not None:
            title, source_stem = mapped
            safe_title = source_stem
        else:
            title = _source_title(path, rel)
            safe_title = sanitize_filename(title)
        refs.append(
            SourceRef(
                id=f"S{len(refs) + 1}",
                raw_path=rel,
                title=title,
                safe_title=safe_title,
                wiki_target=f"sources/{safe_title}",
            )
        )
    return refs


def _gather_sources(
    source_paths: list[str],
    vault: Path,
    max_chars: int = 20000,
    source_refs: list[SourceRef] | None = None,
) -> tuple[str, list[str]]:
    """
    Read source files, return (combined_text, resolved_paths).
    Truncates if combined content exceeds max_chars.
    """
    parts = []
    resolved = []
    ref_by_raw = {r.raw_path: r for r in source_refs or []}
    for sp in source_paths:
        resolved_path = _resolve_source_path(sp, vault)
        if resolved_path is None:
            log.warning("Source not found: %s", sp)
            continue
        p, rel = resolved_path
        try:
            _, body = parse_note(p)
            body = _strip_placeholder_embeds(body)
            ref = ref_by_raw.get(rel)
            if ref is not None:
                parts.append(f"## Source [{ref.id}]: {ref.title} ({ref.raw_path})\n{body}")
            else:
                parts.append(f"## Source: {p.name}\n{body}")
            resolved.append(rel)
        except Exception as e:
            log.warning("Could not read %s: %s", sp, e)

    combined = "\n\n---\n\n".join(parts)
    return _truncate_to_budget(combined, max_chars), resolved


def _source_quality_summary(source_paths: list[str], db: StateDB) -> str:
    """Return best quality (high > medium > low) across all source paths."""
    best = "low"
    for sp in source_paths:
        rec = db.get_raw(sp)
        if rec and rec.quality:
            if rec.quality == "high":
                return "high"
            if rec.quality == "medium" and best != "high":
                best = "medium"
    return best


def _compute_confidence(source_paths: list[str], db: StateDB) -> float:
    """Compute confidence: 0.25 per source + quality bonus from best source."""
    best = _source_quality_summary(source_paths, db)
    return min(1.0, len(source_paths) * 0.25 + _QUALITY_BONUS.get(best, 0.0))


def _mask_citation_rewrite_regions(content: str) -> tuple[str, list[tuple[str, str]]]:
    """Protect markdown regions where [S1] markers must not be rewritten."""
    return mask_markdown_regions(content)


def _restore_masked_regions(content: str, replacements: list[tuple[str, str]]) -> str:
    return restore_markdown_regions(content, replacements)


def _repair_bare_bracket_links(content: str, known_titles: list[str] | None = None) -> str:
    """Convert LLM-produced [Concept] slips into Obsidian [[Concept]] links."""
    masked, replacements = _mask_citation_rewrite_regions(content)
    bare_link_re = re.compile(r"(?<![!\[])\[(?!\[)([^\]\n]+)\](?![\[(])")
    known = {title.casefold() for title in known_titles or []}

    def replace(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        if not target:
            return match.group(0)
        if re.fullmatch(r"S\d+(?:\s*,\s*S\d+)*", target):
            return match.group(0)
        if known and target.casefold() not in known:
            return target
        return f"[[{target}]]"

    return _restore_masked_regions(bare_link_re.sub(replace, masked), replacements)


def _strip_unknown_wikilinks(
    content: str,
    known_titles: list[str],
    stem_to_title: dict[str, str] | None = None,
) -> str:
    """Resolve concept links to their filename stem; unwrap links to no existing page.

    A wikilink resolves only to a note's filename stem, so a concept whose title carries
    filename-forbidden chars (e.g. "TCP/IP" -> TCPIP.md) is matched by stem, not by raw
    title. ``stem_to_title`` maps ``sanitize_filename(title).lower()`` -> title; a link whose
    target sanitizes to a known stem is rewritten to ``[[<stem>|<title>]]`` (so it resolves
    and stays readable), whether the body wrote ``[[TCPIP]]`` or ``[[TCP/IP]]``. Links that
    resolve to no known concept or source page are unwrapped to plain text.
    """
    stem_map = stem_to_title or {}
    masked, replacements = _mask_code_blocks(content)
    known = {title.lower() for title in known_titles}
    wikilink_re = re.compile(r"\[\[([^\]|#]+)(#[^\]|]*)?(?:\|([^\]]*))?\]\]")

    def replace(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        fragment = match.group(2) or ""
        display = match.group(3)
        # Concept resolution by stem wins first, so a raw [[TCP/IP]] is rewritten to its
        # resolving target rather than kept as a broken link by the raw-title fast path.
        stem = sanitize_filename(target)
        canonical = stem_map.get(stem.lower())
        if canonical is not None:
            if target == stem:
                # Target already equals its filename stem → it resolves; leave verbatim
                # (covers normal titles and case variants — only forbidden-char links change).
                return match.group(0)
            new_display = display if display is not None else canonical
            return f"[[{stem}{fragment}|{new_display}]]"
        if target.lower().startswith("sources/") or target.lower() in known:
            return match.group(0)
        return display or f"{target}{fragment}"

    return _restore_code_blocks(wikilink_re.sub(replace, masked), replacements)


def _strip_self_wikilinks(content: str, article_title: str) -> str:
    """Unwrap links that point to the article itself.

    Compares on filename stem so a resolved self-link (e.g. [[TCPIP]] for an article
    titled "TCP/IP") is still recognized.
    """
    title_key = sanitize_filename(article_title).lower()
    wikilink_re = re.compile(r"\[\[([^\]|#]+)(#[^\]|]*)?(?:\|([^\]]*))?\]\]")

    def replace(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        if sanitize_filename(target).lower() != title_key:
            return match.group(0)
        display = match.group(3)
        return display or target

    return wikilink_re.sub(replace, content)


def _strip_empty_wikilinks(content: str) -> str:
    """Remove malformed empty wikilinks like [[]] or [[|display]]."""
    masked, replacements = _mask_code_blocks(content)
    empty_wikilink_re = re.compile(r"\[\[(?:\s*\|\s*([^\]]*))?\s*\]\]")

    def replace(match: re.Match[str]) -> str:
        display = match.group(1)
        if display is None:
            return ""
        return display.strip()

    return _restore_code_blocks(empty_wikilink_re.sub(replace, masked), replacements)


def _repair_wikilink_placeholders(content: str) -> str:
    """Collapse stray placeholder tokens like [[wikilinks][[Title]]] into valid markdown."""
    masked, replacements = _mask_code_blocks(content)
    repaired = masked
    repaired = re.sub(r"\[\[wikilinks\]\[\[([^\]]+)\]\]\]", r"[[\1]]", repaired, flags=re.I)
    repaired = re.sub(r"\[\[wikilinks\]\[\[([^\]]+)\]\]", r"[[\1]]", repaired, flags=re.I)
    repaired = re.sub(r"\[\[wikilinks\]\s*([^\]\n]+)\]", r"\1", repaired, flags=re.I)
    return _restore_code_blocks(repaired, replacements)


def _repair_literal_newlines(content: str) -> str:
    """Repair LLM output that escaped Markdown newlines into literal \n text."""
    if "\\n" not in content:
        return content
    if content.count("\\n") < 2:
        return content
    return content.replace("\\n", "\n")


_MEDIA_EXT_RE = r"(?:pdf|png|jpe?g|gif|svg|webp)"
_PLACEHOLDER_EMBED_RE = re.compile(r"!\[\[[^\]]*\bunknown_filename\b[^\]]*\]\]", re.I)
_MALFORMED_MEDIA_EMBED_RE = re.compile(rf"(?<!\S)!([^\s\[]+\.{_MEDIA_EXT_RE})", re.I)
_MALFORMED_MARKDOWN_MEDIA_RE = re.compile(
    rf"\\?!\\?\[([^\[\]\n]*?\.{_MEDIA_EXT_RE})(?:\\?\])?(?!\()", re.I
)
_OBSIDIAN_MEDIA_EMBED_RE = re.compile(rf"!\[\[([^\]]+\.{_MEDIA_EXT_RE})\]\]", re.I)
_DANGLING_OPEN_BRACKET_RE = re.compile(r"(?m)(?<!\[)[ \t]+\[[ \t]*$")
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(#[^\]|]*)?(?:\|([^\]]*))?\]\]")
_WIKILINK_EDGE_QUOTES = "\"'“”‘’«»‹›„「」『』《》"


def _repair_malformed_embeds(content: str) -> str:
    """Repair LLM/media post-processing slips like !./file.pdf into ![[./file.pdf]]."""
    masked, replacements = _mask_code_blocks(content)

    def replace(match: re.Match[str]) -> str:
        return f"![[{match.group(1).strip()}]]"

    repaired = _MALFORMED_MARKDOWN_MEDIA_RE.sub(replace, masked)
    repaired = _MALFORMED_MEDIA_EMBED_RE.sub(replace, repaired)
    repaired = re.sub(r"\]\]!\[\[", "]]\n![[", repaired)
    return _restore_code_blocks(repaired, replacements)


def _strip_placeholder_embeds(content: str) -> str:
    """Drop Obsidian clipper placeholder embeds before source text reaches the writer."""
    return _PLACEHOLDER_EMBED_RE.sub("", content)


def _remove_dangling_open_brackets(content: str) -> str:
    """Remove truncated markdown-link starts that otherwise survive into drafts."""
    return _DANGLING_OPEN_BRACKET_RE.sub("", content)


def _clean_wikilink_target(target: str) -> str:
    cleaned = target.strip()
    cleaned = re.sub(r"\s*[,;]\s*S\d+(?:\s*,\s*S\d+)*\s*$", "", cleaned)
    return cleaned.strip().strip(_WIKILINK_EDGE_QUOTES).strip()


def _repair_malformed_wikilinks(content: str, known_titles: list[str]) -> str:
    """Trim quote/citation debris from wikilink targets before broken-link checks."""
    masked, replacements = _mask_code_blocks(content)
    known = {title.casefold() for title in known_titles}

    def replace(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        fragment = match.group(2) or ""
        display = match.group(3)
        cleaned = _clean_wikilink_target(target)
        if not cleaned or cleaned == target:
            return match.group(0)
        if cleaned.casefold() in known or cleaned.casefold().startswith("sources/"):
            if display:
                clean_display = display.strip().strip(_WIKILINK_EDGE_QUOTES).strip()
                return f"[[{cleaned}{fragment}|{clean_display}]]"
            return f"[[{cleaned}{fragment}]]"
        return (display or cleaned).strip().strip(_WIKILINK_EDGE_QUOTES).strip()

    return _restore_code_blocks(_WIKILINK_RE.sub(replace, masked), replacements)


def _apply_draft_media_mode(content: str, mode: str) -> str:
    """Control media embeds in synthesized drafts to reduce graph noise."""
    if mode == "embed":
        return content

    def replace(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        if re.search(r"\bunknown_filename\b", target, re.I):
            return ""  # Obsidian clipper placeholder — drop in all modes
        if mode == "omit":
            return ""
        return f"Media reference: {target}"

    return _OBSIDIAN_MEDIA_EMBED_RE.sub(replace, content)


def _rewrite_citation_markers(
    body: str, source_refs: list[SourceRef], *, link_inline: bool = True
) -> str:
    """Normalize [S1] markers. Optionally rewrite them to source wikilinks."""
    if not source_refs:
        return body
    by_id = {ref.id: ref for ref in source_refs}
    masked, replacements = _mask_citation_rewrite_regions(body)
    marker_re = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")

    def replace(match: re.Match[str]) -> str:
        ids = [part.strip() for part in match.group(1).split(",")]
        valid_ids = [id_ for id_ in ids if id_ in by_id]
        if not valid_ids:
            return ""
        if not link_inline:
            text = ",".join(valid_ids)
            return f"[{text}](#Sources)"
        links = [f"[[{by_id[id_].wiki_target}|{id_}]]" for id_ in valid_ids]
        return "(" + ", ".join(links) + ")"

    try:
        return _restore_masked_regions(marker_re.sub(replace, masked), replacements)
    except Exception as exc:  # noqa: BLE001 - citations must never fail compilation
        log.warning("Citation rewrite failed: %s", exc)
        return body


def _inject_body_sections(
    body: str,
    source_paths: list[str],
    config: Config,
    source_refs: list[SourceRef] | None = None,
    article_title: str | None = None,
) -> str:
    """
    Append ## Sources and ## See Also sections to article body.

    ## Sources — [[wikilinks]] to source summary pages in wiki/sources/
    ## See Also — [[wikilinks]] derived from wikilinks already in body
    """
    # Strip any existing ## Sources / ## See Also the LLM may have written
    body = re.sub(r"\n## Sources\b.*", "", body, flags=re.DOTALL).rstrip()
    body = re.sub(r"\n## See Also\b.*", "", body, flags=re.DOTALL).rstrip()

    refs = (
        source_refs if source_refs is not None else _build_source_refs(source_paths, config.vault)
    )

    # ## Sources: link to wiki/sources/{title}.md pages
    source_lines = []
    if config.pipeline.inline_source_citations:
        for ref in refs:
            source_lines.append(f"- [{ref.id}] [[{ref.wiki_target}|{ref.title}]]")
    else:
        for ref in refs:
            link = f"[[{ref.wiki_target}|{ref.title}]]"
            source_lines.append(f"- {link}")

        # Keep historical fallback for unresolved paths when the feature is disabled.
        resolved_raw = {ref.raw_path for ref in refs}
        for sp in source_paths:
            if sp in resolved_raw:
                continue
            src_title = Path(sp).stem.replace("-", " ").title()
            safe_src = sanitize_filename(src_title)
            link = f"[[sources/{safe_src}|{src_title}]]"
            source_lines.append(f"- {link}")

    # ## See Also: wikilinks already in body (sorted, deduplicated)
    linked = sorted(set(extract_wikilinks(body)))
    see_also_lines = []
    for target in linked:
        if not target:
            continue
        if target.lower().startswith("sources/"):
            continue
        if article_title and target.lower() == article_title.lower():
            continue
        see_also_lines.append(f"- [[{target}]]")

    sections = "\n\n## Sources"
    if source_lines:
        sections += "\n" + "\n".join(source_lines)
    if see_also_lines:
        sections += "\n\n## See Also\n" + "\n".join(see_also_lines)

    return body + sections


def _write_draft(
    content_result: SingleArticle,
    config: Config,
    source_paths: list[str],
    db: StateDB,
    confidence: float = 0.5,
    existing_meta: dict | None = None,
    existing_titles: list[str] | None = None,
    concept_aliases: list[str] | None = None,
    alias_map: dict[str, str] | None = None,
    canonical_title: str | None = None,
    prompt_degraded: bool = False,
    run_ulid: str | None = None,
    pipeline: PipelineVersion | None = None,
) -> Path:
    """Write SingleArticle to wiki/.drafts/ and record in state DB."""
    config.drafts_dir.mkdir(parents=True, exist_ok=True)

    article_title = canonical_title or content_result.title
    safe_name = sanitize_filename(article_title)
    draft_path = config.drafts_dir / f"{safe_name}.md"

    # Inject wikilinks for known article titles mentioned in body
    source_refs = _build_source_refs(source_paths, config.vault)
    known_titles = (existing_titles or []) + [ref.title for ref in source_refs]
    body = _repair_literal_newlines(content_result.content)
    body = sanitize_obsidian_math(body)
    body = _repair_malformed_embeds(body)
    body = _repair_bare_bracket_links(body, known_titles)
    body = ensure_wikilinks(body, existing_titles or [])
    # Normalize alias-based links to canonical targets
    if alias_map:
        known = {t.lower() for t in (existing_titles or [])}
        body = normalize_wikilinks(body, alias_map, known)
    if config.pipeline.inline_source_citations:
        body = _rewrite_citation_markers(
            body,
            source_refs,
            link_inline=config.pipeline.source_citation_style == "inline-wikilink",
        )
    source_targets = [ref.wiki_target for ref in source_refs]
    body = _repair_malformed_wikilinks(body, (existing_titles or []) + source_targets)
    body = _repair_wikilink_placeholders(body)
    # Resolve concept links by filename stem so titles with forbidden chars (e.g. "TCP/IP"
    # -> TCPIP.md) survive as readable [[stem|title]] links instead of being stripped.
    stem_to_title: dict[str, str] = {}
    for t in existing_titles or []:
        stem_to_title.setdefault(sanitize_filename(t).lower(), t)
    body = _strip_unknown_wikilinks(body, (existing_titles or []) + source_targets, stem_to_title)
    body = _strip_self_wikilinks(body, article_title)
    body = _strip_empty_wikilinks(body)
    body = _repair_malformed_embeds(body)
    body = _remove_dangling_open_brackets(body)
    body = _apply_draft_media_mode(body, config.pipeline.draft_media)
    body = _inject_body_sections(
        body,
        source_paths,
        config,
        source_refs=source_refs,
        article_title=article_title,
    )

    # Prepend quality annotations (invisible HTML comments, stripped on approve)
    annotations = _build_draft_annotations(
        confidence,
        source_paths,
        db,
        prompt_degraded=prompt_degraded,
    )
    if annotations:
        annotation_block = "\n".join(annotations) + "\n\n"
        # Insert before first ## heading if present, else prepend
        heading_match = re.search(r"^##\s", body, re.MULTILINE)
        if heading_match:
            body = body[: heading_match.start()] + annotation_block + body[heading_match.start() :]
        else:
            body = annotation_block + body

    lineage_entry: list[dict] = []
    if run_ulid:
        lineage_entry = [
            {
                "compile_run": run_ulid,
                "pipeline": pipeline.model_dump() if pipeline else {},
                "timestamp": datetime.now().isoformat(),
            }
        ]
    # Derive quality signals from source paths.
    # `single_source` reflects *document* identity, sourced from source_documents
    # (the v9 canonical store). Maps each raw_notes.path through filename
    # convention to source_documents.id. Falls back to path uniqueness when
    # any source is a manually-dropped raw note (no source_documents row),
    # which is also the correct answer in the current one-raw-file-per-source
    # ingest design.
    if source_paths:
        source_count = len(source_paths)
        source_quality = _source_quality_summary(source_paths, db)

        origin_map = db.get_origin_uris_for_raw_notes(source_paths)
        if all(origin_map.get(sp) for sp in source_paths):
            single_source = len({origin_map[sp] for sp in source_paths}) == 1
        else:
            single_source = len(set(source_paths)) == 1
    else:
        source_count = 0
        single_source = False
        source_quality = None

    meta = build_wiki_frontmatter(
        title=article_title,
        tags=content_result.tags,
        sources=source_paths,
        confidence=confidence,
        is_draft=True,
        existing_meta=existing_meta,
        aliases=concept_aliases or [],
        lineage=lineage_entry if lineage_entry else None,
        source_count=source_count,
        single_source=single_source,
        source_quality=source_quality,
    )

    post = fm_lib.Post(body, **meta)
    atomic_write(draft_path, fm_lib.dumps(post))

    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(config.vault)),
            title=article_title,
            sources=source_paths,
            content_hash=_content_hash(body),
            status="draft",
        )
    )

    return draft_path


# ── Concept-driven compile (default) ──────────────────────────────────────────


def _write_concept_prompt(
    concept: str,
    sources: str,
    existing_titles: list[str],
    existing_content: str = "",
    vault_schema: str = "",
    rejection_history: list[str] | None = None,
    language: str | None = None,
    inline_source_citations: bool = False,
) -> str:
    titles_str = ", ".join(existing_titles[:50]) if existing_titles else "none yet"
    lang_instruction = (
        f"Output language: {language} (ISO 639-1).\n"
        if language
        else "Write in the same language as the source notes.\n"
    )
    prompt = f'Write the wiki article: "{concept}"\n'
    if vault_schema:
        prompt += f"\nVAULT CONVENTIONS:\n{vault_schema}\n"
    prompt += (
        f"\n{lang_instruction}"
        f"IMPORTANT: Keep the content field under 800 words. Be concise.\n"
        f"Tags must be lowercase, hyphen-separated, no spaces or special characters. "
        f"Good: machine-learning, quantum-computing. Bad: Machine Learning, C++.\n"
        f"Do NOT use inline hashtags (#tag) in the content body — use [[wikilinks]] only.\n"
        f"Use Obsidian math syntax: inline $...$ and display $$...$$. Do not use \\[...\\].\n"
        f"If source material references images or diagrams, mention their filenames "
        f"so they can be embedded later (e.g. ![[diagram.png]]).\n"
        f"Use [[wikilinks]] inline in prose to link to related concepts.\n\n"
        f"Existing wiki articles to link to: {titles_str}\n\n"
    )
    if inline_source_citations:
        prompt += (
            "Inline source citations: cite factual sentences/paragraphs with [S1] or "
            "[S1,S2]. Use only ids listed in SOURCE MATERIAL. Do not invent ids. "
            "Do not emit raw [[sources/...]] links. Example: Quantum states can be "
            "entangled [S1,S2].\n\n"
        )
    prompt += f"SOURCE MATERIAL:\n{sources}"
    if existing_content:
        prompt += f"\n\nEXISTING ARTICLE (you are updating this):\n{existing_content}"
    if rejection_history:
        # Deduplicate while preserving order (dict.fromkeys trick)
        unique = list(dict.fromkeys(rejection_history))
        prompt += "\n\nPREVIOUS REJECTIONS — address these issues in this version:\n"
        prompt += "\n".join(f"- {fb}" for fb in unique)
    return prompt


def _build_concept_write_prompt(
    name: str,
    source_paths: list[str],
    config: Config,
    link_titles: list[str],
    existing_content: str,
    vault_schema: str,
    rejection_history: list[str] | None,
    db: StateDB,
    heavy_ctx: int | None = None,
) -> tuple[str, list[str], float, str | None]:
    if heavy_ctx is None:
        heavy_ctx = config.effective_provider.heavy_ctx
    source_refs = (
        _build_source_refs(source_paths, config.vault)
        if config.pipeline.inline_source_citations
        else None
    )
    sources_text, resolved_paths = _gather_sources(
        source_paths,
        config.vault,
        max_chars=heavy_ctx // 2,
        source_refs=source_refs,
    )
    confidence = _compute_confidence(resolved_paths, db)
    lang = _resolve_language([str(p) for p in resolved_paths], db, config)
    write_prompt = _write_concept_prompt(
        name,
        sources_text,
        link_titles,
        existing_content,
        vault_schema,
        rejection_history,
        language=lang,
        inline_source_citations=config.pipeline.inline_source_citations,
    )
    return write_prompt, resolved_paths, confidence, lang


def compile_concepts(
    config: Config,
    router: ModelRouter,
    db: StateDB,
    force: bool = False,
    dry_run: bool = False,
    on_progress: Callable[[int, int, str], None] | None = None,
    concepts: list[str] | None = None,
) -> tuple[list[Path], list[str], dict[str, float]]:
    """
    Concept-driven compile: one article per concept needing compile.

    A concept needs compile if any linked source has status='ingested', or if it
    is a stub (created by synto maintain for broken wikilinks).
    Skips articles whose on-disk content_hash differs from DB (manually edited).
    Pass force=True to recompile even manually-edited articles.

    Pass concepts= to compile only a specific subset (e.g. concepts linked to
    recently changed source files). None = compile all needing compile.
    """
    all_needing = db.concepts_needing_compile()
    if concepts is not None:
        requested: list[str] = []
        seen: set[str] = set()
        for name in concepts:
            canonical = db.resolve_alias(name) or name
            key = canonical.casefold()
            if key not in seen:
                seen.add(key)
                requested.append(canonical)
        concept_names = requested
    else:
        concept_names = all_needing

    if not concept_names:
        log.info("No concepts needing compile")
        return [], [], {}

    fast = router.endpoint("fast")
    heavy = router.endpoint("heavy")

    # Start compile run tracking
    try:
        import ulid as _ulid_mod

        run_ulid = str(_ulid_mod.ULID())
    except ImportError:
        import uuid

        run_ulid = uuid.uuid4().hex.upper()
    pipeline = PipelineVersion(
        fast_model=config.model_name("fast"), heavy_model=config.model_name("heavy")
    )
    if not dry_run and db._has_table("compile_runs"):
        db.start_compile_run(
            run_ulid,
            pipeline.model_dump_json(),
            config.model_name("fast"),
            config.model_name("heavy"),
        )

    log.info("Compiling %d concept(s)", len(concept_names))
    existing_titles = [t for t, _ in list_wiki_articles(config.wiki_dir)]
    draft_titles = [t for t, _, _ in list_draft_articles(config.drafts_dir)]
    link_titles = existing_titles + [t for t in draft_titles if t not in existing_titles]
    for concept_name in concept_names:
        if concept_name not in link_titles:
            link_titles.append(concept_name)
    vault_schema = _load_vault_schema(config)
    total = len(concept_names)
    # Build alias resolution map once per compile run
    alias_map = db.list_alias_map()

    draft_paths: list[Path] = []
    failed: list[str] = []
    failure_categories: dict[str, list[str]] = {}
    concept_timings: dict[str, float] = {}

    def _record_failure(name: str, category: str) -> None:
        if name not in failed:
            failed.append(name)
        failure_categories.setdefault(category, []).append(name)

    for idx, name in enumerate(concept_names, 1):
        if on_progress:
            on_progress(idx, total, name)
        _t_concept = time.monotonic()

        source_paths = db.get_sources_for_concept(name)
        is_stub = db.has_stub(name)
        prompt_degraded = False

        if not source_paths and not is_stub:
            continue

        # Manual edit protection
        safe_name = sanitize_filename(name)
        wiki_path = config.wiki_dir / f"{safe_name}.md"
        draft_path = config.drafts_dir / f"{safe_name}.md"
        existing_meta: dict | None = None

        if draft_path.exists() and not force:
            if dry_run:
                print(f"  [skip] {name} — draft already pending review")
                continue
            log.info(
                "Skipping '%s' — draft already pending review (use --force to overwrite)", name
            )
            db.mark_concept_compile_state(name, source_paths, "deferred_draft")
            continue

        if wiki_path.exists():
            try:
                existing_meta, existing_body = parse_note(wiki_path)
                if not force:
                    art_rec = db.get_article(str(wiki_path.relative_to(config.vault)))
                    if art_rec and art_rec.content_hash != _content_hash(existing_body):
                        if dry_run:
                            print(f"  [skip] {name} — published article manually edited")
                            continue
                        log.info("Skipping '%s' — manually edited (use --force to override)", name)
                        db.mark_concept_compile_state(name, source_paths, "deferred_manual_edit")
                        continue
            except Exception:
                pass

        if force:
            db.clear_deferred_state(name, source_paths)

        if dry_run:
            stub_tag = " [stub]" if is_stub else ""
            print(
                f"  [concept{stub_tag}] {name} — {len(source_paths)} source(s): "
                f"{', '.join(Path(s).name for s in source_paths)}"
            )
            continue

        # For stubs: compile with empty sources using a lightweight stub prompt
        if is_stub and not source_paths:
            stub_prompt = (
                f'Write a brief stub wiki article for the concept: "{name}"\n'
                f"This concept is referenced by other articles but has no source material yet.\n"
                f"Keep it under 150 words. Include a note that this is a stub needing sources."
            )
            try:
                result: SingleArticle = request_structured(
                    client=fast.client,
                    prompt=stub_prompt,
                    model_class=SingleArticle,
                    model=fast.model,
                    system=_STUB_WRITE_SYSTEM,
                    num_ctx=fast.ctx,
                    num_predict=min(_MAX_STUB_PREDICT, fast.ctx),
                    stage="compile_article",
                    model_role="fast",
                    think=fast.think,
                    options=fast.options,
                    temperature=fast.temperature,
                )
            except (StructuredOutputError, LLMBadRequestError, LLMTruncatedError) as e:
                log.error("Failed to write stub '%s': %s", name, e)
                reason = _categorize_failure(e)
                _record_failure(name, reason)
                db.mark_concept_compile_state(
                    name,
                    source_paths,
                    "failed",
                    error=_structured_compile_error(reason, str(e)),
                )
                continue
            draft_path = _write_draft(
                content_result=result,
                config=config,
                source_paths=[],
                db=db,
                confidence=0.0,
                existing_meta=existing_meta,
                existing_titles=link_titles,
                canonical_title=name,
                concept_aliases=db.get_aliases(name),
                alias_map=alias_map,
            )
            draft_paths.append(draft_path)
            db.delete_stub(name)
            db.mark_concept_compile_state(name, source_paths, "compiled")
            elapsed = time.monotonic() - _t_concept
            concept_timings[name] = elapsed
            log.info("Stub draft written: %s (%.1fs)", draft_path.name, elapsed)
            continue

        # Include snippet of existing article for update prompts
        existing_content = ""
        if existing_meta and wiki_path.exists():
            try:
                _, ex_body = parse_note(wiki_path)
                existing_content = ex_body[:2000]
            except Exception:
                pass

        # Inject rejection history into prompt
        rejection_records = db.get_rejections(name, limit=3)
        rejection_history = (
            [r["feedback"] for r in rejection_records] if rejection_records else None
        )

        # Gather source material within context budget
        write_prompt, resolved_paths, confidence, _ = _build_concept_write_prompt(
            name,
            source_paths,
            config,
            link_titles,
            existing_content,
            vault_schema,
            rejection_history,
            db,
            heavy_ctx=heavy.ctx,
        )
        if not resolved_paths:
            log.warning("No readable sources for concept '%s', skipping", name)
            _record_failure(name, "no_sources")
            db.mark_concept_compile_state(
                name,
                source_paths,
                "failed",
                error=_structured_compile_error("no_sources", "No readable sources"),
            )
            continue

        try:
            # Concept drafts stay on a configurable soft output budget so users can
            # keep the default safety rail or explicitly opt back into article_max_tokens.
            num_predict = _concept_draft_num_predict(
                config,
                write_prompt,
                (
                    _WRITE_SYSTEM_WITH_CITATIONS
                    if config.pipeline.inline_source_citations
                    else _WRITE_SYSTEM
                ),
                heavy_ctx=heavy.ctx,
            )
        except ValueError as e:
            log.error("Failed to write '%s': %s", name, e)
            _record_failure(name, "context_too_large")
            db.mark_concept_compile_state(
                name,
                resolved_paths or source_paths,
                "failed",
                error=_structured_compile_error("context_too_large", str(e)),
            )
            continue

        try:
            result = request_structured(
                client=heavy.client,
                prompt=write_prompt,
                model_class=SingleArticle,
                model=heavy.model,
                system=(
                    _WRITE_SYSTEM_WITH_CITATIONS
                    if config.pipeline.inline_source_citations
                    else _WRITE_SYSTEM
                ),
                num_ctx=heavy.ctx,
                num_predict=num_predict,
                stage="compile_article",
                model_role="heavy",
                think=heavy.think,
                options=heavy.options,
                temperature=heavy.temperature,
            )
        except (StructuredOutputError, LLMBadRequestError, LLMTruncatedError) as e:
            retry_succeeded = False
            if _is_prompt_context_overflow(e):
                retry_plan = [
                    (max(512, heavy.ctx // 4), 1000),
                    (max(512, heavy.ctx // 8), 500),
                    (max(512, heavy.ctx // 16), 0),
                    (512, 0),
                ]
                source_refs = (
                    _build_source_refs(source_paths, config.vault)
                    if config.pipeline.inline_source_citations
                    else None
                )
                seen_retry_budgets: set[tuple[int, int]] = set()
                for source_budget, existing_budget in retry_plan:
                    retry_key = (source_budget, existing_budget)
                    if retry_key in seen_retry_budgets:
                        continue
                    seen_retry_budgets.add(retry_key)
                    log.warning(
                        "Retrying '%s' after provider context overflow (sources=%d, existing=%d)",
                        name,
                        source_budget,
                        existing_budget,
                    )
                    fallback_sources_text, fallback_resolved_paths = _gather_sources(
                        source_paths,
                        config.vault,
                        max_chars=source_budget,
                        source_refs=source_refs,
                    )
                    if not fallback_resolved_paths:
                        continue
                    fallback_lang = _resolve_language(
                        [str(p) for p in fallback_resolved_paths], db, config
                    )
                    fallback_prompt = _write_concept_prompt(
                        name,
                        fallback_sources_text,
                        link_titles,
                        existing_content[:existing_budget] if existing_budget else "",
                        vault_schema,
                        rejection_history,
                        language=fallback_lang,
                        inline_source_citations=config.pipeline.inline_source_citations,
                    )
                    try:
                        fallback_num_predict = _concept_draft_num_predict(
                            config,
                            fallback_prompt,
                            (
                                _WRITE_SYSTEM_WITH_CITATIONS
                                if config.pipeline.inline_source_citations
                                else _WRITE_SYSTEM
                            ),
                            heavy_ctx=heavy.ctx,
                        )
                    except ValueError as retry_error:
                        e = retry_error
                        continue
                    try:
                        result = request_structured(
                            client=heavy.client,
                            prompt=fallback_prompt,
                            model_class=SingleArticle,
                            model=heavy.model,
                            system=(
                                _WRITE_SYSTEM_WITH_CITATIONS
                                if config.pipeline.inline_source_citations
                                else _WRITE_SYSTEM
                            ),
                            num_ctx=heavy.ctx,
                            num_predict=fallback_num_predict,
                            stage="compile_article",
                            model_role="heavy",
                            think=heavy.think,
                            options=heavy.options,
                            temperature=heavy.temperature,
                        )
                        resolved_paths = fallback_resolved_paths
                        confidence = _compute_confidence(resolved_paths, db)
                        prompt_degraded = True
                        log.warning(
                            "Compiled '%s' with reduced prompt context; some source or existing "
                            "article text was trimmed. Review the draft and align the loaded "
                            "model context with heavy_ctx if this recurs.",
                            name,
                        )
                        retry_succeeded = True
                        break
                    except (
                        StructuredOutputError,
                        LLMBadRequestError,
                        LLMTruncatedError,
                    ) as retry_error:
                        e = retry_error
                        if not _is_prompt_context_overflow(retry_error):
                            break
            if retry_succeeded:
                pass
            else:
                failure_category = (
                    "context_too_large" if isinstance(e, ValueError) else _categorize_failure(e)
                )
                log.error("Failed to write '%s': %s", name, e)
                _record_failure(name, failure_category)
                db.mark_concept_compile_state(
                    name,
                    resolved_paths or source_paths,
                    "failed",
                    error=_structured_compile_error(failure_category, str(e)),
                )
                continue

        draft_path = _write_draft(
            content_result=result,
            config=config,
            source_paths=resolved_paths,
            db=db,
            confidence=confidence,
            existing_meta=existing_meta,
            existing_titles=link_titles,
            canonical_title=name,
            concept_aliases=db.get_aliases(name),
            alias_map=alias_map,
            prompt_degraded=prompt_degraded,
            run_ulid=run_ulid,
            pipeline=pipeline,
        )
        draft_paths.append(draft_path)
        db.mark_concept_compile_state(name, resolved_paths, "compiled")
        # Track per-article compile run
        if not dry_run and db._has_table("compile_runs"):
            rel = str(draft_path.relative_to(config.vault))
            db.update_article_compile_run(rel, run_ulid)
        elapsed = time.monotonic() - _t_concept
        concept_timings[name] = elapsed
        log.info("Draft written: %s (%.1fs)", draft_path.name, elapsed)

    if failure_categories:
        summary = ", ".join(f"{len(v)} {k}" for k, v in sorted(failure_categories.items()))
        log.warning("Compile failures by category: %s", summary)

    # Finish compile run
    if not dry_run and db._has_table("compile_runs"):
        db.finish_compile_run(run_ulid, article_count=len(draft_paths))

    return draft_paths, failed, concept_timings


# ── Legacy compile (CompilePlan → SingleArticle) ──────────────────────────────


def _plan_prompt(
    source_summary: str,
    existing_titles: list[str],
) -> str:
    titles_str = ", ".join(existing_titles[:50]) if existing_titles else "none yet"
    return (
        f"EXISTING WIKI ARTICLES: {titles_str}\n\n"
        f"SOURCE NOTES TO PROCESS:\n{source_summary}\n\n"
        f"Plan what wiki articles to create or update. Keep scope atomic."
    )


def _write_prompt_legacy(
    article: ArticlePlan,
    sources: str,
    existing_titles: list[str],
    language: str | None = None,
) -> str:
    titles_str = ", ".join(existing_titles[:50]) if existing_titles else "none yet"
    lang_instruction = (
        f"Output language: {language} (ISO 639-1).\n"
        if language
        else "Write in the same language as the source notes.\n"
    )
    return (
        f'Write the wiki article: "{article.title}"\n'
        f"Action: {article.action}\n"
        f"Reasoning: {article.reasoning}\n"
        f"{lang_instruction}"
        f"IMPORTANT: Keep the content field under 800 words. Be concise.\n"
        f"Tags must be lowercase, hyphen-separated, no spaces or special characters. "
        f"Good: machine-learning, quantum-computing. Bad: Machine Learning, C++.\n"
        f"Do NOT use inline hashtags (#tag) in the content body — use [[wikilinks]] only.\n"
        f"Use Obsidian math syntax: inline $...$ and display $$...$$. Do not use \\[...\\].\n\n"
        f"Existing wiki articles to link to: {titles_str}\n\n"
        f"SOURCE MATERIAL:\n{sources}"
    )


def compile_notes(
    config: Config,
    router: ModelRouter,
    db: StateDB,
    rag=None,
    source_paths: list[str] | None = None,
    dry_run: bool = False,
) -> tuple[list[Path], list[str]]:
    """
    Legacy compile: LLM plans articles from source summaries, then writes each one.
    Use compile_concepts() instead for incremental, concept-driven compilation.
    """
    # Resolve source files to compile
    if source_paths is None:
        records = db.list_raw(status="ingested")
        paths = [config.vault / r.path for r in records]
    else:
        paths = [config.vault / sp for sp in source_paths]

    if not paths:
        log.info("No ingested notes to compile")
        return [], []

    fast = router.endpoint("fast")
    heavy = router.endpoint("heavy")

    # Build source summary for planning (use fast model — keep it short)
    summaries = []
    for p in paths:
        try:
            meta, body = parse_note(p)
            short = body[:500].replace("\n", " ")
            summaries.append(f"- {p.name}: {short}")
        except Exception:
            summaries.append(f"- {p.name}: (unreadable)")

    source_summary = "\n".join(summaries)
    existing_titles = [t for t, _ in list_wiki_articles(config.wiki_dir)]

    # ── Step 1: Plan ──────────────────────────────────────────────────────────
    log.info("Planning compilation from %d source notes...", len(paths))
    plan_prompt = _plan_prompt(source_summary, existing_titles)

    try:
        plan: CompilePlan = request_structured(
            client=fast.client,
            prompt=plan_prompt,
            model_class=CompilePlan,
            model=fast.model,
            system=_PLAN_SYSTEM,
            num_ctx=fast.ctx,
            stage="compile_plan",
            model_role="fast",
            think=fast.think,
            options=fast.options,
            temperature=fast.temperature,
        )
    except (StructuredOutputError, LLMBadRequestError, LLMTruncatedError) as e:
        log.error("Planning failed: %s", e)
        return [], ["__planning_failed__"]

    if not plan.articles:
        log.info("Plan produced no articles")
        return [], []

    log.info(
        "Plan: %d articles to %s",
        len(plan.articles),
        "/".join(set(a.action for a in plan.articles)),
    )

    if dry_run:
        for a in plan.articles:
            print(f"  [{a.action}] {a.path} — {a.title}")
            print(f"    sources: {', '.join(a.source_paths)}")
        return [], []

    # ── Step 2: Write each article ────────────────────────────────────────────
    draft_paths: list[Path] = []
    failed: list[str] = []
    failure_categories: dict[str, list[str]] = {}
    source_rel_paths = [str(p.relative_to(config.vault)) for p in paths]

    for article in plan.articles:
        log.info("Writing: %s", article.title)

        relevant = [sp for sp in article.source_paths if sp] or source_rel_paths
        sources_text, resolved_paths = _gather_sources(
            relevant,
            config.vault,
            max_chars=heavy.ctx // 2,
        )

        lang = _resolve_language([str(p) for p in resolved_paths], db, config)
        write_prompt = _write_prompt_legacy(article, sources_text, existing_titles, language=lang)

        try:
            # Legacy compile intentionally follows only article_max_tokens plus remaining
            # context budget. concept_draft_soft_cap applies to the default concept-driven
            # path and does not change --legacy behavior.
            num_predict = _article_num_predict(
                config, write_prompt, _WRITE_SYSTEM, heavy_ctx=heavy.ctx
            )
        except ValueError as e:
            log.error("Failed to write '%s': %s", article.title, e)
            failed.append(article.title)
            failure_categories.setdefault("context_too_large", []).append(article.title)
            continue

        try:
            result: SingleArticle = request_structured(
                client=heavy.client,
                prompt=write_prompt,
                model_class=SingleArticle,
                model=heavy.model,
                system=_WRITE_SYSTEM,
                num_ctx=heavy.ctx,
                num_predict=num_predict,
                stage="compile_article",
                model_role="heavy",
                think=heavy.think,
                options=heavy.options,
                temperature=heavy.temperature,
            )
        except (StructuredOutputError, LLMBadRequestError, LLMTruncatedError) as e:
            log.error("Failed to write '%s': %s", article.title, e)
            failed.append(article.title)
            failure_categories.setdefault(_categorize_failure(e), []).append(article.title)
            continue

        # Preserve existing_meta if updating an existing article
        existing_path = config.wiki_dir / article.path
        existing_meta = None
        if existing_path.exists():
            try:
                existing_meta, _ = parse_note(existing_path)
            except Exception:
                pass

        confidence = _compute_confidence(resolved_paths, db)
        draft_path = _write_draft(
            content_result=result,
            config=config,
            source_paths=resolved_paths,
            db=db,
            confidence=confidence,
            existing_meta=existing_meta,
            existing_titles=existing_titles,
        )
        draft_paths.append(draft_path)
        log.info("Draft written: %s", draft_path.name)

    # Mark source notes as compiled
    if draft_paths:
        for p in paths:
            rel = str(p.relative_to(config.vault))
            db.mark_raw_status(rel, "compiled")

    if failure_categories:
        summary = ", ".join(f"{len(v)} {k}" for k, v in sorted(failure_categories.items()))
        log.warning("Compile failures by category: %s", summary)

    return draft_paths, failed


# ── Approve / Reject ──────────────────────────────────────────────────────────


def approve_drafts(
    config: Config,
    db: StateDB,
    paths: list[Path] | None = None,
    notes: str = "",
    min_confidence: float = 0.0,
) -> list[Path]:
    return publish_drafts(
        config,
        db,
        paths=paths,
        notes=notes,
        min_confidence=min_confidence,
    )


def verify_drafts(
    config: Config,
    db: StateDB,
    paths: list[Path] | None = None,
    notes: str = "",
    min_confidence: float = 0.0,
) -> list[Path]:
    """
    Mark draft(s) verified in place.

    Re-running on an already-verified draft is a no-op so the operation is
    safe to script. Drafts below min_confidence are skipped.
    """
    if paths is None:
        paths = list(config.drafts_dir.rglob("*.md")) if config.drafts_dir.exists() else []

    affected: list[Path] = []
    held_back = 0
    for draft_path in paths:
        draft_path = draft_path.resolve()
        vault_root = config.vault.resolve()
        if not draft_path.exists():
            log.warning("Draft not found: %s", draft_path)
            continue

        try:
            rel_to_drafts = draft_path.relative_to(config.drafts_dir.resolve())
        except ValueError:
            log.warning("Draft is outside wiki/.drafts/: %s", draft_path)
            continue

        published_path = config.wiki_dir / rel_to_drafts
        published_rel = str(published_path.relative_to(vault_root))

        meta, body = parse_note(draft_path)
        if min_confidence > 0.0 and float(meta.get("confidence", 1.0)) < min_confidence:
            held_back += 1
            log.info(
                "Held back (confidence %.2f < %.2f): %s",
                meta.get("confidence"),
                min_confidence,
                draft_path.name,
            )
            continue

        draft_rel = str(draft_path.relative_to(vault_root))
        existing = db.get_article(draft_rel)
        if existing is not None and existing.status == "verified":
            log.debug("Already verified, skipping: %s", draft_path.name)
            continue
        published_existing = db.get_article(published_rel)
        if published_existing is not None and published_existing.status == "published":
            # Publish won the race; never resurrect the draft row or file.
            if draft_path.exists():
                draft_path.unlink()
            continue
        if existing is None:
            db.upsert_article(
                WikiArticleRecord(
                    path=draft_rel,
                    title=str(meta.get("title", draft_path.stem)),
                    sources=(
                        meta.get("sources", []) if isinstance(meta.get("sources"), list) else []
                    ),
                    content_hash=_content_hash(body),
                    status="draft",
                )
            )

        meta["status"] = "verified"
        meta["updated"] = datetime.now().strftime("%Y-%m-%d")
        if isinstance(meta.get("tags"), list):
            meta["tags"] = sanitize_tags([str(t) for t in meta["tags"] if t is not None])
        if (published_existing is None and published_path.exists()) or (
            published_existing is not None and published_existing.status == "published"
        ):
            db.delete_article(draft_rel)
            if draft_path.exists():
                draft_path.unlink()
            continue
        write_note(draft_path, meta, body)
        db.verify_article(draft_rel, notes=notes)
        affected.append(draft_path)
        log.info("Verified: %s", draft_path.name)

    if held_back:
        log.info("Held back %d draft(s) below confidence %.2f", held_back, min_confidence)
    return affected


def publish_drafts(
    config: Config,
    db: StateDB,
    paths: list[Path] | None = None,
    notes: str = "",
    min_confidence: float = 0.0,
) -> list[Path]:
    """
    Move draft(s) from wiki/.drafts/ to wiki/.

    Returns list of published paths. Drafts below min_confidence are skipped
    and remain in .drafts/.
    """
    if paths is None:
        paths = list(config.drafts_dir.rglob("*.md")) if config.drafts_dir.exists() else []

    affected: list[Path] = []
    held_back = 0
    for draft_path in paths:
        draft_path = draft_path.resolve()
        vault_root = config.vault.resolve()
        if not draft_path.exists():
            log.warning("Draft not found: %s", draft_path)
            continue

        try:
            rel_to_drafts = draft_path.relative_to(config.drafts_dir.resolve())
        except ValueError:
            log.warning("Draft is outside wiki/.drafts/: %s", draft_path)
            continue

        meta, body = parse_note(draft_path)
        if min_confidence > 0.0 and float(meta.get("confidence", 1.0)) < min_confidence:
            held_back += 1
            log.info(
                "Held back (confidence %.2f < %.2f): %s",
                meta.get("confidence"),
                min_confidence,
                draft_path.name,
            )
            continue

        target = config.wiki_dir / rel_to_drafts
        target.parent.mkdir(parents=True, exist_ok=True)
        meta["status"] = "published"
        meta["updated"] = datetime.now().strftime("%Y-%m-%d")
        if isinstance(meta.get("tags"), list):
            meta["tags"] = sanitize_tags([str(t) for t in meta["tags"] if t is not None])
        body = _strip_draft_annotations(body)
        write_note(target, meta, body)  # write to destination first

        # Update state DB before removing the draft — if the process crashes between
        # these two steps the draft is a dangling orphan but the DB is consistent.
        target_rel = str(target.resolve().relative_to(vault_root))
        draft_rel = str(draft_path.relative_to(vault_root))
        db.publish_article(draft_rel, target_rel)

        art = db.get_article(target_rel)
        if art is None:
            art = WikiArticleRecord(
                path=target_rel,
                title=str(meta.get("title", target.stem)),
                sources=meta.get("sources", []) if isinstance(meta.get("sources"), list) else [],
                content_hash="",
                status="published",
            )
        if art:
            try:
                _, pub_body = parse_note(target)
                db.upsert_article(
                    WikiArticleRecord(
                        path=target_rel,
                        title=art.title,
                        sources=art.sources,
                        content_hash=_content_hash(pub_body),
                        created_at=art.created_at,
                        updated_at=art.updated_at,
                        status="published",
                        # Carry the verify-time audit fields forward so the
                        # upsert does not blank them and force approve_article
                        # to re-stamp with the publish timestamp.
                        approved_at=art.approved_at,
                        approval_notes=art.approval_notes,
                    )
                )
            except Exception:
                pass
            db.approve_article(target_rel, notes=notes)

        draft_path.unlink()  # remove draft only after DB is consistent
        affected.append(target)
        log.info("Published: %s", target.name)

    if held_back:
        log.info("Held back %d draft(s) below confidence %.2f", held_back, min_confidence)
    return affected


def reject_draft(
    draft_path: Path,
    config: Config,
    db: StateDB,
    feedback: str = "",
) -> None:
    """Delete a draft, store rejection feedback and body for future recompiles."""
    # Resolve to canonical path so relative_to(config.vault) works on macOS
    # where /var is a symlink to /private/var but config.vault is always resolved.
    draft_path = draft_path.resolve()
    # Read before deleting — title and body needed for rejection record
    title = draft_path.stem
    draft_body = ""
    if draft_path.exists():
        try:
            meta, draft_body = parse_note(draft_path)
            title = meta.get("title", draft_path.stem)
        except Exception:
            pass

    try:
        draft_rel = str(draft_path.relative_to(config.vault.resolve()))
    except ValueError:
        log.warning("Draft is outside vault: %s", draft_path)
        return
    article_record = db.get_article(draft_rel)
    db.delete_article(draft_rel)
    if draft_path.exists():
        draft_path.unlink()

    source_paths = article_record.sources if article_record is not None else []
    if source_paths:
        db.mark_concept_compile_state(title, source_paths, "pending")

    if feedback:
        db.add_rejection(title, feedback, body=draft_body)
        count = db.rejection_count(title)
        if count >= StateDB._REJECTION_CAP:
            log.warning(
                "Concept '%s' blocked after %d rejections — use `synto unblock` to re-enable",
                title,
                count,
            )
        else:
            log.info("Draft rejected with feedback: %s", feedback)
