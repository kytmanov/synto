"""
Query pipeline: index-based routing → grounded answer.

Flow:
  1. Read wiki/index.md (no embeddings — index is the routing layer)
  2. Fast model selects relevant pages (PageSelection)
  3. Load page content (up to MAX_PAGES, MAX_CHARS_PER_PAGE each)
  4. Heavy model generates answer (QueryAnswer)
  5. Optionally save to wiki/queries/
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import frontmatter

from ..config import Config
from ..engines import QueryConfig, QueryEngine
from ..indexer import append_log, generate_index
from ..markdown_math import sanitize_obsidian_math
from ..metrics import AppEvent, emit_app_event
from ..models import PageSelection, QueryAnswer, WikiArticleRecord
from ..readers import VaultReader
from ..sanitize import clean_display_name
from ..state import (
    DuplicateArticlePathError,
    DuplicateSynthesisQuestionHashError,
    StateDB,
    SynthesisInsertConflictError,
)
from ..structured_output import request_structured
from ..vault import (
    atomic_write,
    list_wiki_articles,
    next_available_path,
    parse_note,
    sanitize_filename,
    write_note,
)

if TYPE_CHECKING:
    from ..client_factory import ModelRouter, RoleEndpoint

MAX_PAGES = 5
MAX_CHARS_PER_PAGE = 8_000
_MIN_ALIAS_LEN = 3  # filters stop-noise ("in", "of", "to")
_MAX_BRIDGE_MATCHES = 10  # caps routing-hint token cost
_GRAPH_EXPANSION_MIN_CONFIDENCE = 0.7  # relations below this are too noisy to widen context
_GRAPH_EXPANSION_MAX_EXTRAS = 2  # total extra pages across all selected pages, not per page

log = logging.getLogger(__name__)


_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(#[^\]|]*)?(?:\|([^\]]*))?\]\]")


def _expand_query(
    question: str,
    alias_map: dict[str, list[str]],
    known_titles: set[str] | None = None,
) -> str:
    """Augment the routing question with concept names whose aliases appear as
    whole words in the question.

    `known_titles` filters to concepts with a published article — avoids
    hinting the LLM toward titles the index doesn't contain.

    Word-boundary matching uses Python's \\b; silently no-ops on CJK/Thai
    (no inter-word spaces). Documented limitation for v1.

    Hint truncation preserves order of first appearance in the question —
    not alphabetical — so the cap drops late-mentioned concepts, not Z-named
    ones.

    All-caps surfaces (e.g., "AI", "ML") bypass the length floor. Language-
    agnostic: str.isupper() returns False for scripts without casing (CJK),
    so the floor still applies there.

    Concept names themselves are NOT matchable surfaces — the LLM already
    sees them in the index. Only aliases bridge vocabulary.

    Pure function: no DB, no LLM, no state mutation.
    """
    if not question or not alias_map:
        return question

    surface_lookup: dict[str, set[str]] = {}
    for concept_name, aliases in alias_map.items():
        if known_titles is not None and concept_name not in known_titles:
            continue
        for surface in aliases:
            if surface.isupper() and len(surface) >= 2:
                pass  # all-caps acronyms bypass the length floor
            elif len(surface) < _MIN_ALIAS_LEN:
                continue
            surface_lookup.setdefault(surface.lower(), set()).add(concept_name)

    if not surface_lookup:
        return question

    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(s) for s in surface_lookup) + r")\b",
        re.IGNORECASE,
    )
    # dict preserves insertion order — truncation by occurrence, not alphabet.
    matched_ordered: dict[str, None] = {}
    for m in pattern.finditer(question):
        for concept in surface_lookup.get(m.group(0).lower(), ()):
            matched_ordered.setdefault(concept, None)

    if not matched_ordered:
        return question

    related = ", ".join(list(matched_ordered)[:_MAX_BRIDGE_MATCHES])
    return f"{question}\n\n(Routing hint — related wiki concepts: {related})"


@dataclass
class QuerySaveResult:
    path: Path
    resolution: str = "saved_new"
    duplicate_detected: bool = False
    file_written: bool = True
    error: str | None = None


@dataclass
class QueryRunResult:
    answer: str
    selected_pages: list[str]
    synthesis: QuerySaveResult | None = None
    query_save: QuerySaveResult | None = None

    def __iter__(self):
        yield self.answer
        yield self.selected_pages


class SynthesisSaveError(ValueError):
    resolution = "save_failed"

    def __init__(
        self,
        message: str,
        *,
        path: Path | None = None,
        duplicate_detected: bool = False,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.duplicate_detected = duplicate_detected


class SynthesisChainError(SynthesisSaveError):
    resolution = "rejected_synthesis_chain"


class SynthesisManualEditConflictError(SynthesisSaveError):
    resolution = "manual_edit_conflict"


class SynthesisPathAllocationError(SynthesisSaveError):
    resolution = "save_failed"


@dataclass
class _QueryCoreResult:
    answer: str
    selected_pages: list[str]
    title: str | None = None
    index_found: bool = True


# ── Internal helpers ──────────────────────────────────────────────────────────


def _load_index(config: Config) -> str:
    index_path = config.wiki_dir / "index.md"
    if not index_path.exists():
        return ""
    return index_path.read_text(encoding="utf-8")


def _find_page(config: Config, title: str, db: StateDB | None = None) -> Path | None:
    """Resolve a title to a file path with concept > source > synthesis precedence."""

    def priority(path: Path) -> tuple[int, str]:
        rel = path.relative_to(config.vault)
        rel_text = str(rel)
        if "sources" in rel.parts:
            return (1, rel_text.casefold())
        if "synthesis" in rel.parts:
            return (2, rel_text.casefold())
        return (0, rel_text.casefold())

    if title.lower().startswith("sources/"):
        source_title = title.split("/", 1)[1]
        candidate = config.wiki_dir / f"{title}.md"
        if candidate.exists():
            return candidate
        candidate = config.sources_dir / f"{source_title}.md"
        if candidate.exists():
            return candidate
    # Exact filename match (wiki root)
    candidate = config.wiki_dir / f"{title}.md"
    if candidate.exists():
        return candidate
    # Exact filename match (sources/)
    candidate2 = config.sources_dir / f"{title}.md"
    if candidate2.exists():
        return candidate2
    # Frontmatter title scan (case-insensitive fallback)
    matches: list[Path] = []
    for md in config.wiki_dir.rglob("*.md"):
        if ".drafts" in md.parts:
            continue
        try:
            meta, _ = parse_note(md)
            if meta.get("title", "").lower() == title.lower():
                matches.append(md)
        except Exception:
            pass
    if matches:
        return sorted(matches, key=priority)[0]
    # Alias resolution fallback
    if db is not None:
        canonical = db.resolve_alias(title)
        if canonical is not None:
            return _find_page(config, canonical, db=None)
    return None


def _load_pages(
    config: Config,
    page_titles: list[str],
    db: StateDB | None = None,
    *,
    max_pages: int = MAX_PAGES,
) -> str:
    """Return concatenated content of selected pages."""
    parts: list[str] = []
    for title in page_titles[:max_pages]:
        page = _find_page(config, title, db=db)
        if page is None:
            continue
        try:
            meta, body = parse_note(page)
            page_title = meta.get("title", title)
            parts.append(f"# {page_title}\n\n{body[:MAX_CHARS_PER_PAGE]}")
        except Exception:
            pass
    return "\n\n---\n\n".join(parts)


def _derive_synthesis_title(question: str, model_title: str | None) -> str:
    # Drop dangling punctuation so a model title like "Foo)" doesn't become Foo).md.
    candidate = clean_display_name(model_title or "")
    if candidate and len(candidate.split()) <= 12 and sanitize_filename(candidate) != "untitled":
        return candidate

    normalized = unicodedata.normalize("NFKC", question).strip()
    normalized = normalized.rstrip("?").strip()
    normalized = re.sub(r"\s+", " ", normalized)
    fallback = " ".join(normalized.split()[:8]).title()
    return fallback or "untitled-synthesis"


def _strip_unknown_wikilinks(content: str, known_titles: list[str]) -> str:
    """Unwrap wikilinks that do not target an existing wiki/source page."""
    known = {title.casefold() for title in known_titles}

    def replace(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        fragment = match.group(2) or ""
        display = match.group(3)
        if target.casefold().startswith("sources/") or target.casefold() in known:
            return match.group(0)
        return display or f"{target}{fragment}"

    return _WIKILINK_RE.sub(replace, content)


def _sanitize_query_answer(answer: str, source_pages: list[str], known_titles: list[str]) -> str:
    """Strip invented wikilinks from query answers before returning or saving."""
    allowed_titles = [*known_titles, *source_pages]
    sanitized = answer.replace("\\r\\n", "\n").replace("\\n", "\n")
    return _strip_unknown_wikilinks(sanitize_obsidian_math(sanitized), allowed_titles)


def _body_hash(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _normalize_question(question: str) -> str:
    normalized = unicodedata.normalize("NFKC", question).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized.endswith("?"):
        normalized = normalized[:-1].rstrip()
    return normalized


def _question_hash(question: str) -> str:
    return hashlib.sha256(_normalize_question(question).encode("utf-8")).hexdigest()[:16]


def find_existing_synthesis(db: StateDB, question: str) -> WikiArticleRecord | None:
    return db.find_synthesis_by_question_hash(_question_hash(question))


def _resolve_source_paths(config: Config, source_pages: list[str], db: StateDB) -> list[Path]:
    resolved: list[Path] = []
    for title in source_pages:
        page = _find_page(config, title, db=db)
        if page is not None:
            resolved.append(page)
    return resolved


def _source_hashes(config: Config, source_paths: list[Path]) -> list[dict[str, str]]:
    hashes: list[dict[str, str]] = []
    for path in source_paths:
        _, body = parse_note(path)
        hashes.append(
            {
                "path": str(path.relative_to(config.vault)),
                "hash": _body_hash(body),
            }
        )
    return hashes


def _is_synthesis_source(config: Config, db: StateDB, path: Path) -> bool:
    rel_path = str(path.relative_to(config.vault))
    article = db.get_article(rel_path)
    if article is not None and article.kind == "synthesis":
        return True
    try:
        meta, _ = parse_note(path)
    except Exception:
        return False
    tags = meta.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    return any(str(tag).casefold() == "synthesis" for tag in tags)


def _render_synthesis_body(answer: str, source_pages: list[str]) -> str:
    source_lines = ["## Sources", ""]
    if source_pages:
        source_lines.extend(f"- [[{page}]]" for page in source_pages)
    else:
        source_lines.append("- No source pages were selected.")
    return f"{answer.rstrip()}\n\n" + "\n".join(source_lines)


def _build_synthesis_file_text(
    body: str,
    *,
    title: str,
    question: str,
    source_pages: list[str],
    source_page_hashes: list[dict[str, str]],
    question_hash: str,
    content_hash: str,
    created: str,
) -> str:
    meta = {
        "title": title,
        "tags": ["synthesis"],
        "kind": "synthesis",
        "source_question": question,
        "source_pages": source_pages,
        "source_page_hashes": source_page_hashes,
        "question_hash": question_hash,
        "content_hash": content_hash,
        "created": created,
        "status": "published",
        "source_count": len(source_pages),
        "single_source": len(set(source_pages)) == 1,
    }
    return frontmatter.dumps(frontmatter.Post(body, **meta))


def _reserved_synthesis_names(config: Config, db: StateDB) -> set[str]:
    synthesis_parent = config.synthesis_dir.relative_to(config.vault)
    return {
        Path(article.path).name
        for article in db.list_articles()
        if Path(article.path).parent == synthesis_parent
    }


def _save_synthesis_new(
    config: Config,
    db: StateDB,
    *,
    base_path: Path,
    title: str,
    body: str,
    file_text: str,
    content_hash: str,
    question: str,
    source_pages: list[str],
    question_hash: str | None,
    source_paths: list[str],
    source_page_hashes: list[dict[str, str]],
    duplicate_strategy: str,
    duplicate_detected: bool,
    created_at: datetime,
) -> QuerySaveResult:
    reserved_names = _reserved_synthesis_names(config, db)
    attempts = 0

    while attempts < 16:
        path = next_available_path(base_path, reserved_names=reserved_names)
        relative_path = str(path.relative_to(config.vault))
        record = WikiArticleRecord(
            path=relative_path,
            title=title,
            sources=[],
            content_hash=content_hash,
            created_at=created_at,
            updated_at=datetime.now(),
            status="published",
            kind="synthesis",
            question_hash=question_hash,
            synthesis_sources=source_paths,
            synthesis_source_hashes=[[item["path"], item["hash"]] for item in source_page_hashes],
        )
        try:
            with db._tx():
                db.insert_synthesis_atomic(record)
                atomic_write(path, file_text)
            resolution = (
                "saved_new"
                if not duplicate_detected and question_hash is not None and path == base_path
                else "saved_with_suffix"
            )
            return QuerySaveResult(
                path=path,
                resolution=resolution,
                duplicate_detected=duplicate_detected,
            )
        except DuplicateArticlePathError:
            attempts += 1
            reserved_names.add(path.name)
            continue
        except DuplicateSynthesisQuestionHashError:
            if question_hash is None:
                raise RuntimeError("synthesis question hash conflict without question hash")

            existing = db.find_synthesis_by_question_hash(question_hash)
            if existing is None:
                raise RuntimeError("duplicate synthesis detected without existing row")

            if duplicate_strategy == "keep_existing":
                return QuerySaveResult(
                    path=config.vault / existing.path,
                    resolution="kept_existing",
                    duplicate_detected=True,
                    file_written=False,
                )
            if duplicate_strategy == "update_in_place":
                return _update_existing_synthesis(
                    config,
                    db,
                    existing=existing,
                    title=title,
                    question=question,
                    answer_body=body,
                    source_pages=source_pages,
                    source_paths=source_paths,
                    source_page_hashes=source_page_hashes,
                    duplicate_detected=True,
                )

            duplicate_detected = True
            question_hash = None
            attempts += 1

    raise SynthesisPathAllocationError(
        f"Could not allocate a unique synthesis path for {base_path.name}.",
        path=base_path,
        duplicate_detected=duplicate_detected,
    )


def _update_existing_synthesis(
    config: Config,
    db: StateDB,
    *,
    existing,
    title: str,
    question: str,
    answer_body: str,
    source_pages: list[str],
    source_paths: list[str],
    source_page_hashes: list[dict[str, str]],
    duplicate_detected: bool,
) -> QuerySaveResult:
    path = config.vault / existing.path
    created_text = existing.created_at.strftime("%Y-%m-%d")
    if path.exists():
        try:
            meta, existing_body = parse_note(path)
        except Exception as exc:
            raise SynthesisManualEditConflictError(
                f"Existing synthesis at {path} could not be parsed safely.",
                path=path,
                duplicate_detected=duplicate_detected,
            ) from exc
        # A blank content_hash is a "not yet hashed" placeholder, never a manual edit (an empty
        # body still hashes to a non-empty digest) — treat it as regenerable, matching compile
        # and lint's manual-edit guards (#83).
        if existing.content_hash and existing.content_hash != _body_hash(existing_body):
            raise SynthesisManualEditConflictError(
                f"Existing synthesis at {path} was manually edited; refusing to overwrite.",
                path=path,
                duplicate_detected=duplicate_detected,
            )
        # Preserve a manually adjusted frontmatter created date on in-place updates.
        created_text = str(meta.get("created") or created_text)

    content_hash = _body_hash(answer_body)
    file_text = _build_synthesis_file_text(
        answer_body,
        title=title,
        question=question,
        source_pages=source_pages,
        source_page_hashes=source_page_hashes,
        question_hash=existing.question_hash or _question_hash(question),
        content_hash=content_hash,
        created=created_text,
    )
    record = WikiArticleRecord(
        path=existing.path,
        title=title,
        sources=[],
        content_hash=content_hash,
        created_at=existing.created_at,
        updated_at=datetime.now(),
        status="published",
        kind="synthesis",
        question_hash=existing.question_hash or _question_hash(question),
        synthesis_sources=source_paths,
        synthesis_source_hashes=[[item["path"], item["hash"]] for item in source_page_hashes],
    )
    with db._tx():
        db._upsert_article_row(record)
        atomic_write(path, file_text)
    return QuerySaveResult(
        path=path,
        resolution="updated_in_place",
        duplicate_detected=duplicate_detected,
    )


def _save_synthesis(
    config: Config,
    db: StateDB,
    question: str,
    answer: str,
    source_pages: list[str],
    title: str,
    duplicate_strategy: str = "keep_existing",
) -> QuerySaveResult:
    config.synthesis_dir.mkdir(parents=True, exist_ok=True)

    question_hash = _question_hash(question)
    existing = find_existing_synthesis(db, question)
    if existing is not None:
        if duplicate_strategy == "keep_existing":
            return QuerySaveResult(
                path=config.vault / existing.path,
                resolution="kept_existing",
                duplicate_detected=True,
                file_written=False,
            )

    resolved_sources = _resolve_source_paths(config, source_pages, db)
    if any(_is_synthesis_source(config, db, path) for path in resolved_sources):
        raise SynthesisChainError("Synthesis sources cannot include another synthesis page")

    source_paths = [str(path.relative_to(config.vault)) for path in resolved_sources]
    source_page_hashes = _source_hashes(config, resolved_sources)
    body = _render_synthesis_body(answer, source_pages)
    content_hash = _body_hash(body)
    base_path = config.synthesis_dir / f"{sanitize_filename(title)}.md"
    duplicate_detected = existing is not None
    if existing is not None and duplicate_strategy == "update_in_place":
        result = _update_existing_synthesis(
            config,
            db,
            existing=existing,
            title=title,
            question=question,
            answer_body=body,
            source_pages=source_pages,
            source_paths=source_paths,
            source_page_hashes=source_page_hashes,
            duplicate_detected=True,
        )
    else:
        created_text = datetime.now().strftime("%Y-%m-%d")
        file_text = _build_synthesis_file_text(
            body,
            title=title,
            question=question,
            source_pages=source_pages,
            source_page_hashes=source_page_hashes,
            question_hash=question_hash,
            content_hash=content_hash,
            created=created_text,
        )
        result = _save_synthesis_new(
            config,
            db,
            base_path=base_path,
            title=title,
            body=body,
            file_text=file_text,
            content_hash=content_hash,
            question=question,
            source_pages=source_pages,
            question_hash=(
                None
                if duplicate_detected and duplicate_strategy == "save_with_suffix"
                else question_hash
            ),
            source_paths=source_paths,
            source_page_hashes=source_page_hashes,
            duplicate_strategy=duplicate_strategy,
            duplicate_detected=duplicate_detected,
            created_at=datetime.now(),
        )

    try:
        generate_index(config, db)
        append_log(config, f"query synthesize | {question[:60]}")
    except Exception as exc:
        log.warning("query synthesize post-save maintenance failed: %s", exc)
    return result


def _emit_synthesis_event(question: str, source_pages: list[str], result: QuerySaveResult) -> None:
    emit_app_event(
        AppEvent(
            name="query_synthesize",
            payload={
                "question_hash": _question_hash(question),
                "resolution": result.resolution,
                "file_written": result.file_written,
                "path": str(result.path) if result.path else None,
                "duplicate_detected": result.duplicate_detected,
                "source_page_count": len(source_pages),
                "error": result.error,
            },
        )
    )


def _query_core(
    config: Config,
    fast_ep: RoleEndpoint,
    heavy_ep: RoleEndpoint,
    db: StateDB | None,
    question: str,
    *,
    max_pages: int = MAX_PAGES,
    graph_hops: int = 0,
) -> _QueryCoreResult:
    index_content = _load_index(config)
    if not index_content:
        return _QueryCoreResult(
            answer="No wiki index found. Run `synto ingest` and `synto compile` first.",
            selected_pages=[],
            index_found=False,
        )

    all_articles = list_wiki_articles(config.wiki_dir)
    # Expansion filter must see ALL published titles — capping at 80 here would
    # silently hide concepts past that rank from the routing hint.
    expansion_titles_set = {title for title, _ in all_articles}
    # Answer-prompt wikilink whitelist is token-budgeted; the 80-cap stays here.
    known_title_list = [title for title, _ in all_articles[:80]]
    known_titles = ", ".join(known_title_list)

    alias_map = db.load_concept_alias_map() if db is not None else {}
    routing_question = _expand_query(question, alias_map, known_titles=expansion_titles_set)
    if routing_question != question:
        # Stable wire format for smoke + debugging. Parsed by scripts/smoke_test.sh.
        hinted = (
            routing_question.split("(Routing hint — related wiki concepts: ", 1)[1]
            .rstrip()
            .rstrip(")")
        )
        log.info("query.routing_hint matched_concepts=%s", hinted)

    selection_prompt = (
        "You are a routing agent for a personal knowledge wiki.\n\n"
        f"Wiki index:\n{index_content}\n\n"
        f"User question: {routing_question}\n\n"
        "Synthesis pages capture prior answers; consider them when relevant "
        "but never prefer them over a fresh concept page. "
        f"Select 1-{max_pages} page titles from the index that are most relevant "
        "to answer this question. "
        'Return JSON: {"pages": ["Title 1", "Title 2"]}'
    )
    selection = request_structured(
        client=fast_ep.client,
        prompt=selection_prompt,
        model_class=PageSelection,
        model=fast_ep.model,
        num_ctx=fast_ep.ctx,
        max_retries=2,
        stage="query_select",
        model_role="fast",
        think=fast_ep.think,
        options=fast_ep.options,
        temperature=fast_ep.temperature,
    )

    extras: list[str] = []
    if graph_hops > 0 and db is not None and db.count_relations() > 0:
        # Neighbors are concept names; resolve against actual article titles (not the
        # 80-capped known_title_list) so a match is guaranteed loadable by _find_page.
        title_by_casefold = {title.casefold(): title for title, _ in all_articles}
        selected_casefold = {page.casefold() for page in selection.pages}
        for page in selection.pages:
            if len(extras) >= _GRAPH_EXPANSION_MAX_EXTRAS:
                break
            for neighbor in db.list_relation_neighbors(
                page, min_confidence=_GRAPH_EXPANSION_MIN_CONFIDENCE
            ):
                if len(extras) >= _GRAPH_EXPANSION_MAX_EXTRAS:
                    break
                neighbor_casefold = neighbor.casefold()
                if neighbor_casefold in selected_casefold:
                    continue
                resolved_title = title_by_casefold.get(neighbor_casefold)
                if resolved_title is None:
                    continue
                extras.append(resolved_title)
                selected_casefold.add(neighbor_casefold)

    pages = [*selection.pages, *extras]
    context = _load_pages(config, pages, db=db, max_pages=max_pages + len(extras))
    if not context:
        context = "(No matching wiki pages found.)"
    answer_prompt = (
        "You are answering a question using a personal knowledge wiki.\n\n"
        f"Relevant wiki content:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using the wiki content. Use [[wikilinks]] only for existing wiki pages from "
        f"this list: {known_titles}. Do not create links for terms missing from that list. "
        "Use Obsidian math syntax: inline $...$ and display $$...$$. Do not use \\[...\\]. "
        "Answer in the same language as the user's question. "
        "Also provide a short topic title for the answer subject. "
        'Return JSON: {"answer": "your full markdown answer here", "title": "short title"}'
    )
    result = request_structured(
        client=heavy_ep.client,
        prompt=answer_prompt,
        model_class=QueryAnswer,
        model=heavy_ep.model,
        num_ctx=heavy_ep.ctx,
        max_retries=2,
        stage="query_answer",
        model_role="heavy",
        think=heavy_ep.think,
        options=heavy_ep.options,
        temperature=heavy_ep.temperature,
    )

    sanitized_answer = _sanitize_query_answer(result.answer, pages, known_title_list)
    return _QueryCoreResult(
        answer=sanitized_answer,
        selected_pages=pages,
        title=result.title,
    )


# ── Public API ────────────────────────────────────────────────────────────────


def run_query(
    config: Config,
    router: ModelRouter,
    db: StateDB,
    question: str,
    save: bool = False,
    synthesize: bool = False,
    duplicate_strategy: str = "keep_existing",
) -> QueryRunResult:
    """
    Run a query against the wiki.
    Returns answer, selected pages, and any save metadata.
    """
    engine = QueryEngine(
        reader=VaultReader(config.vault),
        fast_ep=router.endpoint("fast"),
        heavy_ep=router.endpoint("heavy"),
        config=config,
        db=db,
        query_config=QueryConfig(max_pages=MAX_PAGES),
    )
    answer = engine.query(question)
    selected_pages = list(engine.last_selected_pages)

    if not engine.last_index_found:
        return QueryRunResult(
            answer=answer.text,
            selected_pages=[],
        )

    sanitized_answer = answer.text
    synthesis_title = _derive_synthesis_title(question, answer.title)

    query_save = None
    if save:
        query_path = _save_query(config, db, question, sanitized_answer, selected_pages)
        query_save = QuerySaveResult(path=query_path, resolution="saved_new")

    synthesis_save = None
    if synthesize:
        try:
            synthesis_save = _save_synthesis(
                config,
                db,
                question,
                sanitized_answer,
                selected_pages,
                synthesis_title,
                duplicate_strategy,
            )
        except SynthesisSaveError as exc:
            synthesis_save = QuerySaveResult(
                path=exc.path or config.synthesis_dir / f"{sanitize_filename(synthesis_title)}.md",
                resolution=exc.resolution,
                duplicate_detected=exc.duplicate_detected,
                file_written=False,
                error=str(exc),
            )
            _emit_synthesis_event(question, selected_pages, synthesis_save)
            raise
        except SynthesisInsertConflictError as exc:
            synthesis_save = QuerySaveResult(
                path=config.synthesis_dir / f"{sanitize_filename(synthesis_title)}.md",
                resolution="save_failed",
                duplicate_detected=False,
                file_written=False,
                error=str(exc),
            )
            _emit_synthesis_event(question, selected_pages, synthesis_save)
            raise
        except Exception as exc:
            synthesis_save = QuerySaveResult(
                path=config.synthesis_dir / f"{sanitize_filename(synthesis_title)}.md",
                resolution="save_failed",
                duplicate_detected=False,
                file_written=False,
                error=str(exc),
            )
            _emit_synthesis_event(question, selected_pages, synthesis_save)
            raise
        _emit_synthesis_event(question, selected_pages, synthesis_save)

    return QueryRunResult(
        answer=sanitized_answer,
        selected_pages=selected_pages,
        synthesis=synthesis_save,
        query_save=query_save,
    )


def _save_query(
    config: Config,
    db: StateDB,
    question: str,
    answer: str,
    source_pages: list[str],
) -> Path:
    """Write answer to wiki/queries/, update index + log."""
    config.queries_dir.mkdir(parents=True, exist_ok=True)

    slug = sanitize_filename(question[:60])
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{date_str}-{slug}.md"
    path = config.queries_dir / filename

    meta = {
        "title": question[:80],
        "tags": ["query"],
        "source_pages": source_pages,
        "created": date_str,
        "status": "published",
    }
    write_note(path, meta, answer)
    append_log(config, f"query | {question[:60]}")
    generate_index(config, db)
    return path
