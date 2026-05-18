"""
Ingest pipeline: raw note → chunk → analyze → embed → update state.

Uses fast model (gemma4:e4b) for analysis.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel as _BaseModel

from ..config import Config
from ..models import AnalysisResult, Concept, RawNoteRecord, SourceSegment, TermExtractionResult
from ..protocols import LLMClientProtocol
from ..state import StateDB
from ..structured_output import request_structured
from ..vault import (
    chunk_text,
    generate_aliases,
    parse_note,
    sanitize_filename,
    sanitize_wikilink_target,
    write_note,
)
from .items import extract_named_reference_items, extract_quoted_title_items, store_extracted_items
from .prompts import load_prompt

log = logging.getLogger(__name__)

INGEST_ANALYSIS_PROMPT_VERSION = "analysis-v2-language-policy"

# Kept for backward compatibility (imported by tests).  Actual system prompts live in
# pipeline/prompts/ and are loaded per source_type via load_prompt().
_SYSTEM = load_prompt("notes")


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _PromptConceptContext:
    canonical: str
    aliases: tuple[str, ...] = ()


def _ingest_prompt_version(config: Config) -> str:
    language = config.pipeline.language or "auto"
    return f"{INGEST_ANALYSIS_PROMPT_VERSION}|language={language}"


def _canonical_prompt_contexts(
    canonical_names: list[str], alias_map: dict[str, str] | None = None
) -> list[_PromptConceptContext]:
    alias_map = alias_map or {}
    aliases_by_canonical: dict[str, list[str]] = {name: [] for name in canonical_names}
    for alias, canonical in sorted(alias_map.items()):
        if canonical not in aliases_by_canonical:
            continue
        cleaned_alias = _clean_concept_text(alias)
        if cleaned_alias and cleaned_alias not in aliases_by_canonical[canonical]:
            aliases_by_canonical[canonical].append(cleaned_alias)
    return [
        _PromptConceptContext(canonical=name, aliases=tuple(aliases_by_canonical.get(name, [])))
        for name in canonical_names
    ]


def _prompt_context_names(contexts: list[_PromptConceptContext]) -> list[str]:
    return [ctx.canonical for ctx in contexts]


def _format_prompt_concept_hint(context: _PromptConceptContext) -> str:
    if not context.aliases:
        return context.canonical
    aliases = ", ".join(context.aliases[:3])
    return f"{context.canonical} (aliases: {aliases})"


def _prompt_context_matches_body(
    context: _PromptConceptContext,
    body: str,
    path_name: str = "",
) -> bool:
    if _has_title_or_body_evidence(context.canonical, body, path_name):
        return True
    return any(_has_title_or_body_evidence(alias, body, path_name) for alias in context.aliases)


def _checkpoint_hash(
    content_hash: str,
    config: Config,
    prompt_contexts: list[_PromptConceptContext],
) -> str:
    payload = {
        "content_hash": content_hash,
        "prompt_version": _ingest_prompt_version(config),
        "fast_model": config.models.fast,
        "contexts": [
            {"canonical": ctx.canonical, "aliases": list(ctx.aliases)} for ctx in prompt_contexts
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_analysis_prompt(
    body: str,
    existing_concepts: list[str],
    path_name: str = "",
    chunk_label: str = "",
    language: str | None = None,
    prompt_concepts: list[_PromptConceptContext] | None = None,
) -> str:
    concepts_hint = "none yet"
    if prompt_concepts:
        matched = [
            context
            for context in prompt_concepts
            if _prompt_context_matches_body(context, body, path_name)
        ]
        unmatched = [context for context in prompt_concepts if context not in matched]
        concepts_hint = ", ".join(
            _format_prompt_concept_hint(context) for context in [*matched, *unmatched][:30]
        )
    elif existing_concepts:
        matched = [
            concept
            for concept in existing_concepts
            if _has_title_or_body_evidence(concept, body, path_name)
        ]
        unmatched = [concept for concept in existing_concepts if concept not in matched]
        concepts_hint = ", ".join([*matched, *unmatched][:30])
    label = f" {chunk_label}" if chunk_label else ""

    if language:
        # Names follow the configured language; existing canonicals are always kept by their
        # exact listed name so _normalize_concepts() can match them. Aliases are intentionally
        # multi-lingual: the model should include the note's native surface form so the evidence
        # filter can find body evidence even when the concept name is a translation.
        lang_instruction = (
            f"Output concept names in {language} (ISO 639-1). "
            f"Exception: for concepts already in the wiki (listed above), always use the exact "
            f"canonical name shown — do not translate it.\n\n"
            f"Aliases may be in any language. Include short forms and abbreviations used in the "
            f"note text. If the concept name is a translation of how it appears in the note text, "
            f"include the note's surface form as the first alias.\n\n"
        )
    else:
        # Use whatever surface form the note itself uses. Scope "international form" to pure
        # acronyms/initialisms only — methodology and domain terms have native-language forms
        # and must not be forced into any specific language.
        lang_instruction = (
            "Use concept names in the form they appear in the note text, unless reusing an "
            "existing wiki concept exactly. Do not translate concept names on your own.\n\n"
            "Do not create separate concepts for the same topic in different languages. "
            "Only keep an internationally-fixed form when the concept is a pure acronym or "
            "initialism with no natural native spelling (e.g. 'API', 'HTTP', 'JSON', 'PDF'). "
            "Methodology and domain terms such as 'Scrum' or 'Extreme Programming' are not in "
            "this category — use the form from the note.\n\n"
        )

    return (
        f"Analyze this note{label} and extract structured metadata.\n\n"
        f"Existing wiki concepts (reuse these names where applicable): {concepts_hint}\n\n"
        f"{lang_instruction}"
        f"For each concept, provide 3-5 short surface forms used in running text "
        f"(abbreviations, short names). Example: name='Program Counter (PC)', "
        f"aliases=['PC', 'program counter']. Use empty list if no natural aliases exist.\n\n"
        f"Also return named_references: exact named references copied from the note that "
        f"may be useful later but may not deserve concept articles: people, organizations, "
        f"products, events, works, named projects. Do not translate. Do not infer. "
        f"Do not include broad topics or concepts. Max 8.\n\n"
        f"NOTE CONTENT:\n{body}"
    )


def _merge_chunk_results(results: list[AnalysisResult]) -> AnalysisResult:
    """Merge AnalysisResults from multiple chunks into one.

    Concepts and topics: union (deduplicated, insertion order preserved).
    Aliases for the same concept are merged across chunks.
    Summary: first chunk's (intro is most representative).
    Quality: minimum across chunks (conservative).
    """
    if len(results) == 1:
        return results[0]

    # Dedup concepts by canonical name (case-insensitive), merge aliases
    seen: dict[str, list[str]] = {}  # lower(name) -> accumulated aliases
    order: list[str] = []  # canonical names in insertion order
    canonical_by_lower: dict[str, str] = {}

    for r in results:
        for c in r.concepts:
            key = c.name.lower()
            if key not in seen:
                seen[key] = list(c.aliases)
                order.append(key)
                canonical_by_lower[key] = c.name
            else:
                existing_lower = {a.lower() for a in seen[key]}
                for a in c.aliases:
                    if a.lower() not in existing_lower:
                        seen[key].append(a)
                        existing_lower.add(a.lower())

    all_concepts = [Concept(name=canonical_by_lower[k], aliases=seen[k]) for k in order][:8]

    seen_topics: set[str] = set()
    all_topics: list[str] = []
    for r in results:
        for t in r.suggested_topics:
            if t.lower() not in seen_topics:
                seen_topics.add(t.lower())
                all_topics.append(t)

    seen_refs: set[str] = set()
    all_named_references: list[str] = []
    for r in results:
        for ref in r.named_references:
            key = ref.casefold().strip()
            if key and key not in seen_refs:
                seen_refs.add(key)
                all_named_references.append(ref)

    quality_rank = {"high": 2, "medium": 1, "low": 0}
    min_result = min(results, key=lambda r: quality_rank.get(r.quality, 1))

    merged_language = next((r.language for r in results if r.language), None)

    return AnalysisResult(
        summary=results[0].summary,
        concepts=all_concepts,
        suggested_topics=all_topics[:5],
        named_references=all_named_references[:8],
        quality=min_result.quality,
        language=merged_language,
    )


def _analyze_body(
    body: str,
    existing_concepts: list[str],
    path_name: str,
    client: LLMClientProtocol,
    config: Config,
    *,
    on_chunk_result=None,
    skip_completed: set[int] | None = None,
    prompt_contexts: list[_PromptConceptContext] | None = None,
    source_type: str = "notes",
) -> AnalysisResult:
    """Analyze note body, splitting into chunks if body exceeds fast_ctx // 2 chars."""
    chunk_size = config.effective_provider.fast_ctx // 2

    if len(body) <= chunk_size:
        prompt = _build_analysis_prompt(
            body,
            existing_concepts,
            path_name,
            language=config.pipeline.language,
            prompt_concepts=prompt_contexts,
        )
        return request_structured(
            client=client,
            prompt=prompt,
            model_class=AnalysisResult,
            model=config.models.fast,
            system=load_prompt(source_type),
            num_ctx=config.effective_provider.fast_ctx,
            temperature=0,
            stage="ingest",
            model_role="fast",
        )

    # Split into chunks — no overlap needed for concept extraction
    chunks = [body[i : i + chunk_size] for i in range(0, len(body), chunk_size)]
    log.info(
        "Note %s split into %d chunks for analysis (%d chars, chunk_size=%d)",
        path_name or "unknown",
        len(chunks),
        len(body),
        chunk_size,
    )

    def _analyze_chunk(chunk: str, idx: int) -> AnalysisResult:
        import time

        label = f"[part {idx + 1}/{len(chunks)}]"
        log.info("Analyzing %s %s …", path_name or "note", label)
        t0 = time.monotonic()
        lang = config.pipeline.language
        prompt = _build_analysis_prompt(
            chunk,
            existing_concepts,
            path_name,
            chunk_label=label,
            language=lang,
            prompt_concepts=prompt_contexts,
        )
        result = request_structured(
            client=client,
            prompt=prompt,
            model_class=AnalysisResult,
            model=config.models.fast,
            system=load_prompt(source_type),
            num_ctx=config.effective_provider.fast_ctx,
            temperature=0,
            stage="ingest",
            model_role="fast",
        )
        log.info("Analyzed %s %s (%.1fs)", path_name or "note", label, time.monotonic() - t0)
        return result

    completed = skip_completed or set()

    if config.pipeline.ingest_parallel:
        chunk_results: list[AnalysisResult | None] = [None] * len(chunks)
        errors: list[Exception] = []
        pending = [(i, chunk) for i, chunk in enumerate(chunks) if i not in completed]
        if pending:
            with ThreadPoolExecutor(max_workers=len(pending)) as executor:
                futures = {executor.submit(_analyze_chunk, chunk, i): i for i, chunk in pending}
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001
                        errors.append(exc)
                        continue
                    chunk_results[idx] = result
                    if on_chunk_result is not None:
                        on_chunk_result(idx, result)
            if errors:
                raise errors[0]
        results = [r for r in chunk_results if r is not None]
    else:
        results = []
        for i, chunk in enumerate(chunks):
            if i in completed:
                continue
            result = _analyze_chunk(chunk, i)
            if on_chunk_result is not None:
                on_chunk_result(i, result)
            results.append(result)

    return _merge_chunk_results(results)


def _analyze_body_with_checkpoints(
    body: str,
    existing_concepts: list[str],
    path: Path,
    content_hash: str,
    client: LLMClientProtocol,
    config: Config,
    db: StateDB,
    *,
    force: bool = False,
    prompt_contexts: list[_PromptConceptContext] | None = None,
    source_type: str = "notes",
) -> AnalysisResult:
    chunk_size = config.effective_provider.fast_ctx // 2
    rel_path = str(path.relative_to(config.vault))
    checkpoint_hash = _checkpoint_hash(content_hash, config, prompt_contexts or [])

    if len(body) <= chunk_size:
        if force:
            db.purge_ingest_chunks(rel_path)
        else:
            db.purge_ingest_chunks(rel_path, keep_hash=checkpoint_hash)
        return _analyze_body(
            body,
            existing_concepts,
            path.name,
            client,
            config,
            prompt_contexts=prompt_contexts,
            source_type=source_type,
        )

    chunks = [body[i : i + chunk_size] for i in range(0, len(body), chunk_size)]
    if force:
        db.purge_ingest_chunks(rel_path)
    else:
        db.purge_ingest_chunks(rel_path, keep_hash=checkpoint_hash)

    stored = db.list_ingest_chunks(rel_path, checkpoint_hash, len(chunks), chunk_size)
    chunk_results: list[AnalysisResult | None] = [None] * len(chunks)
    completed: set[int] = set()

    for row in stored:
        try:
            result = AnalysisResult.model_validate_json(row["result_json"])
        except Exception:  # noqa: BLE001 - corrupt checkpoints are re-analyzed
            continue
        chunk_results[row["chunk_index"]] = result
        completed.add(row["chunk_index"])

    if completed:
        log.info(
            "Resume ingest: %s using %d/%d completed chunks",
            path.name,
            len(completed),
            len(chunks),
        )

    def _save_chunk(idx: int, result: AnalysisResult) -> None:
        chunk_results[idx] = result
        db.upsert_ingest_chunk(
            rel_path,
            checkpoint_hash,
            idx,
            len(chunks),
            chunk_size,
            result.model_dump_json(),
        )

    if len(completed) < len(chunks):
        _analyze_body(
            body,
            existing_concepts,
            path.name,
            client,
            config,
            on_chunk_result=_save_chunk,
            skip_completed=completed,
            prompt_contexts=prompt_contexts,
            source_type=source_type,
        )

    results = [result for result in chunk_results if result is not None]
    merged = _merge_chunk_results(results)
    db.delete_ingest_chunks(rel_path, checkpoint_hash, len(chunks), chunk_size)
    return merged


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "is",
        "it",
        "in",
        "on",
        "at",
        "to",
        "by",
        "for",
        "of",
        "as",
        "from",
        "with",
        "this",
        "that",
        "these",
        "those",
        "be",
        "are",
    }
)

_NOISE_CONCEPT_KEYS = frozenset(
    {
        "document",
        "file",
        "image content unknown",
        "unknown content",
        "unknown file",
        "unknown filename",
        "untitled",
    }
)

_PAREN_ABBR_RE = re.compile(r"^(?P<base>.+?)\s*\((?P<abbr>[A-ZА-Я0-9][A-ZА-Я0-9.+-]{1,8})\)$")
_SURROUNDING_QUOTES_RE = re.compile(r"^[`'\"“”‘’«»]+|[`'\"“”‘’«»]+$")
_REWRITE_ALIAS_PREFIX = "__synto_rewrite_alias__:"
_LEGACY_REWRITE_ALIAS_PREFIX = "__olw_rewrite_alias__:"


def _clean_concept_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip()
    text = _SURROUNDING_QUOTES_RE.sub("", text).strip()
    return re.sub(r"\s+", " ", text)


def _concept_key(text: str) -> str:
    """Deterministic key for safe concept matching; not used as display text."""
    text = _clean_concept_text(text).casefold()
    text = re.sub(r"[_\-/:]+", " ", text)
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _base_concept_name(text: str) -> str:
    """Strip only safe parenthetical abbreviations, e.g. Extreme Programming (XP)."""
    cleaned = _clean_concept_text(text)
    match = _PAREN_ABBR_RE.match(cleaned)
    if not match:
        return cleaned
    abbr = match.group("abbr")
    if not abbr.isupper():
        return cleaned
    return match.group("base").strip()


def _safe_aliases_for_name(text: str) -> list[str]:
    """Aliases safe enough for deterministic matching, independent of LLM aliases."""
    cleaned = _clean_concept_text(text)
    aliases: list[str] = []
    base = _base_concept_name(cleaned)
    if base != cleaned:
        aliases.append(base)
        match = _PAREN_ABBR_RE.match(cleaned)
        if match:
            aliases.append(match.group("abbr"))
    lower = cleaned.casefold()
    if lower != cleaned:
        aliases.append(lower)

    seen: set[str] = set()
    result: list[str] = []
    for alias in aliases:
        key = _concept_key(alias)
        if key and key != _concept_key(cleaned) and key not in seen:
            seen.add(key)
            result.append(alias)
    return result


def _encode_rewrite_alias(alias: str) -> str:
    return f"{_REWRITE_ALIAS_PREFIX}{alias}"


def _decode_rewrite_alias(alias: str) -> tuple[str, bool]:
    if alias.startswith(_REWRITE_ALIAS_PREFIX):
        return alias[len(_REWRITE_ALIAS_PREFIX) :], True
    if alias.startswith(_LEGACY_REWRITE_ALIAS_PREFIX):
        return alias[len(_LEGACY_REWRITE_ALIAS_PREFIX) :], True
    return alias, False


def _is_noise_concept(text: str) -> bool:
    key = _concept_key(text)
    if not key:
        return True
    if key in _NOISE_CONCEPT_KEYS:
        return True
    if key.startswith("unknown ") or key.endswith(" unknown"):
        return True
    return False


def _has_title_or_body_evidence(concept_name: str, body: str, path_name: str = "") -> bool:
    key = _concept_key(concept_name)
    if not key:
        return False
    haystack_key = _concept_key(f"{path_name} {body}")
    if key in haystack_key:
        return True
    base_key = _concept_key(_base_concept_name(concept_name))
    return bool(base_key and base_key != key and base_key in haystack_key)


def _filter_concept_candidates(
    concepts: list[Concept],
    result: AnalysisResult,
    body: str,
    path_name: str = "",
) -> list[Concept]:
    """Conservatively drop weak LLM concepts before canonical normalization."""
    filtered: list[Concept] = []
    for concept in concepts:
        name = _clean_concept_text(concept.name)
        if not name or _is_noise_concept(name):
            continue
        has_evidence = _has_title_or_body_evidence(name, body, path_name)
        if result.quality in ("low", "medium") and not has_evidence:
            # Translated concept names may not appear in the source body; check validated
            # aliases (e.g., the native-language surface form returned as the first alias).
            # Guard with _validate_aliases + _is_noise_concept to prevent generic aliases
            # ("document", "file") from rescuing hallucinated concepts.
            validated_aliases = [
                a for a in _validate_aliases(name, concept.aliases) if not _is_noise_concept(a)
            ]
            if not any(_has_title_or_body_evidence(a, body, path_name) for a in validated_aliases):
                continue
        filtered.append(concept)
    return filtered


def _dedup_by_shared_alias(candidates: list[Concept]) -> list[Concept]:
    """Drop candidates whose name is already an alias of an earlier candidate."""
    seen_aliases: set[str] = set()
    result: list[Concept] = []
    for concept in candidates:
        key = _concept_key(concept.name)
        if key in seen_aliases:
            continue
        result.append(concept)
        for alias in concept.aliases:
            seen_aliases.add(_concept_key(alias))
    return result


def _validate_aliases(canonical: str, raw_aliases: list[str]) -> list[str]:
    """Filter LLM-produced aliases: remove too-short, stopwords, self-matches, duplicates."""
    seen = {canonical.lower()}
    valid: list[str] = []
    for alias in raw_aliases:
        decoded, is_rewrite_alias = _decode_rewrite_alias(alias)
        a = decoded.strip()
        if not a or a.lower() in seen:
            continue
        if len(a) < 2:
            continue
        if len(a) <= 3 and not a.isupper():
            continue
        if a.lower() in _STOPWORDS:
            continue
        seen.add(a.lower())
        valid.append(_encode_rewrite_alias(a) if is_rewrite_alias else a)
    return valid[:5]


def _persistable_aliases(aliases: list[str]) -> list[str]:
    persistable: list[str] = []
    for alias in aliases:
        decoded, is_rewrite_alias = _decode_rewrite_alias(alias)
        if not is_rewrite_alias:
            persistable.append(decoded)
    return persistable


def _display_aliases(aliases: list[str]) -> list[str]:
    return [_decode_rewrite_alias(alias)[0] for alias in aliases]


def _build_safe_concept_index(
    names: list[str], alias_map: dict[str, str] | None = None
) -> dict[str, str]:
    """Return unambiguous deterministic match keys for canonicals and stored aliases."""
    candidates: dict[str, set[str]] = {}

    def _add_keys(surface: str, canonical: str) -> None:
        keys = {_concept_key(surface), _concept_key(_base_concept_name(surface))}
        keys.update(_concept_key(alias) for alias in _safe_aliases_for_name(surface))
        for key in keys:
            if key:
                candidates.setdefault(key, set()).add(canonical)

    for name in names:
        _add_keys(name, name)
    for alias, canonical in (alias_map or {}).items():
        _add_keys(alias, canonical)

    return {key: next(iter(values)) for key, values in candidates.items() if len(values) == 1}


def _load_seed_concepts_from_index(config: Config) -> list[str]:
    """Load canonical names and aliases from preserved INDEX.json for fresh rebuilds."""
    index_path = config.app_dir / "INDEX.json"
    if not index_path.exists():
        return []
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    seeded: list[str] = []
    seen: set[str] = set()
    for article in payload.get("articles", []):
        if not isinstance(article, dict):
            continue
        for surface in [article.get("name"), *(article.get("aliases") or [])]:
            if not isinstance(surface, str):
                continue
            cleaned = _clean_concept_text(surface)
            key = cleaned.casefold()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            seeded.append(cleaned)
    return seeded


def _load_seed_canonical_names_from_index(config: Config) -> list[str]:
    """Load canonical article names from preserved INDEX.json for fresh rebuilds."""
    index_path = config.app_dir / "INDEX.json"
    if not index_path.exists():
        return []
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    seeded: list[str] = []
    seen: set[str] = set()
    for article in payload.get("articles", []):
        if not isinstance(article, dict):
            continue
        canonical = article.get("name")
        if not isinstance(canonical, str):
            continue
        cleaned = _clean_concept_text(canonical)
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        seeded.append(cleaned)
    return seeded


def _load_seed_alias_map_from_index(config: Config) -> dict[str, str]:
    """Load unambiguous alias -> canonical mappings from preserved INDEX.json."""
    index_path = config.app_dir / "INDEX.json"
    if not index_path.exists():
        return {}
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    candidates: dict[str, set[str]] = {}
    for article in payload.get("articles", []):
        if not isinstance(article, dict):
            continue
        canonical = article.get("name")
        if not isinstance(canonical, str):
            continue
        for alias in article.get("aliases") or []:
            if not isinstance(alias, str):
                continue
            key = _clean_concept_text(alias).casefold()
            if key:
                candidates.setdefault(key, set()).add(canonical)
    return {key: next(iter(values)) for key, values in candidates.items() if len(values) == 1}


def _load_source_concept_seeds_from_index(config: Config) -> dict[str, tuple[str, list[str]]]:
    """Load raw source concept memberships from preserved INDEX.json."""
    index_path = config.app_dir / "INDEX.json"
    if not index_path.exists():
        return {}
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    seeded: dict[str, tuple[str, list[str]]] = {}
    for entry in payload.get("source_concepts", []):
        if not isinstance(entry, dict):
            continue
        source_path = entry.get("source_path")
        content_hash = entry.get("content_hash")
        raw_concepts = entry.get("concepts")
        if not isinstance(source_path, str) or not isinstance(content_hash, str):
            continue
        if not isinstance(raw_concepts, list):
            continue
        concepts: list[str] = []
        seen: set[str] = set()
        for concept in raw_concepts:
            if not isinstance(concept, str):
                continue
            cleaned = _clean_concept_text(concept)
            key = cleaned.casefold()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            concepts.append(cleaned)
        if source_path and content_hash and concepts:
            seeded[source_path] = (content_hash, concepts)
    return seeded


def _matching_seeded_source_concepts(
    source_concept_seeds: dict[str, tuple[str, list[str]]] | None,
    rel_path: str,
    content_hash: str,
) -> list[str]:
    if not source_concept_seeds:
        return []
    seeded = source_concept_seeds.get(rel_path)
    if seeded is None:
        return []
    seeded_hash, concepts = seeded
    if seeded_hash != content_hash:
        return []
    return list(concepts)


def _build_trusted_alias_rewrite_index(
    existing_names: list[str],
    db_alias_map: dict[str, str],
    seed_alias_map: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return exact normalized alias rewrites safe enough for pre-cap canonicalization.

    An alias is unsafe if it resolves to multiple canonicals or if it is also the exact
    normalized key of a different canonical concept.
    """
    canonical_by_key: dict[str, str] = {}
    canonical_collisions: set[str] = set()
    for canonical in existing_names:
        key = _concept_key(canonical)
        if not key:
            continue
        if key in canonical_by_key and canonical_by_key[key] != canonical:
            canonical_collisions.add(key)
            continue
        canonical_by_key[key] = canonical

    merged_aliases: dict[str, set[str]] = {}
    for alias_map in [seed_alias_map or {}, db_alias_map]:
        for alias, canonical in alias_map.items():
            key = _concept_key(alias)
            if not key:
                continue
            merged_aliases.setdefault(key, set()).add(canonical)

    trusted: dict[str, str] = {}
    for key, canonicals in merged_aliases.items():
        if len(canonicals) != 1:
            continue
        canonical = next(iter(canonicals))
        canonical_key = _concept_key(canonical)
        if key in canonical_collisions:
            continue
        if key in canonical_by_key and key != canonical_key:
            continue
        trusted[key] = canonical
    return trusted


def _rewrite_candidates_to_canonicals(
    candidates: list[Concept],
    existing_names: list[str],
    db_alias_map: dict[str, str],
    seed_alias_map: dict[str, str] | None = None,
) -> list[Concept]:
    rewrite_index = _build_trusted_alias_rewrite_index(existing_names, db_alias_map, seed_alias_map)
    if not rewrite_index:
        return candidates

    rewritten: list[Concept] = []
    seen_keys: set[str] = set()
    for candidate in candidates:
        original_name = _clean_concept_text(candidate.name)
        if not original_name:
            continue
        canonical = rewrite_index.get(_concept_key(original_name))
        if canonical is None:
            trusted_matches = {
                rewrite_index[_concept_key(alias)]
                for alias in _validate_aliases(original_name, candidate.aliases)
                if _concept_key(alias) in rewrite_index
            }
            canonical = next(iter(trusted_matches)) if len(trusted_matches) == 1 else original_name
        canonical_key = _concept_key(canonical)
        if canonical_key in seen_keys:
            continue
        seen_keys.add(canonical_key)

        aliases = list(candidate.aliases)
        if _concept_key(original_name) != canonical_key:
            aliases = [_encode_rewrite_alias(original_name), *aliases]
        rewritten.append(Concept(name=canonical, aliases=aliases))
    return rewritten


def _normalize_concepts(
    raw_concepts: list[Concept],
    db: StateDB,
    seed_concepts: list[str] | None = None,
    seed_alias_map: dict[str, str] | None = None,
) -> list[tuple[str, list[str]]]:
    """Dedup against existing canonical concept names using safe deterministic keys.

    Returns (canonical_name, validated_aliases) pairs.
    """
    existing_names = db.list_all_concept_names()
    if seed_concepts:
        seen_seed = {name.casefold() for name in seed_concepts}
        existing_names = [
            *seed_concepts,
            *(name for name in existing_names if name.casefold() not in seen_seed),
        ]
    alias_map = db.list_alias_map()
    if seed_alias_map:
        alias_map = {**seed_alias_map, **alias_map}
    existing = _build_safe_concept_index(existing_names, alias_map)
    seen: set[str] = set()
    result: list[tuple[str, list[str]]] = []
    for concept in raw_concepts:
        name = _clean_concept_text(concept.name)
        if not name or _is_noise_concept(name):
            continue
        safe_keys = [_concept_key(name), _concept_key(_base_concept_name(name))]
        safe_keys.extend(_concept_key(alias) for alias in _safe_aliases_for_name(name))
        canonical = next(
            (existing[key] for key in safe_keys if key in existing), _base_concept_name(name)
        )
        canonical_key = _concept_key(canonical)
        if canonical_key in seen:
            continue
        seen.add(canonical_key)
        alias_candidates = [*_safe_aliases_for_name(name), *concept.aliases]
        if _concept_key(name) != canonical_key:
            alias_candidates.insert(0, name)
        aliases = _validate_aliases(canonical, alias_candidates)
        result.append((canonical, aliases))
    return result


_HEADER_SCAN_LINES = 30  # only strip short lines from the opening section

# Media reference patterns for source page preservation
_OBSIDIAN_EMBED_RE = re.compile(
    r"!\[\[([^\]]+\.(?:png|jpg|jpeg|gif|svg|webp|bmp|tiff|avif|pdf|mp4|webm|mov|mp3|wav|ogg))\]\]",
    re.IGNORECASE,
)
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_BARE_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def _meaningful_text_stats(body: str) -> tuple[int, int]:
    """Return (chars, words) after removing media, URLs, and markdown boilerplate."""
    text = _OBSIDIAN_EMBED_RE.sub(" ", body)
    text = _MD_IMAGE_RE.sub(" ", text)
    text = _BARE_URL_RE.sub(" ", text)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"[#>*_\-\[\]()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = re.findall(r"\w{2,}", text, flags=re.UNICODE)
    return len(text), len(words)


def _topic_has_text_evidence(topic: str, body: str, path_name: str = "") -> bool:
    topic_key = _concept_key(topic)
    if not topic_key:
        return False
    return _has_title_or_body_evidence(topic, body, path_name)


def _suggested_topic_candidates(
    result: AnalysisResult,
    body: str,
    path_name: str = "",
) -> list[Concept]:
    chars, words = _meaningful_text_stats(body)
    if chars < 80 and words < 12:
        return []
    candidates: list[Concept] = []
    for topic in result.suggested_topics:
        if _is_noise_concept(topic):
            continue
        if result.quality in ("low", "medium") and not _topic_has_text_evidence(
            topic, body, path_name
        ):
            continue
        candidates.append(Concept(name=topic, aliases=[]))
    return candidates


def _preprocess_web_clip(content: str) -> str:
    """Clean common Obsidian Web Clipper artifacts (nav bars, cookie banners, HTML tags).

    HTML stripping is scoped to the first _HEADER_SCAN_LINES only — body HTML
    (<details>, <kbd>, <sup>, etc.) is intentional and preserved.
    """
    _MD_STARTS = ("#", "-", "*", ">", "[", "!")  # markdown structural chars — always keep
    lines = content.splitlines()

    cleaned = []
    for i, line in enumerate(lines):
        if i < _HEADER_SCAN_LINES:
            # Strip HTML only in header region (nav/banner cleanup)
            line = re.sub(r"<[^>]+>", "", line)
            stripped = line.strip()
            # Skip short non-empty non-markdown lines (nav/banner heuristic)
            if stripped and len(stripped.split()) <= 5 and not stripped.startswith(_MD_STARTS):
                continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _collect_media_refs(body: str) -> list[str]:
    """Extract media references from note body for preservation in source pages."""
    refs: list[str] = []
    for m in _OBSIDIAN_EMBED_RE.finditer(body):
        refs.append(f"- ![[{m.group(1)}]]")
    for m in _MD_IMAGE_RE.finditer(body):
        alt, url = m.group(1), m.group(2)
        refs.append(f"- ![{alt}]({url})")
    return refs


def _create_source_summary_page(
    path: Path,
    src_meta: dict,
    result: AnalysisResult,
    config: Config,
    body: str = "",
    canonical_concepts: list[str] | None = None,
) -> Path:
    """
    Generate wiki/sources/{Title}.md from AnalysisResult. No extra LLM call.
    Returns the path written.
    """
    # Derive title from note frontmatter > file stem
    title = src_meta.get("title") or path.stem.replace("-", " ").strip()
    safe_name = sanitize_filename(title)
    out_path = config.sources_dir / f"{safe_name}.md"
    config.sources_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d")
    rel_raw = str(path.relative_to(config.vault))
    source_url = src_meta.get("source") or src_meta.get("url") or ""
    aliases = generate_aliases(title, "")  # source pages rarely have abbreviations

    # Build concept list as [[wikilinks]]
    concept_names = (
        canonical_concepts if canonical_concepts is not None else [c.name for c in result.concepts]
    )
    concept_lines = "\n".join(
        f"- [[{sanitize_wikilink_target(name)}]]" for name in concept_names[:8] if name.strip()
    )

    out_meta: dict = {
        "title": title,
        "aliases": aliases,
        "tags": ["source"],
        "status": "published",
        "source_file": rel_raw,
        "quality": result.quality,
        "created": now,
    }
    if source_url:
        out_meta["source_url"] = source_url

    body_parts = [
        f"# {title}",
        "",
        "## Summary",
        result.summary,
        "",
        "## Concepts",
        concept_lines,
        "",
        "## Source Info",
        f"- **Quality:** {result.quality}",
        f"- **Raw file:** {rel_raw}",
        f"- **Ingested:** {now}",
    ]
    if source_url:
        body_parts.append(f"- **URL:** {source_url}")

    media_refs = _collect_media_refs(body)
    if media_refs:
        body_parts += ["", "## Media"] + media_refs

    write_note(out_path, out_meta, "\n".join(body_parts))
    log.info("Source summary written: %s", out_path.name)
    return out_path


def ingest_note(
    path: Path,
    config: Config,
    client: LLMClientProtocol,
    db: StateDB,
    rag=None,  # Optional RAGStore, injected in Phase 2
    existing_topics: list[str] | None = None,  # existing concept names for prompt
    seed_concepts: list[str] | None = None,
    seed_alias_map: dict[str, str] | None = None,
    source_concept_seeds: dict[str, tuple[str, list[str]]] | None = None,
    force: bool = False,
) -> AnalysisResult | None:
    """
    Ingest a single raw note.

    Returns AnalysisResult or None if skipped (duplicate / already ingested).
    """
    content = path.read_text(encoding="utf-8")
    # Hash body only (strip frontmatter) so copies are detected as duplicates
    # even after ingest has updated the original's frontmatter (olw_status etc.)
    try:
        _, body_for_hash = parse_note(path)
    except Exception:
        body_for_hash = content
    h = _content_hash(body_for_hash)

    # Dedup check
    existing = db.get_raw_by_hash(h)
    if existing and existing.path != str(path.relative_to(config.vault)):
        log.info("Duplicate of %s, skipping %s", existing.path, path.name)
        return None

    rel_path = str(path.relative_to(config.vault))
    record = db.get_raw(rel_path)
    current_prompt_version = _ingest_prompt_version(config)

    if (
        record
        and record.status in {"ingested", "compiled"}
        and record.content_hash == h
        and record.prompt_version == current_prompt_version
        and not force
    ):
        log.info("Already ingested: %s", path.name)
        return None

    if (
        record
        and record.status in {"ingested", "compiled"}
        and record.content_hash == h
        and record.prompt_version != current_prompt_version
        and not force
    ):
        log.info("Re-ingesting %s: ingest language policy changed", path.name)

    # Pre-process web clips
    meta, body = parse_note(path)
    source_type = str(meta.get("source_type", "notes"))
    # Strip Obsidian clipper placeholder embeds before they reach the LLM
    body = re.sub(r"!\[\[[^\]]*unknown_filename[^\]]*\]\]", "", body, flags=re.IGNORECASE)
    if meta.get("source") or meta.get("url"):  # web clipper adds these
        body = _preprocess_web_clip(body)

    # Chunk + embed only when RAG store is wired in (Phase 2)
    if rag is not None:
        chunks = chunk_text(
            body, chunk_size=config.rag.chunk_size, overlap=config.rag.chunk_overlap
        )
        embeddings = client.embed_batch(chunks, model=config.models.embed)
        rag.add_document(
            doc_id=rel_path,
            chunks=chunks,
            embeddings=embeddings,
            metadata={"source": rel_path, "type": "raw"},
        )

    # LLM analysis — use existing concept names so model can reuse canonical names
    if existing_topics is None:
        existing_topics = db.list_all_concept_names()
        if not existing_topics:
            seed_concepts = _load_seed_canonical_names_from_index(config)
            seed_alias_map = _load_seed_alias_map_from_index(config)
            source_concept_seeds = _load_source_concept_seeds_from_index(config)
            existing_topics = list(seed_concepts)
    prompt_contexts = _canonical_prompt_contexts(
        existing_topics,
        seed_alias_map or db.list_alias_map(),
    )
    try:
        result: AnalysisResult = _analyze_body_with_checkpoints(
            body=body,
            existing_concepts=existing_topics,
            path=path,
            content_hash=h,
            client=client,
            config=config,
            db=db,
            force=force,
            prompt_contexts=prompt_contexts,
            source_type=source_type,
        )
    except Exception as e:
        log.error("Analysis failed for %s: %s", path.name, e)
        db.upsert_raw(
            RawNoteRecord(
                path=rel_path,
                content_hash=h,
                status="failed",
                error=str(e),
                prompt_version=current_prompt_version,
            )
        )
        return None

    # Update state DB (raw files stay immutable — metadata lives in state.db only)
    db.upsert_raw(
        RawNoteRecord(
            path=rel_path,
            content_hash=h,
            status="ingested",
            summary=result.summary,
            quality=result.quality,
            language=result.language,
            prompt_version=current_prompt_version,
            ingested_at=datetime.now(),
        )
    )

    # Normalize concept names against existing canonical names, store linkages
    max_concepts = config.pipeline.max_concepts_per_source
    existing_names = db.list_all_concept_names()
    if seed_concepts:
        seen_seed = {name.casefold() for name in seed_concepts}
        existing_names = [
            *seed_concepts,
            *(name for name in existing_names if name.casefold() not in seen_seed),
        ]
    db_alias_map = db.list_alias_map()
    quality_cap = {"high": max_concepts, "medium": min(max_concepts, 4), "low": 2}
    effective_max = quality_cap.get(result.quality or "low", max_concepts)
    filtered_candidates = _filter_concept_candidates(result.concepts, result, body, path.name)
    concept_candidates = _dedup_by_shared_alias(filtered_candidates[:effective_max])
    if not concept_candidates:
        concept_candidates = _suggested_topic_candidates(result, body, path.name)[:3]
    matching_seed_concepts = _matching_seeded_source_concepts(
        source_concept_seeds,
        rel_path,
        h,
    )
    if matching_seed_concepts:
        concept_candidates = [Concept(name=name, aliases=[]) for name in matching_seed_concepts]
    concept_candidates = _rewrite_candidates_to_canonicals(
        concept_candidates,
        existing_names,
        db_alias_map,
        seed_alias_map,
    )[:max_concepts]
    normalized = _normalize_concepts(
        concept_candidates,
        db,
        seed_concepts,
        seed_alias_map,
    )
    canonical_names = [name for name, _ in normalized]
    db.replace_concepts_for_source(rel_path, canonical_names)
    for canonical, aliases in normalized:
        persistable_aliases = _persistable_aliases(aliases)
        if persistable_aliases:
            db.upsert_aliases(canonical, persistable_aliases)

    title_for_items = str(meta.get("title") or path.stem.replace("-", " ").strip())
    item_candidates = [
        *extract_quoted_title_items(title_for_items, rel_path),
        *extract_named_reference_items(
            result.named_references,
            title_for_items,
            body,
            rel_path,
            canonical_names,
        ),
    ]
    store_extracted_items(db, rel_path, item_candidates)

    # Create source summary page in wiki/sources/ (no extra LLM call)
    try:
        _create_source_summary_page(
            path,
            meta,
            result,
            config,
            body=body,
            canonical_concepts=canonical_names,
        )
    except Exception as e:
        log.warning("Source summary page failed for %s: %s", path.name, e)

    log.info(
        "Ingested: %s (quality=%s, concepts=%s)",
        path.name,
        result.quality,
        canonical_names[:3],
    )
    return result


def ingest_all(
    config: Config,
    client: LLMClientProtocol,
    db: StateDB,
    rag=None,
    force: bool = False,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> list[tuple[Path, AnalysisResult | None]]:
    """Ingest all .md files in raw/ (excluding raw/processed/ subfolders)."""
    raw_files = [
        p
        for p in config.raw_dir.rglob("*.md")
        if "processed" not in p.parts and not p.name.startswith(".")
    ]
    # Snapshot concept names once before loop (for consistent prompt context)
    existing_topics = db.list_all_concept_names()
    seed_concepts = None
    seed_alias_map = None
    source_concept_seeds = None
    if not existing_topics:
        seed_concepts = _load_seed_canonical_names_from_index(config)
        seed_alias_map = _load_seed_alias_map_from_index(config)
        source_concept_seeds = _load_source_concept_seeds_from_index(config)
        existing_topics = list(seed_concepts)
    existing_topic_keys = {topic.casefold() for topic in existing_topics}
    results = []
    total = len(raw_files)
    for done, path in enumerate(sorted(raw_files), start=1):
        result = ingest_note(
            path=path,
            config=config,
            client=client,
            db=db,
            rag=rag,
            existing_topics=existing_topics,
            seed_concepts=seed_concepts,
            seed_alias_map=seed_alias_map,
            source_concept_seeds=source_concept_seeds,
            force=force,
        )
        results.append((path, result))
        if result is not None:
            rel_path = str(path.relative_to(config.vault))
            for name in db.get_concepts_for_sources([rel_path]):
                name_key = name.casefold()
                if name_key not in existing_topic_keys:
                    existing_topic_keys.add(name_key)
                    existing_topics.append(name)
        if on_progress is not None:
            on_progress(done, total, str(path.relative_to(config.vault)))
    return results


_TERM_EXTRACTION_SYSTEM = (
    "You are a technical vocabulary extractor. "
    "Extract defined or implied terms from the provided text. "
    "Return JSON only, no explanation."
)


class _TermLLMItem(_BaseModel):
    name: str
    definition: str
    aliases: list[str] = []
    provenance: str = "extracted"
    confidence: float = 0.9


class _TermLLMResponse(_BaseModel):
    terms: list[_TermLLMItem] = []


def extract_terms(
    segment: SourceSegment,
    client: LLMClientProtocol,
    config: Config,
) -> TermExtractionResult:
    """Second LLM pass: extract technical terms from a source segment."""
    from ..models import TermRecord

    prompt = (
        "Extract technical terms with definitions from the following text.\n"
        'Return JSON: {"terms": [{"name": "...", "definition": "...", "aliases": [], '
        '"provenance": "extracted", "confidence": 0.9}]}\n\n'
        f"{segment.text[:4000]}"
    )
    llm_result = request_structured(
        client=client,
        prompt=prompt,
        model_class=_TermLLMResponse,
        model=config.models.fast,
        system=_TERM_EXTRACTION_SYSTEM,
        num_ctx=config.effective_provider.fast_ctx,
        temperature=0,
        stage="term_extraction",
        model_role="fast",
    )
    terms = [
        TermRecord(
            name=t.name,
            definition=t.definition,
            aliases=t.aliases,
            source_segment_id=segment.id,
            provenance=t.provenance,  # type: ignore[arg-type]
            confidence=min(max(t.confidence, 0.0), 1.0),
        )
        for t in llm_result.terms
    ]
    return TermExtractionResult(
        terms=terms,
        source_segment_id=segment.id,
        model=config.models.fast,
    )
