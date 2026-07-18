"""
Ingest pipeline: raw note → chunk → analyze → embed → update state.

Uses fast model (gemma4:e4b) for analysis.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, get_args

from pydantic import BaseModel as _BaseModel

from ..concept_text import _PAREN_ABBR_RE
from ..concept_text import base_concept_name as _base_concept_name
from ..concept_text import clean_concept_text as _clean_concept_text
from ..concept_text import concept_key as _concept_key
from ..concept_text import match_key as _match_key
from ..config import Config
from ..indexer import append_log
from ..models import (
    AnalysisResult,
    Concept,
    RawNoteRecord,
    RelationCandidate,
    RelationExtractionResult,
    SourceSegment,
    TermExtractionResult,
)
from ..paths import rel_posix
from ..state import StateDB
from ..structured_output import request_structured

if TYPE_CHECKING:
    from ..client_factory import ModelRouter, RoleEndpoint
from ..vault import (
    chunk_text,
    generate_aliases,
    parse_note,
    sanitize_filename,
    sanitize_wikilink_target,
    strip_image_text_blocks,
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
    prompt_hash = hashlib.sha256(_SYSTEM.encode("utf-8")).hexdigest()[:12]
    return f"{INGEST_ANALYSIS_PROMPT_VERSION}|language={language}|notes={prompt_hash}"


def _source_prompt_fingerprint(source_type: str) -> str:
    prompt = load_prompt(source_type)
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


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
    source_type: str = "notes",
) -> str:
    fast = config.resolve_role("fast")
    payload = {
        "content_hash": content_hash,
        "prompt_version": _ingest_prompt_version(config),
        "source_type": source_type,
        "source_prompt": _source_prompt_fingerprint(source_type),
        "fast_model": config.model_name("fast"),
        # Fold in the fast-role connection so switching provider/account (same model id) re-ingests.
        # api_key_env is the env-var NAME, not the secret; rotating the key under the same name is
        # the same account and intentionally does not bust the checkpoint.
        "fast_connection": [
            fast.provider_kind,
            fast.url,
            fast.api_key_env or "",
            sorted(fast.headers.items()),
        ],
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
    Quality: most common rating across chunks (ties broken toward the higher rank).
    """
    from collections import Counter

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

    # Full deduplicated union. Capping is the caller's policy: ingest_note applies the
    # quality-scaled, configured max_concepts. A per-call "max 8" must not become a
    # whole-document cap here, or multi-chunk sources silently ignore the config.
    all_concepts = [Concept(name=canonical_by_lower[k], aliases=seen[k]) for k in order]

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

    # Most common quality across chunks, ties broken toward the higher rank. (Was a
    # conservative min, but with segment-aligned chunks a single thin or peripheral section
    # — title page, references — would drag the whole note down and, via the quality cap in
    # ingest_note, halve the extracted concept count. The note's quality should reflect its
    # substantive majority, not its weakest section.)
    quality_rank = {"high": 2, "medium": 1, "low": 0}
    quality_counts = Counter(r.quality for r in results if r.quality)
    merged_quality = (
        max(quality_counts, key=lambda q: (quality_counts[q], quality_rank.get(q, 1)))
        if quality_counts
        else "medium"
    )

    merged_language = next((r.language for r in results if r.language), None)

    return AnalysisResult(
        summary=results[0].summary,
        concepts=all_concepts,
        # suggested_topics keeps a small bound: it is only a concept-fallback, sliced to
        # [:3] downstream, so this is not a silent whole-document ceiling.
        suggested_topics=all_topics[:5],
        # Full evidenced union; named references have no downstream cap, so a per-call
        # limit here would silently bound the whole document (chunk-order biased).
        named_references=all_named_references,
        quality=merged_quality,
        language=merged_language,
    )


def _build_segment_units(segments: list, chunk_size: int) -> list[tuple[str, list[str]]]:
    """Pack whole segments (in `ordinal` order) into analysis units up to chunk_size chars.

    Returns [(unit_text, [segment_id, ...])]. A segment is never split, so a segment longer
    than chunk_size becomes its own unit. This keeps the analysis aligned to structural
    segments so extracted concepts can be attributed to known segment ids.
    """
    units: list[tuple[str, list[str]]] = []
    parts: list[str] = []
    ids: list[str] = []
    cur_len = 0
    for seg in segments:
        text = seg["text"]
        seg_id = seg["id"]
        add_len = len(text) + 2  # joiner allowance
        if ids and cur_len + add_len > chunk_size:
            units.append(("\n\n".join(parts), ids))
            parts, ids, cur_len = [], [], 0
        parts.append(text)
        ids.append(seg_id)
        cur_len += add_len
    if ids:
        units.append(("\n\n".join(parts), ids))
    return units


def _persist_concept_occurrences(
    db: StateDB,
    source_id: str,
    units: list[tuple[str, list[str]]],
    attribution: dict[int, list[str]],
    canonical_names: list[str],
) -> None:
    """Link extracted concepts to the segments that fed each analysis chunk.

    The concepts a chunk yielded are attributed to every segment in that chunk, after
    resolving raw concept names to the source's final canonical names. Only canonical
    names that survived normalization are persisted. Re-ingest replaces the source's rows.
    """
    from collections import namedtuple

    canon_set = set(canonical_names)
    seg_to_concepts: dict[str, set[str]] = {}
    for idx, (_text, seg_ids) in enumerate(units):
        canon_here: set[str] = set()
        for raw in attribution.get(idx, []):
            resolved = db.find_concept_by_name_or_alias(raw)
            canonical = resolved[0] if resolved else (raw if raw in canon_set else None)
            if canonical in canon_set:
                canon_here.add(canonical)
        if not canon_here:
            continue
        for sid in seg_ids:
            seg_to_concepts.setdefault(sid, set()).update(canon_here)

    db.clear_concept_occurrences_for_source(source_id)
    occ = namedtuple("occ", ["name", "confidence"])
    for sid, concepts in seg_to_concepts.items():
        db.upsert_concept_occurrences([occ(name=c, confidence=1.0) for c in sorted(concepts)], sid)


def _analyze_body(
    body: str,
    existing_concepts: list[str],
    path_name: str,
    router: ModelRouter,
    config: Config,
    *,
    on_chunk_result=None,
    skip_completed: set[int] | None = None,
    prompt_contexts: list[_PromptConceptContext] | None = None,
    source_type: str = "notes",
    units: list[tuple[str, list[str]]] | None = None,
    attribution: dict[int, list[str]] | None = None,
) -> AnalysisResult:
    """Analyze note body, splitting into chunks if body exceeds fast_ctx // 2 chars.

    units: optional pre-built [(chunk_text, [segment_id, ...])] aligned to a tracked
    source's segments. When given, these replace fixed-size body splitting so the
    concepts each chunk yields can be attributed to known segment ids.
    attribution: optional dict, populated chunk_index -> [extracted concept name], for
    callers that persist concept_occurrences. Filled on fresh analysis of each chunk.
    """
    fast = router.endpoint("fast")
    # Chunk sizing follows the fast role's resolved context window.
    fast_ctx = config.resolve_role("fast").ctx
    chunk_size = fast_ctx // 2

    if units is None and len(body) <= chunk_size:
        prompt = _build_analysis_prompt(
            body,
            existing_concepts,
            path_name,
            language=config.pipeline.language,
            prompt_concepts=prompt_contexts,
        )
        result = request_structured(
            client=fast.client,
            prompt=prompt,
            model_class=AnalysisResult,
            model=fast.model,
            system=load_prompt(source_type),
            num_ctx=fast_ctx,
            temperature=fast.temperature if fast.temperature is not None else 0,
            stage="ingest",
            model_role="fast",
            think=fast.think,
            options=fast.options,
        )
        if attribution is not None:
            attribution[0] = [c.name for c in result.concepts]
        return result

    # Segment-aligned units (tracked sources) replace fixed-size body slices.
    if units is not None:
        chunks = [text for text, _seg_ids in units]
        log.info(
            "Note %s analyzed in %d segment-aligned chunk(s)",
            path_name or "unknown",
            len(chunks),
        )
    else:
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
            client=fast.client,
            prompt=prompt,
            model_class=AnalysisResult,
            model=fast.model,
            system=load_prompt(source_type),
            num_ctx=fast_ctx,
            temperature=fast.temperature if fast.temperature is not None else 0,
            stage="ingest",
            model_role="fast",
            think=fast.think,
            options=fast.options,
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
                    if attribution is not None:
                        attribution[idx] = [c.name for c in result.concepts]
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
            if attribution is not None:
                attribution[i] = [c.name for c in result.concepts]
            if on_chunk_result is not None:
                on_chunk_result(i, result)
            results.append(result)

    return _merge_chunk_results(results)


def _analyze_body_with_checkpoints(
    body: str,
    existing_concepts: list[str],
    path: Path,
    content_hash: str,
    router: ModelRouter,
    config: Config,
    db: StateDB,
    *,
    force: bool = False,
    prompt_contexts: list[_PromptConceptContext] | None = None,
    source_type: str = "notes",
    units: list[tuple[str, list[str]]] | None = None,
    attribution: dict[int, list[str]] | None = None,
) -> AnalysisResult:
    chunk_size = config.resolve_role("fast").ctx // 2
    rel_path = rel_posix(path, config.vault)
    checkpoint_hash = _checkpoint_hash(
        content_hash,
        config,
        prompt_contexts or [],
        source_type=source_type,
    )

    if units is None and len(body) <= chunk_size:
        if force:
            db.purge_ingest_chunks(rel_path)
        else:
            db.purge_ingest_chunks(rel_path, keep_hash=checkpoint_hash)
        return _analyze_body(
            body,
            existing_concepts,
            path.name,
            router,
            config,
            prompt_contexts=prompt_contexts,
            source_type=source_type,
            attribution=attribution,
        )

    # Tracked sources pass segment-aligned units; plain notes split the body by size.
    if units is not None:
        chunks = [text for text, _seg_ids in units]
    else:
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
        # Resumed chunks must still feed attribution (they aren't re-analyzed below).
        if attribution is not None:
            attribution[row["chunk_index"]] = [c.name for c in result.concepts]

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
            router,
            config,
            on_chunk_result=_save_chunk,
            skip_completed=completed,
            prompt_contexts=prompt_contexts,
            source_type=source_type,
            units=units,
            attribution=attribution,
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

_REWRITE_ALIAS_PREFIX = "__synto_rewrite_alias__:"
_LEGACY_REWRITE_ALIAS_PREFIX = "__olw_rewrite_alias__:"


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


def _strip_demoted_alias_from_host(config: Config, db: StateDB, host_id: str, surface: str) -> None:
    """Remove a demoted alias from a host concept article's frontmatter (no recompile).

    When a surface a host held as a weak alias becomes its own concept, the host's on-disk
    ``aliases:`` would still claim it — making ``[[surface]]`` resolve ambiguously in the vault.
    This strips just that alias; the body is untouched, so content_hash (and manual-edit
    protection) is preserved. No-op when the host has no published article yet — the DB demotion
    is the source of truth and a stale-frontmatter read must never fail the ingest.
    """
    rel = db.published_path_for_entity(host_id)
    if not rel:
        return
    path = config.vault / rel
    if not path.exists():
        return
    try:
        meta, body = parse_note(path)
    except Exception:  # noqa: BLE001
        return
    existing = list(meta.get("aliases") or [])
    surface_key = _concept_key(surface)
    kept = [a for a in existing if _concept_key(str(a)) != surface_key]
    if len(kept) == len(existing):
        return
    if kept:
        meta["aliases"] = kept
    else:
        meta.pop("aliases", None)
    write_note(path, meta, body)
    append_log(config, f"relink: demoted alias {surface!r} from {rel} (now its own concept)")


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
            # Accept both legacy string format and new {name, entity_id} dict format.
            if isinstance(concept, dict):
                raw_name = concept.get("name")
                if not isinstance(raw_name, str):
                    continue
                concept = raw_name
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


def _restore_identity_from_index(config: Config, db: StateDB) -> None:
    """Restore entity ids + identity log from a preserved INDEX.json on a fresh-DB rebuild.

    Durability (decision 13): recreates entities with their ORIGINAL ids (not re-minted) so a
    deleted-state.db rebuild is lossless. Precedence is enforced in restore_entities_from_seed
    (state.db wins if it already holds entities). No-op when INDEX.json is absent — that is a
    normal fresh ingest (both absent → re-mint by label), not a degraded rebuild.
    """
    index_path = config.app_dir / "INDEX.json"
    if not index_path.exists():
        return
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return
    entries: list[tuple[str, str]] = []
    for entry in payload.get("source_concepts", []):
        if not isinstance(entry, dict):
            continue
        for concept in entry.get("concepts", []):
            if isinstance(concept, dict):
                name, eid = concept.get("name"), concept.get("entity_id")
                if isinstance(name, str) and isinstance(eid, str) and eid:
                    entries.append((name, eid))
    for art in payload.get("articles", []):
        if isinstance(art, dict):
            name, eid = art.get("name"), art.get("entity_id")
            if isinstance(name, str) and isinstance(eid, str) and eid:
                entries.append((name, eid))
    restored = db.restore_entities_from_seed(entries)
    if restored:
        log.info("Rebuild: restored %d entity identities from INDEX.json seed", restored)
        db.restore_identity_log(payload.get("identity_log", []))
        # Denials MUST restore before blessed aliases: a blessed alias that was later denied
        # (`concept alias remove` on a rename/merge-blessed surface) must stay gone after
        # rebuild, and restore_blessed_aliases would otherwise resurrect it.
        denied = db.restore_alias_denials(payload.get("alias_denials", []))
        if denied:
            log.info("Rebuild: restored %d alias denial(s) from INDEX.json seed", denied)
        # Blessed (merge/rename) aliases cannot be regenerated by re-ingest, so the seed
        # carries them with provenance — re-arm the re-mint guard on rebuild.
        blessed = db.restore_blessed_aliases(payload.get("entity_aliases", []))
        if blessed:
            log.info("Rebuild: restored %d blessed aliases from INDEX.json seed", blessed)


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
    rel_path: str = "",
    seed_concepts: list[str] | None = None,
    seed_alias_map: dict[str, str] | None = None,
) -> list[tuple[str, list[str]]]:
    """Dedup against entity labels using match_key resolution (Phase 2).

    Returns (canonical_name, validated_aliases) pairs.  Ambiguous concepts (>1 DB match
    with no sticky edge) are recorded as ambiguous occurrences and excluded from the list.

    Resolution order for each surface form:
      1. DB match_key lookup — handles plural/singular variants and stored aliases.
      2. In-memory safe-key index — handles abbreviation expansion (e.g. "XP" →
         "Extreme Programming (XP)") and rebuild scenarios where the DB is empty.
      3. In-run dedup — prevents duplicate minting within one ingest pass.
      4. Mint new entity.
    """
    # Build in-memory fallback index from all existing DB names + seeds.
    # This handles abbreviation expansion and rebuild (DB empty, INDEX.json present).
    existing_names = db.list_all_concept_names()
    if seed_concepts:
        seen_seed = {n.casefold() for n in seed_concepts}
        existing_names = [
            *seed_concepts,
            *(n for n in existing_names if n.casefold() not in seen_seed),
        ]
    alias_map = db.list_alias_map()
    if seed_alias_map:
        alias_map = {**seed_alias_map, **alias_map}
    mem_index = _build_safe_concept_index(existing_names, alias_map)

    in_run: dict[str, str] = {}  # match_key → canonical_name (minted this run, not in DB yet)
    seen_ids: set[str] = set()  # entity_ids already included this run
    seen_keys: set[str] = set()  # concept_keys for new/mem-matched concepts
    result: list[tuple[str, list[str]]] = []

    def _link(entity_id: str) -> None:
        """Append a (canonical, aliases) pair linking the current surface to an existing entity.

        Aliases colliding with another entity's preferred label are NOT filtered here. The
        persistence seam in ingest_note drops them AND records the merge candidate, so every
        preferred-collision drop happens at one place — which is what makes the candidate set a
        function of the claim-set, not of ingest order.
        """
        seen_ids.add(entity_id)
        canonical = db.preferred_label_for_entity(entity_id) or name
        seen_keys.add(_concept_key(canonical))
        alias_candidates = [*_safe_aliases_for_name(name), *concept.aliases]
        if _concept_key(name) != _concept_key(canonical):
            alias_candidates.insert(0, name)
        result.append((canonical, _validate_aliases(canonical, alias_candidates)))

    for concept in raw_concepts:
        name = _clean_concept_text(concept.name)
        if not name or _is_noise_concept(name):
            continue

        # Order-independent role-aware classification (v26). A surface is an existing concept iff
        # it matches a PREFERRED label or a human-BLESSED alias. Matching only WEAK aliases means
        # it is its own concept — minting demotes those weak aliases and records a merge candidate
        # at the write seam, so identity no longer depends on which paper arrived first.

        # 1. Preferred-label match (exact label_key, then match_key fold for plural/singular).
        pref = db.preferred_entity_for_surface(name)
        if pref.ambiguous:
            # >1 preferred (genuine homonym) — sticky-resolve to the source's existing sense, else
            # record for later resolution; never silently pick a sense (decision 18).
            sticky_id = db.get_sticky_entity_for_source(rel_path, pref.ids)
            if sticky_id is not None:
                if sticky_id not in seen_ids:
                    _link(sticky_id)
            else:
                db.record_ambiguous_occurrence(
                    name, pref.ids, surface=name, source_path=rel_path or None
                )
            continue
        if pref.ids:
            if pref.ids[0] not in seen_ids:
                _link(pref.ids[0])
            continue

        # 2. Human-blessed alias match (user/rename) — respect the curation decision.
        blessed = db.blessed_alias_entities_for_surface(name)
        if len(blessed) > 1:
            db.record_ambiguous_occurrence(
                name, blessed, surface=name, source_path=rel_path or None
            )
            continue
        if blessed:
            if blessed[0] not in seen_ids:
                _link(blessed[0])
            continue

        # 3. Managed homonym: a disambiguation stub exists at the bare name (post-split). The bare
        #    label is an intentionally-shared weak alias across senses, so route to resolution
        #    rather than mint — the stub's existence is what distinguishes this from the bug case.
        if db.has_disambiguation_stub(name):
            cands = db.weak_alias_entities_for_surface(name) or db.resolve_label(name).ids
            db.record_ambiguous_occurrence(name, cands, surface=name, source_path=rel_path or None)
            continue

        # 4. Else mint. Consult the abbreviation heuristic ONLY when no entity already holds this
        #    surface as a weak alias: a stored weak alias means the surface is its own concept under
        #    the order-independent rule (demoted + surfaced as a merge candidate at the write seam),
        #    so it must not be silently absorbed by the heuristic mem-index.
        if not db.weak_alias_entities_for_surface(name):
            safe_keys = [_concept_key(name), _concept_key(_base_concept_name(name))]
            safe_keys.extend(_concept_key(alias) for alias in _safe_aliases_for_name(name))
            mem_canonical = next((mem_index[k] for k in safe_keys if k in mem_index), None)
            if mem_canonical is not None:
                ck = _concept_key(mem_canonical)
                if ck in seen_keys:
                    continue
                seen_keys.add(ck)
                in_run[_match_key(name)] = mem_canonical
                alias_candidates = [*_safe_aliases_for_name(name), *concept.aliases]
                if _concept_key(name) != ck:
                    alias_candidates.insert(0, name)
                result.append((mem_canonical, _validate_aliases(mem_canonical, alias_candidates)))
                continue

        # In-run dedup — same match_key was minted earlier in this pass.
        mk = _match_key(name)
        if mk in in_run:
            continue

        # Mint new concept.
        canonical = _base_concept_name(name)
        ck = _concept_key(canonical)
        if ck in seen_keys:
            continue
        seen_keys.add(ck)
        in_run[mk] = canonical
        alias_candidates = [*_safe_aliases_for_name(name), *concept.aliases]
        if _concept_key(name) != ck:
            alias_candidates.insert(0, name)
        # Aliases colliding with another entity's preferred label are dropped (and recorded as a
        # merge candidate) at the persistence seam in ingest_note, not here — keeping every
        # preferred-collision drop at one place so the candidate set is order-independent.
        result.append((canonical, _validate_aliases(canonical, alias_candidates)))

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
    rel_raw = rel_posix(path, config.vault)
    source_url = src_meta.get("source") or src_meta.get("url") or ""
    aliases = generate_aliases(title, "")  # source pages rarely have abbreviations

    # Build concept list as [[wikilinks]]
    concept_names = (
        canonical_concepts if canonical_concepts is not None else [c.name for c in result.concepts]
    )

    # Keep the raw name as display when the stem differs (e.g. "TCP/IP" -> TCPIP.md), so the
    # link both resolves and stays readable. Source pages skip the compile pipeline.
    def _concept_link(name: str) -> str:
        stem = sanitize_wikilink_target(name)
        return f"- [[{stem}|{name}]]" if stem != name else f"- [[{stem}]]"

    concept_lines = "\n".join(_concept_link(name) for name in concept_names[:8] if name.strip())

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


def write_source_content_md(
    source_id: str,
    source_type: str,
    title: str | None,
    segments: list,
    vault_dir: Path,
    *,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Assemble source segments into raw/<source_id>.md for the ingest pipeline."""
    raw_dir = vault_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{source_id}.md"

    lines: list[str] = []
    for seg in segments:
        loc = getattr(seg, "structural_locator", None)
        if loc:
            lines.append(f"## {loc}\n")
        lines.append(seg.text.strip())
        image_refs = getattr(seg, "image_refs", None) or []
        if image_refs:
            lines.append("")
            lines.append("### Media")
            lines.extend(f"- ![[{ref}]]" for ref in image_refs)
        lines.append("")

    meta: dict[str, object] = dict(metadata or {})
    meta["source_type"] = source_type
    if title:
        meta.setdefault("title", title)

    write_note(dest, meta, "\n".join(lines))
    return dest


def ingest_note(
    path: Path,
    config: Config,
    router: ModelRouter,
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
    rel_path = rel_posix(path, config.vault)
    existing = db.get_raw_by_hash(h)
    if existing and existing.path != rel_path:
        if (config.vault / existing.path).exists():
            # Same content at two real paths → genuine duplicate.
            log.info("Duplicate of %s, skipping %s", existing.path, path.name)
            return None
        # Old path is gone → this is a move/rename. Rekey derived state to the new
        # path, then fall through: the rekeyed row now lives at rel_path and the
        # already-ingested check below skips if analysis is current, or re-analyzes
        # if the row was status 'new'/'failed' or the ingest prompt version changed.
        log.info("Moved %s -> %s, rekeying", existing.path, rel_path)
        db.rekey_raw_path(existing.path, rel_path)

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
    # Strip ### Media sections — image paths are opaque to the LLM and waste tokens;
    # the embeds remain in the raw/ file for the human reader in Obsidian.
    body = re.sub(r"\n### Media\n(?:- !\[\[[^\]]*\]\]\n?)*", "\n", body)
    # Strip extractor OCR "picture text" gibberish before it reaches the model or
    # the RAG chunker below — same in-memory-only policy as the strips above.
    body = strip_image_text_blocks(body)
    if meta.get("source") or meta.get("url"):  # web clipper adds these
        body = _preprocess_web_clip(body)

    # Chunk + embed only when RAG store is wired in (Phase 2)
    if rag is not None:
        chunks = chunk_text(
            body, chunk_size=config.rag.chunk_size, overlap=config.rag.chunk_overlap
        )
        embed_ep = router.endpoint("embed")
        embeddings = embed_ep.client.embed_batch(chunks, model=embed_ep.model)
        rag.add_document(
            doc_id=rel_path,
            chunks=chunks,
            embeddings=embeddings,
            metadata={"source": rel_path, "type": "raw"},
        )

    # LLM analysis — use existing concept names so model can reuse canonical names
    fast = router.endpoint("fast")
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
    # Tracked sources (raw/<source_id>.md backed by source_segments) get segment-aligned
    # analysis units so extracted concepts can be attributed to known segment ids
    # (concept_occurrences) — at no extra LLM cost. `chunk_attribution` is filled
    # chunk_index -> [extracted concept name] and mapped to segments after canonicalization.
    source_id = path.stem
    source_segments = db.get_segments_for_source(source_id)
    chunk_units: list[tuple[str, list[str]]] | None = None
    chunk_attribution: dict[int, list[str]] | None = None
    if source_segments:
        # Pack segments into chapter-sized analysis units. fast_ctx is in TOKENS; a target of
        # ~1.5x fast_ctx CHARS (~0.4x fast_ctx tokens of input) keeps each call well within
        # context while grouping whole sections, so thin/peripheral segments (title page,
        # references) aren't analyzed in isolation — which would rate medium/low and, via the
        # min-quality aggregation + quality cap, halve the extracted concept count.
        unit_target = max(fast.ctx * 3 // 2, 4096)
        chunk_units = _build_segment_units(source_segments, unit_target)
        chunk_attribution = {}
    try:
        result: AnalysisResult = _analyze_body_with_checkpoints(
            body=body,
            existing_concepts=existing_topics,
            path=path,
            content_hash=h,
            router=router,
            config=config,
            db=db,
            force=force,
            prompt_contexts=prompt_contexts,
            source_type=source_type,
            units=chunk_units,
            attribution=chunk_attribution,
        )
    except Exception as e:
        log.error("Analysis failed for %s: %s", path.name, e, exc_info=True)
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
    max_concepts = config.pipeline.effective_max_concepts(source_type)
    existing_names = db.list_all_concept_names()
    if seed_concepts:
        seen_seed = {name.casefold() for name in seed_concepts}
        existing_names = [
            *seed_concepts,
            *(name for name in existing_names if name.casefold() not in seen_seed),
        ]
    # Only BLESSED aliases pre-fold a candidate to its canonical here; a weak (LLM-guessed) alias
    # surface re-extracted as a concept must reach _normalize_concepts and mint (order-independent).
    blessed_alias_map = db.list_blessed_alias_map()
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
        blessed_alias_map,
        seed_alias_map,
    )[:max_concepts]
    normalized = _normalize_concepts(
        concept_candidates,
        db,
        rel_path,
        seed_concepts,
        seed_alias_map,
    )
    canonical_names = [name for name, _ in normalized]
    for host_id, surface in db.replace_concepts_for_source(rel_path, canonical_names):
        _strip_demoted_alias_from_host(config, db, host_id, surface)
    for canonical, aliases in normalized:
        # Drop any extracted alias that is the preferred label of another active entity. This must
        # run HERE, after replace_concepts_for_source has minted every entity in this batch:
        # _normalize_concepts ran before any mint, so it cannot catch a collision between two
        # concepts of the SAME note (e.g. an alias "Knowledge Compounding" on "Dynamic Agentic ROI"
        # when "Knowledge Compounding" is itself an extracted concept). Left unfiltered, the label
        # resolves to two entities and that concept silently loses its article at compile.
        owner_id = db.entity_id_for_name(canonical) or ""
        persistable_aliases = []
        for a in _persistable_aliases(aliases):
            if db.alias_collides_with_preferred(_concept_key(a), owner_id):
                # The alias equals another entity's preferred label — dropping it keeps that label
                # unambiguous. Surface the pair as a merge candidate (seam (b), order #2: the
                # surface was already minted as its own concept, now a host wants it as an alias).
                winner_id = db.entity_id_for_name(a)
                if owner_id and winner_id is not None and winner_id != owner_id:
                    db.record_merge_candidate(
                        owner_id, winner_id, a, reason="alias-collides-preferred"
                    )
                continue
            persistable_aliases.append(a)
        if persistable_aliases:
            db.upsert_aliases(canonical, persistable_aliases)

    # Persist concept→segment links for tracked sources (enables get_source_passages).
    # Non-fatal: an attribution failure must not fail the ingest.
    if chunk_attribution is not None and chunk_units is not None:
        try:
            _persist_concept_occurrences(
                db, source_id, chunk_units, chunk_attribution, canonical_names
            )
        except Exception as e:  # noqa: BLE001
            log.warning("concept_occurrences attribution failed for %s: %s", path.name, e)

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

    _concept_preview = canonical_names[:3]
    if len(canonical_names) > 3:
        _concept_preview = canonical_names[:3] + [f"+{len(canonical_names) - 3} more"]
    log.info(
        "Ingested: %s (quality=%s, concepts=%s)",
        path.name,
        result.quality,
        _concept_preview,
    )
    return result


def ingest_all(
    config: Config,
    router: ModelRouter,
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
        # Lossless rebuild: restore entity ids + identity log before ingest re-mints by label.
        _restore_identity_from_index(config, db)
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
            router=router,
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
            rel_path = rel_posix(path, config.vault)
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
    fast: RoleEndpoint,
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
        client=fast.client,
        prompt=prompt,
        model_class=_TermLLMResponse,
        model=fast.model,
        system=_TERM_EXTRACTION_SYSTEM,
        num_ctx=fast.ctx,
        temperature=fast.temperature if fast.temperature is not None else 0,
        stage="term_extraction",
        model_role="fast",
        think=fast.think,
        options=fast.options,
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
        model=fast.model,
    )


_RELATION_PREDICATES = get_args(RelationCandidate.model_fields["predicate"].annotation)
_RELATION_PREDICATE_SET = frozenset(_RELATION_PREDICATES)

_RELATION_EXTRACTION_SYSTEM = (
    "You are a relation extractor. Given a list of known concepts and a text passage, "
    "extract directed relationships between those concepts only — do not invent concepts. "
    f"predicate must be exactly one of: {', '.join(_RELATION_PREDICATES)}. "
    "Include a short verbatim evidence quote from the passage and a confidence 0.0-1.0. "
    "Return JSON only, no explanation."
)


class _RelationLLMItem(_BaseModel):
    subject: str
    predicate: str
    object: str
    evidence: str = ""
    confidence: float = 0.8


class _RelationLLMResponse(_BaseModel):
    relations: list[_RelationLLMItem] = []


def extract_relations(
    segment: SourceSegment,
    concepts: list[str],
    fast: RoleEndpoint,
    config: Config,
) -> RelationExtractionResult:
    """Third LLM pass: extract directed concept-to-concept relations from a source segment."""
    if not concepts:
        return RelationExtractionResult(
            relations=[], source_segment_id=segment.id, model=fast.model
        )

    prompt = (
        f"Known concepts: {', '.join(concepts)}\n\n"
        "Extract directed relations between these concepts from the following text.\n"
        'Return JSON: {"relations": [{"subject": "...", "predicate": "...", "object": "...", '
        '"evidence": "...", "confidence": 0.8}]}\n\n'
        f"{segment.text[:4000]}"
    )
    llm_result = request_structured(
        client=fast.client,
        prompt=prompt,
        model_class=_RelationLLMResponse,
        model=fast.model,
        system=_RELATION_EXTRACTION_SYSTEM,
        num_ctx=fast.ctx,
        temperature=fast.temperature if fast.temperature is not None else 0,
        stage="relation_extraction",
        model_role="fast",
        think=fast.think,
        options=fast.options,
    )
    relations = []
    for r in llm_result.relations:
        if r.predicate not in _RELATION_PREDICATE_SET:
            log.debug("dropping relation with invalid predicate: %r", r.predicate)
            continue
        if not r.subject or not r.object:
            log.debug("dropping relation with empty subject/object")
            continue
        relations.append(
            RelationCandidate(
                subject=r.subject,
                predicate=r.predicate,  # type: ignore[arg-type]
                object=r.object,
                evidence=r.evidence,
                source_segment_id=segment.id,
                provenance="extracted",
                confidence=min(max(r.confidence, 0.0), 1.0),
            )
        )
    return RelationExtractionResult(
        relations=relations,
        source_segment_id=segment.id,
        model=fast.model,
    )
