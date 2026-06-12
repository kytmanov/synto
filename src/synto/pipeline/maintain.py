"""
Wiki maintenance — self-initiated health operations.

Used by `synto maintain` to:
  - Create stub drafts for broken wikilinks
  - Suggest orphan link fixes
  - Suggest concept merges for near-duplicates
  - Report source quality distribution
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import frontmatter as fm_lib

from ..concept_text import concept_key as _ck
from ..config import Config
from ..models import LintIssue, WikiArticleRecord
from ..sanitize import clean_display_name
from ..state import StateDB
from ..vault import (
    _mask_code_blocks,
    _restore_code_blocks,
    atomic_write,
    list_wiki_articles,
    normalize_wikilinks,
    parse_note,
    rename_wikilink_targets,
    sanitize_filename,
    sanitize_wikilink_target,
    write_note,
)

log = logging.getLogger(__name__)

_STUB_BODY = """\
> [!info] This is a stub article — referenced by other pages but no source material yet.

Add raw notes about this topic to `raw/` and run `synto compile` to generate a full article.
"""

_CONCEPT_MERGE_THRESHOLD = 0.7

_WIKILINK_REPAIR_RE = re.compile(r"\[\[([^\]|#]+?)(?:#([^\]|]*))?(?:\|([^\]]*))?\]\]")


@dataclass
class FixReport:
    repaired: int = 0
    repaired_links: list[tuple[str, str, str]] = field(default_factory=list)
    still_broken: list[LintIssue] = field(default_factory=list)
    skipped_files: list[Path] = field(default_factory=list)


def fix_broken_links(
    config: Config,
    db: StateDB,
    broken_link_issues: list[LintIssue],
    dry_run: bool = False,
) -> FixReport:
    """Repair broken wikilinks in place: dangling-punctuation links (``[[Phase II)]]`` →
    ``[[Phase II]]``) and alias links (``[[Alias]]`` → ``[[Canonical|Alias]]``).

    A link whose cleaned form resolves to a known page is healed and removed from still_broken.
    A dangling link whose cleaned form is still unknown is rewritten to the clean target (so the
    stub created for it matches) but stays in still_broken. Other unknown links fall through
    unchanged for stub creation.
    """
    report = FixReport()

    if not broken_link_issues:
        return report

    # Resolve links against lint's authoritative index (concept stems/titles, wiki-relative paths,
    # drafts, and unambiguous aliases) so this repair never diverges from what lint accepts as
    # valid — a second, narrower resolver here is what previously corrupted path-style links.
    from .lint import _build_title_index

    title_index = _build_title_index(config, db)

    # Group issues by source file for one write per file
    issues_by_file: dict[str, list[LintIssue]] = {}
    for issue in broken_link_issues:
        issues_by_file.setdefault(issue.path, []).append(issue)

    for rel_path, file_issues in issues_by_file.items():
        page = config.vault / rel_path
        try:
            meta, body = parse_note(page)
        except Exception as exc:
            log.warning("fix_broken_links: skipping %s — parse error: %s", rel_path, exc)
            report.skipped_files.append(page)
            report.still_broken.extend(file_issues)
            continue

        original_body = body
        # Mask code fences so repair doesn't rewrite [[...]] inside ```code``` or `inline`.
        masked_body, spans = _mask_code_blocks(body)
        repaired_in_file: list[tuple[str, str]] = []
        still_broken_in_file: list[LintIssue] = []

        for issue in file_issues:
            target = _extract_link_target(issue.description)
            if target is None:
                still_broken_in_file.append(issue)
                continue

            clean = clean_display_name(target)
            dangling = clean != target
            key = clean.lower()

            matched = title_index.get(key)
            if matched is not None:
                # Cleaned link resolves to a real page. Emit the form matching its namespace: a
                # genuine path link keeps its wiki-relative path; anything matched by stem/title/
                # alias is emitted as the sanitized filename stem (e.g. "TCP/IP" lives in TCPIP.md).
                relpath = matched.relative_to(config.wiki_dir).with_suffix("").as_posix()
                dest = relpath if ("/" in relpath and key == relpath.lower()) else matched.stem
                resolved = True
            elif dangling and "/" not in clean:
                # Unknown concept: rewrite toward the stem of the stub create_stubs will write
                # (sanitize_filename(clean).md), so the body link and that stub stay consistent.
                dest, resolved = sanitize_wikilink_target(clean), False
            else:
                # Non-dangling unknown, or a path-style link with no matching page — never sanitize
                # a path to a stem and never stub it (create_stubs also skips "/" targets).
                still_broken_in_file.append(issue)
                continue

            # Rewrite [[target ...]] occurrences (target is the dirty form actually in the body).
            # Emit the display-preserving form like ensure_wikilinks: [[dest|display]] when the
            # readable text differs from the link target, else [[dest]].
            def _make_rewriter(t: str, dest: str, default_display: str):
                def _rewrite(m: re.Match) -> str:
                    if m.group(1).strip().lower() != t.lower():
                        return m.group(0)
                    fragment = m.group(2)
                    explicit_display = m.group(3)
                    frag_part = f"#{fragment}" if fragment else ""
                    display = explicit_display if explicit_display is not None else default_display
                    if display == dest:
                        return f"[[{dest}{frag_part}]]"
                    return f"[[{dest}{frag_part}|{display}]]"

                return _rewrite

            new_masked = _WIKILINK_REPAIR_RE.sub(_make_rewriter(target, dest, clean), masked_body)
            changed = new_masked != masked_body
            if changed:
                new_form = f"[[{dest}]]" if clean == dest else f"[[{dest}|{clean}]]"
                repaired_in_file.append((f"[[{target}]]", new_form))
                masked_body = new_masked
            if not resolved:
                # Body link now points at the clean target; the stub will be created for it.
                still_broken_in_file.append(issue)
            elif not changed:
                # Resolved, but the only occurrence is unrewritable (e.g. inside a code fence).
                still_broken_in_file.append(issue)

        body = _restore_code_blocks(masked_body, spans)

        if body != original_body:
            if dry_run:
                for old, new in repaired_in_file:
                    log.info("dry-run: would rewrite %s → %s in %s", old, new, rel_path)
            else:
                write_note(page, meta, body)
            for old, new in repaired_in_file:
                report.repaired += 1
                report.repaired_links.append((rel_path, old, new))

        report.still_broken.extend(still_broken_in_file)

    return report


def normalize_published_alias_links(
    config: Config,
    db: StateDB,
    dry_run: bool = False,
) -> int:
    """Rewrite alias-form [[Alias]] links in published articles to [[Canonical|Alias]].

    Complements compile's normalize_wikilinks pass: articles published before an alias
    was registered never got that normalization. This pass runs on all published articles.

    Only unambiguous alias rewrites are applied. Returns number of files modified.
    """
    alias_map = db.list_alias_map()
    if not alias_map:
        return 0

    known_titles = {t.lower() for t, _ in list_wiki_articles(config.wiki_dir)}
    modified = 0

    for _title, path in list_wiki_articles(config.wiki_dir):
        try:
            meta, body = parse_note(path)
        except Exception as exc:
            log.warning("normalize_published_alias_links: skipping %s — %s", path.name, exc)
            continue

        new_body = normalize_wikilinks(body, alias_map, known_titles)
        if new_body == body:
            continue

        if dry_run:
            log.info("dry-run: would normalize alias links in %s", path.name)
        else:
            write_note(path, meta, new_body)
            log.info("Normalized alias links in %s", path.name)
        modified += 1

    return modified


class ConceptRenameError(Exception):
    """Raised when a concept rename cannot proceed (not found, name taken, etc.)."""


@dataclass
class RenameReport:
    old_name: str = ""
    new_name: str = ""
    files_moved: list[tuple[str, str]] = field(default_factory=list)
    links_rewritten: int = 0  # files in which inbound links were rewritten
    alias_kept: bool = False
    dry_run: bool = False


def _move_file(old: Path, new: Path) -> None:
    """Rename a file, handling case-only renames on case-insensitive filesystems."""
    if old == new:
        return
    case_only = old.parent == new.parent and old.name.casefold() == new.name.casefold()
    if case_only:
        tmp = old.with_name(f"{old.stem}.__synto_rename__{old.suffix}")
        old.rename(tmp)
        tmp.rename(new)
    else:
        # Defense in depth: preflight already rejects target collisions, but never let a
        # rename silently overwrite an unrelated file if that guarantee ever regresses.
        if new.exists():
            raise ConceptRenameError(f"Refusing to overwrite existing file: {new}")
        old.rename(new)


def rename_concept(
    config: Config,
    db: StateDB,
    old_name: str,
    new_name: str,
    *,
    keep_alias: bool = True,
    dry_run: bool = False,
) -> RenameReport:
    """Rename a concept: migrate DB identity, move its article, rewrite all inbound links.

    See plan/issue #29. The optional alias (old → new) is the durability mechanism:
    ingest's _normalize_concepts canonicalizes re-extracted old names back to the new
    one, so the rename survives future re-ingest. Dropping it can resurrect the old
    concept if a raw note still yields the old surface form.
    """
    old_name = old_name.strip()
    new_name = new_name.strip()
    report = RenameReport(
        old_name=old_name, new_name=new_name, alias_kept=keep_alias, dry_run=dry_run
    )

    # ── Preflight (no writes) ───────────────────────────────────────────────
    if not new_name or sanitize_filename(new_name) == "untitled":
        raise ConceptRenameError(f"Invalid new name: {new_name!r}")

    # Exact resolution only: a fuzzy substring match (e.g. "Net" -> "Network") would
    # silently rename the wrong concept. On a miss, reuse the fuzzy helper for a
    # suggestion only — never to act.
    resolved = db.find_concept_exact(old_name)
    if resolved is None:
        fuzzy = db.find_concept_by_name_or_alias(old_name)
        hint = f" Did you mean {fuzzy[0]!r}?" if fuzzy else ""
        raise ConceptRenameError(f"Concept not found: {old_name!r}.{hint}")
    canonical_old = resolved[0]

    case_only = canonical_old.casefold() == new_name.casefold()
    if canonical_old == new_name:
        raise ConceptRenameError(f"{old_name!r} is already named {new_name!r}")
    if not case_only and db.concept_name_exists_exact(new_name, exclude_concept=canonical_old):
        raise ConceptRenameError(
            f"Cannot rename to {new_name!r}: a concept/alias with that name already exists"
        )

    old_stem = sanitize_filename(canonical_old)
    new_stem = sanitize_filename(new_name)

    # The concept's own article(s): the deterministic file <dir>/<old_stem>.md, plus any
    # row whose title is exactly the canonical name. Matching on alias titles/stems (as
    # find_article_candidates does) is unsafe here — it could pull in an unrelated
    # article that merely shares one of this concept's aliases and clobber it.
    old_key = canonical_old.casefold()
    candidates = [
        art
        for art in db.list_articles()
        if art.kind != "synthesis"
        and (
            Path(art.path).stem.casefold() == old_stem.casefold() or art.title.casefold() == old_key
        )
    ]
    # Validate everything before any write so a rename can't fail partway through.
    moves: list[tuple[Path, Path, str, str]] = []  # (old_path, new_path, new_rel, old_rel)
    seen_targets: set[str] = set()
    for art in candidates:
        old_rel = art.path
        old_path = config.vault / old_rel
        new_path = old_path.parent / f"{new_stem}.md"
        new_rel = str(new_path.relative_to(config.vault))
        if not old_path.exists():
            raise ConceptRenameError(
                f"Tracked article {old_rel} is missing on disk; "
                "reconcile with `synto lint` before renaming."
            )
        if not case_only and new_path.exists():
            raise ConceptRenameError(f"Target file already exists: {new_rel}")
        existing = db.get_article(new_rel)
        if existing is not None and new_rel != old_rel:
            raise ConceptRenameError(f"A tracked article already exists at {new_rel}")
        if new_rel in seen_targets:
            raise ConceptRenameError(f"Multiple articles would collide at {new_rel}")
        seen_targets.add(new_rel)
        moves.append((old_path, new_path, new_rel, old_rel))

    if dry_run:
        log.info("dry-run: would rename concept %r → %r", canonical_old, new_name)
        for _old_path, _new_path, new_rel, old_rel in moves:
            log.info("dry-run: would move %s → %s", old_rel, new_rel)
        report.files_moved = [(old_rel, new_rel) for _o, _n, new_rel, old_rel in moves]
        # Count pages whose inbound links would change, without writing.
        report.links_rewritten = _rewrite_inbound_links(
            config, db, old_stem, new_stem, new_name, dry_run=True
        )
        return report

    # ── Mutations ───────────────────────────────────────────────────────────
    for old_path, new_path, new_rel, old_rel in moves:
        _move_file(old_path, new_path)
        meta, body = parse_note(new_path)
        meta["title"] = new_name
        meta["updated"] = datetime.now().strftime("%Y-%m-%d")
        if keep_alias:
            existing = list(meta.get("aliases") or [])
            if not any(a.casefold() == canonical_old.casefold() for a in existing):
                meta["aliases"] = [*existing, canonical_old]
        # Body is unchanged (frontmatter-only edit), so content_hash is preserved by
        # update_article_identity — this keeps manual-edit protection on a hand-fixed page.
        write_note(new_path, meta, body)
        db.update_article_identity(old_rel, new_rel, new_name)
        report.files_moved.append((old_rel, new_rel))

    db.rename_concept(canonical_old, new_name)
    if keep_alias:
        # Bless the kept old name (source='rename') so a re-extracted old name links to the new
        # concept under the order-independent ingest rule instead of re-minting (rename durability).
        db.upsert_aliases(new_name, [canonical_old], source="rename")

    report.links_rewritten = _rewrite_inbound_links(
        config, db, old_stem, new_stem, new_name, dry_run=False
    )
    return report


def _rewrite_inbound_links(
    config: Config,
    db: StateDB,
    old_stem: str,
    new_stem: str,
    new_name: str,
    *,
    dry_run: bool,
) -> int:
    """Repoint every wikilink resolving to old_stem across the wiki tree.

    Covers concept articles, drafts, sources/, queries/, synthesis/ so no broken
    links are left anywhere. raw/ is immutable and untouched. content_hash is
    refreshed only for tracked rows — _write_fixed_note no-ops the DB update for
    untracked pages, so the broad rewrite scope stays cheap.
    """
    from .lint import _write_fixed_note

    changed = 0
    for page in sorted(config.wiki_dir.rglob("*.md")):
        try:
            meta, body = parse_note(page)
        except Exception as exc:
            log.warning("rename: skipping %s — parse error: %s", page.name, exc)
            continue
        new_body = rename_wikilink_targets(body, old_stem, new_stem, new_name)
        if new_body == body:
            continue
        changed += 1
        if dry_run:
            log.info("dry-run: would rewrite links in %s", page.name)
            continue
        rel_path = str(page.relative_to(config.vault))
        _write_fixed_note(page, rel_path, meta, new_body, db)
    return changed


def create_stubs(
    config: Config,
    db: StateDB,
    broken_link_issues: list[LintIssue] | None = None,
    max_stubs: int = 5,
) -> list[Path]:
    """
    Create stub drafts for broken wikilinks.

    Finds [[Target]] references that have no matching article, creates placeholder
    drafts and registers them in the stubs table so compile can pick them up.

    Pass broken_link_issues to avoid re-running lint. If None, lint runs internally.
    """
    if broken_link_issues is None:
        from .lint import run_lint

        result = run_lint(config, db)
        broken_link_issues = [i for i in result.issues if i.issue_type == "broken_link"]

    # Extract target concept names from broken link descriptions
    # LintIssue description format: "[[Target]] not found" or similar
    created: list[Path] = []
    seen: set[str] = set()

    for issue in broken_link_issues:
        if len(created) >= max_stubs:
            log.info("Stub cap (%d) reached — stopping", max_stubs)
            break

        # Extract target from description (e.g. "[[Quantum Computing]] not found")
        target = _extract_link_target(issue.description)
        if not target:
            # Fall back to path stem
            target = Path(issue.path).stem
        # Strip trailing .md — model sometimes generates [[raw-note.md]] wikilinks
        if target.lower().endswith(".md"):
            target = target[:-3]
        if not target:
            continue

        # Path-prefixed targets like sources/SomePaper are pipeline-managed pages,
        # not concept stubs. sanitize_filename would strip the slash and produce a
        # wrong name (sourcesSomePaper). Skip them — the fix is to run synto run.
        if "/" in target:
            log.debug("Skipping path-fragment broken link target: %s", target)
            continue

        # Drop dangling/unbalanced punctuation (e.g. "Phase II)") before any dedup check or write,
        # so a malformed link can't spawn a stub that diverges from the real "Phase II.md".
        target = clean_display_name(target)
        if not target:
            continue

        if target in seen:
            continue
        seen.add(target)

        # Skip if already has a stub, draft, or published article
        if db.has_stub(target):
            continue
        safe_name = sanitize_filename(target)
        draft_path = config.drafts_dir / f"{safe_name}.md"
        wiki_path = config.wiki_dir / f"{safe_name}.md"
        if draft_path.exists() or wiki_path.exists():
            continue

        # Register in stubs table
        db.add_stub(target, source="auto")

        # Write placeholder draft
        config.drafts_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "title": target,
            "status": "stub",
            "tags": ["stub"],
            "sources": [],
            "confidence": 0.0,
            "created": datetime.now().strftime("%Y-%m-%d"),
            "updated": datetime.now().strftime("%Y-%m-%d"),
        }
        post = fm_lib.Post(_STUB_BODY, **meta)
        atomic_write(draft_path, fm_lib.dumps(post))
        created.append(draft_path)
        log.info("Stub created: %s", draft_path.name)

    return created


def _extract_link_target(description: str) -> str | None:
    """Extract [[Target]] from a lint issue description string."""
    match = re.search(r"\[\[([^\]]+)\]\]", description)
    return match.group(1) if match else None


def suggest_orphan_links(config: Config, db: StateDB) -> list[tuple[str, list[str]]]:
    """
    For each orphan article, find other articles that mention its title unlinked.

    Returns list of (orphan_title, [paths_that_mention_it]).
    """
    from .lint import run_lint

    result = run_lint(config, db)
    orphan_issues = [i for i in result.issues if i.issue_type == "orphan"]
    if not orphan_issues:
        return []

    # Load all published article bodies
    wiki_pages: dict[str, str] = {}
    if config.wiki_dir.exists():
        for p in config.wiki_dir.rglob("*.md"):
            if ".drafts" in p.parts:
                continue
            try:
                meta, body = parse_note(p)
                wiki_pages[str(p.relative_to(config.vault))] = body
            except Exception:
                pass

    suggestions = []
    for issue in orphan_issues:
        orphan_path = config.vault / issue.path
        try:
            meta, _ = parse_note(orphan_path)
            orphan_title = meta.get("title", orphan_path.stem)
        except Exception:
            orphan_title = orphan_path.stem

        # Find pages that mention the orphan title in plain text (not as wikilink)
        mentions = []
        title_pattern = re.compile(
            r"(?<!\[\[)\b" + re.escape(orphan_title) + r"\b(?!\]\])",
            re.IGNORECASE,
        )
        for page_path, body in wiki_pages.items():
            if page_path == issue.path:
                continue
            if title_pattern.search(body):
                mentions.append(page_path)

        if mentions:
            suggestions.append((orphan_title, mentions))

    return suggestions


def suggest_concept_merges(config: Config, db: StateDB) -> list[tuple[str, str, float]]:
    """Find near-duplicate concept pairs via Jaccard similarity and match_key collisions.

    Returns list of (concept_a, concept_b, similarity_score), sorted by score descending.
    Score = 1.0 for match_key fold collisions (plural/singular, issue #54);
    Jaccard threshold applies to token-overlap pairs.
    No LLM required.
    """
    seen: set[tuple[str, str]] = set()
    suggestions: list[tuple[str, str, float]] = []

    # Signal 1: match_key collisions (fold-exact — score 1.0).
    for _a_id, _b_id, _mk in db.find_match_key_collisions():
        a_label = db.preferred_label_for_entity(_a_id)
        b_label = db.preferred_label_for_entity(_b_id)
        if a_label and b_label:
            key = (min(a_label, b_label), max(a_label, b_label))
            if key not in seen:
                seen.add(key)
                suggestions.append((a_label, b_label, 1.0))

    # Signal 2: Jaccard token similarity.
    concepts = db.list_all_concept_names()
    if len(concepts) >= 2:

        def tokenize(name: str) -> frozenset[str]:
            tokens = (re.sub(r"\W+", "", t) for t in re.split(r"[\s\-_]+", name.lower()))
            return frozenset(t for t in tokens if len(t) > 1)

        tokenized = [(c, tokenize(c)) for c in concepts]
        for i, (a, tokens_a) in enumerate(tokenized):
            for b, tokens_b in tokenized[i + 1 :]:
                if not tokens_a or not tokens_b:
                    continue
                intersection = len(tokens_a & tokens_b)
                union = len(tokens_a | tokens_b)
                if union == 0:
                    continue
                score = intersection / union
                if score >= _CONCEPT_MERGE_THRESHOLD:
                    key = (min(a, b), max(a, b))
                    if key not in seen:
                        seen.add(key)
                        suggestions.append((a, b, round(score, 2)))

    suggestions.sort(key=lambda x: -x[2])
    return suggestions


def suggest_concept_splits(config: Config, db: StateDB) -> list[tuple[str, str]]:
    """Flag entities whose source set has low internal token overlap (bimodal).

    Returns [(entity_name, reason_str)] for entities that may cover multiple unrelated
    subjects. Uses the source summary files from wiki/sources/ (first 200 chars).
    Only checks entities with ≥3 sources. Pure heuristic — no LLM.
    """
    results: list[tuple[str, str]] = []
    concepts = db.list_all_concept_names()
    sources_dir = config.wiki_dir / "sources"

    def _load_source_summary(source_path: str) -> str:
        """Read first 200 chars of a source summary from wiki/sources/."""
        stem = sanitize_filename(Path(source_path).stem)
        candidate = sources_dir / f"{stem}.md"
        if candidate.exists():
            try:
                _, body = parse_note(candidate)
                return body[:200]
            except Exception:
                pass
        return ""

    def _tokenize_body(text: str) -> frozenset[str]:
        tokens = re.split(r"\W+", text.lower())
        return frozenset(t for t in tokens if len(t) > 3)

    for name in concepts:
        sources = db.get_sources_for_concept(name)
        if len(sources) < 3:
            continue
        token_sets = [_tokenize_body(_load_source_summary(s)) for s in sources]
        token_sets = [ts for ts in token_sets if ts]
        if len(token_sets) < 3:
            continue

        # Pairwise Jaccard.
        scores: list[float] = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                u = token_sets[i] | token_sets[j]
                if not u:
                    continue
                scores.append(len(token_sets[i] & token_sets[j]) / len(u))

        if not scores:
            continue
        min_score = min(scores)
        avg_score = sum(scores) / len(scores)
        # Bimodal signal: very low min similarity but non-trivial average
        # implies at least one pair of sources is unrelated.
        if min_score < 0.05 and avg_score > 0.1:
            results.append((name, f"min source similarity {min_score:.2f}, avg {avg_score:.2f}"))

    return results


class ConceptMergeError(Exception):
    """Raised when a concept merge cannot proceed."""


class ConceptSplitError(Exception):
    """Raised when a concept split cannot proceed."""


class ConceptUnmergeError(Exception):
    """Raised when a concept unmerge cannot proceed."""


@dataclass
class MergeReport:
    loser: str = ""
    winner: str = ""
    files_retired: list[str] = field(default_factory=list)
    links_rewritten: int = 0
    labels_absorbed: list[str] = field(default_factory=list)
    dry_run: bool = False


@dataclass
class SplitReport:
    original: str = ""
    senses: list[dict] = field(default_factory=list)
    stub_path: str = ""
    dry_run: bool = False


@dataclass
class UnmergeReport:
    winner: str = ""
    loser: str = ""
    sources_restored: list[str] = field(default_factory=list)
    stub_path: str = ""


def _retire_article(article_path: Path, drafts_dir: Path) -> str:
    """Move article to .drafts/ with a date suffix. Returns the new relative path string."""
    drafts_dir.mkdir(parents=True, exist_ok=True)
    stem = article_path.stem
    date_str = datetime.now().strftime("%Y%m%d")
    target = drafts_dir / f"{stem}_retired_{date_str}.md"
    # Avoid clobbering if called multiple times on the same day.
    counter = 0
    while target.exists():
        counter += 1
        target = drafts_dir / f"{stem}_retired_{date_str}_{counter}.md"
    article_path.rename(target)
    return str(target)


def merge_concepts(
    config: Config,
    db: StateDB,
    loser_name: str,
    winner_name: str,
    *,
    absorb_edits: bool = False,
    dry_run: bool = False,
) -> MergeReport:
    """Merge loser concept into winner: DB identity, retire article, rewrite links.

    absorb_edits=True appends loser's manually-edited body into winner's article
    before retiring.  Without it, a body-hash mismatch raises ConceptMergeError.
    """
    import hashlib

    loser_name = loser_name.strip()
    winner_name = winner_name.strip()
    report = MergeReport(loser=loser_name, winner=winner_name, dry_run=dry_run)

    # ── Preflight ──────────────────────────────────────────────────────────────
    if loser_name == winner_name:
        raise ConceptMergeError(f"Cannot merge {loser_name!r} into itself")

    loser_id = db.entity_id_for_name(loser_name)
    winner_id = db.entity_id_for_name(winner_name)
    if loser_id is None:
        raise ConceptMergeError(f"Concept not found: {loser_name!r}")
    if winner_id is None:
        raise ConceptMergeError(f"Concept not found: {winner_name!r}")
    if loser_id == winner_id:
        raise ConceptMergeError(f"{loser_name!r} and {winner_name!r} are the same entity")

    loser_stem = sanitize_filename(loser_name)
    winner_stem = sanitize_filename(winner_name)

    # Locate loser article (same stem-match logic as rename_concept).
    loser_key = loser_name.casefold()
    loser_articles = [
        art
        for art in db.list_articles()
        if art.kind not in ("synthesis", "disambiguation")
        and (
            Path(art.path).stem.casefold() == loser_stem.casefold()
            or art.title.casefold() == loser_key
        )
    ]

    # Manual-edit check.
    for art in loser_articles:
        art_path = config.vault / art.path
        if art_path.exists():
            _, body = parse_note(art_path)
            body_hash = hashlib.sha256(body.encode()).hexdigest()
            if art.content_hash and body_hash != art.content_hash and not absorb_edits:
                raise ConceptMergeError(
                    f"Article {art.path!r} has been manually edited "
                    "(on-disk hash ≠ DB content_hash). "
                    "Pass --absorb-edits to carry the edited body into the winner."
                )

    if dry_run:
        report.files_retired = [art.path for art in loser_articles]
        report.links_rewritten = _rewrite_inbound_links(
            config, db, loser_stem, winner_stem, winner_name, dry_run=True
        )
        # Must NOT call db.merge_entities here — it commits in a real transaction
        # and would corrupt the DB/vault on a dry run. Derive the absorbed labels
        # from current DB state without mutating anything. The real merge drops any
        # label whose key matches the winner's, so filter those out to keep the
        # preview honest.
        winner_key = _ck(winner_name)
        report.labels_absorbed = [
            lbl for lbl in (loser_name, *db.get_aliases(loser_name)) if _ck(lbl) != winner_key
        ]
        return report

    # ── Mutations ──────────────────────────────────────────────────────────────
    # Absorb loser's manually-edited body into winner if requested.
    if absorb_edits:
        winner_articles = [
            art
            for art in db.list_articles()
            if art.kind not in ("synthesis", "disambiguation")
            and (
                Path(art.path).stem.casefold() == winner_stem.casefold()
                or art.title.casefold() == winner_name.casefold()
            )
        ]
        for loser_art in loser_articles:
            loser_path = config.vault / loser_art.path
            if not loser_path.exists():
                continue
            _, loser_body = parse_note(loser_path)
            body_hash = hashlib.sha256(loser_body.encode()).hexdigest()
            if loser_art.content_hash and body_hash == loser_art.content_hash:
                continue
            # Append to winner's article.
            for winner_art in winner_articles:
                winner_path = config.vault / winner_art.path
                if not winner_path.exists():
                    continue
                w_meta, w_body = parse_note(winner_path)
                w_body = w_body.rstrip() + f"\n\n## Absorbed from {loser_name}\n\n{loser_body}"
                write_note(winner_path, w_meta, w_body)
                break

    result = db.merge_entities(winner_name, loser_name)
    report.labels_absorbed = result["labels_absorbed"]

    # Retire loser articles. The on-disk file moves to .drafts/; the tracked row must
    # also go — merge_entities only moves DB identity, so a surviving 'published' row would
    # point at a path that no longer exists, emitting a dangling [[Loser]] in the index and
    # breaking query routing / pack export / MCP serve. Delete the row unconditionally: it
    # dangles whether or not the file was still on disk.
    drafts_dir = config.wiki_dir / ".drafts"
    for art in loser_articles:
        art_path = config.vault / art.path
        if art_path.exists():
            retired = _retire_article(art_path, drafts_dir)
            report.files_retired.append(art.path)
            log.info("merge: retired %s → %s", art.path, retired)
        db.delete_article(art.path)

    # Update winner article frontmatter: add absorbed labels to aliases:.
    winner_articles_post = [
        art
        for art in db.list_articles()
        if art.kind not in ("synthesis", "disambiguation")
        and (
            Path(art.path).stem.casefold() == winner_stem.casefold()
            or art.title.casefold() == winner_name.casefold()
        )
    ]
    for art in winner_articles_post:
        art_path = config.vault / art.path
        if not art_path.exists():
            continue
        meta, body = parse_note(art_path)
        existing_aliases = list(meta.get("aliases") or [])
        # Dedup against existing aliases AND within this batch so a repeated absorbed
        # label can never write a duplicate alias line into the published frontmatter.
        seen_aliases = {a.casefold() for a in existing_aliases}
        added: list[str] = []
        for lbl in result["labels_absorbed"]:
            key = lbl.casefold()
            if key in seen_aliases or key == winner_name.casefold():
                continue
            seen_aliases.add(key)
            added.append(lbl)
        if added:
            meta["aliases"] = [*existing_aliases, *added]
            write_note(art_path, meta, body)

    report.links_rewritten = _rewrite_inbound_links(
        config, db, loser_stem, winner_stem, winner_name, dry_run=False
    )
    return report


def split_concept(
    config: Config,
    db: StateDB,
    entity_name: str,
    senses: list[tuple[str, list[str]]],
    *,
    absorb_edits: bool = False,
    dry_run: bool = False,
) -> SplitReport:
    """Split entity_name into multiple senses, each owning a subset of its sources.

    senses is a list of (new_name, [source_paths]).
    Creates stub articles for each sense plus a disambiguation stub at the bare label.

    The original article's body is carried into the primary (first) sense. If that
    article was manually edited (on-disk hash ≠ DB content_hash), the split refuses
    unless absorb_edits=True, mirroring merge's gate so Decision 19 is symmetric.
    """
    import hashlib

    entity_name = entity_name.strip()
    report = SplitReport(original=entity_name, dry_run=dry_run)

    # ── Preflight ──────────────────────────────────────────────────────────────
    entity_id = db.entity_id_for_name(entity_name)
    if entity_id is None:
        raise ConceptSplitError(f"Concept not found: {entity_name!r}")
    if len(senses) < 2:
        raise ConceptSplitError("split_concept requires at least 2 senses")

    bare_key = _ck(entity_name)
    for new_name, srcs in senses:
        if not new_name:
            raise ConceptSplitError("Sense name cannot be empty")
        if bare_key and _ck(new_name) == bare_key:
            raise ConceptSplitError(
                f"Sense {new_name!r} cannot reuse the original label {entity_name!r} — "
                "that name is reserved for the disambiguation page."
            )
        existing_id = db.entity_id_for_name(new_name)
        if existing_id is not None and existing_id != entity_id:
            raise ConceptSplitError(
                f"Cannot create sense {new_name!r}: a different concept already uses that name"
            )

    # Manual-edit gate: the original body is about to be retired (and carried into the
    # primary sense). Refuse to silently discard hand edits unless told to absorb them.
    orig_stem = sanitize_filename(entity_name)
    orig_articles = [
        art
        for art in db.list_articles()
        if art.kind not in ("synthesis", "disambiguation")
        and (
            Path(art.path).stem.casefold() == orig_stem.casefold()
            or art.title.casefold() == entity_name.casefold()
        )
    ]
    for art in orig_articles:
        art_path = config.vault / art.path
        if art_path.exists():
            _, body = parse_note(art_path)
            body_hash = hashlib.sha256(body.encode()).hexdigest()
            if art.content_hash and body_hash != art.content_hash and not absorb_edits:
                raise ConceptSplitError(
                    f"Article {art.path!r} has been manually edited "
                    "(on-disk hash ≠ DB content_hash). "
                    "Pass --absorb-edits to carry the edited body into the primary sense."
                )

    sense_dicts = [{"name": name, "sources": srcs} for name, srcs in senses]

    if dry_run:
        report.senses = [{"name": name, "sources": srcs} for name, srcs in senses]
        report.stub_path = f"wiki/{sanitize_filename(entity_name)}.md"
        return report

    # ── Mutations ──────────────────────────────────────────────────────────────
    result = db.split_entity(entity_name, sense_dicts)

    # Retire the original article (located + hash-gated in the preflight above).
    # split_entity does not move article rows, so orig_articles is still valid here.
    orig_body = ""
    drafts_dir = config.wiki_dir / ".drafts"
    primary_sense = result["senses"][0]["name"]
    for art in orig_articles:
        art_path = config.vault / art.path
        if art_path.exists():
            _, orig_body = parse_note(art_path)
            _retire_article(art_path, drafts_dir)

    # Create stub articles for each sense.
    for sense in result["senses"]:
        new_name = sense["name"]
        stem = sanitize_filename(new_name)
        stub_path = config.wiki_dir / f"{stem}.md"
        body = orig_body if new_name == primary_sense and orig_body else ""
        if not stub_path.exists():
            write_note(
                stub_path,
                {"title": new_name, "aliases": [entity_name]},
                body,
            )
            log.info("split: created stub %s", stub_path.name)
        report.senses.append({"name": new_name, "path": str(stub_path.relative_to(config.vault))})
        db.upsert_article(
            WikiArticleRecord(
                path=str(stub_path.relative_to(config.vault)),
                title=new_name,
                sources=sense["sources"],
                content_hash="",
                status="draft",
            )
        )

    # Create disambiguation stub.
    if result["stub_needed"]:
        dis_stem = sanitize_filename(entity_name)
        dis_path = config.wiki_dir / f"{dis_stem}.md"
        sense_links = "\n".join(f"- [[{s['name']}]]" for s in result["senses"])
        dis_body = f"**{entity_name}** may refer to:\n\n{sense_links}\n"
        write_note(dis_path, {"title": entity_name, "kind": "disambiguation"}, dis_body)
        report.stub_path = str(dis_path.relative_to(config.vault))
        db.upsert_article(
            WikiArticleRecord(
                path=report.stub_path,
                title=entity_name,
                sources=[],
                content_hash="",
                status="published",
                kind="disambiguation",
            )
        )
        log.info("split: created disambiguation stub %s", dis_path.name)

    return report


def unmerge_concept(config: Config, db: StateDB, merged_name: str) -> UnmergeReport:
    """Reverse the most recent merge that retired ``merged_name``, restoring it as an entity.

    Identity is restored from the merge log (see StateDB.unmerge_entities for the full
    reversibility contract). The loser's article is recreated as an empty stub and reseeded
    pending, so the next ``synto compile`` regenerates it. Wiki links that were repointed to
    the winner during the merge are NOT reverted — that mapping is not safely reversible.
    """
    merged_name = merged_name.strip()
    loser_id = db.get_merged_entity_id(merged_name)
    if loser_id is None:
        raise ConceptUnmergeError(f"No merged concept named {merged_name!r} to unmerge.")

    try:
        result = db.unmerge_entities(loser_id)
    except ValueError as exc:
        raise ConceptUnmergeError(str(exc)) from exc

    report = UnmergeReport(
        winner=result["winner"],
        loser=result["loser"],
        sources_restored=result["sources_restored"],
    )

    # Recreate a stub so the reactivated entity has a page again (next compile fills it).
    stub_path = config.wiki_dir / f"{sanitize_filename(result['loser'])}.md"
    if not stub_path.exists():
        write_note(stub_path, {"title": result["loser"]}, "")
        report.stub_path = str(stub_path.relative_to(config.vault))
        db.upsert_article(
            WikiArticleRecord(
                path=report.stub_path,
                title=result["loser"],
                sources=db.get_sources_for_concept(result["loser"]),
                content_hash="",
                status="draft",
            )
        )
        log.info("unmerge: recreated stub %s", stub_path.name)

    return report
