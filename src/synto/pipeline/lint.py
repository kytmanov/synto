"""
Lint pipeline: all structural checks, no LLM required.

Checks:
  orphan           — concept page with no inbound [[wikilinks]] from other pages
  broken_link      — [[Target]] in body that resolves to no file
  missing_frontmatter — required fields (title, status, tags) absent
  stale            — file hash on disk != DB content_hash (manually edited)
  low_confidence   — confidence < LOW_CONFIDENCE_THRESHOLD
  invalid_tag      — tag that is not a valid Obsidian tag name

Fix mode (--fix):
  Auto-fixes missing_frontmatter and invalid_tag fields.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..markdown_math import (
    mask_markdown_regions,
    restore_markdown_regions,
    sanitize_obsidian_math,
)
from ..models import _ADVISORY_ISSUE_TYPES, LintIssue, LintResult
from ..sanitize import sanitize_tag, sanitize_tags
from ..state import StateDB
from ..vault import _MEDIA_EXTENSIONS, extract_wikilinks, parse_note, sanitize_filename, write_note

log = logging.getLogger(__name__)

_REQUIRED_FIELDS: frozenset[str] = frozenset({"title", "status", "tags"})
_LOW_CONFIDENCE_THRESHOLD = 0.3

# Pages excluded from orphan + link checks (meta / system pages)
_SYSTEM_STEMS = frozenset({"index", "log"})

# Inline hashtag pattern — Obsidian indexes these as tags
_INLINE_TAG_RE = re.compile(r"(?<![/\w])#([a-zA-Z][^\s#\]]*)")

# Common LLM markdown slip: reference-style link syntax with no URL, e.g.
# [astronomy] or [Zodiac] (text), which Obsidian will not resolve as a link.
_MALFORMED_BRACKET_LINK_RE = re.compile(r"(?<![!\[])\[(?!\[)([^\]\n]+)\](?![\[(])")
_MALFORMED_EMBED_RE = re.compile(r"(?<!\S)!([^\s\[]+\.(?:pdf|png|jpe?g|gif|svg|webp))", re.I)
_OBSIDIAN_EMBED_RE = re.compile(r"!\[\[[^\]]+\.(?:pdf|png|jpe?g|gif|svg|webp)\]\]", re.I)
_PLAIN_CITATION_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\](?!\()")
_EMBED_TARGET_RE = re.compile(r"!\[\[([^\]|#]+)")
_DISPLAY_LATEX_RE = re.compile(r"\\{1,2}\[.*?\\{1,2}\]", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"\$[^$\n]+\$")
_DISPLAY_MATH_DOLLAR_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_BARE_LATEX_LINE_RE = re.compile(
    r"(?m)^(?![ \t]*(?:#|>|[-*+] |\d+\. |\|))[ \t]*\\{1,2}[A-Za-z{_].*$"
)

# Vault-internal directory names that LLMs sometimes write as wikilinks
_VAULT_DIRS = frozenset({"wiki", "raw", "source", "sources", "queries", ".drafts", ".olw"})

# ── Helpers ───────────────────────────────────────────────────────────────────


def _check_tags(
    rel_path: str,
    meta: dict,
    issues: list[LintIssue],
    fix: bool,
    page: Path,
    body: str,
) -> None:
    """Emit invalid_tag issues and optionally fix them. Shared by all page loops."""
    tags = meta.get("tags", [])
    if not isinstance(tags, list):
        issues.append(
            LintIssue(
                path=rel_path,
                issue_type="invalid_tag",
                description=f"tags field is not a list: {tags!r}",
                suggestion="Convert tags to a YAML list.",
                auto_fixable=True,
            )
        )
        if fix:
            meta["tags"] = sanitize_tags([str(tags)])
            write_note(page, meta, body)
    else:
        non_str = [t for t in tags if not isinstance(t, str)]
        str_tags = [t for t in tags if isinstance(t, str)]
        # also catch empty strings — sanitize_tags drops them but t == sanitize_tag(t) == ""
        invalid = non_str + [t for t in str_tags if not sanitize_tag(t) or t != sanitize_tag(t)]
        if invalid:
            issues.append(
                LintIssue(
                    path=rel_path,
                    issue_type="invalid_tag",
                    description=f"Invalid tags: {', '.join(str(t) for t in invalid)}",
                    suggestion=f"Sanitized: {', '.join(sanitize_tag(str(t)) for t in invalid)}",
                    auto_fixable=True,
                )
            )
            if fix:
                meta["tags"] = sanitize_tags([str(t) for t in tags])
                write_note(page, meta, body)


def _body_hash(body: str) -> str:
    """Hash page body only (matches compile._content_hash — excludes frontmatter)."""
    return hashlib.sha256(body.encode()).hexdigest()


def _check_malformed_links(rel_path: str, body: str, issues: list[LintIssue]) -> None:
    # Mask code fences/inline code (and math/links) so code samples like
    # ["apps/*", "packages/*"] in a ```json fence are never flagged (#93), then
    # strip the remaining math forms the mask doesn't cover so $H_i \in [0, 1]$
    # and \[...\] display math do not trigger false positives.
    masked, _ = mask_markdown_regions(body)
    clean = _DISPLAY_MATH_DOLLAR_RE.sub(" ", masked)
    clean = _INLINE_MATH_RE.sub(" ", clean)
    clean = _DISPLAY_LATEX_RE.sub(" ", clean)

    seen: set[str] = set()
    for match in _MALFORMED_BRACKET_LINK_RE.finditer(clean):
        if match.start() > 0 and clean[match.start() - 1] == "\\":
            continue
        text = match.group(1).strip()
        if not text or text in seen:
            continue
        if text.startswith("!"):
            continue
        if text.startswith("S") and re.fullmatch(r"S\d+(?:\s*,\s*S\d+)*", text):
            continue
        seen.add(text)
        issues.append(
            LintIssue(
                path=rel_path,
                issue_type="malformed_link",
                description=f"[{text}] is not a valid Markdown or Obsidian link",
                suggestion=f"Use [[{text}]] for an Obsidian link or remove the brackets.",
                auto_fixable=False,
            )
        )

    for line in masked.splitlines():
        stripped = line.rstrip()
        if not stripped.endswith("[") or stripped.endswith(("[[", "![", "![[")):
            continue
        if stripped.startswith(("\\[", "\\\\[")):
            continue
        if "dangling_open_bracket" in seen:
            continue
        seen.add("dangling_open_bracket")
        issues.append(
            LintIssue(
                path=rel_path,
                issue_type="malformed_link",
                description="Dangling '[' at end of line is not a valid Markdown or Obsidian link",
                suggestion="Complete the link target or remove the trailing bracket.",
                auto_fixable=False,
            )
        )


def _check_broken_wikilinks(
    rel_path: str,
    body: str,
    title_index: dict[str, Path],
    issues: list[LintIssue],
) -> None:
    seen_broken: set[str] = set()
    for link in extract_wikilinks(body):
        if link.lower() in title_index or link.lower() in seen_broken:
            continue
        # Skip bare URLs and vault path fragments accidentally wrapped in [[...]]
        is_url = link.startswith(("http://", "https://")) or (
            "/" in link and "." in link.split("/")[0]
        )
        stripped_link = link.rstrip("/")
        is_path_fragment = stripped_link in _VAULT_DIRS or (
            not link.lower().startswith("sources/")
            and link.startswith(tuple(d + "/" for d in _VAULT_DIRS))
        )
        if is_url or is_path_fragment:
            continue
        seen_broken.add(link.lower())
        if link.lower().startswith("sources/"):
            suggestion = "Run `synto run` to regenerate source summary pages, or remove the link."
        else:
            suggestion = f"Create a page for '{link}' or remove the link."
        issues.append(
            LintIssue(
                path=rel_path,
                issue_type="broken_link",
                description=f"[[{link}]] has no matching wiki page",
                suggestion=suggestion,
                auto_fixable=False,
            )
        )


def _check_malformed_embeds(rel_path: str, body: str, issues: list[LintIssue]) -> None:
    masked, _ = mask_markdown_regions(body)
    seen: set[str] = set()
    for match in _MALFORMED_EMBED_RE.finditer(masked):
        target = match.group(1).strip()
        if not target or target in seen:
            continue
        seen.add(target)
        issues.append(
            LintIssue(
                path=rel_path,
                issue_type="malformed_embed",
                description=f"!{target} is not valid Obsidian embed syntax",
                suggestion=f"Use ![[{target}]] or remove the media reference.",
                auto_fixable=True,
            )
        )


def _repair_malformed_embeds(body: str) -> str:
    masked, replacements = mask_markdown_regions(body)
    fixed = _MALFORMED_EMBED_RE.sub(lambda m: f"![[{m.group(1).strip()}]]", masked)
    return restore_markdown_regions(fixed, replacements)


def _check_malformed_latex(rel_path: str, body: str, issues: list[LintIssue]) -> None:
    # Mask code regions so LaTeX *source examples* in fences aren't flagged; the fix
    # path (sanitize_obsidian_math) masks the same regions, keeping check and fix agreed.
    masked, _ = mask_markdown_regions(body)
    if not (_DISPLAY_LATEX_RE.search(masked) or _BARE_LATEX_LINE_RE.search(masked)):
        return
    issues.append(
        LintIssue(
            path=rel_path,
            issue_type="malformed_latex",
            description=(
                "LaTeX math uses delimiters or line forms Obsidian may not render reliably."
            ),
            suggestion="Use $...$ for inline math and $$...$$ for display math.",
            auto_fixable=True,
        )
    )


def _check_stale_lock(config: Config, issues: list[LintIssue]) -> None:
    import os

    from .lock import _IS_POSIX, effective_app_dir, has_invalid_lock_file, lock_holder_pid

    lock_path = effective_app_dir(config.vault) / "pipeline.lock"
    if not lock_path.exists():
        return
    if has_invalid_lock_file(config.vault):
        issues.append(
            LintIssue(
                path=".synto/pipeline.lock",
                issue_type="stale_lock",
                description="pipeline.lock contains an invalid (non-integer) PID.",
                suggestion="Delete .synto/pipeline.lock to clear the stale lock.",
                auto_fixable=False,
            )
        )
        return
    # Lint often runs inside the held pipeline lock (synto run/maintain). Our own
    # lock is not stale, and probing it would open a second fd whose close drops
    # the lock under NFS POSIX-lock emulation — so short-circuit on our own PID.
    try:
        if int(lock_path.read_text(encoding="utf-8").strip()) == os.getpid():
            return
    except (ValueError, OSError):
        pass
    if lock_holder_pid(config.vault) is None:
        # On POSIX the flock lives on the open fd, not the filename. A leftover
        # pipeline.lock file after release is normal and does not block a future
        # acquire, so only invalid lock-file contents are worth flagging here.
        if _IS_POSIX:
            return
        try:
            pid: int | str = int(lock_path.read_text(encoding="utf-8").strip())
        except Exception:
            pid = "unknown"
        issues.append(
            LintIssue(
                path=".synto/pipeline.lock",
                issue_type="stale_lock",
                description=f"pipeline.lock has PID {pid} but that process is not running.",
                suggestion="Delete .synto/pipeline.lock to clear the stale lock.",
                auto_fixable=False,
            )
        )


@dataclass(frozen=True)
class FilenameDrift:
    old_rel: str
    new_rel: str
    old_stem: str
    new_stem: str
    title: str
    # Canonical target already taken (file on disk or tracked DB row) — the fixer must
    # skip; the remedy is a concept rename, not --fix.
    collides: bool


def find_filename_drift(config: Config, db: StateDB) -> list[FilenameDrift]:
    """Tracked articles whose on-disk stem re-sanitizes to a different canonical stem.

    Wikilink targets derive from ``sanitize_filename(title)``, so when its rules get
    stricter (Windows reserved names, trailing dots) a file the OLD sanitizer produced
    stops matching newly written links. Drift is only claimed when the existing stem
    itself re-sanitizes to the canonical form — a stem that doesn't (e.g. after a manual
    frontmatter retitle) is a deliberate user state, not a sanitizer-rule change.

    Collision detection lives here — single source of truth for the check and the fixer.
    """
    drifts: list[FilenameDrift] = []
    for art in db.list_articles():
        path = config.vault / art.path
        if not path.exists():
            continue
        old_stem = path.stem
        new_stem = sanitize_filename(art.title)
        if old_stem == new_stem or sanitize_filename(old_stem) != new_stem:
            continue
        parent, _, _ = art.path.rpartition("/")
        new_rel = f"{parent}/{new_stem}.md" if parent else f"{new_stem}.md"
        collides = (config.vault / new_rel).exists() or db.get_article(new_rel) is not None
        drifts.append(FilenameDrift(art.path, new_rel, old_stem, new_stem, art.title, collides))
    return drifts


def _check_filename_drift(config: Config, db: StateDB, issues: list[LintIssue]) -> None:
    for drift in find_filename_drift(config, db):
        if drift.collides:
            suggestion = (
                f"Target '{drift.new_stem}.md' is taken by another article; `maintain "
                f"--fix` will skip this — rename one concept with `synto concept rename`."
            )
        else:
            suggestion = "Run `synto maintain --fix` to rename it and repoint inbound links."
        issues.append(
            LintIssue(
                path=drift.old_rel,
                issue_type="filename_drift",
                description=(
                    f"Filename stem {drift.old_stem!r} no longer matches its canonical form "
                    f"{drift.new_stem!r}; new [[{drift.new_stem}]] links will not resolve "
                    f"to this file."
                ),
                suggestion=suggestion,
                auto_fixable=False,
            )
        )


def _check_missing_media(rel_path: str, body: str, vault: Path, issues: list[LintIssue]) -> None:
    seen: set[str] = set()
    for match in _EMBED_TARGET_RE.finditer(body):
        target = match.group(1).strip().split("#")[0].strip()
        if not target or target in seen:
            continue
        if Path(target).suffix.lower() not in _MEDIA_EXTENSIONS:
            continue  # note transclusion, not a media embed
        seen.add(target)
        if "/" in target:
            found = (vault / target).exists()
        else:
            found = (vault / "_resources" / target).exists() or (vault / target).exists()
        if not found:
            issues.append(
                LintIssue(
                    path=rel_path,
                    issue_type="missing_media",
                    description=f"![[{target}]] — embedded file does not exist",
                    suggestion="Add the file to _resources/ or remove the embed.",
                    auto_fixable=False,
                )
            )


def _repair_plain_citations(body: str) -> str:
    # mask_links=False: the Sources-section cleanup below must still see
    # [S1](#Sources) links; code fences/inline code stay protected (#93).
    masked, replacements = mask_markdown_regions(body, mask_links=False)
    if "## Sources" not in masked:
        return body
    before_sources, sources = masked.split("## Sources", 1)
    before_sources = _PLAIN_CITATION_RE.sub(
        lambda match: f"[{match.group(1)}](#Sources)", before_sources
    )
    sources = re.sub(r"\[(S\d+(?:\s*,\s*S\d+)*)\]\(#Sources\)", r"[\1]", sources)
    return restore_markdown_regions(before_sources + "## Sources" + sources, replacements)


def _mask_markdown_links(body: str) -> tuple[str, list[tuple[str, str]]]:
    replacements: list[tuple[str, str]] = []

    def repl(match: re.Match[str]) -> str:
        token = f"@@SYNTO_LINK_{len(replacements)}@@"
        replacements.append((token, match.group(0)))
        return token

    return re.sub(r"\[[^\]\n]+\]\([^)]*\)", repl, body), replacements


def _restore_markdown_links(body: str, replacements: list[tuple[str, str]]) -> str:
    for token, original in replacements:
        body = body.replace(token, original)
    return body


def _update_article_hash(db: StateDB, rel_path: str, new_hash: str) -> None:
    # model_copy preserves every field (kind, question_hash, synthesis provenance, …); a
    # partial WikiArticleRecord would blank them, since upsert overwrites those columns from
    # the record. Only content_hash changes.
    art = db.get_article(rel_path)
    if art is None or art.content_hash == new_hash:
        return
    db.upsert_article(art.model_copy(update={"content_hash": new_hash}))


def _write_fixed_note(page: Path, rel_path: str, meta: dict, body: str, db: StateDB) -> None:
    # Copy so the synthesis content_hash writes below don't leak back into the caller's dict.
    meta = dict(meta)
    is_synthesis = str(meta.get("kind", "")).casefold() == "synthesis" or "question_hash" in meta
    if is_synthesis:
        meta["content_hash"] = _body_hash(body)
    write_note(page, meta, body)
    # Hash the body as it round-trips on disk (frontmatter strips the trailing newline), so the
    # stored hash matches what compile/lint later recompute via parse_note. Matches the publish/
    # absorb pattern and is the exact asymmetry #83 was about. If the reparse of the just-written
    # file fails (transient FS fault), fall back to hashing the intended body so the DB row is still
    # updated best-effort — leaving the old hash would reintroduce the #83 stale/manual-edit false
    # positive, and letting it propagate would abort the caller's whole fix pass.
    try:
        _, ondisk = parse_note(page)
        new_hash = _body_hash(ondisk)
    except Exception as exc:
        log.warning(
            "post-write reparse of %s failed; hashing intended body instead — %s", page, exc
        )
        new_hash = _body_hash(body)
    else:
        if is_synthesis and meta.get("content_hash") != new_hash:
            meta["content_hash"] = new_hash
            write_note(page, meta, body)
    _update_article_hash(db, rel_path, new_hash)


def _title_from_file(path: Path) -> str:
    try:
        meta, _ = parse_note(path)
        return str(meta.get("title", path.stem))
    except Exception:
        return path.stem


def _vault_rel_path(path: Path, vault: Path) -> str:
    return path.relative_to(vault).as_posix()


def _normalized_graph_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.replace("-", " ").strip()).casefold()


def _source_page_hash_map(meta: dict) -> dict[str, str]:
    entries = meta.get("source_page_hashes", [])
    if not isinstance(entries, list):
        return {}
    result: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        hash_value = entry.get("hash")
        if isinstance(path, str) and isinstance(hash_value, str):
            result[path] = hash_value
    return result


def _add_graph_quality_issues(
    config: Config,
    title_index: dict[str, Path],
    issues: list[LintIssue],
) -> None:
    welcome = config.vault / "Welcome.md"
    if welcome.exists():
        issues.append(
            LintIssue(
                path="Welcome.md",
                issue_type="graph_noise",
                description="Obsidian starter Welcome.md appears in graph view",
                suggestion="Delete Welcome.md or filter it from graph view with -file:Welcome.",
                auto_fixable=False,
            )
        )

    if config.drafts_dir.exists() and config.pipeline.draft_media != "embed":
        for draft in sorted(config.drafts_dir.rglob("*.md")):
            try:
                _, body = parse_note(draft)
            except Exception:
                continue
            if _OBSIDIAN_EMBED_RE.search(body):
                issues.append(
                    LintIssue(
                        path=_vault_rel_path(draft, config.vault),
                        issue_type="graph_noise",
                        description=(
                            "Draft contains media embeds that create attachment nodes in graph view"
                        ),
                        suggestion=(
                            'Use draft_media = "reference" or move media embeds to source pages.'
                        ),
                        auto_fixable=False,
                    )
                )

    duplicate_examples: list[tuple[Path, Path]] = []
    if config.raw_dir.exists() and config.sources_dir.exists():
        raw_titles = {
            _normalized_graph_title(_title_from_file(path)): path
            for path in config.raw_dir.rglob("*.md")
        }
        for source in sorted(config.sources_dir.glob("*.md")):
            source_title = _title_from_file(source)
            key = _normalized_graph_title(source_title)
            raw_path = raw_titles.get(key)
            if raw_path is None:
                continue
            duplicate_examples.append((source, raw_path))

    if duplicate_examples:
        example_source, example_raw = duplicate_examples[0]
        suffix = "" if len(duplicate_examples) == 1 else f" and {len(duplicate_examples) - 1} more"
        issues.append(
            LintIssue(
                path=_vault_rel_path(example_source, config.vault),
                issue_type="graph_noise",
                description=(
                    "Source summary titles closely duplicate raw note titles, e.g. "
                    f"{_vault_rel_path(example_raw, config.vault)}{suffix}"
                ),
                suggestion="Filter raw/ or wiki/sources/ from Obsidian graph view.",
                auto_fixable=False,
            )
        )

    disconnected: list[Path] = []
    draft_data: list[tuple[Path, str, str]] = []
    if config.drafts_dir.exists():
        for draft in sorted(config.drafts_dir.rglob("*.md")):
            try:
                meta, body = parse_note(draft)
            except Exception:
                continue
            draft_data.append((draft, str(meta.get("title", draft.stem)).lower(), body))

    concept_targets = {title for _, title, _ in draft_data}
    if len(concept_targets) >= 2:
        for draft, own_title, body in draft_data:
            concept_links = {
                link.lower()
                for link in extract_wikilinks(body)
                if link.lower() in concept_targets and link.lower() != own_title
            }
            if not concept_links:
                disconnected.append(draft)

    if disconnected:
        example = disconnected[0]
        suffix = "" if len(disconnected) == 1 else f" and {len(disconnected) - 1} more"
        issues.append(
            LintIssue(
                path=_vault_rel_path(example, config.vault),
                issue_type="graph_connectivity",
                description=(
                    "Generated drafts have no links to other concept drafts, e.g. "
                    f"{example.name}{suffix}"
                ),
                suggestion="Add related concept links or recompile after related concepts exist.",
                auto_fixable=False,
            )
        )


def _wiki_rel_key(path: Path, wiki_dir: Path) -> str:
    # Forward-slash, suffix-less, vault-relative key. as_posix() (not str()) so
    # [[sources/X]] links — always forward-slash — resolve on Windows too. See #26.
    return path.relative_to(wiki_dir).with_suffix("").as_posix().lower()


def _build_title_index(config: Config, db: StateDB | None = None) -> dict[str, Path]:
    """Map lowercase title/stem → path for every wiki page, including drafts.

    Also indexes frontmatter aliases and (when db provided) DB alias map.
    Ambiguous aliases (same alias → multiple pages) are excluded so they stay broken.
    """
    index: dict[str, Path] = {}
    alias_targets: dict[str, list[Path]] = {}  # alias_lower → candidate paths

    for md in config.wiki_dir.rglob("*.md"):
        index[md.stem.lower()] = md
        try:
            index[_wiki_rel_key(md, config.wiki_dir)] = md
        except ValueError:
            pass
        try:
            meta, _ = parse_note(md)
            title = meta.get("title", "")
            if title:
                index[title.lower()] = md
                base_title = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
                if base_title and base_title != title:
                    alias_targets.setdefault(base_title.lower(), []).append(md)
            aliases = meta.get("aliases", [])
            if isinstance(aliases, str):
                aliases = [aliases]
            elif not isinstance(aliases, list):
                aliases = []
            for alias in aliases:
                if isinstance(alias, str) and alias.strip():
                    alias_targets.setdefault(alias.strip().lower(), []).append(md)
        except Exception:
            pass

    # Add DB alias map: alias → canonical title → path (via title index)
    if db is not None:
        for alias_lower, canonical in db.list_alias_map().items():
            target = index.get(canonical.lower())
            if target is not None:
                alias_targets.setdefault(alias_lower, []).append(target)

    # Commit unambiguous aliases to index (don't overwrite canonical title/stem entries)
    for alias_lower, targets in alias_targets.items():
        unique = list({id(t): t for t in targets}.values())
        if len(unique) == 1 and alias_lower not in index:
            index[alias_lower] = unique[0]

    return index


def _build_inbound_index(config: Config) -> dict[str, set[str]]:
    """Map target title (lower) → set of page stems that link to it."""
    inbound: dict[str, set[str]] = {}
    for md in config.wiki_dir.rglob("*.md"):
        if ".drafts" in md.parts:
            continue
        try:
            _, body = parse_note(md)
        except Exception:
            continue
        for link in extract_wikilinks(body):
            key = link.lower()
            inbound.setdefault(key, set()).add(md.stem)
    return inbound


def _concept_pages(config: Config) -> list[Path]:
    """Root-level wiki pages that are concept articles (not system files)."""
    if not config.wiki_dir.exists():
        return []
    pages = []
    for md in sorted(config.wiki_dir.glob("*.md")):
        if md.stem.lower() in _SYSTEM_STEMS:
            continue
        pages.append(md)
    return pages


def _all_wiki_pages(config: Config) -> list[Path]:
    """All wiki pages including drafts, sources/ and queries/ (excluded: system stems)."""
    if not config.wiki_dir.exists():
        return []
    pages = []
    for md in sorted(config.wiki_dir.rglob("*.md")):
        if md.parent == config.wiki_dir and md.stem.lower() in _SYSTEM_STEMS:
            continue
        pages.append(md)
    return pages


def _check_manual_relabel(config: Config, db: StateDB, issues: list[LintIssue], fix: bool) -> None:
    """Decision 10: adopt a wiki file the user renamed on disk as the new preferred label.

    Signal: a tracked, published concept article whose file is gone from disk, paired with
    an untracked wiki file whose body hash equals the tracked row's content_hash (a pure
    rename, not an edit). Under ``--fix`` the new filename stem becomes the entity's
    preferred label (old → alias) and inbound links are rewritten. A new stem that already
    names a *different* active entity is a collision: flag it loudly and never adopt.
    """
    from .maintain import _rewrite_inbound_links

    all_articles = db.list_articles()
    tracked_paths = {a.path for a in all_articles}

    # Untracked ROOT-level wiki files indexed by body hash, for rename matching.
    # Scoped to concept pages (wiki/*.md) so a merge-retired .drafts/ copy — which shares
    # the missing article's body hash — can never be mistaken for a manual rename.
    untracked_by_hash: dict[str, Path] = {}
    for page in _concept_pages(config):
        rel = _vault_rel_path(page, config.vault)
        if rel in tracked_paths:
            continue
        try:
            _, body = parse_note(page)
        except Exception:
            continue
        untracked_by_hash.setdefault(_body_hash(body), page)

    active_preferred = {lbl.casefold() for lbl in db.list_active_preferred_labels()}

    for art in all_articles:
        if not art.is_published or art.kind != "concept" or not art.content_hash:
            continue
        if (config.vault / art.path).exists():
            continue  # file still where the DB expects it
        renamed = untracked_by_hash.get(art.content_hash)
        if renamed is None:
            continue
        new_rel = _vault_rel_path(renamed, config.vault)
        new_label = renamed.stem
        old_label = db.preferred_label_for_entity(art.entity_id) if art.entity_id else art.title
        if not old_label or new_label.casefold() == old_label.casefold():
            continue

        # Collision: the new stem already names a different active entity.
        if new_label.casefold() in active_preferred:
            issues.append(
                LintIssue(
                    path=new_rel,
                    issue_type="homonym_filename_collision",
                    description=(
                        f"File renamed to {new_label!r} but that label already belongs to "
                        f"another active concept — not adopting the rename of {old_label!r}."
                    ),
                    suggestion="Pick a distinct filename or merge the two concepts.",
                    auto_fixable=False,
                )
            )
            continue

        if not fix:
            issues.append(
                LintIssue(
                    path=new_rel,
                    issue_type="manual_relabel_adopted",
                    description=(
                        f"File for concept {old_label!r} was renamed to {new_label!r} on disk."
                    ),
                    suggestion="Run `synto lint --fix` to adopt it as the preferred label.",
                    auto_fixable=True,
                )
            )
            continue

        # Adopt under --fix: demote old label to alias, repoint the article, rewrite links.
        old_stem = sanitize_filename(old_label)
        new_stem = sanitize_filename(new_label)
        db.rename_concept(old_label, new_label)
        # Bless the demoted old label (source='rename') so a re-extracted old name links to the
        # winner instead of minting a new entity — parity with `concept rename` (maintain.py),
        # otherwise the adopted rename re-fragments on the next ingest.
        db.upsert_aliases(new_label, [old_label], source="rename")
        db.update_article_identity(art.path, new_rel, new_label)
        try:
            meta, body = parse_note(renamed)
            meta["title"] = new_label
            existing = list(meta.get("aliases") or [])
            if not any(a.casefold() == old_label.casefold() for a in existing):
                meta["aliases"] = [*existing, old_label]
            write_note(renamed, meta, body)
        except Exception:
            pass
        links = _rewrite_inbound_links(config, db, old_stem, new_stem, new_label, dry_run=False)
        # Keep the in-run snapshot consistent for any later relabel this pass.
        active_preferred.discard(old_label.casefold())
        active_preferred.add(new_label.casefold())
        issues.append(
            LintIssue(
                path=new_rel,
                issue_type="manual_relabel_adopted",
                description=(
                    f"Adopted manual rename {old_label!r} → {new_label!r}; "
                    f"{links} inbound link(s) rewritten."
                ),
                suggestion="None — the preferred label now matches the filename.",
                auto_fixable=False,
            )
        )


# ── Public API ────────────────────────────────────────────────────────────────


def partition_acked(
    issues: list[LintIssue], ack_entries: list[str]
) -> tuple[list[LintIssue], list[LintIssue]]:
    """Split issues into (active, acked) per [maintain].ack entries (#94).

    An entry is "<issue_type>" (matches every path) or "<issue_type>:<path>" (matches only
    that vault-relative path), split on the first colon; check and path are stripped so
    natural TOML spacing ("stale_lock: wiki/X.md") doesn't silently never match. Paths are
    compared with posix separators on both sides — ack entries may be hand-written with
    backslashes on Windows, while LintIssue.path is always vault-relative posix (see
    _vault_rel_path). Only ADVISORY issue types can be acked — an acked structural issue
    would keep lowering the health score while hidden from the list, so it stays active
    (config warns about such entries at load). Display-only: callers must keep using the
    unpartitioned `issues` list for health score / auto-fix passes.
    """
    bare_checks: set[str] = set()
    scoped_checks: set[tuple[str, str]] = set()
    for entry in ack_entries:
        check, sep, path = entry.partition(":")
        check = check.strip()
        if check not in _ADVISORY_ISSUE_TYPES:
            continue
        if sep:
            scoped_checks.add((check, path.strip().replace("\\", "/")))
        else:
            bare_checks.add(check)

    active: list[LintIssue] = []
    acked: list[LintIssue] = []
    for issue in issues:
        norm_path = issue.path.replace("\\", "/")
        if issue.issue_type in bare_checks or (issue.issue_type, norm_path) in scoped_checks:
            acked.append(issue)
        else:
            active.append(issue)
    return active, acked


def run_lint(config: Config, db: StateDB, fix: bool = False) -> LintResult:
    issues: list[LintIssue] = []

    # ── Config sanity checks ──────────────────────────────────────────────────
    # The default article_max_tokens was raised from 4096 to 16384 to avoid
    # silent truncation on long-form articles. Existing synto.toml files written
    # by older synto init runs still pin 4096 and won't pick up the new default.
    if config.pipeline.article_max_tokens == 4096:
        issues.append(
            LintIssue(
                path="synto.toml",
                issue_type="config_outdated",
                description=(
                    f"pipeline.article_max_tokens is {config.pipeline.article_max_tokens} "
                    "(matches the legacy default 4096). Long articles may truncate "
                    "silently on local LLM providers."
                ),
                suggestion=(
                    "Raise to 16384 in synto.toml [pipeline] section, or delete the line "
                    "to pick up the new default."
                ),
                auto_fixable=False,
            )
        )

    _check_stale_lock(config, issues)
    _check_filename_drift(config, db, issues)

    title_index = _build_title_index(config, db=db)
    inbound_index = _build_inbound_index(config)

    # DB records keyed by vault-relative path
    db_articles = {a.path: a for a in db.list_articles(drafts_only=False) if a.is_published}

    pages = _concept_pages(config)
    all_pages = _all_wiki_pages(config)

    for page in pages:
        rel_path = _vault_rel_path(page, config.vault)

        try:
            meta, body = parse_note(page)
        except Exception as exc:
            issues.append(
                LintIssue(
                    path=rel_path,
                    issue_type="missing_frontmatter",
                    description=f"Failed to parse frontmatter: {exc}",
                    suggestion="Fix or recreate the file.",
                    auto_fixable=False,
                )
            )
            continue

        title = meta.get("title", page.stem)

        # Disambiguation stubs are system pages (auto-generated by `concept split`). Existing
        # vaults hold ones written before the creation-site fix — bare frontmatter + empty DB
        # content_hash — so don't flag them missing_frontmatter/stale or let --fix inject fields.
        is_disambiguation = str(meta.get("kind", "")).casefold() == "disambiguation"

        # ── Missing frontmatter ───────────────────────────────────────────────
        missing = _REQUIRED_FIELDS - set(meta.keys())
        if missing and not is_disambiguation:
            issues.append(
                LintIssue(
                    path=rel_path,
                    issue_type="missing_frontmatter",
                    description=f"Missing fields: {', '.join(sorted(missing))}",
                    suggestion=f"Add: {', '.join(f'{f}: ...' for f in sorted(missing))}",
                    auto_fixable=True,
                )
            )
            if fix:
                for field in sorted(missing):
                    if field == "title":
                        meta["title"] = page.stem
                    elif field == "status":
                        meta["status"] = "published"
                    elif field == "tags":
                        meta["tags"] = []
                write_note(page, meta, body)

        # ── Invalid tags ──────────────────────────────────────────────────────
        _check_tags(rel_path, meta, issues, fix, page, body)

        # ── Low confidence ────────────────────────────────────────────────────
        confidence = meta.get("confidence")
        if confidence is not None:
            try:
                conf_val = float(confidence)
                if conf_val < _LOW_CONFIDENCE_THRESHOLD:
                    issues.append(
                        LintIssue(
                            path=rel_path,
                            issue_type="low_confidence",
                            description=(
                                f"Confidence {conf_val:.2f} below "
                                f"threshold {_LOW_CONFIDENCE_THRESHOLD}"
                            ),
                            suggestion="Add more source notes covering this concept.",
                            auto_fixable=False,
                        )
                    )
            except (ValueError, TypeError):
                pass

        # ── Manually edited (stale hash) ──────────────────────────────────────
        db_rec = db_articles.get(rel_path)
        if db_rec and not is_disambiguation:
            # A blank content_hash is a "not yet hashed" placeholder (a real empty body hashes
            # to a non-empty digest), never a manual edit — same semantics as compile's guard (#83).
            if db_rec.content_hash and _body_hash(body) != db_rec.content_hash:
                issues.append(
                    LintIssue(
                        path=rel_path,
                        issue_type="stale",
                        description="File modified manually since last compile.",
                        suggestion=(
                            "Run `synto compile --force` to recompile, "
                            "or keep edits (page is protected)."
                        ),
                        auto_fixable=False,
                    )
                )

        # ── Broken wikilinks ──────────────────────────────────────────────────
        _check_broken_wikilinks(rel_path, body, title_index, issues)

        # ── Malformed markdown links ─────────────────────────────────────────
        _check_malformed_links(rel_path, body, issues)
        _check_malformed_latex(rel_path, body, issues)
        _check_malformed_embeds(rel_path, body, issues)
        _check_missing_media(rel_path, body, config.vault, issues)
        if fix:
            fixed_body = sanitize_obsidian_math(body)
            fixed_body = _repair_plain_citations(_repair_malformed_embeds(fixed_body))
            if fixed_body != body:
                body = fixed_body
                _write_fixed_note(page, rel_path, meta, body, db)

        # ── Inline hashtags ───────────────────────────────────────────────────
        masked_body, markdown_links = _mask_markdown_links(body)
        inline_tags = _INLINE_TAG_RE.findall(masked_body)
        body = _restore_markdown_links(masked_body, markdown_links)
        if inline_tags:
            issues.append(
                LintIssue(
                    path=rel_path,
                    issue_type="inline_tag",
                    description=f"Inline #tags in body: {', '.join(f'#{t}' for t in inline_tags)}",
                    suggestion="Replace inline #tags with [[wikilinks]] or frontmatter tags.",
                    auto_fixable=False,
                )
            )

        # ── Orphan ───────────────────────────────────────────────────────────
        # Linked-by: pages that contain [[title]] or [[stem]] in their body
        linked_by = inbound_index.get(title.lower(), set()) | inbound_index.get(
            page.stem.lower(), set()
        )
        # Exclude self-links and the index page
        linked_by -= {page.stem, "index", "log"}
        if not linked_by:
            issues.append(
                LintIssue(
                    path=rel_path,
                    issue_type="orphan",
                    description="No other wiki page links to this page.",
                    suggestion="Reference this concept from related pages or run `synto compile`.",
                    auto_fixable=False,
                )
            )

    # ── Tag + frontmatter checks for sources/ and queries/ ────────────────────
    concept_page_paths = {p for p in pages}
    for page in all_pages:
        if page in concept_page_paths:
            continue  # already checked above
        rel_path = _vault_rel_path(page, config.vault)
        try:
            meta, body = parse_note(page)
        except Exception as exc:
            issues.append(
                LintIssue(
                    path=rel_path,
                    issue_type="missing_frontmatter",
                    description=f"Failed to parse frontmatter: {exc}",
                    suggestion="Fix YAML syntax in frontmatter.",
                    auto_fixable=False,
                )
            )
            continue

        # Invalid tags
        _check_tags(rel_path, meta, issues, fix, page, body)

        # Malformed links in sources/queries are useful to surface too.
        _check_malformed_links(rel_path, body, issues)
        _check_malformed_latex(rel_path, body, issues)
        _check_malformed_embeds(rel_path, body, issues)
        _check_missing_media(rel_path, body, config.vault, issues)
        if fix:
            fixed_body = sanitize_obsidian_math(body)
            fixed_body = _repair_plain_citations(_repair_malformed_embeds(fixed_body))
            if fixed_body != body:
                body = fixed_body
                _write_fixed_note(page, rel_path, meta, body, db)

        # Draft/source/query links should be valid too; otherwise review sees a
        # healthy vault while pending drafts contain invented pages.
        _check_broken_wikilinks(rel_path, body, title_index, issues)

        # Missing required frontmatter
        missing = _REQUIRED_FIELDS - set(meta.keys())
        if missing:
            issues.append(
                LintIssue(
                    path=rel_path,
                    issue_type="missing_frontmatter",
                    description=f"Missing fields: {', '.join(sorted(missing))}",
                    suggestion=f"Add: {', '.join(f'{f}: ...' for f in sorted(missing))}",
                    auto_fixable=True,
                )
            )
            if fix:
                for field in sorted(missing):
                    if field == "title":
                        meta["title"] = page.stem
                    elif field == "status":
                        meta["status"] = "published"
                    elif field == "tags":
                        meta["tags"] = []
                write_note(page, meta, body)

    if config.pipeline.graph_quality_checks:
        _add_graph_quality_issues(config, title_index, issues)

    concept_titles = {
        article.title.casefold()
        for article in db_articles.values()
        if article.kind == "concept" and article.is_published
    }
    synthesis_db_paths = {
        path
        for path, article in db_articles.items()
        if article.kind == "synthesis" and article.is_published
    }

    # Scan raw/ for missing media (outside wiki denominator → advisory)
    if config.raw_dir.exists():
        for raw_note in sorted(config.raw_dir.glob("*.md")):
            rel_path = _vault_rel_path(raw_note, config.vault)
            try:
                _, body = parse_note(raw_note)
            except Exception:
                continue
            _check_missing_media(rel_path, body, config.vault, issues)

    for page in sorted(config.synthesis_dir.glob("*.md")) if config.synthesis_dir.exists() else []:
        rel_path = _vault_rel_path(page, config.vault)
        db_rec = db_articles.get(rel_path)
        if db_rec is None:
            issues.append(
                LintIssue(
                    path=rel_path,
                    issue_type="orphan",
                    description="Synthesis file exists without a matching state row.",
                    suggestion="Re-save the synthesis article or remove the orphan file.",
                    auto_fixable=False,
                )
            )
            continue

        try:
            meta, _ = parse_note(page)
        except Exception as exc:
            issues.append(
                LintIssue(
                    path=rel_path,
                    issue_type="missing_frontmatter",
                    description=f"Failed to parse frontmatter: {exc}",
                    suggestion="Fix YAML syntax in frontmatter.",
                    auto_fixable=False,
                )
            )
            continue

        if db_rec.title.casefold() in concept_titles:
            issues.append(
                LintIssue(
                    path=rel_path,
                    issue_type="graph_noise",
                    description="Synthesis title shadows an existing concept title.",
                    suggestion="Rename the synthesis page or prefer the concept page.",
                    auto_fixable=False,
                )
            )

        source_pages = meta.get("source_pages", [])
        if isinstance(source_pages, str):
            source_pages = [source_pages]
        elif not isinstance(source_pages, list):
            source_pages = []

        hash_map = _source_page_hash_map(meta)
        for source_page in source_pages:
            if not isinstance(source_page, str):
                continue
            resolved = title_index.get(source_page.lower())
            if resolved is None:
                issues.append(
                    LintIssue(
                        path=rel_path,
                        issue_type="broken_link",
                        description=f"Source page '{source_page}' no longer resolves.",
                        suggestion="Remove the stale source or restore the page.",
                        auto_fixable=False,
                    )
                )
                continue

            resolved_rel = _vault_rel_path(resolved, config.vault)
            if resolved_rel in synthesis_db_paths or "synthesis" in resolved.parts:
                issues.append(
                    LintIssue(
                        path=rel_path,
                        issue_type="synthesis_chain",
                        description=(
                            "Synthesis page references another synthesis page in source_pages."
                        ),
                        suggestion="Recreate the synthesis using concept or source pages only.",
                        auto_fixable=False,
                    )
                )

            try:
                _, source_body = parse_note(resolved)
            except Exception:
                continue
            recorded_hash = hash_map.get(resolved_rel)
            if recorded_hash and recorded_hash != _body_hash(source_body):
                issues.append(
                    LintIssue(
                        path=rel_path,
                        issue_type="stale",
                        description=f"Recorded source hash drifted for {resolved_rel}.",
                        suggestion="Re-run the synthesis to refresh source provenance.",
                        auto_fixable=False,
                    )
                )

    # ── Identity checks (advisory) ────────────────────────────────────────────
    for a_id, b_id, mk in db.find_match_key_collisions():
        a_label = db.preferred_label_for_entity(a_id)
        b_label = db.preferred_label_for_entity(b_id)
        if a_label and b_label:
            issues.append(
                LintIssue(
                    path="",
                    issue_type="label_collision",
                    description=(
                        f"Concepts {a_label!r} and {b_label!r} share match_key {mk!r} "
                        "— likely plural/singular duplicates."
                    ),
                    suggestion=f"Run `synto concept merge {a_label!r} {b_label!r}` to consolidate.",
                    auto_fixable=False,
                )
            )

    for eid, label in db.list_active_entities_without_articles():
        issues.append(
            LintIssue(
                path="",
                issue_type="orphan_entity",
                description=f"Active entity {label!r} has no published article.",
                suggestion="Run `synto compile` to generate an article, or merge/split to resolve.",
                auto_fixable=False,
            )
        )

    # Ambiguous bare label (homonym) with no disambiguation stub on disk.
    existing_stems = {
        Path(a.path).stem.casefold() for a in db.list_articles() if a.kind == "disambiguation"
    }
    for label, n in db.find_ambiguous_active_labels():
        if sanitize_filename(label).casefold() in existing_stems:
            continue
        issues.append(
            LintIssue(
                path="",
                issue_type="ambiguous_label_needs_disambiguation",
                description=(
                    f"Label {label!r} resolves to {n} entities but has no disambiguation page."
                ),
                suggestion=f"Run `synto concept split` or add a disambiguation stub for {label!r}.",
                auto_fixable=False,
            )
        )

    # Untrusted aliases left over from the v18 legacy backfill.
    legacy_aliases = db.count_legacy_backfill_aliases()
    if legacy_aliases:
        issues.append(
            LintIssue(
                path="",
                issue_type="stale_legacy_backfill_alias",
                description=(
                    f"{legacy_aliases} alias(es) still sourced from the v18 legacy backfill."
                ),
                suggestion="Review with `synto concept inspect`; merge or bless to clear.",
                auto_fixable=False,
            )
        )

    # Two active entities whose preferred labels sanitize to the same filename.
    by_stem: dict[str, list[str]] = {}
    for label in db.list_active_preferred_labels():
        by_stem.setdefault(sanitize_filename(label).casefold(), []).append(label)
    for stem, labels in by_stem.items():
        if len(labels) > 1:
            issues.append(
                LintIssue(
                    path="",
                    issue_type="homonym_filename_collision",
                    description=(
                        f"Entities {labels!r} all map to filename {stem!r}.md — "
                        "their compile output would collide on disk."
                    ),
                    suggestion="Rename one entity to a distinct filename-safe label.",
                    auto_fixable=False,
                )
            )

    _check_manual_relabel(config, db, issues, fix)

    # ── Health score ──────────────────────────────────────────────────────────
    # Score based on structural wiki health. Graph-quality findings are advisory:
    # they should be visible in lint output without turning a structurally healthy
    # vault into a failing one or driving the score negative.
    total = max(len(all_pages), 1)
    pages_with_issues = len(
        {iss.path for iss in issues if iss.issue_type not in _ADVISORY_ISSUE_TYPES}
    )
    score = round(100.0 * (1 - pages_with_issues / total), 1)
    score = max(0.0, min(100.0, score))
    advisory_issue_count = sum(1 for iss in issues if iss.issue_type in _ADVISORY_ISSUE_TYPES)

    # Summary
    if not issues:
        summary = f"Wiki healthy. {len(all_pages)} pages checked, no issues."
    else:
        counts: dict[str, int] = {}
        for iss in issues:
            counts[iss.issue_type] = counts.get(iss.issue_type, 0) + 1
        parts = [f"{v} {k}" for k, v in sorted(counts.items())]
        summary = f"{len(issues)} issue(s): {', '.join(parts)}. {len(all_pages)} pages checked."

    return LintResult(
        issues=issues,
        health_score=round(score, 1),
        summary=summary,
        advisory_issue_count=advisory_issue_count,
    )
