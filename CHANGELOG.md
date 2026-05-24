# Changelog

## [Unreleased]

### Added

- Query vocabulary bridge: `synto query` now augments the routing prompt with
  a hint naming the concepts whose aliases match whole words in the user's
  question (e.g., "ML" → "Machine Learning"). This helps the fast model pick
  the right wiki article when the user types an acronym or surface form
  instead of the canonical title. Ambiguous aliases (claimed by ≥2 concepts)
  are filtered. All-caps acronyms of length ≥2 bypass the length floor.
  Pure-CJK aliases are a silent no-op (v1 limitation: Python's `\b` requires
  word boundaries). No new dependencies, no public-API changes.
- Article frontmatter now includes three machine-readable quality signals emitted
  at compile time: `source_count` (int), `single_source` (bool), and
  `source_quality` ("high" | "medium" | "low"). These let file readers, Obsidian
  plugins, MCP tools, and AI agents assess corroboration needs without DB access.
  `single_source: true` is derived from unique source-document identity where
  available, falling back to path uniqueness. Synthesis articles carry
  `source_count` and `single_source` on the same basis. `read_article` in the MCP
  server inherits all three fields; `list_articles` now includes them in its
  projection.
- Per-source-type ingest overrides: `[pipeline.source_overrides.<type>]` sections in
  `synto.toml` let you raise the `max_concepts_per_source` ceiling for long-form source
  types (e.g. `textbook`, `paper`) without changing the global default. Quality-based
  reduction still applies within the per-type ceiling. Unknown source type keys emit a
  warning at load time. Re-run `synto ingest --force` to apply changes to already-ingested
  sources.

### Fixed

- `synto compile` now reads document identity from `source_documents` (the
  canonical store since schema v9) instead of from a duplicate set of columns
  on `raw_notes` that were never populated. The `single_source` frontmatter
  field is unchanged for every current ingestion scenario (each source is its
  own raw file, so path-uniqueness gives the correct answer either way); the
  change is structural and makes the doc-identity branch alive and tested.

### Changed

- Schema v14: drop the five v8 metadata columns from `raw_notes`
  (`source_type`, `origin_uri`, `imported_at`, `normalized_hash`,
  `extractor_version`) that were superseded by `source_documents` in v9.
  These columns held NULL on every row in every released version except
  `source_type` (which had `DEFAULT 'notes'` but was never read). Vaults
  migrate automatically on next open. External tooling querying those
  columns directly must JOIN through `source_documents.id = stem(raw_notes.path)`.
- `synto` now requires SQLite ≥ 3.35.0 (for the v14 migration's
  `DROP COLUMN`). Modern Python stdlib distributions on macOS and Linux
  already meet this. The error message at startup names the required
  version if the check fails.
- A downgrade guard now blocks opening a vault whose on-disk schema
  version is newer than the installed `synto` binary, with a clear
  upgrade-required error.

## [0.2.2] - 2026-05-22

### Fixed

- `synto undo` now exits with a clear error when the working tree has uncommitted
  changes to tracked files, instead of silently returning nothing. Untracked files
  (e.g. `.gitignore`) are correctly ignored.
- MCP server (`synto serve`) no longer emits INFO-level log lines on stdout, which
  corrupted the JSON-RPC protocol stream.

### Internal

- Smoke test suite overhauled: agent-usable output, `soft_check` / `check`
  distinction, JSON report files, per-section timing, and a new `mcp_smoke.py`
  integration suite.

## [0.2.1] - 2026-05-19

### Fixed

- `synto review` now renders all `[[wikilinks]]` in draft bodies correctly. Previously, Rich's
  markup parser interpreted `[[...]]` as nested markup tags, so only the first link was visible
  and the rest appeared as `[[]]`.
- Review diff output (`d` / `v` actions) no longer concatenates header lines. The `---`/`+++`
  and `@@` lines are now properly newline-terminated.
- Draft titles containing `[` or `]` are now escaped before being passed to Rich in the review
  table, metadata line, and blocked-concept message.

## [0.2.0] - 2026-05-19

### Added

- `synto add SOURCE` — import PDF, Markdown, and text files as tracked source documents.
  The original is archived in `.synto/sources/<id>/`. For PDF files, segments are
  extracted immediately into heading-aware chunks and assembled into canonical `raw/*.md`
  notes for the ingest pipeline. Use `--type` to specify the document type; `--force` to
  re-import; `--extend-pack` remains reserved and is currently a safe no-op.
- Source-type prompt system: built-in prompts for `notes`, `textbook`, `paper`, `spec`,
  `api_docs`, `web_article`, `corp_docs`, `transcript`, plus `unknown_text` fallback are
  loaded during ingest analysis based on the declared source type, steering the fast model
  toward type-appropriate structure and terminology.
- Compile lineage: `compile_runs` table records every compile job (models, token counts,
  timestamps). Published articles carry a `lineage:` frontmatter field listing their
  contributing sources and run ID. `synto trace article <name>` prints the full compile
  history for any article.
- LLM response cache: `llm_cache` table stores SHA-256-keyed responses. `synto maintain
  --clear-cache` flushes all entries; `--older-than N` prunes entries older than N days.
- Term extraction: `extract_terms()` and `VaultReader.list_terms()` added;
  `concept_occurrences` table (schema v13) links concepts to the source segments they
  appear in.

### Fixed

- PDF import now preserves ToC preamble text, surfaces bibliographic metadata into raw-note
  frontmatter, closes extractor resources correctly, and detects duplicate imports by content
  hash instead of filename alone.
- File-based imports are now atomic, and `synto add --force` correctly replaces prior raw,
  asset, and import state instead of leaving stale artifacts behind.
- `synto status` now counts imported on-disk raw notes before ingest, so `synto add` output
  shows up immediately as `Raw: new`.
- Structured-output recovery now repairs malformed JSON escapes more aggressively, fixing
  live compile failures caused by invalid backslash and malformed `\u...` sequences.
- Compile cleanup now strips stray `[[wikilinks]...` placeholder artifacts from generated
  article bodies before draft write.
- Ingest invalidation now respects source-type prompt changes, and imported `### Media`
  blocks are stripped before article synthesis.
- Smoke coverage now matches current runtime behavior, including LM Studio model-id
  resolution and the intentional `--extend-pack` no-op.

## [0.1.1] - 2026-05-17

### Fixed

- Ingest checkpoints now include the language setting and prompt version in their cache key.
  Changing `[pipeline].language` (or upgrading to a new prompt format) correctly invalidates
  cached chunks and triggers a full re-analysis on the next run. Previously, stale chunks from
  a different language or prompt version could be silently reused.
- `synto maintain` and compare reports now show a separate advisory issue count alongside the
  structural health score, so graph noise and missing media don't deflate the headline number.
- `synto eval`: missing `INDEX.json` no longer scores as 0 — it is excluded from the harmonic
  mean and reported as `n/a`. An invalid (unparseable) JSON file still scores 0.
- Orchestrator now records the final lint issue count after auto-approve, reflecting the
  post-publish vault state rather than the pre-approve snapshot.
- Placeholder embeds produced by the Obsidian web clipper (`![[unknown_filename.*]]`) are
  stripped from source notes before they reach the writer model, preventing them from leaking
  into compiled articles. The regex uses word boundaries so files that happen to contain
  `unknown_filename` as a substring in a longer name are not affected.
- Type annotation on `_index_json_validity` corrected (`float | None`, was `float`).
