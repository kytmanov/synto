# Changelog

## [Unreleased]

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
