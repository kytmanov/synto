# Changelog

## [Unreleased]

### Added

- `synto add SOURCE` — import PDF, Markdown, and text files as tracked source documents.
  The original is archived in `.synto/sources/<id>/`. For PDF files, segments are
  extracted immediately into heading-aware chunks and linked to the concepts that cite
  them. Use `--type` to specify the document type; `--force` to re-import; `--extend-pack`
  to register the source under a named pack in `synto.toml`.
- Source-type prompt system: 6 per-type system prompt templates (`notes`, `textbook`,
  `paper`, `api_docs`, `web_article`, `corp_docs`) are loaded at compile time based on
  the declared source type, steering the heavy model toward type-appropriate structure and
  terminology.
- Compile lineage: `compile_runs` table records every compile job (models, token counts,
  timestamps). Published articles carry a `lineage:` frontmatter field listing their
  contributing sources and run ID. `synto trace article <name>` prints the full compile
  history for any article.
- LLM response cache: `llm_cache` table stores SHA-256-keyed responses. `synto maintain
  --clear-cache` flushes all entries; `--older-than N` prunes entries older than N days.
- Term extraction: `extract_terms()` and `VaultReader.list_terms()` added;
  `concept_occurrences` table (schema v13) links concepts to the source segments they
  appear in.

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

