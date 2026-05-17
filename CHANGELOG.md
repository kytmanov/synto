# Changelog

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

