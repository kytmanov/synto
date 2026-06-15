# Changelog

## [Unreleased]

### Fixed

- `synto concept merge` no longer aborts when both concepts have a stub row: the
  name-keyed `stubs` move now collapses on collision instead of hitting a primary-key
  constraint.
- Name-keyed writes (e.g. compile-state marking) for a label whose preferred owner was
  retired by a merge now resolve to the active winner instead of attaching to the dead
  entity (and silently vanishing).
- Upgrades from schema &lt; 6 no longer drop the "already compiled" marks for published
  articles, which had forced a full recompile after upgrade.
- `synto lint --fix` adopting a manual file rename now blesses the old label (parity with
  `synto concept rename`), so it no longer re-fragments into a new concept on the next
  ingest.
- `synto concept unmerge` keeps an absorbed alias that another still-merged concept also
  contributed, and uses the winner's current label when it was renamed after the merge.
- Plural/singular concept folding no longer collapses short singular nouns ending in `s`
  (e.g. `Lens` no longer folds onto an unrelated `Len`).
- The published `INDEX.json` and pack export now emit each article's bound `entity_id`
  rather than re-resolving it by name, which dropped identity for homonym/renamed titles.

## [0.6.0] - 2026-06-14

### Added

- **Concept identity and curation.** Concepts now carry a stable `entity_id` separate from
  their display name, so homonyms and renames no longer collide. New commands curate the
  concept graph: `synto concept merge` / `split` / `unmerge` move identity, labels, and
  sources as a unit, and `synto concept rename` relabels in place (old name kept as an
  alias, inbound links rewritten). `synto doctor`, `synto concept inspect`, and
  `synto concept keep` surface and resolve ambiguity. Every operation is logged and
  supports `--dry-run`. Thanks to @PEKEW (#46) and @wlewis55 (#54) for reporting the
  cross-domain mislinking and duplicate-concept problems this solves.
- **Order-independent ingest.** Whether a surface becomes its own concept no longer depends
  on ingest order. When a later note promotes a previously weak alias into a concept, Synto
  raises an explicit merge candidate (in `synto doctor` / `synto concept inspect`) instead
  of silently splitting or absorbing it. Human-blessed aliases are always respected.
  Existing vaults upgrade automatically to schema v26.
- **Remote MCP over Streamable HTTP.** `synto serve --transport streamable-http` exposes
  the wiki at `/mcp` for remote clients. There is no built-in auth — run it on a trusted
  network or behind a reverse proxy/firewall. DNS-rebinding protection is on by default;
  add hostnames a proxy forwards with `--allowed-host`. Thanks to @wwwangjunjie (#67).

### Fixed

- `synto undo` no longer silently diverges the database on identity operations. Because
  `state.db` is gitignored, reverting a `merge`/`split`/`unmerge`/`rename` commit restored
  files but not the database. `undo` now refuses such a batch and names the correct inverse
  (e.g. `synto concept unmerge`); pass `--force` to revert files anyway.
- `synto concept unmerge` now strips the absorbed aliases from the winner article's
  frontmatter, so retired labels stop resolving links to the winner after an unmerge.
- `synto concept merge` absorbs the loser's canonical label, not the exact casing typed on
  the command line, so the blessed alias and the previewed "Labels absorbed" list match.
- Disambiguation stubs from `synto concept split` are written with complete frontmatter and
  a stored content hash; the linter no longer flags them as `missing_frontmatter` or
  `stale`.
- `synto concept merge --absorb-edits` updates the winner's content hash after appending the
  absorbed body, so a freshly merged article isn't immediately reported as manually edited.

### Known limitations

- `synto concept unmerge` is best-effort and reverses only the most recent merge for a
  concept. It restores the loser's entity, labels, and source edges, but recreates the
  article as an empty stub — run `synto compile` to repopulate it. Source edges that both
  concepts shared at merge time, and name-keyed state such as rejections, are not restored.

## [0.5.1] - 2026-06-08

### Changed

- Long-form source types now get a book-appropriate concept-extraction ceiling out of the
  box: `source_type: textbook` defaults to 25 concepts and `source_type: paper` to 15,
  without any `synto.toml` override. An explicit per-type override still wins, and raising
  the global `max_concepts_per_source` above a built-in default still takes effect (a
  built-in never lowers a configured value). Other source types are unchanged.

### Fixed

- `synto verify` / `synto approve` no longer race each other over the same draft.
  Both commands now take the pipeline lock before mutating draft state, so a
  concurrent lifecycle operation is refused instead of interleaving. `verify`
  also now treats an already-published twin as the winner: if publish moved the
  article to `wiki/`, verify removes the stale draft row/file instead of
  resurrecting it as `verified`.
- Stray, unbalanced edge punctuation on a name no longer mints a divergent file or wikilink
  (#53). An LLM-emitted concept, synthesis title, or link target like `Phase II)` (a trailing
  `)` with no opener) passed straight through `sanitize_filename`, which keeps parens, so it
  became its own `Phase II).md` instead of resolving to `Phase II`. A new `clean_display_name`
  trims only *unmatched* edge brackets — balanced titles (`Extreme Programming (XP)`, `f(x)`,
  `(see note)`) and ordinary trailing punctuation (`Yahoo!`, `etc.`, `C++`) are untouched — and
  is applied at the three points such a name enters the vault: concept extraction, synthesis
  titles, and broken-link repair / stub creation. `synto maintain --fix` now also heals existing
  dangling links: `fix_broken_links` resolves against lint's authoritative title index instead of
  a second, narrower resolver, which also fixes the prior corruption of path-style links (e.g.
  `[[TCP/IP]]`). Existing vaults are repaired by `synto maintain --fix`; new dirty names are
  cleaned on the next ingest.
- `max_concepts_per_source` (and per-source-type overrides) are no longer silently ignored
  for large sources (#52). `_merge_chunk_results` capped concepts at a hardcoded 8 — the
  per-LLM-call limit — when combining the chunks of a multi-chunk source, before the
  configured, quality-scaled cap was ever consulted. Because 8 equalled the default, the
  limit was invisible until raised, and only long (multi-chunk) sources were affected:
  short single-chunk sources skip the merge and always honored the config. The merge now
  returns the full deduplicated union and leaves capping to the one place that owns it.
  The same hardcoded limit silently bounded a multi-chunk source's named references to 8
  total (they have no other cap), dropping the rest by chunk order before evidence
  filtering; named references are now the full evidenced union as well. Existing vaults are
  unaffected until re-ingested (`synto ingest --force`).
- `synto status`, `synto run`, and `synto maintain` no longer crash with
  `OSError: [Errno 9] Bad file descriptor` on NFS vaults (#56). The pipeline-lock
  liveness probe requested an exclusive lock to test whether the lock was held;
  NFS emulates `flock()` as `fcntl()` write-locks, which require a writable fd, so
  the read-only probe returned `EBADF`. The probe now uses a shared lock (which
  needs only a readable fd), so it also works on read-only mounts and mode-0444
  stale lock files; a filesystem that rejects locking entirely (e.g. a `nolock`
  mount) degrades gracefully instead of crashing. Lint no longer reports the
  pipeline's own lock as "stale" while a run is in progress on NFS.
- Vaults are now portable across operating systems (#55). The state DB stored
  vault-relative paths with OS-native separators, so a vault built on Windows
  (`raw\note.md`) and moved to Linux/macOS (`raw/note.md`) had every note treated as a
  duplicate and skipped, ignoring its sources. Paths are now stored as POSIX everywhere,
  and a one-time migration repairs existing DBs on first open — no rebuild needed. If a
  vault was used on both Windows and Linux/macOS, the migration reconciles the resulting
  duplicate path entries by recency, keeping the most recently updated state rather than
  dropping it. The suspected cause (CRLF/LF line endings) was a red herring: content hashes
  are computed on a newline-normalized body, so line endings never affected note identity.
- Compile no longer ships speculative same-run concept links as live `[[wikilinks]]` (#65).
  When an article mentioned another concept from the same compile batch, the draft writer
  treated every in-batch title as resolvable before that target page existed on disk, so a
  not-yet-materialized concept could be emitted as a real wikilink. Draft generation now
  keeps `[[wikilinks]]` only for titles that already resolve in the vault and unwraps the
  rest to plain text, while preserving links to real pages such as existing concepts and
  `wiki/sources/*` pages. The same fix train also tightens a few related edge cases around
  Windows-path lint reporting, released POSIX lock files, compare soft caps for local
  providers, and resolved-provider reporting in stats/compare output.

## [0.5.0] - 2026-06-03

### Added

- `synto concept rename OLD NEW` (#29). Renames a concept everywhere its name is its
  identity: moves the article file and fixes its title, repoints every inbound wikilink
  across the wiki tree, and migrates the name across the state DB (concepts, aliases,
  compile state, occurrences, knowledge items, plus rejection/block/stub state so a
  rename can't silently unblock a concept or lose review guidance). The old name is kept
  as an alias by default so a later re-ingest that still yields the old surface form is
  canonicalized back to the new concept rather than recreating the old one; pass
  `--drop-old-alias` to opt out. `content_hash` is refreshed for every rewritten tracked
  page so manual-edit protection isn't tripped. `--dry-run` previews without writing.
- Per-role LLM providers (#24). Each model role (`fast` / `heavy` / `embed`) can now
  target a different provider and account via named `[providers.<alias>]` blocks that
  roles reference, e.g. the heavy model on a cloud endpoint and the fast model on local
  Ollama. Each block carries its own connection and API key (`api_key_env`, or a
  per-alias key in the user-private global config) — a different key per model, or none
  for local providers. `synto init` now emits this format by default; legacy `[ollama]` /
  `[provider]` vaults keep working unchanged. Added an `nvidia` (NVIDIA NIM) provider.
  `synto setup` offers this after you configure the primary provider and fast model — it
  asks whether the heavy (writing) model should use a different provider, reuses the primary
  as the fast role, and collects only the heavy one. The split is saved to the user-private
  global config (api_key_env references, never raw keys), so `synto init` reproduces a
  multi-provider vault.
- Per-model parameters (#31). Each role accepts `ctx`, `temperature`, an Ollama `think`
  flag, and an `options`/`headers` passthrough for any provider-native parameter (e.g.
  `top_p`, `reasoning_effort`) with no code change. Thinking-model control resolves the
  ingest timeout: the `fast` role defaults to `think=false` (structured extraction wastes
  reasoning and could exhaust the budget), while `heavy`/answer roles keep thinking on.
  All knobs are overridable per role in `synto.toml`.
- Anthropic-compatible API support (#22). A new client speaks Anthropic's
  `/v1/messages` schema, so providers exposing that API (e.g. Kimi) can back any
  model role. Selected via the provider block like any other connection; keys are
  read from the block's `api_key_env`.

### Fixed

- `synto init` no longer silently wires a vault for Ollama when no provider is configured,
  and no longer rewrites an already-configured vault when there is no global config. With no
  global config it warns (pointing at `synto setup`) on a fresh vault and leaves an existing
  `synto.toml` untouched, instead of writing Ollama's URL into a non-Ollama provider block.
  Also fixes a config-write regex that, when re-syncing a new-format vault, could delete the
  `[models.*]`/`[pipeline]` sections. `synto init` now leaves any existing multi-provider
  (per-role split) vault untouched, instead of collapsing its split when the global default
  provider name happens to match but a role's provider differs.
- Re-running `synto setup` no longer carries a saved per-alias API key over to a different
  provider. When an alias is repointed to another connection (e.g. the default switches from
  Groq to OpenRouter) the stale key is dropped, so it is never sent to the new provider;
  keys for unchanged aliases are preserved.
- `synto compile` no longer crashes (and `synto ingest` no longer fails notes with
  a blank reason) when an OpenAI-compatible provider returns an error as an
  HTTP-2xx body with no usable `choices` (#25). This is common on OpenRouter's free
  tier, which returns rate-limit/overload errors as `{"error": {...}}` with a 200
  status (bypassing the 429 backoff) and pads the body with keep-alive whitespace
  that blanked the old error message. synto now surfaces the provider's real error
  message, retries transient throttling with a bounded budget, and classifies the
  failure so it is isolated to the affected note/concept instead of aborting the run.
- synto no longer crashes with `UnicodeEncodeError` on the ✓/✗ status glyphs on
  Windows legacy (cp1252) consoles or ascii/POSIX locales (#23). `synto` and the
  `install.py` installer now reconfigure stdout/stderr to UTF-8 at startup when
  the console encoding can't represent them.
- `synto serve` no longer looks like it hangs on startup (#30). The stdio MCP
  server now prints a "ready / waiting for an MCP client" line to **stderr**, and
  all server-side logging is routed to stderr so no log record can corrupt the
  stdout JSON-RPC stream. The README now shows the MCP client-config form instead
  of a bare command. The `\n` validation error users saw is expected stdio
  behavior (a terminal is not a valid MCP client); the fix makes the wait legible.
- Wikilink resolution and rewriting hardened across several edge cases (#27, #35,
  #38, #45). Targets now resolve by filename stem so a `[[Note]]` link reaches
  `raw/note.md` regardless of path depth; `sources/` links resolve against
  forward-slash index keys (#26); leading/trailing backslashes are stripped from
  link targets before lookup; and backslashes in the replacement string are
  escaped so links containing them rewrite correctly instead of corrupting the
  output.

## [0.4.0] - 2026-05-29

### Added

- Four new MCP tools for verbatim source access (Feature 42):
  - `read_source_segment(segment_id)` — fetch one paragraph by id.
  - `search_source_segments(query, limit)` — BM25 search across raw segments.
  - `get_source_passages(concept_name, max_passages)` — verbatim passages backing
    a concept, ordered by extraction confidence.
  - `list_segments(source_id, limit, offset)` — enumerate a source's segments.
- New `[mcp.source_access]` config block. `mode = "permissive_only"` (default)
  filters segments by license; `"all"` returns everything; `"deny"` refuses all
  raw-passage tools. Default `permissive_licenses` covers CC-BY, CC-BY-SA, MIT,
  Apache-2.0, BSD-3-Clause, and public-domain.
- SQLite FTS5 virtual table `source_segments_fts` with sync triggers and v16
  migration backfill.
- `synto doctor --backlog [--since 1d|7d|30d|all]` (Feature 29 Stage 5) — an
  opt-in MCP demand-vs-coverage report mined from the audit log: zero-result
  queries, single-source concepts in active demand, repeat weak queries, and
  per-session tool-mix (verbatim vs `answer_question`). The opt-in backlog report
  is hidden unless `--backlog` is passed.
- `synto doctor` now always prints a short **"Verbatim source index"** section
  reporting whether the FTS5 search index is in sync, and an **MCP source-access**
  line showing the effective privacy posture. This is a small addition to the
  default `synto doctor` output; the exit code is unchanged.
- MCP audit rows now record `result_count` and `resolved_label` inside
  `metadata_json` (no schema change), so a successful zero-result call is
  distinguishable from a call that raised.
- New `[mcp] audit_detailed` option (default `false`). When `false`, query text
  and resolved labels are stored as 8-char hashes, preserving the v0.3.0 privacy
  posture. When `true`, the **raw user query text is written in plaintext to the
  local state DB** so the backlog report can show literal queries — enable only
  if storing query text locally is acceptable. The backlog report works under
  either setting (it matches concepts by hash when labels are hashed).
- The `answer_question` MCP tool now carries a description that routes callers
  between it (ready-made synthesis) and the verbatim primitives. Behavior is
  unchanged.
- Ingest now analyzes tracked sources in **segment-aligned chunks** (whole
  `source_segments` packed to the context budget instead of fixed-size byte
  slices) and records which segments produced each concept in
  `concept_occurrences`. This populates `get_source_passages` from the model's own
  extraction at no extra LLM cost, and avoids splitting paragraphs mid-text. Plain
  notes (no segments) keep the existing fixed-size chunking. **Existing vaults:**
  run `synto ingest --force` once to backfill concept→segment links for
  already-ingested sources (re-runs analysis only; published articles are
  untouched). `synto doctor` reports link coverage and prints this tip when links
  are absent.

### Notes

- **Upgrade is seamless and non-destructive.** v0.3.0 vaults are schema v15; only
  the additive v16 migration runs (it creates the FTS index and backfills it). It
  is atomic and idempotent — a re-run after an interrupted upgrade is a no-op. No
  existing command, MCP tool, or output format changes behavior; the four verbatim
  tools and the doctor additions above are the only user-visible changes.
- **The state DB now migrates on the first synto command of any kind after upgrade**
  (including `synto serve`, which now opens the DB even when `[mcp] audit` is off,
  because the verbatim tools query it). On a large vault the first command pays a
  one-time FTS backfill cost.
- **SQLite without FTS5 degrades gracefully.** If your Python/SQLite build lacks the
  FTS5 module, the v16 migration skips the search index (and logs a warning) instead
  of failing — every other command keeps working. Only `search_source_segments` is
  unavailable; the other three verbatim tools query segments directly. `synto doctor`
  reports this state.
- **Privacy gate on upgrade (read this if your vault holds private or copyrighted
  sources).** Upgraded vaults have no declared license on any source. To keep the
  feature working day one, `permissive_only` is relaxed to `"all"` when no source
  declares a license — meaning **all raw source text is readable by any MCP client
  connected to `synto serve`.** This is surfaced loudly: a WARNING at `synto serve`
  startup and a warning in `synto doctor`. To lock it down, declare licenses on your
  sources or set `[mcp.source_access] mode` explicitly in `synto.toml`, then restart
  `serve`.

## [0.3.0] - 2026-05-25

### Added

- MCP server expansion: `synto serve` now exposes 8 tools instead of 3.
  The new five:
  - `search_articles` — lexical search across article name, summary, and aliases.
  - `get_concept` — concept lookup that returns the canonical article body,
    aliases, and frontmatter in one call.
  - `list_sources` — registered source documents (id, title, type).
  - `trace_lineage` — the article's compile lineage from frontmatter (same data
    `synto trace article` shows on the CLI).
  - `answer_question` — runs the same routed query as `synto query` end-to-end
    and returns the answer plus the selected pages. Uses both the fast and the
    heavy model, so it may cost money on paid providers.
- `list_articles` now exposes `status` and `kind` on every result and
  accepts three new filters: `min_status` (defaults to `"published"`, so
  drafts are hidden unless the caller opts in with `"verified"` or
  `"draft"`), `kind` (`"concept"` or `"synthesis"`), and
  `exclude_single_source` (drops articles whose frontmatter has
  `single_source: true`).
- `synto serve` reaches synthesis articles too. Previously only concept
  articles under `wiki/` were browsable; now `wiki/synthesis/*.md` shows
  up in `list_articles` and is readable via `read_article`, tagged
  `kind="synthesis"`. `synto pack export` still groups synthesis under
  its own directory in exported packs.
- Three-state article lifecycle: drafts now progress through `draft` →
  `verified` → `published`. `synto verify` marks reviewed drafts as
  `verified` in place (frontmatter `status: verified`, file stays in
  `.drafts/`). `synto approve` still publishes, preserving the v0.2.2
  workflow. `synto status` reports the verified-pending count alongside
  the draft and published counts. `synto review` adds a verify action
  per draft, and `synto compile` skips verified drafts so human review
  isn't overwritten by regeneration.
- Drafts can be rejected even after they've been verified. Useful when
  you want to walk back a review without editing the database by hand.
- Query vocabulary bridge: `synto query` now hints the routing prompt
  with concepts whose aliases match whole words in the user's question
  (e.g. "ML" surfaces "Machine Learning"). Helps the fast model pick
  the right page when the user types an acronym instead of the
  canonical title. Aliases claimed by two or more concepts are
  filtered. All-caps acronyms of length ≥ 2 bypass the length floor.
  Pure-CJK aliases are a silent no-op for now (Python's `\b` needs
  word boundaries).
- Article frontmatter now includes three quality signals written at
  compile time: `source_count` (int), `single_source` (bool), and
  `source_quality` (`"high"` | `"medium"` | `"low"`). File readers,
  Obsidian plugins, MCP tools, and agents can use these without a DB
  query. `single_source: true` is derived from unique source-document
  identity where available, falling back to path uniqueness. Synthesis
  articles carry `source_count` and `single_source` on the same basis.
- Per-source-type ingest overrides: `[pipeline.source_overrides.<type>]`
  sections in `synto.toml` let you raise `max_concepts_per_source` for
  long-form sources (e.g. `textbook`, `paper`) without changing the
  global default. Quality-based reduction still applies within the
  per-type ceiling. Unknown type keys warn at load time. Re-run
  `synto ingest --force` to apply changes to already-ingested sources.

### Changed

- `synto serve` ships out of the box: the `mcp` library moved from the
  `[mcp]` optional extra to a required dependency. The `synto[mcp]`
  extras flag still resolves but is no longer needed.
- `list_articles` defaults to `min_status="published"`. v0.2.2 callers
  passing no filter previously saw drafts too; they now need to pass
  `min_status="draft"` to get the old behavior.
- `mcp.audit=true` audit rows in `metric_events.metadata_json` now
  record `bool`, `int`, and `float` argument values directly instead
  of hashing them. String arguments are still hashed, so user-supplied
  text never lands raw.
- Schema v15: `wiki_articles.is_draft` (boolean) replaced with
  `wiki_articles.status` (text) constrained to `draft`, `verified`, or
  `published`. Migration is automatic and idempotent on vault open.
  `WikiArticleRecord` no longer serializes `is_draft` in `model_dump()` —
  callers using it directly must switch to `record.status` or one of
  the derived properties (`is_draft`, `is_published`, `is_trusted`).
  External tooling querying `wiki_articles` directly must read `status`
  instead of `is_draft`.
- Schema v14: dropped the five v8 metadata columns from `raw_notes`
  (`source_type`, `origin_uri`, `imported_at`, `normalized_hash`,
  `extractor_version`). They were superseded by `source_documents` in
  v9 and held NULL on every row in every released version. Vaults
  migrate automatically on next open. External tooling reading those
  columns directly must JOIN through
  `source_documents.id = stem(raw_notes.path)`.
- `synto` now requires SQLite ≥ 3.35.0 (for the v14 migration's
  `DROP COLUMN`). Modern Python stdlib distributions on macOS and
  Linux already meet this. The error message at startup names the
  required version if the check fails.
- A downgrade guard now blocks opening a vault whose on-disk schema
  version is newer than the installed `synto` binary, with a clear
  upgrade-required error.

### Fixed

- Structured-output JSON parser repairs odd-length backslash runs
  before LaTeX commands (e.g. `\\\in` → `\\\\in`), fixing transient
  compile failures on real LLM output. Valid `\\uXXXX` escapes and
  standard JSON escapes (`\n`, `\t`, `\"`, etc.) are preserved.
- Saved query answer pages no longer contain literal `\\n` text —
  escaped newlines from the model's JSON output are decoded before
  writing to disk.
- `synto compile` now reads document identity from `source_documents`
  (the canonical store since schema v9) instead of from a duplicate
  set of columns on `raw_notes` that were never populated. The
  `single_source` frontmatter field is unchanged for every current
  ingestion scenario; the fix makes the doc-identity branch alive and
  tested.

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
