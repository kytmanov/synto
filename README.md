# Synto

<p align="center">
     <a href="https://pepy.tech/projects/synto"><img alt="PyPI downloads" src="https://static.pepy.tech/badge/synto"></a>
     <a href="https://github.com/kytmanov/synto/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/kytmanov/synto?style=flat"></a>
     <a href="https://github.com/kytmanov/synto/forks"><img alt="GitHub forks" src="https://img.shields.io/github/forks/kytmanov/synto?style=flat"></a>
     <a href="https://github.com/kytmanov/synto/commits/master"><img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/kytmanov/synto?style=flat"></a> 
     <a href="https://github.com/kytmanov/synto/actions/workflows/ci.yml"><img alt="CI status" src="https://img.shields.io/github/actions/workflow/status/kytmanov/synto/ci.yml?style=flat&amp;label=CI"></a> 
     <a href="https://pypi.org/project/synto/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/synto?style=flat"></a>
     <a href="https://tip.md/kytmanov"><img alt="Tip in Crypto" src="https://tip.md/badge.svg" height="20"></a>
     <a href="https://buymeacoffee.com/kytmanov"><img alt="Buy Me A Coffee" src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-%E2%98%95-yellow?style=flat&logo=buymeacoffee&logoColor=white"></a>
</p>

<p align="center">
<a href="#install">Install</a> · <a href="#quick-start">Quick start</a> · <a href="#how-it-works">How it works</a> · <a href="#features">Features</a> · <a href="#use-cases">Use cases</a> · <a href="#provider-support">Provider support</a> · <a href="#whats-in-a-pack">What's in a pack</a>
</p>

**Turn your raw notes into a self-improving, interlinked wiki — powered by a local LLM.**

You drop Markdown notes in a folder. Synto reads them with a local LLM, extracts the concepts they contain, and writes one cross-linked article per concept. Every note you add makes the wiki richer. Every article stays on your machine unless you decide otherwise.

After setup (~5 minutes) you have: a structured wiki built from your notes, a queryable knowledge base that works without embeddings or a vector database, and an agent-ready pack that Claude, Cursor, or any file-aware AI can install and reason over — including reading your sources' exact words on demand over MCP.

<p align=center>
<img width="341" height="463" alt="image" src="https://github.com/user-attachments/assets/b3e13203-bb4a-42a4-a16d-0d4d93404f71" />
</p>

> [!NOTE]
> Synto succeeds [obsidian-llm-wiki-local](https://github.com/kytmanov/obsidian-llm-wiki-local) (608 ★, 9k+ downloads) — same proven local pipeline, redesigned for distributable knowledge packs.

---

## The idea

[Andrej Karpathy](https://karpathy.ai) described it as the LLM Wiki: a personal knowledge base where the model doesn't just store what you tell it — it synthesizes, cross-references, and keeps everything current. You add raw material; it does the bookkeeping.

The key insight: **treat your notes as source material, not as the final artifact.** A raw note is a claim about the world. A wiki article is the compiled, cross-linked explanation of a concept derived from many notes. The LLM does the compilation step.

```
You write raw notes  →  LLM extracts concepts  →  Wiki articles grow  →  Agent-ready pack
      raw/                    (automatic)              wiki/             .synto/exports/
  quantum.md           "Qubit", "Superposition"     Qubit.md
  ml-basics.md         "Neural Net", "SGD"          Superposition.md ←── [[wikilinks]]
  physics.md           "Qubit"  ← same concept      Neural_Network.md
                              │
                     Duplicates merge, not multiply.
                     One article per concept, fed by all relevant notes.
```

Unlike a chatbot that forgets, the wiki **persists and compounds**. Every note you add and every draft you review makes it smarter.

---

## How it works

Four stages. Two LLM tiers.

```
┌────────────────────────────────────────────────────────────────────────┐
│  Stage 1: Import  Stage 2: Ingest  Stage 3: Compile  Stage 4: Export  │
│                                                                        │
│  synto add ─────┐                                                      │
│  (PDF/md/txt)   ├─ raw/*.md ──────► wiki/.drafts/ ──────► exports/    │
│  .synto/sources/┘  fast model       heavy model      agent-ready      │
│                    (4B params)      (14B+ params)     directory        │
│                                                                        │
│  archives:         extracts:        writes:           produces:       │
│  · original file   · concepts       · one article     · INDEX.json    │
│  · segments        · summaries        per concept     · AGENTS.md     │
│  · source type     · relationships  · cross-linked    · CLAUDE.md     │
│                    · language         [[wikilinks]]   · articles/     │
│                                     · source-type                     │
│                                       prompt                          │
└────────────────────────────────────────────────────────────────────────┘
```

**Why two LLM tiers?** Analysis is pattern-matching — a 4B model running locally can extract "this note is about Qubit, Superposition, and Entanglement" reliably and fast. Writing a coherent, cross-linked article requires more reasoning — a 14B+ model does this well. Splitting the work keeps the pipeline cheap and fast on consumer hardware.

**Your vault layout after `synto init`:**

```
~/my-wiki/
  raw/              ← drop your notes here (Synto never modifies these)
  wiki/             ← published articles (Obsidian-compatible Markdown)
    .drafts/        ← LLM-generated articles waiting for your review
    sources/        ← per-note source summary pages
    queries/        ← saved Q&A sessions
    synthesis/      ← synthesized answers published as wiki pages
  .synto/
    state.db        ← SQLite: note lifecycle, concept registry, metrics
    sources/        ← originals archived via synto add (PDF, md, txt)
    exports/agents/ ← agent-ready export (synto pack export; --out to relocate)
  synto.toml        ← vault config (provider, models, pipeline settings)
```

### Key mechanisms

**Incremental compilation.** Change one note — only the articles derived from that note recompile. A vault with 200 notes doesn't restart from scratch on every run.

**Rejection feedback loop.** Reject a draft and explain why. The reason is stored and injected into the LLM prompt the next time that concept compiles, so the model can address it. Five rejections auto-block the concept until you re-enable it.

```bash
synto reject wiki/.drafts/Qubit.md --feedback "Too abstract, needs a hardware analogy"
# next compile: prompt includes your feedback → better draft
```

**Confidence scores.** Each compiled draft gets a confidence score (0–1). Approve selectively or set a threshold — drafts below it stay in `.drafts/` for manual review.

```bash
synto verify --min-confidence 0.8               # mark trusted drafts as verified
synto approve --min-confidence 0.8              # publish reviewed or fresh drafts to wiki/
```

**Three-state lifecycle.** Drafts move through `draft` → `verified` → `published`. `synto verify` marks a reviewed draft as `verified` in place (frontmatter `status: verified`, file stays in `.drafts/`). `synto approve` publishes drafts to `wiki/`, preserving the existing publish workflow while allowing an explicit staged review step.

**Hand-edit protection.** Edit a published article in Obsidian or any editor. Synto tracks a SHA-256 content hash and detects your change on the next run — your edits are never overwritten by a recompile.

**No embeddings, no vector database.** `synto query` routes questions to relevant articles using `INDEX.json`. It works on any machine without a GPU, FAISS, or Chroma.

**Source-type analysis.** Imported documents carry a type — `notes`, `textbook`, `paper`,
`spec`, `api_docs`, `web_article`, `corp_docs`, `transcript`, or `unknown_text`. During ingest
analysis the fast model receives a matching system prompt: a `paper` prompt extracts
abstract/methods/results structure; an `api_docs` prompt preserves parameter names; a
`textbook` prompt follows chapter/definition flow. If `--type` is omitted, PDFs default to
`paper` and everything else to `notes` — pass `--type` for anything more specific. Long-form
source types also get higher built-in concept ceilings during ingest: `textbook` defaults to
25 concepts and `paper` to 15 unless you set an explicit override in `synto.toml`.

**Concept identity and curation.** A concept's identity is a stable `entity_id`, not its
name. Names and surface forms are just labels pointing at that id, so you can rename,
merge, and split concepts without the changes depending on ingest order. Three situations
come up as your vault grows:

- **Aliases** — several names for one concept (`qubit`, `Qubit`, `qubits`).
- **Homonyms** — one surface form that means two different concepts (`mercury` the planet
  vs. the element).
- **Senses** — one concept that was over-merged and should split into separate ones.

```
IDENTITY   many surface forms + aliases  →  one stable entity_id

   surface forms + aliases  ──►  [ entity_id ] ─┬─►  rename  relabel; old name kept as alias
   "qubit"  "Qubit"  "qubits"                   ├─►  merge   fold a second entity into this one
                                                └─►  split   one entity → several named senses

HOMONYM    one surface, two existing entities  →  pick which one

   "mercury"  (ambiguous)  ┬─►  Planet
                           └─►  Element     →  keep "mercury" Planet  (assign its occurrences)
```

Ingest that later extracts a prior weak alias as its own concept records an explicit merge
candidate instead of silently absorbing it, so you decide.

**Diagnose:**

- `synto doctor` / `synto maintain` — surface duplicate-name collisions and merge candidates.
- `synto concept inspect NAME` — show the backing entity, its aliases, sources, ambiguous
  occurrences, and suggested actions.

**Reshape identity** (all support `--dry-run`, auto-commit, and are reversible with `synto undo`):

- `synto concept rename OLD NEW [--keep-old-alias]` — relabel in place, keeping the old name
  as an alias by default so re-ingested notes don't recreate it.
- `synto concept merge LOSER WINNER [--absorb-edits]` — move sources and edges onto the
  winner, retire the loser article, and keep the loser's labels as aliases on the winner.
- `synto concept split NAME --sense SENSE SOURCE_PATH ...` — partition a concept's sources
  across senses (repeat `--sense` per source) and leave a disambiguation page at the old name.
- `synto concept unmerge NAME` — best-effort reverse of the most recent merge for that name.
  Several things are not restored; see `synto concept unmerge --help` for the exact contract.

**Resolve homonyms:**

- `synto concept keep SURFACE ENTITY` — assign the ambiguous occurrences of a surface form to
  one entity. This updates the state DB directly; it does not auto-commit and is not reversible
  with `synto undo`.

Example — splitting a homonym into two concepts:

```bash
synto concept split Mercury --sense planet raw/astronomy.md --sense element raw/chemistry.md
```

`git revert` on a curation commit will not restore the database (`.synto/state.db` is
gitignored). For the auto-committing commands above, `synto undo` detects these batches and
names the correct inverse command.

---

## Use cases

**Architectural reference from a textbook**

[@dobryakov](https://github.com/dobryakov) turned Tanenbaum's *Distributed Systems* into a cross-linked wiki and used it as context for Claude/Cursor. Instead of falling back to generic integration advice, the AI started reasoning from distributed-systems concepts like persistent messaging, idempotency, eventual consistency, and two-phase commit when designing a production-grade event-driven architecture. Read the write-up: [Book-as-context](https://www.dobryakov.com/howto/book-as-context.html). Any technical book, spec, or documentation set can become a live context layer for your AI tools.

**Course material as a wiki**

[Andrej Karpathy's LLM Wiki idea](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) works well for learning. Start with lectures, transcripts, and notes from a course like Karpathy's neural-network series. Instead of searching through raw files every time you ask a question, Synto turns them into a linked wiki that grows as you add more material. Then an agent can answer questions about backpropagation, attention, or training using your own course material. Good answers can also be saved back into the wiki, so it becomes more useful over time.

**Your own research notes**

Works today with Markdown. Drop notes in `raw/`, run `synto run`, and get a cross-linked wiki exported as an agent-ready pack. Expose it via `synto serve` as a local MCP server, or ship the pack directory for any file-aware agent to use.

---

## Features

**Incremental compiles.** Each concept gets its own article. When you change a source note, only the articles tied to that note recompile — not the whole vault.

**Rejection feedback.** Reject a draft and attach a reason. The next compile of that concept includes your feedback in the prompt. Five rejections auto-block the concept until you run `synto unblock`.

**Hand-edit protection.** Edit any published article directly. Synto tracks a content hash — if the file changed since last compile, it won't overwrite your work.

**Interactive review.** `synto review` opens a terminal UI per draft: read the article, approve, reject with feedback, launch your editor, or diff against the previous version. Full control over what enters your wiki.

**File watcher.** `synto watch` runs in the background and processes anything you drop into `raw/` automatically. Ingest and compile happen while you keep writing.

**Query and synthesize.** `synto query "what is X?"` answers from your published wiki without embeddings or a vector database. Add `--synthesize` to save the answer as a permanent wiki page with source citations and hand-edit protection.

**Pack export.** `synto pack export --target agents` produces a portable directory any file-aware agent can read: articles, `INDEX.json` for fast concept lookup, source provenance, and agent-readable entry points.

**MCP server.** `synto serve` exposes your wiki as a local MCP server with 12 tools. Eight cover the published wiki: `list_articles`, `read_article`, `find_concept`, `search_articles`, `get_concept`, `list_sources`, `trace_lineage`, and `answer_question`. Four more (below) expose the raw source text. Wire it into Claude Code, Cursor, or any MCP-compatible client in one command. Drafts are hidden by default; `answer_question` runs the same routed query as `synto query` end-to-end (uses both fast and heavy models), so it may cost money on paid providers.

### Verbatim source tools

Four additional MCP tools expose raw source paragraphs to frontier-model callers.
Use them when you want the source's own words, not a synto-generated synthesis.
(For a ready-made answer from your own local models instead, use `answer_question`.)

- `read_source_segment(segment_id)` — fetch one paragraph by id.
- `search_source_segments(query, limit=10)` — BM25 search across raw segments.
- `get_source_passages(concept_name, max_passages=5)` — verbatim passages backing
  a known concept.
- `list_segments(source_id, limit=200, offset=0)` — enumerate a source's segments
  in reading order.

#### Privacy and source access

Raw-paragraph access is governed by `[mcp.source_access]` in `synto.toml`:

```toml
[mcp.source_access]
mode = "permissive_only"  # "all" returns every segment; "deny" disables raw-passage tools
permissive_licenses = ["CC-BY", "CC-BY-SA", "MIT", "Apache-2.0", "BSD-3-Clause", "public-domain"]
```

- **`permissive_only`** (default) — only sources whose `license` is in
  `permissive_licenses` return raw text; everything else is hidden.
- **`all`** — every segment is returned. Use for a private personal vault where you
  trust the connected MCP client.
- **`deny`** — the four verbatim tools refuse all raw passages.

**Legacy-vault behavior (important).** A vault with no declared license on any source
cannot return anything under `permissive_only`, so to keep the feature working `synto`
treats such vaults as `"all"` until you declare a license or set `mode` explicitly. That
means **all raw source text is readable by any connected MCP client.** This is surfaced as
a WARNING at `synto serve` startup and in `synto doctor`. If your vault holds private or
copyrighted material, declare licenses on your sources or set `mode` explicitly, then
restart `serve`.

**Audit detail.** `[mcp] audit_detailed` (default `false`) keeps query text and resolved
labels as 8-char hashes in the local state DB. Set it to `true` only if you want the
`synto doctor --backlog` report to show literal query text — it writes raw queries to the
DB. The backlog report works either way.

**SQLite without FTS5.** If your SQLite build lacks the FTS5 module, the search index is
skipped on upgrade (with a warning) and only `search_source_segments` is unavailable; the
other three verbatim tools and every other command keep working.

**Self-maintenance.** `synto maintain` repairs broken wikilinks, creates stubs for missing targets, and reports orphans, stale articles, and missing frontmatter. `--dry-run` shows the structural-health report without changing anything.

**A/B model comparison.** `synto compare` runs your query set against two different models in isolated copies of your vault so you can evaluate a model switch without touching anything live.

**Quality evaluation.** `synto eval` scores your wiki offline: concept coverage, citation support, link resolution, `INDEX.json` validity. Run it before a pack export or in CI.

**Diagnostics.** `synto doctor` checks your configuration, provider connectivity, and vault integrity — start here when something isn't working.

**Multi-language.** Each note's language is auto-detected at ingest. Articles are written in that language. No hard-coded word lists or language config required.

**Git-aware.** Every automatic operation commits with a `[synto]` prefix. `synto undo` reverts the last N auto-commits. Raw notes are never modified.

**Source import.** `synto add` imports PDFs, Markdown, and text files as tracked source
documents. PDFs are segmented into heading-aware chunks, archived under `.synto/sources/`,
and written back as canonical `raw/*.md` notes for the normal ingest flow. Pick the right
type for your document to get the matching ingest-analysis prompt:

| Type | Use for |
|---|---|
| `notes` | Your own notes, meeting minutes, personal writing |
| `textbook` | Educational material with chapters and exercises |
| `paper` | Academic papers (abstract / methods / results) |
| `spec` | Technical specifications, RFCs, standards |
| `api_docs` | API references, SDK docs, OpenAPI specs |
| `web_article` | Blog posts, news articles, web clips |
| `corp_docs` | Internal wikis, runbooks, design documents |
| `transcript` | Video/audio transcripts, interview notes |
| `unknown_text` | Fallback when text doesn't fit a richer source type |

**Compile lineage.** Every compiled article records which source notes and compile run it
came from. `synto trace article <name>` prints the full history: timestamp, model,
contributing sources.

**LLM response cache.** Identical prompts reuse cached responses from a local SQLite
table instead of hitting the model. `synto maintain --clear-cache` flushes it;
`--older-than N` prunes entries older than N days.

**Pack extension flag.** `synto add --extend-pack NAME` is reserved for future pack-scoped
imports and is currently a safe no-op that does not mutate `synto.toml`.

---

## Install

Requires **Python 3.11+** and an LLM provider ([Ollama](https://ollama.com) recommended for local use).

```bash
pip install synto
# or
uv tool install synto
```

The MCP server (`synto serve`) ships in the base install — no extras flag needed.

---

## Quick start

### 1. Pull models

```bash
# Install Ollama: https://ollama.com/download
ollama pull gemma4:e4b      # fast model — analysis and concept extraction
ollama pull qwen2.5:14b     # heavy model — article writing
```

> **Minimal setup:** pull only `gemma4:e4b` and assign it to both fast and heavy in the wizard below. Enough to get started.

### 2. Run the setup wizard

```bash
synto setup
```

An interactive wizard selects your provider, configures the endpoint, picks models, and optionally sets a default vault. Takes about 30 seconds.

```
╭──────────────────────────────────────────────╮
│         synto  ·  setup                      │
╰──────────────────────────────────────────────╯

  Step 1  Provider

    Local (no API key needed):
      1. Ollama         http://localhost:11434  [default]
      2. LM Studio      http://localhost:1234/v1
      3. vLLM           http://localhost:8000/v1
      ...
    Cloud (API key required):
      9. Groq           https://api.groq.com/openai/v1
     10. Together AI    https://api.together.xyz/v1
      ...

    Select provider [1]: _

  Step 2  Fast model  (analysis · 3–8B recommended)
    1  gemma4:e4b    2  phi4-mini    3  other
    Select [1]: _

  Step 3  Heavy model  (writing · 14B+ recommended)
    1  qwen2.5:14b   2  same as fast   3  other
    Select [1]: _
```

Settings are saved to `~/.config/synto/config.toml`. API keys are stored only in this user-private file, never inside your vault.

### 3. Create a vault

```bash
synto init ~/my-wiki
```

Creates the folder structure and a `synto.toml` pre-filled with your wizard settings.

### 4. Add notes and sources

Drop any `.md` files into `~/my-wiki/raw/`. Web clips, book notes, meeting notes, transcripts — anything.

```
~/my-wiki/raw/
  quantum-computing.md
  ml-fundamentals.md
  distributed-systems-chapter3.md
```

**To import a PDF or other structured document:**

```bash
synto add paper.pdf --type paper --vault ~/my-wiki
synto add textbook_chapter.pdf --type textbook --vault ~/my-wiki
synto add api-reference.md --type api_docs --vault ~/my-wiki
```

Use `--type` to select the matching ingest-analysis prompt (see the table in Features).
If omitted, PDFs are treated as `paper` and everything else as `notes` — pass `--type` for
anything more specific (`spec`, `transcript`, `api_docs`, …).

`synto add --force` re-imports an existing source in place. `--extend-pack` is reserved for
future pack integration and is currently a safe no-op.

### 5. Run the pipeline

```bash
synto run --vault ~/my-wiki
```

Ingest + compile. Drafts appear in `wiki/.drafts/`. Then review and publish:

```bash
synto review --vault ~/my-wiki                    # inspect each draft: approve / verify / reject / edit
# or
synto verify --all --vault ~/my-wiki              # mark all drafts verified in place
synto approve --all --vault ~/my-wiki             # publish drafts to wiki/
```

**One-command flow** (skip review):
```bash
synto run --auto-approve --vault ~/my-wiki
```

**Set-it-and-forget-it** (auto-process every time you save a note):
```bash
synto watch --vault ~/my-wiki
```

**Query your wiki** once articles are published:
```bash
synto query "what is consistent hashing?" --vault ~/my-wiki
synto query "explain backpropagation" --vault ~/my-wiki --synthesize
```

**Expose as MCP server** (Claude Code, Cursor, any MCP client). For local stdio mode,
`synto serve` is launched *by* the client, not run by hand — point your client's config
at it:
```json
{ "mcpServers": { "synto": { "command": "synto", "args": ["serve", "--vault", "~/my-wiki"] } } }
```
Run by hand it will print a "waiting for a client" line on stderr and otherwise sit
idle — that is expected, not a hang.

For remote MCP clients that support Streamable HTTP, run:
```bash
synto serve --vault ~/my-wiki --transport streamable-http --host 127.0.0.1 --port 8000
```
Then connect the client to `http://127.0.0.1:8000/mcp`. To reach it from another machine,
bind a routable address — `--host 0.0.0.0` — and treat it as a trusted-network-only or
behind-a-proxy deployment.

This mode has **no authentication**; control access with a trusted network, firewall, or
reverse proxy. DNS-rebinding protection is enforced: only `Host` headers for loopback and
the bind address are accepted. When fronting it with a reverse proxy (which forwards its
own public hostname), add that hostname so requests are not rejected:
```bash
synto serve --vault ~/my-wiki --transport streamable-http --host 127.0.0.1 \
  --allowed-host synto.example.com
```

If you bind wildcard IPv6 with `--host ::`, Synto still auto-allows only loopback.
Add any remote IPv6 literal or public hostname explicitly, for example:
```bash
synto serve --vault ~/my-wiki --transport streamable-http --host :: \
  --allowed-host "[2001:db8::5]"
```

Note: source-access privacy applies over HTTP exactly as locally. If no source declares a
license, the privacy gate opens to `all` and raw source text becomes readable by any
connected client — far riskier over a network. Declare licenses or set `[mcp.source_access]`
in `synto.toml` before exposing the server.

---

## Provider support

| Local (offline, no API key) | Cloud (API key required) |
|---|---|
| Ollama | Groq |
| LM Studio | Together AI |
| vLLM | Fireworks AI |
| llama.cpp | DeepInfra |
| LocalAI | OpenRouter |
| TGI | Mistral AI |
| SGLang | DeepSeek |
| Llamafile | SiliconFlow |
| Lemonade | Perplexity |
| | xAI (Grok) |
| | NVIDIA NIM |
| | Azure OpenAI |
| | Kimi (Anthropic-compatible) |
| | Custom OpenAI-compatible |

Any OpenAI-compatible endpoint works. Use `synto setup` to configure interactively, or edit `~/.config/synto/config.toml` directly.

### Per-role providers

Each model role can use a different provider and account. Define connections as named
`[providers.<alias>]` blocks and point roles at them — for example, the fast model on local
Ollama and the heavy model on a cloud account:

```toml
[providers.default]
name = "ollama"
url  = "http://localhost:11434"

[providers.ngc]
name = "nvidia"
url  = "https://integrate.api.nvidia.com/v1"
api_key_env = "NVIDIA_API_KEY"   # env var name only — never the key itself

[models.fast]
provider = "default"
model    = "gemma4:e4b"

[models.heavy]
provider = "ngc"
model    = "qwen2.5:14b"
ctx      = 32768
```

The API key belongs to the provider block, so each model can have its own key (point two
blocks at the same provider with different `api_key_env`) or none at all (local providers).
Keys are read from the named env var, the provider's conventional env var, `SYNTO_API_KEY`,
or the user-private `~/.config/synto/config.toml` — **never** from the vault's `synto.toml`.

`synto setup` can configure this interactively: after you pick the primary provider and fast
model, answer "yes" to "Use a different provider for the heavy (writing) model?" and the primary
is reused as the fast role while you set up the heavy one. It saves the split to the global config
so `synto init` reproduces it for new vaults.
Re-running `synto init` only re-syncs a simple single-provider vault whose provider matches your
global default; in that case any per-model `options`/`think` you hand-edited there are replaced.
It leaves a vault untouched when there is no global config, when the vault is set to a different
provider than the global default, or when the vault already splits roles across providers (a
per-role setup). Change those via `synto setup` or by editing `synto.toml` directly.

#### NVIDIA / NGC

NVIDIA inference is OpenAI-compatible; pick the block that matches how your model is served:

- **Self-hosted NIM** (a model you host with NGC — the NIM container is pulled from NGC and run
  on your own infra). It exposes `http://<host>:8000/v1` and usually needs **no key** on inference
  requests. Use `name = "custom"` + your `url` so no environment key is auto-sent:

  ```toml
  [providers.ngc]
  name    = "custom"
  url     = "http://your-host:8000/v1"
  timeout = 600   # raise for a large self-hosted model — the cloud default is 120s

  [models.heavy]
  provider = "ngc"
  model    = "meta/llama-3.1-70b-instruct"
  ```

- **NVIDIA Cloud Functions (NVCF)** — set the OpenAI-compatible LLM Gateway base URL for your
  function (from NVIDIA's console, ending in `/v1`) and `api_key_env = "NGC_API_KEY"`.
- **API Catalog** (`build.nvidia.com`) — `name = "nvidia"` (default URL) with
  `api_key_env = "NVIDIA_API_KEY"` (an `nvapi-` key).

The legacy `/v2/nvcf/pexec` invocation form is not OpenAI-compatible and is not supported — use
the LLM Gateway URL. `synto doctor` may list an NVCF model as "not found" if the per-function
gateway has no `/v1/models`; that doesn't mean inference is broken.

### Advanced model parameters

These are hand-edit-only (not prompted by `synto setup`):

```toml
[models.heavy]
model       = "qwen3.5:9b"
think       = true     # Ollama thinking models: on for heavy by default, off for fast
temperature = 0.3      # override the per-stage default

[models.heavy.options]   # raw passthrough — any provider-native parameter
top_p = 0.9
```

`think` controls Ollama's reasoning flag (a no-op on OpenAI/Anthropic-compatible providers;
use `options` for those). Keys under `[models.<role>.options]` are merged into the provider
request as-is and override the matching first-class field, so set computed values like
`num_predict` only if you mean to.

---

## What ships now

- Full ingest → compile → approve pipeline; supports Markdown notes and Obsidian vaults
- `synto pack export --target agents` — portable knowledge pack with `INDEX.json` and agent metadata
- `synto serve` — MCP server with stdio and Streamable HTTP transports, exposing 12 tools. Eight wiki tools: `list_articles`, `read_article`, `find_concept`, `search_articles`, `get_concept`, `list_sources`, `trace_lineage`, `answer_question`. Four verbatim-source tools: `search_source_segments`, `get_source_passages`, `read_source_segment`, `list_segments`. Quality signals (`status`, `confidence`, `source_count`, `single_source`) are surfaced on every article ref so agents can self-filter; `min_status` defaults to `"published"` to keep drafts out of agent context.
- `synto doctor --backlog` — reads the MCP audit log to show what to ingest next: zero-result queries, single-source concepts in active demand, and the verbatim-vs-`answer_question` tool mix.
- `synto query` — index-routed Q&A with optional synthesis to `wiki/synthesis/`
- `synto review` — interactive draft review: approve, reject, edit, or diff before publishing
- `synto concept rename|merge|split|unmerge|inspect|keep` — stable entity identity and
  curation: `synto doctor`/`maintain` surface candidates and collisions; `inspect` diagnoses
  and `keep` resolves homonyms; `merge` folds two concepts into one and `split` creates a
  disambiguation page; `unmerge` is best-effort with explicit limitations (see command help);
  `--dry-run` on `rename`/`merge`/`split`.
- `synto watch` — file watcher: auto-ingest and compile on every save
- `synto maintain` — wiki health check, stub creation, orphan cleanup
- `synto eval` — offline structural evaluation (coverage, citation support, link resolution)
- `synto compare` — A/B model comparison without touching your vault
- `synto doctor` — configuration and connectivity diagnostics
- Multi-language: notes are ingested and compiled in their source language
- 20+ LLM providers supported via OpenAI-compatible API
- `synto add SOURCE` — import PDF, Markdown, or text files as tracked source documents;
  PDFs are segmented automatically into heading-aware chunks and written back as canonical raw notes
- Source-type prompts: built-in templates for `notes`, `textbook`, `paper`, `spec`,
  `api_docs`, `web_article`, `corp_docs`, `transcript`, plus `unknown_text` fallback,
  select the optimal ingest strategy per document type
- Compile lineage: every article records its source notes and compile run;
  `synto trace article <name>` shows the full history
- LLM response cache: identical prompts reuse cached responses;
  `synto maintain --clear-cache` manages it

---

## What's in a pack

```
.synto/exports/agents/   ← default output dir (synto pack export; --out DIR to relocate)
  articles/           one Markdown file per concept
  synthesis/          published synthesis articles (when present)
  index/
    INDEX.json        machine-readable index with stable article IDs
  agent/
    manifest.json     capabilities and pack metadata
    concepts.json     concept registry with aliases
    sources.json      source provenance
    routes.json       concept/source routing hints
  AGENTS.md           agent-readable entrypoint and usage guide
  CLAUDE.md           Claude Code context file
```

Any file-aware agent can read the articles directly. `INDEX.json` enables fast concept lookup without a database. `synto serve` exposes 12 MCP tools — browse (`list_articles`), read (`read_article`, `get_concept`, `trace_lineage`, `list_sources`), search (`search_articles`, `find_concept`), answer (`answer_question`, which runs the full query pipeline), and read raw source text (`search_source_segments`, `get_source_passages`, `read_source_segment`, `list_segments`).

---

## Data & Privacy

- **Local by default.** Ollama and LM Studio process all content on your machine — notes never leave it.
- **Cloud providers.** If you configure a cloud provider, note content and wiki text are sent to that service. Review their privacy policy before use.
- **No remote analytics.** Synto does not send usage data anywhere. Local runtime and cost metrics stay in your vault database.
- **Pack exports.** Exported packs include raw notes, sources, wiki articles, queries, and synthesis by default. Review your vault before sharing a pack.
- **API keys.** Stored in `~/.config/synto/config.toml` (user-owned, not inside the vault). Never commit that file.
- **Verbatim MCP tools gate raw text by source license.** The four verbatim-source tools return raw paragraphs only from sources whose license is permissive (`[mcp.source_access]` in `synto.toml`). A vault with no declared licenses is treated as `"all"` — every segment is readable by a connected MCP client — and `synto serve`/`synto doctor` warn you about it. See [Privacy and source access](#privacy-and-source-access).

---

## Requirements

- Python 3.11+
- An LLM provider: [Ollama](https://ollama.com) (local), LM Studio, or any OpenAI-compatible endpoint

**Minimum recommended:** a 4B model for ingest (e.g. Gemma 4 `gemma4:e4b`) and a 14B+ model for compilation (e.g. Qwen 2.5 14B `qwen2.5:14b`). This is the ground-floor setup for quality output — works offline on 16 GB RAM. A single 4B model for both stages works if you're just getting started.

---

## Migrate from obsidian-llm-wiki-local

**Existing obsidian-llm-wiki-local vaults must be migrated first.** Run `migrate-olw` to convert the vault layout:

```bash
synto migrate-olw --vault ~/my-old-vault
```

Copies `wiki.toml` → `synto.toml` and `.olw/` → `.synto/`. Notes and articles are untouched. Old files are preserved — delete them once you've verified everything works.
