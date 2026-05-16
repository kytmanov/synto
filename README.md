# Synto

<p align="center">
     <a href="https://github.com/kytmanov/synto/commits/master"><img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/kytmanov/obsidian-llm-wiki-local?style=flat"></a> 
     <a href="https://github.com/kytmanov/synto/actions/workflows/ci.yml"><img alt="CI status" src="https://img.shields.io/github/actions/workflow/status/kytmanov/synto/ci.yml?style=flat&amp;label=CI"></a> 
     <a href="https://pypi.org/project/synto/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/synto?style=flat"></a>
</p>

**Turn your raw notes into a self-improving, interlinked wiki — powered by a local LLM.**

Synto is a knowledge compiler. Point it at any source of expertise and it produces a portable wiki pack any AI agent can install and query.


**Local-first, provider-flexible.** Runs 100% locally with Ollama or LM Studio. Switch to a cloud provider — Groq, OpenRouter, Mistral — when you need more power. Your notes and source material never leave your machine unless you choose.

Synto succeeds [obsidian-llm-wiki-local](https://github.com/kytmanov/obsidian-llm-wiki-local) (608 ★, 9k+ downloads) — same proven local pipeline, redesigned for distributable knowledge packs.

<p align=center>
<img width="341" height="463" alt="image" src="https://github.com/user-attachments/assets/b3e13203-bb4a-42a4-a16d-0d4d93404f71" />
</p>

---

## Why it exists

AI agents are only as good as the context you give them. A folder of notes or a raw PDF is not context — it's noise. Synto processes your source material through a local LLM pipeline, extracts concepts, writes cross-linked articles, and exports a structured directory: articles, an index, a concept graph, and agent-readable metadata. The pack installs like a package. Any agent gets a domain expert layer, not a file dump.

---

## Use cases

**Architectural reference from a textbook**

[Dobryakov took Tanenbaum's *Distributed Systems*](https://www.facebook.com/dobryakov/posts/pfbid021pwPPJZ77KsyWMs3zTdMQEbgpmxgeE9wzoX1SDFE12y5vC3HtdFZV5i5HnS2dUAul), compiled it into a cross-linked wiki, and installed it into Claude/Cursor. Instead of generic suggestions, the AI started reasoning from established distributed systems principles — recommending consistent hashing and two-phase commit instead of "use a shared database." Any technical book, spec, or documentation set becomes a live context layer for your AI.

**Karpathy's LLM Wiki**

[Andrej Karpathy's](https://karpathy.ai) neural-network series is 20 hours of dense, high-quality material. Compile the transcripts and blog posts into a Synto pack and the content becomes queryable: any agent in your project can explain backpropagation, walk through the attention mechanism, or cite a specific lecture — from the source, not from model weights.

**Your own research notes**

Works today with Markdown. Drop notes in `raw/`, run `synto run`, and get a cross-linked wiki exported as an agent-ready pack. Expose it via `synto serve` as a local MCP server, or ship the pack directory for any file-aware agent to use.

---

## How it works

Three stages, two LLM tiers:

1. **Ingest** — fast model reads each source, extracts concepts and summaries
2. **Compile** — heavy model writes one cross-linked article per concept
3. **Export** — `synto pack export` produces an agent-ready directory

The fast model handles analysis (4B parameters is enough). The heavy model handles writing (14B+ recommended locally, or any cloud model). Both tiers are configurable independently.

---

## Install

```bash
pip install synto
# or
uv tool install synto
```

For MCP server support:

```bash
pip install "synto[mcp]"
```

---

## Quick start

```bash
synto setup                                   # configure LLM provider
synto init ~/my-vault                         # create vault

# add Markdown notes to ~/my-vault/raw/
synto run --vault ~/my-vault                  # ingest + compile into wiki/.drafts/
synto review --vault ~/my-vault               # inspect drafts interactively
# or: synto approve --all --vault ~/my-vault

synto pack export --target agents --vault ~/my-vault
synto serve --vault ~/my-vault                # expose as MCP server (requires [mcp])
```

If you want a one-command flow, use `synto run --auto-approve --vault ~/my-vault`.
That skips draft review and publishes directly into `wiki/`.

---

## What's in a pack

```
pack/
  articles/           one Markdown file per concept
  index/INDEX.json    machine-readable index with stable article IDs
  agent/
    manifest.json     capabilities, pack metadata
    concepts.json     concept registry
    sources.json      source provenance
  AGENTS.md           agent-readable entrypoint
  CLAUDE.md           Claude Code context file
```

Any file-aware agent can read the articles directly. `INDEX.json` enables fast concept lookup without a database. `synto serve` exposes `list_articles`, `read_article`, and `find_concept` as MCP tools.

---

## What ships now

- Full ingest → compile → approve pipeline; currently supports Markdown notes and Obsidian vaults
- `synto pack export --target agents` — portable knowledge pack with INDEX.json and agent metadata
- `synto serve` — read-only MCP server (`list_articles`, `read_article`, `find_concept`)
- `synto query` — index-routed Q&A with optional synthesis to `wiki/synthesis/`
- `synto review` — interactive draft review: approve, reject, edit, or diff before publishing
- `synto watch` — file watcher: auto-ingest and compile on every save
- `synto maintain` — wiki health check, stub creation, orphan cleanup
- `synto eval` — offline structural evaluation (coverage, citation support, link resolution)
- `synto compare` — A/B model comparison without touching your vault
- Multi-language: notes are ingested and compiled in their source language
- 20+ LLM providers supported via OpenAI-compatible API

---

## Data & Privacy

- **Local by default.** Ollama and LM Studio process all content on your machine — notes never leave it.
- **Cloud providers.** If you configure an OpenAI-compatible cloud provider, note content and wiki text are sent to that service. Review their privacy policy before use.
- **No remote analytics.** Synto does not send usage analytics anywhere. Local runtime and cost metrics stay in your vault database.
- **Pack exports.** Exported packs include raw notes, sources, wiki articles, queries, and synthesis by default. Review your vault before sharing a pack.
- **API keys.** Stored in `~/.config/synto/config.toml` (user-owned, not inside the vault). Never commit that file.

---

## Requirements

- Python 3.11+
- An LLM provider: Ollama (local), LM Studio, or any OpenAI-compatible endpoint

Recommended local setup: a 4B model for ingest (e.g. Gemma 4), a 14B+ model for compilation (e.g. Qwen 2.5 14B). Works entirely offline.

---

## Migrate from obsidian-llm-wiki-local

**Existing obsidian-llm-wiki-local vaults must be migrated first.** Run `migrate-olw` to copy `wiki.toml` and `.olw/` into the current Synto layout before using normal commands.

```bash
synto migrate-olw --vault ~/my-old-vault
```

Copies `wiki.toml` → `synto.toml` and `.olw/` → `.synto/`. Notes and articles are untouched. Old files are preserved — delete them once you've verified everything works.
