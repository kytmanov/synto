# Obsidian Web Clipper → raw/ Setup

Save web articles directly to your `raw/` folder so `synto` can ingest them automatically.

---

## Option 1: Obsidian Web Clipper (Official)

1. Install [Obsidian Web Clipper](https://obsidian.md/clipper) (Chrome/Firefox/Safari)
2. Open the extension settings → **Template**
3. Set **Save location** to `raw/` in your vault
4. Use this template:

```markdown
---
source_url: {{url}}
source_title: {{title}}
captured: {{date}}
---

# {{title}}

{{content}}
```

5. Clip any article → it lands in `raw/` → run `synto ingest raw/{{filename}}.md`

With `synto watch` running, ingest + compile happen automatically after each clip.

---

## Option 2: Markdownload (Browser Extension)

[Markdownload](https://github.com/deathau/markdownload) is a community alternative with more template options.

1. Install from [Chrome Web Store](https://chrome.google.com/webstore/detail/markdownload-markdown-web/pcmpcfapbekmbjjkdalcgopdkipoggdi) or Firefox Add-ons
2. Settings → **Download folder**: point to your vault's `raw/` directory
3. Template:

```
---
source_url: {baseURI}
source_title: {title}
captured: {date:YYYY-MM-DD}
---

{content}
```

---

## Option 3: Share Sheet (iOS/macOS)

On iOS, use the Obsidian share sheet to save web content. Set the default folder to `raw/` in Obsidian settings → **Core plugins → Note composer → Default location**.

---

## Option 4: Manual + watch

Simplest approach: copy-paste article text into a new `.md` file in `raw/`. With `synto watch` running, it will be ingested within a few seconds (configurable `watch_debounce` in `synto.toml`).

---

## Automating with `synto watch`

Start the watcher in a dedicated terminal (or as a background process):

```bash
synto watch --vault ~/my-vault
# With auto-approve (skips draft review):
synto watch --vault ~/my-vault --auto-approve
```

**Flow:** clip article → saved to `raw/` → watcher triggers after debounce → `ingest` → `compile` → draft in `.drafts/` (or auto-published if `--auto-approve`)

Check `wiki/log.md` in Obsidian to see what was ingested and which concepts were created or updated.
