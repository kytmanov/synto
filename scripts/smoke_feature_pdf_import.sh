#!/usr/bin/env bash
# Smoke tests for feature/pdf-import: synto add, source-type prompts,
# compile lineage, and semantic cache infrastructure.
# Runs standalone — does not depend on smoke_test.sh.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'

# ── provider env vars (same conventions as smoke_test.sh) ─────────────────────
PROVIDER="${PROVIDER:-ollama}"
case "$PROVIDER" in
  ollama)
    PROVIDER_URL="${PROVIDER_URL:-${OLLAMA_URL:-http://localhost:11434}}"
    FAST_MODEL="${FAST_MODEL:-gemma4:e4b}"
    HEAVY_MODEL="${HEAVY_MODEL:-gemma4:e4b}"
    FAST_CTX=8192; HEAVY_CTX=8192
    ;;
  lm_studio)
    PROVIDER_URL="${PROVIDER_URL:-http://localhost:1234/v1}"
    FAST_MODEL="${FAST_MODEL:-google/gemma-4-e4b}"
    HEAVY_MODEL="${HEAVY_MODEL:-google/gemma-4-e4b}"
    FAST_CTX=8192; HEAVY_CTX=8192
    ;;
  *)
    if [[ -z "${PROVIDER_URL:-}" || -z "${FAST_MODEL:-}" ]]; then
      echo "ERROR: PROVIDER=$PROVIDER requires PROVIDER_URL and FAST_MODEL to be set."
      exit 1
    fi
    FAST_CTX="${FAST_CTX:-8192}"; HEAVY_CTX="${HEAVY_CTX:-8192}"
    HEAVY_MODEL="${HEAVY_MODEL:-$FAST_MODEL}"
    ;;
esac

# ── vault + config ─────────────────────────────────────────────────────────────
VAULT_DIR="$(mktemp -d)"
DB="$VAULT_DIR/.synto/state.db"
export SYNTO_VAULT="$VAULT_DIR"
OLW="uv run --project $REPO_DIR synto"
mkdir -p "$VAULT_DIR/raw"

# Write model config before any section so LLM sections work
if [[ "$PROVIDER" == "ollama" ]]; then
  cat > "$VAULT_DIR/synto.toml" <<TOML
[models]
fast = "$FAST_MODEL"
heavy = "$HEAVY_MODEL"

[ollama]
url = "$PROVIDER_URL"
timeout = 900
fast_ctx = $FAST_CTX
heavy_ctx = $HEAVY_CTX

[pipeline]
auto_approve = false
auto_commit = false
TOML
else
  cat > "$VAULT_DIR/synto.toml" <<TOML
[models]
fast = "$FAST_MODEL"
heavy = "$HEAVY_MODEL"

[provider]
name = "$PROVIDER"
url = "$PROVIDER_URL"
timeout = 900
fast_ctx = $FAST_CTX
heavy_ctx = $HEAVY_CTX

[pipeline]
auto_approve = false
auto_commit = false
TOML
fi

# ── helpers ────────────────────────────────────────────────────────────────────
PASS_COUNT=0
pass()   { echo -e "${GREEN}✓${NC} $1"; }
fail()   { echo -e "${RED}✗ FAIL: $1${NC}"; exit 1; }
header() { echo -e "\n${BOLD}$1${NC}"; }
check() {
  local desc="$1"; shift; local rc=0
  ( set +o pipefail; eval "$@" ) >/dev/null 2>&1 || rc=$?
  if [[ $rc -eq 0 ]]; then pass "$desc"; PASS_COUNT=$((PASS_COUNT + 1)); else fail "$desc"; fi
}

# ── Section A: synto add — text file (offline) ────────────────────────────────
header "synto add — text file"
echo "Some source notes." > "$VAULT_DIR/source_note.txt"
ADD_TXT_RC=0; $OLW add "$VAULT_DIR/source_note.txt" 2>&1 || ADD_TXT_RC=$?
check "synto add text exits 0" \
  "test $ADD_TXT_RC -eq 0"
check "source_documents row created" \
  "python3 -c \"import sqlite3; c=sqlite3.connect('$DB'); assert c.execute('SELECT COUNT(*) FROM source_documents').fetchone()[0]==1\""
check "original.txt copied to .synto/sources" \
  "find '$VAULT_DIR/.synto/sources' -name 'original.txt' | grep -q ."

# ── Section B: synto add — PDF import + extraction (offline) ──────────────────
header "synto add — PDF import"
ADD_PDF="$VAULT_DIR/test_source.pdf"
export ADD_PDF
uv run --project "$REPO_DIR" python3 - <<'PYEOF'
import fitz, os
doc = fitz.open()
page = doc.new_page()
page.insert_text((72, 72), "# Introduction\nThis is a test source document for smoke testing.")
doc.save(os.environ["ADD_PDF"])
doc.close()
PYEOF
ADD_PDF_RC=0; $OLW add "$ADD_PDF" --type textbook 2>&1 || ADD_PDF_RC=$?
check "synto add PDF exits 0" \
  "test $ADD_PDF_RC -eq 0"
check "source_type stored as textbook" \
  "python3 -c \"import sqlite3; c=sqlite3.connect('$DB'); rows=c.execute('SELECT source_type FROM source_documents').fetchall(); assert any(r[0]=='textbook' for r in rows)\""
check "source_segments rows created" \
  "python3 -c \"import sqlite3; c=sqlite3.connect('$DB'); assert c.execute('SELECT COUNT(*) FROM source_segments').fetchone()[0]>=1\""
check "original.pdf copied to .synto/sources" \
  "find '$VAULT_DIR/.synto/sources' -name 'original.pdf' | grep -q ."

# ── Section C: duplicate detection (offline) ──────────────────────────────────
header "synto add — duplicate detection"
DUP_RC=0;   $OLW add "$ADD_PDF" --type textbook 2>&1 || DUP_RC=$?
check "second add blocked without --force" \
  "test $DUP_RC -ne 0"
FORCE_RC=0; $OLW add --force "$ADD_PDF" --type textbook 2>&1 || FORCE_RC=$?
check "add --force succeeds" \
  "test $FORCE_RC -eq 0"

# ── Section D: --extend-pack (offline) ────────────────────────────────────────
header "synto add --extend-pack"
$OLW add --force "$VAULT_DIR/source_note.txt" --extend-pack smoke-pack 2>&1 || true
check "[[pack.sources]] written to synto.toml" \
  "grep -q '\[\[pack.sources\]\]' '$VAULT_DIR/synto.toml'"

# ── Section E: source-type prompts — offline load check ───────────────────────
header "source-type prompts — load check"
PROMPT_RC=0
uv run --project "$REPO_DIR" python3 - <<'PYEOF' 2>&1 || PROMPT_RC=$?
from synto.pipeline.prompts import load_prompt
for t in ["notes", "textbook", "paper", "api_docs", "web_article", "corp_docs"]:
    p = load_prompt(t)
    assert len(p) > 50, f"Prompt {t!r} too short ({len(p)} chars)"
PYEOF
check "all 6 source-type prompts load without error" \
  "test $PROMPT_RC -eq 0"

# ── Section F: compile lineage (requires LLM) ─────────────────────────────────
header "compile lineage"
cat > "$VAULT_DIR/raw/smoke_note.md" <<'EOF'
# Machine Learning Basics
Supervised learning uses labelled training data. Unsupervised learning finds
hidden structure without labels. Reinforcement learning trains via reward signals.
EOF

$OLW ingest --all 2>&1
$OLW compile  2>&1
$OLW approve --all 2>&1

check "compile_runs has a finished row" \
  "python3 -c \"import sqlite3; c=sqlite3.connect('$DB'); assert c.execute('SELECT COUNT(*) FROM compile_runs WHERE finished_at IS NOT NULL').fetchone()[0]>=1\""
check "at least one wiki article published" \
  "find '$VAULT_DIR/wiki' -maxdepth 1 -name '*.md' | grep -q ."
check "published article has lineage: frontmatter" \
  "find '$VAULT_DIR/wiki' -maxdepth 1 -name '*.md' -exec grep -l '^lineage:' {} \; | grep -q ."

ARTICLE_FILE=$(find "$VAULT_DIR/wiki" -maxdepth 1 -name '*.md' -exec grep -l '^lineage:' {} \; | head -1)
ARTICLE_NAME=$(basename "$ARTICLE_FILE" .md)
TRACE_OUT_FILE="$(mktemp)"
TRACE_RC=0; $OLW trace article "$ARTICLE_NAME" > "$TRACE_OUT_FILE" 2>&1 || TRACE_RC=$?
check "synto trace article exits 0" \
  "test $TRACE_RC -eq 0"
check "trace output contains a timestamp" \
  "grep -qE '[0-9]{4}-[0-9]{2}-[0-9]{2}' '$TRACE_OUT_FILE'"

# ── Section G: semantic cache — infrastructure + clear (offline) ──────────────
# Note: cache pipeline wiring (client_factory → LLMCache) is incomplete;
# hit_count end-to-end check is deferred until that is wired in.
header "semantic cache — infrastructure"
python3 -c "
import sqlite3
c = sqlite3.connect('$DB')
c.execute(\"INSERT OR REPLACE INTO llm_cache (cache_key, model, response_json, created_at, hit_count) VALUES ('smoke-key', 'test', '{}', datetime('now'), 2)\")
c.commit()
"
check "llm_cache table is writable" \
  "python3 -c \"import sqlite3; c=sqlite3.connect('$DB'); assert c.execute('SELECT COUNT(*) FROM llm_cache').fetchone()[0]>=1\""
CLEAR_RC=0; $OLW maintain --clear-cache 2>&1 || CLEAR_RC=$?
check "synto maintain --clear-cache exits 0" \
  "test $CLEAR_RC -eq 0"
check "llm_cache empty after clear" \
  "python3 -c \"import sqlite3; c=sqlite3.connect('$DB'); assert c.execute('SELECT COUNT(*) FROM llm_cache').fetchone()[0]==0\""

# ── summary ────────────────────────────────────────────────────────────────────
header "Results"
echo -e "${BOLD}All checks passed: $PASS_COUNT${NC}"
echo ""
echo "Vault left at: $VAULT_DIR"
echo "  export SYNTO_VAULT=$VAULT_DIR"
