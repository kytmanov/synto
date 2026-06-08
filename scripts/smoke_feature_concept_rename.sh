#!/usr/bin/env bash
# Smoke test for `synto concept rename` (issue #29): rename a published concept end to end
# and verify the article file moves, its frontmatter title updates, inbound wikilinks
# repoint, and the vault stays consistent. Runs standalone — does not depend on smoke_test.sh.
#
# The published concept's name is LLM-non-deterministic, so the script discovers it from the
# first published article rather than hardcoding it (see CLAUDE.md).
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── colours ───────────────────────────────────────────────────────────────────
if [ -t 1 ]; then
    GREEN='\033[0;32m' RED='\033[0;31m' YELLOW='\033[1;33m' BOLD='\033[1m' NC='\033[0m'
else
    GREEN='' RED='' YELLOW='' BOLD='' NC=''
fi

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

resolve_loaded_model() {
  local model="$1"
  uv run python - <<'PY' "$PROVIDER" "$PROVIDER_URL" "$model"
import re
import sys
import tempfile
from pathlib import Path

from synto.client_factory import build_client
from synto.config import Config

provider, url, model = sys.argv[1:4]


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


with tempfile.TemporaryDirectory(prefix="smoke-model-resolve-") as tmp:
    vault = Path(tmp)
    (vault / "raw").mkdir()
    (vault / "wiki").mkdir()
    (vault / ".synto").mkdir()
    (vault / "synto.toml").write_text(
        f"[models]\nfast = \"{model}\"\nheavy = \"{model}\"\n\n"
        f"[provider]\nname = \"{provider}\"\nurl = \"{url}\"\n",
        encoding="utf-8",
    )
    cfg = Config.from_vault(vault)
    client = build_client(cfg)
    try:
        client.require_healthy()
        models = client.list_models()
    finally:
        try:
            client.close()
        except Exception:
            pass

if model in models:
    print(model)
    raise SystemExit(0)

wanted = norm(model)
matches = [m for m in models if wanted == norm(m) or wanted in norm(m) or norm(m) in wanted]
if len(matches) == 1:
    print(matches[0])
    raise SystemExit(0)

raise SystemExit(
    f"Model {model!r} is not loaded in {provider} at {url}. Available: {models}"
)
PY
}

if [[ "$PROVIDER" != "ollama" ]]; then
  FAST_MODEL="$(resolve_loaded_model "$FAST_MODEL")"
  HEAVY_MODEL="$(resolve_loaded_model "$HEAVY_MODEL")"
fi

# ── vault + config ─────────────────────────────────────────────────────────────
VAULT_DIR="$(mktemp -d)"
export SYNTO_VAULT="$VAULT_DIR"
OLW="uv run --project $REPO_DIR synto"
mkdir -p "$VAULT_DIR/raw"

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
_RESULTS=()
_T0=$(date +%s)
PASS_COUNT=0

pass()   { echo -e "${GREEN}✓${NC} $1"; }
fail() {
    local desc="$1" detail="${2:-}"
    echo -e "  ${RED}✗ FAIL: $desc${NC}${detail:+$'\n'    ${detail:0:1000}}"
    echo -e "  ${YELLOW}▶ Vault left at: $VAULT_DIR${NC}"
    exit 1
}
header() {
    if [[ -n "${_SECTION_START:-}" ]]; then
        echo -e "  ${YELLOW}($(( SECONDS - _SECTION_START ))s)${NC}"
    fi
    _SECTION_START=$SECONDS
    echo -e "\n${BOLD}$1${NC}"
}

check() {
    local desc="$1"; shift
    local out rc=0
    out=$(set +o pipefail; eval "$@" 2>&1) || rc=$?
    if [[ $rc -eq 0 ]]; then
        pass "$desc"; PASS_COUNT=$((PASS_COUNT + 1))
        _RESULTS+=("PASS|$desc|")
    else
        _RESULTS+=("FAIL|$desc|${out:0:1000}")
        fail "$desc" "$out"
    fi
}

_write_report() {
    [[ -z "${REPORT_FILE:-}" ]] && return
    local passed=0 failed=0 elapsed r status label detail
    elapsed=$(( $(date +%s) - _T0 ))
    declare -a checks=()
    for r in "${_RESULTS[@]}"; do
        IFS='|' read -r status label detail <<< "$r"
        if [[ "$status" == "PASS" ]]; then
            ((passed++))
            checks+=("{\"suite\":\"\",\"passed\":true,\"name\":$(printf '%s' "$label" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))'),\"detail\":null}")
        else
            ((failed++))
            checks+=("{\"suite\":\"\",\"passed\":false,\"name\":$(printf '%s' "$label" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))'),\"detail\":$(printf '%s' "$detail" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')}")
        fi
    done
    printf '{"passed":%d,"failed":%d,"duration_s":%d,"checks":[%s]}\n' \
        "$passed" "$failed" "$elapsed" "$(IFS=,; printf '%s' "${checks[*]}")" \
        > "$REPORT_FILE"
}
trap _write_report EXIT

# ── Section A: build a published article (requires LLM) ────────────────────────
header "ingest + compile + approve"
cat > "$VAULT_DIR/raw/smoke_note.md" <<'EOF'
# Machine Learning Basics
Supervised learning uses labelled training data. Unsupervised learning finds
hidden structure without labels. Reinforcement learning trains via reward signals.
EOF

$OLW ingest --all 2>&1
$OLW compile 2>&1
$OLW approve --all 2>&1

check "at least one wiki article published" \
  "find '$VAULT_DIR/wiki' -maxdepth 1 -name '*.md' ! -name 'index.md' ! -name 'log.md' | grep -q ."

# ── Section B: discover the concept + plant an inbound link ────────────────────
header "concept rename"
ARTICLE_FILE=$(find "$VAULT_DIR/wiki" -maxdepth 1 -name '*.md' ! -name 'index.md' ! -name 'log.md' | head -1)
OLD_NAME=$(uv run --project "$REPO_DIR" python3 - "$ARTICLE_FILE" <<'PY'
import sys

import frontmatter

print(frontmatter.load(sys.argv[1]).get("title"))
PY
)
check "resolved a non-empty concept title" \
  "test -n \"$OLD_NAME\""

# A second published article that links to the concept by its canonical name.
cat > "$VAULT_DIR/wiki/Smoke Linker.md" <<EOF
---
title: Smoke Linker
status: published
tags: []
---

This page references [[$OLD_NAME]] so the rename must repoint it.
EOF

NEW_NAME="Renamed Smoke Concept"
RENAME_RC=0; $OLW concept rename "$OLD_NAME" "$NEW_NAME" --drop-old-alias 2>&1 || RENAME_RC=$?
check "concept rename exits 0" \
  "test $RENAME_RC -eq 0"

# ── Section C: assert the rename took effect ───────────────────────────────────
header "post-rename invariants"
check "renamed article file exists" \
  "test -f '$VAULT_DIR/wiki/$NEW_NAME.md'"
check "old article file is gone" \
  "test ! -f '$ARTICLE_FILE'"
check "renamed article frontmatter title updated" \
  "grep -q \"^title: ${NEW_NAME}\$\" '$VAULT_DIR/wiki/$NEW_NAME.md'"
check "inbound wikilink repointed to new name" \
  "grep -q '\[\[${NEW_NAME}\]\]' '$VAULT_DIR/wiki/Smoke Linker.md'"
check "old name no longer linked anywhere in wiki" \
  "! grep -rq '\[\[${OLD_NAME}\]\]' '$VAULT_DIR/wiki'"
check "synto status exits 0 after rename" \
  "$OLW status"

# ── summary ────────────────────────────────────────────────────────────────────
header "Results"
echo -e "${BOLD}All checks passed: $PASS_COUNT${NC}"
echo ""
echo "Vault left at: $VAULT_DIR"
echo "  export SYNTO_VAULT=$VAULT_DIR"
