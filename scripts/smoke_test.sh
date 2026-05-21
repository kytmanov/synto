#!/usr/bin/env bash
# smoke_test.sh — end-to-end test against a real LLM backend
#
# Supports Ollama (default) and LM Studio (PROVIDER=lm_studio).
#
# Usage:
#   ./scripts/smoke_test.sh                              # Ollama, default models
#   PROVIDER=lm_studio ./scripts/smoke_test.sh           # LM Studio
#   PROVIDER=lm_studio FAST_MODEL=google/gemma-4-e4b ./scripts/smoke_test.sh
#   FAST_MODEL=llama3.2:latest ./scripts/smoke_test.sh   # Ollama, custom model
#   VAULT_DIR=/tmp/my-vault ./scripts/smoke_test.sh      # keep vault after run
#   SKIP_PULL=1 ./scripts/smoke_test.sh                  # skip ollama pull
#
# Requirements:
#   - uv (https://docs.astral.sh/uv/)
#   - Ollama running (ollama serve)  — OR —  LM Studio running with a model loaded

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROVIDER="${PROVIDER:-ollama}"

case "$PROVIDER" in
    ollama)
        # OLLAMA_URL kept for backward compatibility
        PROVIDER_URL="${PROVIDER_URL:-${OLLAMA_URL:-http://localhost:11434}}"
        FAST_MODEL="${FAST_MODEL:-gemma4:e4b}"
        HEAVY_MODEL="${HEAVY_MODEL:-gemma4:e4b}"
        FAST_CTX=8192
        HEAVY_CTX=16384
        ;;
    lm_studio)
        PROVIDER_URL="${PROVIDER_URL:-http://localhost:1234/v1}"
        FAST_MODEL="${FAST_MODEL:-google/gemma-4-e4b}"
        HEAVY_MODEL="${HEAVY_MODEL:-google/gemma-4-e4b}"
        FAST_CTX=8192
        # Keep output budget + input within 8192: source uses heavy_ctx//2 tokens,
        # output uses _MAX_ARTICLE_PREDICT=4096. Total = 4096+4096+~800 overhead > 8192.
        # Use 8192 so _gather_sources truncates aggressively enough for short test notes.
        HEAVY_CTX=8192
        ;;
    *)
        # Generic OpenAI-compatible provider — caller must set PROVIDER_URL and models
        PROVIDER_URL="${PROVIDER_URL:-}"
        FAST_MODEL="${FAST_MODEL:-}"
        HEAVY_MODEL="${HEAVY_MODEL:-}"
        FAST_CTX="${FAST_CTX:-8192}"
        HEAVY_CTX="${HEAVY_CTX:-16384}"
        HEAVY_MODEL="${HEAVY_MODEL:-$FAST_MODEL}"
        if [[ -z "$PROVIDER_URL" || -z "$FAST_MODEL" || -z "$HEAVY_MODEL" ]]; then
            echo "ERROR: PROVIDER=$PROVIDER requires PROVIDER_URL and FAST_MODEL to be set."
            exit 1
        fi
        ;;
esac

SKIP_PULL="${SKIP_PULL:-0}"
KEEP_VAULT="${KEEP_VAULT:-0}"
INLINE_SOURCE_CITATIONS="${INLINE_SOURCE_CITATIONS:-0}"
if [[ "$INLINE_SOURCE_CITATIONS" == "1" || "$INLINE_SOURCE_CITATIONS" == "true" ]]; then
    INLINE_SOURCE_CITATIONS_TOML="true"
else
    INLINE_SOURCE_CITATIONS_TOML="false"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Use provided VAULT_DIR or create a temp one
if [[ -n "${VAULT_DIR:-}" ]]; then
    KEEP_VAULT=1
    mkdir -p "$VAULT_DIR"
else
    VAULT_DIR="$(mktemp -d)"
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
if [ -t 1 ]; then
    GREEN='\033[0;32m' RED='\033[0;31m' YELLOW='\033[1;33m' BOLD='\033[1m' NC='\033[0m'
else
    GREEN='' RED='' YELLOW='' BOLD='' NC=''
fi

_RESULTS=()
_T0=$(date +%s)

pass() { echo -e "${GREEN}✓${NC} $1"; }
fail() {
    local desc="$1" detail="${2:-}"
    echo -e "  ${RED}✗ FAIL: $desc${NC}${detail:+$'\n'    ${detail:0:1000}}"
    echo -e "  ${YELLOW}▶ Re-run with VAULT_DIR=/tmp/keep-vault to inspect state after failure${NC}"
    exit 1
}
info() { echo -e "${YELLOW}▶${NC} $1"; }
header() {
    if [[ -n "${_SECTION_START:-}" ]]; then
        echo -e "  ${YELLOW}($(( SECONDS - _SECTION_START ))s)${NC}"
    fi
    _SECTION_START=$SECONDS
    echo -e "\n${BOLD}$1${NC}"
}

PASS_COUNT=0
check() {
    local desc="$1"
    shift
    local out rc=0
    out=$(set +o pipefail; eval "$@" 2>&1) || rc=$?
    if [[ $rc -eq 0 ]]; then
        pass "$desc"
        PASS_COUNT=$((PASS_COUNT + 1))
        _RESULTS+=("PASS|$desc|")
    else
        _RESULTS+=("FAIL|$desc|${out:0:1000}")
        fail "$desc" "$out"
    fi
}

soft_check() {
    local desc="$1"
    shift
    local out rc=0
    out=$(set +o pipefail; eval "$@" 2>&1) || rc=$?
    if [[ $rc -eq 0 ]]; then
        pass "$desc"
        PASS_COUNT=$((PASS_COUNT + 1))
        _RESULTS+=("PASS|$desc|")
    else
        echo -e "  ${RED}✗ SOFT FAIL:${NC} $desc${out:+ — ${out:0:1000}}"
        _RESULTS+=("FAIL|$desc|${out:0:1000}")
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

cleanup() {
    _write_report
    if [[ "$KEEP_VAULT" == "0" ]]; then
        rm -rf "$VAULT_DIR"
        rm -rf "${UNDO_VAULT_DIR:-}"
    else
        echo -e "\nVault kept at: ${BOLD}$VAULT_DIR${NC}"
        if [[ -n "${UNDO_VAULT_DIR:-}" ]]; then
            echo -e "Undo test vault kept at: ${BOLD}$UNDO_VAULT_DIR${NC}"
        fi
    fi
}
trap cleanup EXIT

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

# ── Prerequisites ─────────────────────────────────────────────────────────────
header "Prerequisites (provider: $PROVIDER)"

check "uv available" "command -v uv"

if [[ "$PROVIDER" == "ollama" ]]; then
    check "Ollama reachable at $PROVIDER_URL" "curl -sf $PROVIDER_URL/api/tags"

    if [[ "$SKIP_PULL" == "0" ]]; then
        info "Pulling models (skippable with SKIP_PULL=1)"
        ollama pull "$FAST_MODEL"  || fail "Could not pull $FAST_MODEL"
        if [[ "$FAST_MODEL" != "$HEAVY_MODEL" ]]; then
            ollama pull "$HEAVY_MODEL" || fail "Could not pull $HEAVY_MODEL"
        fi
    fi

    soft_check "Fast model present: $FAST_MODEL"  "curl -sf $PROVIDER_URL/api/tags | grep -F -q '$FAST_MODEL'"
    soft_check "Heavy model present: $HEAVY_MODEL" "curl -sf $PROVIDER_URL/api/tags | grep -F -q '$HEAVY_MODEL'"
else
    # LM Studio and other OpenAI-compatible providers: just verify the endpoint is up.
    # Model presence can't be checked reliably via /v1/models on all backends.
    check "$PROVIDER reachable at $PROVIDER_URL" "curl -sf $PROVIDER_URL/models"
    info "Model pull skipped ($PROVIDER manages its own models — load $FAST_MODEL in $PROVIDER before running)"
    FAST_MODEL="$(resolve_loaded_model "$FAST_MODEL")"
    HEAVY_MODEL="$(resolve_loaded_model "$HEAVY_MODEL")"
    pass "Fast model resolved: $FAST_MODEL"
    pass "Heavy model resolved: $HEAVY_MODEL"
    PASS_COUNT=$((PASS_COUNT + 2))
fi

# ── Install ───────────────────────────────────────────────────────────────────
header "Install"

info "Installing synto from $REPO_DIR"
uv sync --project "$REPO_DIR" --quiet
pass "uv sync"

OLW="uv run --project $REPO_DIR synto"
export SYNTO_VAULT="$VAULT_DIR"

# ── Structured output resilience (PR #32 + _make_template recursion) ──────────
# Pure-Python regression guard — runs without any LLM. Verifies:
#   1. AnalysisResult.coerce_concepts wraps list[str] → list[Concept] (PR #32)
#   2. Mixed list[str | dict] still validates
#   3. _make_template renders list[Concept] as nested object example, not the
#      array description string (root cause behind PR #32)
#   4. request_structured end-to-end handles a string-concept LLM response
header "Structured output resilience"

_SO_SCRIPT=$(mktemp /tmp/synto_so_smoke.XXXXXX)
cat > "$_SO_SCRIPT" <<'PYEOF'
import json
from unittest.mock import MagicMock

from synto.models import AnalysisResult
from synto.structured_output import _make_template, request_structured

r = AnalysisResult(
    summary="s",
    concepts=["Foo", "Bar"],
    suggested_topics=[],
    quality="high",
    language=None,
)
assert [c.name for c in r.concepts] == ["Foo", "Bar"]
assert all(c.aliases == [] for c in r.concepts)

r = AnalysisResult(
    summary="s",
    concepts=[{"name": "A", "aliases": ["a"]}, "B"],
    suggested_topics=[],
    quality="high",
    language=None,
)
assert r.concepts[0].aliases == ["a"]
assert r.concepts[1].name == "B" and r.concepts[1].aliases == []

tpl = json.loads(_make_template(AnalysisResult))
assert isinstance(tpl["concepts"][0], dict), tpl["concepts"]
assert set(tpl["concepts"][0]) == {"name", "aliases"}

fake = json.dumps({
    "summary": "s",
    "concepts": ["Alpha", "Beta"],
    "suggested_topics": [],
    "quality": "high",
})
client = MagicMock()
client.generate.return_value = fake
parsed = request_structured(
    client=client,
    prompt="x",
    model_class=AnalysisResult,
    model="fake",
    max_retries=0,
)
assert [c.name for c in parsed.concepts] == ["Alpha", "Beta"]
print("ok")
PYEOF

_SO_OUT=$(uv run --project "$REPO_DIR" python "$_SO_SCRIPT" 2>&1); _SO_RC=$?
rm -f "$_SO_SCRIPT"
if [[ $_SO_RC -ne 0 ]]; then echo "$_SO_OUT"; fi
check "string-concept list coerced end-to-end (PR #32 + template fix)" \
    "test $_SO_RC -eq 0"

# ── Init ──────────────────────────────────────────────────────────────────────
header "synto init"

$OLW init "$VAULT_DIR" 2>&1 | grep -v "^$" || true

soft_check "raw/ created"           "test -d $VAULT_DIR/raw"
soft_check "wiki/ created"          "test -d $VAULT_DIR/wiki"
soft_check "wiki/.drafts/ created"  "test -d $VAULT_DIR/wiki/.drafts"
soft_check "wiki/sources/ created"  "test -d $VAULT_DIR/wiki/sources"
soft_check ".synto/ created"          "test -d $VAULT_DIR/.synto"
soft_check "synto.toml created"      "test -f $VAULT_DIR/synto.toml"
soft_check "git repo initialised"   "test -d $VAULT_DIR/.git"
# #27: init must write wiki/index.md with lowercase name.
# Can't use `test -f INDEX.md` on macOS APFS (case-insensitive) — it matches index.md.
# Instead, check the actual on-disk filename via ls (ls preserves stored casing).
_ACTUAL_INDEX=$(ls "$VAULT_DIR/wiki/" | { grep -i '^index\.md$' || true; } | head -1)
soft_check "wiki index file is lowercase index.md (issue #27)" "test '$_ACTUAL_INDEX' = 'index.md'"

# Write provider-appropriate synto.toml
if [[ "$PROVIDER" == "ollama" ]]; then
    cat > "$VAULT_DIR/synto.toml" <<TOML
[models]
fast = "$FAST_MODEL"
heavy = "$HEAVY_MODEL"
embed = "nomic-embed-text"

[ollama]
url = "$PROVIDER_URL"
timeout = 900
fast_ctx = $FAST_CTX
heavy_ctx = $HEAVY_CTX

[pipeline]
auto_approve = false
auto_commit = true
watch_debounce = 3.0
inline_source_citations = ${INLINE_SOURCE_CITATIONS_TOML}

[rag]
chunk_size = 512
chunk_overlap = 50
similarity_threshold = 0.7
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
auto_commit = true
watch_debounce = 3.0
inline_source_citations = ${INLINE_SOURCE_CITATIONS_TOML}

[rag]
chunk_size = 512
chunk_overlap = 50
similarity_threshold = 0.7
TOML
fi
pass "synto.toml configured (provider=$PROVIDER fast=$FAST_MODEL heavy=$HEAVY_MODEL)"

# ── Doctor ───────────────────────────────────────────────────────────────────
header "synto doctor"
$OLW doctor 2>&1 || true
# Doctor exit code not checked (models may not be present before pull)

# ── Seed raw notes ────────────────────────────────────────────────────────────
header "Seed raw notes"

cat > "$VAULT_DIR/raw/quantum-computing.md" <<'EOF'
---
title: Quantum Computing Fundamentals
source: https://example.com/quantum
---

Quantum computers use qubits instead of classical bits. Unlike bits which are
either 0 or 1, qubits exploit superposition to be in multiple states simultaneously.

Entanglement links qubits: measuring one instantly determines the state of its
partner regardless of distance. This enables quantum parallelism.

Key algorithms:
- Shor's algorithm: factors large integers exponentially faster than classical
- Grover's algorithm: searches unsorted databases with quadratic speedup
- Quantum Fourier Transform: underpins most quantum speedups

Hardware approaches: superconducting qubits (IBM, Google), trapped ions (IonQ),
photonic (PsiQuantum), topological (Microsoft).

Current state (2024): NISQ era — noisy, ~1000 qubits, error rates ~0.1%.
Fault-tolerant quantum computing requires ~1M physical qubits per logical qubit.
EOF

cat > "$VAULT_DIR/raw/machine-learning-basics.md" <<'EOF'
---
title: Machine Learning Fundamentals
---

Machine learning enables computers to learn from data without being explicitly
programmed. Three main paradigms:

Supervised learning: labeled training data. Examples: classification (spam
detection), regression (price prediction). Algorithms: linear regression,
decision trees, neural networks, SVMs.

Unsupervised learning: finds hidden structure in unlabeled data. Clustering
(k-means), dimensionality reduction (PCA), generative models.

Reinforcement learning: agent learns by interacting with environment, maximising
cumulative reward. Used in game playing (AlphaGo), robotics, recommendation systems.

Deep learning: neural networks with many layers. Excels at images (CNNs), text
(Transformers), audio. Requires large datasets and compute.

Key concepts: gradient descent, backpropagation, overfitting/underfitting,
train/val/test split, cross-validation.
EOF

soft_check "raw note 1 created" "test -f $VAULT_DIR/raw/quantum-computing.md"
soft_check "raw note 2 created" "test -f $VAULT_DIR/raw/machine-learning-basics.md"

# Snapshot checksums so we can verify raw files stay immutable after ingest
RAW_HASH_1=$(shasum "$VAULT_DIR/raw/quantum-computing.md" | awk '{print $1}')
RAW_HASH_2=$(shasum "$VAULT_DIR/raw/machine-learning-basics.md" | awk '{print $1}')

# ── Ingest ────────────────────────────────────────────────────────────────────
header "synto ingest --all"
info "Calling $PROVIDER ($FAST_MODEL) — may take 30-120s..."

$OLW ingest --all 2>&1

check "state.db created" "test -f $VAULT_DIR/.synto/state.db"

# Raw files must remain unchanged (immutability contract)
soft_check "raw note 1 unchanged after ingest" \
    "test \"\$(shasum '$VAULT_DIR/raw/quantum-computing.md' | awk '{print \$1}')\" = '$RAW_HASH_1'"
soft_check "raw note 2 unchanged after ingest" \
    "test \"\$(shasum '$VAULT_DIR/raw/machine-learning-basics.md' | awk '{print \$1}')\" = '$RAW_HASH_2'"

# Source summary pages created in wiki/sources/
SOURCE_COUNT=$(find "$VAULT_DIR/wiki/sources" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
soft_check "source summary pages created" "test '$SOURCE_COUNT' -ge 1"

if [[ "$SOURCE_COUNT" -gt 0 ]]; then
    FIRST_SOURCE=$(find "$VAULT_DIR/wiki/sources" -name "*.md" | sort | head -1)
    soft_check "source page has YAML frontmatter"  "grep -q '^---' \"$FIRST_SOURCE\""
    soft_check "source page has tags: [source]"    "grep -q 'source' \"$FIRST_SOURCE\""
    soft_check "source page has concept wikilinks" "grep -q '\[\[' \"$FIRST_SOURCE\""

    SRC_YAML_ERR=$(uv run --project "$REPO_DIR" python - "$FIRST_SOURCE" 2>/dev/null <<'PYEOF'
import sys
try:
    import frontmatter
    frontmatter.load(sys.argv[1])
except Exception as e:
    print(f"error: {e}")
    sys.exit(1)
PYEOF
)
    soft_check "source page YAML is parseable" "test -z \"$SRC_YAML_ERR\""

    SRC_ALIAS_ERR=$(uv run --project "$REPO_DIR" python - "$FIRST_SOURCE" 2>/dev/null <<'PYEOF'
import sys
try:
    import frontmatter
    m = frontmatter.load(sys.argv[1])
    aliases = m.get('aliases', [])
    assert isinstance(aliases, list), f'aliases not a list: {aliases!r}'
except AssertionError as e:
    print(str(e))
    sys.exit(1)
except Exception as e:
    print(f"error: {e}")
    sys.exit(1)
PYEOF
)
    soft_check "source page aliases is a list" "test -z \"$SRC_ALIAS_ERR\""
fi

# index.md and log.md created
soft_check "wiki/index.md created" "test -f $VAULT_DIR/wiki/index.md"
soft_check "wiki/log.md created"   "test -f $VAULT_DIR/wiki/log.md"
soft_check "index.md has wikilinks" "grep -q '\[\[' $VAULT_DIR/wiki/index.md"

# ── Status after ingest ───────────────────────────────────────────────────────
header "synto status (after ingest)"
STATUS_OUT=$($OLW status 2>&1)
echo "$STATUS_OUT"

soft_check "status shows ingested notes" "echo \"$STATUS_OUT\" | grep -q 'ingested'"

# ── Concept extraction check ──────────────────────────────────────────────────
header "Concept extraction"
# Source summary pages should have wikilinks pointing to extracted concepts
if [[ "$SOURCE_COUNT" -gt 0 ]]; then
    # Verify concept wikilinks exist in source pages (extracted during ingest)
    CONCEPT_LINKS=$(grep -r '\[\[' "$VAULT_DIR/wiki/sources/" 2>/dev/null | wc -l | tr -d ' ')
    soft_check "source pages have concept wikilinks" "test '$CONCEPT_LINKS' -ge 1"
fi

# #28: concept_aliases table should be populated after ingest
ALIAS_COUNT=$(python3 - <<PYEOF
import sqlite3
conn = sqlite3.connect("$VAULT_DIR/.synto/state.db")
try:
    n = conn.execute("SELECT COUNT(*) FROM concept_aliases").fetchone()[0]
    print(n)
except Exception:
    print(0)
conn.close()
PYEOF
)
soft_check "concept_aliases table populated after ingest (issue #28)" \
    "test '$ALIAS_COUNT' -gt 0"
info "Aliases stored in DB: $ALIAS_COUNT"

# ── Language detection check ──────────────────────────────────────────────────
header "Language detection (ingest)"

cat > "$VAULT_DIR/raw/note-francais.md" <<'EOF'
---
title: Apprentissage automatique
---

L'apprentissage automatique est une branche de l'intelligence artificielle.
Les algorithmes apprennent à partir des données sans être explicitement programmés.

Les principales approches sont l'apprentissage supervisé, non supervisé et par renforcement.
EOF

$OLW ingest "$VAULT_DIR/raw/note-francais.md" 2>&1

LANG_IN_DB=$(python3 - <<PYEOF
import sqlite3
conn = sqlite3.connect("$VAULT_DIR/.synto/state.db")
row = conn.execute("SELECT language FROM raw_notes WHERE path='raw/note-francais.md'").fetchone()
print(row[0] if row else "")
conn.close()
PYEOF
)
soft_check "language column populated after ingest" "test -n \"$LANG_IN_DB\""
info "Detected language: '$LANG_IN_DB'"

# ── Compile (concept-driven) ──────────────────────────────────────────────────
header "synto compile (concept-driven)"
info "Calling $PROVIDER ($HEAVY_MODEL) — may take 2-5 min..."

$OLW compile 2>&1

DRAFT_COUNT=$(find "$VAULT_DIR/wiki/.drafts" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
check "at least 1 draft created" "test '$DRAFT_COUNT' -ge 1"

if [[ "$DRAFT_COUNT" -gt 0 ]]; then
    FIRST_DRAFT=$(find "$VAULT_DIR/wiki/.drafts" -name "*.md" | sort | head -1)
    soft_check "draft has YAML frontmatter"   "grep -q '^---' \"$FIRST_DRAFT\""
    soft_check "draft has title field"        "grep -q 'title:' \"$FIRST_DRAFT\""
    soft_check "draft has status: draft"      "grep -q 'status: draft' \"$FIRST_DRAFT\""
    soft_check "draft has sources field"      "grep -q 'sources:' \"$FIRST_DRAFT\""
    soft_check "draft has content"            "test \$(wc -l < \"$FIRST_DRAFT\") -ge 10"
    soft_check "draft has ## Sources section" "grep -q '^## Sources' \"$FIRST_DRAFT\""
    soft_check "draft has confidence field"   "grep -q 'confidence:' \"$FIRST_DRAFT\""
    if [[ "$INLINE_SOURCE_CITATIONS_TOML" == "true" ]]; then
        soft_check "draft source legend has citation ids" "grep -q '^- \[S[0-9]\+\] \[\[sources/' \"$FIRST_DRAFT\""
        if grep -q '\[\[sources/.*|S[0-9]' "$FIRST_DRAFT"; then
            pass "draft has inline source citation link"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            info "No inline source citation link found in first draft; model may have omitted markers."
        fi
    fi
    DRAFT_YAML_OK=$(uv run --project "$REPO_DIR" python - "$FIRST_DRAFT" 2>/dev/null <<'PYEOF'
import sys
try:
    import frontmatter
    frontmatter.load(sys.argv[1])
except Exception as e:
    print(f"error: {e}")
    sys.exit(1)
PYEOF
)
    soft_check "draft YAML is parseable" "test -z \"$DRAFT_YAML_OK\""
    DRAFT_TAG_BAD=$(uv run --project "$REPO_DIR" python - "$FIRST_DRAFT" 2>/dev/null <<'PYEOF'
import sys
try:
    import re, frontmatter
    m = frontmatter.load(sys.argv[1])
    valid_re = re.compile(r'^[a-z0-9][a-zA-Z0-9_/\-]*$')
    bad = [t for t in m.get('tags', []) if not isinstance(t, str) or ' ' in t or t != t.lower() or not valid_re.match(t)]
    if bad:
        print(f"Bad tags: {bad}")
        sys.exit(1)
except Exception as e:
    print(f"error: {e}")
    sys.exit(1)
PYEOF
)
    soft_check "draft tags are valid (lowercase, no spaces, no special chars)" "test -z \"$DRAFT_TAG_BAD\""
fi

# ── Status after compile ──────────────────────────────────────────────────────
header "synto status (after compile)"
$OLW status 2>&1

# ── Approve ───────────────────────────────────────────────────────────────────
header "synto approve --all"
$OLW approve --all 2>&1

WIKI_COUNT=$(find "$VAULT_DIR/wiki" -name "*.md" -not -path "*/.drafts/*" 2>/dev/null | wc -l | tr -d ' ')
check "articles published to wiki/"    "test '$WIKI_COUNT' -ge 1"
check "drafts directory now empty"     "test \$(find $VAULT_DIR/wiki/.drafts -name '*.md' 2>/dev/null | wc -l) -eq 0"
soft_check "git commit created"             "git -C $VAULT_DIR log --oneline | grep -q '\[synto\]'"

# Bulk YAML validity + tag check on all published wiki pages
# Use -print0 / read -d '' to handle spaces and special chars in filenames
header "YAML validity of published pages"
YAML_FAIL=0
TAG_FAIL=0

# Write validator to temp file (avoids heredoc-inside-process-substitution bash quirk)
_YAML_VALIDATOR=$(mktemp /tmp/olw_yaml_check.XXXXXX)
cat > "$_YAML_VALIDATOR" << 'PYEOF'
import sys, re, frontmatter
try:
    m = frontmatter.load(sys.argv[1])
except Exception as e:
    print(f"  YAML parse failed: {sys.argv[1]}: {e}")
    sys.exit(1)
tags = m.get('tags', [])
valid_re = re.compile(r'^[a-z0-9][a-zA-Z0-9_/\-]*$')
bad = [t for t in tags if not isinstance(t, str) or ' ' in t or t != t.lower() or not valid_re.match(t)]
if bad:
    print(f"  Bad tags in {sys.argv[1]}: {bad}")
    sys.exit(2)
PYEOF

while IFS= read -r -d '' md; do
    result=$(uv run --project "$REPO_DIR" python "$_YAML_VALIDATOR" "$md" 2>&1)
    exit_code=$?
    if [ $exit_code -eq 1 ]; then
        echo "$result"
        YAML_FAIL=1
    elif [ $exit_code -eq 2 ]; then
        echo "$result"
        TAG_FAIL=1
    fi
done < <(find "$VAULT_DIR/wiki" -name "*.md" -not -path "*/.drafts/*" -print0 2>/dev/null)
rm -f "$_YAML_VALIDATOR"

soft_check "all published pages have valid YAML" "test $YAML_FAIL -eq 0"
soft_check "no published pages have invalid tags (spaces/uppercase/special)" "test $TAG_FAIL -eq 0"

# Empty wikilinks [[]] in published articles indicate a model output bug
EMPTY_WIKILINK_FILES=$({ grep -rl '\[\[\]\]' "$VAULT_DIR/wiki/" \
    --include='*.md' --exclude-dir='.drafts' --exclude-dir='sources' \
    --exclude-dir='queries' 2>/dev/null || true; } | wc -l | tr -d ' ')
soft_check "no published article contains empty [[]] wikilinks" \
    "test '$EMPTY_WIKILINK_FILES' -eq 0"

# #28: at least one published article must carry an aliases: field in frontmatter.
# Aliases table populated is already asserted above; this asserts compile actually
# propagated them into article frontmatter (the user-visible contract).
ARTICLES_WITH_ALIASES=$({ grep -rl '^aliases:' "$VAULT_DIR/wiki/" \
    --include='*.md' --exclude-dir='.drafts' --exclude-dir='sources' \
    --exclude-dir='queries' 2>/dev/null || true; } | wc -l | tr -d ' ')
info "Articles with aliases frontmatter: $ARTICLES_WITH_ALIASES"
soft_check "at least one published article has aliases: frontmatter (issue #28)" \
    "test '$ARTICLES_WITH_ALIASES' -ge 1"

# ── Maintain issue-type coverage (corrupt → assert → restore) ─────────────────
# Exercises health-check code paths that would otherwise only fire in the wild:
# inline_tag, missing_frontmatter, invalid_tag, stale, low_confidence.
# Vault is clean at this point (post-approve, pre-undo).
header "Maintain issue-type coverage"

_VICTIM=$(find "$VAULT_DIR/wiki" -maxdepth 1 -name '*.md' \
    ! -name 'index.md' ! -name 'log.md' 2>/dev/null | head -1)

if [[ -n "$_VICTIM" ]]; then
    _VICTIM_BACKUP=$(mktemp)
    cp "$_VICTIM" "$_VICTIM_BACKUP"

    # inline_tag — regex-scanned in body
    echo "" >> "$_VICTIM"
    echo "#smoke-inline-tag" >> "$_VICTIM"
    _LT_RC=0; LT_OUT=$($OLW maintain --dry-run 2>&1) || _LT_RC=$?
    check "maintain exits 0 after inline_tag corruption" "test $_LT_RC -eq 0"
    soft_check "maintain detects inline_tag" "echo \"\$LT_OUT\" | grep -qE 'inline_tag'"
    cp "$_VICTIM_BACKUP" "$_VICTIM"

    # missing_frontmatter — rewrite victim with only body, no frontmatter
    python3 - "$_VICTIM" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, 'w') as f:
    f.write("Plain body, no frontmatter.\n")
PYEOF
    _LT_RC=0; LT_OUT=$($OLW maintain --dry-run 2>&1) || _LT_RC=$?
    check "maintain exits 0 after missing_frontmatter corruption" "test $_LT_RC -eq 0"
    soft_check "maintain detects missing_frontmatter" \
        "echo \"\$LT_OUT\" | grep -qE 'missing_frontmatter'"
    cp "$_VICTIM_BACKUP" "$_VICTIM"

    # invalid_tag — inject uppercase/spaced tag into frontmatter.
    # Use uv run python because system python3 has no 'frontmatter' module.
    uv run --project "$REPO_DIR" python - "$_VICTIM" <<'PYEOF'
import sys, frontmatter
path = sys.argv[1]
m = frontmatter.load(path)
m['tags'] = ['Bad Tag', 'UPPER']
with open(path, 'w') as f:
    f.write(frontmatter.dumps(m))
PYEOF
    _LT_RC=0; LT_OUT=$($OLW maintain --dry-run 2>&1) || _LT_RC=$?
    check "maintain exits 0 after invalid_tag corruption" "test $_LT_RC -eq 0"
    soft_check "maintain detects invalid_tag" "echo \"\$LT_OUT\" | grep -qE 'invalid_tag'"
    cp "$_VICTIM_BACKUP" "$_VICTIM"

    # low_confidence — set confidence below threshold (0.3, strict <)
    uv run --project "$REPO_DIR" python - "$_VICTIM" <<'PYEOF'
import sys, frontmatter
path = sys.argv[1]
m = frontmatter.load(path)
m['confidence'] = 0.1
with open(path, 'w') as f:
    f.write(frontmatter.dumps(m))
PYEOF
    _LT_RC=0; LT_OUT=$($OLW maintain --dry-run 2>&1) || _LT_RC=$?
    check "maintain exits 0 after low_confidence corruption" "test $_LT_RC -eq 0"
    soft_check "maintain detects low_confidence" \
        "echo \"\$LT_OUT\" | grep -qE 'low_confidence'"
    cp "$_VICTIM_BACKUP" "$_VICTIM"

    # stale — append text without recompile so body hash diverges from DB record
    echo "" >> "$_VICTIM"
    echo "Untracked manual edit to trigger stale detection." >> "$_VICTIM"
    _LT_RC=0; LT_OUT=$($OLW maintain --dry-run 2>&1) || _LT_RC=$?
    check "maintain exits 0 after stale corruption" "test $_LT_RC -eq 0"
    soft_check "maintain detects stale" "echo \"\$LT_OUT\" | grep -qE 'stale'"
    cp "$_VICTIM_BACKUP" "$_VICTIM"

    rm -f "$_VICTIM_BACKUP"
else
    info "Lint coverage block skipped (no published article available)"
fi

# ── Git log ───────────────────────────────────────────────────────────────────
header "Git history"
git -C "$VAULT_DIR" log --oneline

# ── Undo ─────────────────────────────────────────────────────────────────────
header "synto undo"

UNDO_VAULT_DIR="$(mktemp -d)"
rsync -a "$VAULT_DIR/" "$UNDO_VAULT_DIR/"

UNDO_OUT=$($OLW undo --vault "$UNDO_VAULT_DIR" 2>&1)
echo "$UNDO_OUT"

soft_check "undo reverted publish commit" \
    "git -C $UNDO_VAULT_DIR log --oneline | grep -q 'Revert'"
soft_check "undo leaves main smoke vault published" \
    "test \$(find \"$VAULT_DIR/wiki\" -maxdepth 1 -name '*.md' ! -name 'index.md' ! -name 'log.md' 2>/dev/null | wc -l | tr -d ' ') -ge 1"

# ── Incremental compile (3rd note → only new concepts compiled) ───────────────
header "Incremental compile"
info "Adding 3rd note to test concept-based incremental updates..."

cat > "$VAULT_DIR/raw/deep-learning.md" <<'EOF'
---
title: Deep Learning
---

Deep learning is a subset of machine learning using neural networks with many layers.

Convolutional Neural Networks (CNNs) excel at image recognition tasks.
Transformers (e.g. BERT, GPT) dominate natural language processing.
Recurrent Neural Networks (RNNs) handle sequential data.

Training requires large datasets and GPUs. Key challenges: vanishing gradients,
overfitting, interpretability. Techniques: dropout, batch normalization,
learning rate scheduling.
EOF

$OLW ingest "$VAULT_DIR/raw/deep-learning.md" 2>&1
INGEST3_OUT=$($OLW compile --dry-run 2>&1)
echo "$INGEST3_OUT"
_TMP=$(mktemp); echo "$INGEST3_OUT" > "$_TMP"
soft_check "dry run shows only new concepts" \
    "grep -qiE 'concept|compile|deep|neural|no concept' \"$_TMP\""
rm -f "$_TMP"

# ── Manual edit protection ────────────────────────────────────────────────────
header "Manual edit protection"
# Find any published wiki article (not index, log, sources)
WIKI_ARTICLE=$(find "$VAULT_DIR/wiki" -maxdepth 1 -name "*.md" \
    ! -name "index.md" ! -name "log.md" 2>/dev/null | head -1)

if [[ -n "$WIKI_ARTICLE" ]]; then
    info "Manually editing: $WIKI_ARTICLE"
    echo -e "\n\nManually added content." >> "$WIKI_ARTICLE"

    # Re-ingest to create a new 'ingested' note that would normally trigger compile
    # Use the already ingested note (force it back to ingested)
    _MEC_RC=0
    COMPILE_OUT=$($OLW compile 2>&1) || _MEC_RC=$?
    echo "$COMPILE_OUT"
    check "compile after manual edit exits 0" "test $_MEC_RC -eq 0"
    # Manually edited article should be skipped (not recompiled)
    DRAFT_AFTER_EDIT=$(find "$VAULT_DIR/wiki/.drafts" -name "$(basename $WIKI_ARTICLE)" 2>/dev/null | wc -l | tr -d ' ')
    soft_check "manually edited article skipped in compile" \
        "test \"$DRAFT_AFTER_EDIT\" -eq 0"
fi

# ── Duplicate detection ───────────────────────────────────────────────────────
header "Duplicate detection"
cp "$VAULT_DIR/raw/quantum-computing.md" "$VAULT_DIR/raw/quantum-computing-copy.md" 2>/dev/null || true

_DUP_RC=0
INGEST_OUT=$($OLW ingest "$VAULT_DIR/raw/quantum-computing-copy.md" 2>&1) || _DUP_RC=$?
check "duplicate ingest exits 0" "test $_DUP_RC -eq 0"
_TMP=$(mktemp); echo "$INGEST_OUT" > "$_TMP"
soft_check "duplicate skipped" "grep -qiE 'skip|duplicate|already' \"$_TMP\""
rm -f "$_TMP"
rm -f "$VAULT_DIR/raw/quantum-computing-copy.md"

# ── Query (Stage 3) ───────────────────────────────────────────────────────────
header "synto query (Stage 3)"
PUBLISHED_ARTICLE_COUNT=$(find "$VAULT_DIR/wiki" -maxdepth 1 -name '*.md' \
    ! -name 'index.md' ! -name 'log.md' 2>/dev/null | wc -l | tr -d ' ')
soft_check "query stage has published articles to search" "test '$PUBLISHED_ARTICLE_COUNT' -ge 1"

info "Running query against wiki..."
_Q_RC=0
QUERY_OUT=$($OLW query "What is a qubit?" 2>&1) || _Q_RC=$?
echo "$QUERY_OUT"
check "query exits 0" "test $_Q_RC -eq 0"
_TMP=$(mktemp); echo "$QUERY_OUT" > "$_TMP"
soft_check "query returns an answer" \
    "grep -qiE 'qubit|quantum|superposition|bit' \"$_TMP\""
rm -f "$_TMP"

info "Running query with --save..."
_QS_RC=0
QUERY_SAVE_OUT=$($OLW query --save "What algorithms are used in quantum computing?" 2>&1) || _QS_RC=$?
echo "$QUERY_SAVE_OUT"
check "query --save exits 0" "test $_QS_RC -eq 0"
QUERY_COUNT=$(find "$VAULT_DIR/wiki/queries" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
soft_check "query --save creates file in wiki/queries/" "test \"$QUERY_COUNT\" -ge 1"

info "Running query with --synthesize..."
_QSY_RC=0
QUERY_SYNTH_OUT=$($OLW query --synthesize "What algorithms are used in quantum computing?" 2>&1) || _QSY_RC=$?
echo "$QUERY_SYNTH_OUT"
check "query --synthesize exits 0" "test $_QSY_RC -eq 0"
SYNTH_COUNT=$(find "$VAULT_DIR/wiki/synthesis" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
soft_check "query --synthesize creates file in wiki/synthesis/" "test \"$SYNTH_COUNT\" -ge 1"
_SYNTH_FILE=$(find "$VAULT_DIR/wiki/synthesis" -name "*.md" | head -1)
soft_check "synthesis frontmatter records kind" "grep -q '^kind: synthesis' \"$_SYNTH_FILE\""
soft_check "synthesis frontmatter records question hash" "grep -q '^question_hash:' \"$_SYNTH_FILE\""
soft_check "synthesis includes sources section" "grep -q '^## Sources' \"$_SYNTH_FILE\""
soft_check "index lists synthesis section" "grep -q '^## Synthesis' \"$VAULT_DIR/wiki/index.md\""

info "Re-running query with --synthesize to check duplicate handling..."
_QSY2_RC=0
QUERY_SYNTH_DUP_OUT=$($OLW query --synthesize "What algorithms are used in quantum computing?" 2>&1) || _QSY2_RC=$?
echo "$QUERY_SYNTH_DUP_OUT"
check "duplicate query --synthesize exits 0" "test $_QSY2_RC -eq 0"
SYNTH_COUNT_AFTER=$(find "$VAULT_DIR/wiki/synthesis" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
soft_check "duplicate query --synthesize keeps existing file" "test \"$SYNTH_COUNT_AFTER\" = \"$SYNTH_COUNT\""

# ── Report (Stage 3) ──────────────────────────────────────────────────────────
header "synto report (Stage 3)"
_STATS_RC=0
STATS_OUT=$($OLW report 2>&1) || _STATS_RC=$?
echo "$STATS_OUT"
check "report exits 0" "test $_STATS_RC -eq 0"
_TMP=$(mktemp); echo "$STATS_OUT" > "$_TMP"
soft_check "report prints raw note count" "grep -q 'Raw notes:' \"$_TMP\""
soft_check "report prints published article count" "grep -q 'Published articles:' \"$_TMP\""
soft_check "report prints synthesis article count" "grep -q 'Synthesis articles:' \"$_TMP\""
soft_check "report prints provider" "grep -q 'Provider:' \"$_TMP\""
soft_check "report shows synthesis article after query --synthesize" \
    "grep -qE 'Synthesis articles: [1-9][0-9]*' \"$_TMP\""
rm -f "$_TMP"

_STATS_JSON_RC=0
STATS_JSON=$($OLW report --json 2>&1) || _STATS_JSON_RC=$?
check "report --json exits 0" "test $_STATS_JSON_RC -eq 0"
_TMP=$(mktemp); echo "$STATS_JSON" > "$_TMP"
soft_check "report --json is parseable and complete" "uv run --project \"$REPO_DIR\" python - \"$_TMP\" <<'PYEOF'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as f:
    payload = json.load(f)

assert payload['vault']['raw_notes'] >= 1
assert payload['vault']['published_articles'] >= 1
assert payload['vault']['synthesis_articles'] >= 1
assert 'rollup_calls' in payload['metrics']
assert 'estimated_cost_usd' in payload['metrics']
PYEOF"
rm -f "$_TMP"

# ── Eval (Stage 3) ────────────────────────────────────────────────────────────
header "synto eval (Stage 3)"
_EVAL_RC=0
EVAL_OUT=$($OLW eval 2>&1) || _EVAL_RC=$?
echo "$EVAL_OUT"
check "eval exits 0" "test $_EVAL_RC -eq 0"
_TMP=$(mktemp); echo "$EVAL_OUT" > "$_TMP"
soft_check "eval prints article coverage" "grep -q 'Article coverage:' \"$_TMP\""
soft_check "eval prints index validity" "grep -q 'INDEX.json validity:' \"$_TMP\""
soft_check "eval prints wikilink resolution" "grep -q 'Wikilink resolution:' \"$_TMP\""
soft_check "eval prints harmonic mean" "grep -q 'Harmonic mean:' \"$_TMP\""
rm -f "$_TMP"

_EVAL_JSON_RC=0
EVAL_JSON=$($OLW eval --json 2>&1) || _EVAL_JSON_RC=$?
check "eval --json exits 0" "test $_EVAL_JSON_RC -eq 0"
_TMP=$(mktemp); echo "$EVAL_JSON" > "$_TMP"
soft_check "eval --json is parseable and complete" "uv run --project \"$REPO_DIR\" python - \"$_TMP\" <<'PYEOF'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as f:
    payload = json.load(f)

assert 'article_coverage' in payload
assert 'index_json_validity' in payload
assert 'wikilink_resolution' in payload
assert 'harmonic_mean' in payload
assert payload['details']['queries_evaluated'] >= 1
PYEOF"
rm -f "$_TMP"

# ── Pack export (Phase 1A) ───────────────────────────────────────────────────
header "synto pack export"
PACK_OUT="$VAULT_DIR/.synto/exports/agents-smoke"
rm -rf "$PACK_OUT"
_PACK_RC=0
PACK_OUT_TEXT=$($OLW pack export --target agents --out "$PACK_OUT" 2>&1) || _PACK_RC=$?
echo "$PACK_OUT_TEXT"
check "pack export exits 0" "test $_PACK_RC -eq 0"
soft_check "pack export writes pack.toml" "test -f '$PACK_OUT/pack.toml'"
soft_check "pack export writes routes.json" "test -f '$PACK_OUT/agent/routes.json'"
soft_check "pack export writes INDEX.json" "test -f '$PACK_OUT/index/INDEX.json'"
soft_check "routes.json has populated Phase 1A payload" "uv run --project \"$REPO_DIR\" python - '$PACK_OUT/agent/routes.json' <<'PYEOF'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as f:
    payload = json.load(f)

assert payload['schema_version'] == 1
assert isinstance(payload['routes'], list)
assert len(payload['routes']) > 0
PYEOF"
soft_check "pack export INDEX.json includes papers" "uv run --project \"$REPO_DIR\" python - '$PACK_OUT/index/INDEX.json' <<'PYEOF'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as f:
    payload = json.load(f)

assert 'papers' in payload
assert payload['papers'] == []
PYEOF"

# ── report clear (Phase 1A) ──────────────────────────────────────────────────
header "synto report clear"
METRIC_ROWS_BEFORE=$(python3 - <<PYEOF
import sqlite3
conn = sqlite3.connect("$VAULT_DIR/.synto/state.db")
events = conn.execute("SELECT COUNT(*) FROM metric_events").fetchone()[0]
rollups = conn.execute("SELECT COUNT(*) FROM metric_daily_rollups").fetchone()[0]
print(events + rollups)
conn.close()
PYEOF
)
soft_check "report has rows before clear" "test '$METRIC_ROWS_BEFORE' -gt 0"
_TC_RC=0
METRICS_CLEAR_OUT=$($OLW report clear --yes 2>&1) || _TC_RC=$?
echo "$METRICS_CLEAR_OUT"
check "report clear exits 0" "test $_TC_RC -eq 0"
soft_check "report clear reports deleted rows" "echo \"$METRICS_CLEAR_OUT\" | grep -q 'rows deleted'"
METRIC_ROWS_AFTER=$(python3 - <<PYEOF
import sqlite3
conn = sqlite3.connect("$VAULT_DIR/.synto/state.db")
events = conn.execute("SELECT COUNT(*) FROM metric_events").fetchone()[0]
rollups = conn.execute("SELECT COUNT(*) FROM metric_daily_rollups").fetchone()[0]
print(events + rollups)
conn.close()
PYEOF
)
soft_check "report clear rows cleared" "test '$METRIC_ROWS_AFTER' -eq 0"

# ── serve help (Phase 1A) ────────────────────────────────────────────────────
header "synto serve --help"
_SERVE_HELP_RC=0
SERVE_HELP_OUT=$($OLW serve --help 2>&1) || _SERVE_HELP_RC=$?
echo "$SERVE_HELP_OUT"
check "serve --help exits 0" "test $_SERVE_HELP_RC -eq 0"
soft_check "serve help mentions stdio transport" "echo \"$SERVE_HELP_OUT\" | grep -q 'stdio'"
soft_check "serve help describes exposed tools" "echo \"$SERVE_HELP_OUT\" | grep -q 'Exposes three tools only'"

# ── Maintain --dry-run (Stage 3) ──────────────────────────────────────────────
header "synto maintain --dry-run (Stage 3)"
LINT_OUT=$($OLW maintain --dry-run 2>&1); _LINT_RC=$?
echo "$LINT_OUT"
check "maintain --dry-run exits 0" "test $_LINT_RC -eq 0"
_TMP=$(mktemp); echo "$LINT_OUT" > "$_TMP"
# CLI prints: "Structural health: {score}/100  {summary}"
soft_check "maintain --dry-run prints Structural health: <score>/100 header" \
    "grep -qiE 'health: [0-9]+(\.[0-9]+)?/100' \"$_TMP\""
rm -f "$_TMP"

# maintain --fix must exit 0
$OLW maintain --fix > /dev/null 2>&1; _LINTFIX_RC=$?
check "maintain --fix exits 0" "test $_LINTFIX_RC -eq 0"

# ── Retry failed (Stage 4) ────────────────────────────────────────────────────
header "synto compile --retry-failed (Stage 4)"
# Inject a fake failed record directly, then verify --retry-failed notices it
python3 - <<PYEOF
import sqlite3, pathlib
db_path = "$VAULT_DIR/.synto/state.db"
conn = sqlite3.connect(db_path)
# Only insert if not already present
conn.execute("""
    INSERT OR IGNORE INTO raw_notes (path, content_hash, status, error)
    VALUES ('raw/fake-failed.md', 'badhash', 'failed', 'simulated failure')
""")
conn.commit()
conn.close()
PYEOF

# status --failed should narrow to just the failed record
_SF_RC=0
STATUS_FAILED=$($OLW status --failed 2>&1) || _SF_RC=$?
check "status --failed exits 0" "test $_SF_RC -eq 0"
soft_check "status --failed lists the failed note" \
    "echo \"\$STATUS_FAILED\" | grep -qF 'raw/fake-failed.md'"

_RETRY_TMP=$(mktemp)
_RETRY_RC=0
$OLW compile --retry-failed > "$_RETRY_TMP" 2>&1 || _RETRY_RC=$?
cat "$_RETRY_TMP"
check "compile --retry-failed exits 0" "test $_RETRY_RC -eq 0"
soft_check "retry-failed reports failed notes" \
    "grep -qiE 'retry|failed|not found|re-ingest' \"$_RETRY_TMP\""
rm -f "$_RETRY_TMP"

# ── synto run (orchestrator) ───────────────────────────────────────────────────
header "synto run (pipeline orchestrator)"
info "Adding 4th note to drive synto run..."

cat > "$VAULT_DIR/raw/reinforcement-learning.md" <<'EOF'
---
title: Reinforcement Learning
---

Reinforcement learning (RL) trains agents to make decisions by maximising
cumulative reward from an environment.

Key components: agent, environment, state, action, reward, policy.
Algorithms: Q-learning, SARSA, PPO, A3C. Applications: game playing (AlphaGo,
Atari), robotics control, recommendation systems, autonomous driving.

Model-free methods learn directly from experience. Model-based methods build
an internal model of the environment for planning.
EOF

_RUN_RC=0
RUN_OUT=$($OLW run 2>&1) || _RUN_RC=$?
echo "$RUN_OUT"
check "synto run exits 0" "test $_RUN_RC -eq 0"
_TMP=$(mktemp); echo "$RUN_OUT" > "$_TMP"
soft_check "synto run completes without fatal error" \
    "! grep -qiE 'traceback|exception|fatal' \"$_TMP\""
soft_check "synto run reports ingested or compiled" \
    "grep -qiE 'ingest|compile|draft|publish|rounds' \"$_TMP\""
rm -f "$_TMP"

DRAFTS_BEFORE=$(find "$VAULT_DIR/wiki/.drafts" -name '*.md' 2>/dev/null | wc -l | tr -d ' ')
_RUNDRY_RC=0
RUN_DRYRUN_OUT=$($OLW run --dry-run 2>&1) || _RUNDRY_RC=$?
check "synto run --dry-run exits 0" "test $_RUNDRY_RC -eq 0"
DRAFTS_AFTER=$(find "$VAULT_DIR/wiki/.drafts" -name '*.md' 2>/dev/null | wc -l | tr -d ' ')
_TMP=$(mktemp); echo "$RUN_DRYRUN_OUT" > "$_TMP"
soft_check "synto run --dry-run makes no LLM calls (no new drafts)" \
    "test '$DRAFTS_AFTER' -eq '$DRAFTS_BEFORE'"
rm -f "$_TMP"

# ── Draft annotations ────────────────────────────────────────────────────────
header "Draft annotations"
info "Compiling with low-quality source to trigger annotation..."
# Force a concept to recompile by inserting low quality + low confidence source
$OLW approve --all 2>&1 || true   # clear drafts first

# Inject a single-source, low-confidence concept by direct DB manipulation
python3 - <<PYEOF
import sqlite3
db_path = "$VAULT_DIR/.synto/state.db"
conn = sqlite3.connect(db_path)
# Update one raw note to low quality so annotation triggers
conn.execute("UPDATE raw_notes SET status='ingested', quality='low' WHERE path='raw/reinforcement-learning.md'")
conn.commit()
conn.close()
PYEOF

$OLW compile 2>&1 || true
# Model may or may not annotate (confidence-dependent). The load-bearing assertion is that
# approve strips any annotations that did land — covered below.
DRAFTS_WITH_ANNOTATION=$({ grep -rl 'olw-auto' "$VAULT_DIR/wiki/.drafts/" 2>/dev/null || true; } \
    | wc -l | tr -d ' ')
info "Annotated drafts before approve: $DRAFTS_WITH_ANNOTATION"

# Verify annotations are stripped on approve
$OLW approve --all 2>&1 || true
PUBLISHED_WITH_ANNOTATION=$({ grep -rl 'olw-auto' "$VAULT_DIR/wiki/" \
    --include='*.md' --exclude-dir='.drafts' --exclude-dir='sources' 2>/dev/null || true; } \
    | wc -l | tr -d ' ')
soft_check "no olw-auto annotations in published articles" \
    "test '$PUBLISHED_WITH_ANNOTATION' -eq 0"

# ── Rejection feedback loop ───────────────────────────────────────────────────
header "Rejection feedback loop"
info "Recompiling to produce a draft to reject..."

# Force one concept back to needing compile
python3 - <<PYEOF
import sqlite3
db_path = "$VAULT_DIR/.synto/state.db"
conn = sqlite3.connect(db_path)
conn.execute("UPDATE raw_notes SET status='ingested' WHERE path='raw/quantum-computing.md'")
conn.commit()
conn.close()
PYEOF

$OLW compile 2>&1 || true

REJECT_DRAFT=$(find "$VAULT_DIR/wiki/.drafts" -name "*.md" | head -1)
if [[ -n "$REJECT_DRAFT" ]]; then
    DRAFT_TITLE=$(grep '^title:' "$REJECT_DRAFT" | head -1 | sed 's/title: *//')
    info "Rejecting draft: $DRAFT_TITLE"

    _REJ_RC=0
    REJECT_OUT=$($OLW reject "$REJECT_DRAFT" --feedback "Too brief, needs more concrete examples" 2>&1) || _REJ_RC=$?
    echo "$REJECT_OUT"
    check "reject exits 0" "test $_REJ_RC -eq 0"
    soft_check "reject removes draft file" "test ! -f \"$REJECT_DRAFT\""
    soft_check "reject confirms feedback saved" \
        "echo \"$REJECT_OUT\" | grep -qiE 'feedback|saved|rejection|next compile'"

    # Force concept back to compile again and verify feedback appears in output
    python3 - <<PYEOF
import sqlite3
db_path = "$VAULT_DIR/.synto/state.db"
conn = sqlite3.connect(db_path)
conn.execute("UPDATE raw_notes SET status='ingested' WHERE path='raw/quantum-computing.md'")
conn.commit()
conn.close()
PYEOF
    _CAR_RC=0
    COMPILE_OUT2=$($OLW compile 2>&1) || _CAR_RC=$?
    echo "$COMPILE_OUT2"
    check "compile after rejection exits 0" "test $_CAR_RC -eq 0"
    # We can't easily inspect the prompt, but compile should succeed without crash
    soft_check "recompile after rejection completes" \
        "! echo \"$COMPILE_OUT2\" | grep -qiE 'traceback|fatal'"
else
    pass "rejection test skipped (no draft available)"
fi

# ── synto unblock ──────────────────────────────────────────────────────────────
header "synto unblock"
info "Simulating 5-rejection block..."

python3 - <<PYEOF
import sqlite3
db_path = "$VAULT_DIR/.synto/state.db"
conn = sqlite3.connect(db_path)
conn.execute("""
    INSERT OR IGNORE INTO blocked_concepts (concept, blocked_at)
    VALUES ('Fake Blocked Concept', datetime('now'))
""")
conn.commit()
conn.close()
PYEOF

_SB_RC=0
STATUS_BLOCKED=$($OLW status 2>&1) || _SB_RC=$?
check "status (with block) exits 0" "test $_SB_RC -eq 0"
_TMP=$(mktemp); echo "$STATUS_BLOCKED" > "$_TMP"
soft_check "status shows blocked concept" \
    "grep -qiE 'blocked|Fake Blocked' \"$_TMP\""
rm -f "$_TMP"

_UB_RC=0
UNBLOCK_OUT=$($OLW unblock "Fake Blocked Concept" 2>&1) || _UB_RC=$?
echo "$UNBLOCK_OUT"
check "unblock exits 0" "test $_UB_RC -eq 0"
soft_check "unblock completes without error" \
    "! echo \"$UNBLOCK_OUT\" | grep -qiE 'traceback|error'"
# Capture status output, then grep — prevents false-pass if status itself crashes
# (previous inline-pipe idiom let a status crash look like "no match" via pipefail disable in check())
_SAU_RC=0
STATUS_AFTER_UNBLOCK=$($OLW status 2>&1) || _SAU_RC=$?
check "status (after unblock) exits 0" "test $_SAU_RC -eq 0"
soft_check "concept no longer blocked after unblock" \
    "! echo \"\$STATUS_AFTER_UNBLOCK\" | grep -qiE 'Fake Blocked'"

# ── synto maintain ─────────────────────────────────────────────────────────────
header "synto maintain"
_M_RC=0
MAINTAIN_OUT=$($OLW maintain 2>&1) || _M_RC=$?
echo "$MAINTAIN_OUT"
check "maintain exits 0" "test $_M_RC -eq 0"
_TMP=$(mktemp); echo "$MAINTAIN_OUT" > "$_TMP"
soft_check "maintain runs without fatal error" \
    "! grep -qiE 'traceback|exception|fatal' \"$_TMP\""
soft_check "maintain reports health or quality info" \
    "grep -qiE 'health|quality|lint|stub|orphan|issue|ok' \"$_TMP\""
rm -f "$_TMP"

_MD_RC=0
MAINTAIN_DRY_OUT=$($OLW maintain --dry-run 2>&1) || _MD_RC=$?
check "maintain --dry-run exits 0" "test $_MD_RC -eq 0"
_TMP=$(mktemp); echo "$MAINTAIN_DRY_OUT" > "$_TMP"
soft_check "maintain --dry-run completes" \
    "! grep -qiE 'traceback|fatal' \"$_TMP\""
rm -f "$_TMP"

# ── synto maintain --fix (stubs) ────────────────────────────────────────────────
header "synto maintain --fix (stub creation)"
# Inject a broken wikilink into a published article so maintain can create a stub
FIRST_WIKI=$(find "$VAULT_DIR/wiki" -maxdepth 1 -name "*.md" \
    ! -name "index.md" ! -name "log.md" 2>/dev/null | head -1)

if [[ -n "$FIRST_WIKI" ]]; then
    echo -e "\n[[Nonexistent Stub Topic]]" >> "$FIRST_WIKI"
    _STUB_RC=0
    STUB_OUT=$($OLW maintain --fix 2>&1) || _STUB_RC=$?
    echo "$STUB_OUT"
    check "maintain --fix (stub) exits 0" "test $_STUB_RC -eq 0"
    _TMP=$(mktemp); echo "$STUB_OUT" > "$_TMP"
    soft_check "maintain --fix runs without fatal error" \
        "! grep -qiE 'traceback|fatal' \"$_TMP\""
    # Verify stub draft was created in .drafts or DB has stub entry
    STUB_DRAFT_COUNT=$(find "$VAULT_DIR/wiki/.drafts" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
    STUB_DB_COUNT=$(python3 -c "
import sqlite3
conn = sqlite3.connect('$VAULT_DIR/.synto/state.db')
try:
    n = conn.execute('SELECT COUNT(*) FROM stubs').fetchone()[0]
    print(n)
except Exception:
    print(0)
conn.close()
" 2>/dev/null || echo 0)
    soft_check "maintain --fix created stub draft or DB entry" \
        "test '$STUB_DRAFT_COUNT' -gt 0 || test '$STUB_DB_COUNT' -gt 0"
    # No stub should have a double .md.md extension (bug: model emits [[raw-note.md]] links)
    DOUBLE_MD_STUBS=$(find "$VAULT_DIR/wiki/.drafts" -name "*.md.md" 2>/dev/null | wc -l | tr -d ' ')
    soft_check "no stub has double .md.md extension" "test '$DOUBLE_MD_STUBS' -eq 0"
    # Stub shape — create_stubs writes to drafts_dir with status=stub, confidence=0.0, [!info] callout.
    # Filename is sanitize_filename("Nonexistent Stub Topic") + .md.
    _STUB_FILE=$(find "$VAULT_DIR/wiki/.drafts" -iname 'nonexistent*stub*topic*.md' 2>/dev/null | head -1)
    if [[ -n "$_STUB_FILE" ]]; then
        soft_check "stub body has [!info] callout" "grep -qF '[!info]' \"$_STUB_FILE\""
        soft_check "stub frontmatter has status: stub" \
            "grep -qE '^status: stub' \"$_STUB_FILE\""
        soft_check "stub frontmatter has confidence: 0 or 0.0" \
            "grep -qE '^confidence: 0(\.0)?\$' \"$_STUB_FILE\""
    else
        info "Stub shape check skipped (no stub file matching nonexistent*stub*topic*.md)"
    fi
    rm -f "$_TMP"
    # Restore the file
    # (truncate last line — safe enough for smoke test purposes)
    sed -i '' '$ d' "$FIRST_WIKI" 2>/dev/null || sed -i '$ d' "$FIRST_WIKI" 2>/dev/null || true
else
    pass "stub creation test skipped (no wiki article available)"
fi

# ── synto maintain --fix (alias-based link repair, issue #29) ──────────────────
header "synto maintain --fix (alias link repair, issue #29)"
# Deterministic setup: inject a known concept + alias + published article directly,
# so the test doesn't depend on what the LLM happened to produce earlier in the run.
REPAIR_WIKI=$(find "$VAULT_DIR/wiki" -maxdepth 1 -name "*.md" \
    ! -name "index.md" ! -name "log.md" 2>/dev/null | head -1)

if [[ -n "$REPAIR_WIKI" ]]; then
    # 1. Register a synthetic concept and unambiguous alias in the DB
    python3 - <<PYEOF
import sqlite3
conn = sqlite3.connect("$VAULT_DIR/.synto/state.db")
# Insert concept if not present (idempotent); schema: (name, source_path)
conn.execute("""
    INSERT OR IGNORE INTO concepts (name, source_path)
    VALUES ('Smoke Test Concept', 'raw/smoke-test-concept.md')
""")
# Register unambiguous alias
conn.execute("""
    INSERT OR IGNORE INTO concept_aliases (concept_name, alias)
    VALUES ('Smoke Test Concept', 'STC alias')
""")
conn.commit()
conn.close()
PYEOF

    # 2. Write a minimal published article for the concept so fix_broken_links
    #    considers it a valid repair target (not just stub-worthy)
    _STC_ARTICLE="$VAULT_DIR/wiki/Smoke Test Concept.md"
    cat > "$_STC_ARTICLE" <<'MDEOF'
---
title: Smoke Test Concept
status: published
tags: [test]
sources: []
confidence: 1.0
created: 2026-01-01
updated: 2026-01-01
---

Synthetic article for smoke test alias repair verification.
MDEOF

    # 3. Inject [[STC alias]] as an alias-form link into a real published article
    echo -e "\n[[STC alias]] — this alias link should be normalized by maintain --fix." >> "$REPAIR_WIKI"

    _REP_RC=0
    REPAIR_OUT=$($OLW maintain --fix 2>&1) || _REP_RC=$?
    echo "$REPAIR_OUT"
    check "maintain --fix (alias repair) exits 0" "test $_REP_RC -eq 0"

    _TMP=$(mktemp); echo "$REPAIR_OUT" > "$_TMP"
    soft_check "maintain --fix runs without fatal error (alias repair, issue #29)" \
        "! grep -qiE 'traceback|fatal' \"$_TMP\""
    rm -f "$_TMP"

    # 4. Verify [[STC alias]] was rewritten to [[Smoke Test Concept|STC alias]]
    soft_check "maintain --fix rewrote alias link to canonical form (issue #29)" \
        "grep -qF '[[Smoke Test Concept|STC alias]]' \"$REPAIR_WIKI\""

    # Cleanup injected content and synthetic article
    sed -i '' '$ d' "$REPAIR_WIKI" 2>/dev/null || sed -i '$ d' "$REPAIR_WIKI" 2>/dev/null || true
    rm -f "$_STC_ARTICLE"
    python3 - <<PYEOF
import sqlite3
conn = sqlite3.connect("$VAULT_DIR/.synto/state.db")
conn.execute("DELETE FROM concept_aliases WHERE alias = 'STC alias'")
conn.execute("DELETE FROM concepts WHERE name = 'Smoke Test Concept' AND source_path = 'raw/smoke-test-concept.md'")
conn.commit()
conn.close()
PYEOF
else
    pass "alias repair test skipped (no wiki article available)"
fi

# ── maintain --fix idempotency ────────────────────────────────────────────────
# After the two maintain --fix runs above, a third run must make zero changes.
# Guards against double-rewrite bugs (alias normalization firing twice, stubs
# re-created, etc). Snapshot published markdown hashes, run again, compare.
header "maintain --fix idempotency"
_SNAP_BEFORE=$(find "$VAULT_DIR/wiki" -type f -name '*.md' \
    -not -path '*/.drafts/*' -exec shasum {} \; 2>/dev/null | sort)
_IDEMP_RC=0
$OLW maintain --fix > /dev/null 2>&1 || _IDEMP_RC=$?
check "maintain --fix (idempotency run) exits 0" "test $_IDEMP_RC -eq 0"
_SNAP_AFTER=$(find "$VAULT_DIR/wiki" -type f -name '*.md' \
    -not -path '*/.drafts/*' -exec shasum {} \; 2>/dev/null | sort)
soft_check "maintain --fix is idempotent (no changes on second run)" \
    "test \"\$_SNAP_BEFORE\" = \"\$_SNAP_AFTER\""

# ── compile --legacy smoke pass ───────────────────────────────────────────────
# Legacy two-step LLM path is shipped but never exercised by smoke. Run one
# minimal invocation to guard against complete-bitrot. Don't assert on quality —
# small-model legacy output is noisy — just "ran, produced a draft".
header "synto compile --legacy"
# Force one note back to 'ingested' so legacy has something to compile
python3 - <<PYEOF
import sqlite3
conn = sqlite3.connect("$VAULT_DIR/.synto/state.db")
conn.execute(
    "UPDATE raw_notes SET status='ingested' WHERE path='raw/machine-learning-basics.md'"
)
conn.commit()
conn.close()
PYEOF

_LEGACY_RC=0
LEGACY_OUT=$($OLW compile --legacy 2>&1) || _LEGACY_RC=$?
echo "$LEGACY_OUT"
# Legacy two-step planning is prone to small-model JSON validation failures.
# Exit-0 is the bit-rot guard — draft production is model-dependent, not asserted.
check "compile --legacy exits 0" "test $_LEGACY_RC -eq 0"

# ── synto support ─────────────────────────────────────────────────────────────
header "synto support"
_SUP_RC=0
SUP_OUT=$($OLW support 2>&1) || _SUP_RC=$?
echo "$SUP_OUT"
check "synto support exits 0" "test $_SUP_RC -eq 0"
_TMP=$(mktemp); echo "$SUP_OUT" > "$_TMP"
soft_check "synto support output contains issues URL keyword" \
    "grep -qiE 'github|issues' \"$_TMP\""
rm -f "$_TMP"

# ── synto doctor (exit code, post-init) ───────────────────────────────────────
header "synto doctor (post-init exit code)"
_DOC_RC=0
DOC_OUT=$($OLW doctor 2>&1) || _DOC_RC=$?
echo "$DOC_OUT"
check "doctor exits 0 when provider reachable and vault valid" "test $_DOC_RC -eq 0"
_TMP=$(mktemp); echo "$DOC_OUT" > "$_TMP"
soft_check "doctor output contains Vault structure section" \
    "grep -q 'Vault structure' \"$_TMP\""
rm -f "$_TMP"

# ── synto config inline-source-citations ──────────────────────────────────────
header "synto config inline-source-citations"
_CISC_RC=0
CISC_STATUS_OUT=$($OLW config inline-source-citations status 2>&1) || _CISC_RC=$?
echo "$CISC_STATUS_OUT"
check "config inline-source-citations status exits 0" "test $_CISC_RC -eq 0"

_CISC_ON_RC=0
CISC_ON_OUT=$($OLW config inline-source-citations on 2>&1) || _CISC_ON_RC=$?
echo "$CISC_ON_OUT"
check "config inline-source-citations on exits 0" "test $_CISC_ON_RC -eq 0"
soft_check "config inline-source-citations on writes true to synto.toml" \
    "grep -q 'inline_source_citations = true' '$VAULT_DIR/synto.toml'"

_CISC_STATUS2_RC=0
CISC_STATUS2_OUT=$($OLW config inline-source-citations status 2>&1) || _CISC_STATUS2_RC=$?
_TMP=$(mktemp); echo "$CISC_STATUS2_OUT" > "$_TMP"
soft_check "config inline-source-citations status after on prints enabled" \
    "grep -qiE 'enabled|on|true' \"$_TMP\""
rm -f "$_TMP"

_CISC_OFF_RC=0
CISC_OFF_OUT=$($OLW config inline-source-citations off 2>&1) || _CISC_OFF_RC=$?
echo "$CISC_OFF_OUT"
check "config inline-source-citations off exits 0" "test $_CISC_OFF_RC -eq 0"
soft_check "config inline-source-citations off writes false to synto.toml" \
    "grep -q 'inline_source_citations = false' '$VAULT_DIR/synto.toml'"

_CISC_STATUS3_RC=0
CISC_STATUS3_OUT=$($OLW config inline-source-citations status 2>&1) || _CISC_STATUS3_RC=$?
_TMP=$(mktemp); echo "$CISC_STATUS3_OUT" > "$_TMP"
soft_check "config inline-source-citations status after off prints disabled" \
    "grep -qiE 'disabled|off|false' \"$_TMP\""
rm -f "$_TMP"

# ── synto ingest --force ──────────────────────────────────────────────────────
header "synto ingest --force"
_IF_RC=0
INGEST_FORCE_OUT=$($OLW ingest --force "$VAULT_DIR/raw/quantum-computing.md" 2>&1) || _IF_RC=$?
echo "$INGEST_FORCE_OUT"
check "ingest --force exits 0" "test $_IF_RC -eq 0"
_IF_STATUS=$(python3 -c "
import sqlite3
conn = sqlite3.connect('$VAULT_DIR/.synto/state.db')
row = conn.execute(\"SELECT status FROM raw_notes WHERE path='raw/quantum-computing.md'\").fetchone()
print(row[0] if row else 'missing')
conn.close()
")
soft_check "ingest --force note still has status ingested" "test '$_IF_STATUS' = 'ingested'"

# ── synto compile --auto-approve ──────────────────────────────────────────────
header "synto compile --auto-approve"
# Force one note back to needing compile (both raw_notes and concept_compile_state)
python3 - <<PYEOF
import sqlite3
conn = sqlite3.connect("$VAULT_DIR/.synto/state.db")
conn.execute("UPDATE raw_notes SET status='ingested' WHERE path='raw/machine-learning-basics.md'")
conn.execute("UPDATE concept_compile_state SET status='pending', error=NULL, compiled_at=NULL, updated_at=datetime('now') WHERE source_path='raw/machine-learning-basics.md'")
conn.commit()
conn.close()
PYEOF
# Clear any existing drafts first
$OLW approve --all 2>&1 || true
_CA_WIKI_BEFORE=$(find "$VAULT_DIR/wiki" -maxdepth 1 -name "*.md" ! -name "index.md" ! -name "log.md" 2>/dev/null | wc -l | tr -d ' ')
_CA_DRAFTS_BEFORE=$(find "$VAULT_DIR/wiki/.drafts" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
_CA_RC=0
COMPILE_AA_OUT=$($OLW compile --auto-approve 2>&1) || _CA_RC=$?
echo "$COMPILE_AA_OUT"
check "compile --auto-approve exits 0" "test $_CA_RC -eq 0"
_CA_DRAFTS_AFTER=$(find "$VAULT_DIR/wiki/.drafts" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
soft_check "compile --auto-approve leaves no drafts" "test '$_CA_DRAFTS_AFTER' -eq 0"
_CA_OUT_TMP=$(mktemp); echo "$COMPILE_AA_OUT" > "$_CA_OUT_TMP"
soft_check "compile --auto-approve published at least one article" \
    "grep -qi 'published' '$_CA_OUT_TMP'"
rm -f "$_CA_OUT_TMP"

# ── synto compile --force ─────────────────────────────────────────────────────
header "synto compile --force"
# Reset note to needing compile (both raw_notes and concept_compile_state)
python3 - <<PYEOF
import sqlite3
conn = sqlite3.connect("$VAULT_DIR/.synto/state.db")
conn.execute("UPDATE raw_notes SET status='ingested' WHERE path='raw/quantum-computing.md'")
conn.execute("UPDATE concept_compile_state SET status='pending', error=NULL, compiled_at=NULL, updated_at=datetime('now') WHERE source_path='raw/quantum-computing.md'")
conn.commit()
conn.close()
PYEOF
# Count drafts before
_CF_DRAFTS_BEFORE=$(find "$VAULT_DIR/wiki/.drafts" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
# Manually edit a published article that shares concepts with quantum-computing.md
_CF_WIKI=$(find "$VAULT_DIR/wiki" -maxdepth 1 -name "*.md" \
    ! -name "index.md" ! -name "log.md" 2>/dev/null | head -1)
if [[ -n "$_CF_WIKI" ]]; then
    echo -e "\n\nManual edit for compile --force test." >> "$_CF_WIKI"
    _CF_RC=0
    COMPILE_FORCE_OUT=$($OLW compile --force 2>&1) || _CF_RC=$?
    echo "$COMPILE_FORCE_OUT"
    check "compile --force exits 0" "test $_CF_RC -eq 0"
    _CF_DRAFTS_AFTER=$(find "$VAULT_DIR/wiki/.drafts" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
    soft_check "compile --force creates new draft bypassing manual edit protection" \
        "test '$_CF_DRAFTS_AFTER' -gt '$_CF_DRAFTS_BEFORE'"
    sed -i '' '$ d' "$_CF_WIKI" 2>/dev/null || sed -i '$ d' "$_CF_WIKI" 2>/dev/null || true
else
    pass "compile --force skipped (no published article to edit)"
fi

# ── synto approve (individual + --min-confidence) ─────────────────────────────
header "synto approve (individual draft + --min-confidence)"
# Use compile --force since concepts are already compiled at this point
$OLW approve --all 2>&1 >/dev/null || true
# Reset note to needing compile (both raw_notes and concept_compile_state)
python3 - <<PYEOF
import sqlite3
conn = sqlite3.connect("$VAULT_DIR/.synto/state.db")
conn.execute("UPDATE raw_notes SET status='ingested' WHERE path='raw/quantum-computing.md'")
conn.execute("UPDATE concept_compile_state SET status='pending', error=NULL, compiled_at=NULL, updated_at=datetime('now') WHERE source_path='raw/quantum-computing.md'")
conn.commit()
conn.close()
PYEOF
_APP_RC=0
APP_DRAFT_OUT=$($OLW compile --force 2>&1) || _APP_RC=$?
echo "$APP_DRAFT_OUT"
_APP_DRAFT_COUNT=$(find "$VAULT_DIR/wiki/.drafts" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$_APP_DRAFT_COUNT" -gt 0 ]]; then
    # Test --min-confidence 2.0 (should hold back all drafts)
    _APP_MC_RC=0
    APP_MC_OUT=$($OLW approve --all --min-confidence 2.0 2>&1) || _APP_MC_RC=$?
    echo "$APP_MC_OUT"
    check "approve --all --min-confidence 2.0 exits 0" "test $_APP_MC_RC -eq 0"
    _APP_MC_DRAFTS=$(find "$VAULT_DIR/wiki/.drafts" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
    soft_check "approve --min-confidence 2.0 holds back all drafts" \
        "test '$_APP_MC_DRAFTS' -eq '$_APP_DRAFT_COUNT'"

    # Test individual draft approval
    _APP_SINGLE=$(find "$VAULT_DIR/wiki/.drafts" -name "*.md" | head -1)
    _APP_SINGLE_RC=0
    APP_SINGLE_OUT=$($OLW approve "$_APP_SINGLE" 2>&1) || _APP_SINGLE_RC=$?
    echo "$APP_SINGLE_OUT"
    check "approve <draft-file> exits 0" "test $_APP_SINGLE_RC -eq 0"
    soft_check "approve <draft-file> removes the draft" "test ! -f \"$_APP_SINGLE\""
else
    pass "approve individual test skipped (no drafts produced)"
fi

# ── synto reject --all ────────────────────────────────────────────────────────
header "synto reject --all"
# Use compile --force to produce drafts
$OLW approve --all 2>&1 >/dev/null || true
# Reset note to needing compile (both raw_notes and concept_compile_state)
python3 - <<PYEOF
import sqlite3
conn = sqlite3.connect("$VAULT_DIR/.synto/state.db")
conn.execute("UPDATE raw_notes SET status='ingested' WHERE path='raw/quantum-computing.md'")
conn.execute("UPDATE concept_compile_state SET status='pending', error=NULL, compiled_at=NULL, updated_at=datetime('now') WHERE source_path='raw/quantum-computing.md'")
conn.commit()
conn.close()
PYEOF
_REJ_RC=0
REJ_DRAFT_OUT=$($OLW compile --force 2>&1) || _REJ_RC=$?
echo "$REJ_DRAFT_OUT"
_REJ_ALL_DRAFTS=$(find "$VAULT_DIR/wiki/.drafts" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$_REJ_ALL_DRAFTS" -gt 0 ]]; then
    _REJ_ALL_RC=0
    REJ_ALL_OUT=$($OLW reject --all --feedback "Not good enough" 2>&1) || _REJ_ALL_RC=$?
    echo "$REJ_ALL_OUT"
    check "reject --all exits 0" "test $_REJ_ALL_RC -eq 0"
    _REJ_ALL_AFTER=$(find "$VAULT_DIR/wiki/.drafts" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
    soft_check "reject --all empties wiki/.drafts" "test '$_REJ_ALL_AFTER' -eq 0"
    _TMP=$(mktemp); echo "$REJ_ALL_OUT" > "$_TMP"
    soft_check "reject --all output mentions rejection" \
        "grep -qiE 'reject' \"$_TMP\""
    rm -f "$_TMP"
else
    pass "reject --all skipped (no drafts to reject)"
fi

# ── synto clean --yes ─────────────────────────────────────────────────────────
header "synto clean --yes"
_CLEAN_VAULT="$(mktemp -d)"
$OLW init "$_CLEAN_VAULT" 2>&1 >/dev/null

if [[ "$PROVIDER" == "ollama" ]]; then
    cat > "$_CLEAN_VAULT/synto.toml" <<CLEANTOML
[models]
fast = "$FAST_MODEL"
heavy = "$HEAVY_MODEL"
embed = "nomic-embed-text"

[ollama]
url = "$PROVIDER_URL"
timeout = 900
fast_ctx = $FAST_CTX
heavy_ctx = $HEAVY_CTX

[pipeline]
auto_approve = false
auto_commit = true
watch_debounce = 3.0

[rag]
chunk_size = 512
chunk_overlap = 50
similarity_threshold = 0.7
CLEANTOML
else
    cat > "$_CLEAN_VAULT/synto.toml" <<CLEANTOML
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
auto_commit = true
watch_debounce = 3.0

[rag]
chunk_size = 512
chunk_overlap = 50
similarity_threshold = 0.7
CLEANTOML
fi

cat > "$_CLEAN_VAULT/raw/clean-test.md" <<'CLEANNOTE'
---
title: Clean Test Note
---
This is a test note for the clean command.
CLEANNOTE

$OLW ingest --all --vault "$_CLEAN_VAULT" 2>&1 >/dev/null || true

soft_check "clean target vault has state.db before clean" "test -f '$_CLEAN_VAULT/.synto/state.db'"
soft_check "clean target vault has wiki files before clean" \
    "test \$(find \"$_CLEAN_VAULT/wiki\" -name '*.md' 2>/dev/null | wc -l) -ge 1"
soft_check "clean target vault raw note exists before clean" "test -f '$_CLEAN_VAULT/raw/clean-test.md'"

_CLEAN_RC=0
CLEAN_OUT=$($OLW clean --vault "$_CLEAN_VAULT" --yes 2>&1) || _CLEAN_RC=$?
echo "$CLEAN_OUT"
check "clean --yes exits 0" "test $_CLEAN_RC -eq 0"
soft_check "clean --yes recreates wiki/.drafts" "test -d '$_CLEAN_VAULT/wiki/.drafts'"
soft_check "clean --yes recreates wiki/sources" "test -d '$_CLEAN_VAULT/wiki/sources'"
soft_check "clean --yes deletes state.db" "test ! -f '$_CLEAN_VAULT/.synto/state.db'"
soft_check "clean --yes leaves raw notes untouched" "test -f '$_CLEAN_VAULT/raw/clean-test.md'"
_CLEAN_WIKI_FILES=$(find "$_CLEAN_VAULT/wiki" -maxdepth 1 -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
soft_check "clean --yes removes wiki article files" "test '$_CLEAN_WIKI_FILES' -eq 0"
rm -rf "$_CLEAN_VAULT"

# ── synto undo --steps 2 ──────────────────────────────────────────────────────
header "synto undo --steps 2"
# wiki/log.md accumulates uncommitted entries (auto_commit=false); restore it so
# git revert can proceed without "would be overwritten by merge" errors.
git -C "$VAULT_DIR" restore wiki/log.md 2>/dev/null || true
# Count existing synto commits
_UNDO_SYNTO_COMMITS=$(git -C "$VAULT_DIR" log --oneline --grep='\[synto\]' 2>/dev/null | wc -l | tr -d ' ')
if [[ "$_UNDO_SYNTO_COMMITS" -ge 2 ]]; then
    _UNDO2_RC=0
    UNDO2_OUT=$($OLW undo --vault "$VAULT_DIR" --steps 2 2>&1) || _UNDO2_RC=$?
    echo "$UNDO2_OUT"
    check "undo --steps 2 exits 0" "test $_UNDO2_RC -eq 0"
    _UNDO2_REVERTS=$(git -C "$VAULT_DIR" log --oneline -4 2>/dev/null | grep -c 'Revert' || true)
    soft_check "undo --steps 2 creates 2 Revert commits" "test '$_UNDO2_REVERTS' -ge 2"
    # Re-approve to restore published state for subsequent tests
    $OLW approve --all 2>&1 >/dev/null || true
else
    pass "undo --steps 2 skipped (fewer than 2 synto commits)"
fi

# ── synto maintain --stubs-only ───────────────────────────────────────────────
header "synto maintain --stubs-only"
# Inject a broken wikilink so maintain --stubs-only has something to report
_MSO_VICTIM=$(find "$VAULT_DIR/wiki" -maxdepth 1 -name "*.md" \
    ! -name "index.md" ! -name "log.md" 2>/dev/null | head -1)
if [[ -n "$_MSO_VICTIM" ]]; then
    echo -e "\n[[Stubs Only Test Link XYZ]]" >> "$_MSO_VICTIM"
fi
_MSO_RC=0
MSO_OUT=$($OLW maintain --stubs-only 2>&1) || _MSO_RC=$?
echo "$MSO_OUT"
check "maintain --stubs-only exits 0" "test $_MSO_RC -eq 0"
_TMP=$(mktemp); echo "$MSO_OUT" > "$_TMP"
soft_check "maintain --stubs-only mentions stub or creates draft" \
    "grep -qiE 'stub|draft' \"$_TMP\""
soft_check "maintain --stubs-only does not mention fix for frontmatter or tags" \
    "! grep -qiE 'fix.*frontmatter|fix.*tag|missing_frontmatter|invalid_tag' \"$_TMP\""
rm -f "$_TMP"
[[ -n "$_MSO_VICTIM" ]] && {
    sed -i '' -e '$ d' -e '$ d' "$_MSO_VICTIM" 2>/dev/null || \
    sed -i -e '$ d' -e '$ d' "$_MSO_VICTIM" 2>/dev/null || true
}

# ── synto maintain --clear-cache ──────────────────────────────────────────────
header "synto maintain --clear-cache"
_MCC_RC=0
MCC_OUT=$($OLW maintain --clear-cache 2>&1) || _MCC_RC=$?
echo "$MCC_OUT"
check "maintain --clear-cache exits 0" "test $_MCC_RC -eq 0"
_TMP=$(mktemp); echo "$MCC_OUT" > "$_TMP"
soft_check "maintain --clear-cache mentions cache or cleared" \
    "grep -qiE 'cache|cleared|deleted' \"$_TMP\""
rm -f "$_TMP"

# ── synto maintain --clear-cache --older-than 0 ───────────────────────────────
header "synto maintain --clear-cache --older-than 0"
_MCC2_RC=0
MCC2_OUT=$($OLW maintain --clear-cache --older-than 0 2>&1) || _MCC2_RC=$?
echo "$MCC2_OUT"
check "maintain --clear-cache --older-than 0 exits 0" "test $_MCC2_RC -eq 0"
_TMP=$(mktemp); echo "$MCC2_OUT" > "$_TMP"
soft_check "maintain --clear-cache --older-than 0 has no traceback" \
    "! grep -qiE 'traceback' \"$_TMP\""
rm -f "$_TMP"

# ── synto items audit ─────────────────────────────────────────────────────────
header "synto items audit"
_IA_RC=0
IA_OUT=$($OLW items audit 2>&1) || _IA_RC=$?
echo "$IA_OUT"
check "items audit exits 0" "test $_IA_RC -eq 0"
_TMP=$(mktemp); echo "$IA_OUT" > "$_TMP"
soft_check "items audit has no traceback" \
    "! grep -qiE 'traceback' \"$_TMP\""
rm -f "$_TMP"

# ── synto items show (missing item) ───────────────────────────────────────────
header "synto items show (missing item)"
_IS_RC=0
IS_OUT=$($OLW items show "NonexistentItemXYZ123" 2>&1) || _IS_RC=$?
echo "$IS_OUT"
check "items show missing item exits non-zero" "test $_IS_RC -ne 0"
_TMP=$(mktemp); echo "$IS_OUT" > "$_TMP"
soft_check "items show missing item output contains not found" \
    "grep -qiE 'not found' \"$_TMP\""
rm -f "$_TMP"

# ── synto trace article ───────────────────────────────────────────────────────
header "synto trace article"
_TRACE_WIKI=$(find "$VAULT_DIR/wiki" -maxdepth 1 -name "*.md" \
    ! -name "index.md" ! -name "log.md" 2>/dev/null | head -1)
if [[ -n "$_TRACE_WIKI" ]]; then
    _TRACE_TITLE=$(grep '^title:' "$_TRACE_WIKI" | head -1 | sed 's/^title: *//')
    _TR_RC=0
    TR_OUT=$($OLW trace article "$_TRACE_TITLE" 2>&1) || _TR_RC=$?
    echo "$TR_OUT"
    check "trace article exits 0" "test $_TR_RC -eq 0"
    _TMP=$(mktemp); echo "$TR_OUT" > "$_TMP"
    soft_check "trace article output contains compile history or title or model" \
        "grep -qiE 'Compile history|model' \"$_TMP\""
    rm -f "$_TMP"
else
    pass "trace article skipped (no published article)"
fi

# ── synto add ─────────────────────────────────────────────────────────────────
header "synto add"
# a) Baseline — .synto/sources is created lazily by synto add; ensure it exists
# so that find doesn't return non-zero and trigger set -e before the first add.
mkdir -p "$VAULT_DIR/.synto/sources"
_ADD_SOURCES_BEFORE=$(find "$VAULT_DIR/.synto/sources" \
    -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
_ADD_RAW_BEFORE=$(find "$VAULT_DIR/raw" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')

# b) Create temp source file
_ADD_TMPFILE=$(mktemp /tmp/synto-add-test-XXXXXX.md)
cat > "$_ADD_TMPFILE" <<'ADDNOTE'
---
title: External Source Note
---
This is an external source imported via synto add.
ADDNOTE

# c) First add
_ADD_RC=0
ADD_OUT=$($OLW add "$_ADD_TMPFILE" 2>&1) || _ADD_RC=$?
echo "$ADD_OUT"
check "add exits 0" "test $_ADD_RC -eq 0"
_ADD_TMP=$(mktemp); echo "$ADD_OUT" > "$_ADD_TMP"
soft_check "add has no traceback" \
    "! grep -qiE 'traceback' \"$_ADD_TMP\""
rm -f "$_ADD_TMP"
_ADD_SOURCES_AFTER=$(find "$VAULT_DIR/.synto/sources" \
    -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
soft_check "add increases .synto/sources directory count by one" \
    "test '$_ADD_SOURCES_AFTER' -eq $((_ADD_SOURCES_BEFORE + 1))"
_ADD_RAW_AFTER=$(find "$VAULT_DIR/raw" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
soft_check "add increases raw note count by one" \
    "test '$_ADD_RAW_AFTER' -eq $((_ADD_RAW_BEFORE + 1))"
_ADD_DB_COUNT=$(python3 -c "
import sqlite3
conn = sqlite3.connect('$VAULT_DIR/.synto/state.db')
n = conn.execute('SELECT COUNT(*) FROM source_documents').fetchone()[0]
print(n)
conn.close()
")
soft_check "add inserts row into source_documents table" "test '$_ADD_DB_COUNT' -ge 1"

# d) Duplicate detection
_ADD_DUP_RC=0
_ADD_DUP_TMP=$(mktemp)
$OLW add "$_ADD_TMPFILE" > "$_ADD_DUP_TMP" 2>&1 || _ADD_DUP_RC=$?
check "add duplicate without --force exits non-zero" "test $_ADD_DUP_RC -ne 0"
soft_check "add duplicate output mentions already imported or force" \
    "grep -qiE 'Already imported|already|force' \"$_ADD_DUP_TMP\""
rm -f "$_ADD_DUP_TMP"

# e) --force re-import
_ADD_FORCE_RC=0
_ADD_FORCE_OUT=$($OLW add --force "$_ADD_TMPFILE" 2>&1) || _ADD_FORCE_RC=$?
echo "$_ADD_FORCE_OUT"
check "add --force exits 0" "test $_ADD_FORCE_RC -eq 0"
_ADD_FORCE_TMP=$(mktemp); echo "$_ADD_FORCE_OUT" > "$_ADD_FORCE_TMP"
soft_check "add --force has no traceback" \
    "! grep -qiE 'traceback' \"$_ADD_FORCE_TMP\""
rm -f "$_ADD_FORCE_TMP"
_ADD_SOURCES_FORCE=$(find "$VAULT_DIR/.synto/sources" \
    -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
soft_check "add --force does not grow source directory count (reuses existing)" \
    "test '$_ADD_SOURCES_FORCE' -eq '$_ADD_SOURCES_AFTER'"

# f) add → ingest continuity
_ADD_WIKI_SRC_BEFORE=$(find "$VAULT_DIR/wiki/sources" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
_ADD_INGEST_RC=0
_ADD_INGEST_OUT=$($OLW ingest --all 2>&1) || _ADD_INGEST_RC=$?
check "ingest after add exits 0" "test $_ADD_INGEST_RC -eq 0"
_ADD_WIKI_SRC_AFTER=$(find "$VAULT_DIR/wiki/sources" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
soft_check "ingest after add creates at least one source summary page" \
    "test '$_ADD_WIKI_SRC_AFTER' -gt '$_ADD_WIKI_SRC_BEFORE'"

# g) Cleanup
rm -f "$_ADD_TMPFILE"

# ── synto pack export --out (external path) ───────────────────────────────────
header "synto pack export --out (external path)"
_PACK2_OUT="/tmp/synto-pack-$$-external"
rm -rf "$_PACK2_OUT"
_PACK2_RC=0
PACK2_OUT_TEXT=$($OLW pack export --target agents --out "$_PACK2_OUT" 2>&1) || _PACK2_RC=$?
echo "$PACK2_OUT_TEXT"
check "pack export --out exits 0" "test $_PACK2_RC -eq 0"
soft_check "pack export --out writes pack.toml" "test -f '$_PACK2_OUT/pack.toml'"
rm -rf "$_PACK2_OUT"

# ── synto report --since 7d ───────────────────────────────────────────────────
header "synto report --since 7d"
_RS_RC=0
RS_OUT=$($OLW report --since 7d 2>&1) || _RS_RC=$?
echo "$RS_OUT"
check "report --since 7d exits 0" "test $_RS_RC -eq 0"
_TMP=$(mktemp); echo "$RS_OUT" > "$_TMP"
soft_check "report --since 7d prints Raw notes" \
    "grep -q 'Raw notes:' \"$_TMP\""
rm -f "$_TMP"

# ── synto eval --queries ──────────────────────────────────────────────────────
header "synto eval --queries"
_EVAL_Q_TOML=$(mktemp /tmp/synto-eval-queries-XXXXXX.toml)
cat > "$_EVAL_Q_TOML" <<'EVALTOML'
[[query]]
id = "q1"
question = "What is a qubit?"
expected_contains = ["qubit"]
EVALTOML

_EQ_RC=0
EQ_OUT=$($OLW eval --queries "$_EVAL_Q_TOML" 2>&1) || _EQ_RC=$?
echo "$EQ_OUT"
check "eval --queries exits 0" "test $_EQ_RC -eq 0"
_TMP=$(mktemp); echo "$EQ_OUT" > "$_TMP"
soft_check "eval --queries prints Article coverage" \
    "grep -q 'Article coverage:' \"$_TMP\""
rm -f "$_TMP"
rm -f "$_EVAL_Q_TOML"

# ── synto serve (transport validation) ────────────────────────────────────────
header "synto serve (transport validation)"
_SV_RC=0
SV_OUT=$($OLW serve --transport invalid_transport 2>&1) || _SV_RC=$?
echo "$SV_OUT"
check "serve --transport invalid exits non-zero" "test $_SV_RC -ne 0"
_TMP=$(mktemp); echo "$SV_OUT" > "$_TMP"
soft_check "serve invalid transport output mentions transport/invalid/choice" \
    "grep -qiE 'transport|invalid|choice' \"$_TMP\""
rm -f "$_TMP"

# ── synto run --auto-approve ──────────────────────────────────────────────────
header "synto run --auto-approve"
# Reset note to needing compile (both raw_notes and concept_compile_state)
python3 - <<PYEOF
import sqlite3
conn = sqlite3.connect("$VAULT_DIR/.synto/state.db")
conn.execute("UPDATE raw_notes SET status='ingested' WHERE path='raw/quantum-computing.md'")
conn.execute("UPDATE concept_compile_state SET status='pending', error=NULL, compiled_at=NULL, updated_at=datetime('now') WHERE source_path='raw/quantum-computing.md'")
conn.commit()
conn.close()
PYEOF
# Clear drafts first
$OLW approve --all 2>&1 >/dev/null || true
_RAA_RC=0
RAA_OUT=$($OLW run --auto-approve 2>&1) || _RAA_RC=$?
echo "$RAA_OUT"
check "run --auto-approve exits 0" "test $_RAA_RC -eq 0"
_TMP=$(mktemp); echo "$RAA_OUT" > "$_TMP"
soft_check "run --auto-approve has no traceback" \
    "! grep -qiE 'traceback' \"$_TMP\""
rm -f "$_TMP"
_RAA_DRAFTS=$(find "$VAULT_DIR/wiki/.drafts" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
soft_check "run --auto-approve leaves no drafts" "test '$_RAA_DRAFTS' -eq 0"

# ── synto run --fix ───────────────────────────────────────────────────────────
header "synto run --fix"
# Inject a broken wikilink into a published article
_RF_WIKI=$(find "$VAULT_DIR/wiki" -maxdepth 1 -name "*.md" \
    ! -name "index.md" ! -name "log.md" 2>/dev/null | head -1)
if [[ -n "$_RF_WIKI" ]]; then
    echo -e "\n[[Definitely Broken Link XYZ]]" >> "$_RF_WIKI"
    _RF_RC=0
    RF_OUT=$($OLW run --fix 2>&1) || _RF_RC=$?
    echo "$RF_OUT"
    check "run --fix exits 0" "test $_RF_RC -eq 0"
    _TMP=$(mktemp); echo "$RF_OUT" > "$_TMP"
    soft_check "run --fix has no traceback" \
        "! grep -qiE 'traceback' \"$_TMP\""
    rm -f "$_TMP"
    # Restore
    sed -i '' '$ d' "$_RF_WIKI" 2>/dev/null || sed -i '$ d' "$_RF_WIKI" 2>/dev/null || true
else
    pass "run --fix skipped (no published article)"
fi

# ── synto run --max-rounds 1 ──────────────────────────────────────────────────
header "synto run --max-rounds 1"
_RMR_RC=0
RMR_OUT=$($OLW run --max-rounds 1 2>&1) || _RMR_RC=$?
echo "$RMR_OUT"
check "run --max-rounds 1 exits 0" "test $_RMR_RC -eq 0"
_TMP=$(mktemp); echo "$RMR_OUT" > "$_TMP"
soft_check "run --max-rounds 1 mentions round/rounds/ingest/compile" \
    "grep -qiE 'round|ingest|compile' \"$_TMP\""
rm -f "$_TMP"

# ── Pipeline lock (concurrent invocation) ─────────────────────────────────────
header "Pipeline lock (concurrent invocation)"
# Test the lock mechanism directly in Python since uv run may use a daemon
# process that doesn't inherit the parent's flock.
_PL_LOCK_SCRIPT=$(mktemp /tmp/synto_lock_test.XXXXXX.py)
cat > "$_PL_LOCK_SCRIPT" <<'PYEOF'
import fcntl, os, subprocess, sys

vault = sys.argv[1]
lock_path = os.path.join(vault, ".synto", "pipeline.lock")
os.makedirs(os.path.dirname(lock_path), exist_ok=True)

f = open(lock_path, "a+")
fcntl.flock(f, fcntl.LOCK_EX)
f.seek(0)
f.truncate()
f.write(str(os.getpid()))
f.flush()

child_code = (
    "import fcntl\n"
    "f = open('" + lock_path + "', 'a+')\n"
    "try:\n"
    "    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
    "    print('ACQUIRED')\n"
    "    fcntl.flock(f, fcntl.LOCK_UN)\n"
    "except BlockingIOError:\n"
    "    print('BLOCKED')\n"
    "f.close()\n"
)
r = subprocess.run([sys.executable, "-c", child_code], capture_output=True, text=True)
child_result = r.stdout.strip()

fcntl.flock(f, fcntl.LOCK_UN)
f.close()

if child_result == "BLOCKED":
    print("PASS: lock blocks concurrent access")
    sys.exit(0)
else:
    print("FAIL: child " + child_result + " (should be BLOCKED)")
    sys.exit(1)
PYEOF

_PL_RC=0
PL_OUT=$(uv run --project "$REPO_DIR" python "$_PL_LOCK_SCRIPT" "$VAULT_DIR" 2>&1) || _PL_RC=$?
echo "$PL_OUT"
check "pipeline lock blocks concurrent access" "test $_PL_RC -eq 0"
rm -f "$_PL_LOCK_SCRIPT"

# ── synto --version ───────────────────────────────────────────────────────────
header "synto --version"
_VER_RC=0
VER_OUT=$($OLW --version 2>&1) || _VER_RC=$?
echo "$VER_OUT"
check "synto --version exits 0" "test $_VER_RC -eq 0"
_TMP=$(mktemp); echo "$VER_OUT" > "$_TMP"
soft_check "synto --version matches version pattern" \
    "grep -qE '[0-9]+\.[0-9]+\.[0-9]+' \"$_TMP\""
rm -f "$_TMP"

# ── synto status (fresh vault, no ingest) ─────────────────────────────────────
header "synto status (fresh vault, no ingest)"
_STATUS_FRESH_VAULT="$(mktemp -d)"
$OLW init "$_STATUS_FRESH_VAULT" 2>&1 >/dev/null
if [[ "$PROVIDER" == "ollama" ]]; then
    cat > "$_STATUS_FRESH_VAULT/synto.toml" <<SFVTOML
[models]
fast = "$FAST_MODEL"
heavy = "$HEAVY_MODEL"
embed = "nomic-embed-text"

[ollama]
url = "$PROVIDER_URL"
timeout = 900
fast_ctx = $FAST_CTX
heavy_ctx = $HEAVY_CTX

[pipeline]
auto_approve = false
auto_commit = true
watch_debounce = 3.0

[rag]
chunk_size = 512
chunk_overlap = 50
similarity_threshold = 0.7
SFVTOML
else
    cat > "$_STATUS_FRESH_VAULT/synto.toml" <<SFVTOML
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
auto_commit = true
watch_debounce = 3.0

[rag]
chunk_size = 512
chunk_overlap = 50
similarity_threshold = 0.7
SFVTOML
fi
_SF_RC=0
SF_OUT=$($OLW status --vault "$_STATUS_FRESH_VAULT" 2>&1) || _SF_RC=$?
echo "$SF_OUT"
check "status on fresh vault exits 0" "test $_SF_RC -eq 0"
_TMP=$(mktemp); echo "$SF_OUT" > "$_TMP"
soft_check "status on fresh vault has no traceback" \
    "! grep -qiE 'traceback' \"$_TMP\""
rm -f "$_TMP"
rm -rf "$_STATUS_FRESH_VAULT"

# ── Config loading order: SYNTO_VAULT env vs --vault flag ─────────────────────
header "Config loading: SYNTO_VAULT env vs --vault flag"
_CLO_ENV_RC=0
CLO_ENV_OUT=$($OLW status 2>&1) || _CLO_ENV_RC=$?
check "status via SYNTO_VAULT env exits 0" "test $_CLO_ENV_RC -eq 0"

_CLO_FLAG_RC=0
CLO_FLAG_OUT=$($OLW status --vault "$VAULT_DIR" 2>&1) || _CLO_FLAG_RC=$?
check "status via --vault flag exits 0" "test $_CLO_FLAG_RC -eq 0"

_TMP_ENV=$(mktemp); echo "$CLO_ENV_OUT" > "$_TMP_ENV"
_TMP_FLAG=$(mktemp); echo "$CLO_FLAG_OUT" > "$_TMP_FLAG"
soft_check "status via env shows ingested or published" \
    "grep -qiE 'ingested|published' \"$_TMP_ENV\""
soft_check "status via flag shows ingested or published" \
    "grep -qiE 'ingested|published' \"$_TMP_FLAG\""
rm -f "$_TMP_ENV" "$_TMP_FLAG"

# ── Inline source citations end-to-end ────────────────────────────────────────
header "Inline source citations end-to-end"
# Save current setting
if grep -q 'inline_source_citations = true' "$VAULT_DIR/synto.toml" 2>/dev/null; then
    _ISC_PREV="on"
else
    _ISC_PREV="off"
fi
# Enable inline citations
$OLW config inline-source-citations on 2>&1 >/dev/null || true
# Reset note to needing compile (both raw_notes and concept_compile_state)
python3 - <<PYEOF
import sqlite3
conn = sqlite3.connect("$VAULT_DIR/.synto/state.db")
conn.execute("UPDATE raw_notes SET status='ingested' WHERE path='raw/quantum-computing.md'")
conn.execute("UPDATE concept_compile_state SET status='pending', error=NULL, compiled_at=NULL, updated_at=datetime('now') WHERE source_path='raw/quantum-computing.md'")
conn.commit()
conn.close()
PYEOF
_ISC_RC=0
# Use --force: by this stage articles are flagged as manually edited from
# earlier test sections, so a plain compile skips them all.
ISC_COMPILE_OUT=$($OLW compile --force 2>&1) || _ISC_RC=$?
echo "$ISC_COMPILE_OUT"
check "inline citations compile exits 0" "test $_ISC_RC -eq 0"

$OLW approve --all 2>&1 >/dev/null || true
# Wrap grep in || true so set -eo pipefail doesn't exit on zero matches.
_ISC_CITATION_COUNT=$({ grep -r '\[S[0-9]' "$VAULT_DIR/wiki/" \
    --include='*.md' --exclude-dir=.drafts 2>/dev/null || true; } | wc -l | tr -d ' ')
soft_check "published wiki articles contain inline source citation markers" \
    "test '$_ISC_CITATION_COUNT' -ge 1"
# Restore setting
$OLW config inline-source-citations "$_ISC_PREV" 2>&1 >/dev/null || true

# ── synto migrate-olw ─────────────────────────────────────────────────────────
header "synto migrate-olw"
_MOLW_DIR="$(mktemp -d)"
cat > "$_MOLW_DIR/wiki.toml" <<'MOLWTOML'
[models]
fast = "gemma4:e4b"
heavy = "gemma4:e4b"
MOLWTOML
mkdir -p "$_MOLW_DIR/.olw"
_MOLW_RC=0
_MOLW_TMP=$(mktemp)
$OLW migrate-olw --vault "$_MOLW_DIR" > "$_MOLW_TMP" 2>&1 || _MOLW_RC=$?
cat "$_MOLW_TMP"
check "migrate-olw exits 0" "test $_MOLW_RC -eq 0"
soft_check "migrate-olw creates synto.toml" "test -f '$_MOLW_DIR/synto.toml'"
soft_check "migrate-olw creates .synto directory" "test -d '$_MOLW_DIR/.synto'"
soft_check "migrate-olw .gitignore contains pipeline.lock" \
    "grep -q '.synto/pipeline.lock' '$_MOLW_DIR/.gitignore' 2>/dev/null || grep -qF '.synto/pipeline.lock' '$_MOLW_DIR/.gitignore'"
soft_check "migrate-olw output contains Migrated" \
    "grep -qiE 'Migrated' \"$_MOLW_TMP\""
rm -f "$_MOLW_TMP"
rm -rf "$_MOLW_DIR"

# ── synto doctor (uninitialised vault) ────────────────────────────────────────
header "synto doctor (uninitialised vault)"
_NODOC_DIR="$(mktemp -d)"
_NODOC_RC=0
_NODOC_TMP=$(mktemp)
$OLW doctor --vault "$_NODOC_DIR" > "$_NODOC_TMP" 2>&1 || _NODOC_RC=$?
cat "$_NODOC_TMP"
check "doctor on uninitialised vault exits non-zero" "test $_NODOC_RC -ne 0"
soft_check "doctor on uninitialised vault mentions not initialised or missing or init" \
    "grep -qiE 'not initialised|not initialized|missing|init' \"$_NODOC_TMP\""
rm -f "$_NODOC_TMP"
rm -rf "$_NODOC_DIR"

# ── synto eval --live (not implemented) ───────────────────────────────────────
header "synto eval --live (not implemented)"
_ELIVE_RC=0
_ELIVE_TMP=$(mktemp)
$OLW eval --live > "$_ELIVE_TMP" 2>&1 || _ELIVE_RC=$?
cat "$_ELIVE_TMP"
check "eval --live exits 2" "test $_ELIVE_RC -eq 2"
soft_check "eval --live output mentions not implemented or Phase 1A" \
    "grep -qiE 'not implemented|Phase 1A' \"$_ELIVE_TMP\""
rm -f "$_ELIVE_TMP"

# ── synto approve --all (no drafts) ───────────────────────────────────────────
header "synto approve --all (no drafts)"
$OLW approve --all 2>&1 >/dev/null || true
_EMPTY_DRAFTS=$(find "$VAULT_DIR/wiki/.drafts" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
soft_check "drafts are empty before approve --all (no drafts)" "test '$_EMPTY_DRAFTS' -eq 0"
_NODRAFT_RC=0
_NODRAFT_TMP=$(mktemp)
$OLW approve --all > "$_NODRAFT_TMP" 2>&1 || _NODRAFT_RC=$?
cat "$_NODRAFT_TMP"
check "approve --all with no drafts exits 0" "test $_NODRAFT_RC -eq 0"
soft_check "approve --all with no drafts has no traceback" \
    "! grep -qiE 'traceback' \"$_NODRAFT_TMP\""
rm -f "$_NODRAFT_TMP"

# ── synto items show (existing item) ──────────────────────────────────────────
header "synto items show (existing item)"
_ITEM_NAME=$(python3 -c "
import sqlite3
conn = sqlite3.connect('$VAULT_DIR/.synto/state.db')
row = conn.execute('SELECT name FROM knowledge_items LIMIT 1').fetchone()
print(row[0] if row else '')
conn.close()
")
if [[ -n "$_ITEM_NAME" ]]; then
    _ISHOW_RC=0
    _ISHOW_TMP=$(mktemp)
    $OLW items show "$_ITEM_NAME" > "$_ISHOW_TMP" 2>&1 || _ISHOW_RC=$?
    cat "$_ISHOW_TMP"
    check "items show existing item exits 0" "test $_ISHOW_RC -eq 0"
    soft_check "items show output contains item name or kind or confidence" \
        "grep -qiE '$_ITEM_NAME|kind:|confidence:' \"$_ISHOW_TMP\""
    soft_check "items show has no traceback" \
        "! grep -qiE 'traceback' \"$_ISHOW_TMP\""
    rm -f "$_ISHOW_TMP"
else
    pass "items show existing skipped (no items in DB)"
fi

# ── auto_commit = false (no git commits on approve) ───────────────────────────
header "auto_commit = false (no git commits on approve)"
_NC_VAULT="$(mktemp -d)"
$OLW init "$_NC_VAULT" 2>&1 >/dev/null
if [[ "$PROVIDER" == "ollama" ]]; then
    cat > "$_NC_VAULT/synto.toml" <<NCTOML
[models]
fast = "$FAST_MODEL"
heavy = "$HEAVY_MODEL"
embed = "nomic-embed-text"

[ollama]
url = "$PROVIDER_URL"
timeout = 900
fast_ctx = $FAST_CTX
heavy_ctx = $HEAVY_CTX

[pipeline]
auto_approve = false
auto_commit = false
watch_debounce = 3.0

[rag]
chunk_size = 512
chunk_overlap = 50
similarity_threshold = 0.7
NCTOML
else
    cat > "$_NC_VAULT/synto.toml" <<NCTOML
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
watch_debounce = 3.0

[rag]
chunk_size = 512
chunk_overlap = 50
similarity_threshold = 0.7
NCTOML
fi
cat > "$_NC_VAULT/raw/no-commit-test.md" <<'NCNOTE'
---
title: No Commit Test
---
This note tests that auto_commit = false prevents git commits on approve.
NCNOTE
_NC_INGEST_RC=0
$OLW ingest --all --vault "$_NC_VAULT" 2>&1 >/dev/null || _NC_INGEST_RC=$?
check "no-commit ingest exits 0" "test $_NC_INGEST_RC -eq 0"
_NC_COMPILE_RC=0
$OLW compile --vault "$_NC_VAULT" 2>&1 >/dev/null || _NC_COMPILE_RC=$?
check "no-commit compile exits 0" "test $_NC_COMPILE_RC -eq 0"
_NC_APPROVE_RC=0
_NC_APPROVE_TMP=$(mktemp)
$OLW approve --all --vault "$_NC_VAULT" > "$_NC_APPROVE_TMP" 2>&1 || _NC_APPROVE_RC=$?
cat "$_NC_APPROVE_TMP"
check "no-commit approve --all exits 0" "test $_NC_APPROVE_RC -eq 0"
rm -f "$_NC_APPROVE_TMP"
_NC_SYNTO_COMMITS=$(git -C "$_NC_VAULT" log --oneline 2>/dev/null | grep -c '\[synto\]' || true)
check "auto_commit = false produces zero git commits with [synto] prefix" \
    "test '$_NC_SYNTO_COMMITS' -eq 0"
rm -rf "$_NC_VAULT"

# ── Summary ───────────────────────────────────────────────────────────────────
header "Results"
echo -e "${GREEN}${BOLD}All checks passed: $PASS_COUNT${NC}"
echo ""
echo "Wiki articles created:"
find "$VAULT_DIR/wiki" -name "*.md" -not -path "*/.drafts/*" | sort | sed 's/^/  /'
echo ""
echo "To inspect the vault:"
echo "  export SYNTO_VAULT=$VAULT_DIR"
echo "  uv run --project $REPO_DIR synto status"
if [[ "$KEEP_VAULT" == "1" ]]; then
    echo "  open $VAULT_DIR in Obsidian"
fi
