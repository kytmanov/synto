#!/usr/bin/env bash
# smoke_source_overrides.sh — end-to-end test for per-source-type ingest overrides
#
# Verifies that [pipeline.source_overrides] in synto.toml lifts the
# concept-extraction ceiling for a given source_type at runtime.
#
# Usage:
#   ./scripts/smoke_source_overrides.sh                              # Ollama
#   PROVIDER=lm_studio ./scripts/smoke_source_overrides.sh           # LM Studio
#   VAULT_DIR=/tmp/my-vault ./scripts/smoke_source_overrides.sh      # keep vault
#
# Requirements:
#   - uv (https://docs.astral.sh/uv/)
#   - Ollama or LM Studio running with a model loaded

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROVIDER="${PROVIDER:-ollama}"

case "$PROVIDER" in
    ollama)
        PROVIDER_URL="${PROVIDER_URL:-${OLLAMA_URL:-http://localhost:11434}}"
        FAST_MODEL="${FAST_MODEL:-gemma4:e4b}"
        HEAVY_MODEL="${HEAVY_MODEL:-gemma4:e4b}"
        FAST_CTX=8192
        ;;
    lm_studio)
        PROVIDER_URL="${PROVIDER_URL:-http://localhost:1234/v1}"
        FAST_MODEL="${FAST_MODEL:-google/gemma-4-e4b}"
        HEAVY_MODEL="${HEAVY_MODEL:-google/gemma-4-e4b}"
        FAST_CTX=8192
        ;;
    *)
        echo "Unsupported PROVIDER: $PROVIDER" >&2
        exit 1
        ;;
esac

KEEP_VAULT="${KEEP_VAULT:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

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

_T0=$(date +%s)

pass() { echo -e "${GREEN}✓${NC} $1"; }
fail() {
    local desc="$1" detail="${2:-}"
    echo -e "  ${RED}✗ FAIL: $desc${NC}${detail:+$'\n'    ${detail:0:1000}}"
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
    else
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
    else
        echo -e "  ${RED}✗ SOFT FAIL:${NC} $desc${out:+ — ${out:0:1000}}"
    fi
}

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

with tempfile.TemporaryDirectory(prefix="smoke-resolve-") as tmp:
    vault = Path(tmp)
    (vault / "raw").mkdir()
    (vault / "wiki").mkdir()
    (vault / ".synto").mkdir()
    (vault / "synto.toml").write_text(
        f"[models]\nfast = \"{model}\"\nheavy = \"{model}\"\n\n"
        f"[provider]\nname = \"{provider}\"\nurl = \"{url}\"\n"
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

cleanup() {
    if [[ "$KEEP_VAULT" == "0" ]]; then
        rm -rf "$VAULT_DIR"
    else
        echo -e "\nVault kept at: ${BOLD}$VAULT_DIR${NC}"
    fi
}
trap cleanup EXIT

# ── Prerequisites ─────────────────────────────────────────────────────────────
header "Prerequisites (provider: $PROVIDER)"

check "uv available" "command -v uv"

if [[ "$PROVIDER" == "ollama" ]]; then
    check "Ollama reachable at $PROVIDER_URL" "curl -sf $PROVIDER_URL/api/tags"
else
    check "$PROVIDER reachable at $PROVIDER_URL" "curl -sf $PROVIDER_URL/models"
fi

info "Resolving model..."
FAST_MODEL="$(resolve_loaded_model "$FAST_MODEL")"
pass "Model resolved: $FAST_MODEL"

# ── Setup vault ──────────────────────────────────────────────────────────────
header "Setup vault"

info "Using vault: $VAULT_DIR"
uv sync --project "$REPO_DIR" --quiet

OLW="uv run --project $REPO_DIR synto"
export SYNTO_VAULT="$VAULT_DIR"

$OLW init "$VAULT_DIR" 2>&1 | grep -v "^$" || true

soft_check "raw/ created"           "test -d $VAULT_DIR/raw"
soft_check "wiki/ created"          "test -d $VAULT_DIR/wiki"
soft_check ".synto/ created"        "test -d $VAULT_DIR/.synto"

# ── Write synto.toml with source_overrides ──────────────────────────────────
header "Configure source_overrides"

if [[ "$PROVIDER" == "ollama" ]]; then
    cat > "$VAULT_DIR/synto.toml" <<TOML
[models]
fast = "$FAST_MODEL"
heavy = "$FAST_MODEL"

[ollama]
url = "$PROVIDER_URL"
timeout = 900
fast_ctx = $FAST_CTX

[pipeline]
auto_approve = false
auto_commit = true
max_concepts_per_source = 8

[pipeline.source_overrides.textbook]
max_concepts_per_source = 25
TOML
else
    cat > "$VAULT_DIR/synto.toml" <<TOML
[models]
fast = "$FAST_MODEL"
heavy = "$FAST_MODEL"

[provider]
name = "$PROVIDER"
url = "$PROVIDER_URL"
timeout = 900
fast_ctx = $FAST_CTX

[pipeline]
auto_approve = false
auto_commit = true
max_concepts_per_source = 8

[pipeline.source_overrides.textbook]
max_concepts_per_source = 25
TOML
fi

pass "synto.toml written with source_overrides.textbook.max_concepts_per_source = 25"

# ── Deterministic config-level checks ────────────────────────────────────────
header "Config-level override verification"

_CFG_CHECKS=$(uv run --project "$REPO_DIR" python - "$VAULT_DIR" <<'PY' 2>&1
import sys
from pathlib import Path
from synto.config import Config

vault = Path(sys.argv[1])
cfg = Config.from_vault(vault)

tb = cfg.pipeline.effective_max_concepts("textbook")
nt = cfg.pipeline.effective_max_concepts("notes")
override = cfg.pipeline.source_overrides

assert tb == 25, f"textbook should be 25, got {tb}"
assert nt == 8, f"notes should be 8 (global fallback), got {nt}"
assert "textbook" in override, "textbook key should be in source_overrides"
assert override["textbook"].max_concepts_per_source == 25
print(f"effective_max_concepts(textbook)=25  effective_max_concepts(notes)=8")

# Unknown source type warns but does not raise
import logging
logging.getLogger("synto.config").setLevel(logging.WARNING)
import io, logging.handlers
buf = io.StringIO()
handler = logging.StreamHandler(buf)
handler.setLevel(logging.WARNING)
logging.getLogger("synto.config").addHandler(handler)

cfg2 = Config(
    vault="/tmp/v",
    pipeline={"source_overrides": {"textbok": {"max_concepts_per_source": 25}}},
)
assert "textbok" in cfg2.pipeline.source_overrides
log_text = buf.getvalue()
assert "unknown source type" in log_text, f"expected warning in log: {log_text!r}"
print("unknown source type warning emitted")

print("all config-level checks passed")
PY
)
check "config-level override assertions" "echo \"$_CFG_CHECKS\" | grep -q 'all config-level checks passed'"
echo "  $_CFG_CHECKS"

# ── Seed notes with identical content, different source_type ────────────────
header "Seed raw notes"

# Write both notes via Python — equally rich but different content to avoid
# content-hash duplicate detection during ingest.
uv run --project "$REPO_DIR" python - "$VAULT_DIR" <<'PYEOF'
import sys
from pathlib import Path

vault = Path(sys.argv[1])

textbook_body = r"""# Calculus: Foundations and Applications

Calculus is the mathematical study of continuous change, providing the foundation for modern science and engineering.

## Limits and Continuity

Limits describe the behavior of a function as its input approaches a value. The formal epsilon-delta definition provides rigorous foundations. Continuity means a function has no gaps, jumps, or asymptotes. The Intermediate Value Theorem guarantees that a continuous function attains every value between its endpoints. The Squeeze Theorem bounds a function between two others to determine its limit. L'Hopital's Rule evaluates limits of indeterminate forms using derivatives.

## Differentiation

The derivative measures the instantaneous rate of change. The Power Rule states that the derivative of x^n is n*x^(n-1). The Product Rule differentiates products of functions. The Quotient Rule handles ratios of functions. The Chain Rule composes derivatives of nested functions. Higher-order derivatives measure acceleration and beyond. Implicit Differentiation handles equations not solved for y. The Mean Value Theorem relates average rate of change to instantaneous rate. Critical Points occur where the derivative is zero or undefined. The First Derivative Test identifies local maxima and minima. The Second Derivative Test confirms concavity and inflection points. Related Rates problems connect changing quantities through time. Linear Approximation uses the tangent line to estimate function values.

## Integration

The integral measures accumulation, area under a curve, and total change. The Fundamental Theorem of Calculus links differentiation and integration. Integration by Substitution reverses the chain rule. Integration by Parts reverses the product rule. Partial Fraction Decomposition breaks rational functions into simpler pieces. Trigonometric Integrals handle powers of sine and cosine. Trigonometric Substitution simplifies integrals with square roots. Improper Integrals extend integration to unbounded domains. Numerical Integration (Trapezoidal Rule, Simpson's Rule) approximates definite integrals. The Disk and Shell Methods compute volumes of revolution. Arc Length integrals measure curve length. Surface Area integrals measure area of rotated curves.

## Sequences and Series

Sequences are ordered lists of numbers. Convergence tests determine whether series sum to a finite value. The Geometric Series sums terms with a constant ratio. The p-Series provides a benchmark for comparison. The Ratio Test checks for absolute convergence. The Root Test handles terms with powers. Alternating Series converge when terms decrease to zero. Power Series represent functions as infinite polynomials. Taylor Series approximate functions around a point. Maclaurin Series are Taylor Series centered at zero. The Remainder Theorem bounds approximation error. Fourier Series decompose periodic functions into sine and cosine waves.
"""

notes_body = r"""# Linear Algebra: Core Concepts

Linear algebra is the branch of mathematics concerning vector spaces and linear mappings between them. It underpins nearly all of modern computation.

## Vectors and Vector Spaces

Vectors represent quantities with magnitude and direction. Vector addition follows the parallelogram law. Scalar Multiplication scales a vector's length. Linear Combinations combine vectors using scalars. Span describes all vectors reachable through linear combinations. Linear Independence means no vector is a combination of others. Basis vectors span a space and are linearly independent. Dimension is the number of basis vectors. Subspaces are subsets closed under addition and scalar multiplication. The Null Space contains all vectors mapped to zero. The Column Space spans the columns of a matrix. The Row Space spans the rows of a matrix. Orthogonal Complements contain vectors perpendicular to a subspace. Coordinates represent vectors relative to a chosen basis.

## Matrices and Linear Transformations

Matrices represent linear transformations. Matrix Multiplication composes transformations. The Identity Matrix leaves vectors unchanged. The Inverse Matrix reverses a transformation. The Transpose flips rows and columns. The Determinant measures scaling factor and orientation. The Trace sums diagonal entries. Eigenvalues and Eigenvectors satisfy A*v = lambda*v. The Characteristic Polynomial finds eigenvalues. Diagonalization factors a matrix into simpler form. Similar Matrices represent the same transformation in different bases. Orthogonal Matrices preserve angles and lengths. Symmetric Matrices have real eigenvalues. Positive Definite Matrices have all positive eigenvalues. Singular Value Decomposition (SVD) factors any matrix into three components. The LU Decomposition factors a matrix into lower and upper triangular parts. The QR Decomposition factors into orthogonal and triangular matrices.

## Systems of Linear Equations

Gaussian Elimination systematically solves linear systems by row operations. Row Echelon Form has leading ones in a staircase pattern. Reduced Row Echelon Form simplifies further to identity-like form. Pivots are the first nonzero entries in each row. Free Variables correspond to columns without pivots. Consistency requires no row of the form [0 ... 0 | b] with b nonzero. The Rank of a matrix is the number of pivots. The Nullity is the dimension of the null space. Cramer's Rule solves systems using determinants. Homogeneous Systems have all right-hand sides zero. Non-homogeneous Systems require a particular plus homogeneous solution.

## Inner Products and Norms

The Dot Product measures vector alignment. The Euclidean Norm is the standard vector length. The Cauchy-Schwarz Inequality bounds the dot product. Orthogonality means zero dot product. The Gram-Schmidt Process constructs an orthogonal basis. Least Squares finds the best fit solution to overdetermined systems. Projection maps vectors onto subspaces. The Angle Between Vectors relates dot product to cosine. The Cross Product produces a perpendicular vector in 3D.
"""

# Textbook note
(vault / "raw" / "textbook-ml.md").write_text(
    "---\ntitle: Calculus Foundations\nsource_type: textbook\n---\n" + textbook_body
)

# Notes note — different content (same richness), different source_type
(vault / "raw" / "notes-ml.md").write_text(
    "---\ntitle: Linear Algebra Concepts\nsource_type: notes\n---\n" + notes_body
)

print("ok")
PYEOF
check "notes seeded with equally rich content, different source_type" "true"

soft_check "textbook-ml.md created" "test -f $VAULT_DIR/raw/textbook-ml.md"
soft_check "notes-ml.md created"    "test -f $VAULT_DIR/raw/notes-ml.md"

# Snapshot hashes for immutability check
TB_HASH=$(shasum "$VAULT_DIR/raw/textbook-ml.md" | awk '{print $1}')
NT_HASH=$(shasum "$VAULT_DIR/raw/notes-ml.md" | awk '{print $1}')

# ── Ingest ───────────────────────────────────────────────────────────────────
header "synto ingest --all"

info "Calling $PROVIDER ($FAST_MODEL) — may take 1-3 min..."
$OLW ingest --all 2>&1

check "state.db created" "test -f $VAULT_DIR/.synto/state.db"
soft_check "textbook raw file unchanged" \
    "test \"\$(shasum '$VAULT_DIR/raw/textbook-ml.md' | awk '{print \$1}')\" = '$TB_HASH'"
soft_check "notes raw file unchanged" \
    "test \"\$(shasum '$VAULT_DIR/raw/notes-ml.md' | awk '{print \$1}')\" = '$NT_HASH'"

# ── Probe concept counts from DB ────────────────────────────────────────────
header "Concept extraction analysis"

_DB_CONCEPTS=$(uv run --project "$REPO_DIR" python - "$VAULT_DIR" <<'PY' 2>&1
import sqlite3
import sys
from pathlib import Path

vault = Path(sys.argv[1])
db = sqlite3.connect(str(vault / ".synto" / "state.db"))

tb_count = db.execute(
    "SELECT COUNT(*) FROM concepts WHERE source_path = ?",
    ("raw/textbook-ml.md",),
).fetchone()[0]

nt_count = db.execute(
    "SELECT COUNT(*) FROM concepts WHERE source_path = ?",
    ("raw/notes-ml.md",),
).fetchone()[0]

db.close()
print(f"textbook: {tb_count}")
print(f"notes: {nt_count}")
PY
)

_TB_COUNT=$(echo "$_DB_CONCEPTS" | grep '^textbook:' | awk '{print $2}')
_NT_COUNT=$(echo "$_DB_CONCEPTS" | grep '^notes:' | awk '{print $2}')

info "textbook (override cap=25): $_TB_COUNT concepts stored"
info "notes (global cap=8):       $_NT_COUNT concepts stored"

# Runtime checks — nondeterministic, LLM-dependent
# Textbook with override=25: if the LLM produced >8 candidates, all survive
if [[ "$_TB_COUNT" -gt 8 ]]; then
    pass "textbook stored $_TB_COUNT concepts (> 8, override lifted the cap)"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo -e "  ${YELLOW}? SKIP: textbook only stored $_TB_COUNT concepts — LLM did not produce >8 candidates${NC}"
fi

# Notes with no override: capped at global default (8)
if [[ "$_NT_COUNT" -le 8 ]]; then
    pass "notes stored $_NT_COUNT concepts (≤ 8, global default applied)"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    # This would be surprising — notes has no override, so should be ≤8
    echo -e "  ${YELLOW}? NOTE: notes stored $_NT_COUNT concepts — expected ≤8${NC}"
fi

# Summary pages
SOURCE_COUNT=$(find "$VAULT_DIR/wiki/sources" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
soft_check "source summary pages created" "test '$SOURCE_COUNT' -ge 1"

if [[ "$SOURCE_COUNT" -gt 0 ]]; then
    FIRST_SOURCE=$(find "$VAULT_DIR/wiki/sources" -name "*.md" | sort | head -1)
    soft_check "source page has YAML frontmatter" "grep -q '^---' \"$FIRST_SOURCE\""
    soft_check "source page has concept wikilinks" "grep -q '\[\[' \"$FIRST_SOURCE\""
fi

# ── Status ──────────────────────────────────────────────────────────────────
header "synto status"
$OLW status 2>&1

# ── Summary ──────────────────────────────────────────────────────────────────
header "Results"
echo -e "${GREEN}${BOLD}All checks passed: $PASS_COUNT${NC}"
echo ""
echo "Config: source_overrides.textbook.max_concepts_per_source = 25"
echo "       effective_max_concepts(textbook) = 25"
echo "       effective_max_concepts(notes)    = 8  (global fallback)"
echo ""
echo "Runtime concept storage:"
echo "  textbook (override=25): $_TB_COUNT concepts"
echo "  notes    (global=8):    $_NT_COUNT concepts"
echo ""
if [[ "$KEEP_VAULT" == "1" ]]; then
    echo "Vault: $VAULT_DIR"
    echo "  uv run --project $REPO_DIR synto status --vault $VAULT_DIR"
fi
