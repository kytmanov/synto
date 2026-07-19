#!/usr/bin/env bash
# smoke_per_role_providers.sh — live end-to-end test of per-role providers (#24).
#
# The other smoke scripts use the legacy single-provider [provider] config. This one
# exercises the NEW named-provider-block format ([providers.<alias>] + [models.<role>])
# end-to-end against a real OpenAI-compatible backend, and ASSERTS — via the persisted
# metric_events — that the fast and heavy roles actually routed to their *distinct*
# configured models. That is the business logic of #24: a different model per role.
#
# Usage (defaults target LM Studio with two loaded models):
#   PROVIDER_NAME=lm_studio PROVIDER_URL=http://localhost:1234/v1 \
#     FAST_MODEL=google/gemma-4-e4b HEAVY_MODEL=nvidia/nemotron-3-nano-4b \
#     ./scripts/smoke_per_role_providers.sh
#
# Requirements: uv, and a backend serving BOTH models (distinct) at PROVIDER_URL.

set -euo pipefail

# Rich wraps CLI output at 80 cols when stdout is not a tty, which can split
# phrases that checks grep for. Pin a wide width so greps are deterministic.
export COLUMNS=200

PROVIDER_NAME="${PROVIDER_NAME:-lm_studio}"
PROVIDER_URL="${PROVIDER_URL:-http://localhost:1234/v1}"
FAST_MODEL="${FAST_MODEL:-google/gemma-4-e4b}"
HEAVY_MODEL="${HEAVY_MODEL:-nvidia/nemotron-3-nano-4b}"
CTX="${CTX:-8192}"

if [[ "$FAST_MODEL" == "$HEAVY_MODEL" ]]; then
    echo "ERROR: set FAST_MODEL and HEAVY_MODEL to two *different* models so per-role" >&2
    echo "       routing is observable." >&2
    exit 1
fi

GREEN=$'\033[0;32m'; RED=$'\033[0;31m'; BLUE=$'\033[0;34m'; NC=$'\033[0m'
PASS_COUNT=0
pass() { echo -e "${GREEN}✓${NC} $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { echo -e "${RED}✗${NC} $1"; [[ -n "${2:-}" ]] && echo "$2"; exit 1; }
header() { echo; echo -e "${BLUE}$1${NC}"; }
check() {
    local desc="$1"; shift
    local out rc=0
    out=$(set +o pipefail; eval "$@" 2>&1) || rc=$?
    if [[ $rc -eq 0 ]]; then pass "$desc"; else fail "$desc" "${out:0:800}"; fi
}

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLW="uv run --project $REPO_DIR synto"
VAULT="$(mktemp -d)"
trap 'rm -rf "$VAULT"' EXIT

header "Preflight (provider: $PROVIDER_NAME)"
check "backend reachable at $PROVIDER_URL" "curl -sf -m 5 '$PROVIDER_URL/models'"

header "Init + write NEW per-role config (two distinct models, one connection)"
$OLW init "$VAULT" >/dev/null 2>&1 || true
# Both roles share ONE provider connection (dedup) but use DIFFERENT models — the
# router must build a single client and route each role to its own model.
cat > "$VAULT/synto.toml" <<TOML
[providers.local]
name = "$PROVIDER_NAME"
url = "$PROVIDER_URL"
timeout = 600

[models.fast]
provider = "local"
model = "$FAST_MODEL"
ctx = $CTX

[models.heavy]
provider = "local"
model = "$HEAVY_MODEL"
ctx = $CTX

[pipeline]
auto_commit = false
auto_approve = false

[metrics]
# detailed=true persists a per-call metric_events row (model + role) instead of the
# default model-agnostic daily rollup — needed to assert which model each role used.
detailed = true
TOML
check "doctor resolves fast role to $FAST_MODEL" \
    "$OLW doctor --vault '$VAULT' | grep -qF 'fast: $FAST_MODEL'"
check "doctor resolves heavy role to $HEAVY_MODEL" \
    "$OLW doctor --vault '$VAULT' | grep -qF 'heavy: $HEAVY_MODEL'"

header "Seed + run pipeline (ingest=fast role, compile=heavy role)"
mkdir -p "$VAULT/raw"
cat > "$VAULT/raw/quantum.md" <<'MD'
# Quantum Computing

Quantum computing uses qubits, superposition, and entanglement to process information.
Shor's algorithm factors large integers efficiently. Grover's algorithm searches
unsorted data faster than classical methods. Quantum gates manipulate qubit states.
MD
check "ingest --all exits 0" "$OLW ingest --all --vault '$VAULT'"
check "compile exits 0" "$OLW compile --vault '$VAULT'"
check "at least one draft produced" \
    "test \$(find '$VAULT/wiki/.drafts' -name '*.md' 2>/dev/null | wc -l) -ge 1"

header "Per-role routing assertion (metric_events)"
# Business logic of #24: fast-role calls must hit the fast model, heavy-role calls the
# heavy model — and both must appear (proving the per-role split actually took effect).
ROUTE_OUT=$(SYNTO_VAULT="$VAULT" FAST="$FAST_MODEL" HEAVY="$HEAVY_MODEL" \
    uv run --project "$REPO_DIR" python - <<'PY' || true
import os, sqlite3, sys

vault, fast, heavy = os.environ["SYNTO_VAULT"], os.environ["FAST"], os.environ["HEAVY"]
con = sqlite3.connect(os.path.join(vault, ".synto", "state.db"))
rows = con.execute(
    "SELECT model, tier FROM metric_events WHERE event_type='llm_call' AND tier != ''"
).fetchall()
fast_models = sorted({m for m, t in rows if t == "fast"})
heavy_models = sorted({m for m, t in rows if t == "heavy"})
print(f"rows={len(rows)} fast_role={fast_models} heavy_role={heavy_models}")
problems = []
if not rows:
    problems.append("no llm_call metric rows recorded")
if fast_models != [fast]:
    problems.append(f"fast role used {fast_models}, expected ['{fast}']")
if heavy_models != [heavy]:
    problems.append(f"heavy role used {heavy_models}, expected ['{heavy}']")
if problems:
    print("FAILED: " + "; ".join(problems))
    sys.exit(1)
print("OK: per-role routing sent each role to its own model")
PY
)
if grep -q "^OK:" <<<"$ROUTE_OUT"; then
    pass "per-role routing: fast→$FAST_MODEL, heavy→$HEAVY_MODEL"
    echo "    $(grep '^rows=' <<<"$ROUTE_OUT")"
else
    fail "per-role routing did not split by model" "$ROUTE_OUT"
fi

header "Results"
echo -e "${GREEN}All checks passed: $PASS_COUNT${NC}"
