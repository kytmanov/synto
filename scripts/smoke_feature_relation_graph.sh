#!/usr/bin/env bash
# Smoke test for Features 26+27 (PR #109): relation extraction + concept graph, end to end
# against a live model. Covers: opt-in `pipeline.relation_extraction` ingest pass, the v29
# DB invariants (subject_key/object_key, endpoint-must-resolve, evidence/candidates),
# `relations:` article frontmatter, `synto find`, `synto trace term|relation|citation`,
# graph/graph.json in pack export (closed graph + capability), query-time graph expansion,
# and replace-on-reingest semantics. Runs standalone — does not depend on smoke_test.sh.
#
# Extracted concept and relation names are LLM-non-deterministic, so every assertion is
# outcome-agnostic: discover names from the DB rather than hardcoding them (see CLAUDE.md).
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
relation_extraction = true
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
relation_extraction = true
TOML
fi

# ── helpers ────────────────────────────────────────────────────────────────────
_RESULTS=()
_T0=$(date +%s)
PASS_COUNT=0
SOFT_FAIL_COUNT=0

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

soft_check() {
    local desc="$1"; shift
    local out rc=0
    out=$(set +o pipefail; eval "$@" 2>&1) || rc=$?
    if [[ $rc -eq 0 ]]; then
        pass "$desc"; PASS_COUNT=$((PASS_COUNT + 1))
        _RESULTS+=("PASS|$desc|")
    else
        SOFT_FAIL_COUNT=$((SOFT_FAIL_COUNT + 1))
        _RESULTS+=("SOFTFAIL|$desc|${out:0:1000}")
        echo -e "  ${YELLOW}⚠ SOFT FAIL: $desc${NC}${out:+$'\n'    ${out:0:400}}"
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

# ── Section A: seed notes with blatant relations + ingest (requires LLM) ───────
# Two self-contained notes; each states its relation in the headline sentence so a
# 4B-class fast model reliably extracts both the concepts and the relation between
# them. The relations>=1 check below is a HARD check on purpose: a relation smoke
# that extracts nothing from this text is a real regression, not model noise.
header "seed + ingest (relation_extraction = true)"
cat > "$VAULT_DIR/raw/raft_note.md" <<'EOF'
# Raft
Raft is a consensus algorithm for managing replicated logs. Raft depends on
Consensus among a majority of servers. Raft uses leader election, and Consensus
is required before any log entry is committed.
EOF
cat > "$VAULT_DIR/raw/vector_clocks_note.md" <<'EOF'
# Vector Clocks
Vector Clocks are logical clocks for distributed systems. Causal Consistency is
implemented by Vector Clocks: every replica tracks causality with a Vector Clock,
and Causal Consistency depends on comparing those clocks.
EOF

$OLW ingest --all 2>&1

check "ingest recorded concepts" \
  "uv run --project '$REPO_DIR' python3 - '$VAULT_DIR' <<'PY'
import sys
from pathlib import Path
from synto.state import StateDB
db = StateDB(Path(sys.argv[1]) / '.synto' / 'state.db')
names = db.list_all_concept_names()
assert names, 'no concepts extracted'
print(f'{len(names)} concepts: {names}')
PY"

# ── Section B: v29 DB invariants ───────────────────────────────────────────────
header "relations DB invariants"
check "relations table has >=1 row with valid keys and known endpoints" \
  "uv run --project '$REPO_DIR' python3 - '$VAULT_DIR' <<'PY'
import sys
from pathlib import Path
from synto.concept_text import concept_key
from synto.state import StateDB
db = StateDB(Path(sys.argv[1]) / '.synto' / 'state.db')
relations = db.list_relations()
assert relations, 'no relations extracted'
known = set(db.list_all_concept_names())
for r in relations:
    assert r['subject_key'] == concept_key(r['subject']), r
    assert r['object_key'] == concept_key(r['object']), r
    assert r['subject'] in known, f'unknown subject persisted: {r[\"subject\"]!r}'
    assert r['object'] in known, f'unknown object persisted: {r[\"object\"]!r}'
    assert 0.0 <= r['confidence'] <= 1.0, r
print(f'{len(relations)} relations, all endpoints known + keys consistent')
PY"

check "relation_evidence has >=1 row with non-empty evidence_text" \
  "uv run --project '$REPO_DIR' python3 - '$VAULT_DIR' <<'PY'
import sys
from pathlib import Path
from synto.state import StateDB
db = StateDB(Path(sys.argv[1]) / '.synto' / 'state.db')
rows = db._conn.execute('SELECT * FROM relation_evidence').fetchall()
assert rows, 'no relation evidence recorded'
assert any(r['evidence_text'].strip() for r in rows), 'all evidence_text empty'
assert all(r['source_segment_id'] for r in rows), 'evidence missing segment id'
print(f'{len(rows)} evidence rows')
PY"

check "relation_candidates audit log has >=1 row" \
  "uv run --project '$REPO_DIR' python3 - '$VAULT_DIR' <<'PY'
import sys
from pathlib import Path
from synto.state import StateDB
db = StateDB(Path(sys.argv[1]) / '.synto' / 'state.db')
rows = db._conn.execute('SELECT * FROM relation_candidates').fetchall()
assert rows, 'no raw candidates logged'
assert all(r['created_at'] for r in rows)
print(f'{len(rows)} candidate rows')
PY"

# ── Section C: compile + approve → relations frontmatter block ────────────────
header "compile + approve + relations frontmatter"
$OLW compile 2>&1
$OLW approve --all 2>&1

check "at least one wiki article published" \
  "find '$VAULT_DIR/wiki' -maxdepth 1 -name '*.md' ! -name 'index.md' ! -name 'log.md' | grep -q ."

# Outcome-agnostic: relation endpoints are canonical concepts and concept-driven
# compile writes one article per concept, so at least one published article must
# carry the top-10 relations block.
check "a published article carries a relations: frontmatter block" \
  "uv run --project '$REPO_DIR' python3 - '$VAULT_DIR' <<'PY'
import sys
from pathlib import Path
import frontmatter
wiki = Path(sys.argv[1]) / 'wiki'
hits = []
for f in wiki.glob('*.md'):
    if f.name in ('index.md', 'log.md'):
        continue
    rels = frontmatter.load(f).get('relations')
    if rels:
        assert isinstance(rels, list) and len(rels) <= 10, f
        for r in rels:
            assert set(r) == {'subject', 'predicate', 'object', 'confidence'}, r
        hits.append(f.name)
assert hits, 'no published article has a relations block'
print(f'relations block in: {hits}')
PY"

# ── Section D: synto find + synto trace term/relation/citation ────────────────
header "synto find + trace"
# Discover a relation endpoint (guaranteed to be a canonical concept, per Section B).
REL_SUBJECT=$(uv run --project "$REPO_DIR" python3 - "$VAULT_DIR" <<'PY'
import sys
from pathlib import Path
from synto.state import StateDB
db = StateDB(Path(sys.argv[1]) / '.synto' / 'state.db')
rels = sorted(db.list_relations(), key=lambda r: -r['confidence'])
print(rels[0]['subject'])
PY
)
REL_ID=$(uv run --project "$REPO_DIR" python3 - "$VAULT_DIR" <<'PY'
import sys
from pathlib import Path
from synto.state import StateDB
db = StateDB(Path(sys.argv[1]) / '.synto' / 'state.db')
rels = sorted(db.list_relations(), key=lambda r: -r['confidence'])
print(rels[0]['id'])
PY
)
REL_SEGMENT=$(uv run --project "$REPO_DIR" python3 - "$VAULT_DIR" <<'PY'
import sys
from pathlib import Path
from synto.state import StateDB
db = StateDB(Path(sys.argv[1]) / '.synto' / 'state.db')
row = db._conn.execute('SELECT source_segment_id FROM relation_evidence LIMIT 1').fetchone()
print(row[0])
PY
)
check "discovered relation subject, id, and evidence segment" \
  "test -n '$REL_SUBJECT' && test -n '$REL_ID' && test -n '$REL_SEGMENT'"

FIND_OUT=$($OLW find "$REL_SUBJECT" 2>&1); FIND_RC=$?
echo "$FIND_OUT"
check "synto find <concept> exits 0" "test $FIND_RC -eq 0"
check "synto find returns a match (not 'No articles found')" \
  "! grep -q 'No articles found' <<< \"\$FIND_OUT\""

TRACE_TERM_RC=0; $OLW trace term "$REL_SUBJECT" >/dev/null 2>&1 || TRACE_TERM_RC=$?
check "synto trace term exits 0" "test $TRACE_TERM_RC -eq 0"

TRACE_REL_OUT=$($OLW trace relation "$REL_ID" 2>&1); TRACE_REL_RC=$?
echo "$TRACE_REL_OUT"
check "synto trace relation exits 0" "test $TRACE_REL_RC -eq 0"
check "trace relation shows the subject and an arrow chain" \
  "grep -qF \"\$REL_SUBJECT\" <<< \"\$TRACE_REL_OUT\" && grep -q '→' <<< \"\$TRACE_REL_OUT\""
check "trace relation shows evidence table" \
  "grep -q 'Evidence' <<< \"\$TRACE_REL_OUT\""

TRACE_CIT_RC=0; $OLW trace citation "$REL_SEGMENT" >/dev/null 2>&1 || TRACE_CIT_RC=$?
check "synto trace citation exits 0 on a real evidence segment id" "test $TRACE_CIT_RC -eq 0"

# ── Section E: pack export → graph/graph.json ─────────────────────────────────
header "pack export graph"
PACK_OUT="$VAULT_DIR/.synto/exports/relation-graph-smoke"
rm -rf "$PACK_OUT"
PACK_RC=0; $OLW pack export --target agents --out "$PACK_OUT" 2>&1 || PACK_RC=$?
check "pack export exits 0" "test $PACK_RC -eq 0"
check "graph capability + closed graph.json consistent with concepts.json" \
  "uv run --project '$REPO_DIR' python3 - '$PACK_OUT' <<'PY'
import json
import sys
from pathlib import Path
from synto.concept_text import concept_key
pack = Path(sys.argv[1])
manifest = json.loads((pack / 'agent' / 'manifest.json').read_text(encoding='utf-8'))
assert 'graph' in manifest['pack']['capabilities'], manifest['pack']['capabilities']
graph = json.loads((pack / 'graph' / 'graph.json').read_text(encoding='utf-8'))
assert graph['schema_version'] == 1
assert graph['nodes'] and graph['edges'], 'graph must have nodes and edges'
node_ids = {n['id'] for n in graph['nodes']}
concepts = json.loads((pack / 'agent' / 'concepts.json').read_text(encoding='utf-8'))['concepts']
assert node_ids == {concept_key(c['name']) for c in concepts}
for e in graph['edges']:
    assert e['from_id'] in node_ids and e['to_id'] in node_ids, f'open edge: {e}'
print(f\"{len(node_ids)} nodes, {len(graph['edges'])} edges, closed graph\")
PY"

# ── Section F: query with graph expansion live (requires LLM) ─────────────────
header "query with graph expansion"
QUERY_OUT=$($OLW query "How does Raft achieve agreement across servers?" 2>&1); QUERY_RC=$?
echo "$QUERY_OUT"
check "synto query exits 0 with a populated relation graph" "test $QUERY_RC -eq 0"
check "query produced a non-empty answer" "test -n \"\$(printf '%s' \"\$QUERY_OUT\" | tr -d '[:space:]')\""
# Which pages the fast model selects is non-deterministic; expansion itself is
# unit-covered. Soft: just confirm the Sources footer rendered.
soft_check "query printed a Sources line" "grep -q 'Sources:' <<< \"\$QUERY_OUT\""

# ── Section G: re-ingest = replace (v29) ──────────────────────────────────────
header "re-ingest replaces relation artifacts"
sleep 1
WATERMARK=$(uv run --project "$REPO_DIR" python3 -c 'from datetime import datetime; print(datetime.now().isoformat())')
sleep 1
cat >> "$VAULT_DIR/raw/raft_note.md" <<'EOF'

Raft also depends on Consensus for configuration changes.
EOF
$OLW ingest --all 2>&1

check "re-ingest replaced raft_note candidates (all newer than watermark), left the other note's alone" \
  "uv run --project '$REPO_DIR' python3 - '$VAULT_DIR' '$WATERMARK' <<'PY'
import sys
from pathlib import Path
from synto.state import StateDB
db = StateDB(Path(sys.argv[1]) / '.synto' / 'state.db')
watermark = sys.argv[2]
rows = db._conn.execute('SELECT source_segment_id, created_at FROM relation_candidates').fetchall()
raft = [r for r in rows if r['source_segment_id'].startswith('note:raft_note:')]
other = [r for r in rows if r['source_segment_id'].startswith('note:vector_clocks_note:')]
# raft may legitimately be empty: candidate yield on the re-run is model-dependent
# (observed live), and cleared-with-no-new-rows still satisfies replace semantics.
# The invariant is that nothing from BEFORE the re-ingest survives.
stale = [dict(r) for r in raft if r['created_at'] < watermark]
assert not stale, f'stale candidates survived re-ingest: {stale}'
assert other, 'other note lost its candidates'
assert all(r['created_at'] < watermark for r in other), 'untouched note was cleared too'
print(f'{len(raft)} replaced, {len(other)} untouched')
PY"

check "synto status exits 0" "$OLW status"

# ── summary ────────────────────────────────────────────────────────────────────
header "Results"
echo -e "${BOLD}All checks passed: $PASS_COUNT${NC}"
if [[ $SOFT_FAIL_COUNT -gt 0 ]]; then
  echo -e "${YELLOW}Soft failures: $SOFT_FAIL_COUNT${NC}"
fi
echo ""
echo "Vault left at: $VAULT_DIR"
echo "  export SYNTO_VAULT=$VAULT_DIR"
