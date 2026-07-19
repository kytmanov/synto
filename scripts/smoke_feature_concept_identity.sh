#!/usr/bin/env bash
# End-to-end smoke for Feature 45 (concept identity core): the entity_id identity layer and the
# curation commands built on it — merge, unmerge, split/homonym/disambiguation, the durability
# rebuild from the INDEX.json seed, the doctor identity section, identity lint checks, and
# inspect/keep. Runs standalone — does not depend on smoke_test.sh.
#
# Design: one real ingest+compile+approve proves the live pipeline + v18→v25 migration chain mint
# entities and bind entity_id end to end. The identity operations are then exercised against
# deterministically seeded entities/articles (via the public StateDB API, the same precedent as
# smoke_test.sh's repair section) so the assertions test the real merge/split/unmerge/lint/doctor
# code paths without depending on LLM-non-deterministic extraction output (see CLAUDE.md).
set -uo pipefail

# Rich wraps CLI output at 80 cols when stdout is not a tty, which can split
# phrases that checks grep for. Pin a wide width so greps are deterministic.
export COLUMNS=200

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
DB="$VAULT_DIR/.synto/state.db"
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

pass()   { echo -e "${GREEN}✓${NC} $1"; PASS_COUNT=$((PASS_COUNT + 1)); _RESULTS+=("PASS|$1|"); }
fail() {
    local desc="$1" detail="${2:-}"
    echo -e "  ${RED}✗ FAIL: $desc${NC}${detail:+$'\n'    ${detail:0:1200}}"
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

# check DESC CMD...  — runs CMD, hard-fails the whole smoke on non-zero (business-logic gate).
check() {
    local desc="$1"; shift
    local out rc=0
    out=$(set +o pipefail; eval "$@" 2>&1) || rc=$?
    if [[ $rc -eq 0 ]]; then
        pass "$desc"
    else
        _RESULTS+=("FAIL|$desc|${out:0:1200}")
        fail "$desc" "$out"
    fi
}

# pyassert DESC <<PY ... PY  — run a python snippet against the DB; exit 0 => pass.
pyassert() {
    local desc="$1"
    local out rc=0
    out=$(uv run --project "$REPO_DIR" python3 2>&1) || rc=$?
    if [[ $rc -eq 0 ]]; then
        pass "$desc"
    else
        _RESULTS+=("FAIL|$desc|${out:0:1200}")
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

# ═══════════════════════════════════════════════════════════════════════════════
# Section 0 — Live pipeline: prove ingest+compile+approve mint entities (end-user path)
# ═══════════════════════════════════════════════════════════════════════════════
header "live pipeline: ingest → compile → approve mints an entity"
cat > "$VAULT_DIR/raw/ml_note.md" <<'EOF'
# Gradient Descent
Gradient descent is an optimization algorithm that iteratively adjusts parameters
to minimize a loss function by following the negative gradient. Stochastic gradient
descent uses mini-batches for efficiency.
EOF

$OLW ingest --all 2>&1
$OLW compile 2>&1
$OLW approve --all 2>&1

check "at least one wiki article published" \
  "find '$VAULT_DIR/wiki' -maxdepth 1 -name '*.md' ! -name 'index.md' ! -name 'log.md' | grep -q ."

SYNTO_VAULT="$VAULT_DIR" DB="$DB" pyassert "published concepts carry a stable entity_id (migration chain v25 live)" <<'PY'
import os
from pathlib import Path
from synto.state import StateDB

db = StateDB(Path(os.environ["DB"]))
ver = db.schema_version()
assert ver >= 25, f"schema not migrated to v25+: {ver}"
names = db.list_all_concept_names()
assert names, "no concepts extracted from the live ingest"
with_id = [n for n in names if db.entity_id_for_name(n)]
assert with_id, f"no concept resolved to an entity_id: {names}"
# round-trip: name -> id -> preferred label resolves back to the same entity
eid = db.entity_id_for_name(with_id[0])
assert db.preferred_label_for_entity(eid), "entity has no preferred label"
print(f"v{ver}, {len(with_id)} concepts with entity_id")
PY

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Merge: identity move + article retire + link rewrite + alias absorb + log
# ═══════════════════════════════════════════════════════════════════════════════
header "concept merge (loser → winner)"
# Seed two real entities + published articles + an inbound link, deterministically.
SYNTO_VAULT="$VAULT_DIR" DB="$DB" VAULT="$VAULT_DIR" pyassert "seed two concepts + articles for merge" <<'PY'
import os
from pathlib import Path
from synto.state import StateDB
from synto.models import WikiArticleRecord

vault = Path(os.environ["VAULT"])
db = StateDB(Path(os.environ["DB"]))
db.upsert_concepts("raw/apple-inc.md", ["Apple Inc"])
db.upsert_concepts("raw/apple-computer.md", ["Apple Computer"])
assert db.entity_id_for_name("Apple Inc"), "winner entity not minted"
loser_eid = db.entity_id_for_name("Apple Computer")
assert loser_eid, "loser entity not minted"
assert db.entity_id_for_name("Apple Inc") != loser_eid, "winner/loser collapsed to one entity"

for name in ("Apple Inc", "Apple Computer"):
    p = vault / "wiki" / f"{name}.md"
    p.write_text(
        f"---\ntitle: {name}\nstatus: published\ntags: []\nsources: []\n---\n\nArticle body for {name}.\n",
        encoding="utf-8",
    )
    db.upsert_article(WikiArticleRecord(
        path=f"wiki/{name}.md", title=name, sources=[], content_hash="", status="published",
    ))

# A third published article links to the loser by its canonical name.
linker = vault / "wiki" / "Apple Linker.md"
linker.write_text(
    "---\ntitle: Apple Linker\nstatus: published\ntags: []\n---\n\nSee [[Apple Computer]] for history.\n",
    encoding="utf-8",
)
db.upsert_article(WikiArticleRecord(
    path="wiki/Apple Linker.md", title="Apple Linker", sources=[], content_hash="", status="published",
))
db.close()
print("seeded")
PY

# Dry-run must mutate nothing (Stage-0 bug guard: dry-run once called merge_entities and committed).
ENT_BEFORE=$(uv run --project "$REPO_DIR" python3 -c "
from pathlib import Path; from synto.state import StateDB
db=StateDB(Path('$DB')); print(db._conn.execute('SELECT COUNT(*) FROM concept_entities').fetchone()[0]); db.close()")
$OLW concept merge "Apple Computer" "Apple Inc" --dry-run 2>&1
ENT_AFTER_DRY=$(uv run --project "$REPO_DIR" python3 -c "
from pathlib import Path; from synto.state import StateDB
db=StateDB(Path('$DB')); print(db._conn.execute('SELECT COUNT(*) FROM concept_entities').fetchone()[0]); db.close()")
check "merge --dry-run mutates no entities" "test '$ENT_BEFORE' = '$ENT_AFTER_DRY'"
check "merge --dry-run leaves loser article on disk" "test -f '$VAULT_DIR/wiki/Apple Computer.md'"

# Real merge (confirmation prompt fed 'y').
MERGE_RC=0; printf 'y\n' | $OLW concept merge "Apple Computer" "Apple Inc" 2>&1 || MERGE_RC=$?
check "concept merge exits 0" "test $MERGE_RC -eq 0"
check "loser article removed from wiki/" "test ! -f '$VAULT_DIR/wiki/Apple Computer.md'"
check "loser article retired to .drafts/" \
  "find '$VAULT_DIR/wiki/.drafts' -name 'Apple Computer*retired*' 2>/dev/null | grep -q ."
check "winner article still present" "test -f '$VAULT_DIR/wiki/Apple Inc.md'"
check "winner frontmatter absorbed loser label as alias" \
  "grep -qi 'Apple Computer' '$VAULT_DIR/wiki/Apple Inc.md'"
check "inbound [[Apple Computer]] link repointed to winner" \
  "grep -qF '[[Apple Inc]]' '$VAULT_DIR/wiki/Apple Linker.md'"
check "no dangling [[Apple Computer]] link remains" \
  "! grep -rqF '[[Apple Computer]]' '$VAULT_DIR/wiki'"

SYNTO_VAULT="$VAULT_DIR" DB="$DB" pyassert "merge moved identity: loser entity merged, label resolves to winner" <<'PY'
import os
from pathlib import Path
from synto.state import StateDB

db = StateDB(Path(os.environ["DB"]))
winner = db.entity_id_for_name("Apple Inc")
assert winner, "winner entity vanished"
# 'Apple Computer' is now an alias of the winner (resolve_label returns the winner).
res = db.resolve_label("Apple Computer")
assert winner in res.ids, f"loser label no longer resolves to winner: {res.ids}"
ops = [r[0] for r in db._conn.execute("SELECT op FROM concept_identity_log").fetchall()]
assert "merge" in ops, f"no merge logged: {ops}"
print(f"winner={winner} ops={ops}")
PY

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Unmerge: documented best-effort reversal (entity/labels/edges + stub)
# ═══════════════════════════════════════════════════════════════════════════════
header "concept unmerge (best-effort reversal)"
UNMERGE_RC=0; $OLW concept unmerge "Apple Computer" 2>&1 || UNMERGE_RC=$?
check "concept unmerge exits 0" "test $UNMERGE_RC -eq 0"
check "unmerge recreated a stub article for the loser" "test -f '$VAULT_DIR/wiki/Apple Computer.md'"

SYNTO_VAULT="$VAULT_DIR" DB="$DB" pyassert "unmerge reactivated the loser entity and logged it" <<'PY'
import os
from pathlib import Path
from synto.state import StateDB

db = StateDB(Path(os.environ["DB"]))
loser = db.entity_id_for_name("Apple Computer")
assert loser, "loser entity not restored"
status = db._conn.execute(
    "SELECT status FROM concept_entities WHERE id=?", (loser,)
).fetchone()[0]
assert status == "active", f"restored entity not active: {status}"
ops = [r[0] for r in db._conn.execute("SELECT op FROM concept_identity_log").fetchall()]
assert "unmerge" in ops, f"no unmerge logged: {ops}"
print(f"loser={loser} status={status}")
PY

# Documented limitation: the body comes back as an EMPTY stub, not the original prose.
SYNTO_VAULT="$VAULT_DIR" VAULT="$VAULT_DIR" pyassert "unmerge stub body is empty (documented best-effort limitation)" <<'PY'
import os
from pathlib import Path
from synto.vault import parse_note

p = Path(os.environ["VAULT"]) / "wiki" / "Apple Computer.md"
_, body = parse_note(p)
assert body.strip() == "", f"expected empty stub body, got: {body!r}"
print("empty stub as documented")
PY

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Split: homonym senses + disambiguation stub
# ═══════════════════════════════════════════════════════════════════════════════
header "concept split (homonym + disambiguation stub)"
SYNTO_VAULT="$VAULT_DIR" DB="$DB" VAULT="$VAULT_DIR" pyassert "seed one entity with two sources for split" <<'PY'
import os
from pathlib import Path
from synto.state import StateDB
from synto.models import WikiArticleRecord

vault = Path(os.environ["VAULT"])
db = StateDB(Path(os.environ["DB"]))
# Same label across two sources => one entity, two source edges.
db.upsert_concepts("raw/mercury-planet.md", ["Mercury"])
db.upsert_concepts("raw/mercury-element.md", ["Mercury"])
eid = db.entity_id_for_name("Mercury")
assert eid, "Mercury entity not minted"
srcs = db.get_sources_for_concept("Mercury")
assert len(srcs) >= 2, f"expected >=2 sources, got {srcs}"

p = vault / "wiki" / "Mercury.md"
p.write_text(
    "---\ntitle: Mercury\nstatus: published\ntags: []\nsources: []\n---\n\nMercury bare article.\n",
    encoding="utf-8",
)
db.upsert_article(WikiArticleRecord(
    path="wiki/Mercury.md", title="Mercury", sources=srcs, content_hash="", status="published",
))
db.close()
print(f"seeded Mercury with {len(srcs)} sources")
PY

SPLIT_RC=0; $OLW concept split "Mercury" \
  --sense "Mercury (planet)" "raw/mercury-planet.md" \
  --sense "Mercury (element)" "raw/mercury-element.md" 2>&1 || SPLIT_RC=$?
check "concept split exits 0" "test $SPLIT_RC -eq 0"
check "planet sense article created" "test -f '$VAULT_DIR/wiki/Mercury (planet).md'"
check "element sense article created" "test -f '$VAULT_DIR/wiki/Mercury (element).md'"
check "disambiguation stub created at bare label" "test -f '$VAULT_DIR/wiki/Mercury.md'"
check "disambiguation stub marked kind: disambiguation" \
  "grep -qi 'kind: disambiguation' '$VAULT_DIR/wiki/Mercury.md'"
check "disambiguation stub lists both senses" \
  "grep -qF '[[Mercury (planet)]]' '$VAULT_DIR/wiki/Mercury.md' && grep -qF '[[Mercury (element)]]' '$VAULT_DIR/wiki/Mercury.md'"

SYNTO_VAULT="$VAULT_DIR" DB="$DB" pyassert "bare label now resolves to BOTH senses (ambiguous) and split logged" <<'PY'
import os
from pathlib import Path
from synto.state import StateDB

db = StateDB(Path(os.environ["DB"]))
res = db.resolve_label("Mercury")
assert res.ambiguous and len(res.ids) >= 2, f"bare label not ambiguous: {res.ids}"
planet = db.entity_id_for_name("Mercury (planet)")
element = db.entity_id_for_name("Mercury (element)")
assert planet and element and planet != element, "senses not distinct entities"
assert {planet, element} <= set(res.ids), f"senses not both candidates for bare label: {res.ids}"
ops = [r[0] for r in db._conn.execute("SELECT op FROM concept_identity_log").fetchall()]
assert "split" in ops, f"no split logged: {ops}"
print(f"candidates={res.ids}")
PY

# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Inspect + keep (ambiguous-occurrence drain)
# ═══════════════════════════════════════════════════════════════════════════════
header "concept inspect + keep"
INSPECT_OUT=$($OLW concept inspect "Mercury (planet)" 2>&1) || true
echo "$INSPECT_OUT"
_TMP=$(mktemp); echo "$INSPECT_OUT" > "$_TMP"
check "inspect prints the entity_id" "grep -qi 'entity_id' '$_TMP'"
rm -f "$_TMP"

# Seed an ambiguous occurrence of 'Mercury' and drain it onto the planet sense.
SYNTO_VAULT="$VAULT_DIR" DB="$DB" pyassert "seed an ambiguous occurrence of 'Mercury'" <<'PY'
import os
from pathlib import Path
from synto.state import StateDB

db = StateDB(Path(os.environ["DB"]))
db._conn.execute(
    "INSERT INTO concept_occurrences (concept_name, source_path, surface, resolution_status)"
    " VALUES ('Mercury', 'raw/mercury-planet.md', 'Mercury', 'ambiguous')"
)
db._conn.commit()
n = db._conn.execute(
    "SELECT COUNT(*) FROM concept_occurrences WHERE resolution_status='ambiguous'"
).fetchone()[0]
assert n >= 1
print("ambiguous occurrence seeded")
PY

KEEP_RC=0; KEEP_OUT=$($OLW concept keep "Mercury" "Mercury (planet)" 2>&1) || KEEP_RC=$?
echo "$KEEP_OUT"
check "concept keep exits 0" "test $KEEP_RC -eq 0"
SYNTO_VAULT="$VAULT_DIR" DB="$DB" pyassert "keep resolved the ambiguous occurrence onto the chosen sense" <<'PY'
import os
from pathlib import Path
from synto.state import StateDB

db = StateDB(Path(os.environ["DB"]))
amb = db._conn.execute(
    "SELECT COUNT(*) FROM concept_occurrences WHERE concept_name='Mercury' AND resolution_status='ambiguous'"
).fetchone()[0]
assert amb == 0, f"ambiguous occurrence not drained: {amb} left"
planet = db.entity_id_for_name("Mercury (planet)")
resolved = db._conn.execute(
    "SELECT COUNT(*) FROM concept_occurrences WHERE entity_id=? AND resolution_status='resolved'",
    (planet,),
).fetchone()[0]
assert resolved >= 1, "no occurrence bound to the chosen sense"
print("drained")
PY

# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Identity lint + doctor (match_key dedup worklist, issue #54)
# ═══════════════════════════════════════════════════════════════════════════════
header "lint + doctor identity checks (match_key collision, issue #54)"
# Seed a plural/singular fold collision: User vs Users (match_key collide, label_key distinct).
SYNTO_VAULT="$VAULT_DIR" DB="$DB" pyassert "seed User/Users fold collision" <<'PY'
import os
from pathlib import Path
from synto.state import StateDB

db = StateDB(Path(os.environ["DB"]))
db.upsert_concepts("raw/user.md", ["User"])
db.upsert_concepts("raw/users.md", ["Users"])
cols = db.find_match_key_collisions()
labels = {tuple(sorted((db.preferred_label_for_entity(a), db.preferred_label_for_entity(b)))) for a, b, _ in cols}
assert ("User", "Users") in labels, f"User/Users not detected as a collision: {cols}"
print("collision seeded")
PY

# `synto maintain` (no --fix) runs run_lint and reports identity health issues.
LINT_OUT=$($OLW maintain 2>&1) || true
echo "$LINT_OUT" | head -40
_TMP=$(mktemp); echo "$LINT_OUT" > "$_TMP"
check "lint reports the label/match_key collision" \
  "grep -qiE 'label_collision|match_key|share match_key|User.*Users' '$_TMP'"
rm -f "$_TMP"

DOCTOR_OUT=$($OLW doctor 2>&1) || true
_TMP=$(mktemp); echo "$DOCTOR_OUT" > "$_TMP"
check "doctor renders the Concept identity section" "grep -qi 'Concept identity' '$_TMP'"
check "doctor reports active entities count" "grep -qiE '[0-9]+ active entities' '$_TMP'"
check "doctor surfaces the match_key dedup worklist" "grep -qi 'match_key collision' '$_TMP'"
rm -f "$_TMP"

# concept inspect should surface the User/Users merge suggestion (issue #54 detection layer).
INSPECT2=$($OLW concept inspect "User" 2>&1) || true
echo "$INSPECT2"
_TMP=$(mktemp); echo "$INSPECT2" > "$_TMP"
check "inspect surfaces a merge suggestion for the fold duplicate" \
  "grep -qiE 'merge with|Merge suggestion' '$_TMP'"
rm -f "$_TMP"

# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — Durability: lossless rebuild from the committed INDEX.json seed
# ═══════════════════════════════════════════════════════════════════════════════
header "durability: rebuild identity from INDEX.json seed (decision 13)"
# Merge/split already ran generate_index, so .synto/INDEX.json carries entity ids + the log.
check "INDEX.json seed exists" "test -f '$VAULT_DIR/.synto/INDEX.json'"
check "INDEX.json carries entity_id" "grep -q 'entity_id' '$VAULT_DIR/.synto/INDEX.json'"

# Snapshot an entity id + the log size, then destroy state.db (the gitignored layer).
WINNER_ID_BEFORE=$(uv run --project "$REPO_DIR" python3 -c "
from pathlib import Path; from synto.state import StateDB
db=StateDB(Path('$DB')); print(db.entity_id_for_name('Apple Inc') or ''); db.close()")
LOG_BEFORE=$(uv run --project "$REPO_DIR" python3 -c "
from pathlib import Path; from synto.state import StateDB
db=StateDB(Path('$DB')); print(db._conn.execute('SELECT COUNT(*) FROM concept_identity_log').fetchone()[0]); db.close()")
check "captured a winner entity_id before rebuild" "test -n '$WINNER_ID_BEFORE'"

rm -f "$DB"
RECON_OUT=$($OLW doctor --reconcile 2>&1) || true
_TMP=$(mktemp); echo "$RECON_OUT" > "$_TMP"
check "doctor --reconcile restores entities from the seed" \
  "grep -qiE 'reconcile: restored [0-9]+ entit' '$_TMP'"
rm -f "$_TMP"

SYNTO_VAULT="$VAULT_DIR" DB="$DB" WINNER_ID_BEFORE="$WINNER_ID_BEFORE" LOG_BEFORE="$LOG_BEFORE" \
  pyassert "rebuild is lossless: same entity_id + identity log restored" <<'PY'
import os
from pathlib import Path
from synto.state import StateDB

db = StateDB(Path(os.environ["DB"]))
after = db.entity_id_for_name("Apple Inc")
assert after == os.environ["WINNER_ID_BEFORE"], (
    f"entity_id changed on rebuild: {after} != {os.environ['WINNER_ID_BEFORE']}"
)
log_after = db._conn.execute("SELECT COUNT(*) FROM concept_identity_log").fetchone()[0]
assert log_after >= int(os.environ["LOG_BEFORE"]), (
    f"identity log not restored: {log_after} < {os.environ['LOG_BEFORE']}"
)
print(f"id stable={after}, log={log_after}")
PY

# Precedence: with state.db live again and the seed present, doctor reports a match (no overwrite).
DOCTOR2=$($OLW doctor 2>&1) || true
_TMP=$(mktemp); echo "$DOCTOR2" > "$_TMP"
check "doctor confirms identity matches the seed (precedence: state.db wins)" \
  "grep -qiE 'identity matches the INDEX.json seed|whose entity_id differs from the seed' '$_TMP'"
rm -f "$_TMP"

# ── summary ────────────────────────────────────────────────────────────────────
header "Results"
echo -e "${BOLD}All checks passed: $PASS_COUNT${NC}"
echo ""
echo "Vault left at: $VAULT_DIR"
echo "  export SYNTO_VAULT=$VAULT_DIR"
