#!/usr/bin/env bash
# test_byte_parity_gate.sh — local harness for the byte-parity CI gate
# (VECTOR_SEARCH_PLAN §9.2 / §8.6).
#
# Proves the gate:
#   (A) Does NOT fire on the current real-world parity (no false positive).
#   (B) DOES fire when Python and Go diverge (catches real drift).
#
# Two scenarios run sequentially, both mirroring the exact same filter /
# sort / diff pipeline as the workflow files (.github/workflows/vector-
# parity.yml):
#
#   Scenario A — Real Python --self-test vs real Go fixture.
#                Expectations: empty diff (parity holds).
#
#   Scenario B — Monkeypatched Python (utf-8 → ascii/ignore) vs real Go.
#                Expectations: non-empty diff that includes at least the
#                bio line (multi-byte Spanish text → divergent hash).
#
# Exit code:
#   0  both scenarios behave as expected (gate is verified).
#   1  one or both scenarios misfire (gate cannot be trusted).
#
# Usage:
#   ./scripts/test_byte_parity_gate.sh                  # from helper/
#   ./scripts/test_byte_parity_gate.sh /path/to/helper  # anywhere
#
# Why we DON'T enable `set -o pipefail`:
#
#   `diff` legitimately exits 1 when files differ — that's how we know
#   drift exists. With pipefail on, any pipeline that contains `diff`
#   ALWAYS exits non-zero, so `if diff -u | grep -q PATTERN; then` would
#   fall through to the else branch even when grep matches. That
#   silently broke the bio-line diagnostic check in scenario B.
#
# Why we capture diff body into a variable instead of piping:
#
#   Two ways to express `if files-differ-with-bio-grep` are both wrong:
#     (a) if diff -u ... | grep -q BIO; then  — diff's exit-1 poisons
#         the pipeline under pipefail, condition is always FALSE.
#     (b) if diff -u ... && grep -q BIO; then  — && short-circuits
#         when diff returns 1 (always in scenario B), so grep never
#         runs, condition is always FALSE.
#   The correct pattern is: capture diff body into a variable while
#   swallowing diff's exit-1 with `|| true`, then inspect the string.
#
set -eu

HELPER_ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
echo "helper root: $HELPER_ROOT"

if [[ ! -f "$HELPER_ROOT/scripts/backfill_embeddings.py" ]]; then
  echo "FATAL: $HELPER_ROOT/scripts/backfill_embeddings.py not found" >&2
  exit 2
fi

# ── Locate backend (sibling repo, cached shallow clone, then fresh clone) ──
BACKEND_DIR="${BACKEND_DIR:-}"
if [[ -z "$BACKEND_DIR" && -d "$HELPER_ROOT/../backend" ]]; then
  BACKEND_DIR="$(cd "$HELPER_ROOT/../backend" && pwd)"
fi
if [[ -z "$BACKEND_DIR" && -d "/tmp/test_parity_backend" ]]; then
  BACKEND_DIR="/tmp/test_parity_backend"
fi
if [[ -z "$BACKEND_DIR" ]]; then
  echo "no sibling backend found; shallow-cloning from github…"
  BACKEND_DIR="/tmp/test_parity_backend"
  git clone --depth 1 https://github.com/HelpingPeopleNow/backend.git "$BACKEND_DIR"
fi
echo "backend root: $BACKEND_DIR"

if ! command -v go >/dev/null 2>&1; then
  echo "FATAL: 'go' not found on PATH — install Go 1.25 to run this harness" >&2
  exit 2
fi

GO_FIXTURE="${GO_FIXTURE:-/tmp/test_parity_hash_fixture}"
( cd "$BACKEND_DIR" && go build -o "$GO_FIXTURE" ./cmd/hash_fixture )
echo "go fixture compiled: $GO_FIXTURE"

# ── Pipeline mirrors the workflow's filter/sort/diff exactly ─────────────
canonical_lines() {
  # Two leading spaces + lowercase field name + pipe. Matches BOTH sides.
  grep -E '^  [a-z][a-z_-]+\|' | sort -u
}

PY_GOOD="$HELPER_ROOT/scripts/backfill_embeddings.py"
PY_GOOD_OUT="$(mktemp)"
PY_BAD_OUT="$(mktemp)"
GO_OUT="$(mktemp)"

# ── Scenario A: real Python vs Go ───────────────────────────────────────
echo
echo "=== Scenario A: REAL Python --self-test + Go fixture ==="
echo "    (must PASS — empty diff means current parity holds)"
python3 "$PY_GOOD" --self-test 2>/dev/null | canonical_lines > "$PY_GOOD_OUT"
"$GO_FIXTURE" 2>&1 | canonical_lines > "$GO_OUT"

scenario_a_ok=0
if diff -u "$PY_GOOD_OUT" "$GO_OUT" > /dev/null; then
  echo "  ✓ Scenario A: parity holds ($(wc -l < "$PY_GOOD_OUT") canonical lines, empty diff)"
  scenario_a_ok=1
else
  echo "  ✗ Scenario A: existing parity has drifted — gate would ALREADY be red!"
  diff -u "$PY_GOOD_OUT" "$GO_OUT" | head -20
fi

# ── Scenario B: monkeypatched Python (utf-8 → ascii/ignore) vs Go ───────
echo
echo "=== Scenario B: BUGGY Python (utf-8 → ascii/ignore) + Go fixture ==="
echo "    (must FAIL — non-empty diff means gate catches the drift)"
HERE="$HELPER_ROOT" python3 <<'EOF' 2>/dev/null | canonical_lines > "$PY_BAD_OUT"
import sys, os, importlib.util

helper_root = os.environ["HERE"]
script_path = os.path.join(helper_root, "scripts/backfill_embeddings.py")
spec = importlib.util.spec_from_file_location("real", script_path)
real = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real)

# DELIBERATE encoding drift: replace UTF-8 with ascii + "ignore".
# Production uses utf-8 (Go mirrors). With ascii/ignore, non-ASCII
# characters (Spanish ñ, ç, í, etc.) get stripped before hashing, so
# any fixture line with multi-byte text produces a divergent hash.
def buggy(text):
    import hashlib
    return hashlib.sha256(text.encode("ascii", "ignore")).hexdigest()
real.field_hash = buggy

# Drive --self-test through the public main() entrypoint so all internal
# callers (build_field_texts, the stability check, the UTF-8 byte-span
# line) flow through the patched hash function.
sys.argv = ["backfill_embeddings.py", "--self-test"]
try:
    real.main()
except SystemExit as exc:
    sys.exit(exc.code)
EOF

scenario_b_ok=0
# Capture diff body, swallowing diff's exit-1 (which is the signal we
# WANT in scenario B — not a script failure).
diff_body="$(diff -u "$PY_BAD_OUT" "$GO_OUT" 2>&1 || true)"

if [[ -z "$diff_body" ]]; then
  echo "  ✗ Scenario B: buggy Python MATCHED Go — gate did NOT catch the drift"
  echo "    this means the gate is broken or the drift injection missed"
else
  diff_lines=$(echo "$diff_body" | grep -cE '^[+-]  [a-z]')
  echo "  ✓ Scenario B: buggy Python diverged (gate correctly went red)"
  echo "  ✓   diff has $diff_lines changed canonical lines"

  # Diagnostic-specific assertion: the `bio` line of fixture 1 contains
  # "Electricista con 15 años de experiencia" — definitely multi-byte.
  # A sensible diff shows that line as diverging. If it's missing, the
  # diff is technically correct but harder for a developer to triage.
  #
  # Note: grep runs in BRE mode (no -E flag) so `|` after `bio` is a
  # literal pipe character — exactly what diff output contains.
  if echo "$diff_body" | grep -q '^[+-]  bio|'; then
    echo "  ✓   diff includes the bio line (developer-friendly diagnostic)"
    scenario_b_ok=1
  else
    echo "  ⚠   diff exists but bio line is absent — first 12 lines:"
    echo "$diff_body" | head -12
  fi
fi

# ── Scenario C: profession alias maps must be identical (P-A follow-up) ──
echo
echo "=== Scenario C: backend/internal/core/professions.json"
echo "                 vs helper/scripts/professions.json ==="
echo "    (must PASS — the two must be byte-for-byte equal)"
BACKEND_JSON="$BACKEND_DIR/internal/core/professions.json"
HELPER_JSON="$HELPER_ROOT/scripts/professions.json"

scenario_c_ok=0
if [[ ! -f "$BACKEND_JSON" ]]; then
  echo "  ✗ Scenario C: $BACKEND_JSON not found"
elif [[ ! -f "$HELPER_JSON" ]]; then
  echo "  ✗ Scenario C: $HELPER_JSON not found"
else
  if python3 - "$BACKEND_JSON" "$HELPER_JSON" <<'EOF'
import json, sys

def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

try:
    a = load(sys.argv[1])
    b = load(sys.argv[2])
except (OSError, ValueError) as exc:
    print(f"  ✗ Scenario C: failed to parse JSON: {exc}")
    sys.exit(1)

if a == b:
    print(f"  ✓ Scenario C: profession maps identical ({len(a)} aliases)")
    sys.exit(0)

# Report the precise divergence for easy triage.
only_a = {k: v for k, v in a.items() if k not in b}
only_b = {k: v for k, v in b.items() if k not in a}
diff_vals = {k: (a[k], b[k]) for k in a if k in b and a[k] != b[k]}
print("  ✗ Scenario C: profession maps DIVERGE")
if only_a:
    print(f"    only in backend: {only_a}")
if only_b:
    print(f"    only in helper:  {only_b}")
if diff_vals:
    print(f"    value mismatches: {diff_vals}")
sys.exit(1)
EOF
  then
    scenario_c_ok=1
  fi
fi

# ── Verdict ──────────────────────────────────────────────────────────────
echo
echo "=== result ==="
if (( scenario_a_ok == 1 )) && (( scenario_b_ok == 1 )) && (( scenario_c_ok == 1 )); then
  cat <<EOF

ALL SCENARIOS PASS
  ✓ Scenario A  current parity holds (no false positive)
  ✓ Scenario B  gate catches drift on multi-byte text (bio line)
  ✓ Scenario C  backend + helper profession maps are identical

The byte-parity CI gate is verified to behave as expected.
EOF
  exit 0
fi
echo "FAILED: A=$scenario_a_ok B=$scenario_b_ok C=$scenario_c_ok"
exit 1
