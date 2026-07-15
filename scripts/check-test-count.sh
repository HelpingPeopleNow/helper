#!/usr/bin/env bash
# check-test-count.sh — fails if the helper/AGENTS.md test count claim
# drifts from the actual `def test_` / `async def test_` count in helper/tests/*.py.
#
# Usage:  ./scripts/check-test-count.sh
# Setup:  run from helper/ as the working directory; chmod +x.
# CI:     wire into helper/.github/workflows/docker.yml alongside the existing
#         pytest step so a drift fails the PR-build before any image push.

set -euo pipefail

# Resolve to helper/ regardless of where invoked from.
cd "$(dirname "$0")/.."

# Sanity-check repo layout before counting. A clean checkout without tests/
# would produce ACTUAL=0 forever, masking real drift instead of reporting it.
if [ ! -d tests ]; then
  echo "ERROR: helper/tests/ directory not found at $(pwd)/tests" >&2
  exit 2
fi

# Pull the canonical claimed count out of AGENTS.md, looking for `**NNN pytest tests`.
# `|| true` swallows the pipefail abort when the inner grep has no match — the
# downstream `[ -z "$CLAIMED" ]` check is the actual guard.
CLAIMED=$(grep -oE '\*\*[0-9]+ pytest tests' AGENTS.md | head -1 | grep -oE '[0-9]+' || true)

if [ -z "${CLAIMED:-}" ]; then
  echo "ERROR: no '**NNN pytest tests' claim found in helper/AGENTS.md" >&2
  exit 1
fi

# Count actual `def test_` and `async def test_` functions across tests/*.py.
# `awk '{print $1}'` strips both the leading whitespace and the trailing newline
# that `wc -l` leaves behind (so `$ACTUAL` is a clean integer for `-ne`).
ACTUAL=$(grep -rE '^\s*(async )?def test_' tests/*.py 2>/dev/null | wc -l | awk '{print $1}')

if [ "$ACTUAL" -ne "$CLAIMED" ]; then
  echo "TEST COUNT DRIFT: helper/AGENTS.md claims $CLAIMED pytest tests, found $ACTUAL." >&2
  echo "" >&2
  echo "Per-file breakdown from the filesystem:" >&2
  grep -rcE '^\s*(async )?def test_' tests/*.py 2>/dev/null | sort >&2
  echo "" >&2
  echo "To fix, update helper/AGENTS.md 'Key facts' line and the 'Test files live in' breakdown." >&2
  exit 1
fi

echo "OK ($ACTUAL pytest tests, matches helper/AGENTS.md claim)"
