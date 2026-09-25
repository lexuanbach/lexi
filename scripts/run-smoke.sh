#!/usr/bin/env bash
# Fast wiring check (a few seconds): 3 methods x 3 seeds x 400 steps through the real code.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
cd "$ROOT/lexi"
./run_smoke.sh
