#!/usr/bin/env bash
# Verify the package without re-running experiments (a few seconds): file integrity,
# package hygiene, every paper claim against the committed results, and the paper PDFs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
PY="${PYTHON:-python3}"
# Bytecode caches left by a direct python3 call are not part of the release.
find lexi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
"$PY" scripts/check-manifest.py
"$PY" scripts/check-hygiene.py
"$PY" scripts/check-claims.py
for pdf in paper/camera-ready.pdf paper/extended.pdf; do
  test -s "$pdf"
done
if command -v pdfinfo >/dev/null 2>&1; then
  pages="$(pdfinfo paper/camera-ready.pdf | awk '/^Pages:/{print $2}')"
  test "$pages" -le 10
  echo "PASS: camera-ready PDF has $pages pages"
else
  echo "SKIPPED: pdfinfo unavailable, PDF existence was checked"
fi
echo "PASS: artifact verification complete"
