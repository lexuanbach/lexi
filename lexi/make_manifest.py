#!/usr/bin/env python3
"""
SHA-256 manifest of the committed result files and the real-data inputs.

The manifest lets a reader check the provenance of every number without re-running the
experiments. It covers ../results/*.json, which holds the measured outputs of the
experiment scripts that back Tables 1 and 3 and Fig. 2, and data/*, the real UK carbon and
Azure load traces. The manifest is written to ../MANIFEST.sha256. The script is
deterministic and uses only the standard library.

    python3 make_manifest.py          # write ../MANIFEST.sha256
    python3 make_manifest.py --check  # verify every listed file against its digest

The exit status of --check is 0 when all files match and 1 otherwise. It is an
integrity check on files and does not compare values with the paper. That is done by
check_results.py.
"""
from __future__ import annotations
import hashlib, os, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "..", "MANIFEST.sha256")


def _targets():
    files = sorted(glob.glob(os.path.join(HERE, "..", "results", "*.json")))
    files += sorted(glob.glob(os.path.join(HERE, "data", "*")))
    return [f for f in files if os.path.isfile(f)]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: str) -> str:
    return os.path.relpath(path, os.path.join(HERE, "..")).replace(os.sep, "/")


def generate() -> int:
    lines = [f"{_sha256(f)}  {_rel(f)}" for f in _targets()]
    with open(MANIFEST, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.relpath(MANIFEST, HERE)} with {len(lines)} entries")
    return 0


def check() -> int:
    if not os.path.exists(MANIFEST):
        print("MANIFEST.sha256 not found; run without --check to generate it")
        return 1
    root = os.path.join(HERE, "..")
    bad = missing = 0
    n = 0
    with open(MANIFEST) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n += 1
            digest, rel = line.split("  ", 1)
            path = os.path.join(root, rel)
            if not os.path.exists(path):
                print(f"[MISSING] {rel}"); missing += 1; continue
            if _sha256(path) != digest:
                print(f"[MISMATCH] {rel}"); bad += 1
    if bad or missing:
        print(f"RESULT: FAIL -- {bad} mismatched, {missing} missing of {n}")
        return 1
    print(f"RESULT: PASS -- all {n} files match MANIFEST.sha256")
    return 0


if __name__ == "__main__":
    sys.exit(check() if "--check" in sys.argv[1:] else generate())
