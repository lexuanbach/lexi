#!/usr/bin/env python3
"""Check (default) or rewrite (--write) MANIFEST.sha256.

The manifest lists the SHA-256 digest of every file of the package except the manifest
itself, as "digest  relative/path" lines sorted by path. Three top-level folders are not
part of the release and are skipped: regenerated/ (written by run-full.sh and
run-extended.sh), figures/ (the default output folder of the figure scripts) and .venv/
(a virtual environment created as in README.md).
"""
from pathlib import Path
import hashlib, sys

root = Path(__file__).resolve().parents[1]
manifest = root / 'MANIFEST.sha256'
SKIP_TOP = {'.git', '.venv', 'regenerated', 'figures'}


def released_files():
    out = []
    for p in root.rglob('*'):
        rel = p.relative_to(root)
        if p.is_file() and p != manifest and rel.parts[0] not in SKIP_TOP:
            out.append(rel.as_posix())
    return sorted(out)


def digest(rel):
    return hashlib.sha256((root / rel).read_bytes()).hexdigest()


if '--write' in sys.argv[1:]:
    files = released_files()
    manifest.write_text(''.join(f'{digest(f)}  {f}\n' for f in files), encoding='utf-8')
    print(f'wrote MANIFEST.sha256 with {len(files)} entries')
    sys.exit(0)

expected = {}
for line in manifest.read_text(encoding='utf-8').splitlines():
    if not line.strip():
        continue
    d, rel = line.split('  ', 1)
    expected[rel] = d
actual_files = set(released_files())
ok = actual_files == set(expected)
if not ok:
    for x in sorted(actual_files - set(expected)):
        print('UNMANIFESTED:', x)
    for x in sorted(set(expected) - actual_files):
        print('MISSING:', x)
for rel, d in sorted(expected.items()):
    if (root / rel).is_file() and digest(rel) != d:
        print('HASH MISMATCH:', rel)
        ok = False
print(f"{'PASS' if ok else 'FAIL'}: {len(expected)} manifest entries")
sys.exit(0 if ok else 1)
