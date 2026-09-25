#!/usr/bin/env python3
"""Compare regenerated result files with the released ones, key by key.

Usage:
    python3 scripts/compare-results.py RELEASED_DIR REGENERATED_DIR [FILE ...]

Both directories hold result JSON files with the same names. Without FILE arguments every
*.json file present in both directories is compared. For each file the script reports
each top-level key as SAME or DIFF. Keys that hold wall-clock timings depend on the
machine and are reported as TIMING without failing:
    core_results.json      scalability, _meta (runtime_s)
    robustness_results.json    rx5_scaling, _meta
    deployment_results.json   _meta
A file that exists only in the released directory is reported as NOT REGENERATED.
Exit status 0 means that every compared key other than a timing key is identical.
"""
import json, sys
from pathlib import Path

TIMING = {
    "core_results.json": {"scalability", "_meta"},
    "robustness_results.json": {"rx5_scaling", "_meta"},
    "deployment_results.json": {"_meta"},
}


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    old, new = Path(argv[1]), Path(argv[2])
    names = argv[3:] or sorted(p.name for p in old.glob("*.json"))
    ok = True
    for name in names:
        a, b = old / name, new / name
        if not b.exists():
            print(f"  NOT REGENERATED  {name}")
            continue
        ja, jb = json.loads(a.read_text()), json.loads(b.read_text())
        if a.read_bytes() == b.read_bytes():
            print(f"  IDENTICAL        {name} (byte for byte)")
            continue
        for key in sorted(set(ja) | set(jb)):
            same = ja.get(key) == jb.get(key)
            if same:
                tag = "SAME"
            elif key in TIMING.get(name, ()):
                tag = "TIMING"
            else:
                tag = "DIFF"
                ok = False
            print(f"  {tag:16} {name} : {key}")
    print("PASS: regenerated results equal the released ones (timing keys excepted)"
          if ok else "FAIL: some regenerated values differ from the released ones")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
