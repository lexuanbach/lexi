#!/usr/bin/env python3
"""Package hygiene: no build or cache files, no local paths and no credentials in any
released text file."""
from pathlib import Path
import re, sys

root = Path(__file__).resolve().parents[1]
bad_dirs = {'.git', '.hg', '.svn', '__pycache__', '.pytest_cache', '.mypy_cache', '.gocache',
            '.venv', '.venv_ad', 'venv', 'node_modules'}
skip_top = {'.git', '.venv', 'regenerated', 'figures'}
bad_suffixes = ('.aux', '.bbl', '.bcf', '.blg', '.fdb_latexmk', '.fls', '.glob', '.log', '.out',
                '.run.xml', '.synctex.gz', '.toc', '.vo', '.vok', '.vos', '.pyc')
bad_names = {'.DS_Store'}
# Local home or temporary paths, and credential formats. The path patterns are split so
# that this file does not match itself.
patterns = [
    re.compile(r'/' + r'Users/'), re.compile(r'/' + r'home/'), re.compile(r'Desk' + r'top/'),
    re.compile(r'/private' + r'/tmp'), re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}'),
    re.compile(r'hf_[A-Za-z0-9]{20,}'), re.compile(r'sk-[A-Za-z0-9]{20,}'),
    re.compile(r'AKIA[0-9A-Z]{16}'), re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY'),
]
bad, flagged = [], []
for p in root.rglob('*'):
    rel = p.relative_to(root)
    if rel.parts and rel.parts[0] in skip_top:
        continue
    if any(part in bad_dirs for part in rel.parts) or p.name in bad_names or \
            (p.is_file() and p.name.endswith(bad_suffixes)):
        bad.append(str(rel))
    if p.is_file() and p.stat().st_size <= 10_000_000:
        try:
            text = p.read_text(encoding='utf-8')
        except (UnicodeDecodeError, OSError):
            continue
        for pat in patterns:
            if pat.search(text):
                flagged.append(f'{rel}: {pat.pattern}')
if bad or flagged:
    for x in bad:
        print('FORBIDDEN:', x)
    for x in flagged:
        print('FLAGGED:', x)
    sys.exit(1)
print('PASS: package hygiene, local-path and credential checks')
