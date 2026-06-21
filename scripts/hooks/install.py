#!/usr/bin/env python3
"""
Install the pre-commit hook into .git/hooks/.

Run once from the CityLearn repo root:
    python scripts/hooks/install.py
"""
import shutil
import stat
import subprocess
import sys
from pathlib import Path

repo_root = Path(subprocess.run(
    ['git', 'rev-parse', '--show-toplevel'],
    capture_output=True, text=True,
).stdout.strip())

src  = repo_root / 'scripts' / 'hooks' / 'pre-commit'
dest = repo_root / '.git' / 'hooks' / 'pre-commit'

if dest.exists():
    print(f'Existing hook found at {dest} — backing up to {dest}.bak')
    shutil.copy2(dest, str(dest) + '.bak')

shutil.copy2(src, dest)
dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
print(f'[OK] pre-commit hook installed at {dest}')
print('     It will validate .ipynb syntax before every commit.')
