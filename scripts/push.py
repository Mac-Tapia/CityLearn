#!/usr/bin/env python3
"""
Validate + push the CityLearn submodule and bump the parent repo.

Steps:
  1. Run validate_notebook_syntax.py (syntax + Colab badge URL check)
  2. Push CityLearn to mac-tapia/citylearn-v3-madrl  ← what the badge points to
  3. Bump the submodule pointer in the parent repo (MADRLCitytleranflexresdr)
  4. Push the parent repo to origin/master

Usage (run from CityLearn/ or from the parent repo root):
    python CityLearn/scripts/push.py
    python CityLearn/scripts/push.py --skip-validate   # emergency only
    python CityLearn/scripts/push.py --dry-run
"""

import subprocess
import sys
from pathlib import Path

CITYLEARN_REMOTE  = 'mac-tapia'
CITYLEARN_BRANCH  = 'citylearn-v3-madrl'   # must match badge URL in validate_notebook_syntax.py
PARENT_REMOTE     = 'origin'
PARENT_BRANCH     = 'master'


def run(cmd: list[str], cwd: Path, dry: bool = False) -> int:
    label = ' '.join(str(c) for c in cmd)
    print(f'  $ {label}')
    if dry:
        print('    [dry-run — skipped]')
        return 0
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def find_repos() -> tuple[Path, Path]:
    """Return (citylearn_root, parent_root)."""
    here = Path(__file__).resolve().parent          # scripts/
    citylearn = here.parent                         # CityLearn/
    parent    = citylearn.parent                    # MADRLCitytleranflexresdr/

    if not (citylearn / '.git').exists():
        sys.exit(f'[push] ERROR: {citylearn} is not a git repo')
    if not (parent / '.git').exists():
        sys.exit(f'[push] ERROR: {parent} is not a git repo')
    return citylearn, parent


def main() -> int:
    dry          = '--dry-run'        in sys.argv
    skip_val     = '--skip-validate'  in sys.argv

    citylearn, parent = find_repos()
    validator = citylearn / 'scripts' / 'validate_notebook_syntax.py'

    # ── Step 1: validate ─────────────────────────────────────────────────────
    if skip_val:
        print('[push] WARNING: --skip-validate set, skipping validation.')
    else:
        print('\n[push] Step 1 — validating notebook syntax and badge URL...')
        rc = subprocess.run(
            [sys.executable, str(validator)], cwd=citylearn
        ).returncode
        if rc != 0:
            print('\n[push] BLOCKED: fix the errors above before pushing.')
            return 1
        print('[push] Validation passed.')

    # ── Step 2: push CityLearn to citylearn-v3-madrl ─────────────────────────
    print(f'\n[push] Step 2 — pushing CityLearn to '
          f'{CITYLEARN_REMOTE}/{CITYLEARN_BRANCH}...')

    # Check for uncommitted changes
    dirty = subprocess.run(
        ['git', 'status', '--porcelain'],
        cwd=citylearn, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        print(f'[push] ERROR: CityLearn has uncommitted changes:\n{dirty}')
        print('  Commit them first, then re-run push.py.')
        return 1

    rc = run(
        ['git', 'push', CITYLEARN_REMOTE,
         f'HEAD:{CITYLEARN_BRANCH}'],
        cwd=citylearn, dry=dry,
    )
    if rc != 0:
        print('[push] ERROR: git push CityLearn failed.')
        return rc

    # Verify the remote commit matches local HEAD
    if not dry:
        local_sha = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=citylearn, capture_output=True, text=True
        ).stdout.strip()
        remote_sha = subprocess.run(
            ['git', 'ls-remote', CITYLEARN_REMOTE, CITYLEARN_BRANCH],
            cwd=citylearn, capture_output=True, text=True
        ).stdout.split()[0] if subprocess.run(
            ['git', 'ls-remote', CITYLEARN_REMOTE, CITYLEARN_BRANCH],
            cwd=citylearn, capture_output=True, text=True
        ).stdout.strip() else ''

        if remote_sha and local_sha != remote_sha:
            print(f'[push] ERROR: remote SHA {remote_sha[:12]} != local {local_sha[:12]}')
            return 1
        print(f'[push] CityLearn remote up-to-date at {local_sha[:12]}.')

    # ── Step 3: bump submodule pointer in parent ──────────────────────────────
    print('\n[push] Step 3 — bumping submodule pointer in parent repo...')
    parent_dirty = subprocess.run(
        ['git', 'status', '--porcelain', 'CityLearn'],
        cwd=parent, capture_output=True, text=True
    ).stdout.strip()

    if not parent_dirty:
        print('[push] Parent submodule pointer already up-to-date, no bump needed.')
    else:
        rc = run(['git', 'add', 'CityLearn'], cwd=parent, dry=dry)
        if rc != 0:
            return rc

        local_sha = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=citylearn, capture_output=True, text=True
        ).stdout.strip()[:12]

        msg = (
            f'bump: CityLearn {CITYLEARN_BRANCH} {local_sha}\n\n'
            f'Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>'
        )
        if dry:
            print(f'  [dry-run] would commit: {msg!r}')
        else:
            rc = subprocess.run(
                ['git', 'commit', '-m', msg], cwd=parent
            ).returncode
            if rc != 0:
                print('[push] ERROR: could not commit submodule bump.')
                return rc

    # ── Step 4: push parent ────────────────────────────────────────────────────
    print(f'\n[push] Step 4 — pushing parent repo to '
          f'{PARENT_REMOTE}/{PARENT_BRANCH}...')
    rc = run(
        ['git', 'push', PARENT_REMOTE, f'HEAD:{PARENT_BRANCH}'],
        cwd=parent, dry=dry,
    )
    if rc != 0:
        print('[push] ERROR: git push parent failed.')
        return rc

    print('\n[push] Done. Colab badge now reflects the latest commit.')
    print(f'  https://colab.research.google.com/github/Mac-Tapia/CityLearn'
          f'/blob/{CITYLEARN_BRANCH}/examples/madrl_citylearn_v3_tutorial.ipynb')
    return 0


if __name__ == '__main__':
    sys.exit(main())
