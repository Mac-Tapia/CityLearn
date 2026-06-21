#!/usr/bin/env python3
"""
Validate that every code cell in a Jupyter notebook is free of Python
syntax errors AND that the Colab badge URL is correct for the push target.

Checks performed:
  1. Literal newline inside string literal  (the specific Jupyter/VSCode
     autosave corruption that causes SyntaxError: unterminated string literal)
  2. Full ast.parse() on every code cell
  3. Colab badge URL present and branch matches COLAB_BRANCH

Usage:
    python scripts/validate_notebook_syntax.py [notebook.ipynb ...]
    python scripts/validate_notebook_syntax.py --branch citylearn-v3-madrl

Exit codes:
    0  all checks pass
    1  one or more checks fail
"""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

# The GitHub branch that the Colab badge must point to.
# Must match the branch we push to (see scripts/push.py).
COLAB_BRANCH   = 'citylearn-v3-madrl'
COLAB_REPO     = 'Mac-Tapia/CityLearn'
COLAB_NB_PATH  = 'examples/madrl_citylearn_v3_tutorial.ipynb'
EXPECTED_BADGE_URL = (
    f'https://colab.research.google.com/github/{COLAB_REPO}'
    f'/blob/{COLAB_BRANCH}/{COLAB_NB_PATH}'
)

_BADGE_RE = re.compile(
    r'https://colab\.research\.google\.com/github/[^)"\s]+'
)


# ---------------------------------------------------------------------------
# Check 1 — literal newline inside string literal
# ---------------------------------------------------------------------------

def _check_literal_newlines(source_list: list[str], cell_id: str) -> list[str]:
    errors = []
    for idx, entry in enumerate(source_list):
        body = entry.rstrip('\n')
        if '\n' not in body:
            continue
        for char_pos, ch in enumerate(body):
            if ch != '\n':
                continue
            before = body[:char_pos]
            sq = before.count("'") - before.count("\\'")
            dq = before.count('"') - before.count('\\"')
            if sq % 2 == 1 or dq % 2 == 1:
                errors.append(
                    f"  cell {cell_id!r} source[{idx}]: "
                    f"raw newline inside string literal at char {char_pos} "
                    f"— use '\\n' escape instead\n"
                    f"    entry: {entry!r}"
                )
    return errors


# ---------------------------------------------------------------------------
# Check 2 — ast.parse
# ---------------------------------------------------------------------------

def _check_ast(source: str, cell_id: str, cell_index: int) -> list[str]:
    try:
        ast.parse(source)
        return []
    except SyntaxError as exc:
        lines = source.split('\n')
        context = []
        if exc.lineno:
            lo = max(0, exc.lineno - 2)
            hi = min(len(lines), exc.lineno + 1)
            for i in range(lo, hi):
                marker = '>>>' if i + 1 == exc.lineno else '   '
                context.append(f"    {marker} {i+1:4d} | {lines[i]}")
        return [
            f"  cell {cell_id!r} (index {cell_index}): "
            f"SyntaxError line {exc.lineno}: {exc.msg}\n" + '\n'.join(context)
        ]


# ---------------------------------------------------------------------------
# Check 3 — Colab badge URL
# ---------------------------------------------------------------------------

def _check_badge(nb: dict, path: Path) -> list[str]:
    errors: list[str] = []
    found: list[str] = []

    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'markdown':
            continue
        src = ''.join(cell.get('source', []))
        for m in _BADGE_RE.findall(src):
            found.append(m)

    if not found:
        errors.append(
            f"  {path.name}: no Colab badge URL found.\n"
            f"  Expected: {EXPECTED_BADGE_URL}"
        )
        return errors

    for url in found:
        if url == EXPECTED_BADGE_URL:
            continue
        # Parse what's actually there
        m = re.match(
            r'https://colab\.research\.google\.com/github/([^/]+/[^/]+)'
            r'/blob/([^/]+)/(.+)', url
        )
        if not m:
            errors.append(f"  badge URL malformed: {url}")
            continue
        repo, branch, nb_path = m.group(1), m.group(2), m.group(3)
        if branch != COLAB_BRANCH:
            errors.append(
                f"  badge branch mismatch: found {branch!r}, "
                f"expected {COLAB_BRANCH!r}\n"
                f"  URL: {url}\n"
                f"  Fix: change to {EXPECTED_BADGE_URL}"
            )
        if repo != COLAB_REPO:
            errors.append(
                f"  badge repo mismatch: found {repo!r}, "
                f"expected {COLAB_REPO!r}"
            )

    return errors


# ---------------------------------------------------------------------------
# Per-notebook validation
# ---------------------------------------------------------------------------

def validate(path: Path, check_badge: bool = True) -> bool:
    try:
        nb = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        print(f"[FAIL] {path}: cannot parse JSON — {exc}", file=sys.stderr)
        return False

    all_errors: list[str] = []

    for ci, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') != 'code':
            continue
        cell_id   = cell.get('id', f'cell_{ci}')
        src_list  = cell.get('source', [])
        source    = ''.join(src_list)
        if not source.strip():
            continue
        all_errors.extend(_check_literal_newlines(src_list, cell_id))
        all_errors.extend(_check_ast(source, cell_id, ci))

    if check_badge:
        all_errors.extend(_check_badge(nb, path))

    if all_errors:
        print(f"[FAIL] {path} — {len(all_errors)} error(s):")
        for e in all_errors:
            print(e)
        return False

    code_cells = sum(1 for c in nb['cells'] if c.get('cell_type') == 'code')
    print(f"[OK]   {path} — {code_cells} code cells, badge OK, all clean")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    no_badge = '--no-badge' in sys.argv[1:]

    if not args:
        default = (
            Path(__file__).parent.parent
            / 'examples' / 'madrl_citylearn_v3_tutorial.ipynb'
        )
        paths = [default]
    else:
        paths = [Path(a) for a in args]

    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"[ERROR] not found: {p}", file=sys.stderr)
        return 1

    results = [validate(p, check_badge=not no_badge) for p in paths]
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
