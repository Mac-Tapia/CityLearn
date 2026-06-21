#!/usr/bin/env python3
"""
Validate that every code cell in a Jupyter notebook is free of Python
syntax errors, including the specific 'literal newline inside string
literal' corruption that Jupyter/VSCode autosave can introduce.

Usage:
    python scripts/validate_notebook_syntax.py [notebook.ipynb ...]

Exit codes:
    0  all cells pass
    1  one or more cells have errors
"""

import ast
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check_literal_newlines(source_list: list[str], cell_id: str) -> list[str]:
    """
    Detect source-list entries that contain a raw newline character (chr 10)
    *inside* a string literal — the specific corruption Jupyter editors cause
    when they convert the escape sequence \\n to a real newline.

    A source-list entry normally ends with exactly one \\n (its line
    terminator).  Any \\n that appears before the final character means the
    entry spans multiple logical lines, which is only valid for multi-line
    constructs (triple-quoted strings, parenthesised expressions, etc.).
    We flag entries where such an embedded \\n appears to be inside a
    single-quoted string.
    """
    errors = []
    for idx, entry in enumerate(source_list):
        # Strip the trailing newline (legal line terminator)
        body = entry.rstrip('\n')
        # Any remaining chr(10) is an embedded newline — suspicious
        if '\n' not in body:
            continue
        # Quick heuristic: is there an odd number of single-quotes before
        # the embedded newline?  That signals an open string literal.
        for char_pos, ch in enumerate(body):
            if ch != '\n':
                continue
            before = body[:char_pos]
            # Count unescaped single-quotes; odd count → open string
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


def _check_ast(source: str, cell_id: str, cell_index: int) -> list[str]:
    """Run ast.parse on the full joined source; return error strings."""
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
        context_str = '\n'.join(context)
        return [
            f"  cell {cell_id!r} (index {cell_index}): "
            f"SyntaxError line {exc.lineno}: {exc.msg}\n{context_str}"
        ]


# ---------------------------------------------------------------------------
# Per-notebook validation
# ---------------------------------------------------------------------------

def validate(path: Path) -> bool:
    """Return True if notebook is clean, False if errors found."""
    try:
        nb = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        print(f"[FAIL] {path}: cannot parse JSON — {exc}", file=sys.stderr)
        return False

    all_errors: list[str] = []

    for ci, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') != 'code':
            continue
        cell_id = cell.get('id', f'cell_{ci}')
        source_list: list[str] = cell.get('source', [])
        source: str = ''.join(source_list)

        if not source.strip():
            continue

        # Check 1: literal newlines embedded inside string literals
        all_errors.extend(_check_literal_newlines(source_list, cell_id))

        # Check 2: full AST parse (catches every syntax error)
        all_errors.extend(_check_ast(source, cell_id, ci))

    if all_errors:
        print(f"[FAIL] {path} — {len(all_errors)} error(s) found:")
        for e in all_errors:
            print(e)
        return False

    print(f"[OK]   {path} — {sum(1 for c in nb['cells'] if c.get('cell_type')=='code')} code cells, all clean")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    args = sys.argv[1:]
    if not args:
        # Default: validate the tutorial notebook
        default = Path(__file__).parent.parent / 'examples' / 'madrl_citylearn_v3_tutorial.ipynb'
        paths = [default]
    else:
        paths = [Path(a) for a in args]

    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"[ERROR] not found: {p}", file=sys.stderr)
        return 1

    results = [validate(p) for p in paths]
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
