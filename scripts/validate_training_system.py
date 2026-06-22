#!/usr/bin/env python3
"""
validate_training_system.py
Complete pre-push / pre-training validation for the MADRL CityLearn v3 system.

Checks (all must PASS before training or pushing):
  1. Syntax     — all .py scripts in CityLearn/scripts/ parse cleanly
  2. Launcher   — execution_mode default=two_phase, constants, MASAC Phase 2 settings, OOM retry
  3. Monitor    — FASE 1/2 phase headers, no [FAILED] tag, _Tee class, file saving
  4. Notebook   — cell 1.2 uses CITYLEARN_DEV_BRANCH, launcher_base_args has --execution-mode two_phase
  5. Git sync   — both branches (citylearn-v3-madrl + fix/platformdirs-colab) at same SHA on remote
  6. MADRL tests — test_masac_oom_retry_fix.py (no external deps, fast)

Usage:
    python CityLearn/scripts/validate_training_system.py
    python CityLearn/scripts/validate_training_system.py --skip-git   # offline mode
    python CityLearn/scripts/validate_training_system.py --skip-tests  # skip test suite
    python CityLearn/scripts/validate_training_system.py --verbose

Exit codes:
    0  all checks PASS
    1  one or more checks FAIL
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Tuple

# ── Repo layout ──────────────────────────────────────────────────────────────
SCRIPTS_DIR   = Path(__file__).resolve().parent          # CityLearn/scripts/
CITYLEARN_DIR = SCRIPTS_DIR.parent                       # CityLearn/
EXAMPLES_DIR  = CITYLEARN_DIR / 'examples'
TESTS_DIR     = CITYLEARN_DIR / 'tests'

LAUNCHER_PATH  = SCRIPTS_DIR / 'colab_a100_official_launcher.py'
MONITOR_PATH   = SCRIPTS_DIR / 'colab_a100_live_monitor.py'
NOTEBOOK_PATH  = EXAMPLES_DIR / 'madrl_citylearn_v3_tutorial.ipynb'

CITYLEARN_REMOTE     = 'mac-tapia'
CITYLEARN_BRANCH     = 'citylearn-v3-madrl'
CITYLEARN_DEV_BRANCH = 'fix/platformdirs-colab'

VERBOSE = '--verbose' in sys.argv

# ── Result tracking ──────────────────────────────────────────────────────────
_results: List[Tuple[str, bool, str]] = []   # (check_name, passed, detail)


def _ok(name: str, detail: str = '') -> bool:
    _results.append((name, True, detail))
    if VERBOSE:
        print(f'  [PASS] {name}' + (f' — {detail}' if detail else ''))
    return True


def _fail(name: str, detail: str) -> bool:
    _results.append((name, False, detail))
    print(f'  [FAIL] {name}: {detail}')
    return False


def _check(name: str, cond: bool, ok_msg: str, fail_msg: str) -> bool:
    return _ok(name, ok_msg) if cond else _fail(name, fail_msg)


# ── Helper: load a module from file path ─────────────────────────────────────
def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(args: List[str], cwd: Path = CITYLEARN_DIR) -> str:
    return subprocess.check_output(
        ['git'] + args, cwd=cwd, text=True, stderr=subprocess.DEVNULL
    ).strip()


# ============================================================================
# CHECK 1 — Syntax: all scripts/*.py
# ============================================================================
def check_syntax() -> int:
    """Returns number of failures."""
    print('\n[1] Syntax — CityLearn/scripts/*.py')
    failures = 0
    for py in sorted(SCRIPTS_DIR.glob('*.py')):
        try:
            ast.parse(py.read_text(encoding='utf-8'))
            _ok(f'syntax:{py.name}')
        except SyntaxError as exc:
            _fail(f'syntax:{py.name}', f'line {exc.lineno}: {exc.msg}')
            failures += 1
    return failures


# ============================================================================
# CHECK 2 — Launcher configuration
# ============================================================================
def check_launcher() -> int:
    print('\n[2] Launcher — colab_a100_official_launcher.py')
    failures = 0

    try:
        mod = _load_module(LAUNCHER_PATH, '_launcher_val')
    except Exception as exc:
        _fail('launcher:import', str(exc))
        return 1

    ns = mod

    # 2a. execution_mode default
    try:
        args = ns.parse_args([])
        mode = args.execution_mode
        if not _check('launcher:execution_mode_default',
                      mode == 'two_phase',
                      f'default={mode}',
                      f'default={mode!r} != "two_phase" — Colab will not use two-phase!'):
            failures += 1
    except Exception as exc:
        _fail('launcher:parse_args', str(exc)); failures += 1

    # 2b. TWO_PHASE_LIGHT / TWO_PHASE_HEAVY constants
    light = getattr(ns, 'TWO_PHASE_LIGHT', None)
    heavy = getattr(ns, 'TWO_PHASE_HEAVY', None)
    if not _check('launcher:TWO_PHASE_LIGHT',
                  light == ('happo', 'matd3', 'maac'),
                  str(light),
                  f'{light!r} != ("happo","matd3","maac")'):
        failures += 1
    if not _check('launcher:TWO_PHASE_HEAVY',
                  heavy == ('masac',),
                  str(heavy),
                  f'{heavy!r} != ("masac",)'):
        failures += 1

    # 2c. Phase 2 torch threads default
    try:
        args = ns.parse_args([])
        val = args.two_phase_heavy_torch_threads
        if not _check('launcher:heavy_torch_threads',
                      val == 4,
                      f'default={val}',
                      f'default={val} != 4'):
            failures += 1
        val2 = args.two_phase_masac_cuda_fraction
        if not _check('launcher:masac_cuda_fraction',
                      abs(val2 - 0.26) < 1e-9,
                      f'default={val2}',
                      f'default={val2} != 0.26 — MASAC will OOM in Phase 2!'):
            failures += 1
    except Exception as exc:
        _fail('launcher:phase2_args', str(exc)); failures += 1

    # 2d. make_oom_retry_job includes cuda-memory-fraction
    src = LAUNCHER_PATH.read_text(encoding='utf-8')
    retry_fn_match = None
    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'make_oom_retry_job':
                retry_src = ast.get_source_segment(src, node) or ''
                retry_fn_match = retry_src
                break
    except Exception:
        retry_fn_match = src[src.find('def make_oom_retry_job'):][:2000]

    if retry_fn_match is not None:
        for flag, name in [
            ('--cuda-memory-fraction', 'retry:cuda_memory_fraction'),
            ('--masac-preload-batch-device', 'retry:preload_batch_device'),
            ('--buffer-size', 'retry:buffer_size'),
        ]:
            if not _check(name,
                          flag in retry_fn_match,
                          f'{flag} present',
                          f'{flag} missing from make_oom_retry_job — OOM retry will fail!'):
                failures += 1
    else:
        _fail('launcher:make_oom_retry_job', 'function not found'); failures += 1

    # 2e. _single_thread_env NOT present (renamed to _perf_env)
    if not _check('launcher:no_single_thread_env',
                  '_single_thread_env' not in src,
                  '_single_thread_env absent (renamed to _perf_env)',
                  '_single_thread_env still referenced — NameError in Phase 2!'):
        failures += 1

    # 2f. MALLOC_ARENA_MAX in _perf_env
    if not _check('launcher:malloc_arena_max',
                  'MALLOC_ARENA_MAX' in src,
                  'MALLOC_ARENA_MAX present',
                  'MALLOC_ARENA_MAX missing — glibc lock contention on 167 GiB RAM!'):
        failures += 1

    return failures


# ============================================================================
# CHECK 3 — Monitor configuration
# ============================================================================
def check_monitor() -> int:
    print('\n[3] Monitor — colab_a100_live_monitor.py')
    failures = 0

    src = MONITOR_PATH.read_text(encoding='utf-8')

    checks = [
        ('monitor:TWO_PHASE_LIGHT',   'TWO_PHASE_LIGHT',           'FASE 1 constant present',        'TWO_PHASE_LIGHT constant missing'),
        ('monitor:TWO_PHASE_HEAVY',   'TWO_PHASE_HEAVY',           'FASE 2 constant present',        'TWO_PHASE_HEAVY constant missing'),
        ('monitor:_Tee',              'class _Tee',                 '_Tee class present',             '_Tee class missing — output not saved to file!'),
        ('monitor:_render_inner',     'def _render_inner',          '_render_inner present',          '_render_inner missing'),
        ('monitor:render_once',       'def render_once',            'render_once present',            'render_once missing'),
        ('monitor:monitor_snapshots', 'monitor_snapshots',          'snapshot dir present',           'monitor_snapshots missing — output not saved!'),
        ('monitor:monitor_latest',    'monitor_latest.txt',         'monitor_latest.txt present',     'monitor_latest.txt missing'),
        ('monitor:FASE_1_header',     'FASE 1',                     'FASE 1 header present',          'FASE 1 header missing — no phase organization!'),
        ('monitor:FASE_2_header',     'FASE 2',                     'FASE 2 header present',          'FASE 2 header missing — no phase organization!'),
        ('monitor:_phase_sort_key',   'def _phase_sort_key',        '_phase_sort_key present',        '_phase_sort_key missing'),
    ]
    for name, token, ok_msg, fail_msg in checks:
        if not _check(name, token in src, ok_msg, fail_msg):
            failures += 1

    # Verify [FAILED] tag is NOT in print_progress source
    try:
        mod = _load_module(MONITOR_PATH, '_monitor_val')
        pp_src = inspect.getsource(mod.print_progress)
        if not _check('monitor:no_FAILED_tag',
                      'FAILED' not in pp_src and 'failed_with_progress' not in pp_src,
                      '[FAILED] tag absent',
                      '[FAILED] tag still in print_progress — remove it!'):
            failures += 1
    except Exception as exc:
        _fail('monitor:print_progress_inspect', str(exc)); failures += 1

    return failures


# ============================================================================
# CHECK 4 — Notebook configuration
# ============================================================================
def check_notebook() -> int:
    print('\n[4] Notebook — madrl_citylearn_v3_tutorial.ipynb')
    failures = 0

    try:
        nb = json.loads(NOTEBOOK_PATH.read_text(encoding='utf-8'))
    except Exception as exc:
        _fail('notebook:parse', str(exc))
        return 1

    # Cell 1.2 (index 18) — git setup
    cell_12_src = ''.join(nb['cells'][18]['source'])

    checks_12 = [
        ('notebook:CITYLEARN_DEV_BRANCH',
         'CITYLEARN_DEV_BRANCH',
         'CITYLEARN_DEV_BRANCH present in cell 1.2',
         'CITYLEARN_DEV_BRANCH missing from cell 1.2 — Colab will use stale branch!'),
        ('notebook:dev_branch_value',
         "fix/platformdirs-colab",
         "fix/platformdirs-colab referenced in cell 1.2",
         "fix/platformdirs-colab missing from cell 1.2!"),
        ('notebook:checkout_uses_dev',
         'mac-tapia/{CITYLEARN_DEV_BRANCH}',
         'checkout uses CITYLEARN_DEV_BRANCH',
         'checkout uses CITYLEARN_BRANCH — will get stale code!'),
    ]
    for name, token, ok_msg, fail_msg in checks_12:
        if not _check(name, token in cell_12_src, ok_msg, fail_msg):
            failures += 1

    # Cell 7.0 (launcher_base_args) — search all cells
    launcher_cell_src = ''
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            s = ''.join(cell['source'])
            if 'def launcher_base_args' in s:
                launcher_cell_src = s
                break

    if not launcher_cell_src:
        _fail('notebook:launcher_base_args', 'def launcher_base_args not found in any cell')
        failures += 1
    else:
        if not _check('notebook:execution_mode_two_phase',
                      '--execution-mode' in launcher_cell_src and 'two_phase' in launcher_cell_src,
                      '--execution-mode two_phase present',
                      '--execution-mode two_phase missing from launcher_base_args!'):
            failures += 1
        if not _check('notebook:no_max_parallel_12',
                      "'--max-parallel', '12'" not in launcher_cell_src and
                      '"--max-parallel", "12"' not in launcher_cell_src,
                      '--max-parallel 12 absent (good)',
                      '--max-parallel 12 still in launcher_base_args — overrides two_phase!'):
            failures += 1

    return failures


# ============================================================================
# CHECK 5 — Git sync (both branches at same SHA on remote)
# ============================================================================
def check_git_sync() -> int:
    print('\n[5] Git sync — both remote branches must be identical')
    failures = 0

    try:
        _git(['fetch', '--quiet', CITYLEARN_REMOTE,
              CITYLEARN_BRANCH, CITYLEARN_DEV_BRANCH])
    except Exception as exc:
        _fail('git:fetch', f'git fetch failed: {exc}')
        return 1

    try:
        sha_badge = _git(['rev-parse', f'{CITYLEARN_REMOTE}/{CITYLEARN_BRANCH}'])
        sha_dev   = _git(['rev-parse', f'{CITYLEARN_REMOTE}/{CITYLEARN_DEV_BRANCH}'])
        local_sha = _git(['rev-parse', 'HEAD'])
    except Exception as exc:
        _fail('git:rev-parse', str(exc))
        return 1

    if not _check('git:branches_in_sync',
                  sha_badge == sha_dev,
                  f'both at {sha_badge[:12]}',
                  f'DIVERGED! {CITYLEARN_BRANCH}={sha_badge[:12]} '
                  f'vs {CITYLEARN_DEV_BRANCH}={sha_dev[:12]} '
                  f'— run: python CityLearn/scripts/push.py'):
        failures += 1

    if not _check('git:local_pushed',
                  local_sha == sha_dev,
                  f'local HEAD pushed ({local_sha[:12]})',
                  f'local HEAD {local_sha[:12]} NOT on remote — run: python CityLearn/scripts/push.py'):
        failures += 1

    return failures


# ============================================================================
# CHECK 6 — MADRL-specific test suite (no external deps required)
# ============================================================================
def check_tests() -> int:
    print('\n[6] MADRL tests — test_masac_oom_retry_fix.py')
    failures = 0

    test_file = TESTS_DIR / 'test_masac_oom_retry_fix.py'
    if not test_file.exists():
        _fail('tests:file_exists', f'{test_file} not found')
        return 1

    # Load and run the tests manually (no pytest dependency)
    try:
        import types
        pytest_shim = types.ModuleType('pytest')
        pytest_shim.approx = lambda x, **kw: x
        def _mark_param(*a, **kw):
            def dec(f): return f
            return dec
        def _fixture(*a, **kw):
            def dec(f): return f
            return dec
        pytest_shim.mark    = types.SimpleNamespace(parametrize=_mark_param)
        pytest_shim.fixture = _fixture
        sys.modules['pytest'] = pytest_shim

        spec = importlib.util.spec_from_file_location('_oom_tests', test_file)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as exc:
        _fail('tests:import', str(exc))
        return 1

    # Run all test_ functions
    passed = 0
    failed = 0
    for name in dir(mod):
        if not name.startswith('test_'):
            continue
        fn = getattr(mod, name)
        if not callable(fn):
            continue
        try:
            import inspect as _inspect
            sig = _inspect.signature(fn)
            params = [p for p in sig.parameters.values()
                      if p.default is _inspect.Parameter.empty]
            if params:
                continue  # parametrized — skip (needs pytest)
            fn()
            _ok(f'test:{name}')
            passed += 1
        except AssertionError as exc:
            _fail(f'test:{name}', str(exc))
            failed += 1
        except Exception as exc:
            _fail(f'test:{name}', f'{type(exc).__name__}: {exc}')
            failed += 1

    failures = failed
    if passed:
        _ok('tests:summary', f'{passed} passed, {failed} failed')

    return failures


# ============================================================================
# MAIN
# ============================================================================
def main() -> int:
    skip_git   = '--skip-git'   in sys.argv
    skip_tests = '--skip-tests' in sys.argv

    print('=' * 70)
    print('  MADRL CityLearn v3 — Training System Validation')
    print('=' * 70)

    total_failures = 0
    total_failures += check_syntax()
    total_failures += check_launcher()
    total_failures += check_monitor()
    total_failures += check_notebook()

    if skip_git:
        print('\n[5] Git sync — SKIPPED (--skip-git)')
    else:
        total_failures += check_git_sync()

    if skip_tests:
        print('\n[6] MADRL tests — SKIPPED (--skip-tests)')
    else:
        total_failures += check_tests()

    # ── Summary ──────────────────────────────────────────────────────────────
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)

    print('\n' + '=' * 70)
    if total_failures == 0:
        print(f'  ALL CHECKS PASSED ({passed} checks)')
        print('  Sistema listo para entrenar en Colab A100.')
    else:
        print(f'  FAILED: {failed} check(s) failed — fix before pushing/training!')
        print()
        for name, ok, detail in _results:
            if not ok:
                print(f'    [FAIL] {name}: {detail}')
    print('=' * 70)

    return 0 if total_failures == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
