"""Validate the two-phase dynamic-backfill scheduling rule.

Rule under test (`_run_backfill_schedule`):
  * Phase 1 = HAPPO + MASAC start first; phase 2 = MATD3 + MAAC.
  * Phase 2 NEVER starts before a phase-1 job completes.
  * Each phase-1 completion admits exactly ONE phase-2 job (one-for-one),
    lightest-first (MAAC before MATD3), as soon as a slot frees.
  * Works dynamically/flexibly regardless of completion order and on full resume
    (all phase-1 jobs skipped/instant).
"""
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import colab_a100_official_launcher as launcher


def _jobs():
    phase1 = [
        {"name": "happo", "scenario": "E1"},
        {"name": "happo", "scenario": "E2"},
        {"name": "happo", "scenario": "E3"},
        {"name": "masac", "scenario": "E1"},
        {"name": "masac", "scenario": "E2"},
        {"name": "masac", "scenario": "E3"},
    ]
    phase2 = [
        {"name": "matd3", "scenario": "E1"},
        {"name": "matd3", "scenario": "E2"},
        {"name": "matd3", "scenario": "E3"},
        {"name": "maac", "scenario": "E1"},
        {"name": "maac", "scenario": "E2"},
        {"name": "maac", "scenario": "E3"},
    ]
    return phase1, phase2


class _Recorder:
    """submit_fn that records start order and lets the test release completions."""

    def __init__(self, *, instant=False, exit_codes=None):
        self.lock = threading.Lock()
        self.start_order = []
        self.max_concurrent = 0
        self._active = 0
        self._instant = instant
        self._exit_codes = exit_codes or {}
        # gate per job key so the test controls completion order
        self._gates = {}

    def _key(self, job):
        return f"{job['name']}/{job['scenario']}"

    def gate(self, key):
        ev = self._gates.setdefault(key, threading.Event())
        return ev

    def submit(self, pool, job):
        key = self._key(job)
        with self.lock:
            self.start_order.append(key)
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)

        def _work():
            if not self._instant:
                self.gate(key).wait(timeout=10)
            with self.lock:
                self._active -= 1
            return int(self._exit_codes.get(key, 0))

        return pool.submit(_work)


def test_phase2_never_before_phase1_and_one_for_one():
    phase1, phase2 = _jobs()
    rec = _Recorder(instant=False)

    # Release phase-1 jobs one at a time, asserting phase-2 admission is one-for-one.
    def _driver():
        # Wait until all 6 phase-1 jobs have started (cap=6) and NO phase-2 yet.
        for _ in range(200):
            with rec.lock:
                started = list(rec.start_order)
            if len(started) >= 6:
                break
            time.sleep(0.01)
        with rec.lock:
            started = list(rec.start_order)
        assert all(s.startswith(("happo", "masac")) for s in started[:6]), started
        assert len(started) == 6, f"phase-2 prefilled before any phase-1 done: {started}"

        # Release phase-1 completions one by one; each opens exactly one phase-2 (lightest).
        expected_phase2 = ["maac/E1", "maac/E2", "maac/E3", "matd3/E1", "matd3/E2", "matd3/E3"]
        for i, p1 in enumerate(["happo/E1", "happo/E2", "happo/E3", "masac/E1", "masac/E2", "masac/E3"]):
            rec.gate(p1).set()
            # wait for the i-th phase-2 admission
            for _ in range(300):
                with rec.lock:
                    p2_started = [s for s in rec.start_order if s.startswith(("maac", "matd3"))]
                if len(p2_started) >= i + 1:
                    break
                time.sleep(0.01)
            with rec.lock:
                p2_started = [s for s in rec.start_order if s.startswith(("maac", "matd3"))]
            assert p2_started == expected_phase2[: i + 1], (i, p2_started)
        # Release all phase-2 to let the pool drain.
        for p2 in expected_phase2:
            rec.gate(p2).set()

    t = threading.Thread(target=_driver, daemon=True)
    t.start()
    rc = launcher._run_backfill_schedule(
        phase1_jobs=phase1, phase2_jobs=phase2, max_workers=6, submit_fn=rec.submit
    )
    t.join(timeout=15)
    assert rc == 0
    assert rec.max_concurrent <= 6
    # All 12 jobs ran exactly once.
    assert sorted(rec.start_order) == sorted(
        f"{j['name']}/{j['scenario']}" for j in (phase1 + phase2)
    )


def test_full_resume_phase1_all_instant_admits_all_phase2():
    """On resume where every phase-1 job is already complete (skipped -> instant 0),
    all 6 phase-2 jobs must still be admitted (no deadlock)."""
    phase1, phase2 = _jobs()
    rec = _Recorder(instant=True)
    rc = launcher._run_backfill_schedule(
        phase1_jobs=phase1, phase2_jobs=phase2, max_workers=6, submit_fn=rec.submit
    )
    assert rc == 0
    assert len(rec.start_order) == 12
    assert sorted(rec.start_order) == sorted(
        f"{j['name']}/{j['scenario']}" for j in (phase1 + phase2)
    )


def test_failure_exit_code_surfaces():
    phase1, phase2 = _jobs()
    rec = _Recorder(instant=True, exit_codes={"matd3/E2": 7})
    rc = launcher._run_backfill_schedule(
        phase1_jobs=phase1, phase2_jobs=phase2, max_workers=6, submit_fn=rec.submit
    )
    assert rc == 7


def test_lightest_first_ordering():
    _, phase2 = _jobs()
    ordered = sorted(phase2, key=launcher._job_backfill_weight)
    names = [f"{j['name']}/{j['scenario']}" for j in ordered]
    assert names == [
        "maac/E1", "maac/E2", "maac/E3", "matd3/E1", "matd3/E2", "matd3/E3",
    ]
