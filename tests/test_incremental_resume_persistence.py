"""Tests for resume-safe incremental CSV persistence (50-episode integrity)."""
import sys
from pathlib import Path
from typing import Dict, Optional

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import citylearn_v3_training_common as common


def _make_timeseries_row(global_step: int, episode_time_steps: int) -> dict:
    ep = global_step // episode_time_steps
    return {
        "global_step": global_step,
        "episode": ep,
        "episode_step": global_step % episode_time_steps,
        "reward_mean": -0.5,
        "all_done": (global_step + 1) % episode_time_steps == 0,
    }


def _episode_rows(episode: int, episode_time_steps: int, extra_keys: Optional[Dict] = None) -> list:
    rows = []
    base = episode * episode_time_steps
    for step in range(episode_time_steps):
        row = _make_timeseries_row(base + step, episode_time_steps)
        if extra_keys:
            row.update(extra_keys)
        rows.append(row)
    return rows


@pytest.fixture
def job_dir(tmp_path):
    root = tmp_path / "HAPPO" / "E1"
    data = root / "data"
    data.mkdir(parents=True)
    live = root / "live_progress.json"
    live.write_text("{}", encoding="utf-8")
    return root, data, live


def test_write_csv_atomic_roundtrip(tmp_path):
    path = tmp_path / "out.csv"
    rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    common.write_csv(path, rows)
    assert path.is_file()
    loaded = common.read_csv_rows(path)
    assert len(loaded) == 2
    assert loaded[0]["a"] == "1"


def test_append_stable_schema(tmp_path):
    path = tmp_path / "ts.csv"
    fn = ["episode", "global_step", "reward_mean"]
    common._append_csv_rows_stable_schema(path, [{"episode": 0, "global_step": 0, "reward_mean": -1}], fn)
    common._append_csv_rows_stable_schema(path, [{"episode": 1, "global_step": 10, "reward_mean": -2}], fn)
    loaded = common.read_csv_rows(path)
    assert len(loaded) == 2
    assert loaded[1]["episode"] == "1"


def test_schema_growth_triggers_full_rewrite(job_dir):
    root, data, live = job_dir
    ts_path = data / "timeseries.csv"
    adapter = common.CityLearnV3BackendAdapter(
        scenario="E1",
        seed=0,
        episode_time_steps=4,
        algorithm="HAPPO",
        live_progress_path=str(live),
    )
    adapter.timeseries_records = _episode_rows(0, 4)
    adapter._flush_incremental_artifacts()
    assert len(common.read_csv_rows(ts_path)) == 4

    # Episode 1 adds a new column -> full rewrite from memory (must not crash).
    adapter.timeseries_records.extend(_episode_rows(1, 4, {"new_col": 99}))
    adapter._flush_incremental_artifacts()
    loaded = common.read_csv_rows(ts_path)
    assert len(loaded) == 8
    assert "new_col" in loaded[0]
    assert loaded[-1]["new_col"] == "99"


def test_preload_truncates_partial_episode(job_dir):
    root, data, live = job_dir
    ts_path = data / "timeseries.csv"
    ep_steps = 8760
    rows = []
    for ep in range(3):
        rows.extend(_episode_rows(ep, ep_steps))
    rows.extend(_make_timeseries_row(3 * ep_steps + 500, ep_steps) for _ in range(500))
    common.write_csv(ts_path, [{**r, "reward_mean": -0.5} for r in rows])

    adapter = common.CityLearnV3BackendAdapter(
        scenario="E1",
        seed=0,
        episode_time_steps=ep_steps,
        algorithm="HAPPO",
        live_progress_path=str(live),
    )
    summary = adapter.preload_resume_artifacts(3)
    assert summary["completed_episodes"] == 3
    assert summary["timeseries_rows"] == 3 * ep_steps
    assert adapter.global_step == 3 * ep_steps
    assert len(adapter.timeseries_records) == 3 * ep_steps


def test_fifty_episode_integrity_via_incremental_flush(job_dir):
    """Simulate 50 complete episodes: incremental flush must yield intact CSV."""
    root, data, live = job_dir
    ts_path = data / "timeseries.csv"
    ep_steps = 10
    n_episodes = 50

    adapter = common.CityLearnV3BackendAdapter(
        scenario="E1",
        seed=0,
        episode_time_steps=ep_steps,
        algorithm="HAPPO",
        live_progress_path=str(live),
    )

    for ep in range(n_episodes):
        adapter.timeseries_records.extend(_episode_rows(ep, ep_steps))
        adapter._flush_incremental_artifacts()

    loaded = common.read_csv_rows(ts_path)
    assert len(loaded) == n_episodes * ep_steps

    episodes = sorted({int(r["episode"]) for r in loaded})
    assert episodes == list(range(n_episodes))
    assert int(loaded[-1]["global_step"]) == n_episodes * ep_steps - 1


def test_resume_continues_fifty_episode_file(job_dir):
    """Interrupt at ep 30, resume to 50: file must contain all 50 episodes seamlessly."""
    root, data, live = job_dir
    ts_path = data / "timeseries.csv"
    ep_steps = 10
    n_episodes = 50
    interrupt_at = 30

    # Phase 1: train 0..29, flush each episode.
    adapter1 = common.CityLearnV3BackendAdapter(
        scenario="E1",
        seed=0,
        episode_time_steps=ep_steps,
        algorithm="HAPPO",
        live_progress_path=str(live),
    )
    for ep in range(interrupt_at):
        adapter1.timeseries_records.extend(_episode_rows(ep, ep_steps))
        adapter1._flush_incremental_artifacts()

    # Phase 2: resume from ep 30.
    adapter2 = common.CityLearnV3BackendAdapter(
        scenario="E1",
        seed=0,
        episode_time_steps=ep_steps,
        algorithm="HAPPO",
        live_progress_path=str(live),
    )
    adapter2.preload_resume_artifacts(interrupt_at)
    assert adapter2.global_step == interrupt_at * ep_steps

    for ep in range(interrupt_at, n_episodes):
        adapter2.timeseries_records.extend(_episode_rows(ep, ep_steps))
        adapter2._flush_incremental_artifacts()

    loaded = common.read_csv_rows(ts_path)
    assert len(loaded) == n_episodes * ep_steps
    episodes = sorted({int(r["episode"]) for r in loaded})
    assert episodes == list(range(n_episodes))

    # global_step must be continuous 0..499
    steps = [int(r["global_step"]) for r in loaded]
    assert steps == list(range(n_episodes * ep_steps))


def test_preload_empty_when_no_prior_csv(job_dir):
    _, _, live = job_dir
    adapter = common.CityLearnV3BackendAdapter(
        scenario="E1",
        seed=0,
        episode_time_steps=4,
        algorithm="HAPPO",
        live_progress_path=str(live),
    )
    summary = adapter.preload_resume_artifacts(0)
    assert summary["timeseries_rows"] == 0
    assert adapter.global_step == 0


def test_preload_advances_global_step_without_csv(job_dir):
    """Checkpoint resume may have completed_episodes>0 but no CSV yet (first incremental run)."""
    _, _, live = job_dir
    adapter = common.CityLearnV3BackendAdapter(
        scenario="E1",
        seed=0,
        episode_time_steps=4,
        algorithm="HAPPO",
        live_progress_path=str(live),
    )
    summary = adapter.preload_resume_artifacts(5)
    assert summary["timeseries_rows"] == 0
    assert adapter.global_step == 5 * 4
