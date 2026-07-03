"""Regression tests for KPI-grounded --skip-completed and HAPPO evaluation unwrap."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
REPO_ROOT = Path(__file__).resolve().parents[2]
HARL_ROOT = REPO_ROOT / "external" / "HARL"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(HARL_ROOT) not in sys.path:
    sys.path.insert(0, str(HARL_ROOT))

from citylearn_v3_training_common import (  # noqa: E402
    DATA_DIR_NAME,
    JOB_LAUNCHER_COMPLETE_MARKER,
    job_counts_as_launcher_complete,
    job_meets_launcher_complete_requirements,
    reconcile_stale_job_launcher_marker,
    write_job_launcher_complete_marker,
)
from citylearn.v3.objectives import _unwrap_env  # noqa: E402


def _write_results(run_dir: Path, payload: dict) -> None:
    data = run_dir / DATA_DIR_NAME
    data.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    (data / "results.json").write_text(text, encoding="utf-8")
    (run_dir / "results.json").write_text(text, encoding="utf-8")


def _write_timeseries_steps(run_dir: Path, n_steps: int, episode_time_steps: int = 8760) -> None:
    data = run_dir / DATA_DIR_NAME
    data.mkdir(parents=True, exist_ok=True)
    path = data / "timeseries.csv"
    lines = ["episode,episode_step,global_step,time_step,reward_mean,all_done\n"]
    for gs in range(n_steps):
        ep = gs // episode_time_steps
        ep_step = gs % episode_time_steps
        all_done = "True" if ep_step >= episode_time_steps - 1 else "False"
        lines.append(f"{ep},{ep_step},{gs},1,-0.5,{all_done}\n")
    path.write_text("".join(lines), encoding="utf-8")


def _touch_checkpoint(
    run_dir: Path,
    name: str = "actor.pt",
    *,
    algorithm: str = "matd3",
) -> None:
    algo = algorithm.lower()
    if algo in {"matd3", "maddpg"}:
        ckpt = run_dir / "checkpoints" / "offpolicy_run" / "models" / "policy_0"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "actor.pt").write_bytes(b"pt")
        return
    if algo == "masac":
        ckpt = run_dir / "checkpoints" / "models"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "100_rnn_net_params.pkl").write_bytes(b"pkl")
        return
    if algo == "happo":
        ckpt = run_dir / "checkpoints" / "gym" / "run" / "happo" / "models"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "actor_agent0.pt").write_bytes(b"pt")
        return
    ckpt = run_dir / "checkpoints" / "models"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / name).write_bytes(b"pt")


def _kpi_report() -> dict:
    return {
        "all_values": {"peak_average": 1.0, "electricity_cost_delta": 100.0},
        "project_axis_metrics": {"OE1": 0.5},
        "report_source": {"completed_episode_count": 50},
    }


def test_resume_slice_episodes_field_does_not_block_skip(tmp_path: Path) -> None:
    """Drive: episodes=40 (last resume slice) but timeseries/KPIs show 50 → SKIP."""
    run = tmp_path / "MATD3" / "E1"
    run.mkdir(parents=True)
    _write_results(
        run,
        {
            "algorithm": "MATD3",
            "scenario": "E1",
            "episodes": 40,
            "episodes_recorded": 50,
            "episode_time_steps": 8760,
            "citylearn_v3_report": _kpi_report(),
            "hyperparameters": {"episodes": 50, "target_episodes": 50},
        },
    )
    _write_timeseries_steps(run, 50 * 8760)
    _touch_checkpoint(run, algorithm="matd3")

    assert job_meets_launcher_complete_requirements(run, target_episodes=50)
    assert job_counts_as_launcher_complete(run, target_episodes=50)


def test_happo_salvage_without_kpis_does_not_skip(tmp_path: Path) -> None:
    run = tmp_path / "HAPPO" / "E1"
    run.mkdir(parents=True)
    _write_results(
        run,
        {
            "algorithm": "HAPPO",
            "scenario": "E1",
            "status": "completed_with_salvage",
            "salvage_reason": "NameError: name 'VecEnvWrapper' is not defined",
            "episodes_recorded": 50,
            "episode_time_steps": 8760,
            "citylearn_v3_report": {"all_values": {}},
            "hyperparameters": {
                "target_episodes": 50,
                "n_rollout_threads": 12,
                "job_resume": {"completed_episodes": 49},
            },
        },
    )
    _write_timeseries_steps(run, 49 * 8760)
    _touch_checkpoint(run, algorithm="happo")

    assert not job_meets_launcher_complete_requirements(run, target_episodes=50)
    assert not job_counts_as_launcher_complete(run, target_episodes=50)


def test_full_kpi_job_skips(tmp_path: Path) -> None:
    run = tmp_path / "MATD3" / "E2"
    run.mkdir(parents=True)
    report = _kpi_report()
    report["report_source"] = {"completed_episode_count": 50}
    _write_results(
        run,
        {
            "algorithm": "MATD3",
            "scenario": "E2",
            "episodes": 50,
            "episodes_recorded": 50,
            "episode_time_steps": 8760,
            "citylearn_v3_report": report,
            "hyperparameters": {"episodes": 50, "target_episodes": 50},
        },
    )
    _write_timeseries_steps(run, 50 * 8760)
    _touch_checkpoint(run, algorithm="matd3")

    assert job_meets_launcher_complete_requirements(run, target_episodes=50)
    assert job_counts_as_launcher_complete(run, target_episodes=50)


def test_maac_resume_slice_with_full_timeseries_skips(tmp_path: Path) -> None:
    """Drive MAAC E1/E2: episodes=11 slice, episodes_recorded=50, KPIs present."""
    run = tmp_path / "MAAC" / "E2"
    run.mkdir(parents=True)
    _write_results(
        run,
        {
            "algorithm": "MAAC",
            "scenario": "E2",
            "episodes": 11,
            "episodes_recorded": 50,
            "episode_time_steps": 8760,
            "citylearn_v3_report": _kpi_report(),
            "hyperparameters": {"episodes": 50, "target_episodes": 50},
        },
    )
    _write_timeseries_steps(run, 50 * 8760)
    _touch_checkpoint(run, algorithm="maac", name="checkpoint_episode_11.pt")

    assert job_meets_launcher_complete_requirements(run, target_episodes=50)
    assert job_counts_as_launcher_complete(run, target_episodes=50)


def test_stale_marker_removed_when_kpis_missing(tmp_path: Path) -> None:
    run = tmp_path / "MAAC" / "E3"
    run.mkdir(parents=True)
    _write_results(
        run,
        {
            "algorithm": "MAAC",
            "scenario": "E3",
            "episodes": 11,
            "episodes_recorded": 50,
            "episode_time_steps": 8760,
            "citylearn_v3_report": {"all_values": {}},
            "hyperparameters": {"episodes": 50, "target_episodes": 50},
        },
    )
    _write_timeseries_steps(run, 50 * 8760)
    _touch_checkpoint(run, algorithm="maac", name="checkpoint_episode_50.pt")

    write_job_launcher_complete_marker(
        run,
        algorithm="MAAC",
        scenario="E3",
        target_episodes=50,
        episodes_completed=50,
        source="test_stale",
    )
    marker = run / DATA_DIR_NAME / JOB_LAUNCHER_COMPLETE_MARKER
    assert marker.is_file()

    assert reconcile_stale_job_launcher_marker(run, target_episodes=50)
    assert not marker.is_file()
    assert not job_counts_as_launcher_complete(run, target_episodes=50)


def test_harl_share_vec_env_unwrapped_no_vecenvwrapper_nameerror() -> None:
    from harl.envs.env_wrappers import ShareDummyVecEnv

    class _MiniEnv:
        observation_space = share_observation_space = action_space = None
        n_agents = 1

        def seed(self, _seed):
            return None

        def reset(self):
            return None, None, None

        def step(self, _actions):
            return None, None, None, None, None, None

        def close(self):
            return None

    vec = ShareDummyVecEnv([lambda: _MiniEnv()])
    try:
        assert vec.unwrapped is vec
    finally:
        vec.close()


def test_unwrap_env_avoids_broken_harl_unwrapped() -> None:
    class _Inner:
        def evaluate_v2(self):
            return "ok"

    class _HarLLike:
        env = _Inner()

        @property
        def unwrapped(self):
            raise NameError("name 'VecEnvWrapper' is not defined")

    wrapper = _HarLLike()
    assert _unwrap_env(wrapper) is wrapper.env
    assert _unwrap_env(wrapper).evaluate_v2() == "ok"


@pytest.mark.parametrize(
    "fixture_name,expect_complete",
    [
        ("matd3_E1_results.json", True),
        ("maac_E1_results.json", True),
        ("happo_E1_results.json", False),
    ],
)
def test_drive_kpi_fixtures_completion(tmp_path: Path, fixture_name: str, expect_complete: bool) -> None:
    fixture = REPO_ROOT / "outputs" / "_drive_madrl" / "kpis" / fixture_name
    if not fixture.is_file():
        pytest.skip(f"fixture missing: {fixture}")

    payload = json.loads(fixture.read_text(encoding="utf-8"))
    algo = str(payload.get("algorithm", "X")).lower()
    file_parts = fixture.stem.split("_")
    file_scen = file_parts[1].upper() if len(file_parts) >= 2 else str(payload.get("scenario", "E1")).upper()
    payload_scen = str(payload.get("scenario") or "").upper()
    mislabeled = bool(payload_scen and payload_scen != file_scen)
    scen = file_scen
    run = tmp_path / algo.upper() / scen
    run.mkdir(parents=True)
    _write_results(run, payload)

    recorded = int(payload.get("episodes_recorded") or payload.get("episodes") or 0)
    if not mislabeled and recorded > 0:
        _write_timeseries_steps(run, recorded * 8760)
        _touch_checkpoint(run, algorithm=algo, name=f"checkpoint_episode_{recorded}.pt")

    assert job_counts_as_launcher_complete(run, target_episodes=50) is expect_complete


def test_drive_exports_mislabeled_detected() -> None:
    tools_dir = REPO_ROOT / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from validate_drive_kpi_downloads import validate_results_json

    kpis = REPO_ROOT / "outputs" / "_drive_madrl" / "kpis"
    if not kpis.is_dir():
        pytest.skip("drive kpis folder missing")

    mislabeled = [
        row
        for path in sorted(kpis.glob("*_results.json"))
        for row in [validate_results_json(path)]
        if not row.get("ok")
    ]
    if not mislabeled:
        pytest.skip("no mislabeled drive exports in kpis/ (downloads look consistent)")

    for row in mislabeled:
        assert row.get("mismatches"), row
