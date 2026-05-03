import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from citylearn_v3_training_common import write_training_artifacts


class _Args:
    scenario = "E3"
    seed = 7
    episode_time_steps = 2


class _Adapter:
    timeseries_records = [
        {
            "global_step": 0,
            "episode": 0,
            "episode_step": 0,
            "reward_sum": 1.0,
            "reward_mean": 0.5,
            "scenario": "E3",
        },
        {
            "global_step": 1,
            "episode": 0,
            "episode_step": 1,
            "reward_sum": 2.0,
            "reward_mean": 1.0,
            "scenario": "E3",
        },
    ]
    trace_records = [
        {
            "global_step": 0,
            "episode": 0,
            "episode_step": 0,
            "agent": "Building_1",
            "reward": 0.5,
        }
    ]


class _Candidate:
    adapter = _Adapter()


def _report():
    return {
        "project_axis_metrics": {"OE1": {}, "OE2": {}, "OE3": {}},
        "objective_axis_kpis": {
            "OE1": {
                "name": "Flexibilidad Energetica",
                "baseline_comparison": {
                    "comparable_kpis": 1,
                    "improved_kpis": 1,
                    "not_improved_kpis": 0,
                },
                "kpis": {
                    "peak_average": {
                        "value": 0.9,
                        "trace": {"source": "citylearn_v2.evaluate_v2", "lower_is_better": True},
                        "comparison": {
                            "available": True,
                            "baseline": 1.0,
                            "delta_vs_baseline": -0.1,
                            "improved_vs_baseline": True,
                        },
                    }
                },
            },
            "OE2": {
                "name": "Emisiones de CO2",
                "baseline_comparison": {
                    "comparable_kpis": 1,
                    "improved_kpis": 0,
                    "not_improved_kpis": 1,
                },
                "kpis": {
                    "carbon_emissions": {
                        "value": 1.1,
                        "trace": {"source": "citylearn_v2.evaluate_v2", "lower_is_better": True},
                        "comparison": {
                            "available": True,
                            "baseline": 1.0,
                            "delta_vs_baseline": 0.1,
                            "improved_vs_baseline": False,
                        },
                    }
                },
            },
            "OE3": {
                "name": "Costos Energeticos",
                "baseline_comparison": {
                    "comparable_kpis": 1,
                    "improved_kpis": 1,
                    "not_improved_kpis": 0,
                },
                "kpis": {
                    "electricity_cost": {
                        "value": 0.8,
                        "trace": {"source": "citylearn_v2.evaluate_v2", "lower_is_better": True},
                        "comparison": {
                            "available": True,
                            "baseline": 1.0,
                            "delta_vs_baseline": -0.2,
                            "improved_vs_baseline": True,
                        },
                    }
                },
            },
        },
        "axis_kpis": {
            "peak_average": 0.9,
            "carbon_emissions": 1.1,
            "electricity_cost": 0.8,
        },
    }


def test_training_artifacts_use_data_checkpoints_and_figures_layout(tmp_path):
    output_dir = tmp_path / "happo" / "E3_seed_7"
    checkpoint_dir = output_dir / "checkpoints" / "backend"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "actor_agent0.pt").write_bytes(b"checkpoint")

    artifacts = write_training_artifacts(
        output_dir=output_dir,
        algorithm="HAPPO",
        backend="external/HARL",
        args=_Args(),
        report=_report(),
        candidate=_Candidate(),
        hyperparameters={"episodes": 1},
    )

    data_dir = output_dir / "data"
    figures_dir = output_dir / "figures"
    tables_dir = figures_dir / "tables"

    for file_name in ["results.json", "timeseries.csv", "trace.csv", "checkpoint_manifest.json"]:
        assert (data_dir / file_name).is_file()
        assert (output_dir / file_name).is_file()

    assert artifacts["checkpoint_count"] == 1
    assert artifacts["timeseries_rows"] == 2
    assert artifacts["trace_rows"] == 1
    assert (figures_dir / "figures_manifest.json").is_file()
    assert (figures_dir / "reward_timeseries.png").is_file()
    assert (figures_dir / "episode_reward_summary.png").is_file()
    assert (figures_dir / "axis_baseline_comparison.png").is_file()
    assert (figures_dir / "core_kpis.png").is_file()
    assert (tables_dir / "episode_summary.csv").is_file()
    assert (tables_dir / "objective_kpis.csv").is_file()
    assert (tables_dir / "checkpoint_inventory.csv").is_file()

    results = json.loads((data_dir / "results.json").read_text(encoding="utf-8"))
    assert results["artifact_layout"]["data"] == str(data_dir)
    assert results["artifact_layout"]["checkpoints"] == str(output_dir / "checkpoints")
    assert results["figures"]["figure_count"] >= 4
