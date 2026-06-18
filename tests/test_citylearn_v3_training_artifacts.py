import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np  # noqa: E402

from citylearn_v3_training_common import (  # noqa: E402
    CityLearnV3BackendAdapter,
    _artifact_consistency_audit,
    install_finite_optimizer_step_guard,
    write_training_artifacts,
    write_training_summary,
)


class _Args:
    scenario = "E3"
    seed = 7
    episode_time_steps = 2


class _EfficientArgs(_Args):
    artifact_profile = "efficient"


class _LegacyComparisonArgs(_Args):
    artifact_profile = "full"
    legacy_root_artifacts = True
    statistical_comparison_artifacts = True


class _Space:
    def __init__(self, dim):
        self.low = [-1.0] * dim
        self.high = [1.0] * dim


class _KPIFrame:
    def to_dict(self, orient):
        assert orient == "records"
        return [
            {
                "level": "building",
                "name": "Building_1",
                "cost_function": "building_energy_grid_total_import_control_kwh",
                "value": 10.0,
            },
            {
                "level": "building",
                "name": "Building_1",
                "cost_function": "building_energy_grid_total_export_control_kwh",
                "value": 2.0,
            },
            {
                "level": "building",
                "name": "Building_2",
                "cost_function": "building_energy_grid_total_import_control_kwh",
                "value": 1.0,
            },
            {
                "level": "building",
                "name": "Building_2",
                "cost_function": "building_energy_grid_total_export_control_kwh",
                "value": 4.0,
            },
        ]


class _CoreEnv:
    action_names = [["electrical_storage"], ["electrical_storage"]]
    observation_names = [
        ["net_electricity_consumption", "electrical_storage_soc"],
        ["net_electricity_consumption", "electrical_storage_soc"],
    ]


class _ObjectiveEnv:
    possible_agents = ["Building_1", "Building_2"]
    env = _CoreEnv()

    def action_space(self, agent):
        return _Space(1)

    def observation_space(self, agent):
        return _Space(2)

    def get_kpi_frame(self):
        return _KPIFrame()


class _Adapter:
    env = _ObjectiveEnv()
    agents = ["Building_1", "Building_2"]
    action_dims = {"Building_1": 1, "Building_2": 1}
    observation_dims = {"Building_1": 2, "Building_2": 2}
    normalization_metadata = {
        "normalize_observations": True,
        "observation_method": "test normalization",
    }
    timeseries_records = [
        {
            "global_step": 0,
            "episode": 0,
            "episode_step": 0,
            "reward_sum": 1.0,
            "reward_mean": 0.5,
            "district_net_electricity_consumption": 2.0,
            "district_net_electricity_consumption_without_storage": 2.4,
            "district_net_electricity_consumption_cost": 0.3,
            "district_net_electricity_consumption_emission": 0.5,
            "electricity_price_mean": 0.15,
            "carbon_intensity_mean": 0.2,
            "scenario": "E3",
        },
        {
            "global_step": 1,
            "episode": 0,
            "episode_step": 1,
            "reward_sum": 2.0,
            "reward_mean": 1.0,
            "district_net_electricity_consumption": 1.0,
            "district_net_electricity_consumption_without_storage": 1.4,
            "district_net_electricity_consumption_cost": 0.2,
            "district_net_electricity_consumption_emission": 0.3,
            "electricity_price_mean": 0.14,
            "carbon_intensity_mean": 0.18,
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
            "action_l2": 0.25,
            "action_mean": 0.1,
        }
    ]


class _Candidate:
    adapter = _Adapter()


class _EfficientAdapter(_Adapter):
    trace_record_interval = 10
    trace_detail = "compact"


class _EfficientCandidate:
    adapter = _EfficientAdapter()


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


def test_citylearn_v3_backend_adapter_normalizes_observations_before_training():
    adapter = CityLearnV3BackendAdapter(
        schema_path="CityLearn/data/datasets/baeda_3dem/schema.json",
        scenario="E1",
        seed=0,
        episode_time_steps=4,
        normalize_observations=True,
    )

    try:
        observations = adapter.reset()
        values = np.concatenate([observation.reshape(-1) for observation in observations])

        assert adapter.normalization_metadata["normalize_observations"] is True
        assert np.nanmin(values) >= 0.0
        assert np.nanmax(values) <= 1.0
        assert all(float(space.low.min()) == 0.0 for space in adapter.continuous_observation_space)
        assert all(float(space.high.max()) == 1.0 for space in adapter.ctde_share_observation_space)
    finally:
        adapter.close()


def test_training_artifacts_use_data_checkpoints_and_figures_layout(tmp_path):
    output_dir = tmp_path / "happo" / "E3_seed_7"
    checkpoint_dir = output_dir / "checkpoints" / "backend"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "actor_agent0.pt").write_bytes(b"checkpoint")
    (output_dir / "live_progress.json").write_text('{"global_step": 1}', encoding="utf-8")

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
        assert not (output_dir / file_name).exists()

    assert artifacts["checkpoint_count"] == 1
    assert artifacts["timeseries_rows"] == 2
    assert artifacts["trace_rows"] == 1
    assert artifacts["results_json_root"] is None
    assert artifacts["checkpoint_manifest_root"] is None
    assert artifacts["statistical_comparison_dir"] is None
    assert artifacts["completed_live_progress_removed"] is True
    assert not (output_dir / "live_progress.json").exists()
    assert (figures_dir / "figures_manifest.json").is_file()
    assert (figures_dir / "reward_timeseries.png").is_file()
    assert (figures_dir / "convergence_returns.png").is_file()
    assert (figures_dir / "episode_reward_summary.png").is_file()
    assert (figures_dir / "learning_efficiency.png").is_file()
    assert (figures_dir / "citylearn_v2_district_timeseries.png").is_file()
    assert (figures_dir / "exploration_action_l2.png").is_file()
    assert (figures_dir / "agent_reward_contribution.png").is_file()
    assert (figures_dir / "axis_baseline_comparison.png").is_file()
    assert (figures_dir / "baseline_gain_by_kpi.png").is_file()
    assert (figures_dir / "core_kpis.png").is_file()
    assert (figures_dir / "OE1_flexibility_kpis.png").is_file()
    assert (figures_dir / "OE2_co2_kpis.png").is_file()
    assert (figures_dir / "OE3_cost_kpis.png").is_file()
    assert (tables_dir / "episode_summary.csv").is_file()
    assert (tables_dir / "objective_kpis.csv").is_file()
    assert (tables_dir / "training_efficiency.csv").is_file()
    assert (tables_dir / "exploration_summary.csv").is_file()
    assert (tables_dir / "agent_reward_summary.csv").is_file()
    assert (tables_dir / "building_behavior_summary.csv").is_file()
    assert (tables_dir / "building_kpis.csv").is_file()
    assert (tables_dir / "building_observation_action_schema.csv").is_file()
    assert (tables_dir / "building_trace_sample.csv").is_file()
    assert (tables_dir / "checkpoint_inventory.csv").is_file()
    assert (data_dir / "building_behavior_summary.csv").is_file()
    assert not (output_dir / "building_behavior_summary.csv").exists()

    results = json.loads((data_dir / "results.json").read_text(encoding="utf-8"))
    assert results["artifact_layout"]["data"] == str(data_dir)
    assert results["artifact_layout"]["checkpoints"] == str(output_dir / "checkpoints")
    assert results["figures"]["figure_count"] >= 12
    assert results["building_count"] == 2
    assert results["building_detail"]["building_behavior_summary"]["rows"] == 2
    assert results["normalization"]["normalize_observations"] is True

    checkpoint_manifest = json.loads((data_dir / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert checkpoint_manifest["normalization"]["observation_method"] == "test normalization"


def test_default_artifact_profile_avoids_duplicate_root_and_comparison_artifacts(tmp_path):
    output_dir = tmp_path / "happo" / "E3_seed_7"

    artifacts = write_training_artifacts(
        output_dir=output_dir,
        algorithm="HAPPO",
        backend="external/HARL",
        args=_EfficientArgs(),
        report=_report(),
        candidate=_EfficientCandidate(),
        hyperparameters={"episodes": 1},
    )

    data_dir = output_dir / "data"

    assert (data_dir / "timeseries.csv").is_file()
    assert (data_dir / "trace.csv").is_file()
    assert not (output_dir / "timeseries.csv").exists()
    assert not (output_dir / "trace.csv").exists()
    assert not (output_dir / "results.json").exists()
    assert not (output_dir / "training_summary.json").exists()
    assert not (output_dir.parent.parent / "statistical_comparison").exists()
    assert not (output_dir / "building_behavior_summary.csv").exists()
    assert artifacts["statistical_comparison_dir"] is None

    results = json.loads((data_dir / "results.json").read_text(encoding="utf-8"))
    policy = results["artifact_write_policy"]
    assert policy["legacy_root_artifacts"] is False
    assert policy["root_timeseries_csv"] is False
    assert policy["root_trace_csv"] is False
    assert policy["statistical_comparison_artifacts"] is False
    assert policy["statistical_comparison_trace_csv"] is False
    assert policy["trace_is_sampled"] is True
    assert policy["trace_record_interval"] == 10

    summary_paths = write_training_summary(output_dir, {"artifacts": artifacts})
    assert (data_dir / "training_summary.json").is_file()
    assert summary_paths["training_summary_json_root"] is None
    assert not (output_dir / "training_summary.json").exists()


def test_legacy_root_and_statistical_comparison_artifacts_are_explicit(tmp_path):
    output_dir = tmp_path / "happo" / "E3_seed_7"

    artifacts = write_training_artifacts(
        output_dir=output_dir,
        algorithm="HAPPO",
        backend="external/HARL",
        args=_LegacyComparisonArgs(),
        report=_report(),
        candidate=_Candidate(),
        hyperparameters={"episodes": 1},
    )

    data_dir = output_dir / "data"
    comparison_dir = Path(artifacts["statistical_comparison_dir"])

    for file_name in ["results.json", "timeseries.csv", "trace.csv", "checkpoint_manifest.json"]:
        assert (data_dir / file_name).is_file()
        assert (output_dir / file_name).is_file()

    assert (comparison_dir / "result_happo_E3.json").is_file()
    assert (comparison_dir / "timeseries_happo_E3.csv").is_file()
    assert (comparison_dir / "trace_happo_E3.csv").is_file()

    results = json.loads((data_dir / "results.json").read_text(encoding="utf-8"))
    policy = results["artifact_write_policy"]
    assert policy["legacy_root_artifacts"] is True
    assert policy["statistical_comparison_artifacts"] is True
    assert policy["root_timeseries_csv"] is True
    assert policy["root_trace_csv"] is True
    assert policy["statistical_comparison_trace_csv"] is True

    summary_paths = write_training_summary(output_dir, {"artifacts": artifacts})
    assert Path(summary_paths["training_summary_json"]).is_file()
    assert Path(summary_paths["training_summary_json_root"]).is_file()


def test_artifact_audit_warns_on_short_timeseries():
    rows = [
        {
            "global_step": 0,
            "episode": 0,
            "episode_step": 0,
            "all_done": True,
            "reward_sum": 1.0,
            "reward_mean": 1.0,
        }
    ]
    audit = _artifact_consistency_audit(
        report={"all_values": {}, "report_source": {}},
        timeseries_rows=rows,
        trace_rows=[],
        episode_summaries=[
            {
                "episode": 0,
                "steps": 1,
                "last_all_done": True,
                "last_global_step": 0,
                "last_time_step": 0,
            }
        ],
        expected_episode_time_steps=2,
        expected_episodes=1,
    )

    warning_codes = {item["code"] for item in audit["warnings"]}
    assert audit["status"] == "warning"
    assert audit["expected_timeseries_rows"] == 2
    assert audit["timeseries_rows"] == 1
    assert audit["timeseries_row_delta_vs_expected"] == -1
    assert "unexpected_timeseries_rows" in warning_codes
    assert "episode_step_count_mismatch" in warning_codes


def test_finite_optimizer_guard_initializes_audit_file(tmp_path):
    import torch

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    audit_path = tmp_path / "finite_gradient_guard.jsonl"

    guard = install_finite_optimizer_step_guard(
        [
            {
                "owner": "test_model",
                "module": model,
                "optimizer": optimizer,
            }
        ],
        audit_path,
    )

    records = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert guard["installed_optimizers"] == 1
    assert guard["audit_file_exists"] is True
    assert guard["audit_initialized"] is True
    assert records[0]["event"] == "finite_optimizer_step_guard_installed"
    assert records[0]["installed_optimizers"] == 1
    assert records[0]["owners"] == ["test_model"]
