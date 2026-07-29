import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np

from citylearn_v3_training_common import CityLearnV3BackendAdapter, write_training_artifacts


class _Args:
    scenario = "E3"
    seed = 7
    episode_time_steps = 2


class _EfficientArgs(_Args):
    artifact_profile = "efficient"


class _Space:
    def __init__(self, dim):
        self.low = [-1.0] * dim
        self.high = [1.0] * dim


class _KPIFrame:
    def to_dict(self, orient):
        assert orient == "records"
        return [
            {
                "level": "district",
                "name": "District",
                "cost_function": "district_energy_grid_total_import_control_kwh",
                "value": 100.0,
            },
            {
                "level": "district",
                "name": "District",
                "cost_function": "district_emissions_total_control_kgco2",
                "value": 50.0,
            },
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
                "name": "Building_1",
                "cost_function": "building_emissions_total_control_kgco2",
                "value": 5.0,
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

    def action_space(self, _agent):
        return _Space(1)

    def observation_space(self, _agent):
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
        "building_axis_kpis": {},
        "building_objective_kpis": [],
        "building_count": 0,
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
    assert (tables_dir / "building_objective_kpis.csv").is_file()
    assert (tables_dir / "district_kpis.csv").is_file()
    assert (tables_dir / "citylearn_kpi_frame.csv").is_file()
    assert (tables_dir / "building_observation_action_schema.csv").is_file()
    assert (tables_dir / "building_trace_sample.csv").is_file()
    assert (tables_dir / "checkpoint_inventory.csv").is_file()
    assert (data_dir / "building_behavior_summary.csv").is_file()
    assert (data_dir / "building_kpis.csv").is_file()
    assert (data_dir / "building_objective_kpis.csv").is_file()
    assert (data_dir / "district_kpis.csv").is_file()
    assert (data_dir / "citylearn_kpi_frame.csv").is_file()
    assert (output_dir / "building_behavior_summary.csv").is_file()

    results = json.loads((data_dir / "results.json").read_text(encoding="utf-8"))
    assert results["artifact_layout"]["data"] == str(data_dir)
    assert results["artifact_layout"]["checkpoints"] == str(output_dir / "checkpoints")
    assert results["figures"]["figure_count"] >= 12
    assert results["building_count"] == 2
    assert results["building_detail"]["building_behavior_summary"]["rows"] == 2
    assert results["building_detail"]["building_kpis"]["rows"] >= 4
    assert results["building_detail"]["building_objective_kpis"]["rows"] >= 1
    assert results["building_detail"]["district_kpis"]["rows"] >= 1
    assert results["kpi_levels"]["building_evaluate_v2_rows"] >= 4
    assert results["normalization"]["normalize_observations"] is True

    checkpoint_manifest = json.loads((data_dir / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert checkpoint_manifest["normalization"]["observation_method"] == "test normalization"


def test_efficient_artifact_profile_avoids_duplicate_heavy_trace_csv(tmp_path):
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
    comparison_dir = Path(artifacts["statistical_comparison_dir"])

    assert (data_dir / "timeseries.csv").is_file()
    assert (data_dir / "trace.csv").is_file()
    assert (output_dir / "timeseries.csv").is_file()
    assert not (output_dir / "trace.csv").exists()
    assert not (output_dir / "building_behavior_summary.csv").exists()
    assert not (comparison_dir / "trace_happo_E3.csv").exists()

    results = json.loads((data_dir / "results.json").read_text(encoding="utf-8"))
    policy = results["artifact_write_policy"]
    assert policy["root_timeseries_csv"] is True
    assert policy["root_trace_csv"] is False
    assert policy["statistical_comparison_trace_csv"] is False
    assert policy["trace_is_sampled"] is True
    assert policy["trace_record_interval"] == 10


def test_resolve_output_dir_uses_simple_madrl_scenario_layout(tmp_path):
    from citylearn_v3_training_common import resolve_output_dir, resolve_job_run_dir

    out = resolve_output_dir(str(tmp_path / "HAPPO"), "happo", "E2", 0)
    assert out == tmp_path / "HAPPO" / "E2"
    assert (out / "data").mkdir(exist_ok=True) or True

    run = resolve_job_run_dir(tmp_path, "masac", "E3", 0)
    assert run == tmp_path / "MASAC" / "E3"
