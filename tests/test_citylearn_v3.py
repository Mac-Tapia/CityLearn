import numpy as np
import pytest

from citylearn.v3 import (
    CityLearnV3ExperimentConfig,
    CityLearnV3MARLlibEnv,
    describe_environment,
    evaluate_objectives,
    make_citylearn_v3_env,
    make_citylearn_v3_project_env,
    objective_manifest,
)
from citylearn.v3.backends import citylearn_v3_backend_manifest


def test_citylearn_v3_config_keeps_thesis_experiment_design():
    config = CityLearnV3ExperimentConfig()

    assert config.algorithms == ("HAPPO", "MASAC", "MATD3", "MAAC")
    assert config.scenarios == ("E1", "E2", "E3")
    assert config.experiment_count == 120
    assert config.central_agent is False
    assert config.use_citylearn_v2_kpis is True
    assert config.expose_all_citylearn_v2_kpis is True


def test_citylearn_v3_objective_manifest_covers_thesis_axes():
    manifest = objective_manifest()

    assert set(manifest["axes"]) == {"OE1", "OE2", "OE3"}
    assert manifest["project_axis_metrics"] == ("OE1", "OE2", "OE3")
    assert set(manifest["axes"]["OE1"]["kpis"]).issuperset({
        "peak_average",
        "ramping_average",
        "one_minus_load_factor_average",
        "pv_self_consumption_ratio",
        "battery_throughput_total",
        "ev_v2g_export_total",
    })
    assert manifest["axes"]["OE2"]["name"] == "Emisiones de CO2"
    assert set(manifest["axes"]["OE2"]["kpis"]).issuperset({
        "carbon_emissions",
        "carbon_emissions_control",
        "carbon_emissions_baseline",
        "carbon_emissions_delta",
    })
    assert set(manifest["axes"]["OE3"]["kpis"]).issuperset({
        "electricity_cost",
        "electricity_cost_control",
        "electricity_cost_delta",
        "cost_peak_average",
        "price_signal_deviation",
    })
    assert manifest["axis_kpis"]["price_signal_deviation"]["source"] == "derived_from_citylearn_v2_timeseries"
    assert manifest["axis_kpis"]["carbon_emissions_control"]["citylearn_v2_names"] == (
        "district_emissions_total_control_kgco2",
    )


def test_citylearn_v3_env_uses_citylearn_v2_17_building_ev_schema():
    config = CityLearnV3ExperimentConfig().for_smoke_test(episode_time_steps=4)
    env = make_citylearn_v3_project_env(config, scenario="E1", seed=0)

    try:
        observations, infos = env.reset()
        description = describe_environment(env)

        assert description["simulator"] == "citylearn-v2"
        assert description["num_agents"] == 17
        assert description["has_ev_actions"] is True
        assert description["has_ev_observations"] is True
        assert description["supports_all_citylearn_v2_kpis"] is True
        assert description["state_dim"] == env.state().shape[0]
        assert set(observations) == set(env.possible_agents)
        assert infos["Building_1"]["agent_index"] == 0
    finally:
        env.close()


def test_citylearn_v3_objective_report_uses_citylearn_v2_kpis_and_derived_dr_metric():
    config = CityLearnV3ExperimentConfig().for_smoke_test(episode_time_steps=4)
    env = make_citylearn_v3_project_env(config, scenario="E3", seed=0)

    try:
        env.reset()
        for _ in range(3):
            actions = {
                agent: np.zeros(env.action_space(agent).shape, dtype=np.float32)
                for agent in env.agents
            }
            env.step(actions)

        report = evaluate_objectives(env)

        assert set(report["axes"]) == {"OE1", "OE2", "OE3"}
        assert report["axes"]["OE1"]["kpis"]["peak_average"]["trace"]["source"] == "citylearn_v2.evaluate_v2"
        assert report["axes"]["OE2"]["kpis"]["carbon_emissions_control"]["trace"]["source"] == "citylearn_v2.evaluate_v2"
        assert report["axes"]["OE3"]["kpis"]["electricity_cost"]["value"] == pytest.approx(
            report["axis_kpis"]["electricity_cost"]
        )
        assert report["project_axis_metrics"]["OE2"]["name"] == "Emisiones de CO2"
        assert "carbon_emissions" in report["axis_kpis"]
        assert "carbon_emissions_control" in report["axis_kpis"]
        assert "price_signal_deviation" in report["axis_kpis"]
    finally:
        env.close()


def test_citylearn_v3_env_is_generic_for_citylearn_v2_datasets():
    env = make_citylearn_v3_env(
        schema_path="data/datasets/baeda_3dem/schema.json",
        episode_time_steps=4,
        seed=0,
    )

    try:
        observations, _ = env.reset()
        description = describe_environment(env)
        actions = {
            agent: np.zeros(env.action_space(agent).shape, dtype=np.float32)
            for agent in env.agents
        }
        env.step(actions)
        kpi_frame = env.get_kpi_frame()
        all_kpis = env.get_all_kpis()

        assert description["num_agents"] == 4
        assert description["has_ev_actions"] is False
        assert description["supports_all_citylearn_v2_kpis"] is True
        assert set(observations) == set(env.possible_agents)
        assert {"cost_function", "value", "name", "level"}.issubset(kpi_frame.columns)
        assert "District" in all_kpis
    finally:
        env.close()


def test_citylearn_v3_marllib_adapter_pads_heterogeneous_spaces():
    config = CityLearnV3ExperimentConfig().for_smoke_test(episode_time_steps=4)
    env = CityLearnV3MARLlibEnv({"config": config, "scenario": "E1", "seed": 0})

    try:
        observations = env.reset()
        env_info = env.get_env_info()

        assert env.num_agents == 17
        assert env_info["original_action_dims"]["Building_1"] == 3
        assert env.action_space.shape[0] == max(env.original_action_dims.values())
        assert observations["Building_1"]["obs"].shape == env.observation_space["obs"].shape

        actions = {
            agent: np.zeros(env.action_space.shape, dtype=np.float32)
            for agent in env.agents
        }
        next_observations, rewards, dones, infos = env.step(actions)

        assert len(next_observations) == 17
        assert len(rewards) == 17
        assert dones["__all__"] is False
        assert "individual_reward" in infos["Building_1"]
    finally:
        env.close()


def test_citylearn_v3_backend_manifest_points_to_cloned_official_sources():
    manifest = citylearn_v3_backend_manifest(CityLearnV3ExperimentConfig().for_smoke_test())

    assert manifest["simulator"] == "citylearn-v2"
    assert manifest["backends"]["HAPPO"]["available"] is True
    assert manifest["backends"]["MASAC"]["available"] is True
    assert manifest["backends"]["MATD3"]["available"] is True
    assert manifest["backends"]["MAAC"]["available"] is True
    assert manifest["backends"]["MARLLIB"]["available"] is True
    assert manifest["locked_commits"]["HAPPO"]["commit"] == "b1af98b0dbab72a2eee9d160751cd09aedbb8ce2"


def test_citylearn_agents_do_not_export_local_madrl_implementations():
    import citylearn.agents as agents

    assert "HAPPO" not in agents.__all__
    assert "MASAC" not in agents.__all__
    assert "MATD3" not in agents.__all__
    assert "MAAC" not in agents.__all__

    with pytest.raises(ImportError, match="not implemented locally"):
        getattr(agents, "HAPPO")
