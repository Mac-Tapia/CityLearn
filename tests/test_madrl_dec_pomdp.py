import numpy as np
import pandas as pd
import pytest
from gymnasium import spaces

from citylearn.dec_pomdp import CityLearnDecPOMDPEnv, make_citylearn_17_building_ev_dec_pomdp
from citylearn.madrl_kpis import extract_citylearn_v2_kpis, extract_citylearn_v2_metrics
from citylearn.official_madrl import OFFICIAL_MADRL_SOURCES, official_backend_status


pytest.importorskip("pettingzoo")


class _Building:
    def __init__(self, name):
        self.name = name


class _DummyCityLearn:
    def __init__(self, n_agents=3):
        self.unwrapped = self
        self.buildings = [_Building(f"Building_{i + 1}") for i in range(n_agents)]
        self.observation_space = [
            spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
            for _ in range(n_agents)
        ]
        self.action_space = [
            spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            for _ in range(n_agents)
        ]
        self.observations = []
        self.render_called = False

    def reset(self, seed=None, options=None):
        self.observations = [
            np.full(space.shape, i, dtype=np.float32)
            for i, space in enumerate(self.observation_space)
        ]
        return self.observations, {"seed": seed}

    def step(self, actions):
        self.observations = [
            np.asarray(obs, dtype=np.float32) + 1.0
            for obs in self.observations
        ]
        rewards = [-float(np.abs(action).sum()) for action in actions]
        return self.observations, rewards, False, False, {"raw_reward_count": len(rewards)}

    def evaluate_v2(self):
        return pd.DataFrame(
            [
                {
                    "level": "district",
                    "name": "District",
                    "cost_function": "district_energy_grid_shape_quality_peak_all_time_average_to_baseline_ratio",
                    "value": 0.91,
                },
                {
                    "level": "district",
                    "name": "District",
                    "cost_function": "district_energy_grid_shape_quality_ramping_average_to_baseline_ratio",
                    "value": 0.87,
                },
                {
                    "level": "district",
                    "name": "District",
                    "cost_function": "district_cost_total_control_eur",
                    "value": 123.0,
                },
                {
                    "level": "district",
                    "name": "District",
                    "cost_function": "district_cost_total_baseline_eur",
                    "value": 150.0,
                },
                {
                    "level": "district",
                    "name": "District",
                    "cost_function": "district_cost_ratio_to_baseline_total_ratio",
                    "value": 0.82,
                },
                {
                    "level": "district",
                    "name": "District",
                    "cost_function": "district_emissions_ratio_to_baseline_total_ratio",
                    "value": 0.76,
                },
            ]
        )

    def close(self):
        pass

    def render(self):
        self.render_called = True


def test_dec_pomdp_parallel_env_contract_and_team_reward():
    env = CityLearnDecPOMDPEnv(_DummyCityLearn(), reward_aggregation="team_mean", scenario="E1")
    obs, infos = env.reset(seed=11)

    assert env.possible_agents == ["Building_1", "Building_2", "Building_3"]
    assert set(obs) == set(env.possible_agents)
    assert env.state().shape == env.state_space.shape
    assert infos["Building_1"]["agent_index"] == 0

    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    next_obs, rewards, terminations, truncations, step_infos = env.step(actions)

    assert set(next_obs) == set(env.possible_agents)
    assert len(set(round(value, 8) for value in rewards.values())) == 1
    assert not any(terminations.values())
    assert not any(truncations.values())
    assert step_infos["Building_2"]["scenario"] == "E1"

    kpis = env.get_kpis()
    assert kpis["peak_average"] == pytest.approx(0.91)
    assert kpis["electricity_cost"] == pytest.approx(0.82)


def test_citylearn_v2_kpi_extractor_uses_exact_v2_names():
    frame = pd.DataFrame(
        [
            {
                "level": "district",
                "name": "District",
                "cost_function": "district_energy_grid_shape_quality_load_factor_penalty_daily_average_to_baseline_ratio",
                "value": 0.25,
            },
            {
                "level": "district",
                "name": "District",
                "cost_function": "district_emissions_ratio_to_baseline_total_ratio",
                "value": 0.76,
            },
            {
                "level": "district",
                "name": "District",
                "cost_function": "district_emissions_total_control_kgco2",
                "value": 45.0,
            },
            {
                "level": "district",
                "name": "District",
                "cost_function": "district_emissions_total_baseline_kgco2",
                "value": 60.0,
            },
            {
                "level": "district",
                "name": "District",
                "cost_function": "district_emissions_total_delta_kgco2",
                "value": -15.0,
            },
            {
                "level": "district",
                "name": "District",
                "cost_function": "district_solar_self_consumption_ratio_self_consumption_ratio",
                "value": 0.62,
            },
            {
                "level": "district",
                "name": "District",
                "cost_function": "district_battery_total_throughput_kwh",
                "value": 12.5,
            },
            {
                "level": "district",
                "name": "District",
                "cost_function": "district_ev_performance_departure_success_ratio",
                "value": 0.99,
            },
            {
                "level": "district",
                "name": "District",
                "cost_function": "district_emissions_ratio_to_baseline_total_ratio",
                "value": 0.76,
            },
        ]
    )

    kpis = extract_citylearn_v2_kpis(frame)
    metrics = extract_citylearn_v2_metrics(frame)

    assert kpis["one_minus_load_factor_average"] == pytest.approx(0.25)
    assert kpis["carbon_emissions"] == pytest.approx(0.76)
    assert kpis["carbon_emissions_control"] == pytest.approx(45.0)
    assert kpis["carbon_emissions_delta"] == pytest.approx(-15.0)
    assert kpis["pv_self_consumption_ratio"] == pytest.approx(0.62)
    assert kpis["battery_throughput_total"] == pytest.approx(12.5)
    assert kpis["ev_departure_success_rate"] == pytest.approx(0.99)
    assert metrics == {}


def test_official_madrl_sources_are_registered_without_local_invention():
    status = official_backend_status(["HAPPO", "MASAC", "MATD3", "MAAC", "MARLLIB"])

    assert set(status) == {"HAPPO", "MASAC", "MATD3", "MAAC", "MARLLIB"}
    assert OFFICIAL_MADRL_SOURCES["HAPPO"].repository_url == "https://github.com/PKU-MARL/HARL"
    assert OFFICIAL_MADRL_SOURCES["MASAC"].repository_url == "https://github.com/puyuan1996/MARL"
    assert OFFICIAL_MADRL_SOURCES["MATD3"].repository_url == "https://github.com/JohannesAck/MATD3implementation"
    assert OFFICIAL_MADRL_SOURCES["MAAC"].repository_url == "https://github.com/shariqiqbal2810/MAAC"
    assert status["MARLLIB"]["backend_package"] == "marllib"


def test_real_17_building_ev_dataset_dec_pomdp_smoke():
    env = make_citylearn_17_building_ev_dec_pomdp(
        episode_time_steps=4,
        random_seed=3,
        reward_aggregation="team_mean",
    )

    try:
        observations, _ = env.reset()
        assert len(env.possible_agents) == 17
        assert len(observations) == 17
        assert env.state().shape == env.state_space.shape

        action_names = sum(getattr(env.env, "action_names", []), [])
        assert any(name.startswith("electric_vehicle_storage") for name in action_names)

        actions = {
            agent: np.zeros(env.action_space(agent).shape, dtype=np.float32)
            for agent in env.agents
        }
        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        assert len(next_observations) == 17
        assert len(rewards) == 17
        assert len(set(round(value, 8) for value in rewards.values())) == 1
        assert "individual_reward" in infos["Building_1"]
        assert not any(terminations.values())
        assert not any(truncations.values())
    finally:
        env.close()
