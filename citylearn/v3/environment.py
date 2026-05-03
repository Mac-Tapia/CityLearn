"""CityLearn v3 environment factory and introspection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from citylearn.dec_pomdp import (
    DEFAULT_17_BUILDING_EV_SCHEMA,
    CityLearnDecPOMDPEnv,
    make_citylearn_dec_pomdp,
)
from citylearn.v3.config import CityLearnV3ExperimentConfig


def make_citylearn_v3_env(
    config: Optional[CityLearnV3ExperimentConfig] = None,
    *,
    schema_path: Optional[Path | str] = None,
    scenario: Optional[str] = None,
    seed: Optional[int] = None,
    episode_time_steps: Optional[int] = None,
    reward_aggregation: Optional[str] = None,
    **citylearn_kwargs,
) -> CityLearnDecPOMDPEnv:
    """Build a CityLearn v3 Dec-POMDP for any CityLearn v2 schema."""

    config = CityLearnV3ExperimentConfig() if config is None else config
    merged_citylearn_kwargs = {**config.citylearn_kwargs, **citylearn_kwargs}

    return make_citylearn_dec_pomdp(
        schema_path or config.schema_path,
        episode_time_steps=episode_time_steps or config.episode_time_steps,
        random_seed=seed,
        scenario=scenario,
        reward_aggregation=reward_aggregation or config.reward_aggregation,
        offline=True,
        **merged_citylearn_kwargs,
    )


def make_citylearn_v3_project_env(
    config: Optional[CityLearnV3ExperimentConfig] = None,
    *,
    scenario: Optional[str] = None,
    seed: Optional[int] = None,
    episode_time_steps: Optional[int] = None,
    reward_aggregation: Optional[str] = None,
) -> CityLearnDecPOMDPEnv:
    """Build this project's default 17-building + EV CityLearn v3 environment."""

    config = CityLearnV3ExperimentConfig() if config is None else config

    return make_citylearn_v3_env(
        config,
        schema_path=DEFAULT_17_BUILDING_EV_SCHEMA,
        scenario=scenario,
        seed=seed,
        episode_time_steps=episode_time_steps,
        reward_aggregation=reward_aggregation,
    )


def describe_environment(env: CityLearnDecPOMDPEnv) -> Dict[str, object]:
    """Return a compact, testable description of a CityLearn v3 environment."""

    action_names = [
        name
        for building_actions in getattr(env.env, "action_names", [])
        for name in building_actions
    ]
    observation_names = [
        name
        for building_observations in getattr(env.env, "observation_names", [])
        for name in building_observations
    ]

    return {
        "version_layer": "citylearn-v3-madrl",
        "simulator": "citylearn-v2",
        "num_agents": env.num_agents,
        "agents": env.possible_agents,
        "state_dim": int(env.state_space.shape[0]),
        "observation_dims": {
            agent: int(env.observation_space(agent).shape[0])
            for agent in env.possible_agents
        },
        "action_dims": {
            agent: int(env.action_space(agent).shape[0])
            for agent in env.possible_agents
        },
        "has_ev_actions": any(name.startswith("electric_vehicle_storage") for name in action_names),
        "has_ev_observations": any("electric_vehicle" in name for name in observation_names),
        "supports_all_citylearn_v2_kpis": hasattr(env.env.unwrapped, "evaluate_v2"),
        "reward_aggregation": env.reward_aggregation,
        "scenario": env.scenario,
    }
