"""CityLearn v3 environment factory and introspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

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
    normalize_observations: Optional[bool] = None,
    madrl_algorithm: Optional[str] = None,
    use_citylearn_v3_reward: bool = True,
    **citylearn_kwargs,
) -> CityLearnDecPOMDPEnv:
    """Build a CityLearn v3 Dec-POMDP for any CityLearn v2 schema."""

    config = CityLearnV3ExperimentConfig() if config is None else config
    merged_citylearn_kwargs = {**config.citylearn_kwargs, **citylearn_kwargs}

    if use_citylearn_v3_reward:
        reward_kwargs = dict(merged_citylearn_kwargs.get("reward_function_kwargs") or {})
        reward_kwargs.setdefault("algorithm", madrl_algorithm or "MADRL")
        reward_kwargs.setdefault("scenario", scenario or "E1")
        merged_citylearn_kwargs.setdefault(
            "reward_function",
            "citylearn.reward_function.CityLearnV3MADRLRewardFunction",
        )
        merged_citylearn_kwargs["reward_function_kwargs"] = reward_kwargs

    return make_citylearn_dec_pomdp(
        schema_path or config.schema_path,
        episode_time_steps=episode_time_steps or config.episode_time_steps,
        random_seed=seed,
        scenario=scenario,
        reward_aggregation=reward_aggregation or config.reward_aggregation,
        normalize_observations=(
            config.normalize_observations if normalize_observations is None else normalize_observations
        ),
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
    normalize_observations: Optional[bool] = None,
    madrl_algorithm: Optional[str] = None,
    use_citylearn_v3_reward: bool = True,
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
        normalize_observations=normalize_observations,
        madrl_algorithm=madrl_algorithm,
        use_citylearn_v3_reward=use_citylearn_v3_reward,
    )


def describe_environment(env: CityLearnDecPOMDPEnv) -> Dict[str, Any]:
    """Return a compact, testable description of a CityLearn v3 environment."""

    exposed_env = getattr(env, "env", env)
    core_env = getattr(exposed_env, "unwrapped", exposed_env)
    action_names = [
        name
        for building_actions in getattr(core_env, "action_names", [])
        for name in building_actions
    ]
    observation_names = [
        name
        for building_observations in getattr(exposed_env, "observation_names", getattr(core_env, "observation_names", []))
        for name in building_observations
    ]
    reward_function = getattr(core_env, "reward_function", None)

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
        "supports_all_citylearn_v2_kpis": hasattr(core_env, "evaluate_v2"),
        "reward_aggregation": env.reward_aggregation,
        "reward_function": reward_function.__class__.__name__ if reward_function is not None else None,
        "reward_metadata": getattr(reward_function, "metadata", {}),
        "normalize_observations": exposed_env.__class__.__name__ == "NormalizedObservationWrapper",
        "scenario": env.scenario,
    }
