"""MARLlib/RLlib-facing CityLearn v3 environment adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
from gymnasium import spaces

from citylearn.dec_pomdp import DEFAULT_17_BUILDING_EV_SCHEMA
from citylearn.v3.config import CityLearnV3ExperimentConfig
from citylearn.v3.environment import make_citylearn_v3_env


try:  # pragma: no cover - depends on optional MARLlib/RLlib stack.
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except Exception:  # pragma: no cover
    class MultiAgentEnv:  # type: ignore[no-redef]
        """Fallback base so the adapter remains importable without RLlib."""


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MARLLIB_ROOT = PROJECT_ROOT / "external" / "MARLlib"


def _pad_vector(values, target_dim: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)

    if array.size > target_dim:
        raise ValueError(f"Cannot pad vector of size {array.size} to smaller size {target_dim}.")

    output = np.zeros(target_dim, dtype=np.float32)
    output[: array.size] = array
    return output


def _max_box(spaces_by_agent: Mapping[str, spaces.Box], *, low_default: float, high_default: float) -> spaces.Box:
    max_dim = max(int(space.shape[0]) for space in spaces_by_agent.values())
    low = np.full(max_dim, low_default, dtype=np.float32)
    high = np.full(max_dim, high_default, dtype=np.float32)

    for space in spaces_by_agent.values():
        dim = int(space.shape[0])
        low[:dim] = np.minimum(low[:dim], np.asarray(space.low, dtype=np.float32).reshape(-1))
        high[:dim] = np.maximum(high[:dim], np.asarray(space.high, dtype=np.float32).reshape(-1))

    return spaces.Box(low=low, high=high, dtype=np.float32)


class CityLearnV3MARLlibEnv(MultiAgentEnv):
    """RLlib MultiAgentEnv adapter for the CityLearn v3 Dec-POMDP.

    MARLlib examples expect a single observation and action space. CityLearn's
    17-building EV dataset is heterogeneous, so this adapter pads observations
    and actions to the maximum local dimension and slices actions back before
    stepping CityLearn.
    """

    def __init__(self, env_config: Optional[Mapping[str, object]] = None):
        env_config = dict(env_config or {})

        config = env_config.pop("config", None)
        if config is None:
            schema_path = Path(env_config.pop("schema_path", DEFAULT_17_BUILDING_EV_SCHEMA))
            config = CityLearnV3ExperimentConfig(
                schema_path=schema_path,
                episode_time_steps=int(env_config.pop("episode_time_steps", 8760)),
                reward_aggregation=str(env_config.pop("reward_aggregation", "team_mean")),
            )

        if not isinstance(config, CityLearnV3ExperimentConfig):
            raise TypeError("config must be a CityLearnV3ExperimentConfig instance.")

        scenario = env_config.pop("scenario", env_config.pop("map_name", None))
        scenario = None if scenario in {None, "citylearn_v3"} else str(scenario)
        seed = env_config.pop("seed", None)

        self.citylearn_v3 = make_citylearn_v3_env(
            config,
            scenario=scenario,
            seed=None if seed is None else int(seed),
            **env_config,
        )
        self.config = config
        self.env_config = {"scenario": scenario, **env_config}
        self.agents = self.citylearn_v3.possible_agents[:]
        self.num_agents = len(self.agents)
        self._agent_to_id = {agent: i for i, agent in enumerate(self.agents)}

        self.original_observation_spaces = {
            agent: self.citylearn_v3.observation_space(agent)
            for agent in self.agents
        }
        self.original_action_spaces = {
            agent: self.citylearn_v3.action_space(agent)
            for agent in self.agents
        }
        self.original_observation_dims = {
            agent: int(space.shape[0])
            for agent, space in self.original_observation_spaces.items()
        }
        self.original_action_dims = {
            agent: int(space.shape[0])
            for agent, space in self.original_action_spaces.items()
        }

        self._obs_box = _max_box(
            self.original_observation_spaces,
            low_default=-np.inf,
            high_default=np.inf,
        )
        self._act_box = _max_box(
            self.original_action_spaces,
            low_default=-1.0,
            high_default=1.0,
        )
        self.observation_space = spaces.Dict({"obs": self._obs_box})
        self.action_space = self._act_box
        self.observation_spaces = {
            agent: self.observation_space
            for agent in self.agents
        }
        self.action_spaces = {
            agent: self.action_space
            for agent in self.agents
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[Mapping] = None):
        observations, _infos = self.citylearn_v3.reset(seed=seed, options=options)
        return self._format_observations(observations)

    def step(self, action_dict: Mapping[str, np.ndarray]):
        actions = {
            agent: self._unpad_action(agent, action_dict[agent])
            for agent in action_dict
        }
        observations, rewards, terminations, truncations, infos = self.citylearn_v3.step(actions)
        dones = {
            agent: bool(terminations.get(agent, False) or truncations.get(agent, False))
            for agent in self.agents
        }
        dones["__all__"] = any(dones.values())
        return self._format_observations(observations), rewards, dones, infos

    def close(self):
        self.citylearn_v3.close()

    def render(self, mode=None):
        return self.citylearn_v3.render()

    def get_env_info(self) -> Dict[str, object]:
        return {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": int(self.config.episode_time_steps),
            "policy_mapping_info": {
                "citylearn_v3": {
                    "description": "17-building CityLearn v2 + EV Dec-POMDP",
                    "team_prefix": tuple(self.agents),
                    "all_agents_one_policy": False,
                    "one_agent_one_policy": True,
                }
            },
            "agent_order": self.agents,
            "original_observation_dims": self.original_observation_dims,
            "original_action_dims": self.original_action_dims,
        }

    def _format_observations(self, observations: Mapping[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        return {
            agent: {"obs": _pad_vector(observation, self._obs_box.shape[0])}
            for agent, observation in observations.items()
        }

    def _unpad_action(self, agent: str, action: np.ndarray) -> np.ndarray:
        dim = self.original_action_dims[agent]
        return np.asarray(action, dtype=np.float32).reshape(-1)[:dim]


def register_citylearn_v3_marllib_env(name: str = "citylearn_v3") -> str:
    """Register the adapter in MARLlib's environment registry when available."""

    if str(MARLLIB_ROOT) not in sys.path and MARLLIB_ROOT.exists():
        sys.path.insert(0, str(MARLLIB_ROOT))

    try:
        from marllib.envs.base_env import ENV_REGISTRY
    except Exception as exc:
        raise ImportError(
            "MARLlib could not be imported. Install MARLlib dependencies in a "
            "compatible environment before launching training."
        ) from exc

    ENV_REGISTRY[name] = CityLearnV3MARLlibEnv
    return name
