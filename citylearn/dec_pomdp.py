"""PettingZoo Dec-POMDP wrapper for CityLearn MADRL experiments.

The thesis experiments use CityLearn as a decentralized partially observable
Markov game: each building observes and controls only its own local features,
while training algorithms may use the concatenated global state for CTDE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from citylearn.citylearn import CityLearnEnv
from citylearn.madrl_kpis import (
    evaluate_citylearn_v2_all_kpis,
    evaluate_citylearn_v2_kpi_frame,
    evaluate_citylearn_v2_kpis,
)


DEFAULT_17_BUILDING_EV_SCHEMA = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "datasets"
    / "citylearn_challenge_2022_phase_all_plus_evs"
    / "schema.json"
)
CITYLEARN_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def resolve_citylearn_schema_path(schema_path: Path | str) -> Path | str:
    """Resolve local schema paths without blocking CityLearn dataset names."""

    path = Path(schema_path).expanduser()

    if path.exists():
        return path

    package_relative_path = CITYLEARN_PACKAGE_ROOT / path

    if package_relative_path.exists():
        return package_relative_path

    return schema_path


class CityLearnDecPOMDPEnv(ParallelEnv):
    """Parallel PettingZoo environment backed by decentralized CityLearn.

    Agents are CityLearn buildings. EV flexibility remains attached to the
    building action/observation spaces through CityLearn's charger features.
    Rewards can be exposed as individual values or as a shared team reward for
    cooperative MARL.
    """

    metadata = {"name": "citylearn_dec_pomdp_v0", "render_modes": []}

    def __init__(
        self,
        env: CityLearnEnv,
        *,
        reward_aggregation: str = "team_mean",
        scenario: Optional[str] = None,
    ):
        self.env = env
        self.reward_aggregation = reward_aggregation
        self.scenario = scenario

        self.possible_agents = self._agent_names()
        self.agents = self.possible_agents[:]
        self._agent_index = {agent: i for i, agent in enumerate(self.possible_agents)}

        self._observation_spaces = {
            agent: self._copy_box_space(space)
            for agent, space in zip(self.possible_agents, self.env.observation_space)
        }
        self._action_spaces = {
            agent: self._copy_box_space(space)
            for agent, space in zip(self.possible_agents, self.env.action_space)
        }
        self.state_space = self._build_state_space()
        self._last_observations: Dict[str, np.ndarray] = {}

    @property
    def num_agents(self) -> int:
        return len(self.possible_agents)

    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_spaces[agent]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Mapping] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        observations, info = self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        self._last_observations = self._observations_to_dict(observations)
        infos = {agent: dict(info) for agent in self.agents}

        for agent in self.agents:
            infos[agent]["agent_index"] = self._agent_index[agent]

        return self._last_observations, infos

    def step(self, actions: Mapping[str, Iterable[float]]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        missing = [agent for agent in self.agents if agent not in actions]
        if missing:
            raise KeyError(f"Missing actions for active CityLearn agents: {missing}")

        ordered_actions = [
            self._clip_action(agent, actions[agent])
            for agent in self.agents
        ]

        result = self.env.step(ordered_actions)
        if len(result) == 5:
            observations, rewards, terminated, truncated, info = result
        elif len(result) == 4:
            observations, rewards, done, info = result
            terminated, truncated = bool(done), False
        else:
            raise ValueError(f"Unexpected CityLearn step result with {len(result)} values.")

        observation_dict = self._observations_to_dict(observations)
        individual_rewards = self._reward_vector(rewards)
        reward_dict = self._aggregate_rewards(individual_rewards)
        terminations = {agent: bool(terminated) for agent in self.agents}
        truncations = {agent: bool(truncated) for agent in self.agents}
        infos = self._infos(info, individual_rewards)

        self._last_observations = observation_dict

        if bool(terminated) or bool(truncated):
            self.agents = []

        return observation_dict, reward_dict, terminations, truncations, infos

    def state(self) -> np.ndarray:
        """Return the CTDE global state as concatenated local observations."""

        if not self._last_observations:
            observations = getattr(self.env, "observations", [])
            self._last_observations = self._observations_to_dict(observations)

        values = [
            np.asarray(self._last_observations[agent], dtype=np.float32).reshape(-1)
            for agent in self.possible_agents
            if agent in self._last_observations
        ]

        if not values:
            return np.zeros(self.state_space.shape, dtype=np.float32)

        return np.concatenate(values).astype(np.float32, copy=False)

    def get_kpis(self) -> Dict[str, float]:
        """Return CityLearn v2 KPI summary for reporting."""

        return evaluate_citylearn_v2_kpis(self.env.unwrapped)

    def get_kpi_frame(self):
        """Return the complete CityLearn v2 KPI DataFrame."""

        return evaluate_citylearn_v2_kpi_frame(self.env.unwrapped)

    def get_all_kpis(self) -> Dict[str, Dict[str, float]]:
        """Return all CityLearn v2 KPIs grouped by district/building name."""

        return evaluate_citylearn_v2_all_kpis(self.env.unwrapped)

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()

    def _agent_names(self) -> List[str]:
        buildings = getattr(self.env.unwrapped, "buildings", [])
        names = [getattr(building, "name", None) for building in buildings]
        names = [str(name) for name in names if name is not None]

        if len(names) == len(self.env.action_space):
            return names

        return [f"Building_{i + 1}" for i in range(len(self.env.action_space))]

    @staticmethod
    def _copy_box_space(space: spaces.Space) -> spaces.Box:
        if not isinstance(space, spaces.Box):
            raise TypeError(f"CityLearn MADRL expects Box spaces, got {type(space).__name__}.")

        return spaces.Box(
            low=np.asarray(space.low, dtype=np.float32),
            high=np.asarray(space.high, dtype=np.float32),
            dtype=np.float32,
        )

    def _build_state_space(self) -> spaces.Box:
        lows = []
        highs = []

        for agent in self.possible_agents:
            space = self._observation_spaces[agent]
            lows.append(np.asarray(space.low, dtype=np.float32).reshape(-1))
            highs.append(np.asarray(space.high, dtype=np.float32).reshape(-1))

        return spaces.Box(
            low=np.concatenate(lows).astype(np.float32, copy=False),
            high=np.concatenate(highs).astype(np.float32, copy=False),
            dtype=np.float32,
        )

    def _observations_to_dict(self, observations) -> Dict[str, np.ndarray]:
        if isinstance(observations, Mapping):
            return {
                str(agent): np.asarray(value, dtype=np.float32)
                for agent, value in observations.items()
            }

        return {
            agent: np.asarray(observations[i], dtype=np.float32)
            for i, agent in enumerate(self.possible_agents)
        }

    def _clip_action(self, agent: str, action) -> np.ndarray:
        space = self._action_spaces[agent]
        array = np.asarray(action, dtype=np.float32)
        return np.clip(array, space.low, space.high).astype(np.float32, copy=False)

    def _reward_vector(self, rewards) -> np.ndarray:
        array = np.asarray(rewards, dtype=np.float32).reshape(-1)

        if array.size == self.num_agents:
            return array

        scalar = float(array.mean()) if array.size > 0 else 0.0
        return np.full(self.num_agents, scalar, dtype=np.float32)

    def _aggregate_rewards(self, individual_rewards: np.ndarray) -> Dict[str, float]:
        if self.reward_aggregation == "individual":
            values = individual_rewards
        elif self.reward_aggregation == "team_sum":
            values = np.full(self.num_agents, float(individual_rewards.sum()), dtype=np.float32)
        elif self.reward_aggregation == "mixed":
            team_mean = float(individual_rewards.mean()) if individual_rewards.size > 0 else 0.0
            values = 0.7 * individual_rewards + 0.3 * team_mean
        elif self.reward_aggregation == "team_mean":
            team_mean = float(individual_rewards.mean()) if individual_rewards.size > 0 else 0.0
            values = np.full(self.num_agents, team_mean, dtype=np.float32)
        else:
            raise ValueError(
                "reward_aggregation must be one of "
                "{'individual', 'team_mean', 'team_sum', 'mixed'}."
            )

        return {
            agent: float(values[self._agent_index[agent]])
            for agent in self.possible_agents
        }

    def _infos(self, info: Mapping, individual_rewards: np.ndarray) -> Dict[str, dict]:
        base_info = dict(info)
        team_reward = float(individual_rewards.mean()) if individual_rewards.size > 0 else 0.0

        return {
            agent: {
                **base_info,
                "agent_index": self._agent_index[agent],
                "individual_reward": float(individual_rewards[self._agent_index[agent]]),
                "team_reward": team_reward,
                "scenario": self.scenario,
            }
            for agent in self.possible_agents
        }


def make_citylearn_dec_pomdp(
    schema_path: Path | str,
    *,
    episode_time_steps: Optional[int] = None,
    random_seed: Optional[int] = None,
    scenario: Optional[str] = None,
    reward_aggregation: str = "team_mean",
    normalize_observations: bool = False,
    offline: bool = True,
    **citylearn_kwargs,
) -> CityLearnDecPOMDPEnv:
    """Create a Dec-POMDP wrapper for any CityLearn v2 schema/dataset."""

    schema = resolve_citylearn_schema_path(schema_path)
    env = CityLearnEnv(
        str(schema),
        central_agent=False,
        episode_time_steps=episode_time_steps,
        random_seed=random_seed,
        offline=offline,
        **citylearn_kwargs,
    )

    if scenario is not None:
        from citylearn.scenario_manager import ScenarioManager

        manager = ScenarioManager()
        manager.select_scenario(scenario)
        manager.apply_scenario_modifications(env)

    if normalize_observations:
        from citylearn.wrappers import NormalizedObservationWrapper

        env = NormalizedObservationWrapper(env)

    return CityLearnDecPOMDPEnv(
        env,
        reward_aggregation=reward_aggregation,
        scenario=scenario,
    )


def make_citylearn_17_building_ev_dec_pomdp(
    schema_path: Optional[Path | str] = None,
    *,
    episode_time_steps: Optional[int] = None,
    random_seed: Optional[int] = None,
    scenario: Optional[str] = None,
    reward_aggregation: str = "team_mean",
    normalize_observations: bool = False,
    offline: bool = True,
    **citylearn_kwargs,
) -> CityLearnDecPOMDPEnv:
    """Create the project default 17-building + EV Dec-POMDP environment."""

    schema = Path(schema_path) if schema_path is not None else DEFAULT_17_BUILDING_EV_SCHEMA
    return make_citylearn_dec_pomdp(
        schema,
        episode_time_steps=episode_time_steps,
        random_seed=random_seed,
        scenario=scenario,
        reward_aggregation=reward_aggregation,
        normalize_observations=normalize_observations,
        offline=offline,
        **citylearn_kwargs,
    )
