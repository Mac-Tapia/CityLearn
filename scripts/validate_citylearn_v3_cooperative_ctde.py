"""Validate the CityLearn v3 MADRL cooperative CTDE contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from citylearn.v3 import describe_environment, make_citylearn_v3_project_env


ALGORITHMS = ["HAPPO", "MASAC", "MATD3", "MAAC"]
SCENARIOS = ["E1", "E2", "E3"]


def validate_contract(seed: int, episode_time_steps: int) -> Dict[str, Any]:
    """Validate cooperative rewards and CTDE state for all MADRL backends."""

    rows: List[Dict[str, Any]] = []

    for algorithm in ALGORITHMS:
        for scenario in SCENARIOS:
            env = make_citylearn_v3_project_env(
                scenario=scenario,
                seed=seed,
                episode_time_steps=episode_time_steps,
                madrl_algorithm=algorithm,
            )
            description = describe_environment(env)
            observation_dim_sum = sum(description["observation_dims"].values())

            if description["reward_aggregation"] != "team_mean":
                raise AssertionError(
                    f"{algorithm} {scenario} reward aggregation is "
                    f"{description['reward_aggregation']!r}, expected 'team_mean'."
                )

            if description["num_agents"] != 17:
                raise AssertionError(
                    f"{algorithm} {scenario} has {description['num_agents']} agents, "
                    "expected 17."
                )

            if description["state_dim"] != observation_dim_sum:
                raise AssertionError(
                    f"{algorithm} {scenario} CTDE state_dim={description['state_dim']} "
                    f"does not match observation sum={observation_dim_sum}."
                )

            if description["reward_metadata"]["not_using_marl_base_weights"] is not True:
                raise AssertionError(
                    f"{algorithm} {scenario} is using MARL base reward weights."
                )

            _, _ = env.reset(seed=seed)
            actions = {
                agent: np.zeros(env.action_space(agent).shape, dtype=np.float32)
                for agent in env.agents
            }
            _, rewards, _, _, infos = env.step(actions)
            rounded_rewards = {round(float(value), 8) for value in rewards.values()}

            if len(rounded_rewards) != 1:
                raise AssertionError(
                    f"{algorithm} {scenario} does not share one team reward: {rewards}."
                )

            if not all("team_reward" in info for info in infos.values()):
                raise AssertionError(f"{algorithm} {scenario} is missing team_reward.")

            if not all("individual_reward" in info for info in infos.values()):
                raise AssertionError(
                    f"{algorithm} {scenario} is missing individual_reward."
                )

            rows.append(
                {
                    "algorithm": algorithm,
                    "scenario": scenario,
                    "agents": description["num_agents"],
                    "reward_aggregation": description["reward_aggregation"],
                    "state_dim": description["state_dim"],
                    "obs_dim_sum": observation_dim_sum,
                    "shared_reward_value": next(iter(rounded_rewards)),
                    "reward_profile": description["reward_metadata"].get("profile"),
                    "not_using_marl_base_weights": description["reward_metadata"].get(
                        "not_using_marl_base_weights"
                    ),
                }
            )
            env.close()

    return {"status": "passed", "rows": rows}


def main() -> None:
    """Run the CLI validator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/validation/cooperative_ctde_validation.json"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-time-steps", type=int, default=2)
    args = parser.parse_args()

    payload = validate_contract(
        seed=args.seed,
        episode_time_steps=args.episode_time_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
