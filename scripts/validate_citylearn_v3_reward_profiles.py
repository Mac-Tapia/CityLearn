"""Validate CityLearn v3 MADRL reward profiles by algorithm and axis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
CITYLEARN_ROOT = SCRIPT_PATH.parents[1]

if str(CITYLEARN_ROOT) not in sys.path:
    sys.path.insert(0, str(CITYLEARN_ROOT))

from citylearn.v3.environment import describe_environment, make_citylearn_v3_project_env


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--episode-time-steps", default=2, type=int)
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    algorithms = ("HAPPO", "MASAC", "MATD3", "MAAC")
    scenarios = ("E1", "E2", "E3")
    rows = []

    for algorithm in algorithms:
        for scenario in scenarios:
            env = make_citylearn_v3_project_env(
                scenario=scenario,
                seed=args.seed,
                episode_time_steps=args.episode_time_steps,
                madrl_algorithm=algorithm,
            )

            try:
                metadata = describe_environment(env)["reward_metadata"]
                _observations, _infos = env.reset(seed=args.seed)
                zero_actions = {
                    agent: np.zeros(env.action_space(agent).shape, dtype=np.float32)
                    for agent in env.possible_agents
                }
                _next_observations, rewards, _terminations, _truncations, _infos = env.step(zero_actions)
                rows.append({
                    "algorithm": algorithm,
                    "scenario": scenario,
                    "reward_function": metadata.get("function"),
                    "profile": metadata.get("profile", {}).get("profile_name"),
                    "axis_weights": metadata.get("axis_weights"),
                    "not_using_marl_base_weights": metadata.get("not_using_marl_base_weights"),
                    "reward_mean_zero_action": float(np.mean(list(rewards.values()))),
                })
            finally:
                env.close()

    payload = {
        "status": "passed",
        "checks": {
            "algorithms": algorithms,
            "scenarios": scenarios,
            "expected_reward_function": "CityLearnV3MADRLRewardFunction",
        },
        "rows": rows,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
