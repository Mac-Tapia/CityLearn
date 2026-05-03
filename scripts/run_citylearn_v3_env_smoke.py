"""Run a short CityLearn v3 Dec-POMDP environment smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
CITYLEARN_ROOT = SCRIPT_PATH.parents[1]

if str(CITYLEARN_ROOT) not in sys.path:
    sys.path.insert(0, str(CITYLEARN_ROOT))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.integer, np.floating)):
        return value.item()

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]

    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-path", default=None, help="Optional CityLearn v2 schema path.")
    parser.add_argument("--scenario", default="E1", help="Scenario label for the smoke run.")
    parser.add_argument("--seed", default=0, type=int, help="Environment seed.")
    parser.add_argument("--episode-time-steps", default=4, type=int, help="Tiny episode length for smoke testing.")
    parser.add_argument("--steps", default=3, type=int, help="Number of zero-action steps to execute.")
    args = parser.parse_args()

    from citylearn.v3 import describe_environment, make_citylearn_v3_env, make_citylearn_v3_project_env

    if args.schema_path is None:
        env = make_citylearn_v3_project_env(
            scenario=args.scenario,
            seed=args.seed,
            episode_time_steps=args.episode_time_steps,
        )
    else:
        env = make_citylearn_v3_env(
            schema_path=args.schema_path,
            scenario=args.scenario,
            seed=args.seed,
            episode_time_steps=args.episode_time_steps,
        )

    try:
        observations, infos = env.reset(seed=args.seed)
        executed_steps = 0
        last_rewards = {}
        last_terminations = {}
        last_truncations = {}

        for _ in range(max(args.steps, 0)):
            actions = {
                agent: np.zeros(env.action_space(agent).shape, dtype=np.float32)
                for agent in env.possible_agents
            }
            observations, last_rewards, last_terminations, last_truncations, infos = env.step(actions)
            executed_steps += 1

            if any(last_terminations.values()) or any(last_truncations.values()):
                break

        kpi_frame = env.get_kpi_frame()
        result = {
            "description": describe_environment(env),
            "executed_steps": executed_steps,
            "active_agents": sorted(observations),
            "last_reward_mean": float(np.mean(list(last_rewards.values()))) if last_rewards else None,
            "terminated": any(last_terminations.values()) if last_terminations else False,
            "truncated": any(last_truncations.values()) if last_truncations else False,
            "kpi_frame_shape": kpi_frame.shape,
            "kpi_summary": env.get_kpis(),
        }
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
