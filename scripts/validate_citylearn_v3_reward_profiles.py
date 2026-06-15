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

from citylearn.reward_function import CityLearnV3MADRLRewardFunction
from citylearn.v3.environment import describe_environment, make_citylearn_v3_project_env


EXPECTED_AXIS_WEIGHTS = {
    "E1": {"flex": 0.70, "carbon": 0.15, "cost": 0.15},
    "E2": {"flex": 0.15, "carbon": 0.70, "cost": 0.15},
    "E3": {"flex": 0.25, "carbon": 0.15, "cost": 0.60},
}

EXPECTED_PROFILE = {
    "team_reward_ratio": 0.70,
    "ev_weight": 0.25,
    "reward_scale": 1.00,
    "ramp_weight": 0.35,
    "peak_weight": 0.45,
    "ev_soc_tolerance": 0.05,
    "ev_soc_critical_deficit": 0.25,
    "ev_urgency_hours": 4.0,
    "ev_departure_deficit_weight": 0.55,
    "ev_urgency_deficit_weight": 0.30,
    "ev_idle_deficit_weight": 0.15,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--episode-time-steps", default=2, type=int)
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def validate_reinforced_ev_soc_penalty(errors: list[str]) -> dict:
    reward_function = CityLearnV3MADRLRewardFunction(
        {"central_agent": False},
        algorithm="MASAC",
        scenario="E3",
    )
    observation = {
        "electric_vehicles_chargers_dict": {
            "charger_1": {
                "connected": True,
                "previous_battery_soc": 0.40,
                "battery_soc": 0.40,
                "battery_capacity": 40.0,
                "min_capacity": 0.0,
                "last_charged_kwh": 0.0,
                "required_soc": 0.85,
                "hours_until_departure": 0.0,
                "max_charging_power": 7.4,
                "max_discharging_power": 0.0,
            }
        },
        "charging_constraint_violation_kwh": 0.0,
        "net_electricity_consumption": 5.0,
    }
    base_ev_raw = reward_function.calculate_ev_penalty(observation, current_reward=0.0)
    service_constraint = reward_function._ev_service_constraint_term(observation)
    ev_term = reward_function._ev_term(observation)

    if base_ev_raw >= 0.0:
        errors.append(f"synthetic EV SOC deficit: base EV raw reward is {base_ev_raw}, expected negative")
    if service_constraint > -0.99:
        errors.append(f"synthetic EV SOC deficit: service constraint is {service_constraint}, expected <= -0.99")
    if ev_term > -0.99:
        errors.append(f"synthetic EV SOC deficit: ev_term is {ev_term}, expected <= -0.99")

    return {
        "base_ev_raw": float(base_ev_raw),
        "service_constraint": float(service_constraint),
        "ev_term": float(ev_term),
    }


def main() -> int:
    args = parse_args()
    algorithms = ("HAPPO", "MASAC", "MATD3", "MAAC")
    scenarios = ("E1", "E2", "E3")
    rows = []
    errors = []
    ev_soc_penalty_check = validate_reinforced_ev_soc_penalty(errors)

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
                axis_weights = metadata.get("axis_weights") or {}
                profile = metadata.get("profile") or {}
                expected_profile_name = f"{algorithm.lower()}_unified_comparable_v3"

                if metadata.get("function") != "CityLearnV3MADRLRewardFunction":
                    errors.append(f"{algorithm}/{scenario}: reward function is {metadata.get('function')}")
                if metadata.get("not_using_marl_base_weights") is not True:
                    errors.append(f"{algorithm}/{scenario}: reward uses MARL base weights")
                if profile.get("profile_name") != expected_profile_name:
                    errors.append(f"{algorithm}/{scenario}: profile is {profile.get('profile_name')}, expected {expected_profile_name}")

                for key, expected_value in EXPECTED_AXIS_WEIGHTS[scenario].items():
                    actual_value = float(axis_weights.get(key, np.nan))
                    if not np.isclose(actual_value, expected_value, atol=1.0e-9):
                        errors.append(f"{algorithm}/{scenario}: axis {key}={actual_value}, expected {expected_value}")

                for key, expected_value in EXPECTED_PROFILE.items():
                    actual_value = float(profile.get(key, np.nan))
                    if not np.isclose(actual_value, expected_value, atol=1.0e-9):
                        errors.append(f"{algorithm}/{scenario}: profile {key}={actual_value}, expected {expected_value}")

                reward_values = np.asarray(list(rewards.values()), dtype=float)
                if not np.all(np.isfinite(reward_values)):
                    errors.append(f"{algorithm}/{scenario}: non-finite reward values")

                rows.append({
                    "algorithm": algorithm,
                    "scenario": scenario,
                    "reward_function": metadata.get("function"),
                    "profile": metadata.get("profile", {}).get("profile_name"),
                    "profile_parameters": {
                        key: metadata.get("profile", {}).get(key)
                        for key in EXPECTED_PROFILE
                    },
                    "axis_weights": axis_weights,
                    "not_using_marl_base_weights": metadata.get("not_using_marl_base_weights"),
                    "reward_mean_zero_action": float(np.mean(reward_values)),
                    "reward_min_zero_action": float(np.min(reward_values)),
                    "reward_max_zero_action": float(np.max(reward_values)),
                })
            finally:
                env.close()

    payload = {
        "status": "passed" if not errors else "failed",
        "checks": {
            "algorithms": algorithms,
            "scenarios": scenarios,
            "expected_reward_function": "CityLearnV3MADRLRewardFunction",
            "expected_axis_weights": EXPECTED_AXIS_WEIGHTS,
            "expected_profile": EXPECTED_PROFILE,
            "expected_profile_policy": "unified_comparable_v3 for statistical comparability across MADRL backends with reinforced EV SOC service",
            "reinforced_ev_soc_penalty_check": ev_soc_penalty_check,
        },
        "rows": rows,
        "errors": errors,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
