"""Validate CityLearn v3 MADRL reward profiles by algorithm and axis."""
# This white-box validator intentionally verifies the reward's internal EV
# component terms in addition to its public aggregate.
# pylint: disable=protected-access

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
CITYLEARN_ROOT = SCRIPT_PATH.parents[1]

if str(CITYLEARN_ROOT) not in sys.path:
    sys.path.insert(0, str(CITYLEARN_ROOT))

from citylearn.reward_function import CityLearnV3MADRLRewardFunction
from citylearn.v3.environment import describe_environment, make_citylearn_v3_project_env


EXPECTED_AXIS_WEIGHTS = {
    "E1": {"flex": 0.60, "carbon": 0.15, "cost": 0.25},
    "E2": {"flex": 0.25, "carbon": 0.60, "cost": 0.15},
    "E3": {"flex": 0.15, "carbon": 0.25, "cost": 0.60},
}

EXPECTED_WEIGHTING_METHOD = "compensatory_hierarchical_weighted_sum"
EXPECTED_HIERARCHY_LEVELS = {"primary": 0.60, "secondary": 0.25, "residual": 0.15}
EXPECTED_PRIORITY_ORDER = {
    "E1": ("flex", "cost", "carbon"),
    "E2": ("carbon", "flex", "cost"),
    "E3": ("cost", "carbon", "flex"),
}

EXPECTED_PROFILE = {
    "team_reward_ratio": 0.70,
    "ev_weight": 0.25,
    "reward_scale": 1.00,
    "ramp_weight": 0.35,
    "peak_weight": 0.45,
    "bess_cycle_weight": 0.10,
    "bess_cycle_scale": 0.05,
    "ev_soc_tolerance": 0.05,
    "ev_soc_critical_deficit": 0.25,
    "ev_urgency_hours": 8.0,
    "ev_departure_deficit_weight": 0.70,
    "ev_urgency_deficit_weight": 0.30,
    "ev_idle_deficit_weight": 0.25,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--episode-time-steps", default=2, type=int)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("HAPPO", "MASAC", "MATD3", "MAAC"),
        default=("HAPPO", "MASAC", "MATD3", "MAAC"),
        help="Optional subset for a faster integration check.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("E1", "E2", "E3"),
        default=("E1", "E2", "E3"),
        help="Optional subset for a faster integration check.",
    )
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


def _zero_action(env: Any, agent: str) -> np.ndarray:
    space = env.action_space(agent)
    if space.shape is None:
        raise ValueError(f"Expected a shaped action space for {agent!r}")
    return np.zeros(space.shape, dtype=np.float32)


def validate_hierarchical_design(errors: list[str]) -> dict:
    axes = ("flex", "carbon", "cost")
    expected_levels = sorted(EXPECTED_HIERARCHY_LEVELS.values())
    row_sums = {}
    column_assignments = {}

    primary = EXPECTED_HIERARCHY_LEVELS["primary"]
    secondary = EXPECTED_HIERARCHY_LEVELS["secondary"]
    residual = EXPECTED_HIERARCHY_LEVELS["residual"]
    if not primary > secondary > residual > 0.0:
        errors.append("hierarchy levels do not satisfy primary > secondary > residual > 0")
    if primary <= secondary + residual:
        errors.append("primary hierarchy weight does not exceed the combined non-primary weights")
    if not np.isclose(primary + secondary + residual, 1.0, atol=1.0e-12):
        errors.append("hierarchy levels do not sum to 1.0")

    for scenario, weights in EXPECTED_AXIS_WEIGHTS.items():
        row_sum = float(sum(weights.values()))
        row_sums[scenario] = row_sum
        if not np.isclose(row_sum, 1.0, atol=1.0e-12):
            errors.append(f"{scenario}: hierarchy row sums to {row_sum}, expected 1.0")
        order = EXPECTED_PRIORITY_ORDER[scenario]
        actual_order = tuple(sorted(axes, key=lambda axis: weights[axis], reverse=True))
        if actual_order != order:
            errors.append(f"{scenario}: priority order is {actual_order}, expected {order}")

    for axis in axes:
        assigned = sorted(EXPECTED_AXIS_WEIGHTS[scenario][axis] for scenario in EXPECTED_AXIS_WEIGHTS)
        column_assignments[axis] = assigned
        if not np.allclose(assigned, expected_levels, atol=1.0e-12):
            errors.append(f"axis {axis}: hierarchy assignments {assigned}, expected {expected_levels}")

    return {
        "weighting_method": EXPECTED_WEIGHTING_METHOD,
        "levels": EXPECTED_HIERARCHY_LEVELS,
        "priority_order": {key: list(value) for key, value in EXPECTED_PRIORITY_ORDER.items()},
        "row_sums": row_sums,
        "column_assignments": column_assignments,
        "primary_exceeds_non_primary_sum": primary > secondary + residual,
        "cyclically_balanced": all(
            np.allclose(values, expected_levels, atol=1.0e-12)
            for values in column_assignments.values()
        ),
    }


def main() -> int:
    args = parse_args()
    algorithms = tuple(args.algorithms)
    scenarios = tuple(args.scenarios)
    rows = []
    errors = []
    hierarchical_design_check = validate_hierarchical_design(errors)
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
                raw_metadata = describe_environment(env)["reward_metadata"]
                if not isinstance(raw_metadata, Mapping):
                    raise TypeError("describe_environment returned non-mapping reward_metadata")
                metadata: Mapping[str, Any] = raw_metadata
                _observations, _infos = env.reset(seed=args.seed)
                zero_actions = {
                    agent: _zero_action(env, agent)
                    for agent in env.possible_agents
                }
                _next_observations, rewards, _terminations, _truncations, _infos = env.step(zero_actions)
                raw_axis_weights = metadata.get("axis_weights")
                axis_weights = raw_axis_weights if isinstance(raw_axis_weights, Mapping) else {}
                raw_profile = metadata.get("profile")
                profile = raw_profile if isinstance(raw_profile, Mapping) else {}
                expected_profile_name = f"{algorithm.lower()}_unified_comparable_v4"

                if metadata.get("function") != "CityLearnV3MADRLRewardFunction":
                    errors.append(f"{algorithm}/{scenario}: reward function is {metadata.get('function')}")
                if metadata.get("not_using_marl_base_weights") is not True:
                    errors.append(f"{algorithm}/{scenario}: reward uses MARL base weights")
                if metadata.get("weighting_method") != EXPECTED_WEIGHTING_METHOD:
                    errors.append(
                        f"{algorithm}/{scenario}: weighting method is {metadata.get('weighting_method')}, "
                        f"expected {EXPECTED_WEIGHTING_METHOD}"
                    )
                if dict(metadata.get("hierarchy_levels") or {}) != EXPECTED_HIERARCHY_LEVELS:
                    errors.append(f"{algorithm}/{scenario}: hierarchy levels do not match the validated design")
                if tuple(metadata.get("axis_priority_order") or ()) != EXPECTED_PRIORITY_ORDER[scenario]:
                    errors.append(f"{algorithm}/{scenario}: axis priority order does not match the validated design")
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
                    "profile": profile.get("profile_name"),
                    "profile_parameters": {
                        key: profile.get(key)
                        for key in EXPECTED_PROFILE
                    },
                    "axis_weights": axis_weights,
                    "weighting_method": metadata.get("weighting_method"),
                    "hierarchy_levels": metadata.get("hierarchy_levels"),
                    "axis_priority_order": metadata.get("axis_priority_order"),
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
            "hierarchical_design_check": hierarchical_design_check,
            "expected_profile": EXPECTED_PROFILE,
            "expected_profile_policy": "unified_comparable_v4 for statistical comparability across MADRL backends with reinforced EV SOC service and BESS cycle penalty",
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
