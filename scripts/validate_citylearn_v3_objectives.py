"""Validate CityLearn v3 MADRL objective coverage and baseline comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from citylearn_v3_training_common import (
    add_common_citylearn_args,
    ensure_project_paths,
    resolve_output_dir,
    write_json,
)


ALGORITHM_LAUNCHERS = {
    "HAPPO": "train_citylearn_v3_happo.py",
    "MASAC": "train_citylearn_v3_masac.py",
    "MATD3": "train_citylearn_v3_matd3.py",
    "MAAC": "train_citylearn_v3_maac.py",
}


CTDE_CONTRACT = {
    "HAPPO": {
        "training": "centralized critic/value input via CityLearnHARLEnv.share_observation_space",
        "execution": "per-building local observation and per-building action",
    },
    "MASAC": {
        "training": "SMAC-style global state via CityLearnSMACDiscreteEnv.get_state",
        "execution": "per-building local observation and discrete action mapped to CityLearn actions",
    },
    "MATD3": {
        "training": "centralized critic input from concatenated joint observations/actions",
        "execution": "per-building actor with local observation and continuous action",
    },
    "MAAC": {
        "training": "attention critic over all agents through the official MAAC backend",
        "execution": "per-building policy with local observation and discrete action mapped to CityLearn actions",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_citylearn_args(parser)
    parser.add_argument(
        "--include-citylearn-v2-test-agents",
        action="store_true",
        help="Also roll out CityLearn v2 BaselineAgent and EV RBC for KPI comparison.",
    )
    return parser.parse_args()


def _zero_action_rollout(env, episode_time_steps: int) -> int:
    env.reset()
    steps = 0

    while env.agents and steps < episode_time_steps:
        actions = {
            agent: np.zeros(env.action_space(agent).shape, dtype=np.float32)
            for agent in env.agents
        }
        env.step(actions)
        steps += 1

    return steps


def _rollout_citylearn_v2_agent(agent_cls, schema_path, scenario: str, seed: int, episode_time_steps: int):
    from citylearn.citylearn import CityLearnEnv
    from citylearn.dec_pomdp import DEFAULT_17_BUILDING_EV_SCHEMA, resolve_citylearn_schema_path
    from citylearn.scenario_manager import ScenarioManager
    from citylearn.v3.objectives import evaluate_objectives

    schema = resolve_citylearn_schema_path(schema_path or DEFAULT_17_BUILDING_EV_SCHEMA)
    env = CityLearnEnv(
        str(schema),
        central_agent=True,
        episode_time_steps=episode_time_steps,
        random_seed=seed,
        offline=True,
    )

    if scenario:
        manager = ScenarioManager()
        manager.select_scenario(scenario)
        manager.apply_scenario_modifications(env)

    try:
        agent = agent_cls(env)
        observations, _ = env.reset()
        steps = 0

        while not env.terminated and steps < episode_time_steps:
            actions = agent.predict(observations, deterministic=True)
            result = env.step(actions)
            observations = result[0]
            steps += 1

        report = evaluate_objectives(env)
        return {
            "agent": agent_cls.__name__,
            "steps": steps,
            "objectives": report,
        }
    finally:
        env.close()


def main() -> int:
    args = parse_args()
    ensure_project_paths()

    from citylearn.agents.base import BaselineAgent
    from citylearn.agents.rbc import BasicElectricVehicleRBC_ReferenceController
    from citylearn.v3 import (
        describe_environment,
        evaluate_objectives,
        make_citylearn_v3_env,
        make_citylearn_v3_project_env,
        objective_manifest,
    )
    from citylearn.v3.backends import citylearn_v3_backend_manifest

    output_dir = resolve_output_dir(args.output_dir, "objective_validation", args.scenario, args.seed)
    env = (
        make_citylearn_v3_env(
            schema_path=args.schema_path,
            scenario=args.scenario,
            seed=args.seed,
            episode_time_steps=args.episode_time_steps,
        )
        if args.schema_path
        else make_citylearn_v3_project_env(
            scenario=args.scenario,
            seed=args.seed,
            episode_time_steps=args.episode_time_steps,
        )
    )

    try:
        steps = _zero_action_rollout(env, args.episode_time_steps)
        objective_report = evaluate_objectives(env)
        environment_report = describe_environment(env)
    finally:
        env.close()

    scripts_dir = Path(__file__).resolve().parent
    algorithm_report = {
        algorithm: {
            "launcher": str(scripts_dir / launcher),
            "launcher_exists": (scripts_dir / launcher).is_file(),
            "ctde": CTDE_CONTRACT[algorithm],
            "collaborative": True,
            "multiobjective_evaluation": tuple(objective_manifest()["axes"].keys()),
        }
        for algorithm, launcher in ALGORITHM_LAUNCHERS.items()
    }

    citylearn_v2_agents = {}
    if args.include_citylearn_v2_test_agents:
        for agent_cls in (BaselineAgent, BasicElectricVehicleRBC_ReferenceController):
            citylearn_v2_agents[agent_cls.__name__] = _rollout_citylearn_v2_agent(
                agent_cls,
                args.schema_path,
                args.scenario,
                args.seed,
                args.episode_time_steps,
            )

    report = {
        "scenario": args.scenario,
        "seed": args.seed,
        "episode_time_steps": args.episode_time_steps,
        "zero_action_citylearn_v3_steps": steps,
        "environment": environment_report,
        "algorithms": algorithm_report,
        "backend_manifest": citylearn_v3_backend_manifest(),
        "objectives": objective_report,
        "citylearn_v2_test_agents": citylearn_v2_agents,
    }

    write_json(output_dir / "citylearn_v3_objective_validation.json", report)
    print(output_dir / "citylearn_v3_objective_validation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

