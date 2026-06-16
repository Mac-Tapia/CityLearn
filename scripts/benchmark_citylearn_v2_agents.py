"""Benchmark original CityLearn v2 agents with CityLearn v3 objective reports.

The goal is not to modify the original agents. This script rolls out the
CityLearn v2 agents on the same schema/scenario used by the MADRL experiments
and writes the same KPI/table/figure layout expected by the v2-vs-v3 master
comparator.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CITYLEARN_ROOT = SCRIPT_DIR.parents[0]
PROJECT_ROOT = CITYLEARN_ROOT.parent

if str(CITYLEARN_ROOT) not in sys.path:
    sys.path.insert(0, str(CITYLEARN_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from citylearn.agents.base import Agent, BaselineAgent
from citylearn.agents.marlisa import MARLISA
from citylearn.agents.rbc import BasicRBC, HourRBC, OptimizedRBC
from citylearn.agents.sac import SAC
from citylearn.citylearn import CityLearnEnv
from citylearn.dec_pomdp import DEFAULT_17_BUILDING_EV_SCHEMA
from citylearn.scenario_manager import ScenarioManager

from citylearn_v3_training_common import (
    _as_float,
    _checkpoint_files,
    _compact_array_stats,
    _mean_current_building_signal,
    _series_value,
    _write_csv_mirrors,
    _write_json_mirrors,
    _write_training_figures_and_tables,
    citylearn_v3_training_report,
    ensure_artifact_layout,
)


AGENT_REGISTRY = {
    "baseline": BaselineAgent,
    "hour_rbc": HourRBC,
    "basic_rbc": BasicRBC,
    "optimized_rbc": OptimizedRBC,
    "sac": SAC,
    "marlisa": MARLISA,
    "random": Agent,
}


@dataclass(frozen=True)
class RolloutResult:
    timeseries_rows: List[Dict[str, object]]
    trace_rows: List[Dict[str, object]]
    reward_rows: List[List[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schema-path",
        default=str(DEFAULT_17_BUILDING_EV_SCHEMA),
        help="CityLearn v2 schema path or dataset name.",
    )
    parser.add_argument("--scenario", default="E3")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--episode-time-steps", default=8760, type=int)
    parser.add_argument("--train-episodes", default=0, type=int)
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["baseline", "hour_rbc"],
        choices=sorted(AGENT_REGISTRY),
        help=(
            "Original CityLearn v2 agents to benchmark. SAC/MARLISA can be slow; "
            "enable them explicitly when needed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/citylearn_v2_original_benchmark",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write a failure JSON for agents that error and continue.",
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_env(schema_path: str, *, scenario: str, seed: int, episode_time_steps: int) -> CityLearnEnv:
    env = CityLearnEnv(
        str(schema_path),
        central_agent=False,
        episode_time_steps=episode_time_steps,
        random_seed=seed,
        offline=True,
    )

    if scenario:
        manager = ScenarioManager()
        manager.select_scenario(scenario)
        manager.apply_scenario_modifications(env)

    return env


def _hour_rbc_action_map(env: CityLearnEnv) -> Dict[str, Dict[int, float]]:
    action_names = sorted({name for names in env.unwrapped.action_names for name in names})
    action_map: Dict[str, Dict[int, float]] = {}

    for name in action_names:
        hours: Dict[int, float] = {}

        for hour in range(1, 25):
            if "storage" in name:
                value = -0.08 if 9 <= hour <= 21 else 0.091
            elif name in {"cooling_device", "heating_device", "cooling_or_heating_device"}:
                value = 0.4 if hour <= 8 or hour >= 22 else 0.8
            else:
                value = 0.0

            hours[hour] = value

        action_map[name] = hours

    return action_map


def _make_agent(agent_key: str, env: CityLearnEnv):
    cls = AGENT_REGISTRY[agent_key]

    if agent_key == "hour_rbc":
        return cls(env, action_map=_hour_rbc_action_map(env))

    if agent_key == "marlisa":
        return cls(env, regression_frequency=2500, information_sharing=True, iterations=2)

    return cls(env)


def _district_timeseries_row(
    env: CityLearnEnv,
    *,
    global_step: int,
    episode_step: int,
    scenario: str,
    rewards: Sequence[float],
) -> Dict[str, object]:
    citylearn_env = env.unwrapped
    time_step = int(getattr(citylearn_env, "time_step", global_step))
    reward_values = [_as_float(value) for value in rewards]
    reward_values = [value for value in reward_values if value is not None]
    return {
        "global_step": global_step,
        "episode": 0,
        "episode_step": episode_step,
        "reset_count": 1,
        "time_step": time_step,
        "scenario": scenario,
        "reward_sum": None if not reward_values else float(np.sum(reward_values)),
        "reward_mean": None if not reward_values else float(np.mean(reward_values)),
        "all_done": bool(env.terminated or env.truncated),
        "district_net_electricity_consumption": _series_value(citylearn_env, "net_electricity_consumption", time_step),
        "district_net_electricity_consumption_without_storage": _series_value(citylearn_env, "net_electricity_consumption_without_storage", time_step),
        "district_net_electricity_consumption_cost": _series_value(citylearn_env, "net_electricity_consumption_cost", time_step),
        "district_net_electricity_consumption_emission": _series_value(citylearn_env, "net_electricity_consumption_emission", time_step),
        "electricity_price_mean": _mean_current_building_signal(citylearn_env, "pricing", "electricity_pricing", time_step),
        "carbon_intensity_mean": _mean_current_building_signal(citylearn_env, "carbon_intensity", "carbon_intensity", time_step),
    }


def _trace_rows(
    env: CityLearnEnv,
    *,
    global_step: int,
    episode_step: int,
    scenario: str,
    observations,
    actions,
    rewards,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    building_names = [building.name for building in env.unwrapped.buildings]

    for index, building_name in enumerate(building_names):
        action = np.asarray(actions[index] if index < len(actions) else [], dtype=float).reshape(-1)
        observation = np.asarray(observations[index] if index < len(observations) else [], dtype=float).reshape(-1)
        action_stats = _compact_array_stats(action)
        observation_stats = _compact_array_stats(observation)
        rows.append(
            {
                "global_step": global_step,
                "episode": 0,
                "episode_step": episode_step,
                "time_step": int(getattr(env.unwrapped, "time_step", global_step)),
                "scenario": scenario,
                "agent": building_name,
                "agent_index": index,
                "reward": _as_float(rewards[index] if index < len(rewards) else None),
                "done": bool(env.terminated or env.truncated),
                "action_dim": int(action.size),
                "action_mean": action_stats["mean"],
                "action_min": action_stats["min"],
                "action_max": action_stats["max"],
                "action_l2": action_stats["l2"],
                "observation_dim": int(observation.size),
                "observation_mean": observation_stats["mean"],
                "observation_min": observation_stats["min"],
                "observation_max": observation_stats["max"],
                "observation_l2": observation_stats["l2"],
            }
        )

    return rows


def _rollout(env: CityLearnEnv, agent, *, scenario: str) -> RolloutResult:
    observations, _ = env.reset()
    timeseries_rows: List[Dict[str, object]] = []
    trace_rows: List[Dict[str, object]] = []
    reward_rows: List[List[float]] = []
    global_step = 0

    while not (env.terminated or env.truncated):
        actions = agent.predict(observations, deterministic=True)
        next_observations, rewards, terminated, truncated, _ = env.step(actions)
        rewards = list(np.asarray(rewards, dtype=float).reshape(-1))
        reward_rows.append(rewards)
        timeseries_rows.append(
            _district_timeseries_row(
                env,
                global_step=global_step,
                episode_step=global_step,
                scenario=scenario,
                rewards=rewards,
            )
        )
        trace_rows.extend(
            _trace_rows(
                env,
                global_step=global_step,
                episode_step=global_step,
                scenario=scenario,
                observations=observations,
                actions=actions,
                rewards=rewards,
            )
        )
        observations = [o for o in next_observations]
        global_step += 1

        if terminated or truncated:
            break

    return RolloutResult(timeseries_rows, trace_rows, reward_rows)


def _write_benchmark_artifacts(
    *,
    output_dir: Path,
    agent_key: str,
    scenario: str,
    seed: int,
    episode_time_steps: int,
    train_episodes: int,
    env: CityLearnEnv,
    rollout: RolloutResult,
) -> Dict[str, object]:
    dirs = ensure_artifact_layout(output_dir)
    data_dir = dirs["data"]
    report = citylearn_v3_training_report(env)
    timeseries_path = data_dir / "timeseries.csv"
    trace_path = data_dir / "trace.csv"
    _write_csv_mirrors([timeseries_path, output_dir / "timeseries.csv"], rollout.timeseries_rows)
    _write_csv_mirrors([trace_path, output_dir / "trace.csv"], rollout.trace_rows)

    checkpoints = _checkpoint_files(output_dir, dirs["checkpoints"])
    checkpoint_manifest = {
        "algorithm": agent_key,
        "backend": "citylearn_v2_original",
        "checkpoint_dir": str(dirs["checkpoints"]),
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "hyperparameters": {"train_episodes": train_episodes},
    }
    _write_json_mirrors(
        [data_dir / "checkpoint_manifest.json", output_dir / "checkpoint_manifest.json"],
        checkpoint_manifest,
    )
    figures_manifest = _write_training_figures_and_tables(
        dirs=dirs,
        report=report,
        timeseries_rows=rollout.timeseries_rows,
        trace_rows=rollout.trace_rows,
        episode_summaries=[],
        checkpoints=checkpoints,
    )
    results = {
        "algorithm": agent_key,
        "family": "citylearn_v2_original",
        "backend": "citylearn.agents",
        "scenario": scenario,
        "seed": seed,
        "episode_time_steps": episode_time_steps,
        "train_episodes": train_episodes,
        "output_dir": str(output_dir),
        "timeseries_rows": len(rollout.timeseries_rows),
        "trace_rows": len(rollout.trace_rows),
        "timeseries_csv": str(timeseries_path),
        "trace_csv": str(trace_path),
        "checkpoint_manifest": str(data_dir / "checkpoint_manifest.json"),
        "checkpoint_count": len(checkpoints),
        "figures_manifest": str(dirs["figures"] / "figures_manifest.json"),
        "figures": figures_manifest,
        "project_axis_metrics": report["project_axis_metrics"],
        "citylearn_v3_report": report,
    }
    _write_json_mirrors([data_dir / "results.json", output_dir / "results.json"], results)
    _write_csv(data_dir / "kpis.csv", _flat_kpi_rows(agent_key, "citylearn_v2_original", report))
    return results


def _flat_kpi_rows(method: str, family: str, report: Mapping[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    axes = report.get("project_axis_metrics", {})

    for axis_code, axis in axes.items():
        for kpi, payload in axis.get("kpis", {}).items():
            comparison = payload.get("comparison", {})
            trace = payload.get("trace", {})
            rows.append(
                {
                    "family": family,
                    "method": method,
                    "axis": axis_code,
                    "axis_name": axis.get("name"),
                    "kpi": kpi,
                    "value": payload.get("value"),
                    "baseline": comparison.get("baseline"),
                    "delta_vs_baseline": comparison.get("delta_vs_baseline"),
                    "improved_vs_baseline": comparison.get("improved_vs_baseline"),
                    "lower_is_better": trace.get("lower_is_better"),
                    "source": trace.get("source"),
                }
            )

    return rows


def _write_failure(output_dir: Path, agent_key: str, exc: BaseException) -> None:
    dirs = ensure_artifact_layout(output_dir)
    payload = {
        "algorithm": agent_key,
        "family": "citylearn_v2_original",
        "status": "failed",
        "error_type": exc.__class__.__name__,
        "error": str(exc),
    }
    _write_json_mirrors([dirs["data"] / "results.json", output_dir / "results.json"], payload)


def run_agent(args: argparse.Namespace, agent_key: str) -> Dict[str, object]:
    output_dir = Path(args.output_dir) / agent_key / f"{args.scenario}_seed_{args.seed}"
    env = _make_env(
        args.schema_path,
        scenario=args.scenario,
        seed=args.seed,
        episode_time_steps=args.episode_time_steps,
    )

    try:
        agent = _make_agent(agent_key, env)

        if args.train_episodes > 0 and agent_key not in {"baseline", "hour_rbc", "basic_rbc", "optimized_rbc", "random"}:
            agent.learn(episodes=args.train_episodes)
            env.close()
            env = _make_env(
                args.schema_path,
                scenario=args.scenario,
                seed=args.seed,
                episode_time_steps=args.episode_time_steps,
            )
            agent.env = env

        rollout = _rollout(env, agent, scenario=args.scenario)
        return _write_benchmark_artifacts(
            output_dir=output_dir,
            agent_key=agent_key,
            scenario=args.scenario,
            seed=args.seed,
            episode_time_steps=args.episode_time_steps,
            train_episodes=args.train_episodes,
            env=env,
            rollout=rollout,
        )
    finally:
        env.close()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, object]] = []

    for agent_key in args.agents:
        try:
            print(f"Running CityLearn v2 original agent: {agent_key}", flush=True)
            summaries.append(run_agent(args, agent_key))
        except Exception as exc:
            output_dir = output_root / agent_key / f"{args.scenario}_seed_{args.seed}"
            _write_failure(output_dir, agent_key, exc)
            print(f"FAILED {agent_key}: {exc}", file=sys.stderr, flush=True)

            if not args.continue_on_error:
                raise

    manifest = {
        "family": "citylearn_v2_original",
        "schema_path": args.schema_path,
        "scenario": args.scenario,
        "seed": args.seed,
        "episode_time_steps": args.episode_time_steps,
        "train_episodes": args.train_episodes,
        "agents": args.agents,
        "runs": [
            {
                "algorithm": item.get("algorithm"),
                "output_dir": item.get("output_dir"),
                "timeseries_rows": item.get("timeseries_rows"),
                "trace_rows": item.get("trace_rows"),
            }
            for item in summaries
        ],
    }
    (output_root / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(output_root / "benchmark_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
