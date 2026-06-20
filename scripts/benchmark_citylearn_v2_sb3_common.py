"""Stable-Baselines3 CityLearn v2 comparison baselines for the Iquitos dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CITYLEARN_ROOT = SCRIPT_DIR.parents[0]
PROJECT_ROOT = CITYLEARN_ROOT.parent

for path in (CITYLEARN_ROOT, SCRIPT_DIR, PROJECT_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from citylearn.citylearn import CityLearnEnv
from citylearn.scenario_manager import ScenarioManager
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper

from benchmark_citylearn_v2_agents import (
    RolloutResult,
    _district_timeseries_row,
    _write_benchmark_artifacts,
)


DEFAULT_SCHEMA = "CityLearn/data/datasets/citylearn_iquitos_2023_2025/schema.json"
DEFAULT_OUTPUT = "outputs/citylearn_v2_original_benchmark"
SCENARIOS = ("E1", "E2", "E3")


def parse_args(algorithm: str, argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Train/evaluate {algorithm.upper()} as a CityLearn v2 central-agent "
            "baseline on the local Iquitos schema."
        )
    )
    parser.add_argument("--schema-path", default=DEFAULT_SCHEMA)
    parser.add_argument("--scenario", default="E3", help="E1, E2, E3, or ALL.")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--episode-time-steps", default=8760, type=int)
    parser.add_argument(
        "--train-episodes",
        default=75,
        type=int,
        help="Training episodes before the final evaluation rollout. Use 0 for smoke tests.",
    )
    parser.add_argument("--total-timesteps", default=None, type=int)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--learning-rate", default=3.0e-4, type=float)
    parser.add_argument("--gamma", default=0.9999, type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", default=1, type=int)
    parser.add_argument("--policy", default="MlpPolicy")
    parser.add_argument("--n-steps", default=2048, type=int, help="PPO/A2C rollout steps.")
    parser.add_argument("--batch-size", default=256, type=int, help="PPO/SAC batch size.")
    parser.add_argument("--buffer-size", default=50000, type=int, help="SAC replay buffer size.")
    parser.add_argument("--learning-starts", default=1000, type=int, help="SAC random warmup steps.")
    return parser.parse_args(argv)


def scenario_list(value: str) -> List[str]:
    text = str(value or "E3").upper()
    if text in {"ALL", "TODOS", "3EJES"}:
        return list(SCENARIOS)
    if text not in SCENARIOS:
        raise ValueError(f"Unknown scenario {value!r}. Use E1, E2, E3, or ALL.")
    return [text]


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_sb3_algorithm(algorithm: str):
    try:
        from stable_baselines3 import A2C, PPO, SAC
    except ImportError as exc:
        raise RuntimeError(
            "Stable-Baselines3 is required for PPO/SAC/A2C CityLearn v2 baselines. "
            "Install the project requirements, including stable-baselines3==2.2.1."
        ) from exc

    registry = {"ppo": PPO, "sac": SAC, "a2c": A2C}
    return registry[algorithm]


def make_citylearn_v2_central_env(
    *,
    schema_path: str,
    scenario: str,
    seed: int,
    episode_time_steps: int,
) -> Tuple[CityLearnEnv, StableBaselines3Wrapper]:
    resolved_schema = resolve_project_path(schema_path)
    if not resolved_schema.is_file():
        raise FileNotFoundError(f"Iquitos schema not found: {resolved_schema}")

    env = CityLearnEnv(
        str(resolved_schema),
        central_agent=True,
        episode_time_steps=int(episode_time_steps),
        random_seed=int(seed),
        offline=True,
    )
    manager = ScenarioManager()
    manager.select_scenario(scenario)
    manager.apply_scenario_modifications(env)
    wrapped = StableBaselines3Wrapper(NormalizedObservationWrapper(env))
    return env, wrapped


def total_timesteps(args: argparse.Namespace) -> int:
    if args.total_timesteps is not None:
        return max(0, int(args.total_timesteps))
    return max(0, int(args.train_episodes)) * max(1, int(args.episode_time_steps))


def model_kwargs(algorithm: str, args: argparse.Namespace, timesteps: int) -> Dict[str, object]:
    base: Dict[str, object] = {
        "learning_rate": float(args.learning_rate),
        "gamma": float(args.gamma),
        "verbose": int(args.verbose),
        "seed": int(args.seed),
        "device": args.device,
    }
    if algorithm == "ppo":
        n_steps = max(2, min(int(args.n_steps), max(2, int(args.episode_time_steps))))
        batch_size = max(2, min(int(args.batch_size), n_steps))
        base.update({"n_steps": n_steps, "batch_size": batch_size})
    elif algorithm == "a2c":
        base.update({"n_steps": max(2, min(int(args.n_steps), max(2, int(args.episode_time_steps))))})
    elif algorithm == "sac":
        learning_starts = max(0, min(int(args.learning_starts), max(0, timesteps - 1))) if timesteps else 0
        base.update(
            {
                "batch_size": max(1, int(args.batch_size)),
                "buffer_size": max(int(args.buffer_size), int(args.batch_size), 1),
                "learning_starts": learning_starts,
            }
        )
    return base


def central_trace_row(
    raw_env: CityLearnEnv,
    *,
    global_step: int,
    episode_step: int,
    scenario: str,
    observation,
    action,
    reward: float,
) -> Dict[str, object]:
    observation_array = np.asarray(observation, dtype=float).reshape(-1)
    action_array = np.asarray(action, dtype=float).reshape(-1)
    return {
        "global_step": global_step,
        "episode": 0,
        "episode_step": episode_step,
        "time_step": int(getattr(raw_env.unwrapped, "time_step", global_step)),
        "scenario": scenario,
        "agent": "central_agent",
        "agent_index": 0,
        "reward": float(reward),
        "done": bool(raw_env.terminated or raw_env.truncated),
        "action_dim": int(action_array.size),
        "action_mean": float(np.mean(action_array)) if action_array.size else None,
        "action_min": float(np.min(action_array)) if action_array.size else None,
        "action_max": float(np.max(action_array)) if action_array.size else None,
        "action_l2": float(np.linalg.norm(action_array)) if action_array.size else None,
        "observation_dim": int(observation_array.size),
        "observation_mean": float(np.mean(observation_array)) if observation_array.size else None,
        "observation_min": float(np.min(observation_array)) if observation_array.size else None,
        "observation_max": float(np.max(observation_array)) if observation_array.size else None,
        "observation_l2": float(np.linalg.norm(observation_array)) if observation_array.size else None,
    }


def rollout_sb3(raw_env: CityLearnEnv, wrapped_env: StableBaselines3Wrapper, model, *, scenario: str) -> RolloutResult:
    observations, _ = wrapped_env.reset()
    timeseries_rows: List[Dict[str, object]] = []
    trace_rows: List[Dict[str, object]] = []
    reward_rows: List[List[float]] = []
    global_step = 0

    while not (raw_env.terminated or raw_env.truncated):
        actions, _ = model.predict(observations, deterministic=True)
        next_observations, reward, terminated, truncated, _ = wrapped_env.step(actions)
        reward_value = float(reward)
        reward_rows.append([reward_value])
        timeseries_rows.append(
            _district_timeseries_row(
                raw_env,
                global_step=global_step,
                episode_step=global_step,
                scenario=scenario,
                rewards=[reward_value],
            )
        )
        trace_rows.append(
            central_trace_row(
                raw_env,
                global_step=global_step,
                episode_step=global_step,
                scenario=scenario,
                observation=observations,
                action=actions,
                reward=reward_value,
            )
        )
        observations = next_observations
        global_step += 1
        if terminated or truncated:
            break

    return RolloutResult(timeseries_rows, trace_rows, reward_rows)


def write_sb3_metadata(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def run_scenario(algorithm: str, args: argparse.Namespace, scenario: str) -> Dict[str, object]:
    algo_cls = load_sb3_algorithm(algorithm)
    timesteps = total_timesteps(args)
    raw_train_env, train_env = make_citylearn_v2_central_env(
        schema_path=args.schema_path,
        scenario=scenario,
        seed=args.seed,
        episode_time_steps=args.episode_time_steps,
    )
    output_dir = Path(args.output_dir) / algorithm / f"{scenario}_seed_{args.seed}"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = algo_cls(args.policy, train_env, **model_kwargs(algorithm, args, timesteps))

    try:
        if timesteps > 0:
            model.learn(total_timesteps=timesteps, log_interval=1)
        model_path = checkpoint_dir / f"{algorithm}_citylearn_v2_sb3.zip"
        model.save(str(model_path))
    finally:
        train_env.close()
        raw_train_env.close()

    raw_eval_env, eval_env = make_citylearn_v2_central_env(
        schema_path=args.schema_path,
        scenario=scenario,
        seed=args.seed,
        episode_time_steps=args.episode_time_steps,
    )
    try:
        rollout = rollout_sb3(raw_eval_env, eval_env, model, scenario=scenario)
        results = _write_benchmark_artifacts(
            output_dir=output_dir,
            agent_key=algorithm,
            backend="stable_baselines3",
            scenario=scenario,
            seed=args.seed,
            episode_time_steps=args.episode_time_steps,
            train_episodes=args.train_episodes,
            env=raw_eval_env,
            rollout=rollout,
        )
        metadata = {
            "algorithm": algorithm,
            "family": "citylearn_v2_original",
            "backend": "stable_baselines3",
            "schema_path": str(resolve_project_path(args.schema_path)),
            "scenario": scenario,
            "seed": args.seed,
            "episode_time_steps": args.episode_time_steps,
            "train_episodes": args.train_episodes,
            "total_timesteps": timesteps,
            "central_agent": True,
            "checkpoint": str(model_path),
            "same_dataset_as_madrl": "CityLearn/data/datasets/citylearn_iquitos_2023_2025/schema.json",
        }
        write_sb3_metadata(output_dir / "data" / "sb3_baseline_metadata.json", metadata)
        results["sb3_baseline_metadata"] = metadata
        for results_path in (output_dir / "data" / "results.json", output_dir / "results.json"):
            results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        return results
    finally:
        eval_env.close()
        raw_eval_env.close()


def main(algorithm: str, argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(algorithm, argv)
    summaries = [run_scenario(algorithm, args, scenario) for scenario in scenario_list(args.scenario)]
    manifest = {
        "family": "citylearn_v2_original",
        "backend": "stable_baselines3",
        "algorithm": algorithm,
        "schema_path": str(resolve_project_path(args.schema_path)),
        "scenario": args.scenario,
        "seed": args.seed,
        "episode_time_steps": args.episode_time_steps,
        "train_episodes": args.train_episodes,
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
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"{algorithm}_sb3_benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(manifest_path)
    return 0
