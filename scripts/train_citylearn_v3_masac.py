"""Train MASAC/mSAC on CityLearn v3 using the paper repository backend."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from citylearn_v3_training_common import (
    CityLearnSMACDiscreteEnv,
    add_common_citylearn_args,
    add_external_path,
    citylearn_v3_training_report,
    ensure_project_paths,
    ensure_artifact_layout,
    resolve_output_dir,
    write_training_artifacts,
    write_training_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_citylearn_args(parser)
    parser.add_argument("--episodes", default=None, type=int, help="Number of MASAC training episodes/epochs.")
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--action-bins", default=3, type=int)
    parser.add_argument("--buffer-size", default=2, type=int)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_project_paths()
    add_external_path("MARL", "src")

    from common.arguments import get_common_args, get_mixer_args
    from runner_msac import Runner

    output_dir = resolve_output_dir(args.output_dir, "masac", args.scenario, args.seed)
    artifact_dirs = ensure_artifact_layout(output_dir)
    configured_episodes = int(args.episodes if args.episodes is not None else args.epochs)
    env = CityLearnSMACDiscreteEnv(
        schema_path=args.schema_path,
        scenario=args.scenario,
        seed=args.seed,
        episode_time_steps=args.episode_time_steps,
        action_bins=args.action_bins,
        live_progress_path=str(output_dir / "live_progress.json"),
        live_progress_interval=100,
    )
    env_info = env.get_env_info()

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        backend_args = get_mixer_args(get_common_args())
    finally:
        sys.argv = original_argv

    backend_args.alg = "qmix"
    backend_args.map = f"citylearn_v3_{args.scenario}"
    backend_args.seed = args.seed
    backend_args.n_epoch = configured_episodes
    backend_args.evaluate_cycle = max(configured_episodes + 1, 2)
    backend_args.evaluate_epoch = 1
    backend_args.n_episodes = 1
    backend_args.cuda = bool(args.cuda and torch.cuda.is_available())
    backend_args.result_dir = str(artifact_dirs["data"] / "backend_results")
    backend_args.model_dir = str(artifact_dirs["checkpoints"] / "models")
    backend_args.replay_dir = ""
    backend_args.n_actions = env_info["n_actions"]
    backend_args.n_agents = env_info["n_agents"]
    backend_args.state_shape = env_info["state_shape"]
    backend_args.obs_shape = env_info["obs_shape"]
    backend_args.episode_limit = env_info["episode_limit"]
    backend_args.critic_batch_size = 1
    backend_args.buffer_size = max(int(args.buffer_size), 2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    report = citylearn_v3_training_report(None)
    artifacts = {}
    hyperparameters = {
        "episodes": configured_episodes,
        "n_epoch": backend_args.n_epoch,
        "n_episodes_per_epoch": backend_args.n_episodes,
        "episode_limit": backend_args.episode_limit,
        "action_bins": args.action_bins,
        "n_discrete_actions": env_info["n_actions"],
        "critic_batch_size": backend_args.critic_batch_size,
        "buffer_size": backend_args.buffer_size,
        "cuda": backend_args.cuda,
        "ctde_state_shape": backend_args.state_shape,
    }

    try:
        runner = Runner(env, backend_args)
        runner.run(args.seed)
        report = citylearn_v3_training_report(env)
        artifacts = write_training_artifacts(
            output_dir=output_dir,
            algorithm="MASAC",
            backend="external/MARL",
            args=args,
            report=report,
            candidate=env,
            hyperparameters=hyperparameters,
            extra={
                "episodes": configured_episodes,
                "epochs": configured_episodes,
            },
        )
    finally:
        env.close()

    write_training_summary(
        output_dir,
        {
            "algorithm": "MASAC",
            "backend": "external/MARL",
            "scenario": args.scenario,
            "seed": args.seed,
            "episode_time_steps": args.episode_time_steps,
            "episodes": configured_episodes,
            "epochs": configured_episodes,
            "action_bins": args.action_bins,
            "n_discrete_actions": env_info["n_actions"],
            "output_dir": str(output_dir),
            "artifact_layout": artifacts.get("artifact_layout", {}),
            "hyperparameters": hyperparameters,
            "artifacts": artifacts,
            "project_axis_metrics": report["project_axis_metrics"],
            "citylearn_v3_report": report,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
