"""Train HAPPO on CityLearn v3 using the official HARL backend."""

from __future__ import annotations

import argparse
import sys

import torch

from citylearn_v3_training_common import (
    CityLearnHARLEnv,
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
    parser.add_argument("--episodes", default=None, type=int, help="Number of rollout episodes to train.")
    parser.add_argument("--num-env-steps", default=8, type=int)
    parser.add_argument("--hidden-size", default=128, type=int)
    parser.add_argument("--torch-threads", default=1, type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--exp-name", default="citylearn_v3_happo")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_project_paths()
    add_external_path("HARL")

    from harl.envs.env_wrappers import ShareDummyVecEnv
    import harl.runners.on_policy_base_runner as base_runner
    from harl.runners import RUNNER_REGISTRY
    from harl.utils.configs_tools import get_defaults_yaml_args

    output_dir = resolve_output_dir(args.output_dir, "happo", args.scenario, args.seed)
    artifact_dirs = ensure_artifact_layout(output_dir)
    configured_num_env_steps = max(
        args.num_env_steps,
        args.episode_time_steps,
        (args.episodes or 0) * args.episode_time_steps,
    )
    configured_episodes = max(1, configured_num_env_steps // max(args.episode_time_steps, 1))

    def make_citylearn_train_env(env_name, seed, n_threads, env_args):
        def make_env(rank):
            def init_env():
                env = CityLearnHARLEnv(
                    schema_path=args.schema_path,
                    scenario=args.scenario,
                    seed=args.seed + rank * 1000,
                    episode_time_steps=args.episode_time_steps,
                    live_progress_path=str(output_dir / "live_progress.json"),
                    live_progress_interval=100,
                )
                env.seed(seed + rank * 1000)
                return env

            return init_env

        return ShareDummyVecEnv([make_env(rank) for rank in range(n_threads)])

    base_runner.make_train_env = make_citylearn_train_env

    algo_args, env_args = get_defaults_yaml_args("happo", "gym")
    algo_args["seed"]["seed"] = args.seed
    algo_args["device"]["cuda"] = bool(args.cuda and torch.cuda.is_available())
    algo_args["device"]["torch_threads"] = args.torch_threads
    algo_args["train"]["n_rollout_threads"] = 1
    algo_args["train"]["episode_length"] = args.episode_time_steps
    algo_args["train"]["num_env_steps"] = configured_num_env_steps
    algo_args["train"]["log_interval"] = 1
    algo_args["train"]["eval_interval"] = 1
    algo_args["eval"]["use_eval"] = False
    algo_args["logger"]["log_dir"] = str(artifact_dirs["checkpoints"])
    algo_args["model"]["hidden_sizes"] = [args.hidden_size, args.hidden_size]
    algo_args["algo"]["share_param"] = False
    env_args["scenario"] = f"citylearn_v3_{args.scenario}"
    env_args["state_type"] = "EP"

    runner_args = {"algo": "happo", "env": "gym", "exp_name": args.exp_name}
    runner = RUNNER_REGISTRY["happo"](runner_args, algo_args, env_args)
    report = citylearn_v3_training_report(None)
    artifacts = {}
    hyperparameters = {
        "episodes": configured_episodes,
        "num_env_steps": configured_num_env_steps,
        "episode_length": args.episode_time_steps,
        "hidden_sizes": algo_args["model"]["hidden_sizes"],
        "torch_threads": args.torch_threads,
        "share_param": algo_args["algo"]["share_param"],
        "ctde_state_type": env_args["state_type"],
        "n_rollout_threads": algo_args["train"]["n_rollout_threads"],
        "log_interval": algo_args["train"]["log_interval"],
        "checkpoint_interval_episodes": algo_args["train"]["eval_interval"],
        "cuda": algo_args["device"]["cuda"],
    }

    try:
        runner.run()
        runner.save()
        report = citylearn_v3_training_report(runner.envs)
        artifacts = write_training_artifacts(
            output_dir=output_dir,
            algorithm="HAPPO",
            backend="external/HARL",
            args=args,
            report=report,
            candidate=runner.envs,
            hyperparameters=hyperparameters,
            extra={
                "episodes": configured_episodes,
                "num_env_steps": configured_num_env_steps,
            },
        )
    finally:
        runner.close()

    write_training_summary(
        output_dir,
        {
            "algorithm": "HAPPO",
            "backend": "external/HARL",
            "scenario": args.scenario,
            "seed": args.seed,
            "episode_time_steps": args.episode_time_steps,
            "episodes": configured_episodes,
            "num_env_steps": configured_num_env_steps,
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
