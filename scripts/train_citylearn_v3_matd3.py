"""Train MATD3 on CityLearn v3 using the PyTorch marlbenchmark/off-policy backend."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from citylearn_v3_training_common import (
    CityLearnOffPolicyVecEnv,
    add_common_citylearn_args,
    add_external_path,
    citylearn_v3_training_report,
    ensure_project_paths,
    ensure_artifact_layout,
    install_noop_wandb,
    resolve_output_dir,
    write_training_artifacts,
    write_training_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_citylearn_args(parser)
    parser.add_argument("--episodes", default=None, type=int, help="Number of MATD3 rollout episodes.")
    parser.add_argument("--num-env-steps", default=8, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--buffer-size", default=128, type=int)
    parser.add_argument("--hidden-size", default=64, type=int)
    parser.add_argument("--train-interval", default=1, type=int)
    parser.add_argument("--num-random-episodes", default=1, type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--experiment-name", default="citylearn_v3_matd3")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_project_paths()
    install_noop_wandb()
    add_external_path("off-policy")

    from offpolicy.config import get_config
    from offpolicy.runner.mlp.mpe_runner import MPERunner
    from offpolicy.utils.util import get_cent_act_dim, get_dim_from_space

    output_dir = resolve_output_dir(args.output_dir, "matd3", args.scenario, args.seed)
    artifact_dirs = ensure_artifact_layout(output_dir)
    configured_num_env_steps = max(
        args.num_env_steps,
        args.episode_time_steps,
        (args.episodes or 0) * args.episode_time_steps,
    )
    configured_episodes = max(1, configured_num_env_steps // max(args.episode_time_steps, 1))
    env = CityLearnOffPolicyVecEnv(
        schema_path=args.schema_path,
        scenario=args.scenario,
        seed=args.seed,
        episode_time_steps=args.episode_time_steps,
        algorithm="MATD3",
        live_progress_path=str(output_dir / "live_progress.json"),
        live_progress_interval=args.live_progress_interval,
    )
    eval_env = CityLearnOffPolicyVecEnv(
        schema_path=args.schema_path,
        scenario=args.scenario,
        seed=args.seed + 10000,
        episode_time_steps=args.episode_time_steps,
        algorithm="MATD3",
    )

    parser = get_config()
    all_args = parser.parse_args([])
    all_args.algorithm_name = "matd3"
    all_args.env_name = "CityLearnV3"
    all_args.scenario_name = args.scenario
    all_args.experiment_name = args.experiment_name
    all_args.seed = args.seed
    all_args.cuda = bool(args.cuda and torch.cuda.is_available())
    all_args.use_wandb = False
    all_args.use_eval = False
    all_args.num_env_steps = configured_num_env_steps
    all_args.episode_length = args.episode_time_steps
    all_args.batch_size = args.batch_size
    all_args.buffer_size = args.buffer_size
    all_args.hidden_size = args.hidden_size
    all_args.train_interval = args.train_interval
    all_args.num_random_episodes = args.num_random_episodes
    all_args.log_interval = max(args.episode_time_steps, 1)
    all_args.save_interval = max(args.episode_time_steps, 1)
    all_args.n_rollout_threads = 1
    all_args.n_eval_rollout_threads = 1
    all_args.share_policy = False
    all_args.use_same_share_obs = True
    all_args.use_available_actions = False

    device = torch.device("cuda:0" if all_args.cuda else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy_info = {
        f"policy_{agent_id}": {
            "cent_obs_dim": get_dim_from_space(env.share_observation_space[agent_id]),
            "cent_act_dim": get_cent_act_dim(env.action_space),
            "obs_space": env.observation_space[agent_id],
            "share_obs_space": env.share_observation_space[agent_id],
            "act_space": env.action_space[agent_id],
        }
        for agent_id in range(env.num_agents)
    }

    def policy_mapping_fn(agent_id):
        return f"policy_{agent_id}"

    run_dir = artifact_dirs["checkpoints"] / "offpolicy_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "args": all_args,
        "policy_info": policy_info,
        "policy_mapping_fn": policy_mapping_fn,
        "env": env,
        "eval_env": eval_env,
        "num_agents": env.num_agents,
        "device": device,
        "use_same_share_obs": True,
        "run_dir": run_dir,
    }

    runner = MPERunner(config=config)
    env.adapter.clear_records()
    total_num_steps = 0
    report = citylearn_v3_training_report(None)
    artifacts = {}
    hyperparameters = {
        "episodes": configured_episodes,
        "num_env_steps": configured_num_env_steps,
        "episode_length": args.episode_time_steps,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "hidden_size": args.hidden_size,
        "train_interval": args.train_interval,
        "num_random_episodes": args.num_random_episodes,
        "live_progress_interval": args.live_progress_interval,
        "share_policy": all_args.share_policy,
        "use_same_share_obs": all_args.use_same_share_obs,
        "checkpoint_interval_steps": all_args.save_interval,
        "cuda": all_args.cuda,
        "ctde_share_observation": "padded_joint_observation",
        "reward_function": "CityLearnV3MADRLRewardFunction",
        "reward_profile": "MATD3",
        "reward_metadata": env.adapter.reward_metadata,
    }

    try:
        while total_num_steps < all_args.num_env_steps:
            total_num_steps = runner.run()
        runner.saver()
        report = citylearn_v3_training_report(env)
        artifacts = write_training_artifacts(
            output_dir=output_dir,
            algorithm="MATD3",
            backend="external/off-policy",
            args=args,
            report=report,
            candidate=env,
            hyperparameters=hyperparameters,
            extra={
                "episodes": configured_episodes,
                "num_env_steps": configured_num_env_steps,
                "total_num_steps": total_num_steps,
            },
        )
    finally:
        env.close()
        eval_env.close()
        if hasattr(runner, "writter"):
            runner.writter.export_scalars_to_json(str(Path(runner.log_dir) / "summary.json"))
            runner.writter.close()

    write_training_summary(
        output_dir,
        {
            "algorithm": "MATD3",
            "backend": "external/off-policy",
            "scenario": args.scenario,
            "seed": args.seed,
            "episode_time_steps": args.episode_time_steps,
            "episodes": configured_episodes,
            "num_env_steps": configured_num_env_steps,
            "output_dir": str(output_dir),
            "artifact_layout": artifacts.get("artifact_layout", {}),
            "hyperparameters": hyperparameters,
            "reward_metadata": env.adapter.reward_metadata,
            "artifacts": artifacts,
            "project_axis_metrics": report["project_axis_metrics"],
            "citylearn_v3_report": report,
        },
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("WANDB_MODE", "disabled")
    raise SystemExit(main())
