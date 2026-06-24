"""Train MATD3 on CityLearn v3 using the PyTorch marlbenchmark/off-policy backend."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from citylearn_v3_training_common import (
    CityLearnOffPolicyVecEnv,
    FiniteTensorBoardWriter,
    add_common_citylearn_args,
    add_external_path,
    citylearn_v3_training_report,
    configure_torch_runtime,
    ensure_project_paths,
    ensure_artifact_layout,
    install_finite_optimizer_step_guard,
    install_noop_wandb,
    resolve_output_dir,
    discover_job_resume_plan,
    write_job_resume_manifest,
    start_live_progress_heartbeat,
    stop_live_progress_heartbeat,
    write_training_artifacts,
    write_training_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_citylearn_args(parser)
    parser.add_argument("--episodes", default=None, type=int, help="Number of MATD3 rollout episodes.")
    parser.add_argument("--num-env-steps", default=8, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--buffer-size", default=500, type=int)
    parser.add_argument("--hidden-size", default=256, type=int)
    parser.add_argument("--lr", default=3.0e-4, type=float)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--gamma", default=0.9999, type=float,
                        help="Discount factor. Use 0.9999 for year-long (8760-step) episodes.")
    parser.add_argument("--train-interval", default=1, type=int)
    parser.add_argument("--num-random-episodes", default=1, type=int)
    parser.add_argument("--torch-threads", default=1, type=int)
    parser.add_argument("--live-heartbeat-seconds", default=30, type=int)
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

    resume_plan = discover_job_resume_plan(
        output_dir,
        algorithm="matd3",
        target_episodes=configured_episodes,
        episode_time_steps=args.episode_time_steps,
        allow_resume=bool(getattr(args, "resume", True)),
    )
    if resume_plan.get("active"):
        configured_episodes = int(resume_plan["remaining_episodes"])
        configured_num_env_steps = int(resume_plan["remaining_num_env_steps"])
        print(
            f"[matd3] RESUME ep {resume_plan['completed_episodes']}/{resume_plan['target_episodes']} "
            f"-> {configured_episodes} remaining",
            flush=True,
        )
    write_job_resume_manifest(output_dir, resume_plan)

    env = CityLearnOffPolicyVecEnv(
        schema_path=args.schema_path,
        scenario=args.scenario,
        seed=args.seed,
        episode_time_steps=args.episode_time_steps,
        algorithm="MATD3",
        live_progress_path=str(output_dir / "live_progress.json"),
        live_progress_interval=args.live_progress_interval,
        trace_record_interval=args.trace_record_interval,
        trace_detail=args.trace_detail,
        normalize_observations=args.normalize_observations,
    )
    eval_env = CityLearnOffPolicyVecEnv(
        schema_path=args.schema_path,
        scenario=args.scenario,
        seed=args.seed + 10000,
        episode_time_steps=args.episode_time_steps,
        algorithm="MATD3",
        trace_record_interval=args.trace_record_interval,
        trace_detail=args.trace_detail,
        normalize_observations=args.normalize_observations,
    )
    gpu_runtime = configure_torch_runtime(
        torch,
        use_cuda=args.cuda,
        torch_threads=args.torch_threads,
        gpu_profile=args.gpu_profile,
        cuda_memory_fraction=args.cuda_memory_fraction,
    )

    parser = get_config()
    all_args = parser.parse_args([])
    all_args.algorithm_name = "matd3"
    all_args.env_name = "CityLearnV3"
    all_args.scenario_name = args.scenario
    all_args.experiment_name = args.experiment_name
    all_args.seed = args.seed
    all_args.cuda = bool(gpu_runtime["cuda_enabled"])
    all_args.use_wandb = False
    all_args.use_eval = False
    all_args.num_env_steps = configured_num_env_steps
    all_args.episode_length = args.episode_time_steps
    all_args.batch_size = args.batch_size
    all_args.buffer_size = args.buffer_size
    all_args.hidden_size = args.hidden_size
    all_args.lr = float(args.lr)
    all_args.max_grad_norm = float(args.max_grad_norm)
    all_args.use_max_grad_norm = True
    all_args.train_interval = args.train_interval
    all_args.num_random_episodes = args.num_random_episodes
    all_args.gamma = float(args.gamma)
    all_args.log_interval = max(args.episode_time_steps, 1)
    all_args.save_interval = max(args.episode_time_steps, 1)
    all_args.n_rollout_threads = 1
    all_args.n_eval_rollout_threads = 1
    all_args.share_policy = False
    all_args.use_same_share_obs = True
    all_args.use_available_actions = False
    if resume_plan.get("active") and resume_plan.get("model_dir"):
        model_root = Path(str(resume_plan["model_dir"]))
        all_args.model_dir = model_root.as_posix().rstrip("/") + "/"

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
    if hasattr(runner, "writter"):
        finite_writer = FiniteTensorBoardWriter(
            runner.writter,
            output_dir / "data" / "tensorboard_finite_filter.jsonl",
        )
        runner.writter = finite_writer
    finite_optimizer_guard = install_finite_optimizer_step_guard(
        [
            {
                "owner": f"{policy_id}/actor",
                "module": getattr(policy, "actor", None),
                "optimizer": getattr(policy, "actor_optimizer", None),
            }
            for policy_id, policy in getattr(runner, "policies", {}).items()
        ]
        + [
            {
                "owner": f"{policy_id}/critic",
                "module": getattr(policy, "critic", None),
                "optimizer": getattr(policy, "critic_optimizer", None),
            }
            for policy_id, policy in getattr(runner, "policies", {}).items()
        ],
        output_dir / "data" / "matd3_finite_gradient_guard.jsonl",
    )
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
        "gamma": all_args.gamma,
        "lr": all_args.lr,
        "max_grad_norm": all_args.max_grad_norm,
        "train_interval": args.train_interval,
        "num_random_episodes": args.num_random_episodes,
        "torch_threads": args.torch_threads,
        "live_progress_interval": args.live_progress_interval,
        "share_policy": all_args.share_policy,
        "use_same_share_obs": all_args.use_same_share_obs,
        "checkpoint_interval_steps": all_args.save_interval,
        "live_heartbeat_seconds": args.live_heartbeat_seconds,
        "cuda": all_args.cuda,
        "gpu_runtime": gpu_runtime,
        "algorithm_family": "MADRL",
        "ctde_share_observation": "padded_joint_observation",
        "reward_function": "CityLearnV3MADRLRewardFunction",
        "reward_profile": "MATD3",
        "reward_metadata": env.adapter.reward_metadata,
        "finite_optimizer_step_guard": finite_optimizer_guard,
        "job_resume": resume_plan,
    }

    heartbeat_stop = None
    heartbeat_thread = None
    try:
        heartbeat_stop, heartbeat_thread = start_live_progress_heartbeat(
            env.adapter,
            args.live_heartbeat_seconds,
            active_stage="matd3_backend_training",
            initial_stage="matd3_backend_starting",
            note="MATD3 backend is stepping the vector environment or updating actor/critic networks.",
        )
        while total_num_steps < all_args.num_env_steps:
            total_num_steps = runner.run()
        try:
            env.adapter.write_live_heartbeat(
                stage="matd3_backend_finished",
                note="MATD3 backend runner finished; writing final artifacts.",
            )
        except Exception:
            pass
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
        stop_live_progress_heartbeat(heartbeat_stop, heartbeat_thread)
        env.close()
        eval_env.close()
        if hasattr(runner, "writter"):
            runner.writter.export_scalars_to_json(str(Path(runner.log_dir) / "summary.json"))
            runner.writter.close()

    write_training_summary(
        output_dir,
        {
            "algorithm": "MATD3",
            "algorithm_family": "MADRL",
            "backend": "external/off-policy",
            "scenario": args.scenario,
            "seed": args.seed,
            "episode_time_steps": args.episode_time_steps,
            "episodes": configured_episodes,
            "num_env_steps": configured_num_env_steps,
            "output_dir": str(output_dir),
            "artifact_layout": artifacts.get("artifact_layout", {}),
            "hyperparameters": hyperparameters,
            "gpu_runtime": gpu_runtime,
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
