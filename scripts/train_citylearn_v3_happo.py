"""Train HAPPO on CityLearn v3 using the official HARL backend."""

from __future__ import annotations

import argparse
import sys

import torch

from citylearn_v3_training_common import (
    count_completed_episodes,
    find_latest_harl_models_dir,
    CityLearnHARLEnv,
    FiniteTensorBoardWriter,
    add_common_citylearn_args,
    add_external_path,
    citylearn_v3_training_report,
    configure_torch_runtime,
    ensure_project_paths,
    ensure_artifact_layout,
    install_harl_finite_optimizer_step_guard,
    resolve_output_dir,
    start_live_progress_heartbeat,
    stop_live_progress_heartbeat,
    write_training_artifacts,
    write_training_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_citylearn_args(parser)
    parser.add_argument("--episodes", default=None, type=int, help="Number of rollout episodes to train.")
    parser.add_argument("--num-env-steps", default=8, type=int)
    parser.add_argument("--hidden-size", default=256, type=int)
    parser.add_argument("--torch-threads", default=1, type=int)
    parser.add_argument("--live-heartbeat-seconds", default=30, type=int)
    parser.add_argument("--n-rollout-threads", default=1, type=int)
    parser.add_argument("--log-interval", default=1, type=int)
    parser.add_argument("--eval-interval", default=1, type=int)
    parser.add_argument("--actor-lr", default=1.0e-4, type=float)
    parser.add_argument("--critic-lr", default=5.0e-4, type=float)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--gamma", default=0.9999, type=float,
                        help="Discount factor. Use 0.9999 for year-long (8760-step) episodes.")
    parser.add_argument("--use-recurrent-policy", action="store_true",
                        help="Enable GRU recurrent policy in HARL for temporal sequence learning.")
    parser.add_argument("--action-aggregation", default="mean", choices=("mean", "prod"))
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
    rollout_threads = max(1, int(args.n_rollout_threads))
    configured_num_env_steps = max(
        args.num_env_steps,
        args.episode_time_steps,
        (args.episodes or 0) * args.episode_time_steps * rollout_threads,
    )
    configured_episodes = max(1, configured_num_env_steps // max(args.episode_time_steps, 1) // rollout_threads)


    # ── Checkpoint-resume: detect completed episodes + latest models dir ──────
    _completed = count_completed_episodes(output_dir)
    _ckpt_dir  = artifact_dirs["checkpoints"]
    _models_dir = find_latest_harl_models_dir(_ckpt_dir)
    _remaining_episodes = max(1, configured_episodes - _completed)
    _episode_offset = _completed
    if _completed > 0 and _models_dir is not None:
        print(
            f"[HAPPO/{args.scenario}] Resuming from episode {_completed} "
            f"({_remaining_episodes} remaining). Checkpoint: {_models_dir}",
            flush=True,
        )
        configured_num_env_steps = _remaining_episodes * args.episode_time_steps
        configured_episodes = _remaining_episodes
    elif _completed > 0:
        print(
            f"[HAPPO/{args.scenario}] {_completed} episode(s) detected but no "
            f"checkpoint found — starting fresh.",
            flush=True,
        )
        _episode_offset = 0
    # ─────────────────────────────────────────────────────────────────────────
    def make_citylearn_train_env(env_name, seed, n_threads, env_args):
        def make_env(rank):
            def init_env():
                env = CityLearnHARLEnv(
                    schema_path=args.schema_path,
                    episode_offset=_episode_offset,
                    scenario=args.scenario,
                    seed=args.seed + rank * 1000,
                    episode_time_steps=args.episode_time_steps,
                    algorithm="HAPPO",
                    live_progress_path=str(output_dir / "live_progress.json"),
                    live_progress_interval=args.live_progress_interval,
                    trace_record_interval=args.trace_record_interval,
                    trace_detail=args.trace_detail,
                    normalize_observations=args.normalize_observations,
                )
                env.seed(seed + rank * 1000)
                return env

            return init_env

        return ShareDummyVecEnv([make_env(rank) for rank in range(n_threads)])

    base_runner.make_train_env = make_citylearn_train_env

    algo_args, env_args = get_defaults_yaml_args("happo", "gym")
    gpu_runtime = configure_torch_runtime(
        torch,
        use_cuda=args.cuda,
        torch_threads=args.torch_threads,
        gpu_profile=args.gpu_profile,
        cuda_memory_fraction=args.cuda_memory_fraction,
    )
    algo_args["seed"]["seed"] = args.seed
    algo_args["device"]["cuda"] = bool(gpu_runtime["cuda_enabled"])
    algo_args["device"]["torch_threads"] = args.torch_threads
    algo_args["train"]["n_rollout_threads"] = rollout_threads
    algo_args["train"]["episode_length"] = args.episode_time_steps
    algo_args["train"]["num_env_steps"] = configured_num_env_steps
    algo_args["train"]["log_interval"] = max(1, int(args.log_interval))
    algo_args["train"]["eval_interval"] = max(1, int(args.eval_interval))
    algo_args["eval"]["use_eval"] = False
    algo_args["logger"]["log_dir"] = str(artifact_dirs["checkpoints"])
    if _completed > 0 and _models_dir is not None:
        algo_args["train"]["model_dir"] = str(_models_dir)
    algo_args["model"]["hidden_sizes"] = [args.hidden_size, args.hidden_size]
    algo_args["model"]["lr"] = float(args.actor_lr)
    algo_args["model"]["critic_lr"] = float(args.critic_lr)
    algo_args["algo"]["use_max_grad_norm"] = True
    algo_args["algo"]["max_grad_norm"] = float(args.max_grad_norm)
    algo_args["algo"]["action_aggregation"] = args.action_aggregation
    algo_args["algo"]["share_param"] = False
    algo_args["algo"]["gamma"] = float(args.gamma)
    algo_args["model"]["use_recurrent_policy"] = bool(args.use_recurrent_policy)
    env_args["scenario"] = f"citylearn_v3_{args.scenario}"
    env_args["state_type"] = "EP"

    runner_args = {"algo": "happo", "env": "gym", "exp_name": args.exp_name}
    runner = RUNNER_REGISTRY["happo"](runner_args, algo_args, env_args)
    tensorboard_audit_path = output_dir / "data" / "tensorboard_finite_filter.jsonl"
    finite_writer = FiniteTensorBoardWriter(runner.writter, tensorboard_audit_path)
    runner.writter = finite_writer
    if getattr(runner, "logger", None) is not None:
        runner.logger.writter = finite_writer
    gradient_guard = install_harl_finite_optimizer_step_guard(
        runner,
        output_dir / "data" / "happo_finite_gradient_guard.jsonl",
    )
    reward_metadata = {}
    runner_envs = getattr(runner.envs, "envs", None)
    report_candidate = runner.envs
    if runner_envs:
        report_candidate = runner_envs[0]
        reward_metadata = getattr(report_candidate.adapter, "reward_metadata", {})
    report = citylearn_v3_training_report(None)
    artifacts = {}
    hyperparameters = {
        "episodes": configured_episodes,
        "num_env_steps": configured_num_env_steps,
        "episode_length": args.episode_time_steps,
        "hidden_sizes": algo_args["model"]["hidden_sizes"],
        "actor_lr": algo_args["model"]["lr"],
        "critic_lr": algo_args["model"]["critic_lr"],
        "max_grad_norm": algo_args["algo"]["max_grad_norm"],
        "action_aggregation": algo_args["algo"]["action_aggregation"],
        "gamma": algo_args["algo"]["gamma"],
        "use_recurrent_policy": algo_args["model"]["use_recurrent_policy"],
        "torch_threads": args.torch_threads,
        "share_param": algo_args["algo"]["share_param"],
        "ctde_state_type": env_args["state_type"],
        "n_rollout_threads": algo_args["train"]["n_rollout_threads"],
        "log_interval": algo_args["train"]["log_interval"],
        "checkpoint_interval_episodes": algo_args["train"]["eval_interval"],
        "live_progress_interval": args.live_progress_interval,
        "live_heartbeat_seconds": args.live_heartbeat_seconds,
        "cuda": algo_args["device"]["cuda"],
        "gpu_runtime": gpu_runtime,
        "finite_optimizer_step_guard": gradient_guard,
        "algorithm_family": "MADRL",
        "reward_function": "CityLearnV3MADRLRewardFunction",
        "reward_profile": "HAPPO",
        "reward_metadata": reward_metadata,
    }

    heartbeat_stop = None
    heartbeat_thread = None
    heartbeat_adapter = getattr(report_candidate, "adapter", None)
    try:
        heartbeat_stop, heartbeat_thread = start_live_progress_heartbeat(
            heartbeat_adapter,
            args.live_heartbeat_seconds,
            active_stage="happo_backend_training",
            initial_stage="happo_backend_starting",
            note="HAPPO backend is collecting rollouts or updating policy networks.",
        )
        runner.run()
        if heartbeat_adapter is not None:
            try:
                heartbeat_adapter.write_live_heartbeat(
                    stage="happo_backend_finished",
                    note="HAPPO backend runner finished; writing final artifacts.",
                )
            except Exception:
                pass
        runner.save()
        report = citylearn_v3_training_report(report_candidate)
        artifacts = write_training_artifacts(
            output_dir=output_dir,
            algorithm="HAPPO",
            backend="external/HARL",
            args=args,
            report=report,
            candidate=report_candidate,
            hyperparameters=hyperparameters,
            extra={
                "episodes": configured_episodes,
                "num_env_steps": configured_num_env_steps,
            },
        )
    finally:
        stop_live_progress_heartbeat(heartbeat_stop, heartbeat_thread)
        runner.close()

    write_training_summary(
        output_dir,
        {
            "algorithm": "HAPPO",
            "algorithm_family": "MADRL",
            "backend": "external/HARL",
            "scenario": args.scenario,
            "seed": args.seed,
            "episode_time_steps": args.episode_time_steps,
            "episodes": configured_episodes,
            "num_env_steps": configured_num_env_steps,
            "output_dir": str(output_dir),
            "artifact_layout": artifacts.get("artifact_layout", {}),
            "hyperparameters": hyperparameters,
            "gpu_runtime": gpu_runtime,
            "reward_metadata": reward_metadata,
            "artifacts": artifacts,
            "project_axis_metrics": report["project_axis_metrics"],
            "citylearn_v3_report": report,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
