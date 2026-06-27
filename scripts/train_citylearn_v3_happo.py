"""Train HAPPO on CityLearn v3 using the official HARL backend."""

from __future__ import annotations

import argparse
import sys
import traceback

import torch

from citylearn_v3_training_common import (
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
    discover_job_resume_plan,
    write_job_resume_manifest,
    start_live_progress_heartbeat,
    stop_live_progress_heartbeat,
    write_minimal_results_json,
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
    parser.add_argument(
        "--num-mini-batch",
        default=0,
        type=int,
        help="PPO minibatches para actor y critic. 0=auto: mantiene el minibatch de GPU "
        "~constante al subir n_rollout_threads (mas RAM de sistema, misma VRAM).",
    )
    parser.add_argument(
        "--gpu-rollout-ref",
        default=8,
        type=int,
        help="Rollouts de referencia por minibatch GPU para el modo auto de --num-mini-batch.",
    )
    parser.add_argument("--ppo-epoch", default=5, type=int,
                        help="PPO actor epochs per update. Higher = more GPU work + sample efficiency.")
    parser.add_argument("--critic-epoch", default=5, type=int,
                        help="Critic epochs per update. Higher = more GPU work + sample efficiency.")
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

    from harl.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
    import harl.runners.on_policy_base_runner as base_runner
    from harl.runners import RUNNER_REGISTRY
    from harl.utils.configs_tools import get_defaults_yaml_args

    output_dir = resolve_output_dir(args.output_dir, "happo", args.scenario, args.seed)
    artifact_dirs = ensure_artifact_layout(output_dir)
    rollout_threads = max(1, int(args.n_rollout_threads))
    # Mantener acotado el minibatch de GPU al escalar rollouts: mas workers agrandan el
    # buffer numpy (RAM de sistema) pero num_mini_batch divide el batch -> VRAM por update
    # ~constante. mini_batch_size = n_rollout_threads * episode_length / num_mini_batch.
    import math as _math

    gpu_rollout_ref = max(1, int(getattr(args, "gpu_rollout_ref", 8) or 8))
    if int(getattr(args, "num_mini_batch", 0) or 0) > 0:
        num_mini_batch = int(args.num_mini_batch)
    else:
        num_mini_batch = max(1, _math.ceil(rollout_threads / gpu_rollout_ref))
    num_mini_batch = min(num_mini_batch, rollout_threads)
    configured_num_env_steps = max(
        args.num_env_steps,
        args.episode_time_steps,
        (args.episodes or 0) * args.episode_time_steps * rollout_threads,
    )
    configured_episodes = max(1, configured_num_env_steps // max(args.episode_time_steps, 1) // rollout_threads)

    resume_plan = discover_job_resume_plan(
        output_dir,
        algorithm="happo",
        target_episodes=configured_episodes,
        episode_time_steps=args.episode_time_steps,
        rollout_threads=rollout_threads,
        allow_resume=bool(getattr(args, "resume", True)),
    )
    if resume_plan.get("active"):
        configured_episodes = int(resume_plan["remaining_episodes"])
        configured_num_env_steps = int(resume_plan["remaining_num_env_steps"])
        print(
            f"[happo] RESUME ep {resume_plan['completed_episodes']}/{resume_plan['target_episodes']} "
            f"-> {configured_episodes} remaining",
            flush=True,
        )
    write_job_resume_manifest(output_dir, resume_plan)

    def make_citylearn_train_env(env_name, seed, n_threads, env_args):
        def make_env(rank):
            def init_env():
                env = CityLearnHARLEnv(
                    schema_path=args.schema_path,
                    scenario=args.scenario,
                    seed=args.seed + rank * 1000,
                    episode_time_steps=args.episode_time_steps,
                    algorithm="HAPPO",
                    live_progress_path=(
                        str(output_dir / "live_progress.json") if rank == 0 else None
                    ),
                    live_progress_interval=args.live_progress_interval,
                    trace_record_interval=args.trace_record_interval,
                    trace_detail=args.trace_detail,
                    normalize_observations=args.normalize_observations,
                )
                env.seed(seed + rank * 1000)
                return env

            return init_env

        # SubprocVecEnv: cada env corre en proceso separado (bypass GIL, ~n_threads× FPS).
        # DummyVecEnv solo como fallback si hay error de spawn (ej. entorno Windows/local).
        env_fns = [make_env(rank) for rank in range(n_threads)]
        if n_threads > 1:
            try:
                return ShareSubprocVecEnv(env_fns)
            except Exception as _e:
                print(f"[happo] ShareSubprocVecEnv falló ({_e}), usando DummyVecEnv.", flush=True)
        return ShareDummyVecEnv(env_fns)

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
    algo_args["model"]["hidden_sizes"] = [args.hidden_size, args.hidden_size]
    algo_args["model"]["lr"] = float(args.actor_lr)
    algo_args["model"]["critic_lr"] = float(args.critic_lr)
    algo_args["algo"]["use_max_grad_norm"] = True
    algo_args["algo"]["max_grad_norm"] = float(args.max_grad_norm)
    algo_args["algo"]["actor_num_mini_batch"] = num_mini_batch
    algo_args["algo"]["critic_num_mini_batch"] = num_mini_batch
    algo_args["algo"]["ppo_epoch"] = max(1, int(args.ppo_epoch))
    algo_args["algo"]["critic_epoch"] = max(1, int(args.critic_epoch))
    algo_args["algo"]["action_aggregation"] = args.action_aggregation
    algo_args["algo"]["share_param"] = False
    algo_args["algo"]["gamma"] = float(args.gamma)
    algo_args["model"]["use_recurrent_policy"] = bool(args.use_recurrent_policy)
    env_args["scenario"] = f"citylearn_v3_{args.scenario}"
    env_args["state_type"] = "EP"

    if resume_plan.get("active") and resume_plan.get("model_dir"):
        algo_args["train"]["model_dir"] = str(resume_plan["model_dir"])

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
        if resume_plan.get("active"):
            _adapter = getattr(runner_envs[0], "adapter", None)
            if _adapter is not None:
                _preload = _adapter.preload_resume_artifacts(
                    int(resume_plan["completed_episodes"])
                )
                print(f"[happo] preloaded resume artifacts: {_preload}", flush=True)
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
        "actor_num_mini_batch": num_mini_batch,
        "critic_num_mini_batch": num_mini_batch,
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
        "job_resume": resume_plan,
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
        # HAPPO overwrites actor/critic .pt every eval_interval episode, so weights are
        # always on disk. Still salvage and write artifacts on any failure so a late
        # crash (backend tail work, Drive I/O) never discards real progress.
        run_error: BaseException | None = None
        try:
            runner.run()
        except Exception as exc:  # noqa: BLE001 - salvage on any backend failure
            run_error = exc
            traceback.print_exc()
            print(
                f"[happo] runner.run() raised {type(exc).__name__}: {exc}. "
                "Salvaging trained model and writing artifacts so progress is not lost.",
                flush=True,
            )
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        if heartbeat_adapter is not None:
            try:
                heartbeat_adapter.write_live_heartbeat(
                    stage="happo_backend_finished",
                    note="HAPPO backend runner finished; writing final artifacts.",
                )
            except Exception:
                pass
        try:
            runner.save()
        except Exception as exc:  # noqa: BLE001 - never let checkpointing abort finalization
            print(f"[happo] final save() failed: {exc}", flush=True)
        if run_error is not None:
            hyperparameters["run_completed_with_salvage"] = True
            hyperparameters["run_error"] = f"{type(run_error).__name__}: {run_error}"
        try:
            report = citylearn_v3_training_report(report_candidate)
        except Exception as exc:  # noqa: BLE001 - keep fallback report
            print(f"[happo] training report failed, using fallback report: {exc}", flush=True)
        try:
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
        except Exception as exc:  # noqa: BLE001 - guarantee a results.json regardless
            print(
                f"[happo] write_training_artifacts failed ({type(exc).__name__}: {exc}); "
                "writing minimal salvage results.json.",
                flush=True,
            )
            write_minimal_results_json(
                output_dir=output_dir,
                algorithm="HAPPO",
                backend="external/HARL",
                args=args,
                hyperparameters=hyperparameters,
                report=report,
                error=exc,
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
