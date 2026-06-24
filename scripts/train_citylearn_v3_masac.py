"""Train MASAC/mSAC on CityLearn v3 using the paper repository backend."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from masac_runtime_optimizations import install_masac_runtime_optimizations

from citylearn_v3_training_common import (
    CityLearnSMACDiscreteEnv,
    add_common_citylearn_args,
    add_external_path,
    citylearn_v3_training_report,
    configure_torch_runtime,
    ensure_project_paths,
    ensure_artifact_layout,
    install_finite_optimizer_step_guard,
    resolve_output_dir,
    discover_job_resume_plan,
    write_job_resume_manifest,
    load_masac_checkpoint_bundle,
    find_masac_checkpoint_bundle,
    start_live_progress_heartbeat,
    stop_live_progress_heartbeat,
    write_training_artifacts,
    write_training_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_citylearn_args(parser)
    parser.add_argument("--episodes", default=None, type=int, help="Number of MASAC training episodes/epochs.")
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--action-bins", default=3, type=int)
    parser.add_argument(
        "--discrete-action-mode",
        default="axis",
        choices=("axis", "shared", "cartesian"),
        help="Discrete CityLearn action table. axis is linear in action dimensions; cartesian is exponential.",
    )
    parser.add_argument(
        "--max-replay-buffer-gib",
        default=8.0,
        type=float,
        help="Abort before backend startup if MASAC replay buffer would exceed this estimate.",
    )
    parser.add_argument("--buffer-size", default=20, type=int)
    parser.add_argument("--critic-batch-size", default=64, type=int)
    parser.add_argument("--critic-train-steps", default=1, type=int)
    parser.add_argument("--actor-sample-times", default=5, type=int)
    parser.add_argument(
        "--masac-preload-batch-device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help=(
            "Where the optimized MASAC backend should preload replay batches. "
            "auto tries CUDA and falls back to CPU on OOM."
        ),
    )
    parser.add_argument("--actor-lr", default=3.0e-4, type=float)
    parser.add_argument("--critic-lr", default=5.0e-4, type=float)
    parser.add_argument("--alpha-lr", default=3.0e-4, type=float)
    parser.add_argument("--grad-norm-clip", default=1.0, type=float)
    parser.add_argument("--rnn-hidden-dim", default=256, type=int)
    parser.add_argument("--qmix-hidden-dim", default=128, type=int)
    parser.add_argument("--hyper-hidden-dim", default=256, type=int)
    parser.add_argument("--gamma", default=0.9999, type=float,
                        help="Discount factor. Use 0.9999 for year-long (8760-step) episodes.")
    parser.add_argument("--use-recurrent-policy", action="store_true",
                        help="Enable RNN (GRU) recurrent policy in QMIX backend.")
    parser.add_argument("--torch-threads", default=1, type=int)
    parser.add_argument(
        "--live-heartbeat-seconds",
        default=30,
        type=int,
        help="Seconds between live_progress heartbeat writes while MASAC updates backend networks.",
    )
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def estimate_replay_buffer_gib(
    *,
    buffer_size: int,
    episode_limit: int,
    n_agents: int,
    n_actions: int,
    obs_shape: int,
    state_shape: int,
) -> float:
    """Estimate the external MASAC replay buffer allocation in GiB."""

    transition_elements = (
        n_agents * obs_shape
        + n_agents
        + state_shape
        + 1
        + n_agents * obs_shape
        + state_shape
        + n_agents * n_actions
        + n_agents * n_actions
        + n_agents * n_actions
        + 1
        + 1
    )
    bytes_required = int(buffer_size) * int(episode_limit) * int(transition_elements) * 8
    return bytes_required / float(1024**3)


def main() -> int:
    args = parse_args()
    ensure_project_paths()
    add_external_path("MARL", "src")

    from common.arguments import get_common_args, get_mixer_args
    masac_backend_optimization = install_masac_runtime_optimizations()
    from runner_msac import Runner

    output_dir = resolve_output_dir(args.output_dir, "masac", args.scenario, args.seed)
    artifact_dirs = ensure_artifact_layout(output_dir)
    configured_episodes = int(args.episodes if args.episodes is not None else args.epochs)

    resume_plan = discover_job_resume_plan(
        output_dir,
        algorithm="masac",
        target_episodes=configured_episodes,
        episode_time_steps=args.episode_time_steps,
        allow_resume=bool(getattr(args, "resume", True)),
    )
    if resume_plan.get("active"):
        configured_episodes = int(resume_plan["remaining_episodes"])
        print(
            f"[masac] RESUME ep {resume_plan['completed_episodes']}/{resume_plan['target_episodes']} "
            f"-> {configured_episodes} remaining",
            flush=True,
        )
    write_job_resume_manifest(output_dir, resume_plan)

    env = CityLearnSMACDiscreteEnv(
        schema_path=args.schema_path,
        scenario=args.scenario,
        seed=args.seed,
        episode_time_steps=args.episode_time_steps,
        action_bins=args.action_bins,
        discrete_action_mode=args.discrete_action_mode,
        algorithm="MASAC",
        live_progress_path=str(output_dir / "live_progress.json"),
        live_progress_interval=args.live_progress_interval,
        trace_record_interval=args.trace_record_interval,
        trace_detail=args.trace_detail,
        normalize_observations=args.normalize_observations,
    )
    env_info = env.get_env_info()
    gpu_runtime = configure_torch_runtime(
        torch,
        use_cuda=args.cuda,
        torch_threads=args.torch_threads,
        gpu_profile=args.gpu_profile,
        cuda_memory_fraction=args.cuda_memory_fraction,
    )

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
    backend_args.load_model = False
    backend_args.evaluate_cycle = max(configured_episodes + 1, 2)
    backend_args.evaluate_epoch = 1
    backend_args.n_episodes = 1
    backend_args.cuda = bool(gpu_runtime["cuda_enabled"])
    backend_args.result_dir = str(artifact_dirs["data"] / "backend_results")
    backend_args.model_dir = str(artifact_dirs["checkpoints"] / "models")
    backend_args.replay_dir = ""
    backend_args.n_actions = env_info["n_actions"]
    backend_args.n_agents = env_info["n_agents"]
    backend_args.state_shape = env_info["state_shape"]
    backend_args.obs_shape = env_info["obs_shape"]
    backend_args.episode_limit = env_info["episode_limit"]
    backend_args.critic_batch_size = max(1, int(args.critic_batch_size))
    backend_args.critic_train_steps = max(1, int(args.critic_train_steps))
    backend_args.actor_sample_times = max(1, int(args.actor_sample_times))
    backend_args.citylearn_preload_batch_device = args.masac_preload_batch_device
    backend_args.actor_lr = float(args.actor_lr)
    backend_args.critic_lr = float(args.critic_lr)
    backend_args.grad_norm_clip = float(args.grad_norm_clip)
    backend_args.rnn_hidden_dim = max(1, int(args.rnn_hidden_dim))
    backend_args.qmix_hidden_dim = max(1, int(args.qmix_hidden_dim))
    backend_args.hyper_hidden_dim = max(1, int(args.hyper_hidden_dim))
    backend_args.buffer_size = max(int(args.buffer_size), 2)
    backend_args.gamma = float(args.gamma)
    backend_args.use_rnn = bool(args.use_recurrent_policy)
    estimated_replay_buffer_gib = estimate_replay_buffer_gib(
        buffer_size=backend_args.buffer_size,
        episode_limit=backend_args.episode_limit,
        n_agents=backend_args.n_agents,
        n_actions=backend_args.n_actions,
        obs_shape=backend_args.obs_shape,
        state_shape=backend_args.state_shape,
    )

    if estimated_replay_buffer_gib > float(args.max_replay_buffer_gib):
        raise MemoryError(
            "MASAC replay buffer estimate is too large: "
            f"{estimated_replay_buffer_gib:.2f} GiB > {args.max_replay_buffer_gib:.2f} GiB. "
            "Use --discrete-action-mode axis for CityLearn multi-actuator actions, "
            "or reduce episode length, buffer size or action bins."
        )

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
        "discrete_action_mode": args.discrete_action_mode,
        "discrete_action_metadata": env.adapter.discrete_action_metadata,
        "n_discrete_actions": env_info["n_actions"],
        "estimated_replay_buffer_gib": estimated_replay_buffer_gib,
        "max_replay_buffer_gib": args.max_replay_buffer_gib,
        "critic_batch_size": backend_args.critic_batch_size,
        "critic_train_steps": backend_args.critic_train_steps,
        "actor_sample_times": backend_args.actor_sample_times,
        "masac_preload_batch_device": backend_args.citylearn_preload_batch_device,
        "masac_backend_optimization": masac_backend_optimization,
        "actor_lr": backend_args.actor_lr,
        "critic_lr": backend_args.critic_lr,
        "alpha_lr": args.alpha_lr,
        "grad_norm_clip": backend_args.grad_norm_clip,
        "buffer_size": backend_args.buffer_size,
        "gamma": backend_args.gamma,
        "use_recurrent_policy": backend_args.use_rnn,
        "rnn_hidden_dim": backend_args.rnn_hidden_dim,
        "qmix_hidden_dim": backend_args.qmix_hidden_dim,
        "hyper_hidden_dim": backend_args.hyper_hidden_dim,
        "torch_threads": args.torch_threads,
        "live_progress_interval": args.live_progress_interval,
        "live_heartbeat_seconds": args.live_heartbeat_seconds,
        "cuda": backend_args.cuda,
        "gpu_runtime": gpu_runtime,
        "algorithm_family": "MADRL",
        "backend_repository": "external/MARL/src",
        "ctde_state_shape": backend_args.state_shape,
        "reward_function": "CityLearnV3MADRLRewardFunction",
        "reward_profile": "MASAC",
        "reward_metadata": env.adapter.reward_metadata,
        "job_resume": resume_plan,
    }

    heartbeat_stop = None
    heartbeat_thread = None
    try:
        runner = Runner(env, backend_args)
        learner = getattr(runner, "qmix_pg_learner", None)
        if resume_plan.get("active") and learner is not None:
            bundle = find_masac_checkpoint_bundle(artifact_dirs["checkpoints"] / "models")
            if bundle:
                load_masac_checkpoint_bundle(learner, bundle, use_cuda=bool(backend_args.cuda))
                print("[masac] Loaded checkpoint bundle for resume", flush=True)
        if learner is not None and getattr(learner, "alpha_optimizer", None) is not None:
            for group in learner.alpha_optimizer.param_groups:
                group["lr"] = float(args.alpha_lr)
        finite_optimizer_guard = install_finite_optimizer_step_guard(
            [
                {
                    "owner": "masac_policy",
                    "module": getattr(getattr(learner, "agent", None), "policy", None),
                    "optimizer": getattr(learner, "policy_optimiser", None),
                },
                {
                    "owner": "masac_critic_q1",
                    "parameters": getattr(learner, "eval_parameters", None),
                    "optimizer": getattr(learner, "optimizer", None),
                },
                {
                    "owner": "masac_critic_q2",
                    "parameters": getattr(learner, "eval_parameters_2", None),
                    "optimizer": getattr(learner, "optimizer_2", None),
                },
                {
                    "owner": "masac_alpha",
                    "parameters": [getattr(learner, "log_alpha", None)],
                    "optimizer": getattr(learner, "alpha_optimizer", None),
                },
            ],
            output_dir / "data" / "masac_finite_gradient_guard.jsonl",
        )
        hyperparameters["finite_optimizer_step_guard"] = finite_optimizer_guard
        heartbeat_stop, heartbeat_thread = start_live_progress_heartbeat(
            env.adapter,
            args.live_heartbeat_seconds,
            active_stage="masac_backend_training",
            initial_stage="masac_backend_starting",
            note="MASAC backend is updating critic/actor networks between CityLearn environment steps.",
        )
        runner.run(args.seed)
        final_train_step = max(1, int(configured_episodes) * max(1, int(backend_args.critic_train_steps)))
        if learner is not None and hasattr(learner, "save_model"):
            learner.save_model(final_train_step)
            hyperparameters["final_checkpoint_save_step"] = final_train_step
        try:
            env.adapter.write_live_heartbeat(
                stage="masac_backend_finished",
                note="MASAC backend runner finished; writing final artifacts.",
            )
        except Exception:
            pass
        report = citylearn_v3_training_report(env)
        artifacts = write_training_artifacts(
            output_dir=output_dir,
            algorithm="MASAC",
            backend="external/MADRL-MASAC",
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
        stop_live_progress_heartbeat(heartbeat_stop, heartbeat_thread)
        env.close()

    write_training_summary(
        output_dir,
        {
            "algorithm": "MASAC",
            "algorithm_family": "MADRL",
            "backend": "external/MADRL-MASAC",
            "backend_repository": "external/MARL/src",
            "scenario": args.scenario,
            "seed": args.seed,
            "episode_time_steps": args.episode_time_steps,
            "episodes": configured_episodes,
            "epochs": configured_episodes,
            "action_bins": args.action_bins,
            "discrete_action_mode": args.discrete_action_mode,
            "discrete_action_metadata": env.adapter.discrete_action_metadata,
            "n_discrete_actions": env_info["n_actions"],
            "estimated_replay_buffer_gib": estimated_replay_buffer_gib,
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
    raise SystemExit(main())
