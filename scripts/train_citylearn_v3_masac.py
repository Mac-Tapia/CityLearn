"""Train MASAC/mSAC on CityLearn v3 using the paper repository backend."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

from checkpoint_state import restore_owner_training_state, wrap_save_with_training_state
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
    write_minimal_results_json,
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
        default=None,
        type=float,
        help=(
            "Refuse before backend startup if the MASAC replay estimate exceeds this value. "
            "Default: the measured available system RAM. The buffer is never shrunk."
        ),
    )
    parser.add_argument("--buffer-size", default=5000, type=int,
                        help="Replay capacity in EPISODES (author default 5000); never more than the run holds.")
    parser.add_argument(
        "--critic-batch-size",
        default=32,
        type=int,
        help="Episodes sampled per QMIX critic update (author default 32 episodes, not transitions).",
    )
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
    parser.add_argument("--qmix-hidden-dim", default=None, type=int,
                        help="Default: rnn_hidden_dim // 2 (author ratio rnn 64 / qmix 32).")
    parser.add_argument("--hyper-hidden-dim", default=None, type=int,
                        help="Default: rnn_hidden_dim (author ratio rnn 64 / hyper 64).")
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


def resolve_masac_capacity(
    *,
    episode_limit: int,
    configured_episodes: int,
    critic_batch_size: int,
    critic_train_steps: int,
    actor_sample_times: int,
    rnn_hidden_dim: int,
    qmix_hidden_dim=None,
    hyper_hidden_dim=None,
    buffer_size: int,
) -> dict:
    """Resolve the MASAC/QMIX capacity without any episode-length clamp.

    Author defaults (external/MARL/src/common/arguments.py): critic_batch_size 32
    episodes, critic_train_steps 1, buffer 5000 episodes, rnn 64 / qmix 32 / hyper 64.
    qmix and hyper widths keep the author ratio to rnn_hidden_dim (qmix = rnn/2,
    hyper = rnn) unless given explicitly. The only reduction is lossless: the episode
    buffer never holds more episodes than the run produces. The runner itself samples
    min(current_size, critic_batch_size) episodes.
    """
    rnn = max(1, int(rnn_hidden_dim))
    resolved = {
        "episode_limit": int(episode_limit),
        "critic_batch_size": max(1, int(critic_batch_size)),
        "critic_train_steps": max(1, int(critic_train_steps)),
        "actor_sample_times": max(1, int(actor_sample_times)),
        "rnn_hidden_dim": rnn,
        "qmix_hidden_dim": max(1, int(qmix_hidden_dim)) if qmix_hidden_dim is not None else max(1, rnn // 2),
        "hyper_hidden_dim": max(1, int(hyper_hidden_dim)) if hyper_hidden_dim is not None else rnn,
        "buffer_size_requested": int(buffer_size),
        "notes": [],
    }
    if qmix_hidden_dim is None or hyper_hidden_dim is None:
        resolved["notes"].append("qmix/hyper widths follow the author ratio to rnn_hidden_dim (1/2, 1)")
    run_episodes = max(2, int(configured_episodes))
    resolved["buffer_size"] = max(2, min(int(buffer_size), run_episodes))
    if resolved["buffer_size"] < int(buffer_size):
        resolved["notes"].append(
            f"buffer holds every episode of the run ({run_episodes}); requested {int(buffer_size)} episodes"
        )
    return resolved


def estimate_masac_critic_gpu_gib(
    *,
    batch_episodes: int,
    episode_limit: int,
    n_agents: int,
    obs_shape: int,
    n_actions: int,
    state_shape: int,
    rnn_hidden_dim: int,
    qmix_hidden_dim: int,
    hyper_hidden_dim: int,
) -> float:
    """Conservative float32 estimate of one QMIX critic update unrolled over B x T.

    Per (episode, step): eval and target agent inputs (obs + last action one-hot +
    agent id), state and next state, eval activations kept for backward (fc1 + 3 GRU
    gates + hidden = 5 x rnn per agent), eval/target Q values per action, and the
    hypernetwork mixer activations; x2 for gradients and allocator slack.
    """
    per_step = (
        2 * n_agents * (obs_shape + n_actions + n_agents)
        + 2 * state_shape
        + n_agents * 5 * rnn_hidden_dim
        + 2 * n_agents * n_actions
        + 2 * (n_agents * qmix_hidden_dim + qmix_hidden_dim + 2 * hyper_hidden_dim)
    )
    floats = int(batch_episodes) * int(episode_limit) * per_step
    return 2.0 * floats * 4 / float(1024**3)


def check_masac_capacity(*, replay_gib: float, available_ram_gib, critic_gpu_gib: float, free_vram_gib) -> None:
    """Refuse (never shrink) when the measured device cannot hold the configuration."""
    if available_ram_gib is not None and replay_gib > available_ram_gib:
        raise MemoryError(
            f"insufficient_resources:MASAC:ram replay {replay_gib:.2f} GiB > {available_ram_gib:.2f} GiB available"
        )
    if free_vram_gib is not None and critic_gpu_gib > free_vram_gib:
        raise MemoryError(
            f"insufficient_resources:MASAC:vram critic update {critic_gpu_gib:.2f} GiB > "
            f"{free_vram_gib:.2f} GiB free VRAM"
        )


def _available_ram_gib():
    try:
        import psutil

        return psutil.virtual_memory().available / float(1024**3)
    except Exception:  # noqa: BLE001 - measurement unavailable is reported as None
        return None


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

    # Resume offset injected via the adapter constructor (same pattern as HAPPO) so the
    # live-progress/CSV adapter continues from the resumed episode instead of ep1.
    masac_resume_completed = (
        int(resume_plan["completed_episodes"]) if resume_plan.get("active") else 0
    )
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
        resume_completed_episodes=masac_resume_completed,
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
    capacity = resolve_masac_capacity(
        episode_limit=backend_args.episode_limit,
        configured_episodes=configured_episodes,
        critic_batch_size=args.critic_batch_size,
        critic_train_steps=args.critic_train_steps,
        actor_sample_times=args.actor_sample_times,
        rnn_hidden_dim=args.rnn_hidden_dim,
        qmix_hidden_dim=args.qmix_hidden_dim,
        hyper_hidden_dim=args.hyper_hidden_dim,
        buffer_size=args.buffer_size,
    )
    for note in capacity["notes"]:
        print(f"[masac] {note}", flush=True)
    backend_args.critic_batch_size = capacity["critic_batch_size"]
    backend_args.critic_train_steps = capacity["critic_train_steps"]
    backend_args.actor_sample_times = capacity["actor_sample_times"]
    backend_args.citylearn_preload_batch_device = args.masac_preload_batch_device
    # Persist a resumable checkpoint after every QMIX critic update so an interruption
    # near the end of training never discards trained weights (resume always possible).
    backend_args.citylearn_masac_save_every_steps = 1
    backend_args.actor_lr = float(args.actor_lr)
    backend_args.critic_lr = float(args.critic_lr)
    backend_args.grad_norm_clip = float(args.grad_norm_clip)
    backend_args.rnn_hidden_dim = capacity["rnn_hidden_dim"]
    backend_args.qmix_hidden_dim = capacity["qmix_hidden_dim"]
    backend_args.hyper_hidden_dim = capacity["hyper_hidden_dim"]
    backend_args.buffer_size = capacity["buffer_size"]
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
    estimated_critic_gpu_gib = estimate_masac_critic_gpu_gib(
        batch_episodes=min(backend_args.critic_batch_size, backend_args.buffer_size),
        episode_limit=backend_args.episode_limit,
        n_agents=backend_args.n_agents,
        obs_shape=backend_args.obs_shape,
        n_actions=backend_args.n_actions,
        state_shape=backend_args.state_shape,
        rnn_hidden_dim=backend_args.rnn_hidden_dim,
        qmix_hidden_dim=backend_args.qmix_hidden_dim,
        hyper_hidden_dim=backend_args.hyper_hidden_dim,
    )
    available_ram_gib = (
        float(args.max_replay_buffer_gib) if args.max_replay_buffer_gib is not None else _available_ram_gib()
    )
    free_vram_gib = None
    if backend_args.cuda and torch.cuda.is_available():
        free_vram_gib = torch.cuda.mem_get_info()[0] / float(1024**3)
    print(
        f"[masac] capacity buffer={backend_args.buffer_size} ep replay={estimated_replay_buffer_gib:.2f} GiB "
        f"(available {available_ram_gib}) critic_batch={backend_args.critic_batch_size} ep "
        f"critic_gpu={estimated_critic_gpu_gib:.2f} GiB (free {free_vram_gib})",
        flush=True,
    )
    check_masac_capacity(
        replay_gib=estimated_replay_buffer_gib,
        available_ram_gib=available_ram_gib,
        critic_gpu_gib=estimated_critic_gpu_gib,
        free_vram_gib=free_vram_gib,
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
        # D03: per-building availability mask (get_avail_agent_actions) and the equipment of each action.
        "discrete_action_map": env.adapter.discrete_action_map(),
        "estimated_replay_buffer_gib": estimated_replay_buffer_gib,
        "max_replay_buffer_gib": available_ram_gib,
        "estimated_critic_gpu_gib": estimated_critic_gpu_gib,
        "free_vram_gib_at_start": free_vram_gib,
        "buffer_size_requested": capacity["buffer_size_requested"],
        "capacity_notes": capacity["notes"],
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
        "reward_function": "citylearn.v3.reward.CityLearnV3MADRLRewardFunction",
        "reward_profile": "MASAC",
        "reward_metadata": env.adapter.reward_metadata,
        "job_resume": resume_plan,
    }

    heartbeat_stop = None
    heartbeat_thread = None
    try:
        runner = Runner(env, backend_args)
        learner = getattr(runner, "qmix_pg_learner", None)
        masac_optimizer_specs = [
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
        ]
        if resume_plan.get("active") and learner is not None:
            bundle = find_masac_checkpoint_bundle(artifact_dirs["checkpoints"] / "models")
            if bundle:
                load_masac_checkpoint_bundle(learner, bundle, use_cuda=bool(backend_args.cuda))
                print("[masac] Loaded checkpoint bundle for resume", flush=True)
                hyperparameters["training_state_resume"] = restore_owner_training_state(
                    Path(bundle["rnn"]).parent, masac_optimizer_specs
                )
        if learner is not None and getattr(learner, "alpha_optimizer", None) is not None:
            for group in learner.alpha_optimizer.param_groups:
                group["lr"] = float(args.alpha_lr)
        if learner is not None:
            wrap_save_with_training_state(
                learner,
                "save_model",
                lambda: Path(str(learner.model_dir)),
                lambda: masac_optimizer_specs,
                backend="external/MADRL-MASAC",
            )
        finite_optimizer_guard = install_finite_optimizer_step_guard(
            masac_optimizer_specs,
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
        # The external runner finishes with a matplotlib plt() call and other tail work
        # that can raise inside a headless subprocess. Salvage trained weights and write
        # artifacts no matter what, so a late failure never discards real progress.
        run_error: BaseException | None = None
        try:
            runner.run(args.seed)
        except Exception as exc:  # noqa: BLE001 - salvage on any backend tail failure
            run_error = exc
            traceback.print_exc()
            print(
                f"[masac] runner.run() raised {type(exc).__name__}: {exc}. "
                "Salvaging trained model and writing artifacts so progress is not lost.",
                flush=True,
            )
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        final_train_step = max(1, int(configured_episodes) * max(1, int(backend_args.critic_train_steps)))
        if learner is not None and hasattr(learner, "save_model"):
            try:
                learner.save_model(final_train_step)
                hyperparameters["final_checkpoint_save_step"] = final_train_step
            except Exception as exc:  # noqa: BLE001 - never let checkpointing abort finalization
                print(f"[masac] final save_model failed: {exc}", flush=True)
        if run_error is not None:
            hyperparameters["run_completed_with_salvage"] = True
            hyperparameters["run_error"] = f"{type(run_error).__name__}: {run_error}"
        try:
            env.adapter.write_live_heartbeat(
                stage="masac_backend_finished",
                note="MASAC backend runner finished; writing final artifacts.",
            )
        except Exception:
            pass
        try:
            report = citylearn_v3_training_report(env)
        except Exception as exc:  # noqa: BLE001 - keep fallback report from line above
            print(f"[masac] training report failed, using fallback report: {exc}", flush=True)
        try:
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
        except Exception as exc:  # noqa: BLE001 - guarantee a results.json regardless
            print(
                f"[masac] write_training_artifacts failed ({type(exc).__name__}: {exc}); "
                "writing minimal salvage results.json.",
                flush=True,
            )
            write_minimal_results_json(
                output_dir=output_dir,
                algorithm="MASAC",
                backend="external/MADRL-MASAC",
                args=args,
                hyperparameters=hyperparameters,
                report=report,
                error=exc,
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
