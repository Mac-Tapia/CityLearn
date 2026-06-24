"""Train MAAC on CityLearn v3 using the original MAAC backend."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.autograd import Variable

from citylearn_v3_training_common import (
    CityLearnMAACVecEnv,
    NoOpLogger,
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
    start_live_progress_heartbeat,
    stop_live_progress_heartbeat,
    write_training_artifacts,
    write_training_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_citylearn_args(parser)
    parser.add_argument("--episodes", default=1, type=int)
    parser.add_argument("--action-bins", default=3, type=int)
    parser.add_argument(
        "--discrete-action-mode",
        default="axis",
        choices=("axis", "shared", "cartesian"),
        help="Discrete CityLearn action table. axis is linear in action dimensions; cartesian is exponential.",
    )
    parser.add_argument(
        "--max-discrete-actions",
        default=512,
        type=int,
        help="Abort before MAAC startup if a discrete agent action space is larger than this.",
    )
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--buffer-length", default=50000, type=int)
    parser.add_argument("--steps-per-update", default=1, type=int)
    parser.add_argument("--num-updates", default=1, type=int)
    parser.add_argument("--hidden-size", default=256, type=int)
    parser.add_argument("--attend-heads", default=4, type=int)
    parser.add_argument("--pi-lr", default=3e-4, type=float)
    parser.add_argument("--q-lr", default=1e-3, type=float)
    parser.add_argument("--tau", default=1e-3, type=float)
    parser.add_argument("--gamma", default=0.9999, type=float,
                        help="Discount factor. Use 0.9999 for year-long (8760-step) episodes.")
    parser.add_argument("--reward-scale", default=10.0, type=float)
    parser.add_argument("--torch-threads", default=1, type=int)
    parser.add_argument("--live-heartbeat-seconds", default=30, type=int)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_project_paths()
    add_external_path("MAAC")
    # The MAAC repo has top-level ``utils`` and ``algorithms`` names. Remove
    # earlier modules with those names so imports resolve to external/MAAC.
    import sys

    sys.modules.pop("utils", None)
    sys.modules.pop("algorithms", None)
    sys.path[:] = [
        entry for entry in sys.path
        if not entry.replace("\\", "/").rstrip("/").endswith("/uc3m")
    ]
    from algorithms.attention_sac import AttentionSAC
    from utils.buffer import ReplayBuffer

    output_dir = resolve_output_dir(args.output_dir, "maac", args.scenario, args.seed)
    artifact_dirs = ensure_artifact_layout(output_dir)

    resume_plan = discover_job_resume_plan(
        output_dir,
        algorithm="maac",
        target_episodes=int(args.episodes),
        episode_time_steps=args.episode_time_steps,
        allow_resume=bool(getattr(args, "resume", True)),
    )
    if resume_plan.get("active"):
        args.episodes = int(resume_plan["remaining_episodes"])
        print(
            f"[maac] RESUME ep {resume_plan['completed_episodes']}/{resume_plan['target_episodes']} "
            f"-> {args.episodes} remaining",
            flush=True,
        )
    write_job_resume_manifest(output_dir, resume_plan)

    env = CityLearnMAACVecEnv(
        schema_path=args.schema_path,
        scenario=args.scenario,
        seed=args.seed,
        episode_time_steps=args.episode_time_steps,
        action_bins=args.action_bins,
        discrete_action_mode=args.discrete_action_mode,
        algorithm="MAAC",
        live_progress_path=str(output_dir / "live_progress.json"),
        live_progress_interval=args.live_progress_interval,
        trace_record_interval=args.trace_record_interval,
        trace_detail=args.trace_detail,
        normalize_observations=args.normalize_observations,
    )
    logger = NoOpLogger()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    gpu_runtime = configure_torch_runtime(
        torch,
        use_cuda=args.cuda,
        torch_threads=args.torch_threads,
        gpu_profile=args.gpu_profile,
        cuda_memory_fraction=args.cuda_memory_fraction,
    )
    use_gpu = bool(gpu_runtime["cuda_enabled"])

    if env.adapter.n_discrete_actions > int(args.max_discrete_actions):
        raise MemoryError(
            "MAAC discrete action space is too large: "
            f"{env.adapter.n_discrete_actions} > {args.max_discrete_actions}. "
            "Use --discrete-action-mode axis for CityLearn multi-actuator actions, "
            "or reduce action bins."
        )

    model = AttentionSAC.init_from_env(
        env,
        tau=args.tau,
        pi_lr=args.pi_lr,
        q_lr=args.q_lr,
        gamma=args.gamma,
        pol_hidden_dim=args.hidden_size,
        critic_hidden_dim=args.hidden_size,
        attend_heads=args.attend_heads,
        reward_scale=args.reward_scale,
    )
    if resume_plan.get("active") and resume_plan.get("maac_checkpoint"):
        model = AttentionSAC.init_from_save(str(resume_plan["maac_checkpoint"]), load_critic=True)
        print(f"[maac] Loaded {resume_plan['maac_checkpoint']}", flush=True)
    finite_optimizer_guard = install_finite_optimizer_step_guard(
        [
            {
                "owner": "maac_critic",
                "module": getattr(model, "critic", None),
                "optimizer": getattr(model, "critic_optimizer", None),
            }
        ]
        + [
            {
                "owner": f"maac_agent{agent_index}_policy",
                "module": getattr(agent, "policy", None),
                "optimizer": getattr(agent, "policy_optimizer", None),
            }
            for agent_index, agent in enumerate(getattr(model, "agents", []) or [])
        ],
        output_dir / "data" / "maac_finite_gradient_guard.jsonl",
    )
    maac_start_episode = int(resume_plan.get("maac_start_episode") or 0) if resume_plan.get("active") else 0
    replay_buffer = ReplayBuffer(
        args.buffer_length,
        model.nagents,
        [space.shape[0] for space in env.observation_space],
        [space.n for space in env.action_space],
    )

    t = 0
    report = citylearn_v3_training_report(None)
    artifacts = {}
    hyperparameters = {
        "episodes": args.episodes,
        "episode_length": args.episode_time_steps,
        "action_bins": args.action_bins,
        "discrete_action_mode": args.discrete_action_mode,
        "discrete_action_metadata": env.adapter.discrete_action_metadata,
        "max_discrete_actions": args.max_discrete_actions,
        "n_discrete_actions": env.adapter.n_discrete_actions,
        "batch_size": args.batch_size,
        "buffer_length": args.buffer_length,
        "steps_per_update": args.steps_per_update,
        "num_updates": args.num_updates,
        "hidden_size": args.hidden_size,
        "attend_heads": args.attend_heads,
        "pi_lr": args.pi_lr,
        "q_lr": args.q_lr,
        "tau": args.tau,
        "gamma": args.gamma,
        "reward_scale": args.reward_scale,
        "torch_threads": args.torch_threads,
        "live_progress_interval": args.live_progress_interval,
        "live_heartbeat_seconds": args.live_heartbeat_seconds,
        "cuda": use_gpu,
        "gpu_runtime": gpu_runtime,
        "algorithm_family": "MADRL",
        "critic": "multi-agent attention critic",
        "reward_function": "CityLearnV3MADRLRewardFunction",
        "reward_profile": "MAAC",
        "reward_metadata": env.adapter.reward_metadata,
        "finite_optimizer_step_guard": finite_optimizer_guard,
        "job_resume": resume_plan,
        "maac_start_episode": maac_start_episode,
    }
    heartbeat_stop = None
    heartbeat_thread = None
    try:
        heartbeat_stop, heartbeat_thread = start_live_progress_heartbeat(
            env.adapter,
            args.live_heartbeat_seconds,
            active_stage="maac_backend_training",
            initial_stage="maac_backend_starting",
            note="MAAC backend is collecting transitions or updating attention critics and policies.",
        )
        for episode in range(maac_start_episode, maac_start_episode + args.episodes):
            obs = env.reset()
            model.prep_rollouts(device="gpu" if use_gpu else "cpu")

            for _step in range(args.episode_time_steps):
                torch_obs = [
                    Variable(torch.Tensor(np.vstack(obs[:, agent_i])), requires_grad=False)
                    for agent_i in range(model.nagents)
                ]
                if use_gpu:
                    torch_obs = [item.cuda() for item in torch_obs]

                torch_agent_actions = model.step(torch_obs, explore=True)
                agent_actions = [action.data.cpu().numpy() for action in torch_agent_actions]
                env_actions = [[action[i] for action in agent_actions] for i in range(env.num_envs)]
                next_obs, rewards, dones, infos = env.step(env_actions)
                replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
                obs = next_obs
                t += env.num_envs

                if len(replay_buffer) >= args.batch_size and (t % args.steps_per_update) < env.num_envs:
                    model.prep_training(device="gpu" if use_gpu else "cpu")
                    for _ in range(args.num_updates):
                        sample = replay_buffer.sample(args.batch_size, to_gpu=use_gpu, norm_rews=False)
                        model.update_critic(sample, logger=logger)
                        model.update_policies(sample, logger=logger)
                        model.update_all_targets()
                    model.prep_rollouts(device="gpu" if use_gpu else "cpu")

                if np.all(dones):
                    break

            model.save(artifact_dirs["checkpoints"] / f"checkpoint_episode_{episode + 1}.pt")

        model.prep_rollouts(device="cpu")
        try:
            env.adapter.write_live_heartbeat(
                stage="maac_backend_finished",
                note="MAAC backend runner finished; writing final artifacts.",
            )
        except Exception:
            pass
        model.save(artifact_dirs["checkpoints"] / "model.pt")
        report = citylearn_v3_training_report(env)
        artifacts = write_training_artifacts(
            output_dir=output_dir,
            algorithm="MAAC",
            backend="external/MAAC",
            args=args,
            report=report,
            candidate=env,
            hyperparameters=hyperparameters,
            extra={
                "episodes": args.episodes,
                "environment_steps": t,
            },
        )
    finally:
        stop_live_progress_heartbeat(heartbeat_stop, heartbeat_thread)
        env.close()

    write_training_summary(
        output_dir,
        {
            "algorithm": "MAAC",
            "algorithm_family": "MADRL",
            "backend": "external/MAAC",
            "scenario": args.scenario,
            "seed": args.seed,
            "episode_time_steps": args.episode_time_steps,
            "episodes": args.episodes,
            "action_bins": args.action_bins,
            "discrete_action_mode": args.discrete_action_mode,
            "discrete_action_metadata": env.adapter.discrete_action_metadata,
            "n_discrete_actions": env.adapter.n_discrete_actions,
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
