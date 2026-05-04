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
    ensure_project_paths,
    ensure_artifact_layout,
    resolve_output_dir,
    write_training_artifacts,
    write_training_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_citylearn_args(parser)
    parser.add_argument("--episodes", default=1, type=int)
    parser.add_argument("--action-bins", default=3, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--buffer-length", default=256, type=int)
    parser.add_argument("--steps-per-update", default=1, type=int)
    parser.add_argument("--num-updates", default=1, type=int)
    parser.add_argument("--hidden-size", default=128, type=int)
    parser.add_argument("--attend-heads", default=4, type=int)
    parser.add_argument("--pi-lr", default=1e-3, type=float)
    parser.add_argument("--q-lr", default=1e-3, type=float)
    parser.add_argument("--tau", default=1e-3, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward-scale", default=100.0, type=float)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_project_paths()
    add_external_path("MAAC")
    # The MAAC repo has a top-level ``utils`` package name. Remove any earlier
    # module with that name so imports resolve to external/MAAC/utils.
    import sys

    sys.modules.pop("utils", None)
    from algorithms.attention_sac import AttentionSAC
    from utils.buffer import ReplayBuffer

    output_dir = resolve_output_dir(args.output_dir, "maac", args.scenario, args.seed)
    artifact_dirs = ensure_artifact_layout(output_dir)
    env = CityLearnMAACVecEnv(
        schema_path=args.schema_path,
        scenario=args.scenario,
        seed=args.seed,
        episode_time_steps=args.episode_time_steps,
        action_bins=args.action_bins,
        algorithm="MAAC",
        live_progress_path=str(output_dir / "live_progress.json"),
        live_progress_interval=100,
    )
    logger = NoOpLogger()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_gpu = bool(args.cuda and torch.cuda.is_available())

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
        "cuda": use_gpu,
        "critic": "multi-agent attention critic",
        "reward_function": "CityLearnV3MADRLRewardFunction",
        "reward_profile": "MAAC",
        "reward_metadata": env.adapter.reward_metadata,
    }
    try:
        for episode in range(args.episodes):
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
        env.close()

    write_training_summary(
        output_dir,
        {
            "algorithm": "MAAC",
            "backend": "external/MAAC",
            "scenario": args.scenario,
            "seed": args.seed,
            "episode_time_steps": args.episode_time_steps,
            "episodes": args.episodes,
            "action_bins": args.action_bins,
            "n_discrete_actions": env.adapter.n_discrete_actions,
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
    raise SystemExit(main())
