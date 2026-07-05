"""Complete salvaged HAPPO jobs (49/50 + empty KPIs) after the VecEnvWrapper fix.

Typical Colab workflow after git sync (celda 2.3):

    python CityLearn/scripts/regenerate_happo_kpis.py \\
        --output-root "$OUTPUT_ROOT" --scenario E1 --execute

Or preview the resume command only:

    python CityLearn/scripts/regenerate_happo_kpis.py \\
        --job-dir "$OUTPUT_ROOT/HAPPO/E1" --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional

SCENARIOS = ("E1", "E2", "E3")
DEFAULT_SCHEMA = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "datasets"
    / "citylearn_iquitos_2023_2025"
    / "schema.json"
)


def _read_results(job_dir: Path) -> Dict[str, object]:
    for rel in ("data/results.json", "results.json"):
        path = job_dir / rel
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No results.json under {job_dir}")


def _happo_needs_kpi_regeneration(payload: Mapping[str, object]) -> bool:
    status = str(payload.get("status") or "").lower()
    axis = payload.get("project_axis_metrics") or {}
    salvage = payload.get("salvage_reason") or (payload.get("hyperparameters") or {}).get("run_error")
    if status == "completed_with_salvage":
        return True
    if not axis and salvage:
        return True
    if "VecEnvWrapper" in str(salvage or ""):
        return True
    return not bool(axis)


def _build_train_command(
    repo: Path,
    job_dir: Path,
    payload: Mapping[str, object],
    *,
    episodes: int = 50,
    rollout_threads: Optional[int] = None,
    schema_path: Optional[Path] = None,
) -> List[str]:
    scripts = repo / "CityLearn" / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from citylearn_v3_training_common import clamp_happo_n_rollout_threads, resolve_job_rollout_threads

    hyper = dict(payload.get("hyperparameters") or {})
    scenario = str(payload.get("scenario") or job_dir.name).upper()
    seed = int(payload.get("seed") or 0)
    episode_time_steps = int(
        payload.get("episode_time_steps")
        or hyper.get("episode_length")
        or 8760
    )
    stored_rollout = int(hyper.get("n_rollout_threads") or 0) or None
    if rollout_threads is not None and int(rollout_threads) > 0:
        n_rollout = clamp_happo_n_rollout_threads(int(rollout_threads))
    else:
        n_rollout = resolve_job_rollout_threads(job_dir, "happo", fallback=2)
    if stored_rollout is not None and stored_rollout > n_rollout:
        print(
            f"[happo-kpi] n_rollout_threads clamped {stored_rollout} -> {n_rollout} "
            "(salvage hyperparameters ignored for throughput)",
            flush=True,
        )
    hidden = int((hyper.get("hidden_sizes") or [512, 512])[0])
    num_mini_batch = int(hyper.get("actor_num_mini_batch") or hyper.get("critic_num_mini_batch") or 0)
    gpu_rollout_ref = 8
    ppo_epoch = int(hyper.get("ppo_epoch") or 5)
    critic_epoch = int(hyper.get("critic_epoch") or 5)
    gpu_runtime = dict(hyper.get("gpu_runtime") or {})
    cuda = bool(hyper.get("cuda") or gpu_runtime.get("cuda_enabled"))
    torch_threads = int(gpu_runtime.get("torch_threads") or hyper.get("torch_threads") or 2)
    gpu_profile = str(gpu_runtime.get("profile") or hyper.get("gpu_profile") or "aws")
    cuda_mem_frac = gpu_runtime.get("cuda_memory_fraction_applied") or gpu_runtime.get(
        "cuda_memory_fraction_requested"
    )
    schema = schema_path or DEFAULT_SCHEMA
    num_env_steps = int(episodes) * int(episode_time_steps) * int(n_rollout)
    live_interval = int(hyper.get("live_progress_interval") or 250)

    cmd = [
        sys.executable,
        "-B",
        str(repo / "CityLearn" / "scripts" / "train_citylearn_v3_happo.py"),
        "--scenario",
        scenario,
        "--seed",
        str(seed),
        "--schema-path",
        str(schema),
        "--output-dir",
        str(job_dir),
        "--episodes",
        str(episodes),
        "--episode-time-steps",
        str(episode_time_steps),
        "--num-env-steps",
        str(num_env_steps),
        "--n-rollout-threads",
        str(n_rollout),
        "--hidden-size",
        str(hidden),
        "--num-mini-batch",
        str(num_mini_batch),
        "--gpu-rollout-ref",
        str(gpu_rollout_ref),
        "--ppo-epoch",
        str(ppo_epoch),
        "--critic-epoch",
        str(critic_epoch),
        "--live-heartbeat-seconds",
        str(int(hyper.get("live_heartbeat_seconds") or 120)),
        "--live-progress-interval",
        str(live_interval),
        "--torch-threads",
        str(torch_threads),
        "--gpu-profile",
        gpu_profile,
        "--artifact-profile",
        str(hyper.get("artifact_profile") or "efficient"),
        "--trace-record-interval",
        str(int(hyper.get("trace_record_interval") or 8760)),
        "--trace-detail",
        str(hyper.get("trace_detail") or "compact"),
        "--log-interval",
        "1",
        "--eval-interval",
        "1",
    ]
    if cuda:
        cmd.append("--cuda")
    if cuda_mem_frac is not None:
        try:
            frac = float(cuda_mem_frac)
            if 0.0 < frac <= 1.0:
                cmd.extend(["--cuda-memory-fraction", str(frac)])
        except (TypeError, ValueError):
            pass
    if hyper.get("use_recurrent_policy"):
        cmd.append("--use-recurrent-policy")
    action_agg = hyper.get("action_aggregation")
    if action_agg:
        cmd.extend(["--action-aggregation", str(action_agg)])
    return cmd


def resolve_happo_job_dir(output_root: Path, scenario: str) -> Path:
    scripts = Path(__file__).resolve().parent
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from citylearn_v3_training_common import resolve_existing_job_run_dir, resolve_job_run_dir

    existing = resolve_existing_job_run_dir(output_root, "happo", scenario.upper(), 0)
    if existing is not None:
        return existing
    return resolve_job_run_dir(output_root, "happo", scenario.upper(), 0)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job-dir", help="HAPPO job directory (e.g. OUTPUT_ROOT/HAPPO/E1)")
    group.add_argument("--output-root", help="MADRL run root; use with --scenario or --all-scenarios")
    parser.add_argument("--scenario", choices=SCENARIOS, help="Required with --output-root unless --all-scenarios")
    parser.add_argument("--all-scenarios", action="store_true", help="Process E1, E2 and E3 under --output-root")
    parser.add_argument("--episodes", type=int, default=50, help="Target episodes (resume uses remaining only)")
    parser.add_argument(
        "--n-rollout-threads",
        type=int,
        default=None,
        help="Override n_rollout_threads (default: from job hyperparameters, else 2 for A100)",
    )
    parser.add_argument("--schema-path", default=None, help="CityLearn schema (default: Iquitos 2023-2025)")
    parser.add_argument("--execute", action="store_true", help="Run train_citylearn_v3_happo.py (default: dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--force", action="store_true", help="Run even if project_axis_metrics already present")
    parser.add_argument(
        "--sync-salvage",
        action="store_true",
        help="Copy salvage results.json from outputs/_drive_madrl/kpis into job data/ if missing",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Do not verify HARL checkpoints before --execute (not recommended)",
    )
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    dry_run = args.dry_run or not args.execute
    schema_path = Path(args.schema_path).resolve() if args.schema_path else DEFAULT_SCHEMA
    prepare_script = repo / "CityLearn" / "scripts" / "prepare_happo_colab_resume.py"

    if args.job_dir:
        jobs = [Path(args.job_dir).resolve()]
        output_root = jobs[0].parent.parent
    else:
        if not args.all_scenarios and not args.scenario:
            parser.error("Provide --scenario or --all-scenarios with --output-root")
        output_root = Path(args.output_root).resolve()
        scenarios = SCENARIOS if args.all_scenarios else (args.scenario.upper(),)
        jobs = [resolve_happo_job_dir(output_root, scen) for scen in scenarios]

    if args.execute and not args.skip_preflight and prepare_script.is_file():
        pre_cmd = [
            sys.executable,
            "-B",
            str(prepare_script),
            "--output-root",
            str(output_root),
            "--episodes",
            str(args.episodes),
        ]
        if args.sync_salvage:
            pre_cmd.append("--sync-salvage")
        if args.n_rollout_threads is not None:
            pre_cmd.extend(["--happo-rollout-threads", str(args.n_rollout_threads)])
        if args.scenario and not args.all_scenarios:
            pre_cmd.extend(["--scenario", args.scenario.upper()])
        print("[happo-kpi] preflight:", " ".join(pre_cmd))
        pre = subprocess.run(pre_cmd, cwd=str(repo))
        if pre.returncode != 0:
            print(
                "[happo-kpi] preflight failed: checkpoints or resume plan not ready. "
                "Run on Colab with OUTPUT_ROOT on Drive, or use --skip-preflight.",
                file=sys.stderr,
            )
            return pre.returncode

    exit_code = 0
    for job_dir in jobs:
        payload = _read_results(job_dir)
        if not args.force and not _happo_needs_kpi_regeneration(payload):
            print(f"[skip] {job_dir.name}: KPIs already present and no salvage flag")
            continue
        cmd = _build_train_command(
            repo,
            job_dir,
            payload,
            episodes=args.episodes,
            rollout_threads=args.n_rollout_threads,
            schema_path=schema_path,
        )
        print(f"[happo-kpi] {job_dir}")
        print("  " + " ".join(cmd))
        if dry_run:
            continue
        result = subprocess.run(cmd, cwd=str(repo))
        if result.returncode != 0:
            print(f"[happo-kpi] FAILED ({result.returncode}): {job_dir}", file=sys.stderr)
            exit_code = result.returncode

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
