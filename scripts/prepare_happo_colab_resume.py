"""Preflight HAPPO 49→50 resume on Colab/Drive before ``regenerate_happo_kpis.py --execute``.

Validates checkpoints, syncs salvage ``results.json`` when requested, and prints the
same resume plan used by ``discover_job_resume_plan`` / cell 2.1b.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

SCENARIOS = ("E1", "E2", "E3")
DEFAULT_SCHEMA = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "datasets"
    / "citylearn_iquitos_2023_2025"
    / "schema.json"
)


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parent


def _import_common():
    scripts = _scripts_dir()
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import citylearn_v3_training_common as common  # noqa: WPS433

    return common


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sync_salvage_results(
    job_dir: Path,
    *,
    kpis_dir: Path,
    scenario: str,
) -> bool:
    """Copy ``outputs/_drive_madrl/kpis/happo_{scenario}_results.json`` into job data/."""
    src = kpis_dir / f"happo_{scenario.upper()}_results.json"
    if not src.is_file():
        return False
    data_dir = job_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dst = data_dir / "results.json"
    shutil.copy2(src, dst)
    mirror = job_dir / "results.json"
    try:
        shutil.copy2(src, mirror)
    except OSError:
        pass
    return True


def _resolve_jobs(common, output_root: Path, scenarios: Sequence[str], seed: int) -> List[Path]:
    jobs: List[Path] = []
    for scen in scenarios:
        existing = common.resolve_existing_job_run_dir(output_root, "happo", scen.upper(), seed)
        if existing is not None:
            jobs.append(existing)
        else:
            jobs.append(common.resolve_job_run_dir(output_root, "happo", scen.upper(), seed))
    return jobs


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, help="MADRL run root (OUTPUT_ROOT)")
    parser.add_argument("--scenario", choices=SCENARIOS, help="Single scenario (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=50, help="Target episodes for resume plan")
    parser.add_argument(
        "--episode-time-steps",
        type=int,
        default=8760,
        help="Episode length for resume accounting",
    )
    parser.add_argument(
        "--happo-rollout-threads",
        type=int,
        default=None,
        help="Override rollout threads for resume preview (default: from artifacts)",
    )
    parser.add_argument(
        "--sync-salvage",
        action="store_true",
        help="Copy salvage results.json from outputs/_drive_madrl/kpis into each job",
    )
    parser.add_argument(
        "--repo",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root (for default kpis dir)",
    )
    args = parser.parse_args(argv)

    common = _import_common()
    output_root = Path(args.output_root).resolve()
    repo = Path(args.repo).resolve()
    kpis_dir = repo / "outputs" / "_drive_madrl" / "kpis"
    scenarios = (args.scenario.upper(),) if args.scenario else SCENARIOS

    if not output_root.is_dir():
        print(f"[prepare] OUTPUT_ROOT missing: {output_root}", file=sys.stderr)
        return 2

    jobs = _resolve_jobs(common, output_root, scenarios, int(args.seed))
    failures = 0

    for job_dir, scen in zip(jobs, scenarios):
        print(f"\n[prepare] {scen} -> {job_dir}")
        if args.sync_salvage:
            if _sync_salvage_results(job_dir, kpis_dir=kpis_dir, scenario=scen):
                print(f"  synced salvage results from {kpis_dir / f'happo_{scen}_results.json'}")
            else:
                print(f"  no salvage KPI file to sync for {scen}")

        if not common.find_harl_actor_checkpoint_dir(job_dir / common.CHECKPOINT_DIR_NAME):
            print("  FAIL: no HARL actor checkpoints (actor_agent0.pt)", file=sys.stderr)
            failures += 1
            continue

        roll = common.resolve_job_rollout_threads(
            job_dir,
            "happo",
            fallback=args.happo_rollout_threads,
        )
        plan = common.discover_job_resume_plan(
            job_dir,
            algorithm="happo",
            target_episodes=int(args.episodes),
            episode_time_steps=int(args.episode_time_steps),
            rollout_threads=int(roll),
            output_root=output_root,
        )
        print(
            f"  resume: active={plan.get('active')} "
            f"completed={plan.get('completed_episodes')}/{plan.get('target_episodes')} "
            f"remaining={plan.get('remaining_episodes')} "
            f"rollout_threads={roll} note={plan.get('note')}"
        )
        if plan.get("model_dir"):
            print(f"  model_dir: {plan.get('model_dir')}")
        if not plan.get("active"):
            print("  WARN: resume plan inactive — job may be complete or missing weights", file=sys.stderr)
            failures += 1
        elif int(plan.get("completed_episodes") or 0) < int(args.episodes) - 1:
            print(
                "  WARN: completed_episodes < target-1; not a 49→50 tail resume",
                file=sys.stderr,
            )

        results_path = job_dir / "data" / "results.json"
        if results_path.is_file():
            payload = _read_json(results_path)
            hyper = dict(payload.get("hyperparameters") or {})
            print(
                f"  results: status={payload.get('status')!r} "
                f"cuda={hyper.get('cuda')} n_rollout_threads={hyper.get('n_rollout_threads')}"
            )

    if failures:
        print(f"\n[prepare] FAILED ({failures} scenario(s) not ready)", file=sys.stderr)
        return 1

    if not DEFAULT_SCHEMA.is_file():
        print(f"[prepare] WARN: schema not found at {DEFAULT_SCHEMA}", file=sys.stderr)
    print("\n[prepare] OK — safe to run regenerate_happo_kpis.py --execute")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
