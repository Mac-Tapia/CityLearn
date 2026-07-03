"""Preflight HAPPO E1/E2/E3 for intra-job resume (49→50 ep + KPIs) on Colab/Drive.

Ensures the job directory exists, salvage ``results.json`` is present, and HARL
actor checkpoints are on disk before ``regenerate_happo_kpis.py --execute``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

SCENARIOS = ("E1", "E2", "E3")
DEFAULT_KPIS_EXPORT = Path(__file__).resolve().parents[2] / "outputs" / "_drive_madrl" / "kpis"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_salvage_export(kpis_dir: Path, scenario: str) -> Optional[dict]:
    path = kpis_dir / f"happo_{scenario}_results.json"
    if not path.is_file():
        return None
    return _read_json(path)


def _resolve_job_dir(output_root: Path, scenario: str) -> Path:
    scripts = Path(__file__).resolve().parent
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from citylearn_v3_training_common import resolve_existing_job_run_dir, resolve_job_run_dir

    existing = resolve_existing_job_run_dir(output_root, "happo", scenario, 0)
    if existing is not None:
        return existing
    return resolve_job_run_dir(output_root, "happo", scenario, 0)


def _ensure_results_json(job_dir: Path, salvage: Optional[dict], *, sync: bool) -> Path:
    for rel in ("data/results.json", "results.json"):
        path = job_dir / rel
        if path.is_file():
            return path
    if salvage is None:
        raise FileNotFoundError(f"No results.json under {job_dir} and no KPI export for salvage")
    data_dir = job_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "results.json"
    if sync or not target.is_file():
        target.write_text(json.dumps(salvage, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def preflight_scenario(
    output_root: Path,
    scenario: str,
    *,
    kpis_dir: Path,
    sync_salvage: bool,
) -> Dict[str, object]:
    from citylearn_v3_training_common import (
        discover_job_resume_plan,
        find_harl_actor_checkpoint_dir,
    )

    job_dir = _resolve_job_dir(output_root, scenario)
    salvage = _load_salvage_export(kpis_dir, scenario)
    results_path = _ensure_results_json(job_dir, salvage, sync=sync_salvage)
    payload = _read_json(results_path)

    hp = payload.get("hyperparameters") or {}
    jr = hp.get("job_resume") or {}
    episode_time_steps = int(
        payload.get("episode_time_steps") or hp.get("episode_length") or 8760
    )
    n_rollout = int(hp.get("n_rollout_threads") or 1)
    checkpoints_dir = job_dir / "checkpoints"
    model_dir = find_harl_actor_checkpoint_dir(checkpoints_dir)
    plan = discover_job_resume_plan(
        job_dir,
        algorithm="happo",
        target_episodes=50,
        episode_time_steps=episode_time_steps,
        rollout_threads=n_rollout,
        allow_resume=True,
        output_root=output_root,
    )

    axis = payload.get("project_axis_metrics") or {}
    needs_resume = (
        str(payload.get("status") or "").lower() == "completed_with_salvage"
        or not axis
        or "VecEnvWrapper" in str(payload.get("salvage_reason") or "")
    )

    actor_count = 0
    if model_dir and model_dir.is_dir():
        actor_count = len(list(model_dir.glob("actor_agent*.pt")))

    ready = (
        needs_resume
        and bool(plan.get("active"))
        and model_dir is not None
        and actor_count >= 1
        and int(plan.get("remaining_episodes") or 0) > 0
    )

    return {
        "scenario": scenario,
        "job_dir": str(job_dir),
        "results_json": str(results_path),
        "status": payload.get("status"),
        "salvage_reason": payload.get("salvage_reason"),
        "has_kpis": bool(axis),
        "needs_resume": needs_resume,
        "resume_plan": plan,
        "model_dir": str(model_dir) if model_dir else None,
        "actor_checkpoints": actor_count,
        "ready_to_execute": ready,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, help="Colab OUTPUT_ROOT (madrl_v3_...)")
    parser.add_argument("--scenario", choices=SCENARIOS, help="Single scenario (default: all)")
    parser.add_argument("--kpis-dir", type=Path, default=DEFAULT_KPIS_EXPORT)
    parser.add_argument(
        "--sync-salvage",
        action="store_true",
        help="Copy salvage results.json from kpis export into job data/ if missing",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root).resolve()
    scenarios = (args.scenario,) if args.scenario else SCENARIOS
    rows = [
        preflight_scenario(
            output_root,
            scen,
            kpis_dir=args.kpis_dir.resolve(),
            sync_salvage=args.sync_salvage,
        )
        for scen in scenarios
    ]

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        for row in rows:
            plan = row["resume_plan"]
            print(
                f"[{row['scenario']}] ready={row['ready_to_execute']} "
                f"needs_resume={row['needs_resume']} actors={row['actor_checkpoints']} "
                f"resume={plan.get('completed_episodes')}/{plan.get('target_episodes')} "
                f"remaining={plan.get('remaining_episodes')} model_dir={row['model_dir']}"
            )
            if not row["ready_to_execute"]:
                if row["actor_checkpoints"] == 0:
                    print("    -> FALTA: checkpoints en Drive (monta OUTPUT_ROOT en Colab)")
                elif not plan.get("active"):
                    print(f"    -> plan note: {plan.get('note')}")

    return 0 if all(r["ready_to_execute"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
