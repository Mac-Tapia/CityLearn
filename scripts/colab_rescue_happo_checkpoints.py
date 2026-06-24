"""Rescue partial HAPPO checkpoints from an interrupted Colab MADRL run.

Use before deleting a failed/incomplete output session (e.g. MASAC OOM) to keep
HAPPO progress for archival or intra-job resume in a new OUTPUT_ROOT.

Colab example (run in a notebook cell after stopping training):

    !python CityLearn/scripts/colab_rescue_happo_checkpoints.py rescue \\
        --source-run /content/drive/MyDrive/MADRLCitytleranflexresdr/outputs/madrl_v3_20260624_175429

Optional inject into a fresh run before 7.2 (resume HAPPO from ep ~1 progress):

    !python CityLearn/scripts/colab_rescue_happo_checkpoints.py inject \\
        --archive outputs/rescued_happo_20260624_175429 \\
        --target-run /content/drive/MyDrive/MADRLCitytleranflexresdr/outputs/madrl_v3_NEW
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

SCENARIOS = ("E1", "E2", "E3")
SEED = 0
HAPPO_REL_FILES = (
    "live_progress.json",
    "data/job_resume_manifest.json",
    "data/checkpoint_manifest.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return root / path


def happo_job_dir(run_root: Path, scenario: str) -> Path:
    return run_root / "happo" / f"{scenario}_seed_{SEED}"


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _copy_file(src: Path, dst: Path, *, dry_run: bool) -> None:
    if not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"  [dry-run] file {src} -> {dst}")
        return
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path, *, dry_run: bool) -> None:
    if not src.is_dir():
        return
    if dry_run:
        print(f"  [dry-run] tree {src} -> {dst}")
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _read_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _progress_summary(job_dir: Path) -> Dict[str, object]:
    live = _read_json(job_dir / "live_progress.json") or {}
    episode = int(live.get("episode") or 0)
    ep_step = int(live.get("episode_step") or 0)
    ep_steps = int(live.get("episode_time_steps") or 8760)
    global_step = int(live.get("global_step") or 0)
    return {
        "scenario": live.get("scenario"),
        "episode_display": episode + 1,
        "episode_step": ep_step,
        "episode_time_steps": ep_steps,
        "global_step": global_step,
        "fps": live.get("fps"),
        "has_checkpoints": (job_dir / "checkpoints").is_dir(),
        "checkpoint_bytes": _dir_size_bytes(job_dir / "checkpoints"),
    }


def rescue_scenario(
    *,
    source_run: Path,
    archive_root: Path,
    scenario: str,
    dry_run: bool,
) -> Dict[str, object]:
    src_job = happo_job_dir(source_run, scenario)
    dst_job = archive_root / "happo" / f"{scenario}_seed_{SEED}"
    record: Dict[str, object] = {
        "scenario": scenario,
        "source": str(src_job),
        "destination": str(dst_job),
        "copied": False,
        "reason": "",
    }
    if not src_job.is_dir():
        record["reason"] = "source_missing"
        return record

    summary = _progress_summary(src_job)
    record["progress"] = summary
    if not summary.get("has_checkpoints") and not (src_job / "live_progress.json").is_file():
        record["reason"] = "no_checkpoints_or_progress"
        return record

    print(f"[rescue] HAPPO/{scenario} -> {dst_job}")
    _copy_tree(src_job / "checkpoints", dst_job / "checkpoints", dry_run=dry_run)
    for rel in HAPPO_REL_FILES:
        _copy_file(src_job / rel, dst_job / rel, dry_run=dry_run)

    record["copied"] = True
    record["reason"] = "ok"
    return record


def cmd_rescue(args: argparse.Namespace, root: Path) -> int:
    source_run = resolve_path(root, args.source_run)
    if not source_run.is_dir():
        raise FileNotFoundError(f"Source run not found: {source_run}")

    if args.dest:
        archive_root = resolve_path(root, args.dest)
    else:
        stamp = source_run.name.replace("madrl_v3_", "happo_")
        archive_root = resolve_path(root, f"outputs/rescued_{stamp}")

    scenarios = list(args.scenarios or SCENARIOS)
    print(f"[rescue] source={source_run}")
    print(f"[rescue] archive={archive_root}")
    if not args.dry_run:
        archive_root.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    for scenario in scenarios:
        records.append(
            rescue_scenario(
                source_run=source_run,
                archive_root=archive_root,
                scenario=scenario,
                dry_run=args.dry_run,
            )
        )

    status_path = source_run / "official_full_status.json"
    manifest = {
        "rescued_at": utc_now(),
        "mode": "rescue",
        "source_run": str(source_run),
        "archive_root": str(archive_root),
        "dry_run": bool(args.dry_run),
        "scenarios": records,
        "note": (
            "Partial HAPPO only. Not valid for official 12/12 completion. "
            "Use inject to resume intra-job in a new OUTPUT_ROOT (launcher --skip-completed "
            "skips only jobs with results.json)."
        ),
    }
    if status_path.is_file():
        manifest["source_status_snapshot"] = str(status_path)

    manifest_path = archive_root / "rescue_manifest.json"
    if args.dry_run:
        print(f"[dry-run] would write {manifest_path}")
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[rescue] manifest: {manifest_path}")

    copied = sum(1 for r in records if r.get("copied"))
    print(f"[rescue] done: {copied}/{len(scenarios)} scenarios copied")
    return 0 if copied > 0 else 1


def inject_scenario(
    *,
    archive_root: Path,
    target_run: Path,
    scenario: str,
    dry_run: bool,
) -> Dict[str, object]:
    src_job = archive_root / "happo" / f"{scenario}_seed_{SEED}"
    dst_job = happo_job_dir(target_run, scenario)
    record: Dict[str, object] = {
        "scenario": scenario,
        "source": str(src_job),
        "destination": str(dst_job),
        "injected": False,
        "reason": "",
    }
    if not src_job.is_dir():
        record["reason"] = "archive_missing"
        return record

    print(f"[inject] HAPPO/{scenario} {src_job} -> {dst_job}")
    _copy_tree(src_job / "checkpoints", dst_job / "checkpoints", dry_run=dry_run)
    for rel in HAPPO_REL_FILES:
        _copy_file(src_job / rel, dst_job / rel, dry_run=dry_run)

    record["injected"] = True
    record["progress"] = _progress_summary(dst_job if not dry_run else src_job)
    record["reason"] = "ok"
    return record


def cmd_inject(args: argparse.Namespace, root: Path) -> int:
    archive_root = resolve_path(root, args.archive)
    target_run = resolve_path(root, args.target_run)
    if not archive_root.is_dir():
        raise FileNotFoundError(f"Archive not found: {archive_root}")

    scenarios = list(args.scenarios or SCENARIOS)
    print(f"[inject] archive={archive_root}")
    print(f"[inject] target={target_run}")
    if not args.dry_run:
        target_run.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    for scenario in scenarios:
        records.append(
            inject_scenario(
                archive_root=archive_root,
                target_run=target_run,
                scenario=scenario,
                dry_run=args.dry_run,
            )
        )

    manifest = {
        "injected_at": utc_now(),
        "mode": "inject",
        "archive_root": str(archive_root),
        "target_run": str(target_run),
        "dry_run": bool(args.dry_run),
        "scenarios": records,
        "next_step": (
            "Launch colab_a100_official_launcher with --skip-completed on target_run. "
            "HAPPO jobs resume from checkpoints/live_progress; MASAC restarts fresh."
        ),
    }
    manifest_path = target_run / "happo_rescue_inject_manifest.json"
    if args.dry_run:
        print(f"[dry-run] would write {manifest_path}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[inject] manifest: {manifest_path}")

    injected = sum(1 for r in records if r.get("injected"))
    print(f"[inject] done: {injected}/{len(scenarios)} scenarios injected")
    return 0 if injected > 0 else 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    rescue = sub.add_parser("rescue", help="Copy HAPPO checkpoints from a failed run to an archive")
    rescue.add_argument(
        "--source-run",
        required=True,
        help="OUTPUT_ROOT of interrupted run (e.g. outputs/madrl_v3_20260624_175429)",
    )
    rescue.add_argument(
        "--dest",
        default="",
        help="Archive directory (default: outputs/rescued_happo_<run_name>)",
    )
    rescue.add_argument("--scenario", action="append", dest="scenarios", choices=SCENARIOS)
    rescue.add_argument("--dry-run", action="store_true")

    inject = sub.add_parser("inject", help="Inject archived HAPPO into a new OUTPUT_ROOT before relaunch")
    inject.add_argument("--archive", required=True, help="Rescue archive from rescue command")
    inject.add_argument("--target-run", required=True, help="New OUTPUT_ROOT for relaunch")
    inject.add_argument("--scenario", action="append", dest="scenarios", choices=SCENARIOS)
    inject.add_argument("--dry-run", action="store_true")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = project_root()
    if args.cmd == "rescue":
        return cmd_rescue(args, root)
    if args.cmd == "inject":
        return cmd_inject(args, root)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
