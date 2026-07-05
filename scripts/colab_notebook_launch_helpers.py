"""Notebook cell 7.2 bootstrap — run training without cells 1.5/2.1/6.1/7.0/7.1.

After ``git pull`` on Colab: interrupt stuck 7.2, re-run **only** cell 7.2.
"""

from __future__ import annotations

import importlib.util
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple


def _in_colab() -> bool:
    return importlib.util.find_spec("google.colab") is not None


def _usable_vcpus() -> int:
    try:
        affinity = getattr(os, "sched_getaffinity", None)
        if affinity is not None:
            return len(affinity(0))
    except Exception:
        pass
    return int(os.cpu_count() or 12)


def _detect_vram_gib() -> float:
    try:
        mib = int(
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .splitlines()[0]
        )
        return mib / 1024.0
    except Exception:
        return 80.0


def _alloc_phase1(vcpus: int, max_rollout: int = 16) -> Tuple[int, int]:
    best = (1, 1)
    for torch_t in (1, 2):
        for rollout in range(1, max_rollout + 1):
            if 3 * (torch_t + rollout) + 3 * torch_t <= vcpus:
                best = (torch_t, rollout)
    return best


def _launcher_flags(launcher_path: Path) -> set:
    try:
        src = launcher_path.read_text(encoding="utf-8")
        return set(re.findall(r'add_argument\(["\'](-{1,2}[\w-]+)["\']', src))
    except Exception:
        return set()


def colab_a100_training_config(
    repo: Path,
    *,
    output_root: Path,
    gdrive_root: Optional[Path] = None,
    target_episodes: int = 50,
    episode_steps: int = 8760,
    seed: int = 0,
) -> Dict[str, object]:
    """Mirror notebook cell 6.1 constants for the official launcher."""
    repo = Path(repo)
    vcpus = _usable_vcpus()
    vram = _detect_vram_gib()
    p1_torch, happo_rollout = _alloc_phase1(vcpus)
    p2_torch = max(1, vcpus // 6)
    happo_gpu_ref = 8

    try:
        import psutil

        ram_gib = psutil.virtual_memory().total / 1024**3
    except Exception:
        ram_gib = 177.0

    happo_frac = 0.14 if vram <= 85 else 0.15
    masac_frac = 0.16 if vram <= 85 else 0.22
    phase1_vram = 3 * happo_frac * vram + 3 * masac_frac * vram
    extra_p2 = max(0, int((0.92 * vram - phase1_vram) // max(happo_frac * vram, 1e-6)))
    ram_budget = max(6, int((ram_gib * 0.90) // 18))
    vram_budget = 6 + extra_p2
    cap_auto = min(12, ram_budget, vram_budget)
    max_concurrent = cap_auto if (vcpus > 12 and cap_auto > 6) else 0

    schema = repo / "CityLearn/data/datasets/citylearn_iquitos_2023_2025/schema.json"
    launcher = repo / "CityLearn/scripts/colab_a100_official_launcher.py"
    monitor = repo / "CityLearn/scripts/colab_a100_live_monitor.py"
    guard = repo / "CityLearn/scripts/colab_protocol_guard.py"

    return {
        "REPO": str(repo),
        "GDRIVE_ROOT": str(gdrive_root) if gdrive_root else "",
        "PYTHON": sys.executable,
        "PROJECT_PYTHON": sys.executable,
        "SCHEMA_PATH": str(schema),
        "LAUNCHER": str(launcher),
        "MONITOR": str(monitor),
        "PROTOCOL_GUARD": str(guard),
        "OUTPUT_ROOT": str(output_root),
        "N_EPISODES": int(target_episodes),
        "EPISODES": int(target_episodes),
        "EPISODE_STEPS": int(episode_steps),
        "NUM_ENV_STEPS": int(target_episodes) * int(episode_steps),
        "SEED": int(seed),
        "USABLE_VCPUS": vcpus,
        "TWO_PHASE_P1_TORCH": p1_torch,
        "HAPPO_ROLLOUT_THREADS": happo_rollout,
        "HAPPO_GPU_ROLLOUT_REF": happo_gpu_ref,
        "TWO_PHASE_P2_TORCH": p2_torch,
        "TORCH_THREADS": p2_torch,
        "MAX_CONCURRENT_JOBS": max_concurrent,
        "LIVE_PROGRESS_INT": 300,
        "LIVE_HEARTBEAT_SEC": 120,
        "EST_MIN_PER_EPISODE": 12,
        "EST_MIN_BY_ALGO": {"happo": 11, "masac": 15, "matd3": 12, "maac": 8},
        "SIX_JOB_CUDA_FRAC": happo_frac,
        "SIX_JOB_MASAC_CUDA_FRAC": masac_frac,
        "SIX_JOB_MASAC_BUF": 4,
        "SIX_JOB_MASAC_GIB": 8.0,
        "SIX_JOB_MASAC_BATCH": 1,
        "SIX_JOB_MATD3_BUF": 4096,
        "SIX_JOB_MATD3_BATCH": 256,
        "SIX_JOB_MATD3_HIDDEN": 256,
        "SIX_JOB_MAAC_BUF": 450_000,
        "SIX_JOB_MAAC_BATCH": 768,
        "SIX_JOB_MAAC_HIDDEN": 768,
        "SIX_JOB_MAAC_UPDATES": 12,
        "MONITOR_INTERVAL": 120,
        "AUTO_DISCONNECT_COLAB": False,
        "AUTO_RUN_POST_TRAINING": True,
        "ARTIFACT_PROFILE": "efficient",
        "TRACE_INTERVAL": 8760,
        "TRACE_DETAIL": "compact",
        "GPU_PROFILE": "aws",
        "CUDA_MEMORY_FRACTION": 0.92,
        "EXECUTION_MODE": "two_phase_happo_masac",
        "DYNAMIC_BACKFILL": True,
        "SCENARIOS": ["E1", "E2", "E3"],
        "ALGORITHMS": ["happo", "masac", "matd3", "maac"],
        "QUICK_TEST": False,
        "_created_new_run": False,
    }


def build_official_launcher_argv(config: Mapping[str, object]) -> List[str]:
    """Same argv as notebook ``launcher_base_args()`` (cell 7.0)."""
    launcher = Path(str(config["LAUNCHER"]))
    flags = _launcher_flags(launcher)

    def _opt(flag: str, *values: object) -> List[str]:
        return [flag, *[str(v) for v in values]] if flag in flags else []

    base: List[str] = [
        str(config["PYTHON"]),
        "-B",
        str(config["LAUNCHER"]),
        "--scenario",
        "ALL",
        "--seed",
        str(config["SEED"]),
        "--episode-time-steps",
        str(config["EPISODE_STEPS"]),
        "--episodes",
        str(config["EPISODES"]),
        "--schema-path",
        str(config["SCHEMA_PATH"]),
        "--output-root",
        str(config["OUTPUT_ROOT"]),
        "--torch-threads",
        str(config["TORCH_THREADS"]),
        "--live-progress-interval",
        str(config["LIVE_PROGRESS_INT"]),
        "--live-heartbeat-seconds",
        str(config["LIVE_HEARTBEAT_SEC"]),
        "--artifact-profile",
        str(config["ARTIFACT_PROFILE"]),
        "--trace-record-interval",
        str(config["TRACE_INTERVAL"]),
        "--trace-detail",
        str(config["TRACE_DETAIL"]),
        "--gpu-profile",
        str(config["GPU_PROFILE"]),
        "--cuda-memory-fraction",
        str(config["CUDA_MEMORY_FRACTION"]),
        "--require-a100",
        "--smoke-imports",
        "--oom-retry",
        "--no-live-monitor",
        "--happo-hidden-size",
        "512",
        "--happo-n-rollout-threads",
        str(config["HAPPO_ROLLOUT_THREADS"]),
        "--happo-num-mini-batch",
        "0",
        "--happo-gpu-rollout-ref",
        str(config["HAPPO_GPU_ROLLOUT_REF"]),
        "--happo-ppo-epoch",
        "10",
        "--happo-critic-epoch",
        "10",
        "--masac-critic-batch-size",
        str(config["SIX_JOB_MASAC_BATCH"]),
        "--masac-buffer-size",
        str(config["SIX_JOB_MASAC_BUF"]),
        "--masac-max-replay-buffer-gib",
        str(config["SIX_JOB_MASAC_GIB"]),
        "--masac-rnn-hidden-dim",
        "64",
        "--masac-qmix-hidden-dim",
        "32",
        "--masac-hyper-hidden-dim",
        "64",
        "--masac-preload-batch-device",
        "auto",
        "--masac-actor-sample-times",
        "1",
        "--masac-critic-train-steps",
        "1",
        "--matd3-batch-size",
        str(config["SIX_JOB_MATD3_BATCH"]),
        "--matd3-buffer-size",
        str(config["SIX_JOB_MATD3_BUF"]),
        "--matd3-hidden-size",
        str(config["SIX_JOB_MATD3_HIDDEN"]),
        "--matd3-train-interval",
        "100",
        "--maac-batch-size",
        str(config["SIX_JOB_MAAC_BATCH"]),
        "--maac-buffer-length",
        str(config["SIX_JOB_MAAC_BUF"]),
        "--maac-hidden-size",
        str(config["SIX_JOB_MAAC_HIDDEN"]),
        "--maac-steps-per-update",
        "50",
        "--maac-num-updates",
        str(config["SIX_JOB_MAAC_UPDATES"]),
        "--execution-mode",
        "two_phase_happo_masac",
        "--two-phase-torch-threads",
        str(config["TORCH_THREADS"]),
        "--two-phase-p1-torch-threads",
        str(config["TWO_PHASE_P1_TORCH"]),
        "--two-phase-p2-torch-threads",
        str(config["TWO_PHASE_P2_TORCH"]),
        "--six-job-cuda-fraction",
        str(config["SIX_JOB_CUDA_FRAC"]),
        "--six-job-masac-cuda-fraction",
        str(config["SIX_JOB_MASAC_CUDA_FRAC"]),
        "--six-job-masac-buffer-size",
        str(config["SIX_JOB_MASAC_BUF"]),
        "--six-job-masac-max-replay-gib",
        str(config["SIX_JOB_MASAC_GIB"]),
        "--six-job-masac-critic-batch-size",
        str(config["SIX_JOB_MASAC_BATCH"]),
    ]
    max_jobs = int(config.get("MAX_CONCURRENT_JOBS") or 0)
    if max_jobs > 0:
        base += ["--max-concurrent-jobs", str(max_jobs)]
    return base


def verify_two_phase_protocol(config: Mapping[str, object]) -> None:
    launcher = Path(str(config["LAUNCHER"]))
    monitor = Path(str(config["MONITOR"]))
    if not launcher.is_file() or not monitor.is_file():
        raise RuntimeError(
            f"Scripts no encontrados: launcher={launcher} monitor={monitor}. "
            "Ejecuta celda 1.2 (git pull) antes de 7.2."
        )
    launcher_src = launcher.read_text(encoding="utf-8")
    monitor_src = monitor.read_text(encoding="utf-8")
    required = (
        "run_two_phase_happo_masac_jobs",
        "TWO_PHASE_P1_HM",
        "LAUNCHER_PROTOCOL_ID",
        "two_phase_happo_masac_v3",
    )
    missing = [s for s in required if s not in launcher_src]
    if missing:
        raise RuntimeError(f"Launcher sin protocolo two_phase: faltan {missing}. git pull + celda 1.2.")
    if "two_phase_happo_masac_v3" not in monitor_src:
        raise RuntimeError("Monitor desactualizado. git pull + celda 1.2.")
    print("[protocol] verify_two_phase_protocol PASSED (bootstrap 7.2)")


def launch_signature(config: Mapping[str, object]) -> Tuple[str, int, int, int]:
    return (
        str(config.get("OUTPUT_ROOT", "")),
        int(config.get("EPISODES", 0) or 0),
        int(config.get("HAPPO_ROLLOUT_THREADS", 0) or 0),
        int(config.get("MAX_CONCURRENT_JOBS", 0) or 0),
    )


def bind_cell_72_helpers(ns: MutableMapping[str, object], config: Mapping[str, object]) -> None:
    """Install launcher helpers into notebook globals when 7.0 was not run."""
    sig = launch_signature(config)

    def _launcher_base_args() -> List[str]:
        return build_official_launcher_argv(config)

    def _resolve_output_root_or_latest() -> str:
        return str(config["OUTPUT_ROOT"])

    def _dry_run_already_validated() -> bool:
        return ns.get("_DRY_RUN_VALIDATED") == sig

    def _mark_dry_run_validated() -> None:
        ns["_DRY_RUN_VALIDATED"] = sig

    ns["launcher_base_args"] = _launcher_base_args
    ns["resolve_output_root_or_latest"] = _resolve_output_root_or_latest
    ns["verify_two_phase_protocol"] = lambda: verify_two_phase_protocol(config)
    ns["dry_run_already_validated"] = _dry_run_already_validated
    ns["mark_dry_run_validated"] = _mark_dry_run_validated


def prepare_colab_cell_72_standalone(
    ns: MutableMapping[str, object],
    *,
    repo: Optional[Path] = None,
    resume_output_root: Optional[str] = None,
) -> Dict[str, object]:
    """Bootstrap Colab cell 7.2 without prior notebook cells (Drive audit + plan)."""
    if not _in_colab():
        return {}

    scripts = Path(__file__).resolve().parent
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from citylearn_v3_training_common import (
        assert_canonical_colab_skip_plan,
        build_jobs_resume_report,
        discover_colab_gdrive_workspace,
        notebook_jobs_resume_preview,
        resolve_colab_mydrive_resume_root,
    )

    repo_path = Path(repo or ns.get("REPO") or "/content/MADRLCitytleranflexresdr")
    if not (repo_path / "CityLearn").is_dir():
        raise RuntimeError(
            f"REPO invalido: {repo_path}. Ejecuta celda 1.2 (clone) o define REPO antes de 7.2."
        )

    mount = Path("/content/drive")
    if not (mount / "MyDrive").is_dir():
        from google.colab import drive  # type: ignore[import-not-found]

        print("[7.2 bootstrap] Montando Google Drive...")
        drive.mount("/content/drive")

    gdrive = discover_colab_gdrive_workspace(mount, repo=repo_path, audit_runs=False)
    if gdrive is None:
        raise RuntimeError("No se encontro workspace MADRL en Drive. Ejecuta celda 1.5 o 1.2.")

    manual = str(resume_output_root or ns.get("RESUME_OUTPUT_ROOT") or ns.get("OUTPUT_ROOT") or "").strip()
    target_ep = int(ns.get("N_EPISODES", ns.get("EPISODES", 50)) or 50)
    ep_steps = int(ns.get("EPISODE_STEPS", 8760) or 8760)

    output_root = None
    if manual and Path(manual).is_dir():
        output_root = Path(manual)
    if output_root is None:
        output_root = resolve_colab_mydrive_resume_root(
            gdrive,
            repo=repo_path,
            resume_output_root=manual or None,
            target_episodes=target_ep,
            episode_time_steps=ep_steps,
            require_canonical_plan=True,
        )
    if output_root is None:
        output_root = resolve_colab_mydrive_resume_root(
            gdrive,
            repo=repo_path,
            resume_output_root=manual or None,
            target_episodes=target_ep,
            episode_time_steps=ep_steps,
            require_canonical_plan=False,
        )
    if output_root is None:
        raise RuntimeError(
            "No hay OUTPUT_ROOT reanudable en Drive. Ejecuta celda 2.1 o define RESUME_OUTPUT_ROOT."
        )

    config = colab_a100_training_config(
        repo_path,
        output_root=output_root,
        gdrive_root=gdrive,
        target_episodes=target_ep,
        episode_steps=ep_steps,
    )
    ns.update(config)
    bind_cell_72_helpers(ns, config)

    print("\n[7.2 bootstrap] Modo standalone — sin celdas 1.5/2.1/7.1")
    print(f"[7.2 bootstrap] OUTPUT_ROOT = {output_root}")
    print(f"[7.2 bootstrap] HAPPO rollout_threads = {config['HAPPO_ROLLOUT_THREADS']}")

    report = build_jobs_resume_report(
        output_root,
        target_episodes=target_ep,
        episode_time_steps=ep_steps,
        happo_rollout_threads=int(config["HAPPO_ROLLOUT_THREADS"]),
    )
    notebook_jobs_resume_preview(
        output_root,
        target_episodes=target_ep,
        episode_time_steps=ep_steps,
        happo_rollout_threads=int(config["HAPPO_ROLLOUT_THREADS"]),
        label="7.2 bootstrap",
        show_footer_hint=False,
    )
    try:
        assert_canonical_colab_skip_plan(report, output_root=output_root)
    except RuntimeError:
        salvage = sum(
            1
            for row in report.get("jobs") or []
            if str(row.get("action")) in ("resume", "happo_salvage_kpi")
        )
        done = int(report.get("completed") or 0)
        print(f"[7.2 bootstrap] Plan: {done} SKIP + {salvage} REANUDA (no canonico estricto)")

    ns["_CELL_72_BOOTSTRAPPED"] = True
    return dict(config)
