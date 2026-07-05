"""Notebook cell 7.2 bootstrap — one cell resumes full Colab training.

On Colab reconnect: run **only** cell 7.2. Bootstrap runs git hard sync (1.2),
Drive mount (1.5), OUTPUT_ROOT discovery (2.1), training config (6.1/7.0), and
skip/resume plan — then launches two_phase_happo_masac with --skip-completed.
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

COLAB_REPO_URL = "https://github.com/Mac-Tapia/MADRLCitytleranflexresdr.git"
COLAB_REPO_BRANCH = "codex/fix-madrl-traceability-docs"
COLAB_DEFAULT_REPO = "/content/MADRLCitytleranflexresdr"
CITYLEARN_URL = "https://github.com/Mac-Tapia/CityLearn.git"
CITYLEARN_BRANCH = "codex/iquitos-distillation-madrl-docs"
MAAC_URL = "https://github.com/Mac-Tapia/MAAC.git"
MAAC_BRANCH = "codex/integrar-limpieza-diagnosticos"


def _git_check(args: Sequence[str], *, cwd: Optional[Path] = None) -> None:
    cmd = ["git", *[str(a) for a in args]]
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def _git_out(args: Sequence[str], *, cwd: Path) -> str:
    return subprocess.check_output(
        ["git", *[str(a) for a in args]],
        text=True,
        cwd=str(cwd),
    ).strip()


def colab_git_hard_sync(
    repo: Path,
    *,
    clone_if_missing: bool = True,
) -> Dict[str, str]:
    """Mirror notebook cell 1.2: parent hard reset + CityLearn/MAAC live branches."""
    repo = Path(repo)
    citylearn_dir = repo / "CityLearn"
    maac_dir = repo / "external" / "MAAC"

    if not (repo / ".git").is_dir():
        if not clone_if_missing:
            raise RuntimeError(f"REPO sin .git: {repo}")
        if repo.exists() and any(repo.iterdir()):
            raise RuntimeError(
                f"{repo} existe pero sin .git. Elimina la carpeta y vuelve a ejecutar 7.2."
            )
        print(f"[7.2 bootstrap] Clonando {COLAB_REPO_URL} ({COLAB_REPO_BRANCH})...")
        _git_check(
            [
                "clone",
                "--branch",
                COLAB_REPO_BRANCH,
                "--depth",
                "1",
                "--recurse-submodules",
                "--shallow-submodules",
                COLAB_REPO_URL,
                str(repo),
            ]
        )
    else:
        origin = _git_out(["config", "--get", "remote.origin.url"], cwd=repo)
        if origin != COLAB_REPO_URL:
            raise RuntimeError(f"Repo apunta a {origin}; esperado {COLAB_REPO_URL}.")
        print(f"[7.2 bootstrap] HARD SYNC origin/{COLAB_REPO_BRANCH}...")
        _git_check(["fetch", "origin", COLAB_REPO_BRANCH], cwd=repo)
        _git_check(["reset", "--hard", f"origin/{COLAB_REPO_BRANCH}"], cwd=repo)
        _git_check(["clean", "-fd"], cwd=repo)
        _git_check(["submodule", "sync", "--recursive"], cwd=repo)
        _git_check(["submodule", "update", "--init", "--recursive", "--force"], cwd=repo)

    print(f"[7.2 bootstrap] CityLearn -> {CITYLEARN_BRANCH}")
    remotes = _git_out(["remote"], cwd=citylearn_dir).splitlines()
    if "mac-tapia" not in remotes:
        _git_check(["remote", "add", "mac-tapia", CITYLEARN_URL], cwd=citylearn_dir)
    else:
        _git_check(["remote", "set-url", "mac-tapia", CITYLEARN_URL], cwd=citylearn_dir)
    _git_check(["fetch", "mac-tapia", CITYLEARN_BRANCH], cwd=citylearn_dir)
    _git_check(["checkout", "-B", CITYLEARN_BRANCH, f"mac-tapia/{CITYLEARN_BRANCH}"], cwd=citylearn_dir)
    _git_check(["clean", "-fd"], cwd=citylearn_dir)
    cl_branch = _git_out(["rev-parse", "--abbrev-ref", "HEAD"], cwd=citylearn_dir)
    if cl_branch == "HEAD":
        _git_check(["checkout", "-B", CITYLEARN_BRANCH], cwd=citylearn_dir)
        cl_branch = _git_out(["rev-parse", "--abbrev-ref", "HEAD"], cwd=citylearn_dir)
    if cl_branch != CITYLEARN_BRANCH:
        raise RuntimeError(f"CityLearn en rama {cl_branch!r}, esperado {CITYLEARN_BRANCH!r}")
    cl_commit = _git_out(["rev-parse", "--short", "HEAD"], cwd=citylearn_dir)

    print(f"[7.2 bootstrap] external/MAAC -> {MAAC_BRANCH}")
    maac_remotes = _git_out(["remote"], cwd=maac_dir).splitlines()
    if "mac-tapia" not in maac_remotes:
        _git_check(["remote", "add", "mac-tapia", MAAC_URL], cwd=maac_dir)
    else:
        _git_check(["remote", "set-url", "mac-tapia", MAAC_URL], cwd=maac_dir)
    _git_check(["fetch", "mac-tapia", MAAC_BRANCH], cwd=maac_dir)
    _git_check(["checkout", "-B", MAAC_BRANCH, f"mac-tapia/{MAAC_BRANCH}"], cwd=maac_dir)
    _git_check(["clean", "-fd"], cwd=maac_dir)
    maac_branch = _git_out(["rev-parse", "--abbrev-ref", "HEAD"], cwd=maac_dir)
    if maac_branch != MAAC_BRANCH:
        raise RuntimeError(f"MAAC en rama {maac_branch!r}, esperado {MAAC_BRANCH!r}")
    maac_commit = _git_out(["rev-parse", "--short", "HEAD"], cwd=maac_dir)

    parent_head = _git_out(["rev-parse", "--short", "HEAD"], cwd=repo)
    print(
        f"[7.2 bootstrap] git OK: padre {COLAB_REPO_BRANCH}@{parent_head} | "
        f"CityLearn@{cl_commit} | MAAC@{maac_commit}"
    )
    return {
        "parent": parent_head,
        "citylearn": cl_commit,
        "maac": maac_commit,
    }


def colab_verify_repo_patches(repo: Path) -> None:
    """Protocol guard + critical patches (cells 1.2 E/F)."""
    repo = Path(repo)
    guard = repo / "CityLearn/scripts/colab_protocol_guard.py"
    patches = repo / "CityLearn/scripts/colab_verify_critical_patches.py"
    if not guard.is_file():
        raise FileNotFoundError(f"Falta {guard}")
    if not patches.is_file():
        raise FileNotFoundError(f"Falta {patches}")
    subprocess.check_call([sys.executable, str(guard), "verify-repo", "--repo", str(repo)])
    subprocess.check_call([sys.executable, str(patches), "--repo", str(repo)])
    print("[7.2 bootstrap] protocol-guard + parches criticos OK", flush=True)


def colab_mount_drive_if_needed() -> Path:
    """Mount Google Drive (cell 1.5) when /content/drive/MyDrive is absent."""
    mount = Path("/content/drive")
    if not (mount / "MyDrive").is_dir():
        from google.colab import drive  # type: ignore[import-not-found]

        print("[7.2 bootstrap] Montando Google Drive...", flush=True)
        drive.mount("/content/drive")
    return mount


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
    skip_git_sync: bool = False,
) -> Dict[str, object]:
    """Full Colab reconnect bootstrap: git 1.2 + Drive 1.5 + OUTPUT_ROOT 2.1 + config 6.1/7.0."""
    if not _in_colab():
        return {}

    repo_path = Path(repo or ns.get("REPO") or COLAB_DEFAULT_REPO)
    ns["REPO"] = str(repo_path)

    if not skip_git_sync:
        print("[7.2 bootstrap] (1/4) git hard sync...", flush=True)
        colab_git_hard_sync(repo_path)
        colab_verify_repo_patches(repo_path)
    elif not (repo_path / "CityLearn").is_dir():
        raise RuntimeError(
            f"REPO invalido: {repo_path}. Quita skip_git_sync o ejecuta celda 1.2."
        )

    print("[7.2 bootstrap] (2/4) Google Drive...", flush=True)
    colab_mount_drive_if_needed()

    scripts = repo_path / "CityLearn" / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from citylearn_v3_training_common import (
        assert_canonical_colab_skip_plan,
        bootstrap_colab_notebook_cell_72,
        notebook_jobs_resume_preview,
    )

    manual = str(resume_output_root or ns.get("RESUME_OUTPUT_ROOT") or ns.get("OUTPUT_ROOT") or "").strip()
    if manual and Path(manual).is_dir():
        ns["RESUME_OUTPUT_ROOT"] = manual
        ns["OUTPUT_ROOT"] = manual

    print("[7.2 bootstrap] (3/4) OUTPUT_ROOT + plan skip/resume...", flush=True)
    boot = bootstrap_colab_notebook_cell_72(
        repo_path,
        python_executable=str(ns.get("PROJECT_PYTHON") or ns.get("PYTHON") or sys.executable),
        require_canonical_plan=False,
        verbose=False,
    )
    config = dict(boot["globals"])
    ns.update(config)
    bind_cell_72_helpers(ns, config)

    target_ep = int(config.get("N_EPISODES", 50))
    ep_steps = int(config.get("EPISODE_STEPS", 8760))
    output_root = Path(str(config["OUTPUT_ROOT"]))

    print("\n[7.2 bootstrap] (4/4) listo — celda unica (sin 1.2/1.5/2.1/6.1/7.0/7.1)")
    print(f"[7.2 bootstrap] OUTPUT_ROOT = {output_root}")
    print(f"[7.2 bootstrap] HAPPO rollout_threads = {config['HAPPO_ROLLOUT_THREADS']}")
    print(
        "[7.2 bootstrap] Reanuda mismas carpetas Drive; salvage HAPPO en paralelo si VRAM alcanza"
    )

    report = boot["resume_report"]
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
