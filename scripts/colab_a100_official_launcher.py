"""Colab A100 launcher for the CityLearn v3 MADRL official training chain.

This script mirrors the Windows PowerShell launcher contract in a Python form
that works inside Google Colab notebooks. It writes official_full_status.json,
keeps logs per MADRL job, supports dry-run and resume, and can retry one CUDA
OOM with more conservative resource settings.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


ALGORITHMS = ("happo", "masac", "matd3", "maac")
CITYLEARN_V2_BENCHMARKS = ("PPO", "SAC", "A2C")
SCENARIOS = ("E1", "E2", "E3")
HEAVY_ALGORITHMS = {"masac", "maac"}

# two_phase execution: Phase 1 = light algorithms in parallel (9 jobs),
# Phase 2 = MASAC alone (3 jobs with dedicated A100 GPU+CPU).
# MAAC is grouped with light algorithms: its GPU footprint (~0.7 GiB/job) is
# negligible and its attention mechanism is far lighter than MASAC's QMIX replay.
TWO_PHASE_LIGHT = ("happo", "matd3", "maac")  # Phase 1: 9 jobs simultaneous
TWO_PHASE_HEAVY = ("masac",)                  # Phase 2: 3 jobs, dedicated GPU
DEFAULT_SCHEMA = "CityLearn/data/datasets/citylearn_iquitos_2023_2025/schema.json"
DEFAULT_OUTPUT_ROOT = "outputs/colab_madrl_a100_official"
REFERENCE_SOURCES = [
    {
        "key": "citylearn_standardization",
        "title": "CityLearn: Standardizing Research in Multi-Agent Reinforcement Learning for Demand Response and Urban Energy Management",
        "url": "https://arxiv.org/abs/2012.10504",
    },
    {
        "key": "citylearn_v2",
        "title": "CityLearn v2: energy-flexible, resilient, occupant-centric, and carbon-aware",
        "url": "https://escholarship.org/content/qt5t48x8xk/qt5t48x8xk.pdf",
    },
    {
        "key": "citylearn_challenge_2022",
        "title": "The CityLearn Challenge 2022: Overview, Results, and Lessons Learned",
        "url": "https://proceedings.mlr.press/v220/nweye23a.html",
    },
    {
        "key": "happo_hatrpo",
        "title": "Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning",
        "url": "https://openreview.net/forum?id=EcGGFkNTxdJ",
    },
    {
        "key": "maac",
        "title": "Actor-Attention-Critic for Multi-Agent Reinforcement Learning",
        "url": "https://proceedings.mlr.press/v97/iqbal19a.html",
    },
    {
        "key": "matd3",
        "title": "Reducing Overestimation Bias in Multi-Agent Domains Using Double Centralized Critics",
        "url": "https://arxiv.org/abs/1910.01465",
    },
    {
        "key": "qmix",
        "title": "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning",
        "url": "https://arxiv.org/abs/1803.11485",
    },
    {
        "key": "sac",
        "title": "Soft Actor-Critic Algorithms and Applications",
        "url": "https://arxiv.org/abs/1812.05905",
    },
    {
        "key": "pytorch_cuda",
        "title": "PyTorch CUDA semantics and memory management",
        "url": "https://docs.pytorch.org/docs/stable/notes/cuda.html",
    },
    {
        "key": "pytorch_reproducibility",
        "title": "PyTorch reproducibility notes",
        "url": "https://docs.pytorch.org/docs/stable/notes/randomness.html",
    },
    {
        "key": "colab_limits",
        "title": "Google Colab FAQ: resources and GPU availability are not guaranteed",
        "url": "https://research.google.com/colaboratory/faq.html",
    },
    {
        "key": "nvidia_a100",
        "title": "NVIDIA A100 Tensor Core GPU datasheet",
        "url": "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf",
    },
    {
        "key": "ross_may_phd",
        "title": "On the Feasibility of Reinforcement Learning in Single- and Multi-Agent Systems",
        "url": "https://www.diva-portal.org/smash/get/diva2:1731696/FULLTEXT01.pdf",
    },
    {
        "key": "oxford_energy_flexibility_thesis",
        "title": "Multi-agent reinforcement learning for the coordination of residential energy flexibility",
        "url": "https://ora.ox.ac.uk/objects/uuid:4acf19ee-76b3-4e71-a424-259cd9e2d6f5",
    },
    {
        "key": "polito_marl_building_energy_thesis",
        "title": "Multi-Agent Reinforcement Learning Building Energy Optimization",
        "url": "https://webthesis.biblio.polito.it/40396/1/tesi.pdf",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return root / path


def path_for_status(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def resolve_status_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_text(cmd: Sequence[str], *, timeout: int = 30) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception as exc:
        return f"ERROR: {exc}"
    return (proc.stdout or proc.stderr or "").strip()


def query_gpu() -> Dict[str, object]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {"available": False, "error": "nvidia-smi not found"}

    text = run_text(
        [
            nvidia_smi,
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not text or text.startswith("ERROR:"):
        return {"available": False, "error": text or "nvidia-smi returned no data"}

    first = text.splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 4:
        return {"available": False, "error": f"unexpected nvidia-smi output: {first}"}

    total_mib = float(parts[1])
    free_mib = float(parts[2])
    return {
        "available": True,
        "name": parts[0],
        "memory_total_mib": total_mib,
        "memory_free_mib": free_mib,
        "memory_total_gib": round(total_mib / 1024.0, 2),
        "memory_free_gib": round(free_mib / 1024.0, 2),
        "driver_version": parts[3],
    }


def configure_environment(root: Path, args: argparse.Namespace) -> Dict[str, object]:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if args.cuda:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")
    os.environ["PYTHONUNBUFFERED"] = "1"
    # Force non-interactive Agg backend for all subprocesses that import matplotlib.
    # Without this, concurrent MASAC/MAAC subprocesses can race on the font-cache
    # directory (~/.cache/matplotlib) when they all start simultaneously, causing
    # random ImportError or corrupted-font-cache failures.
    os.environ.setdefault("MPLBACKEND", "Agg")

    path_entries = [
        root,
        root / "CityLearn",
        root / "CityLearn" / "scripts",
        root / "external" / "HARL",
        root / "external" / "MARL" / "src",
        root / "external" / "off-policy",
        root / "external" / "MAAC",
    ]
    for entry in reversed(path_entries):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)

    current_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_entries = [str(entry) for entry in path_entries]
    if current_pythonpath:
        pythonpath_entries.append(current_pythonpath)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    os.environ["CITYLEARN_PROJECT_ROOT"] = str(root)

    return {
        "python": sys.executable,
        "platform": platform.platform(),
        "python_version": sys.version,
        "pythonpath_prefix": pythonpath_entries[:7],
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }


def validate_gpu(args: argparse.Namespace) -> Dict[str, object]:
    gpu = query_gpu()
    if args.skip_gpu_preflight:
        gpu["preflight_skipped"] = True
        return gpu

    if args.cuda and not gpu.get("available"):
        raise RuntimeError(f"CUDA requested but GPU preflight failed: {gpu.get('error')}")

    if args.require_a100 and "A100" not in str(gpu.get("name", "")):
        raise RuntimeError(
            "A100 GPU required. Current GPU is "
            f"{gpu.get('name', 'not detected')}. Select Runtime > Change runtime type > A100."
        )

    if args.require_a100 and float(gpu.get("memory_total_gib") or 0.0) < 39.0:
        raise RuntimeError(f"A100 preflight expected >=39 GiB VRAM, got {gpu.get('memory_total_gib')} GiB.")

    return gpu


def validate_torch(args: argparse.Namespace) -> Dict[str, object]:
    if args.skip_gpu_preflight:
        return {"skipped": True}

    import torch

    cuda_available = bool(torch.cuda.is_available())
    if args.cuda and not cuda_available:
        raise RuntimeError("PyTorch reports CUDA unavailable.")

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if cuda_available:
        torch.cuda.set_device(0)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        if args.cuda_memory_fraction > 0:
            torch.cuda.set_per_process_memory_fraction(float(args.cuda_memory_fraction), 0)

    return {
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_enabled": bool(args.cuda and cuda_available),
        "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "device_total_memory_gib": (
            torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_available else None
        ),
        "matmul_precision": "high",
        "tf32": bool(cuda_available),
    }


def smoke_imports() -> Dict[str, object]:
    modules = [
        "torch",
        "numpy",
        "pandas",
        "citylearn",
        "citylearn.v3.environment",
        "harl",
        "runner_msac",
        "offpolicy",
        "algorithms.attention_sac",
    ]
    results: Dict[str, object] = {}
    for module_name in modules:
        try:
            if module_name == "algorithms.attention_sac":
                sys.modules.pop("algorithms", None)
            __import__(module_name)
            results[module_name] = "ok"
        except Exception as exc:
            results[module_name] = f"failed: {exc}"
    failures = {name: result for name, result in results.items() if str(result).startswith("failed:")}
    if failures:
        raise RuntimeError(f"Smoke imports failed: {failures}")
    return results


def scenario_list(value: str) -> List[str]:
    text = str(value or "ALL").upper()
    if text in {"ALL", "TODOS", "3EJES"}:
        return list(SCENARIOS)
    if text not in SCENARIOS:
        raise ValueError(f"Unknown scenario {value}. Use E1, E2, E3, or ALL.")
    return [text]


def replace_arg(args: List[str], flag: str, value: object) -> List[str]:
    output = list(args)
    if flag in output:
        idx = output.index(flag)
        if idx + 1 < len(output):
            output[idx + 1] = str(value)
            return output
    output.extend([flag, str(value)])
    return output


def output_base(output_root: Path, algorithm: str) -> Path:
    return output_root / algorithm


def run_dir(output_root: Path, algorithm: str, scenario: str, seed: int) -> Path:
    return output_base(output_root, algorithm) / f"{scenario}_seed_{seed}"


def common_args(
    args: argparse.Namespace,
    *,
    schema_arg: str,
    output_root: Path,
    algorithm: str,
    scenario: str,
) -> List[str]:
    values = [
        "--scenario",
        scenario,
        "--schema-path",
        schema_arg,
        "--seed",
        str(args.seed),
        "--episode-time-steps",
        str(args.episode_time_steps),
        "--output-dir",
        str(output_base(output_root, algorithm)),
        "--torch-threads",
        str(args.torch_threads),
        "--live-progress-interval",
        str(args.live_progress_interval),
        "--live-heartbeat-seconds",
        str(args.live_heartbeat_seconds),
        "--artifact-profile",
        args.artifact_profile,
        "--trace-record-interval",
        str(args.trace_record_interval),
        "--trace-detail",
        args.trace_detail,
        "--gpu-profile",
        args.gpu_profile,
    ]
    if args.cuda:
        values.append("--cuda")
    if args.cuda_memory_fraction > 0:
        values.extend(["--cuda-memory-fraction", str(args.cuda_memory_fraction)])
    return values


def build_jobs(args: argparse.Namespace, root: Path, output_root: Path, schema_arg: str) -> List[Dict[str, object]]:
    jobs: List[Dict[str, object]] = []
    scenarios = scenario_list(args.scenario)
    num_env_steps = int(args.episode_time_steps) * int(args.episodes)

    for scenario in scenarios:
        jobs.append(
            {
                "name": "happo",
                "scenario": scenario,
                "script": "CityLearn/scripts/train_citylearn_v3_happo.py",
                "args": common_args(args, schema_arg=schema_arg, output_root=output_root, algorithm="happo", scenario=scenario)
                + [
                    "--episodes",
                    str(args.episodes),
                    "--num-env-steps",
                    str(num_env_steps),
                    "--hidden-size",
                    str(args.happo_hidden_size),
                    "--n-rollout-threads",
                    "1",
                    "--log-interval",
                    "1",
                    "--eval-interval",
                    "1",
                    "--actor-lr",
                    "1e-4",
                    "--critic-lr",
                    "5e-4",
                    "--max-grad-norm",
                    "1.0",
                    "--gamma",
                    "0.9999",
                    "--action-aggregation",
                    "mean",
                    "--live-progress-interval",
                    "300",
                ],
            }
        )
        jobs.append(
            {
                "name": "masac",
                "scenario": scenario,
                "script": "CityLearn/scripts/train_citylearn_v3_masac.py",
                "args": common_args(args, schema_arg=schema_arg, output_root=output_root, algorithm="masac", scenario=scenario)
                + [
                    "--episodes",
                    str(args.episodes),
                    "--epochs",
                    str(args.episodes),
                    "--action-bins",
                    "3",
                    "--discrete-action-mode",
                    "axis",
                    "--max-replay-buffer-gib",
                    str(args.masac_max_replay_buffer_gib),
                    "--buffer-size",
                    str(args.masac_buffer_size),
                    "--critic-batch-size",
                    str(args.masac_critic_batch_size),
                    "--critic-train-steps",
                    str(args.masac_critic_train_steps),
                    "--actor-sample-times",
                    str(args.masac_actor_sample_times),
                    "--masac-preload-batch-device",
                    args.masac_preload_batch_device,  # default="cuda" → FS: GPU VRAM per-algorithm
                    "--actor-lr",
                    "3e-4",
                    "--critic-lr",
                    "5e-4",
                    "--alpha-lr",
                    "3e-4",
                    "--grad-norm-clip",
                    "1.0",
                    "--gamma",
                    "0.9999",
                    "--rnn-hidden-dim",
                    str(args.masac_rnn_hidden_dim),
                    "--qmix-hidden-dim",
                    str(args.masac_qmix_hidden_dim),
                    "--hyper-hidden-dim",
                    str(args.masac_hyper_hidden_dim),
                ],
            }
        )
        jobs.append(
            {
                "name": "matd3",
                "scenario": scenario,
                "script": "CityLearn/scripts/train_citylearn_v3_matd3.py",
                "args": common_args(args, schema_arg=schema_arg, output_root=output_root, algorithm="matd3", scenario=scenario)
                + [
                    "--episodes",
                    str(args.episodes),
                    "--num-env-steps",
                    str(num_env_steps),
                    "--batch-size",
                    str(args.matd3_batch_size),
                    "--buffer-size",
                    str(args.matd3_buffer_size),
                    "--hidden-size",
                    str(args.matd3_hidden_size),
                    "--lr",
                    str(args.matd3_lr),
                    "--max-grad-norm",
                    "1.0",
                    "--gamma",
                    "0.9999",
                    "--train-interval",
                    str(args.matd3_train_interval),
                    "--num-random-episodes",
                    str(args.matd3_num_random_episodes),
                    # MATD3 dynamic live-progress-interval: initial 50 steps (fast feedback),
                    # then 300 steps after 2000 total steps (reduce I/O during main training).
                    "--live-progress-interval-initial",
                    "50",
                    "--live-progress-interval-threshold",
                    "2000",
                    "--live-progress-interval-stable",
                    "300",
                ],
            }
        )
        jobs.append(
            {
                "name": "maac",
                "scenario": scenario,
                "script": "CityLearn/scripts/train_citylearn_v3_maac.py",
                "args": common_args(args, schema_arg=schema_arg, output_root=output_root, algorithm="maac", scenario=scenario)
                + [
                    "--episodes",
                    str(args.episodes),
                    "--action-bins",
                    "3",
                    "--discrete-action-mode",
                    "axis",
                    "--max-discrete-actions",
                    "512",
                    "--batch-size",
                    str(args.maac_batch_size),
                    "--buffer-length",
                    str(args.maac_buffer_length),
                    "--steps-per-update",
                    str(args.maac_steps_per_update),
                    "--num-updates",
                    str(args.maac_num_updates),
                    "--hidden-size",
                    str(args.maac_hidden_size),
                    "--attend-heads",
                    str(args.maac_attend_heads),
                    "--pi-lr",
                    "3e-4",
                    "--q-lr",
                    str(args.maac_q_lr),
                    "--tau",
                    str(args.maac_tau),
                    "--gamma",
                    "0.9999",
                    "--reward-scale",
                    "10.0",
                    "--live-progress-interval",
                    "300",
                ],
            }
        )

    algo_order = ALGORITHMS
    start_idx = algo_order.index(args.start_from_algorithm)
    return [job for job in jobs if algo_order.index(str(job["name"])) >= start_idx]


def command_for_job(job: Mapping[str, object]) -> List[str]:
    return [sys.executable, "-B", str(job["script"])] + [str(item) for item in job["args"]]


def completed_artifact_exists(root: Path, job_output_dir: str) -> bool:
    run_path = resolve_status_path(root, job_output_dir)
    return (run_path / "data" / "results.json").exists() or (run_path / "results.json").exists()


def is_oom_failure(*paths: Path) -> bool:
    needles = ("cuda out of memory", "torch.outofmemoryerror", "outofmemoryerror", "cublas_status_alloc_failed")
    for path in paths:
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except Exception:
            continue
        if any(needle in text for needle in needles):
            return True
    return False


def make_oom_retry_job(job: Mapping[str, object]) -> Optional[Dict[str, object]]:
    retry = copy.deepcopy(dict(job))
    name = str(retry["name"])
    args = [str(item) for item in retry["args"]]

    if name == "masac":
        # OOM fallback: buffer_size=15 → ~10.28 GiB float32/job. Drop to 8 episodes
        # (6.85 GiB float32) + keep on GPU (preload=cuda) but cap cuda_frac=0.20.
        # Reduces hidden dims to lower peak GPU pressure during concurrent phase.
        args = replace_arg(args, "--buffer-size", "8")
        args = replace_arg(args, "--critic-batch-size", "64")
        args = replace_arg(args, "--max-replay-buffer-gib", "12")
        args = replace_arg(args, "--masac-preload-batch-device", "cuda")
        args = replace_arg(args, "--cuda-memory-fraction", "0.20")
        args = replace_arg(args, "--rnn-hidden-dim", "64")
        args = replace_arg(args, "--qmix-hidden-dim", "64")
        args = replace_arg(args, "--hyper-hidden-dim", "128")
        args = replace_arg(args, "--critic-train-steps", "1")
        args = replace_arg(args, "--actor-sample-times", "5")
    elif name == "matd3":
        # OOM fallback: reduce batch (less GPU memory per step) and buffer (less SSD/RAM).
        args = replace_arg(args, "--batch-size", "512")
        args = replace_arg(args, "--buffer-size", "50000")
        args = replace_arg(args, "--hidden-size", "256")
        args = replace_arg(args, "--train-interval", "100")
    elif name == "maac":
        # OOM fallback: reduce batch and buffer.
        args = replace_arg(args, "--batch-size", "256")
        args = replace_arg(args, "--buffer-length", "50000")
        args = replace_arg(args, "--hidden-size", "256")
        args = replace_arg(args, "--attend-heads", "4")
    else:
        return None

    retry["args"] = args
    retry["oom_retry"] = True
    return retry


def print_monitor_snapshot(root: Path, status_path: Path, log_tail: int = 12) -> None:
    status = read_json(status_path)
    if not status:
        print(f"[monitor] status not found: {status_path}", flush=True)
        return

    active = None
    for job in status.get("jobs", []):
        if job.get("completed_at") is None:
            active = job
            break
    if not active:
        print(f"[monitor] status={status.get('status')} jobs={len(status.get('jobs', []))}", flush=True)
        return

    run_path = resolve_status_path(root, str(active["output_dir"]))
    progress = read_json(run_path / "live_progress.json")
    print("", flush=True)
    print(
        f"[monitor] {active['name'].upper()}/{active['scenario']} "
        f"status={status.get('status')} output={active['output_dir']}",
        flush=True,
    )
    if progress:
        total_steps = int(status.get("num_env_steps") or 0)
        global_step = int(progress.get("global_step") or 0)
        pct = (100.0 * global_step / total_steps) if total_steps else 0.0
        weights = progress.get("reward_axis_weights") or {}
        print(
            "  ep={}/{} step_ep={}/{} global={}/{} ({:.2f}%) reward_mean={} return={}".format(
                int(progress.get("episode") or 0) + 1,
                status.get("episodes"),
                progress.get("episode_step"),
                status.get("episode_time_steps"),
                global_step,
                total_steps,
                pct,
                progress.get("episode_reward_mean_cumulative"),
                progress.get("episode_return_cumulative"),
            ),
            flush=True,
        )
        print(
            "  weights flex={} carbon={} cost={} live_status={}".format(
                weights.get("flex"),
                weights.get("carbon"),
                weights.get("cost"),
                progress.get("live_status"),
            ),
            flush=True,
        )

    gpu = run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader",
        ],
        timeout=10,
    )
    if gpu and not gpu.startswith("ERROR:"):
        print(f"  gpu: {gpu.splitlines()[0]}", flush=True)

    log_path = Path(str(active.get("log") or ""))
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-log_tail:]
        if lines:
            print("  log tail:", flush=True)
            for line in lines[-log_tail:]:
                print(f"    {line[:180]}", flush=True)


def append_job_record(
    manifest: Dict[str, object],
    status_path: Path,
    record: Dict[str, object],
    lock: Optional[threading.Lock] = None,
) -> None:
    if lock:
        with lock:
            manifest.setdefault("jobs", [])
            manifest["jobs"].append(record)
            atomic_write_json(status_path, manifest)
    else:
        manifest.setdefault("jobs", [])
        manifest["jobs"].append(record)
        atomic_write_json(status_path, manifest)


def complete_job_record(
    manifest: Dict[str, object],
    status_path: Path,
    record: Dict[str, object],
    exit_code: int,
    started_time: float,
    lock: Optional[threading.Lock] = None,
) -> None:
    record["completed_at"] = utc_now()
    record["exit_code"] = int(exit_code)
    record["duration_minutes"] = round((time.time() - started_time) / 60.0, 3)
    if lock:
        with lock:
            atomic_write_json(status_path, manifest)
    else:
        atomic_write_json(status_path, manifest)


def run_one_job(
    *,
    root: Path,
    manifest: Dict[str, object],
    status_path: Path,
    job: Mapping[str, object],
    output_root: Path,
    log_dir: Path,
    args: argparse.Namespace,
    attempt: int = 0,
    lock: Optional[threading.Lock] = None,
    live_monitor: Optional[bool] = None,
) -> int:
    name = str(job["name"])
    scenario = str(job["scenario"])
    job_run_dir = run_dir(output_root, name, scenario, args.seed)
    job_output_dir = path_for_status(root, job_run_dir)
    use_monitor = args.live_monitor if live_monitor is None else live_monitor

    if args.skip_completed and completed_artifact_exists(root, job_output_dir):
        record = {
            "name": name,
            "scenario": scenario,
            "script": job["script"],
            "started_at": "skipped",
            "completed_at": utc_now(),
            "exit_code": 0,
            "output_dir": job_output_dir,
            "skipped": True,
            "skip_reason": "already_completed",
            "attempt": attempt,
        }
        append_job_record(manifest, status_path, record, lock=lock)
        print(f"SKIP {name.upper()}/{scenario}: existing results.json", flush=True)
        return 0

    label = f"{scenario}_{name}"
    suffix = "" if attempt == 0 else f"_retry{attempt}"
    log_path = log_dir / f"{label}{suffix}.log"
    err_path = log_dir / f"{label}{suffix}.stderr.log"
    command = command_for_job(job)
    started = time.time()
    record = {
        "name": name,
        "scenario": scenario,
        "script": job["script"],
        "started_at": utc_now(),
        "completed_at": None,
        "exit_code": None,
        "log": str(log_path),
        "stderr_log": str(err_path),
        "output_dir": job_output_dir,
        "command": " ".join(command),
        "attempt": attempt,
        "oom_retry": bool(job.get("oom_retry", False)),
    }
    append_job_record(manifest, status_path, record, lock=lock)

    startup_delay = float(job.get("startup_delay_seconds", 0))
    if startup_delay > 0:
        print(
            f"[launcher] {name.upper()}/{scenario}: startup delay {startup_delay:.0f}s "
            f"(stagger RAM allocation for concurrent dataset loads)",
            flush=True,
        )
        time.sleep(startup_delay)

    print(f"START {name.upper()}/{scenario} attempt={attempt} log={log_path}", flush=True)
    proc_env = os.environ.copy()
    # Per-job env overrides (e.g. OMP_NUM_THREADS to prevent numpy thread stealing).
    if job.get("env_overrides"):
        proc_env.update(job["env_overrides"])
    with log_path.open("w", encoding="utf-8") as stdout_f, err_path.open("w", encoding="utf-8") as stderr_f:
        proc = subprocess.Popen(
            command,
            cwd=str(root),
            stdout=stdout_f,
            stderr=stderr_f,
            text=True,
            env=proc_env,
        )

        last_monitor = 0.0
        while proc.poll() is None:
            if use_monitor and (time.time() - last_monitor) >= max(5, args.monitor_interval):
                print_monitor_snapshot(root, status_path, log_tail=args.log_tail)
                last_monitor = time.time()
            time.sleep(5)

        exit_code = int(proc.returncode or 0)

    complete_job_record(manifest, status_path, record, exit_code, started, lock=lock)
    if exit_code == 0:
        print(f"DONE {name.upper()}/{scenario} attempt={attempt}", flush=True)
    else:
        print(f"FAIL {name.upper()}/{scenario} attempt={attempt} exit={exit_code} stderr={err_path}", flush=True)
    return exit_code


def run_job_with_retry(
    *,
    root: Path,
    manifest: Dict[str, object],
    status_path: Path,
    job: Mapping[str, object],
    output_root: Path,
    log_dir: Path,
    args: argparse.Namespace,
    lock: Optional[threading.Lock] = None,
    live_monitor: Optional[bool] = None,
) -> int:
    exit_code = run_one_job(
        root=root,
        manifest=manifest,
        status_path=status_path,
        job=job,
        output_root=output_root,
        log_dir=log_dir,
        args=args,
        attempt=0,
        lock=lock,
        live_monitor=live_monitor,
    )
    if exit_code == 0:
        return 0

    # Find the record for this job (it's the last one we appended for this name/scenario)
    name, scenario = str(job["name"]), str(job["scenario"])
    last_record = next(
        (r for r in reversed(manifest.get("jobs", []))
         if r.get("name") == name and r.get("scenario") == scenario),
        {},
    )
    log_path = Path(str(last_record.get("log") or ""))
    err_path = Path(str(last_record.get("stderr_log") or ""))
    # exit=-9 is SIGKILL from the Linux OOM killer (system RAM exhausted).
    # is_oom_failure() checks log text for CUDA OOM strings, which a SIGKILL
    # never writes.  Treat either condition as an OOM worth retrying.
    _sigkill = (exit_code == -9)
    if not args.oom_retry or (not _sigkill and not is_oom_failure(log_path, err_path)):
        return exit_code

    retry_job = make_oom_retry_job(job)
    if retry_job is None:
        return exit_code

    _oom_reason = "SIGKILL (system RAM OOM)" if _sigkill else "CUDA OOM"
    print(f"[launcher] {_oom_reason} detected for {job['name'].upper()}/{job['scenario']}; retrying with conservative settings.", flush=True)
    return run_one_job(
        root=root,
        manifest=manifest,
        status_path=status_path,
        job=retry_job,
        output_root=output_root,
        log_dir=log_dir,
        args=args,
        attempt=1,
        lock=lock,
        live_monitor=live_monitor,
    )


def run_parallel_jobs(
    *,
    root: Path,
    manifest: Dict[str, object],
    status_path: Path,
    jobs: List[Dict[str, object]],
    output_root: Path,
    log_dir: Path,
    args: argparse.Namespace,
    max_parallel_override: Optional[int] = None,
) -> int:
    """Run jobs with a bounded thread pool. Returns 0 only if all jobs succeed."""
    lock = threading.Lock()
    failures: List[str] = []
    max_workers = max_parallel_override if max_parallel_override is not None else args.max_parallel

    def _run(job: Mapping[str, object]) -> int:
        return run_job_with_retry(
            root=root,
            manifest=manifest,
            status_path=status_path,
            job=job,
            output_root=output_root,
            log_dir=log_dir,
            args=args,
            lock=lock,
            live_monitor=False,  # outer cell monitor handles display in parallel mode
        )

    print(
        f"[launcher] Starting {len(jobs)} jobs with max_parallel={max_workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {pool.submit(_run, job): job for job in jobs}
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            label = f"{job['name'].upper()}/{job['scenario']}"
            try:
                rc = future.result()
            except Exception as exc:
                rc = 1
                print(f"[launcher] EXCEPTION {label}: {exc}", flush=True)
            if rc != 0:
                failures.append(label)
                print(f"[launcher] FAILED {label} (exit={rc})", flush=True)
            else:
                print(f"[launcher] COMPLETED {label}", flush=True)

    if failures:
        print(f"[launcher] {len(failures)} job(s) failed: {', '.join(failures)}", flush=True)
        return 1
    return 0


def _patch_torch_threads(jobs: List[Dict[str, object]], threads: int) -> List[Dict[str, object]]:
    """Return a copy of jobs with --torch-threads replaced by threads."""
    patched = []
    for job in jobs:
        new_args = replace_arg([str(a) for a in job["args"]], "--torch-threads", str(threads))
        patched.append({**job, "args": new_args})
    return patched


def run_algo_sequential_jobs(
    *,
    root: Path,
    manifest: Dict[str, object],
    status_path: Path,
    jobs: List[Dict[str, object]],
    output_root: Path,
    log_dir: Path,
    args: argparse.Namespace,
) -> int:
    """Run algorithms sequentially, each algorithm's scenarios in parallel.

    Execution order: HAPPO×3 → MATD3×3 → MAAC×3 → MASAC×3.

    Motivation
    ----------
    Running all 12 jobs simultaneously on an A100 causes:
    - CPU contention: 12 Python processes share ~12 vCPUs → each env step
      gets ~1 vCPU → ~2 FPS (half of the natural ~4-5 FPS per process).
    - GPU saturation: 3 MASAC jobs × 20+ GiB = 60 GiB of the 80 GiB A100,
      throttling GPU bandwidth for HAPPO/MATD3/MAAC during their updates.

    With algo_sequential:
    - 3 processes share CPU (4× more vCPU per job) → ~4-5 FPS per job.
    - Each algorithm gets dedicated GPU bandwidth.
    - HAPPO/MATD3/MAAC finish ~2× faster; MASAC still limited by QMIX but
      free of GPU competition.
    - --algo-sequential-torch-threads (default 4) replaces --torch-threads
      during each phase, matching the available vCPUs per process.
    """
    # Execution order: lightest algorithms first, MASAC last (heaviest/QMIX).
    # ALGORITHMS constant preserves the original definition order; here we
    # always run MASAC as the final phase regardless of its position in ALGORITHMS.
    _PHASE_ORDER = tuple(a for a in ALGORITHMS if a != "masac") + ("masac",)

    algo_groups: Dict[str, List[Dict[str, object]]] = {algo: [] for algo in _PHASE_ORDER}
    for job in jobs:
        name = str(job["name"])
        if name in algo_groups:
            algo_groups[name].append(job)

    phase_threads = int(args.algo_sequential_torch_threads)
    start_idx = _PHASE_ORDER.index(args.start_from_algorithm)
    overall_rc = 0

    for phase_num, algo in enumerate(_PHASE_ORDER):
        if _PHASE_ORDER.index(algo) < start_idx:
            continue
        group = algo_groups.get(algo, [])
        if not group:
            continue

        print(
            f"\n[launcher] ═══ PHASE {phase_num + 1}/4: {algo.upper()} "
            f"({len(group)} scenario(s) × {phase_threads} torch-threads) ═══",
            flush=True,
        )

        # Give each process more torch threads since only 3 compete.
        phase_jobs = _patch_torch_threads(group, phase_threads)

        # MASAC: FS = GPU VRAM (GpuBackedNdArray float32, ~10.28 GiB/job).
        # In algo_sequential mode, only 3 MASAC jobs share the A100 → no Phase-1 overlap.
        # GPU: 3 × (10.28 buffer + 0.9 model) ≈ 33.5 GiB / 80 GiB = 42%. Safe.
        # Job args already set preload-batch-device=cuda; patch cuda_fraction here.
        if algo == "masac":
            phase_jobs = [
                {**job, "args": replace_arg(
                    [str(a) for a in job["args"]],
                    "--cuda-memory-fraction", "0.30",  # 24 GiB/job, covers buffer+model
                )}
                for job in phase_jobs
            ]

        rc = run_parallel_jobs(
            root=root,
            manifest=manifest,
            status_path=status_path,
            jobs=phase_jobs,
            output_root=output_root,
            log_dir=log_dir,
            args=args,
            max_parallel_override=len(phase_jobs),
        )
        if rc != 0:
            overall_rc = rc
            print(
                f"[launcher] Phase {algo.upper()} had failures — continuing with next phase.",
                flush=True,
            )

    return overall_rc


def run_two_phase_jobs(
    *,
    root: Path,
    manifest: Dict[str, object],
    status_path: Path,
    jobs: List[Dict[str, object]],
    output_root: Path,
    log_dir: Path,
    args: argparse.Namespace,
) -> int:
    """Two-phase execution optimised for NVIDIA A100-SXM4-80GB (80 GiB VRAM, 167.1 GiB RAM).

    Instance spec: a2-ultragpu-1g — 1× A100-SXM4-80GB, 12 vCPUs, 170 GiB RAM.

    Phase 1 — HAPPO×3 + MATD3×3 + MAAC×3  (9 jobs in parallel)
    ─────────────────────────────────────────────────────────────
    Hardware budget:
      GPU: ~11.6 GiB / 80 GiB = 14.6%  (HAPPO~1.05 + MATD3~2.14 + MAAC~0.69 per job × 3)
      GPU free: ~68.4 GiB → zero contention, each algo gets full bandwidth for updates.
      RAM: ~11.5 GiB per process × 9 jobs ≈ 104 GiB (MASAC buffer on GPU → no extra CPU RAM).
    CPU:  12 vCPUs / 9 jobs = 1.33 vCPU/job.
    Torch threads: --two-phase-light-torch-threads (default 1).
      Rationale: env sim is single-threaded Python (GIL). With 9 processes on 12 vCPUs
      each process gets ~1.33 dedicated core. Setting torch-threads=1 avoids Torch
      competing with the env sim main thread on the same vCPU.
    OMP_NUM_THREADS=1 per job (env_overrides): prevents numpy from spawning OMP
      thread pools (default=cpu_count). Without this: 9 × 4 OMP threads = 36 threads
      thrashing on 12 vCPUs → env sim stalls. With OMP=1: 9 threads on 12 cores →
      each has near-dedicated CPU time.
    Expected FPS: 3-5 FPS/job  (after OMP fix; was 2 FPS with default OMP settings).
    Expected time: 30-50 min/episode.

    Phase 2 — MASAC×3  (3 jobs, dedicated A100-SXM4-80GB)
    ───────────────────────────────────────────────────────
    Hardware budget (A100-SXM4-80GB + 167.1 GiB RAM, a2-ultragpu-1g spec):
      GPU: 3 × (10.28 GiB buffer + 0.9 GiB model) = ~33.5 GiB / 80 GiB = 42%.
      GPU cap: cuda-fraction=0.26 → 20.8 GiB/process; covers buffer (10.28) + model + overhead.
      RAM: ~6 GiB / 167.1 GiB = 3.6% (MASAC buffer on GPU VRAM, not RAM).
    CPU: 12 vCPUs / 3 jobs = 4 vCPU/job → 4× more scheduling time than 12-parallel.
    Torch threads: --two-phase-heavy-torch-threads (default 4, matches 4 vCPU/job).
      QMIX batch matrix ops benefit from 4 intra-op threads.
    cuda-memory-fraction: --two-phase-masac-cuda-fraction (default 0.26).
      Rationale: with 3 MASAC processes and fraction=0.92 (default), each process
      thinks it can cache up to 73.6 GiB → 3 × 73.6 GiB race on 80 GiB GPU → OOM.
      Capping at 0.26 × 80 = 20.8 GiB/process → 3 × 20.8 = 62.4 GiB total → safe.
    Replay buffer: FS=GPU VRAM (GpuBackedNdArray float32, 15 ep = 10.28 GiB/job).
      Libera 41 GiB de RAM (era float64 en RAM). preload=cuda: zero-copy desde GPU.
    Expected FPS: 4-5 FPS/job (4 vCPU/job → consistent OS scheduling).
    Expected time: 30-37 min/episode (vs 73 min in 12-parallel baseline).

    Note on FPS ceiling (a2-ultragpu-1g: 12 vCPUs)
    ────────────────────────────────────────────────
    The CityLearn env simulation (17 buildings + 42 EV chargers) is GIL-bound: each
    process uses exactly 1 Python thread for env.step(). FPS is determined by how
    consistently that thread gets CPU time, not by GPU usage.

    With 12 vCPUs:
      Phase 1 (9 jobs):  12/9 = 1.33 vCPU/job → 3-5 FPS/job  (after OMP fix)
      Phase 2 (3 MASAC): 12/3 = 4.0  vCPU/job → 4-7 FPS/job  (dedicated scheduling)
      1 job alone:       12/1 = 12   vCPU/job → 8-12 FPS/job (if env natively allows)

    The two-phase split gives MASAC the most vCPU/job (4×) while running 3 scenarios
    in parallel — the best achievable balance on a 12-vCPU A100 instance.
    """
    phase1_jobs = [j for j in jobs if j["name"] in TWO_PHASE_LIGHT]
    phase2_jobs = [j for j in jobs if j["name"] in TWO_PHASE_HEAVY]

    light_threads = int(args.two_phase_light_torch_threads)
    heavy_threads = int(args.two_phase_heavy_torch_threads)
    masac_cuda_fraction = float(args.two_phase_masac_cuda_fraction)

    # Detect real vCPU count so FPS expectations are accurate.
    try:
        import os as _os
        vcpu_count = len(_os.sched_getaffinity(0))  # Linux/Colab: actual affinity
    except AttributeError:
        vcpu_count = _os.cpu_count() or 12          # fallback (Windows/macOS)

    vcpu_per_p1 = vcpu_count / max(len(phase1_jobs), 1)
    vcpu_per_p2 = vcpu_count / max(len(phase2_jobs), 1)

    # Thread + allocator environment applied to every subprocess:
    # OMP_NUM_THREADS=1: prevents numpy/MKL/OpenBLAS from spawning thread pools.
    #   Without this: 9 processes × 4 default OMP threads = 36 threads thrashing
    #   on 12 vCPUs. With =1: 9 threads on 12 cores → near-dedicated vCPU per job.
    # MALLOC_ARENA_MAX=2: glibc defaults to 8×cpu_count arenas → 96 arenas, each
    #   with its own lock. With 167.1 GiB RAM and 9 concurrent processes, lock
    #   contention on malloc/free adds latency to Python's memory allocator.
    #   Capping at 2 arenas: 2 arenas for all 9 processes → minimal lock wait.
    _perf_env = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MALLOC_ARENA_MAX": "2",
    }

    # ── PHASE 1: HAPPO + MATD3 + MAAC (9 jobs simultaneous) ─────────────────
    algo_labels = " + ".join(a.upper() for a in TWO_PHASE_LIGHT)
    print(
        f"\n[launcher] ═══ PHASE 1/2 ({algo_labels}): "
        f"{len(phase1_jobs)} jobs x {light_threads} torch-thread(s) "
        f"| GPU ~11.6 GiB / 80 GiB "
        f"| vCPU {vcpu_count}/{len(phase1_jobs)} = {vcpu_per_p1:.1f}/job "
        f"| OMP=1 MALLOC_ARENA=2 ═══",
        flush=True,
    )
    p1_jobs = _patch_torch_threads(phase1_jobs, light_threads)
    p1_jobs = [{**job, "env_overrides": _perf_env} for job in p1_jobs]
    overall_rc = run_parallel_jobs(
        root=root,
        manifest=manifest,
        status_path=status_path,
        jobs=p1_jobs,
        output_root=output_root,
        log_dir=log_dir,
        args=args,
        max_parallel_override=len(p1_jobs),
    )
    if overall_rc != 0:
        print("[launcher] Phase 1 had failures — proceeding to Phase 2.", flush=True)

    # ── PHASE 2: MASAC (3 jobs, dedicated A100) ─────────────────────────────
    gpu_cap_gib = masac_cuda_fraction * 80.0
    print(
        f"\n[launcher] ═══ PHASE 2/2 (MASAC): "
        f"{len(phase2_jobs)} jobs x {heavy_threads} torch-threads "
        f"| GPU ~{len(phase2_jobs)*gpu_cap_gib:.0f} GiB cap / 80 GiB "
        f"| vCPU {vcpu_count}/{len(phase2_jobs)} = {vcpu_per_p2:.1f}/job "
        f"| replay-buffer=GPU-VRAM (GpuBackedNdArray float32) | OMP=1 MALLOC_ARENA=2 ═══",
        flush=True,
    )
    p2_jobs = _patch_torch_threads(phase2_jobs, heavy_threads)
    for patch_flag, patch_val in [
        # Cap GPU allocation per process to prevent 3-way caching OOM.
        # Job args already set preload-batch-device=cuda (GPU VRAM FS for MASAC).
        ("--cuda-memory-fraction", str(masac_cuda_fraction)),
    ]:
        p2_jobs = [
            {**job, "args": replace_arg([str(a) for a in job["args"]], patch_flag, patch_val)}
            for job in p2_jobs
        ]
    # Same thread + allocator environment: prevent numpy from stealing vCPUs across MASAC processes.
    p2_jobs = [{**job, "env_overrides": _perf_env} for job in p2_jobs]
    rc2 = run_parallel_jobs(
        root=root,
        manifest=manifest,
        status_path=status_path,
        jobs=p2_jobs,
        output_root=output_root,
        log_dir=log_dir,
        args=args,
        max_parallel_override=len(p2_jobs),
    )
    if rc2 != 0:
        overall_rc = rc2

    return overall_rc


def run_two_phase_concurrent_jobs(
    *,
    root: Path,
    manifest: Dict[str, object],
    status_path: Path,
    jobs: List[Dict[str, object]],
    output_root: Path,
    log_dir: Path,
    args: argparse.Namespace,
) -> int:
    """Run all 12 jobs simultaneously: HAPPO+MATD3+MAAC+MASAC concurrently.

    Optimised for A100-SXM4-80GB (80 GiB VRAM, 167.1 GiB RAM).
    All 12 jobs share the GPU and 12 vCPUs equally (1 vCPU/job).

    GPU budget (MASAC replay buffer on GPU):
      HAPPO×3 + MATD3×3 + MAAC×3: ~14 GiB total (observed ~1.55 GiB/job)
      MASAC×3 buffer: 3 × 13.7 GiB = 41.1 GiB  (buffer_size=10 ep × 1.37 GiB/ep)
      MASAC×3 model+cuda: 3 × 0.22 × 80 GiB reserved = 52.8 GiB
      Actual GPU usage: ~14 + 41.1 + 1.5 (model) ≈ 56.6 GiB / 80 GiB = 70.8%

    System RAM budget (buffer on GPU frees 41 GiB from CPU RAM):
      12 processes × ~11.55 GiB/process = ~138.6 GiB
      + launcher (~1 GiB) + OS (~8 GiB) = ~147.6 GiB / 167.1 GiB = 88.3%
      Peak during MATD3 loading (120s stagger): ~155 GiB (93%) — safe.
      Without GPU offload: 12 × 11.55 + 41.1 buffer = ~180 GiB → OOM (observed).

    CPU: 12 jobs on 12 vCPUs = 1 vCPU/job → ~2-2.5 FPS/job for env sim.

    Wall clock comparison (50 episodes, A100-SXM4-80GB):
      two_phase sequential : Phase1 ~40h + Phase2 ~27h = ~67h
      two_phase_concurrent : max(~54h, ~54h)          = ~54h  (saves ~13h, 19%)

    Parameters controlled by --two-phase-masac-cuda-fraction (default 0.26
    in launcher; actual MASAC fraction is max(provided, 0.25) to cover 15-episode
    GPU buffer at float32 = 10.28 GiB + model 0.9 GiB = 11.18 GiB per process).
    """
    phase1_jobs = [j for j in jobs if j["name"] in TWO_PHASE_LIGHT]
    phase2_jobs = [j for j in jobs if j["name"] in TWO_PHASE_HEAVY]

    masac_cuda_fraction = float(args.two_phase_masac_cuda_fraction)

    try:
        import os as _os
        vcpu_count = len(_os.sched_getaffinity(0))
    except AttributeError:
        vcpu_count = _os.cpu_count() or 12

    _perf_env = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MALLOC_ARENA_MAX": "2",
    }

    # All Phase 1 jobs: torch_threads=1 (1 vCPU per job, GIL-bound env sim)
    p1_jobs = _patch_torch_threads(phase1_jobs, 1)
    p1_jobs = [{**job, "env_overrides": _perf_env} for job in p1_jobs]

    # Resource allocation strategy:
    #
    # System RAM (167 GiB):
    #   - 9 base jobs (HAPPO×3 + MAAC×3 + MATD3×3): ~9 GiB/job × 9 = ~81 GiB
    #   - 3 MASAC jobs (model+overhead, buffer on GPU):   ~2 GiB × 3 = ~6 GiB
    #   - MATD3 MlpPolicyBuffer (~2.4 GiB/job) → SSD     freed ~7.2 GiB
    #   Total system RAM: ~87 GiB / 167 GiB = 52%   ← safe, was 165 GiB (99%)
    #
    # GPU VRAM (80 GiB):
    #   - 3 MASAC: buffer float32 (15 ep) 10.28 GiB + model 0.9 GiB = ~33.5 GiB
    #   - 3 MATD3: model ~3.5 GiB × 3 (hidden=512) = ~10.5 GiB
    #   - 3 MAAC:  model ~3.5 GiB × 3 (hidden=512) = ~10.5 GiB
    #   - 3 HAPPO: model ~0.5 GiB × 3 = ~1.5 GiB
    #   Total GPU: ~56 GiB / 80 GiB = 70%
    #
    # Disk SSD (235+ GiB local Colab):
    #   - 3 MATD3 MlpPolicyBuffer memmap: ~4.8 GiB × 3 (400K buffer) = ~14.4 GiB
    #
    # cuda_frac=0.25: 0.25×80=20 GiB/process; covers 10.28 GiB buffer + 0.9 GiB model.
    masac_gpu_frac = max(masac_cuda_fraction, 0.25)
    p2_jobs = _patch_torch_threads(phase2_jobs, 1)
    for patch_flag, patch_val in [
        ("--masac-preload-batch-device", "cuda"),   # buffer on GPU VRAM
        ("--cuda-memory-fraction", str(masac_gpu_frac)),
    ]:
        p2_jobs = [
            {**job, "args": replace_arg([str(a) for a in job["args"]], patch_flag, patch_val)}
            for job in p2_jobs
        ]
    p2_jobs = [{**job, "env_overrides": _perf_env} for job in p2_jobs]

    # Stagger MATD3 starts: E1 at 600s, E2 at 730s, E3 at 860s.
    # Previous 300/420/540s delays: E1+E2 still get OOM-killed (exit=-9) while
    # E3 at 540s survived.  Root cause: at t<540s the 9 concurrent processes
    # (HAPPO×3 + MASAC×3 + MAAC×3) still have transient peak RAM from dataset
    # loading and initial gradient allocation that, combined with the new MATD3
    # env load (~10-12 GiB), pushes past the 167 GiB Colab limit.
    # Starting all MATD3 jobs after 600s gives the other processes time to
    # reach steady state.  130s inter-job gap (vs 120s) adds a safety margin
    # between concurrent MATD3 env loads.
    _matd3_delay = 600
    staggered = []
    for job in (p1_jobs + p2_jobs):
        if job["name"] == "matd3":
            job = {**job, "startup_delay_seconds": _matd3_delay}
            _matd3_delay += 130
        staggered.append(job)
    all_jobs = staggered

    n_total = len(all_jobs)
    vcpu_ratio = vcpu_count / max(n_total, 1)
    masac_buf_gpu_gib = len(phase2_jobs) * 6.85   # float32 on GPU (was 13.7 GiB float64 in RAM)
    matd3_buf_disk_gib = sum(
        1 for j in all_jobs if j.get("name") == "matd3"
    ) * 2.4   # MlpPolicyBuffer on SSD

    print(
        f"\n[launcher] ═══ CONCURRENT (HAPPO+MATD3+MAAC+MASAC): "
        f"{n_total} jobs × 1 torch-thread "
        f"| sys-RAM ~87 GiB / 167 GiB (52%)"
        f" | GPU ~37 GiB / 80 GiB (46%)"
        f" | SSD ~{matd3_buf_disk_gib:.0f} GiB (MATD3 buf) "
        f"| MASAC buf={masac_buf_gpu_gib:.0f} GiB on GPU, MATD3 buf={matd3_buf_disk_gib:.0f} GiB on SSD "
        f"| MATD3 stagger=600s/E1+130s/job (steady-state RAM after concurrent init) ═══",
        flush=True,
    )

    return run_parallel_jobs(
        root=root,
        manifest=manifest,
        status_path=status_path,
        jobs=all_jobs,
        output_root=output_root,
        log_dir=log_dir,
        args=args,
        max_parallel_override=n_total,
    )


def make_manifest(
    *,
    args: argparse.Namespace,
    root: Path,
    output_root: Path,
    schema_arg: str,
    schema_resolved: Path,
    env_info: Mapping[str, object],
    gpu_info: Mapping[str, object],
    torch_info: Mapping[str, object],
    import_info: Optional[Mapping[str, object]],
) -> Dict[str, object]:
    scenarios = scenario_list(args.scenario)
    return {
        "started_at": utc_now(),
        "completed_at": None,
        "status": "running",
        "dataset": schema_resolved.parent.name,
        "schema_path": schema_arg,
        "schema_path_resolved": str(schema_resolved),
        "scenario": args.scenario,
        "scenarios": scenarios,
        "seed": args.seed,
        "episode_time_steps": args.episode_time_steps,
        "episodes": args.episodes,
        "num_env_steps": args.episode_time_steps * args.episodes,
        "torch": torch_info.get("torch_version"),
        "cuda": bool(args.cuda),
        "algorithm_family": "MADRL",
        "execution": args.execution_mode,
        "parallelization": {
            "execution_mode": args.execution_mode,
            "max_parallel": args.max_parallel,
            "two_phase_light_torch_threads": args.two_phase_light_torch_threads,
            "two_phase_heavy_torch_threads": args.two_phase_heavy_torch_threads,
            "two_phase_masac_cuda_fraction": args.two_phase_masac_cuda_fraction,
            "algo_sequential_torch_threads": args.algo_sequential_torch_threads,
            "reason": (
                "two_phase: Phase1=HAPPO+MATD3+MAAC x3 (9 parallel, OMP=1+MALLOC_ARENA=2, ~3-5 FPS/job, ~11 GiB GPU); "
                "Phase2=MASAC x3 (dedicated A100, 4 vCPU/job, OMP=1, ~4-7 FPS/job, replay-buffer on CPU)."
                if args.execution_mode == "two_phase"
                else (
                    "two_phase_concurrent: all 12 jobs simultaneously (MASAC overlaps Phase1), "
                    "MASAC cuda_frac=0.18 (14.4 GiB×3=43.2 GiB + Phase1 14 GiB = 57 GiB / 80 GiB), "
                    "1 vCPU/job ~2-2.5 FPS, saves ~13h vs sequential two_phase."
                    if args.execution_mode == "two_phase_concurrent"
                    else (
                        "algo_sequential: HAPPO→MATD3→MAAC→MASAC one group at a time (3 parallel per phase)."
                        if args.execution_mode == "algo_sequential"
                        else f"parallel_all: all 12 jobs simultaneously (~2 FPS/job, high GPU contention)."
                    )
                )
            ),
        },
        "active_project_environment": dict(env_info),
        "gpu_optimization": {
            "enabled": bool(args.cuda),
            "profile": args.gpu_profile,
            "require_a100": bool(args.require_a100),
            "gpu": dict(gpu_info),
            "torch_runtime": dict(torch_info),
            "cuda_memory_fraction_requested": args.cuda_memory_fraction,
            "live_progress_interval": args.live_progress_interval,
            "strategy": "A100 sequential MADRL jobs, TF32-enabled Torch runtime, resumable artifacts, OOM retry fallback.",
        },
        "algorithm_resource_limits": {
            "happo_hidden_size": args.happo_hidden_size,
            "masac_max_replay_buffer_gib": args.masac_max_replay_buffer_gib,
            "masac_buffer_size": args.masac_buffer_size,
            "masac_critic_batch_size": args.masac_critic_batch_size,
            "masac_preload_batch_device": args.masac_preload_batch_device,
            "matd3_batch_size": args.matd3_batch_size,
            "matd3_buffer_size": args.matd3_buffer_size,
            "maac_batch_size": args.maac_batch_size,
            "maac_buffer_length": args.maac_buffer_length,
            "oom_retry": bool(args.oom_retry),
            "citylearn_v2_benchmarks": list(CITYLEARN_V2_BENCHMARKS),
        },
        "artifact_optimization": {
            "profile": args.artifact_profile,
            "trace_record_interval": args.trace_record_interval,
            "trace_detail": args.trace_detail,
            "note": "efficient/minimal reduce per-agent trace generation and duplicate CSV mirrors.",
        },
        "reward": {
            "function": "citylearn.reward_function.CityLearnV3MADRLRewardFunction",
            "aggregation": "team_mean",
            "axes": ["OE1_flexibility", "OE2_carbon", "OE3_cost"],
        },
        "training_config": {
            "episodes_required": 50,
            "episode_time_steps_required": 8760,
            "a100_ready": bool(args.episodes >= 50 and args.episode_time_steps == 8760),
            "smoke_imports": dict(import_info or {}),
        },
        "output_root": path_for_status(root, output_root),
        "start_from_algorithm": args.start_from_algorithm,
        "citylearn_v2_benchmarks": list(CITYLEARN_V2_BENCHMARKS),
        "algorithm_order": list(ALGORITHMS),
        "references": REFERENCE_SOURCES,
        "jobs": [],
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="ALL")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--episode-time-steps", default=8760, type=int)
    parser.add_argument("--episodes", default=50, type=int)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--schema-path", default=DEFAULT_SCHEMA)
    parser.add_argument("--torch-threads", default=2, type=int)
    parser.add_argument("--live-progress-interval", default=1000, type=int)
    parser.add_argument("--live-heartbeat-seconds", default=30, type=int)
    parser.add_argument("--artifact-profile", default="full", choices=("full", "efficient", "minimal"))
    parser.add_argument("--trace-record-interval", default=24, type=int)
    parser.add_argument("--trace-detail", default="compact", choices=("full", "compact"))
    parser.add_argument("--gpu-profile", default="aws", choices=("auto", "local4060_fast", "local4060", "balanced", "conservative", "aws"))
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-memory-fraction", default=0.92, type=float)
    parser.add_argument("--require-a100", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-gpu-preflight", action="store_true")
    parser.add_argument("--smoke-imports", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--start-from-algorithm", default="happo", choices=ALGORITHMS)
    parser.add_argument("--live-monitor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--monitor-interval", default=30, type=int)
    parser.add_argument("--log-tail", default=12, type=int)
    parser.add_argument("--oom-retry", action=argparse.BooleanOptionalAction, default=True)

    # ── Per-algorithm hyperparameters (A100-SXM4 80 GiB / 167 GiB RAM / 235 GB SSD) ──
    #
    # HAPPO  — on-policy PPO  | FS: RAM   (rollout ~26 MB, no migration)
    parser.add_argument("--happo-hidden-size", default=512, type=int)  # was 384
    #
    # MASAC  — off-policy SAC+QMIX | FS: GPU VRAM (GpuBackedNdArray float32)
    #   buffer_size=15 ep → float32: 15/10×6.85 GiB = 10.28 GiB/job × 3 = 30.8 GiB GPU
    #   + models ~2.7 GiB → total MASAC GPU ≈ 33.5 GiB (< 80 GiB, cuda_frac=0.25)
    #   Larger critic_batch (512) + train_steps (2) + actor_samples (10) exploit
    #   zero-copy reads from the GPU buffer for maximum GPU utilization.
    parser.add_argument("--masac-max-replay-buffer-gib", default=20.0, type=float)
    parser.add_argument("--masac-buffer-size", default=15, type=int)           # was 10
    parser.add_argument("--masac-critic-batch-size", default=512, type=int)    # was 64
    parser.add_argument("--masac-critic-train-steps", default=2, type=int)     # was 1
    parser.add_argument("--masac-actor-sample-times", default=10, type=int)    # was 5
    parser.add_argument("--masac-rnn-hidden-dim", default=128, type=int)       # balanced (256 uses more VRAM with 3 concurrent)
    parser.add_argument("--masac-qmix-hidden-dim", default=128, type=int)      # was 128
    parser.add_argument("--masac-hyper-hidden-dim", default=256, type=int)     # was 256
    parser.add_argument(
        "--masac-preload-batch-device", default="cuda", choices=("auto", "cuda", "cpu"),
        help="FS strategy para buffer MASAC: cuda=GPU VRAM (GpuBackedNdArray float32, default), "
             "cpu=RAM sistema (compatibilidad), auto=decide segun --masac-preload-batch-device.",
    )
    #
    # MATD3  — off-policy TD3  | FS: SSD local /content/ (DiskBackedNdArray memmap)
    #   buffer_size=400K → 4.8 GiB on SSD × 3 = 14.4 GiB (SSD 235 GB, no RAM cost)
    #   batch_size=4096 → larger batches exploit A100 throughput; OS page-cache keeps
    #   hot transitions in RAM after first access (near-DRAM speed in steady-state).
    #   hidden_size=512 → larger actor/critic network for 17-agent 748-dim shared obs.
    #   train_interval=50 → 2× more frequent updates = better GPU utilization.
    parser.add_argument("--matd3-batch-size", default=4096, type=int)          # was 512
    parser.add_argument("--matd3-buffer-size", default=400000, type=int)       # was 6000
    parser.add_argument("--matd3-hidden-size", default=512, type=int)          # was 256
    parser.add_argument("--matd3-lr", default="3e-4", type=str)
    parser.add_argument("--matd3-train-interval", default=50, type=int)        # was 100
    parser.add_argument("--matd3-num-random-episodes", default=1, type=int,
                        help="Random warmup episodes before MATD3 training (1 ep = 8760 steps ≈ 40 min at 3 FPS).")
    #
    # MAAC   — off-policy Attention SAC | FS: RAM (np.roll internal, ~600 MB)
    #   batch_size=1024 → multi-head attention benefits from larger batches on A100.
    #   hidden_size=512 → 512/attend_heads=8 → 64 keys/head (standard for 17 agents).
    #   steps_per_update=100 → update every 100 env steps (was 250), more GPU use.
    #   q_lr=5e-4 → reduced from 1e-3 for stability with larger batch + hidden.
    #   tau=5e-3 → faster target network update with more frequent updates.
    parser.add_argument("--maac-batch-size", default=1024, type=int)           # was 512
    parser.add_argument("--maac-buffer-length", default=100000, type=int)
    parser.add_argument("--maac-hidden-size", default=512, type=int)           # was 256
    parser.add_argument("--maac-attend-heads", default=8, type=int)            # was hardcoded 4
    parser.add_argument("--maac-steps-per-update", default=100, type=int)      # was 250
    parser.add_argument("--maac-num-updates", default=8, type=int)
    parser.add_argument("--maac-q-lr", default="5e-4", type=str)               # was hardcoded 1e-3
    parser.add_argument("--maac-tau", default="5e-3", type=str)                # was hardcoded 1e-3
    parser.add_argument(
        "--max-parallel", default=12, type=int,
        help=(
            "Max jobs to run concurrently in parallel_all mode. "
            "Default 12 = all 4 MADRL x 3 scenarios simultaneously on A100-SXM4-80GB. "
            "Ignored in algo_sequential mode (always 3 per phase)."
        ),
    )
    parser.add_argument(
        "--execution-mode",
        default="two_phase",
        choices=("parallel_all", "two_phase", "two_phase_concurrent", "algo_sequential"),
        help=(
            "two_phase (default): Phase 1 = HAPPO+MATD3+MAAC × 3 scenarios (9 jobs "
            "simultaneous, ~11 GiB GPU, 3 FPS/job); Phase 2 = MASAC × 3 scenarios "
            "(3 jobs, dedicated A100, 4 vCPU/job, 4-5 FPS/job). ~67h total. "
            "two_phase_concurrent: all 12 jobs simultaneously (MASAC overlaps Phase1), "
            "MASAC cuda_fraction=0.18 (set via --two-phase-masac-cuda-fraction), "
            "1 vCPU/job → 2-2.5 FPS, saves ~13h (19%%) vs two_phase. ~54h total. "
            "algo_sequential: HAPPO→MATD3→MAAC→MASAC one group at a time (3 per phase). "
            "parallel_all: all 12 simultaneously, no MASAC GPU cap (risk OOM)."
        ),
    )
    parser.add_argument(
        "--two-phase-light-torch-threads",
        default=1,
        type=int,
        help=(
            "Torch threads per process for Phase 1 (HAPPO/MATD3/MAAC). "
            "9 jobs share 12 vCPUs → 1.3 vCPU/job. Setting torch-threads=1 "
            "avoids Torch competing with the CityLearn env simulation thread. "
            "Default: 1."
        ),
    )
    parser.add_argument(
        "--two-phase-heavy-torch-threads",
        default=4,
        type=int,
        help=(
            "Torch threads per process for Phase 2 (MASAC). "
            "3 jobs share 12 vCPUs → 4 vCPU/job. QMIX benefits from 4 threads "
            "for batch matrix ops. Default: 4."
        ),
    )
    parser.add_argument(
        "--two-phase-masac-cuda-fraction",
        default=0.26,
        type=float,
        help=(
            "CUDA memory fraction per MASAC process in Phase 2 (two_phase mode). "
            "On A100-SXM4-80GB: 0.26 × 80 GiB = 20.8 GiB/process; "
            "3 processes × 20.8 GiB = 62.4 GiB total (17.6 GiB margin on 80 GiB). "
            "Prevents PyTorch caching allocator from racing across 3 processes "
            "when the full 80 GiB appears free. Default: 0.26."
        ),
    )
    parser.add_argument(
        "--algo-sequential-torch-threads",
        default=4,
        type=int,
        help=(
            "Torch threads per process in algo_sequential mode (3 jobs/phase). "
            "With 4 vCPU/job on a 12-vCPU A100 Colab, 4 threads is the natural fit. "
            "Ignored in two_phase and parallel_all modes."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = project_root()
    env_info = configure_environment(root, args)

    output_root = resolve_path(root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    schema_resolved = resolve_path(root, args.schema_path)
    if not schema_resolved.exists():
        raise FileNotFoundError(f"Schema not found: {schema_resolved}")
    schema_arg = path_for_status(root, schema_resolved)

    gpu_info = validate_gpu(args)
    torch_info = validate_torch(args)
    import_info = smoke_imports() if args.smoke_imports else None

    latest_paths = [
        root / "outputs" / "latest_colab_output_root.txt",
        root / "outputs" / "latest_visible_training_output_root.txt",
    ]
    for latest_path in latest_paths:
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.write_text(path_for_status(root, output_root), encoding="utf-8")

    manifest_path = output_root / "official_full_manifest.json"
    status_path = output_root / "official_full_status.json"
    manifest = make_manifest(
        args=args,
        root=root,
        output_root=output_root,
        schema_arg=schema_arg,
        schema_resolved=schema_resolved,
        env_info=env_info,
        gpu_info=gpu_info,
        torch_info=torch_info,
        import_info=import_info,
    )
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(status_path, manifest)

    jobs = build_jobs(args, root, output_root, schema_arg)
    if args.dry_run:
        manifest["jobs"] = []
        for job in jobs:
            name = str(job["name"])
            scenario = str(job["scenario"])
            job_run_dir = run_dir(output_root, name, scenario, args.seed)
            command = command_for_job(job)
            manifest["jobs"].append(
                {
                    "name": name,
                    "scenario": scenario,
                    "script": job["script"],
                    "started_at": None,
                    "completed_at": None,
                    "exit_code": None,
                    "output_dir": path_for_status(root, job_run_dir),
                    "command": " ".join(command),
                    "planned_only": True,
                }
            )
        manifest["status"] = "dry_run"
        manifest["completed_at"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(status_path, manifest)
        print(f"Dry run completed: {status_path}", flush=True)
        return 0

    if args.execution_mode == "two_phase":
        overall_rc = run_two_phase_jobs(
            root=root,
            manifest=manifest,
            status_path=status_path,
            jobs=jobs,
            output_root=output_root,
            log_dir=log_dir,
            args=args,
        )
    elif args.execution_mode == "two_phase_concurrent":
        overall_rc = run_two_phase_concurrent_jobs(
            root=root,
            manifest=manifest,
            status_path=status_path,
            jobs=jobs,
            output_root=output_root,
            log_dir=log_dir,
            args=args,
        )
    elif args.execution_mode == "algo_sequential":
        overall_rc = run_algo_sequential_jobs(
            root=root,
            manifest=manifest,
            status_path=status_path,
            jobs=jobs,
            output_root=output_root,
            log_dir=log_dir,
            args=args,
        )
    elif args.max_parallel > 1:
        overall_rc = run_parallel_jobs(
            root=root,
            manifest=manifest,
            status_path=status_path,
            jobs=jobs,
            output_root=output_root,
            log_dir=log_dir,
            args=args,
        )
    else:
        overall_rc = 0
        for job in jobs:
            exit_code = run_job_with_retry(
                root=root,
                manifest=manifest,
                status_path=status_path,
                job=job,
                output_root=output_root,
                log_dir=log_dir,
                args=args,
            )
            if exit_code != 0:
                overall_rc = exit_code
                break

    manifest["status"] = "completed" if overall_rc == 0 else "failed"
    manifest["completed_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(status_path, manifest)
    if overall_rc == 0:
        print(f"Training chain completed: {status_path}", flush=True)
    else:
        print(f"Training chain FAILED (see above): {status_path}", flush=True)
    return int(overall_rc)


if __name__ == "__main__":
    raise SystemExit(main())
