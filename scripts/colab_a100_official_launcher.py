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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


ALGORITHMS = ("happo", "masac", "matd3", "maac")
CITYLEARN_V2_BENCHMARKS = ("PPO", "SAC", "A2C")
SCENARIOS = ("E1", "E2", "E3")
HEAVY_ALGORITHMS = {"masac", "maac"}

# two_phase_happo_masac: Phase1 HAPPO+MASAC x3, Phase2 MATD3+MAAC x3 (Colab A100 80GB).
TWO_PHASE_P1_HM = ("happo", "masac")
TWO_PHASE_P2_HM = ("matd3", "maac")

_MANIFEST_LOCK = threading.Lock()
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
                    str(args.happo_n_rollout_threads),
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
                    args.masac_preload_batch_device,
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
                    "3e-4",
                    "--max-grad-norm",
                    "1.0",
                    "--gamma",
                    "0.9999",
                    "--train-interval",
                    str(args.matd3_train_interval),
                    "--num-random-episodes",
                    "1",
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
                    "4",
                    "--pi-lr",
                    "3e-4",
                    "--q-lr",
                    "1e-3",
                    "--tau",
                    "1e-3",
                    "--gamma",
                    "0.9999",
                    "--reward-scale",
                    "10.0",
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
        # OOM retry: 50% of A100-80GB defaults
        args = replace_arg(args, "--buffer-size", "20")           # 40 -> 20 episodes
        args = replace_arg(args, "--critic-batch-size", "512")    # 1024 -> 512
        args = replace_arg(args, "--max-replay-buffer-gib", "20") # 40 -> 20 GiB
        args = replace_arg(args, "--masac-rnn-hidden-dim", "512") # 1024 -> 512
        args = replace_arg(args, "--masac-qmix-hidden-dim", "256")# 512 -> 256
        args = replace_arg(args, "--masac-hyper-hidden-dim", "512")# 1024 -> 512
        args = replace_arg(args, "--masac-preload-batch-device", "cpu")
    elif name == "matd3":
        # OOM retry: 50% of A100-80GB defaults
        args = replace_arg(args, "--batch-size", "512")           # 1024 -> 512
        args = replace_arg(args, "--buffer-size", "1000000")      # 2M -> 1M
        args = replace_arg(args, "--hidden-size", "512")          # 1024 -> 512
    elif name == "maac":
        # OOM retry: 50% of A100-80GB defaults
        args = replace_arg(args, "--batch-size", "512")           # 1024 -> 512
        args = replace_arg(args, "--buffer-length", "500000")     # 1M -> 500K
        args = replace_arg(args, "--hidden-size", "512")          # 1024 -> 512
        args = replace_arg(args, "--num-updates", "8")            # 16 -> 8
    elif name == "happo":
        # OOM retry: reduce hidden size
        args = replace_arg(args, "--hidden-size", "512")          # 1024 -> 512
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

    active_jobs = [job for job in status.get("jobs", []) if job.get("completed_at") is None]
    if not active_jobs:
        print(f"[monitor] status={status.get('status')} jobs={len(status.get('jobs', []))}", flush=True)
        return

    active = active_jobs[0]
    run_path = resolve_status_path(root, str(active["output_dir"]))
    progress = read_json(run_path / "live_progress.json")
    print("", flush=True)
    if len(active_jobs) > 1:
        running = ", ".join(f"{j['name'].upper()}/{j['scenario']}" for j in active_jobs)
        print(f"[monitor] parallel active ({len(active_jobs)}): {running}", flush=True)
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
) -> None:
    with _MANIFEST_LOCK:
        manifest.setdefault("jobs", [])
        manifest["jobs"].append(record)
        atomic_write_json(status_path, manifest)


def complete_job_record(
    manifest: Dict[str, object],
    status_path: Path,
    record: Dict[str, object],
    exit_code: int,
    started_time: float,
) -> None:
    with _MANIFEST_LOCK:
        record["completed_at"] = utc_now()
        record["exit_code"] = int(exit_code)
        record["duration_minutes"] = round((time.time() - started_time) / 60.0, 3)
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
) -> int:
    name = str(job["name"])
    scenario = str(job["scenario"])
    job_run_dir = run_dir(output_root, name, scenario, args.seed)
    job_output_dir = path_for_status(root, job_run_dir)

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
        append_job_record(manifest, status_path, record)
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
    append_job_record(manifest, status_path, record)

    delay = float(job.get("startup_delay_seconds") or 0)
    if delay > 0:
        print(f"DELAY {name.upper()}/{scenario} {delay:.0f}s (stagger)", flush=True)
        time.sleep(delay)

    env = os.environ.copy()
    env.update(dict(job.get("env_overrides") or {}))

    print(f"START {name.upper()}/{scenario} attempt={attempt} log={log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as stdout_f, err_path.open("w", encoding="utf-8") as stderr_f:
        proc = subprocess.Popen(
            command,
            cwd=str(root),
            stdout=stdout_f,
            stderr=stderr_f,
            text=True,
            env=env,
        )

        last_monitor = 0.0
        while proc.poll() is None:
            if args.live_monitor and (time.time() - last_monitor) >= max(5, args.monitor_interval):
                print_monitor_snapshot(root, status_path, log_tail=args.log_tail)
                last_monitor = time.time()
            time.sleep(5)

        exit_code = int(proc.returncode or 0)

    complete_job_record(manifest, status_path, record, exit_code, started)
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
    )
    if exit_code == 0:
        return 0

    last_record = manifest["jobs"][-1]
    log_path = Path(str(last_record.get("log") or ""))
    err_path = Path(str(last_record.get("stderr_log") or ""))
    if not args.oom_retry or not is_oom_failure(log_path, err_path):
        return exit_code

    retry_job = make_oom_retry_job(job)
    if retry_job is None:
        return exit_code

    print(f"OOM detected for {job['name'].upper()}/{job['scenario']}; retrying with conservative settings.", flush=True)
    return run_one_job(
        root=root,
        manifest=manifest,
        status_path=status_path,
        job=retry_job,
        output_root=output_root,
        log_dir=log_dir,
        args=args,
        attempt=1,
    )


def _patch_torch_threads(jobs: List[Dict[str, object]], n_threads: int) -> List[Dict[str, object]]:
    patched: List[Dict[str, object]] = []
    for job in jobs:
        args = replace_arg([str(a) for a in job["args"]], "--torch-threads", str(n_threads))
        patched.append({**job, "args": args})
    return patched


def run_parallel_jobs(
    *,
    root: Path,
    manifest: Dict[str, object],
    status_path: Path,
    jobs: List[Dict[str, object]],
    output_root: Path,
    log_dir: Path,
    args: argparse.Namespace,
    max_workers: int,
) -> int:
    """Run jobs concurrently; returns first non-zero exit code (0 if all succeed)."""
    if not jobs:
        return 0
    workers = max(1, min(int(max_workers), len(jobs)))
    overall_rc = 0
    if workers == 1:
        for job in jobs:
            ec = run_job_with_retry(
                root=root, manifest=manifest, status_path=status_path,
                job=job, output_root=output_root, log_dir=log_dir, args=args,
            )
            if ec != 0:
                overall_rc = ec
        return overall_rc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_job_with_retry,
                root=root, manifest=manifest, status_path=status_path,
                job=job, output_root=output_root, log_dir=log_dir, args=args,
            ): job
            for job in jobs
        }
        for fut in as_completed(futures):
            ec = int(fut.result())
            if ec != 0:
                overall_rc = ec
    return overall_rc


def run_two_phase_happo_masac_jobs(
    *,
    root: Path,
    manifest: Dict[str, object],
    status_path: Path,
    jobs: List[Dict[str, object]],
    output_root: Path,
    log_dir: Path,
    args: argparse.Namespace,
) -> int:
    """Phase1 HAPPO+MASAC x3 parallel, then Phase2 MATD3+MAAC x3 parallel (A100 80GB / 167 GiB RAM)."""
    phase1_jobs = [j for j in jobs if j["name"] in TWO_PHASE_P1_HM]
    phase2_jobs = [j for j in jobs if j["name"] in TWO_PHASE_P2_HM]

    phase_threads = int(args.two_phase_torch_threads)
    masac_cuda_fraction = float(args.two_phase_masac_cuda_fraction)

    _perf_env = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MALLOC_ARENA_MAX": "2",
    }

    p1_labels = " + ".join(a.upper() for a in TWO_PHASE_P1_HM)
    print(
        f"\n[launcher] === PHASE 1/2 ({p1_labels}): {len(phase1_jobs)} jobs, "
        f"torch_threads={phase_threads}, MASAC cuda_frac={masac_cuda_fraction} ===",
        flush=True,
    )
    p1_jobs = _patch_torch_threads(phase1_jobs, phase_threads)
    p1_jobs = [{**job, "env_overrides": _perf_env} for job in p1_jobs]
    p1_patched: List[Dict[str, object]] = []
    for job in p1_jobs:
        if job["name"] == "masac":
            patched_args = [str(a) for a in job["args"]]
            for flag, val in [
                ("--masac-preload-batch-device", "cuda"),
                ("--cuda-memory-fraction", str(masac_cuda_fraction)),
            ]:
                patched_args = replace_arg(patched_args, flag, val)
            job = {**job, "args": patched_args}
        p1_patched.append(job)

    overall_rc = run_parallel_jobs(
        root=root, manifest=manifest, status_path=status_path,
        jobs=p1_patched, output_root=output_root, log_dir=log_dir,
        args=args, max_workers=len(p1_patched),
    )
    if overall_rc != 0:
        print("[launcher] Phase 1 had failures — proceeding to Phase 2.", flush=True)

    p2_labels = " + ".join(a.upper() for a in TWO_PHASE_P2_HM)
    print(
        f"\n[launcher] === PHASE 2/2 ({p2_labels}): {len(phase2_jobs)} jobs, "
        f"torch_threads={phase_threads}, MATD3 stagger 600/3600/6600s ===",
        flush=True,
    )
    p2_jobs = _patch_torch_threads(phase2_jobs, phase_threads)
    p2_jobs = [{**job, "env_overrides": _perf_env} for job in p2_jobs]
    _matd3_delay = 600
    p2_staggered: List[Dict[str, object]] = []
    for job in p2_jobs:
        if job["name"] == "matd3":
            job = {**job, "startup_delay_seconds": _matd3_delay}
            _matd3_delay += 3000
        p2_staggered.append(job)

    rc2 = run_parallel_jobs(
        root=root, manifest=manifest, status_path=status_path,
        jobs=p2_staggered, output_root=output_root, log_dir=log_dir,
        args=args, max_workers=len(p2_staggered),
    )
    if rc2 != 0:
        overall_rc = rc2
    return overall_rc


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
        "execution": getattr(args, "execution_mode", "algo_sequential"),
        "parallelization": {
            "execution_mode": getattr(args, "execution_mode", "algo_sequential"),
            "requested": args.parallel_scenarios > 1,
            "effective": args.parallel_scenarios > 1,
            "parallel_scenarios": args.parallel_scenarios,
            "two_phase_torch_threads": getattr(args, "two_phase_torch_threads", 2),
            "two_phase_masac_cuda_fraction": getattr(args, "two_phase_masac_cuda_fraction", 0.26),
            "strategy": (
                "two_phase_happo_masac: Phase1=HAPPO+MASAC x3 (6 parallel); "
                "Phase2=MATD3+MAAC x3 (6 parallel, MATD3 stagger)."
                if getattr(args, "execution_mode", "") == "two_phase_happo_masac"
                else f"Run {args.parallel_scenarios} scenarios per algorithm concurrently; algorithms are sequential."
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
            "strategy": "A100 MADRL jobs with parallel scenarios, TF32-enabled Torch runtime, resumable artifacts, OOM retry fallback.",
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
            "a100_ready": bool(args.episodes == 50 and args.episode_time_steps == 8760),
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
    parser.add_argument("--torch-threads", default=4, type=int)
    parser.add_argument("--parallel-scenarios", default=3, type=int,
                        help="Number of scenarios to run concurrently per algorithm. "
                             "Set to 1 for sequential. A100-80GB: 3 is safe (MASAC uses CPU buffer).")
    parser.add_argument("--live-progress-interval", default=5000, type=int)
    parser.add_argument("--live-heartbeat-seconds", default=120, type=int)
    parser.add_argument("--artifact-profile", default="efficient", choices=("full", "efficient", "minimal"))
    parser.add_argument("--trace-record-interval", default=8760, type=int)
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
    parser.add_argument("--monitor-interval", default=120, type=int)
    parser.add_argument("--log-tail", default=4, type=int)
    parser.add_argument("--oom-retry", action=argparse.BooleanOptionalAction, default=True)

    # ── A100-SXM4-80GB hyperparameters ───────────────────────────────────────
    # VRAM budget (73.6 GiB usable @ 0.92): 3 parallel scenarios per algo.
    # HAPPO x3: lighter [512,512] policy nets target ~11 FPS on A100-class runs.
    # MATD3 x3: ~1.5 GiB each = 4.5 GiB. MAAC x3: ~1.5 GiB each = 4.5 GiB.
    # RAM budget (167 GiB): MASAC buffer 3x40=120 GiB, MATD3 buffer 3x14=42 GiB.
    parser.add_argument("--happo-hidden-size", default=512, type=int,
                        help="A100-80GB HAPPO speed profile: [512,512] targets ~11 FPS with 3 parallel scenarios.")
    parser.add_argument("--happo-n-rollout-threads", default=4, type=int,
                        help="Parallel env subprocesses per HAPPO job (ShareSubprocVecEnv). "
                             "A100-80GB + 3 parallel scenarios: 4 threads x 3 = 12 procs → ~4x FPS. "
                             "Set to 1 for DummyVecEnv (debug).")
    parser.add_argument("--masac-max-replay-buffer-gib", default=40.0, type=float)
    parser.add_argument("--masac-buffer-size", default=40, type=int,
                        help="Episodes in replay buffer. 40 ep = 350400 steps per instance.")
    parser.add_argument("--masac-critic-batch-size", default=1024, type=int,
                        help="A100-80GB: 1024 fills Tensor Cores. Was 512 (bug).")
    parser.add_argument("--masac-critic-train-steps", default=2, type=int,
                        help="A100-80GB: 2 critic steps per env step (GPU is fast).")
    parser.add_argument("--masac-actor-sample-times", default=10, type=int,
                        help="A100-80GB: 10 actor updates per critic step (2x vs 5).")
    parser.add_argument("--masac-rnn-hidden-dim", default=1024, type=int,
                        help="A100-80GB: GRU actor hidden dim 1024 (2x vs 512).")
    parser.add_argument("--masac-qmix-hidden-dim", default=512, type=int,
                        help="A100-80GB: QMIX monotonic mixing 512 (2x vs 256).")
    parser.add_argument("--masac-hyper-hidden-dim", default=1024, type=int,
                        help="A100-80GB: hypernetwork hidden dim 1024 (2x vs 512).")
    parser.add_argument("--masac-preload-batch-device", default="cpu", choices=("auto", "cuda", "cpu"),
                        help="cpu: keeps 3x40=120 GiB buffer in RAM, freeing VRAM for parallel scenarios.")
    parser.add_argument("--matd3-batch-size", default=1024, type=int,
                        help="A100-80GB: Tensor Cores optimal at batch>=512.")
    parser.add_argument("--matd3-buffer-size", default=2000000, type=int,
                        help="A100-80GB: 2M transitions = 228 ep diversity; 3x~14 GiB = 42 GiB RAM.")
    parser.add_argument("--matd3-hidden-size", default=1024, type=int,
                        help="A100-80GB: actor+critic hidden 1024 (4x params vs 512).")
    parser.add_argument("--matd3-train-interval", default=100, type=int)
    parser.add_argument("--maac-batch-size", default=1024, type=int,
                        help="A100-80GB: Tensor Cores optimal at batch>=512.")
    parser.add_argument("--maac-buffer-length", default=1000000, type=int,
                        help="A100-80GB: 1M steps = 2x vs 500K; 3x~7 GiB = 21 GiB RAM.")
    parser.add_argument("--maac-hidden-size", default=1024, type=int,
                        help="A100-80GB: attention critic hidden 1024 (4x params vs 512).")
    parser.add_argument("--maac-steps-per-update", default=100, type=int)
    parser.add_argument("--maac-num-updates", default=16, type=int,
                        help="A100-80GB: 16 gradient steps per update (2x vs 8; GPU is fast).")
    parser.add_argument(
        "--execution-mode",
        default="algo_sequential",
        choices=("algo_sequential", "two_phase_happo_masac"),
        help=(
            "algo_sequential: HAPPO->MASAC->MATD3->MAAC, 3 scenarios in parallel per algorithm. "
            "two_phase_happo_masac: Phase1 HAPPO+MASAC x3, then Phase2 MATD3+MAAC x3 (Colab A100)."
        ),
    )
    parser.add_argument(
        "--two-phase-torch-threads",
        default=2,
        type=int,
        help="Torch threads per job in two_phase_happo_masac (6 jobs / 12 vCPU = 2).",
    )
    parser.add_argument(
        "--two-phase-masac-cuda-fraction",
        default=0.26,
        type=float,
        help="CUDA memory fraction per MASAC process in Phase 1 (0.26 x 80 GiB = 20.8 GiB/job).",
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

    if args.execution_mode == "two_phase_happo_masac":
        overall_rc = run_two_phase_happo_masac_jobs(
            root=root,
            manifest=manifest,
            status_path=status_path,
            jobs=jobs,
            output_root=output_root,
            log_dir=log_dir,
            args=args,
        )
        manifest["status"] = "completed" if overall_rc == 0 else "failed"
        manifest["completed_at"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(status_path, manifest)
        return int(overall_rc)

    # Group jobs by algorithm in canonical order; run N scenarios per algo concurrently.
    algo_groups: Dict[str, list] = defaultdict(list)
    for job in jobs:
        algo_groups[str(job["name"])].append(job)

    max_p = max(1, int(args.parallel_scenarios))
    if max_p > 1:
        print(f"[launcher] parallel_scenarios={max_p} — running {max_p} scenarios concurrently per algorithm.", flush=True)

    def _run_group(algo_jobs: list, n_parallel: int) -> int:
        """Run a list of jobs with up to n_parallel concurrent workers. Returns 0 or first failing exit code."""
        if n_parallel <= 1:
            for job in algo_jobs:
                ec = run_job_with_retry(
                    root=root, manifest=manifest, status_path=status_path,
                    job=job, output_root=output_root, log_dir=log_dir, args=args,
                )
                if ec != 0:
                    return int(ec)
            return 0
        with ThreadPoolExecutor(max_workers=n_parallel) as pool:
            futures = {
                pool.submit(
                    run_job_with_retry,
                    root=root, manifest=manifest, status_path=status_path,
                    job=job, output_root=output_root, log_dir=log_dir, args=args,
                ): job
                for job in algo_jobs
            }
            for fut in as_completed(futures):
                ec = fut.result()
                if ec != 0:
                    return int(ec)
        return 0

    for algo in ALGORITHMS:
        group = algo_groups.get(algo, [])
        if not group:
            continue
        print(f"[launcher] {algo.upper()} — {len(group)} scenario(s) with n_parallel={min(max_p, len(group))}", flush=True)
        exit_code = _run_group(group, min(max_p, len(group)))
        if exit_code != 0:
            manifest["status"] = "failed"
            manifest["completed_at"] = utc_now()
            atomic_write_json(manifest_path, manifest)
            atomic_write_json(status_path, manifest)
            return int(exit_code)

    manifest["status"] = "completed"
    manifest["completed_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(status_path, manifest)
    print(f"Training chain completed: {status_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
