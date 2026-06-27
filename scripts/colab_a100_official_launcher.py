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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


ALGORITHMS = ("happo", "masac", "matd3", "maac")
CITYLEARN_V2_BENCHMARKS = ("PPO", "SAC", "A2C")
SCENARIOS = ("E1", "E2", "E3")
HEAVY_ALGORITHMS = {"masac", "maac"}

# two_phase_happo_masac: Phase1 HAPPO+MASAC×3 (6 parallel) → Phase2 MATD3+MAAC×3 (Colab A100 80GB).
LAUNCHER_PROTOCOL_ID = "two_phase_happo_masac_v3"
TWO_PHASE_P1_HM = ("happo", "masac")
TWO_PHASE_P2_HM = ("matd3", "maac")
TWO_PHASE_ORDER = (TWO_PHASE_P1_HM, TWO_PHASE_P2_HM)
FOUR_PHASE_ALGO_ORDER = ("happo", "masac", "matd3", "maac")  # monitor ETA fallback
# Wall-time priors (min/ep, 6 parallel jobs/fase, A100) for manifest ETA when FPS not yet measured.
EST_MIN_PER_EPISODE_BY_ALGO = {
    "happo": 14.6,   # prior FPS=10 -> 8760/10/60 ~ 14.6 min/ep (on-policy, CPU-bound)
    "masac": 15.0,   # off-policy GPU: 12 ep replay, batch 1024
    "matd3": 12.2,   # prior FPS=12 -> 8760/12/60 ~ 12.2 min/ep (v4-aligned config)
    "maac": 12.2,    # prior FPS=12 -> 8760/12/60 ~ 12.2 min/ep
}
EST_MIN_PER_EPISODE_PHASE = 12.0  # max(HAPPO,MASAC) ≈ 15 but MASAC often bounds phase 1

_MANIFEST_LOCK = threading.Lock()
# Single-owner monitor: with N parallel jobs each running its own proc-wait loop, every
# job used to call print_monitor_snapshot() (a GLOBAL snapshot) every interval, so the
# same dashboard was printed up to N times per interval with lines interleaved across
# threads. This lock + shared timestamp lets exactly one thread print per interval.
_MONITOR_LOCK = threading.Lock()
_LAST_MONITOR_TS = 0.0
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

    # Accept any datacenter GPU with enough VRAM (A100/H100/H200/RTX PRO 6000 Blackwell/etc.).
    # All expose the TF32 / expandable_segments memory profile this launcher uses. The real
    # requirement is VRAM headroom for 6 jobs/phase, not a specific model name.
    _gpu_name = str(gpu.get("name", ""))
    _vram = float(gpu.get("memory_total_gib") or 0.0)
    _known = ("A100", "H100", "H200", "RTX PRO 6000", "BLACKWELL", "A40", "L40")
    _name_known = any(k in _gpu_name.upper() for k in _known)

    if args.require_a100 and _vram < 39.0:
        raise RuntimeError(
            f"GPU preflight expected >=39 GiB VRAM, got {_vram} GiB ({_gpu_name or 'not detected'}). "
            "Select Runtime > Change runtime type > A100 / H100 / RTX PRO 6000."
        )

    if args.require_a100 and not _name_known:
        # Capable VRAM but unrecognized model: allow and log rather than hard-fail.
        print(
            f"[launcher] GPU '{_gpu_name}' not in known list but has {_vram:.0f} GiB VRAM (>=39) -> accepted.",
            flush=True,
        )

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
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        cuda_runtime = str(getattr(torch.version, "cuda", "") or "")
        needs_cu128 = capability[0] >= 12 or any(
            k in device_name.upper() for k in ("BLACKWELL", "RTX PRO 6000", "RTX 50")
        )
        if needs_cu128 and not cuda_runtime.startswith("12.8"):
            raise RuntimeError(
                f"GPU {device_name} (sm_{capability[0]}{capability[1]}) requiere PyTorch cu128. "
                f"Instalado: torch {torch.__version__} CUDA {cuda_runtime}. "
                "Re-ejecuta celda 1.3 del notebook (auto-instala cu128) o usa runtime A100/H100."
            )
        try:
            probe = torch.zeros(1, device="cuda")
            _ = (probe + 1).item()
            torch.cuda.synchronize()
        except Exception as exc:
            raise RuntimeError(
                f"PyTorch no puede ejecutar kernels CUDA en {device_name}: {exc}. "
                "Blackwell (sm_120) necesita: pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cu128"
            ) from exc
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
        "device_capability": list(torch.cuda.get_device_capability(0)) if cuda_available else None,
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "kernel_smoke_ok": bool(cuda_available),
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


def arg_value(args: Sequence[str], flag: str, default: Optional[object] = None) -> Optional[str]:
    items = [str(item) for item in args]
    if flag in items:
        idx = items.index(flag)
        if idx + 1 < len(items):
            return items[idx + 1]
    return None if default is None else str(default)


def output_base(output_root: Path, algorithm: str) -> Path:
    from citylearn_v3_training_common import normalize_algorithm_dir

    return output_root / normalize_algorithm_dir(algorithm)


def run_dir(output_root: Path, algorithm: str, scenario: str, seed: int) -> Path:
    from citylearn_v3_training_common import resolve_job_run_dir

    return resolve_job_run_dir(output_root, algorithm, scenario, seed)


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
                    "--num-mini-batch",
                    str(args.happo_num_mini_batch),
                    "--gpu-rollout-ref",
                    str(args.happo_gpu_rollout_ref),
                    "--ppo-epoch",
                    str(args.happo_ppo_epoch),
                    "--critic-epoch",
                    str(args.happo_critic_epoch),
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


def is_sigkill_exit(exit_code: int) -> bool:
    """True when the child was SIGKILL'd (Linux OOM killer leaves empty logs).

    Common encodings: 137 = 128+9, 247 = (-9) & 0xFF from some wait() implementations.
    """
    code = int(exit_code)
    if code < 0:
        code = code & 0xFF
    return code in (9, 137, 247)


def is_oom_failure(*paths: Path) -> bool:
    needles = (
        "cuda out of memory",
        "torch.outofmemoryerror",
        "outofmemoryerror",
        "cublas_status_alloc_failed",
        "replay buffer estimate is too large",
        "memoryerror",
        "std::bad_alloc",
        "cannot allocate memory",
        "exceeds the memory limit",
        "exceeds allowed",
        "killed",
        "oom",
    )
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
        cur_frac = float(arg_value(args, "--cuda-memory-fraction", "0.14") or 0.14)
        args = replace_arg(args, "--buffer-size", "2")
        args = replace_arg(args, "--critic-batch-size", "1")
        args = replace_arg(args, "--max-replay-buffer-gib", "6.0")
        args = replace_arg(args, "--rnn-hidden-dim", "64")
        args = replace_arg(args, "--qmix-hidden-dim", "32")
        args = replace_arg(args, "--hyper-hidden-dim", "64")
        args = replace_arg(args, "--masac-preload-batch-device", "cpu")
        args = replace_arg(args, "--cuda-memory-fraction", str(min(0.28, round(cur_frac * 1.35, 3))))
    elif name == "matd3":
        cur_frac = float(arg_value(args, "--cuda-memory-fraction", "0.14") or 0.14)
        args = replace_arg(args, "--batch-size", "64")
        args = replace_arg(args, "--buffer-size", "2048")
        args = replace_arg(args, "--hidden-size", "128")
        args = replace_arg(args, "--train-interval", "150")
        args = replace_arg(args, "--cuda-memory-fraction", str(max(0.08, round(cur_frac * 0.7, 3))))
    elif name == "maac":
        args = replace_arg(args, "--batch-size", "512")
        args = replace_arg(args, "--buffer-length", "750000")
        args = replace_arg(args, "--hidden-size", "512")
        args = replace_arg(args, "--num-updates", "12")
        args = replace_arg(args, "--steps-per-update", "100")
    elif name == "happo":
        args = replace_arg(args, "--hidden-size", "384")
        args = replace_arg(args, "--n-rollout-threads", "1")
    else:
        return None

    retry["args"] = args
    retry["oom_retry"] = True
    return retry


def _job_display_order(job: Mapping[str, object]) -> tuple:
    name = str(job.get("name", "")).lower()
    scenario = str(job.get("scenario", "")).upper()
    try:
        algo_idx = ALGORITHMS.index(name)
    except ValueError:
        algo_idx = 99
    try:
        scen_idx = SCENARIOS.index(scenario)
    except ValueError:
        scen_idx = 99
    return (algo_idx, scen_idx)


def print_monitor_snapshot(root: Path, status_path: Path, log_tail: int = 12) -> None:
    status = read_json(status_path)
    if not status:
        print(f"[monitor] status not found: {status_path}", flush=True)
        return

    active_jobs = sorted(
        [
            job
            for job in status.get("jobs", [])
            if job.get("completed_at") is None and not job.get("planned_only")
        ],
        key=_job_display_order,
    )
    if not active_jobs:
        print(f"[monitor] status={status.get('status')} jobs={len(status.get('jobs', []))}", flush=True)
        return

    total_steps = int(status.get("num_env_steps") or 0)
    episode_steps = int(status.get("episode_time_steps") or 0)
    episodes = int(status.get("episodes") or 0)

    print("", flush=True)
    running = ", ".join(f"{j['name'].upper()}/{j['scenario']}" for j in active_jobs)
    print(f"[monitor] parallel active ({len(active_jobs)}): {running}", flush=True)

    for job in active_jobs:
        label = f"{job['name'].upper()}/{job['scenario']}"
        run_path = resolve_status_path(root, str(job["output_dir"]))
        progress = read_json(run_path / "live_progress.json")
        if not progress:
            print(
                f"[monitor] {label} status=running output={job.get('output_dir')} "
                "(sin live_progress aun)",
                flush=True,
            )
            continue
        global_step = int(progress.get("global_step") or 0)
        episode_step = int(progress.get("episode_step") or 0)
        pct = (100.0 * global_step / total_steps) if total_steps else 0.0
        ep_pct = (100.0 * episode_step / episode_steps) if episode_steps else 0.0
        weights = progress.get("reward_axis_weights") or {}
        try:
            fps = float(progress.get("fps") or 0.0)
        except (TypeError, ValueError):
            fps = 0.0
        # Bloque individual por MADRL: pasos, aprendizaje, means, componentes y KPIs.
        print(f"[monitor] === {label} ===", flush=True)
        print(
            f"  pasos: ep={int(progress.get('episode') or 0) + 1}/{episodes} "
            f"step_ep={episode_step}/{episode_steps} ({ep_pct:.1f}%) "
            f"global={global_step}/{total_steps} ({pct:.2f}%)",
            flush=True,
        )
        print(
            f"  aprendizaje: fps={fps:.1f} live_status={progress.get('live_status')} "
            f"profile={progress.get('reward_profile')}",
            flush=True,
        )
        print(
            f"  means: ep_return={progress.get('episode_return_cumulative')} "
            f"ep_reward_mean={progress.get('episode_reward_mean_cumulative')} "
            f"total_reward_mean={progress.get('total_reward_mean_cumulative')}",
            flush=True,
        )
        print(
            f"  componentes: flex={progress.get('reward_component_flex_mean')} "
            f"carbon={progress.get('reward_component_carbon_mean')} "
            f"cost={progress.get('reward_component_cost_mean')} "
            f"ev={progress.get('reward_component_ev_mean')} "
            f"team={progress.get('reward_team_reward')}",
            flush=True,
        )
        print(
            f"  kpis: cost={progress.get('district_net_electricity_consumption_cost')} "
            f"co2={progress.get('district_net_electricity_consumption_emission')} "
            f"net_load={progress.get('district_net_electricity_consumption')} "
            f"price_mean={progress.get('electricity_price_mean')}",
            flush=True,
        )
        print(
            f"  pesos: flex={weights.get('flex')} carbon={weights.get('carbon')} "
            f"cost={weights.get('cost')}",
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

    if len(active_jobs) == 1:
        active = active_jobs[0]
        log_path = Path(str(active.get("log") or ""))
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-log_tail:]
            if lines:
                print("  log tail:", flush=True)
                for line in lines[-log_tail:]:
                    print(f"    {line[:180]}", flush=True)


def maybe_print_monitor_snapshot(
    root: Path, status_path: Path, interval: float, log_tail: int = 12
) -> None:
    """Print one global monitor snapshot per interval, regardless of how many parallel
    job threads call this. Returns immediately for every caller except the single thread
    that wins the interval, preventing duplicated/interleaved dashboards."""
    global _LAST_MONITOR_TS
    now = time.time()
    with _MONITOR_LOCK:
        if (now - _LAST_MONITOR_TS) < max(5.0, float(interval)):
            return
        _LAST_MONITOR_TS = now
        print_monitor_snapshot(root, status_path, log_tail=log_tail)


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

        while proc.poll() is None:
            if args.live_monitor:
                maybe_print_monitor_snapshot(
                    root, status_path, args.monitor_interval, log_tail=args.log_tail
                )
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
    oom_like = is_sigkill_exit(exit_code) or is_oom_failure(log_path, err_path)
    if not args.oom_retry or not oom_like:
        if is_sigkill_exit(exit_code) and args.oom_retry:
            print(
                f"[launcher] {job['name'].upper()}/{job['scenario']} exit={exit_code} "
                "(SIGKILL/OOM-killer) but logs are empty — no retry settings for this algorithm.",
                flush=True,
            )
        return exit_code

    retry_job = make_oom_retry_job(job)
    if retry_job is None:
        return exit_code

    reason = "SIGKILL/OOM-killer" if is_sigkill_exit(exit_code) else "OOM in logs"
    print(
        f"{reason} for {job['name'].upper()}/{job['scenario']}; "
        "retrying with conservative settings.",
        flush=True,
    )
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


def _patch_job_cuda_fraction(job: Mapping[str, object], cuda_fraction: float) -> Dict[str, object]:
    patched_args = replace_arg([str(a) for a in job["args"]], "--cuda-memory-fraction", cuda_fraction)
    return {**dict(job), "args": patched_args}


def _patch_job_args(job: Mapping[str, object], updates: Mapping[str, object]) -> Dict[str, object]:
    patched_args = [str(a) for a in job["args"]]
    for flag, val in updates.items():
        patched_args = replace_arg(patched_args, flag, val)
    return {**dict(job), "args": patched_args}


def _phase_cuda_fraction(args: argparse.Namespace) -> float:
    return float(getattr(args, "six_job_cuda_fraction", None) or args.three_job_cuda_fraction)


def _masac_cuda_fraction(args: argparse.Namespace) -> float:
    """MASAC critic/QMIX bursts need a higher per-process cap than HAPPO."""
    explicit = getattr(args, "six_job_masac_cuda_fraction", None)
    if explicit is not None:
        return float(explicit)
    base = _phase_cuda_fraction(args)
    vram_gib = float(getattr(args, "_detected_vram_gib", 0.0) or 0.0)
    if vram_gib >= 90.0:
        return max(base, 0.22)
    if vram_gib >= 75.0:
        return max(base, 0.18)
    return max(base, 0.16)


def _patch_happo_a100_job(job: Mapping[str, object], args: argparse.Namespace) -> Dict[str, object]:
    """On-policy: parallel rollouts (CPU) + GPU PPO updates."""
    return _patch_job_args(
        job,
        {
            "--cuda-memory-fraction": str(_phase_cuda_fraction(args)),
            "--n-rollout-threads": str(args.happo_n_rollout_threads),
            "--num-mini-batch": str(args.happo_num_mini_batch),
            "--gpu-rollout-ref": str(args.happo_gpu_rollout_ref),
            "--hidden-size": str(args.happo_hidden_size),
            "--ppo-epoch": str(args.happo_ppo_epoch),
            "--critic-epoch": str(args.happo_critic_epoch),
        },
    )


def _patch_masac_a100_job(job: Mapping[str, object], args: argparse.Namespace) -> Dict[str, object]:
    """Off-policy SAC+QMIX: CPU replay (RAM) + GPU compute & batch (Blackwell 96 GiB)."""
    return _patch_job_args(
        job,
        {
            # Replay buffer stays in CPU RAM; the sampled episode batch is preloaded to GPU
            # once ('auto', with automatic CPU fallback on OOM) so the 8760-step QMIX unroll
            # runs on GPU without per-step CPU->GPU copies. Validated by the v4 winning run.
            "--masac-preload-batch-device": str(args.masac_preload_batch_device),
            "--cuda-memory-fraction": str(_masac_cuda_fraction(args)),
            "--buffer-size": str(args.six_job_masac_buffer_size),
            "--max-replay-buffer-gib": str(args.six_job_masac_max_replay_gib),
            "--critic-batch-size": str(args.six_job_masac_critic_batch_size),
            "--critic-train-steps": str(args.masac_critic_train_steps),
            "--actor-sample-times": str(args.masac_actor_sample_times),
            "--rnn-hidden-dim": str(args.masac_rnn_hidden_dim),
            "--qmix-hidden-dim": str(args.masac_qmix_hidden_dim),
            "--hyper-hidden-dim": str(args.masac_hyper_hidden_dim),
        },
    )


def _patch_matd3_a100_job(job: Mapping[str, object], args: argparse.Namespace) -> Dict[str, object]:
    """Off-policy TD3: large RAM replay + frequent GPU gradient bursts."""
    return _patch_job_args(
        job,
        {
            "--cuda-memory-fraction": str(_phase_cuda_fraction(args)),
            "--batch-size": str(args.matd3_batch_size),
            "--buffer-size": str(args.matd3_buffer_size),
            "--hidden-size": str(args.matd3_hidden_size),
            "--train-interval": str(args.matd3_train_interval),
        },
    )


def _patch_maac_a100_job(job: Mapping[str, object], args: argparse.Namespace) -> Dict[str, object]:
    """Off-policy attention SAC: RAM buffer + multi-update GPU training."""
    return _patch_job_args(
        job,
        {
            "--cuda-memory-fraction": str(_phase_cuda_fraction(args)),
            "--batch-size": str(args.maac_batch_size),
            "--buffer-length": str(args.maac_buffer_length),
            "--hidden-size": str(args.maac_hidden_size),
            "--steps-per-update": str(args.maac_steps_per_update),
            "--num-updates": str(args.maac_num_updates),
        },
    )


_ALGO_A100_PATCHERS = {
    "happo": _patch_happo_a100_job,
    "masac": _patch_masac_a100_job,
    "matd3": _patch_matd3_a100_job,
    "maac": _patch_maac_a100_job,
}


def _prepare_two_phase_jobs(
    jobs: List[Dict[str, object]],
    algo_names: Sequence[str],
    *,
    args: argparse.Namespace,
    phase_threads: int,
    perf_env: Mapping[str, str],
) -> List[Dict[str, object]]:
    patched: List[Dict[str, object]] = []
    cuda_fraction = _phase_cuda_fraction(args)
    for algo_name in algo_names:
        phase_jobs = _patch_torch_threads([j for j in jobs if j["name"] == algo_name], phase_threads)
        phase_jobs = [{**job, "env_overrides": dict(perf_env)} for job in phase_jobs]
        patcher = _ALGO_A100_PATCHERS.get(algo_name, _patch_job_cuda_fraction)
        _matd3_stagger_s = {"E1": 0, "E2": 30, "E3": 60}
        for job in phase_jobs:
            if algo_name in _ALGO_A100_PATCHERS:
                patched_job = patcher(job, args)
            else:
                patched_job = _patch_job_cuda_fraction(job, cuda_fraction)
            if algo_name == "matd3":
                scenario = str(job.get("scenario", "")).upper()
                delay = float(_matd3_stagger_s.get(scenario, 0))
                if delay > 0:
                    patched_job = {**patched_job, "startup_delay_seconds": delay}
            patched.append(patched_job)
    return patched


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
    """Two phases (6 jobs each): HAPPO+MASAC×3 → MATD3+MAAC×3 on A100 (~70 GiB VRAM budget)."""
    cuda_fraction = _phase_cuda_fraction(args)
    masac_cuda_fraction = _masac_cuda_fraction(args)
    vram_gib = float(getattr(args, "_detected_vram_gib", 0.0) or 80.0)

    # Per-phase CPU budget on a 12-vCPU Colab A100 (6 jobs/phase, no oversubscription):
    #   Phase 1 (HAPPO+MASAC): HAPPO adds n_rollout_threads SubprocVecEnv workers, so
    #     demand = 3×(torch + rollout) + 3×torch. With torch=1, rollout=2 → 3×3 + 3×1 = 12.
    #   Phase 2 (MATD3+MAAC): single-env off-policy (no rollout), so torch=2 → 6×2 = 12.
    fallback = int(args.two_phase_torch_threads)
    p1_threads = int(getattr(args, "two_phase_p1_torch_threads", None) or fallback)
    p2_threads = int(getattr(args, "two_phase_p2_torch_threads", None) or fallback)

    _perf_env = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MALLOC_ARENA_MAX": "2",
    }

    overall_rc = 0
    phase_specs = (
        (1, TWO_PHASE_P1_HM, "HAPPO+MASAC", p1_threads),
        (2, TWO_PHASE_P2_HM, "MATD3+MAAC", p2_threads),
    )
    for phase_idx, algo_names, label, phase_threads in phase_specs:
        phase_jobs = _prepare_two_phase_jobs(
            jobs,
            algo_names,
            args=args,
            phase_threads=phase_threads,
            perf_env=_perf_env,
        )
        if not phase_jobs:
            continue
        print(
            f"\n[launcher] === PHASE {phase_idx}/2 ({label}×3 = {len(phase_jobs)} parallel): "
            f"torch_threads={phase_threads}, cuda_frac={cuda_fraction} "
            f"(HAPPO/MATD3 ~{cuda_fraction * vram_gib:.0f} GiB/job cap), "
            f"MASAC cuda_frac={masac_cuda_fraction} (~{masac_cuda_fraction * vram_gib:.0f} GiB/job), "
            f"all start simultaneously (no stagger) ===",
            flush=True,
        )
        if "happo" in algo_names:
            import math as _math_happo_log
            _roll = int(args.happo_n_rollout_threads)
            _ref = max(1, int(getattr(args, "happo_gpu_rollout_ref", 8) or 8))
            _nmb = int(getattr(args, "happo_num_mini_batch", 0) or 0)
            if _nmb <= 0:
                _nmb = max(1, _math_happo_log.ceil(_roll / _ref))
            _nmb = min(_nmb, _roll)
            print(
                f"[launcher] HAPPO: hidden={args.happo_hidden_size}, "
                f"n_rollout_threads={_roll} (SubprocVecEnv, RAM), "
                f"num_mini_batch={_nmb} (GPU minibatch ~{_ref * args.episode_time_steps} steps)",
                flush=True,
            )
        if "masac" in algo_names:
            print(
                f"[launcher] MASAC GPU: buffer={args.six_job_masac_buffer_size} ep, "
                f"max_replay={args.six_job_masac_max_replay_gib} GiB, "
                f"batch={args.six_job_masac_critic_batch_size}, "
                f"rnn={args.masac_rnn_hidden_dim}, "
                f"cuda_frac={masac_cuda_fraction} (~{masac_cuda_fraction * vram_gib:.0f} GiB/job cap)",
                flush=True,
            )
        if "matd3" in algo_names:
            print(
                f"[launcher] MATD3 RAM: buffer={args.matd3_buffer_size:,} transitions, "
                f"batch={args.matd3_batch_size}, train_interval={args.matd3_train_interval}",
                flush=True,
            )
        if "maac" in algo_names:
            print(
                f"[launcher] MAAC RAM: buffer={args.maac_buffer_length:,} steps, "
                f"batch={args.maac_batch_size}, updates={args.maac_num_updates}/"
                f"{args.maac_steps_per_update} steps",
                flush=True,
            )
        rc = run_parallel_jobs(
            root=root,
            manifest=manifest,
            status_path=status_path,
            jobs=phase_jobs,
            output_root=output_root,
            log_dir=log_dir,
            args=args,
            max_workers=len(phase_jobs),
        )
        if rc != 0:
            overall_rc = rc
            print(
                f"[launcher] Phase {phase_idx} ({label}) had failures — continuing.",
                flush=True,
            )
    return overall_rc


# Backfill priority: lightest phase-2 algorithm first, then by scenario. MAAC uses a
# smaller RAM buffer and fewer updates than MATD3, so it is the cheapest job to admit
# the moment a phase-1 slot frees (maximizes GPU/CPU utilization without oversubscribing).
_BACKFILL_ALGO_WEIGHT = {"maac": 0, "matd3": 1, "happo": 2, "masac": 3}


def _job_backfill_weight(job: Mapping[str, object]) -> Tuple[int, str]:
    name = str(job.get("name", ""))
    scenario = str(job.get("scenario", ""))
    return (_BACKFILL_ALGO_WEIGHT.get(name, 9), scenario)


def run_dynamic_backfill_jobs(
    *,
    root: Path,
    manifest: Dict[str, object],
    status_path: Path,
    jobs: List[Dict[str, object]],
    output_root: Path,
    log_dir: Path,
    args: argparse.Namespace,
) -> int:
    """Elastic single-pool scheduler for two_phase_happo_masac.

    Starts the phase-1 jobs (HAPPO+MASAC×3) and, as soon as ANY slot frees, backfills
    the next phase-2 job (MATD3/MAAC, lightest first). Total concurrency stays capped at
    the phase-1 width, so the validated VRAM envelope is never exceeded; the gain is that
    phase-2 begins overlapping the tail of phase-1 instead of waiting for ALL of phase 1.
    Falls back to the strict two-phase scheduler via run_two_phase_happo_masac_jobs.
    """
    cuda_fraction = _phase_cuda_fraction(args)
    masac_cuda_fraction = _masac_cuda_fraction(args)
    vram_gib = float(getattr(args, "_detected_vram_gib", 0.0) or 80.0)
    fallback = int(args.two_phase_torch_threads)
    p1_threads = int(getattr(args, "two_phase_p1_torch_threads", None) or fallback)
    p2_threads = int(getattr(args, "two_phase_p2_torch_threads", None) or fallback)

    _perf_env = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MALLOC_ARENA_MAX": "2",
    }

    phase1_jobs = _prepare_two_phase_jobs(
        jobs, TWO_PHASE_P1_HM, args=args, phase_threads=p1_threads, perf_env=_perf_env
    )
    phase2_jobs = _prepare_two_phase_jobs(
        jobs, TWO_PHASE_P2_HM, args=args, phase_threads=p2_threads, perf_env=_perf_env
    )
    if not phase1_jobs and not phase2_jobs:
        return 0

    total_jobs = len(phase1_jobs) + len(phase2_jobs)
    auto_workers = max(1, len(phase1_jobs) or len(phase2_jobs))
    requested_cap = int(getattr(args, "max_concurrent_jobs", 0) or 0)
    if requested_cap > 0:
        max_workers = max(1, min(requested_cap, total_jobs))
    else:
        max_workers = auto_workers
    backfill_queue = sorted(phase2_jobs, key=_job_backfill_weight)

    print(
        f"\n[launcher] === DYNAMIC BACKFILL (two_phase_happo_masac): "
        f"start {len(phase1_jobs)} phase-1 (HAPPO+MASAC), backfill {len(backfill_queue)} "
        f"phase-2 (MATD3+MAAC, lightest first) as slots free; concurrency cap={max_workers} "
        f"(HAPPO/MATD3/MAAC ~{cuda_fraction * vram_gib:.0f} GiB/job, "
        f"MASAC ~{masac_cuda_fraction * vram_gib:.0f} GiB/job) ===",
        flush=True,
    )
    if backfill_queue:
        order = ", ".join(f"{j['name'].upper()}/{j['scenario']}" for j in backfill_queue)
        print(f"[launcher] backfill order: {order}", flush=True)

    overall_rc = 0

    def _submit(pool: ThreadPoolExecutor, job: Dict[str, object]):
        return pool.submit(
            run_job_with_retry,
            root=root,
            manifest=manifest,
            status_path=status_path,
            job=job,
            output_root=output_root,
            log_dir=log_dir,
            args=args,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending = set()
        future_to_job: Dict[object, Dict[str, object]] = {}
        for job in phase1_jobs:
            fut = _submit(pool, job)
            pending.add(fut)
            future_to_job[fut] = job
        # If phase 1 has fewer jobs than workers, immediately fill spare slots.
        while len(pending) < max_workers and backfill_queue:
            job = backfill_queue.pop(0)
            fut = _submit(pool, job)
            pending.add(fut)
            future_to_job[fut] = job
            print(
                f"[launcher] backfill START {job['name'].upper()}/{job['scenario']} (spare slot)",
                flush=True,
            )

        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                done_job = future_to_job.pop(fut, {})
                try:
                    ec = int(fut.result())
                except Exception as exc:  # noqa: BLE001 - a worker crash must not abort the pool
                    ec = 1
                    print(
                        f"[launcher] worker for {done_job.get('name', '?')}/"
                        f"{done_job.get('scenario', '?')} raised {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                if ec != 0:
                    overall_rc = ec
                if backfill_queue:
                    job = backfill_queue.pop(0)
                    new_fut = _submit(pool, job)
                    pending.add(new_fut)
                    future_to_job[new_fut] = job
                    print(
                        f"[launcher] backfill START {job['name'].upper()}/{job['scenario']} "
                        f"(slot freed by {done_job.get('name', '?')}/{done_job.get('scenario', '?')})",
                        flush=True,
                    )

    if overall_rc != 0:
        print("[launcher] dynamic backfill finished with one or more job failures.", flush=True)
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
    vram_gib = float(gpu_info.get("memory_total_gib") or getattr(args, "_detected_vram_gib", 0.0) or 80.0)
    cuda_frac = float(getattr(args, "six_job_cuda_fraction", 0.14) or 0.14)
    masac_frac = _masac_cuda_fraction(args)
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
        "execution": "two_phase_happo_masac",
        "parallelization": {
            "execution_mode": "two_phase_happo_masac",
            "two_phase_algo_groups": [list(TWO_PHASE_P1_HM), list(TWO_PHASE_P2_HM)],
            "jobs_per_phase": 6,
            "max_concurrent_jobs": int(getattr(args, "max_concurrent_jobs", 0) or 0),
            "dynamic_backfill": bool(getattr(args, "dynamic_backfill", True)),
            "requested": args.parallel_scenarios > 1,
            "effective": args.parallel_scenarios > 1,
            "parallel_scenarios": args.parallel_scenarios,
            "two_phase_torch_threads": getattr(args, "two_phase_torch_threads", 2),
            "two_phase_p1_torch_threads": getattr(args, "two_phase_p1_torch_threads", 1),
            "two_phase_p2_torch_threads": getattr(args, "two_phase_p2_torch_threads", 2),
            "six_job_cuda_fraction": cuda_frac,
            "six_job_masac_buffer_size": getattr(args, "six_job_masac_buffer_size", 12),
            "six_job_masac_max_replay_gib": getattr(args, "six_job_masac_max_replay_gib", 18.0),
            "six_job_masac_critic_batch_size": getattr(args, "six_job_masac_critic_batch_size", 1),
            "six_job_masac_cuda_fraction": masac_frac,
            "two_phase_masac_cuda_fraction": masac_frac,
            "gpu_vram_gib": vram_gib,
            "strategy": (
                (
                    "dynamic_backfill: start 6 (HAPPO+MASAC x3); backfill MATD3+MAAC "
                    "lightest-first as each slot frees (no wait for all phase 1); "
                    f"cap=6 | VRAM {vram_gib:.0f} GiB | HAPPO/MATD3/MAAC cap "
                    f"{cuda_frac * vram_gib:.0f} GiB/job | MASAC cap {masac_frac * vram_gib:.0f} GiB/job"
                )
                if bool(getattr(args, "dynamic_backfill", True))
                else (
                    "two_phase_happo_masac: Phase1=HAPPO+MASAC x3 (6 parallel, no stagger); "
                    "Phase2=MATD3+MAAC x3 (6 parallel, no stagger); "
                    f"VRAM {vram_gib:.0f} GiB | HAPPO cap {cuda_frac * vram_gib:.0f} GiB/job | "
                    f"MASAC cap {masac_frac * vram_gib:.0f} GiB/job"
                )
            ),
            "est_min_per_episode_by_algo": dict(EST_MIN_PER_EPISODE_BY_ALGO),
            "est_min_per_episode": EST_MIN_PER_EPISODE_PHASE,
            "est_phase_wall_hours": round(args.episodes * EST_MIN_PER_EPISODE_PHASE / 60.0, 1),
            # Sequential upper bound (2 phases). With dynamic_backfill the real makespan
            # is lower because phase 2 overlaps phase 1 as slots free; live dashboards
            # report the FPS-based makespan ETA.
            "est_total_wall_hours": round(2 * args.episodes * EST_MIN_PER_EPISODE_PHASE / 60.0, 1),
            "est_total_wall_hours_is_upper_bound": bool(getattr(args, "dynamic_backfill", True)),
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
            "colab_dry_run_ready": bool(
                args.episode_time_steps == 8760 and args.episodes >= 1
            ),
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
    parser.add_argument(
        "--dynamic-backfill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Elastic scheduler: backfill the lightest phase-2 job (MATD3/MAAC) as soon as a "
        "phase-1 slot frees, instead of waiting for ALL of phase 1. Concurrency cap unchanged "
        "(VRAM envelope preserved). Use --no-dynamic-backfill for strict two-phase barriers.",
    )
    parser.add_argument(
        "--max-concurrent-jobs",
        default=0,
        type=int,
        help="Override the dynamic-backfill concurrency cap (default 0 = auto = phase width, 6). "
        "Raising it lets phase 2 (MATD3+MAAC) overlap phase 1 (HAPPO+MASAC) to use idle vCPUs on "
        "machines with more cores than Colab (e.g. H100 ~26 vCPU). RAM-bound: each phase-2 job is "
        "~18 GiB resident, so keep total under system RAM (e.g. 8 jobs ~144 GiB on a 177 GiB box). "
        "Does NOT raise per-job FPS (CityLearn env.step is single-threaded); only shortens the "
        "sweep makespan by overlapping the two phases.",
    )

    # ── A100-SXM4-80GB hyperparameters (two_phase_happo_masac primary) ─────────
    # Two phases (6 jobs each): Phase1 HAPPO+MASAC, Phase2 MATD3+MAAC.
    # 6 × cuda_fraction × 80 GiB ≈ 58 GiB cap at 0.12; MASAC replay ~11 GiB/job RAM (buf=8).
    # Notebook Sección 6 is the single source of truth; these defaults match launcher_base_args().
    parser.add_argument("--happo-hidden-size", default=512, type=int,
                        help="HAPPO [512,512]; stable with n_rollout_threads=2 on A100.")
    parser.add_argument("--happo-n-rollout-threads", default=2, type=int,
                        help="Parallel env rollouts per HAPPO job (SubprocVecEnv; 3 jobs×2=6 envs).")
    parser.add_argument("--happo-num-mini-batch", default=0, type=int,
                        help="PPO minibatches HAPPO (0=auto). Auto sube con n_rollout_threads "
                             "para que el minibatch de GPU (VRAM) quede ~constante.")
    parser.add_argument("--happo-gpu-rollout-ref", default=8, type=int,
                        help="Rollouts de referencia por minibatch GPU (modo auto de num_mini_batch).")
    parser.add_argument("--happo-ppo-epoch", default=10, type=int,
                        help="HAPPO PPO actor epochs/update (10 aprovecha GPU ociosa; HARL default=5).")
    parser.add_argument("--happo-critic-epoch", default=10, type=int,
                        help="HAPPO critic epochs/update (10 aprovecha GPU ociosa; HARL default=5).")
    parser.add_argument("--masac-max-replay-buffer-gib", default=8.0, type=float)
    parser.add_argument("--masac-buffer-size", default=2, type=int,
                        help="Episodes in replay buffer (2 = stable Iquitos profile for 8760-step QMIX).")
    parser.add_argument("--masac-critic-batch-size", default=1, type=int,
                        help="MASAC episodes per QMIX update (NOT transitions; use 1 for 8760-step CityLearn).")
    parser.add_argument("--masac-critic-train-steps", default=1, type=int,
                        help="Critic passes per env epoch (1 for 8760-step episodes in 6-parallel).")
    parser.add_argument("--masac-actor-sample-times", default=1, type=int,
                        help="Actor samples per env epoch (backend caps at 1 for CityLearn).")
    parser.add_argument("--masac-rnn-hidden-dim", default=64, type=int,
                        help="GRU hidden; 64 is stable for 8760-step CityLearn QMIX in 6-parallel.")
    parser.add_argument("--masac-qmix-hidden-dim", default=32, type=int,
                        help="QMIX mixing hidden dim (32 for long CityLearn episodes).")
    parser.add_argument("--masac-hyper-hidden-dim", default=64, type=int,
                        help="Hypernetwork hidden dim for QMIX weights.")
    parser.add_argument("--masac-preload-batch-device", default="auto", choices=("auto", "cuda", "cpu"),
                        help="auto: replay in CPU RAM + episode batch on GPU (OOM falls back to CPU). "
                             "Use cpu to force replay+batch on CPU; cuda to force GPU (single-job).")
    parser.add_argument("--matd3-batch-size", default=256, type=int,
                        help="MATD3 batch aligned to the v4 winning run (stable 3/3 exit_code=0).")
    parser.add_argument("--matd3-buffer-size", default=4096, type=int,
                        help="v4 stable replay: 17 separate policies x double centralized critic.")
    parser.add_argument("--matd3-hidden-size", default=256, type=int,
                        help="MATD3 actor+critic hidden 256 (v4 winning run).")
    parser.add_argument("--matd3-train-interval", default=100, type=int,
                        help="Train every 100 env steps (v4 winning run).")
    parser.add_argument("--maac-batch-size", default=768, type=int,
                        help="MAAC batch (Tensor Cores).")
    parser.add_argument("--maac-buffer-length", default=1000000, type=int,
                        help="1M steps ~7 GiB RAM/job; 3x21 GiB << 167 GiB.")
    parser.add_argument("--maac-hidden-size", default=768, type=int,
                        help="MAAC attention critic hidden 768 (stable 6-parallel phase 2).")
    parser.add_argument("--maac-steps-per-update", default=50, type=int,
                        help="Collect 50 steps then burst GPU updates.")
    parser.add_argument("--maac-num-updates", default=12, type=int,
                        help="12 gradient steps per update (stable 6-parallel burst).")
    parser.add_argument(
        "--execution-mode",
        default="two_phase_happo_masac",
        choices=("two_phase_happo_masac",),
        help="Colab A100 official: Phase1 HAPPO+MASAC×3, Phase2 MATD3+MAAC×3 (6 parallel each, no stagger).",
    )
    parser.add_argument(
        "--two-phase-torch-threads",
        default=2,
        type=int,
        help="Fallback torch threads per job in two_phase when per-phase values are unset.",
    )
    parser.add_argument(
        "--two-phase-p1-torch-threads",
        default=1,
        type=int,
        help="Phase 1 (HAPPO+MASAC) torch threads/job. HAPPO adds rollout workers, so "
        "1 keeps 3×(1+rollout)+3×1 ≈ 12 vCPU without oversubscription.",
    )
    parser.add_argument(
        "--two-phase-p2-torch-threads",
        default=2,
        type=int,
        help="Phase 2 (MATD3+MAAC) torch threads/job. Single-env off-policy (no rollout), "
        "so 2 fills 6×2 = 12 vCPU during GPU update bursts.",
    )
    parser.add_argument(
        "--six-job-cuda-fraction",
        default=0.14,
        type=float,
        help="Per-process VRAM cap for HAPPO/MATD3/MAAC (fraction of total VRAM).",
    )
    parser.add_argument(
        "--six-job-masac-cuda-fraction",
        default=None,
        type=float,
        help="Per-process VRAM cap for MASAC only (higher than HAPPO; QMIX unrolls 8760 steps/episode on GPU).",
    )
    parser.add_argument(
        "--six-job-masac-buffer-size",
        default=2,
        type=int,
        help="MASAC replay episodes in 6-job phase (2 = stable Iquitos profile).",
    )
    parser.add_argument(
        "--six-job-masac-max-replay-gib",
        default=8.0,
        type=float,
        help="MASAC replay cap GiB per job in 6-job phase (must exceed buffer estimate).",
    )
    parser.add_argument(
        "--six-job-masac-critic-batch-size",
        default=1,
        type=int,
        help="MASAC episodes per QMIX update in 6-job phase (1 for CityLearn 8760-step episodes).",
    )
    parser.add_argument(
        "--three-job-cuda-fraction",
        default=0.12,
        type=float,
        help="Deprecated alias for --six-job-cuda-fraction.",
    )
    parser.add_argument(
        "--three-job-masac-buffer-size",
        default=12,
        type=int,
        help="Deprecated alias for --six-job-masac-buffer-size.",
    )
    parser.add_argument(
        "--three-job-masac-max-replay-gib",
        default=18.0,
        type=float,
        help="Deprecated alias for --six-job-masac-max-replay-gib.",
    )
    parser.add_argument(
        "--three-job-masac-critic-batch-size",
        default=1,
        type=int,
        help="Deprecated alias for --six-job-masac-critic-batch-size.",
    )
    parser.add_argument(
        "--two-phase-masac-cuda-fraction",
        default=0.12,
        type=float,
        help="Deprecated alias for --six-job-cuda-fraction.",
    )
    return parser.parse_args(argv)


def _sync_six_job_masac_defaults(args: argparse.Namespace) -> None:
    """Keep build_jobs and phase patchers aligned on six-job MASAC caps."""
    args.masac_buffer_size = int(getattr(args, "six_job_masac_buffer_size", args.masac_buffer_size))
    args.masac_max_replay_buffer_gib = float(
        getattr(args, "six_job_masac_max_replay_gib", args.masac_max_replay_buffer_gib)
    )
    args.masac_critic_batch_size = int(
        getattr(args, "six_job_masac_critic_batch_size", args.masac_critic_batch_size)
    )


def _load_protocol_guard():
    guard_path = Path(__file__).resolve().parent / "colab_protocol_guard.py"
    if not guard_path.is_file():
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location("_colab_protocol_guard", guard_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_launcher_self() -> None:
    script = Path(__file__).resolve()
    guard = _load_protocol_guard()
    if guard is not None:
        guard.assert_not_legacy_path(script, role="launcher")
        guard.validate_launcher_source(script.read_text(encoding="utf-8"), path=str(script))
        return
    if "MADRL_CityLearn_v3" in str(script):
        raise RuntimeError(f"Launcher en clone legacy Drive: {script}")
    src = script.read_text(encoding="utf-8")
    if LAUNCHER_PROTOCOL_ID not in src or "run_two_phase_happo_masac_jobs" not in src:
        raise RuntimeError(f"Launcher corrupto o legacy: {script}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _sync_six_job_masac_defaults(args)
    _assert_launcher_self()
    script_path = Path(__file__).resolve()
    print(
        f"[launcher] protocol={LAUNCHER_PROTOCOL_ID} execution_mode={args.execution_mode} "
        f"script_path={script_path}",
        flush=True,
    )
    if args.execution_mode != "two_phase_happo_masac":
        print(
            f"[launcher] FATAL: execution_mode={args.execution_mode!r} — solo two_phase_happo_masac",
            flush=True,
        )
        return 2
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
    setattr(args, "_detected_vram_gib", float(gpu_info.get("memory_total_gib") or 0.0))
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

    if args.execution_mode != "two_phase_happo_masac":
        raise ValueError(
            f"Unsupported execution_mode={args.execution_mode!r}. "
            "Colab A100 launcher only supports two_phase_happo_masac."
        )

    if getattr(args, "dynamic_backfill", True):
        overall_rc = run_dynamic_backfill_jobs(
            root=root,
            manifest=manifest,
            status_path=status_path,
            jobs=jobs,
            output_root=output_root,
            log_dir=log_dir,
            args=args,
        )
    else:
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


if __name__ == "__main__":
    raise SystemExit(main())
