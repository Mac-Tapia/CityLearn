"""Shared CityLearn v3 training launch utilities for external MADRL backends."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import itertools
import json
import os
import pickle
import shutil
import sys
import threading
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Force a headless matplotlib backend for every MADRL training script. Some external
# backends (e.g. external/MARL MASAC runner_msac.py) import matplotlib.pyplot at module
# load and call plt.savefig() at the end of run(). Inside Colab's headless Popen
# subprocesses an interactive backend raises and silently discards the whole
# finalization (no checkpoint, no results.json). Setting Agg here — imported by all
# train scripts before any backend — makes plotting safe for the entire pipeline.
os.environ.setdefault("MPLBACKEND", "Agg")
try:  # pragma: no cover - defensive: matplotlib may be absent in minimal envs
    import matplotlib

    matplotlib.use("Agg", force=True)
except Exception:
    pass

import numpy as np
from gym import spaces


SCRIPT_PATH = Path(__file__).resolve()
CITYLEARN_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]
EXTERNAL_ROOT = PROJECT_ROOT / "external"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "citylearn_v3_madrl"
_CSV_CACHE_DIR = PROJECT_ROOT / "outputs" / "dataset_cache"
DATA_DIR_NAME = "data"
CHECKPOINT_DIR_NAME = "checkpoints"
FIGURES_DIR_NAME = "figures"
TABLES_DIR_NAME = "tables"
JOB_LAUNCHER_COMPLETE_MARKER = "job_launcher_complete.json"
SPACE_BOUND = 1.0e6


def ensure_project_paths() -> None:
    for path in (CITYLEARN_ROOT, PROJECT_ROOT):
        path_text = str(path)

        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def add_external_path(*parts: str) -> Path:
    path = EXTERNAL_ROOT.joinpath(*parts)
    path_text = str(path)

    if path.exists() and path_text not in sys.path:
        sys.path.insert(0, path_text)

    return path


def install_noop_wandb() -> None:
    """Provide a tiny wandb stub for backends that import it unconditionally."""

    if "wandb" in sys.modules:
        return

    module = types.ModuleType("wandb")
    module.run = types.SimpleNamespace(dir=str(DEFAULT_OUTPUT_ROOT / "wandb_stub"))
    module.init = lambda *args, **kwargs: types.SimpleNamespace(
        dir=str(DEFAULT_OUTPUT_ROOT / "wandb_stub"),
        finish=lambda: None,
    )
    module.log = lambda *args, **kwargs: None
    sys.modules["wandb"] = module


def add_common_citylearn_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--schema-path", default=None, help="Optional CityLearn v2 schema path.")
    parser.add_argument("--scenario", default="E1", help="CityLearn v3 scenario label.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument("--episode-time-steps", default=4, type=int, help="Episode length for the launcher.")
    parser.add_argument("--output-dir", default=None, help="Directory for logs, models and summaries.")
    parser.add_argument(
        "--normalize-observations",
        dest="normalize_observations",
        action="store_true",
        default=True,
        help="Use CityLearn min-max observation normalization and cyclic time encoding before backend training.",
    )
    parser.add_argument(
        "--raw-observations",
        dest="normalize_observations",
        action="store_false",
        help="Disable observation normalization and feed raw CityLearn units to the backend.",
    )
    parser.add_argument(
        "--live-progress-interval",
        default=250,
        type=int,
        help="Environment steps between live_progress.json writes.",
    )
    parser.add_argument(
        "--artifact-profile",
        default="full",
        choices=("full", "efficient", "minimal"),
        help=(
            "Artifact write profile. full preserves legacy root/data CSV mirrors; "
            "efficient keeps canonical data CSVs and avoids duplicate heavy traces; "
            "minimal writes the smallest compatible artifact set."
        ),
    )
    parser.add_argument(
        "--trace-record-interval",
        default=1,
        type=int,
        help=(
            "Record one detailed per-agent trace row every N environment steps. "
            "Use 1 for full traces; larger values reduce memory and CSV I/O."
        ),
    )
    parser.add_argument(
        "--trace-detail",
        default="full",
        choices=("full", "compact"),
        help="full includes named observation/action columns; compact keeps summary and energy columns only.",
    )
    parser.add_argument(
        "--gpu-profile",
        default="auto",
        choices=("auto", "local4060_fast", "local4060", "balanced", "conservative", "aws"),
        help="Torch/CUDA runtime profile. local4060_fast keeps RTX 4060 runs lighter; local4060 enables larger local GPU settings.",
    )
    parser.add_argument(
        "--cuda-memory-fraction",
        default=None,
        type=float,
        help=(
            "Optional per-process CUDA memory cap in the range (0, 1]. "
            "Launchers set this from dedicated VRAM reported by nvidia-smi, "
            "not from Windows shared GPU memory."
        ),
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Disable intra-job checkpoint resume when live_progress/checkpoints exist.",
    )


def configure_torch_runtime(
    torch_module=None,
    *,
    use_cuda: bool = False,
    torch_threads: Optional[int] = None,
    gpu_profile: str = "auto",
    cuda_memory_fraction: Optional[float] = None,
) -> Dict[str, object]:
    """Configure Torch for MADRL training and return runtime metadata."""

    if torch_module is None:
        import torch as torch_module

    profile = str(gpu_profile or "auto").strip().lower()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    if use_cuda:
        default_cuda_alloc_conf = (
            "max_split_size_mb:128"
            if sys.platform.startswith("win")
            else "expandable_segments:True,max_split_size_mb:128"
        )
        current_cuda_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").strip()

        if not current_cuda_alloc_conf:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = default_cuda_alloc_conf
        elif sys.platform.startswith("win") and "expandable_segments" in current_cuda_alloc_conf.lower():
            parts = [
                part.strip()
                for part in current_cuda_alloc_conf.split(",")
                if part.strip() and not part.strip().lower().startswith("expandable_segments")
            ]
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(parts) if parts else "max_split_size_mb:128"

    requested_threads = _as_int(torch_threads)
    if requested_threads is not None and requested_threads > 0:
        try:
            torch_module.set_num_threads(requested_threads)
        except Exception:
            pass

    cuda_available = bool(torch_module.cuda.is_available())
    cuda_enabled = bool(use_cuda and cuda_available)
    matmul_precision = None
    tf32_allowed = False
    requested_cuda_memory_fraction = None
    cuda_memory_fraction_applied = None
    cuda_memory_fraction_error = None

    if cuda_memory_fraction is not None:
        try:
            requested_cuda_memory_fraction = float(cuda_memory_fraction)
        except Exception:
            requested_cuda_memory_fraction = None

    if hasattr(torch_module, "set_float32_matmul_precision"):
        matmul_precision = "high" if profile in {"auto", "local4060_fast", "local4060", "balanced", "aws"} else "highest"
        try:
            torch_module.set_float32_matmul_precision(matmul_precision)
        except Exception:
            matmul_precision = None

    if cuda_enabled:
        try:
            torch_module.cuda.set_device(0)
        except Exception:
            pass

        if requested_cuda_memory_fraction is not None:
            if 0.0 < requested_cuda_memory_fraction <= 1.0:
                try:
                    torch_module.cuda.set_per_process_memory_fraction(requested_cuda_memory_fraction, 0)
                    cuda_memory_fraction_applied = requested_cuda_memory_fraction
                except Exception as exc:
                    cuda_memory_fraction_error = str(exc)
            else:
                cuda_memory_fraction_error = (
                    f"cuda_memory_fraction outside (0, 1]: {requested_cuda_memory_fraction}"
                )

        try:
            torch_module.backends.cuda.matmul.allow_tf32 = profile != "conservative"
            tf32_allowed = bool(torch_module.backends.cuda.matmul.allow_tf32)
        except Exception:
            pass

        try:
            torch_module.backends.cudnn.allow_tf32 = profile != "conservative"
            torch_module.backends.cudnn.benchmark = profile in {"auto", "local4060_fast", "local4060", "balanced", "aws"}
        except Exception:
            pass

    device_name = None
    device_total_memory_gib = None

    if cuda_available:
        try:
            device_name = torch_module.cuda.get_device_name(0)
            device_total_memory_gib = float(torch_module.cuda.get_device_properties(0).total_memory / (1024**3))
        except Exception:
            pass

    return {
        "profile": profile,
        "cuda_requested": bool(use_cuda),
        "cuda_available": cuda_available,
        "cuda_enabled": cuda_enabled,
        "cuda_device": "cuda:0" if cuda_enabled else "cpu",
        "cuda_device_name": device_name,
        "cuda_device_total_memory_gib": device_total_memory_gib,
        "cuda_memory_fraction_requested": requested_cuda_memory_fraction,
        "cuda_memory_fraction_applied": cuda_memory_fraction_applied,
        "cuda_memory_fraction_error": cuda_memory_fraction_error,
        "torch_threads": requested_threads,
        "matmul_precision": matmul_precision,
        "tf32_allowed": tf32_allowed,
        "cudnn_benchmark": bool(cuda_enabled and profile in {"auto", "local4060_fast", "local4060", "balanced", "aws"}),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }


def effective_matd3_per_policy_buffer_size(
    total_buffer: int,
    *,
    num_agents: int,
    share_policy: bool = False,
) -> int:
    """Split a total replay budget across per-agent policies for MATD3.

    The marlbenchmark/off-policy ``MlpReplayBuffer`` allocates one
    ``MlpPolicyBuffer`` per policy when ``share_policy=False``. CityLearn v3
    uses 17 decentralized actor policies, so a CLI ``--buffer-size 2000000`` would
    otherwise allocate 17× that many transitions and OOM-kill the process.
    """
    total_buffer = max(1, int(total_buffer))
    num_agents = max(1, int(num_agents))
    if share_policy or num_agents <= 1:
        return total_buffer
    return max(500, total_buffer // num_agents)


def estimate_matd3_replay_ram_gib(
    *,
    per_policy_buffer: int,
    num_agents: int,
    obs_dim: int,
    share_obs_dim: int,
    act_dim: int,
    share_policy: bool = False,
    use_same_share_obs: bool = True,
) -> float:
    """Conservative RAM estimate for MATD3 numpy replay buffers (GiB)."""
    per_policy = max(1, int(per_policy_buffer))
    n_agents = max(1, int(num_agents))
    policies = 1 if share_policy else n_agents
    obs_dim = max(1, int(obs_dim))
    share_obs_dim = max(1, int(share_obs_dim))
    act_dim = max(1, int(act_dim))

    agents_per_policy = n_agents if share_policy else 1
    share_bytes = share_obs_dim * 4 * 2 if use_same_share_obs else agents_per_policy * share_obs_dim * 4 * 2
    bytes_per_policy_buffer = per_policy * (
        agents_per_policy * obs_dim * 4 * 2
        + share_bytes
        + agents_per_policy * act_dim * 4
        + agents_per_policy * 16
    )
    return (bytes_per_policy_buffer * policies) / float(1024 ** 3)


def normalize_algorithm_dir(algorithm: str) -> str:
    """Folder name for a MADRL backend: HAPPO, MASAC, MATD3, MAAC."""
    return algorithm.strip().upper()


def normalize_scenario_dir(scenario: str, seed: int = 0) -> str:
    """Folder name for a scenario axis: E1, E2, E3 (seed 0 omits suffix)."""
    scenario = scenario.strip().upper()
    seed = int(seed)
    if seed == 0:
        return scenario
    return f"{scenario}_s{seed}"


def job_run_relative_parts(algorithm: str, scenario: str, seed: int = 0) -> Path:
    """Relative path ``<MADRL>/<Escenario>`` under an output root."""
    return Path(normalize_algorithm_dir(algorithm)) / normalize_scenario_dir(scenario, seed)


def iter_job_run_dir_candidates(
    base: Path,
    algorithm: str,
    scenario: str,
    seed: int,
):
    """Possible run directories: new simple layout first, then legacy ``E1_seed_0``."""
    base = Path(base)
    algo_upper = normalize_algorithm_dir(algorithm)
    algo_lower = algorithm.strip().lower()
    scen = scenario.strip().upper()
    seed = int(seed)
    simple = normalize_scenario_dir(scen, seed)
    legacy = f"{scen}_seed_{seed}"
    patterns = (
        base / algo_upper / simple,
        base / algo_lower / simple,
        base / algo_upper / legacy,
        base / algo_lower / legacy,
    )
    seen = set()
    for path in patterns:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        yield path


def resolve_job_run_dir(base: Path, algorithm: str, scenario: str, seed: int) -> Path:
    """Canonical run directory for one MADRL job; creates parents if needed."""
    path = next(iter_job_run_dir_candidates(base, algorithm, scenario, seed))
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_existing_job_run_dir(
    base: Path,
    algorithm: str,
    scenario: str,
    seed: int,
) -> Optional[Path]:
    """Return an existing run directory (new or legacy layout), or None."""
    for candidate in iter_job_run_dir_candidates(base, algorithm, scenario, seed):
        if candidate.is_dir():
            return candidate
    return None


def resolve_output_dir(output_dir: Optional[str], algorithm: str, scenario: str, seed: int) -> Path:
    base = Path(output_dir) if output_dir else DEFAULT_OUTPUT_ROOT / normalize_algorithm_dir(algorithm)
    path = base / normalize_scenario_dir(scenario, seed)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_artifact_layout(output_dir: Path) -> Dict[str, Path]:
    """Create the canonical CityLearn v3 output folders for one MADRL run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dirs = {
        "run": output_dir,
        "data": output_dir / DATA_DIR_NAME,
        "checkpoints": output_dir / CHECKPOINT_DIR_NAME,
        "figures": output_dir / FIGURES_DIR_NAME,
        "tables": output_dir / FIGURES_DIR_NAME / TABLES_DIR_NAME,
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def start_live_progress_heartbeat(
    adapter,
    interval_seconds: int,
    *,
    active_stage: str,
    note: Optional[str] = None,
    initial_stage: Optional[str] = None,
    initial_note: Optional[str] = None,
):
    """Start a daemon heartbeat thread for external MADRL backend update phases."""

    if adapter is None or not hasattr(adapter, "write_live_heartbeat"):
        return None, None

    interval_seconds = int(interval_seconds)
    if interval_seconds <= 0:
        return None, None

    stop_event = threading.Event()

    if initial_stage:
        try:
            adapter.write_live_heartbeat(stage=initial_stage, note=initial_note or note)
        except Exception:
            pass

    def heartbeat_loop() -> None:
        while not stop_event.wait(interval_seconds):
            try:
                adapter.write_live_heartbeat(stage=active_stage, note=note)
            except Exception:
                pass
            # Flush buffered checkpoint/live_progress writes to Drive so an abrupt
            # Colab disconnect cannot discard recent resumable progress.
            flush_filesystem_buffers()

    thread = threading.Thread(
        target=heartbeat_loop,
        name=f"{active_stage}-live-progress-heartbeat",
        daemon=True,
    )
    thread.start()
    return stop_event, thread


def stop_live_progress_heartbeat(stop_event, thread, *, timeout_seconds: float = 5.0) -> None:
    """Stop a heartbeat returned by ``start_live_progress_heartbeat``."""

    if stop_event is not None:
        stop_event.set()

    if thread is not None:
        thread.join(timeout=timeout_seconds)


def flush_filesystem_buffers() -> None:
    """Force OS-level flush of buffered writes to durable storage.

    On Colab the Google Drive FUSE mount buffers writes asynchronously, so an
    abrupt runtime crash/disconnect can silently discard recently written
    checkpoints and ``live_progress.json`` before they reach Drive's backend.
    ``os.sync()`` flushes every dirty buffer (including the FUSE backend) so the
    resumable artifacts survive an interruption. No-op on platforms without
    ``os.sync`` (e.g. Windows during local validation).
    """

    sync = getattr(os, "sync", None)
    if not callable(sync):
        return
    try:
        sync()
    except Exception:
        pass


def fsync_file(path: Path) -> None:
    """Best-effort durable flush of a single file to disk/Drive."""

    try:
        fd = os.open(str(path), os.O_RDONLY)
    except (OSError, ValueError):
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _artifact_layout_payload(dirs: Mapping[str, Path]) -> Dict[str, str]:
    return {name: str(path) for name, path in dirs.items()}


def write_json(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_json_mirrors(paths: Sequence[Path], data: Mapping[str, object]) -> None:
    for path in paths:
        write_json(path, data)


def _as_float(value) -> Optional[float]:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None

    return output if np.isfinite(output) else None


def _as_int(value) -> Optional[int]:
    try:
        output = int(value)
    except (TypeError, ValueError):
        return None

    return output


def _series_value(owner, name: str, index: Optional[int]) -> Optional[float]:
    values = getattr(owner, name, None)

    if values is None:
        return None

    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None

    if array.size == 0:
        return None

    if index is None:
        index = array.size - 1

    index = max(0, min(int(index), array.size - 1))
    return _as_float(array[index])


def _district_current_scalar(citylearn_env, attr: str, time_step) -> Optional[float]:
    """Sum attr[time_step] across buildings without triggering the env-level DataFrame property.

    The env-level properties (e.g. net_electricity_consumption_without_storage) build a full
    pd.DataFrame of shape (n_buildings, n_steps_so_far) on every call — O(n) memory per step,
    O(n^2) total. This helper reads the per-building list directly and sums, which is O(1).
    """
    buildings = getattr(citylearn_env, "buildings", None) or []
    if not buildings:
        return None
    total = 0.0
    for b in buildings:
        series = getattr(b, attr, None)
        if series is None:
            return None
        try:
            idx = int(time_step) if time_step is not None else len(series) - 1
            idx = max(0, min(idx, len(series) - 1))
            total += float(series[idx])
        except (IndexError, TypeError, ValueError):
            return None
    return total


def _mean_current_building_signal(citylearn_env, source_name: str, series_name: str, index: Optional[int]) -> Optional[float]:
    values = []

    for building in getattr(citylearn_env, "buildings", []):
        source = getattr(building, source_name, None)
        value = _series_value(source, series_name, index) if source is not None else None

        if value is not None:
            values.append(value)

    if not values:
        return None

    return float(np.mean(values))


def _compact_array_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        array = np.asarray([], dtype=float)

    array = array[np.isfinite(array)]

    if array.size == 0:
        return {
            "mean": None,
            "min": None,
            "max": None,
            "l2": None,
        }

    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "l2": float(np.linalg.norm(array)),
    }


def _csv_safe(value):
    if value is None:
        return ""

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, np.generic):
        return value.item()

    return json.dumps(value, sort_keys=True, default=str)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Atomically write rows to a CSV (tmp file + replace) so a crash mid-write never
    leaves a half-written/torn file on Drive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})

    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_safe(row.get(key)) for key in fieldnames})
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _write_csv_mirrors(paths: Sequence[Path], rows: Sequence[Mapping[str, object]]) -> None:
    for path in paths:
        write_csv(path, rows)


def read_csv_rows(path: Path) -> List[Dict[str, object]]:
    """Read a CSV written by write_csv back into a list of dict rows (values as str)."""
    path = Path(path)
    if not path.is_file():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as file:
            return [dict(row) for row in csv.DictReader(file)]
    except Exception:
        return []


def _dedup_rows_keep_first(
    rows: Sequence[Mapping[str, object]],
    keys: Sequence[str],
) -> List[Dict[str, object]]:
    """Drop duplicate rows sharing the same `keys`, keeping the FIRST occurrence.

    Used on resume to clean an incremental CSV that a buggy/old run polluted with
    duplicate (episode, step[, agent]) rows, guaranteeing one row per key in file order.
    """
    seen: set = set()
    out: List[Dict[str, object]] = []
    for row in rows:
        signature = tuple(str(row.get(k)) for k in keys)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(dict(row))
    return out


def _append_csv_rows_stable_schema(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    """Append rows to a CSV using a KNOWN-STABLE header (no new keys).

    Callers must guarantee every row's keys are a subset of `fieldnames`; schema
    growth is handled by a full atomic rewrite in the recorder, not here.
    """
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_safe(row.get(key)) for key in fieldnames})


def _write_markdown_table(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("_No rows generated._\n", encoding="utf-8")
        return

    columns = sorted({key for row in rows for key in row.keys()})
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for row in rows:
        values = [str(_csv_safe(row.get(column))).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_objective_env(candidate):
    if candidate is None:
        return None

    if hasattr(candidate, "adapter"):
        adapter = getattr(candidate, "adapter", None)

        if hasattr(adapter, "env"):
            return adapter.env

    if hasattr(candidate, "envs"):
        envs = getattr(candidate, "envs")
        if envs:
            return _resolve_objective_env(envs[0])

    if hasattr(candidate, "env") and candidate.__class__.__name__ not in {"CityLearnDecPOMDPEnv"}:
        nested = getattr(candidate, "env")
        if nested is not candidate:
            return _resolve_objective_env(nested)

    return candidate


def _resolve_adapter(candidate):
    if candidate is None:
        return None

    if hasattr(candidate, "adapter"):
        return candidate.adapter

    if hasattr(candidate, "envs"):
        envs = getattr(candidate, "envs")

        if envs:
            return _resolve_adapter(envs[0])

    if hasattr(candidate, "env") and candidate.__class__.__name__ not in {"CityLearnDecPOMDPEnv"}:
        nested = getattr(candidate, "env")

        if nested is not candidate:
            return _resolve_adapter(nested)

    return None


def _empty_objectives() -> Dict[str, object]:
    from citylearn.v3.objectives import objective_manifest

    return {
        "manifest": objective_manifest(),
        "axes": {},
        "project_axis_metrics": {},
        "axis_kpis": {},
        "supporting_values": {},
        "all_values": {},
        "kpi_frame_rows": 0,
    }


def _adapter_completed_objectives(adapter) -> Optional[Mapping[str, object]]:
    if adapter is None:
        return None

    objectives = getattr(adapter, "last_completed_objectives", None)

    if isinstance(objectives, Mapping) and "all_values" in objectives:
        return objectives

    return None


def _last_timeseries_row(adapter) -> Dict[str, object]:
    rows = getattr(adapter, "timeseries_records", None) if adapter is not None else None

    if not rows:
        return {}

    return dict(rows[-1])


def _report_source_metadata(candidate, adapter, objective_env, source_type: str) -> Dict[str, object]:
    last_row = _last_timeseries_row(adapter)
    return {
        "type": source_type,
        "candidate_class": candidate.__class__.__name__ if candidate is not None else None,
        "objective_env_class": objective_env.__class__.__name__ if objective_env is not None else None,
        "adapter_class": adapter.__class__.__name__ if adapter is not None else None,
        "adapter_algorithm": getattr(adapter, "algorithm", None) if adapter is not None else None,
        "completed_episode_count": getattr(adapter, "completed_episode_count", 0) if adapter is not None else 0,
        "last_completed_episode": getattr(adapter, "last_completed_episode", None) if adapter is not None else None,
        "last_completed_global_step": getattr(adapter, "last_completed_global_step", None) if adapter is not None else None,
        "last_completed_time_step": getattr(adapter, "last_completed_time_step", None) if adapter is not None else None,
        "last_recorded_global_step": last_row.get("global_step"),
        "last_recorded_episode": last_row.get("episode"),
        "last_recorded_episode_step": last_row.get("episode_step"),
        "last_recorded_time_step": last_row.get("time_step"),
        "last_recorded_all_done": last_row.get("all_done"),
        "timeseries_rows": len(getattr(adapter, "timeseries_records", []) or []) if adapter is not None else 0,
        "trace_rows": len(getattr(adapter, "trace_records", []) or []) if adapter is not None else 0,
        "episode_time_steps": getattr(adapter, "episode_time_steps", None) if adapter is not None else None,
        "snapshot_error": getattr(adapter, "last_completed_snapshot_error", None) if adapter is not None else None,
    }


def _report_warnings(report_source: Mapping[str, object], objectives: Mapping[str, object]) -> List[Dict[str, object]]:
    warnings: List[Dict[str, object]] = []
    source_type = report_source.get("type")
    last_all_done = report_source.get("last_recorded_all_done")

    if source_type == "current_environment" and last_all_done is False:
        warnings.append({
            "severity": "warning",
            "code": "current_env_incomplete",
            "message": "The current environment was not at an all_done episode boundary when KPIs were evaluated.",
        })

    if source_type == "last_completed_episode_snapshot" and last_all_done is False:
        warnings.append({
            "severity": "info",
            "code": "using_completed_snapshot_after_backend_reset",
            "message": "The backend current environment is incomplete; KPIs come from the last completed episode snapshot.",
        })

    if report_source.get("snapshot_error"):
        warnings.append({
            "severity": "warning",
            "code": "completed_snapshot_error",
            "message": str(report_source["snapshot_error"]),
        })

    if not objectives.get("all_values") and source_type != "none":
        warnings.append({
            "severity": "warning",
            "code": "empty_objective_values",
            "message": "Objective evaluation returned no all_values.",
        })

    return warnings


def citylearn_v3_training_report(candidate) -> Dict[str, object]:
    """Return standardized CityLearn v2 KPI reporting for a launcher.

    The report is intentionally shared by HAPPO, MASAC, MATD3 and MAAC so each
    algorithm writes the same objective-axis KPIs. Project metrics are the
    three thesis axes: flexibility, CO2 emissions and costs.
    """

    from citylearn.v3.objectives import evaluate_objectives

    adapter = _resolve_adapter(candidate)
    objective_env = _resolve_objective_env(candidate)
    completed_objectives = _adapter_completed_objectives(adapter)

    if completed_objectives is not None:
        objectives = completed_objectives
        source_type = "last_completed_episode_snapshot"
    elif objective_env is None:
        objectives = _empty_objectives()
        source_type = "none"
    else:
        objectives = evaluate_objectives(objective_env)
        source_type = "current_environment"

    report_source = _report_source_metadata(candidate, adapter, objective_env, source_type)

    return {
        "project_axis_metrics": objectives["project_axis_metrics"],
        "objective_axis_kpis": objectives["axes"],
        "axis_kpis": objectives["axis_kpis"],
        "supporting_values": objectives["supporting_values"],
        "all_values": objectives["all_values"],
        "kpi_frame_rows": objectives["kpi_frame_rows"],
        "objective_manifest": objectives["manifest"],
        "report_source": report_source,
        "report_warnings": _report_warnings(report_source, objectives),
    }


def _checkpoint_files(output_dir: Path, checkpoint_dir: Optional[Path] = None) -> List[Dict[str, object]]:
    checkpoint_extensions = {".pt", ".pkl", ".pth", ".ckpt", ".zip"}
    output = []
    checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else output_dir
    search_root = checkpoint_dir if checkpoint_dir.exists() else output_dir

    if not any(path.is_file() and path.suffix.lower() in checkpoint_extensions for path in search_root.rglob("*")):
        search_root = output_dir

    for path in sorted(search_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in checkpoint_extensions:
            continue

        output.append({
            "path": str(path),
            "relative_path": str(path.relative_to(output_dir)),
            "bytes": path.stat().st_size,
        })

    return output


def job_has_final_results(output_dir: Path) -> bool:
    output_dir = Path(output_dir)
    return (output_dir / "data" / "results.json").is_file() or (output_dir / "results.json").is_file()


def read_job_results_json(output_dir: Path) -> Optional[Dict[str, object]]:
    output_dir = Path(output_dir)
    for rel in ("data/results.json", "results.json"):
        path = output_dir / rel
        if not path.is_file():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _resolve_job_target_episodes(
    payload: Mapping[str, object],
    *,
    target_episodes: Optional[int] = None,
) -> Optional[int]:
    hyperparameters = dict(payload.get("hyperparameters") or {})
    target = target_episodes
    if target is None:
        job_resume = dict(hyperparameters.get("job_resume") or {})
        target = job_resume.get("target_episodes")
    if target is None:
        target = hyperparameters.get("target_episodes")
    if target is None:
        target = hyperparameters.get("episodes")
    return _as_int(target)


def _recorded_episodes_from_results(payload: Mapping[str, object]) -> Optional[int]:
    recorded = _as_int(payload.get("episodes_recorded"))
    if recorded is not None:
        return recorded
    summaries = payload.get("episode_summaries")
    if isinstance(summaries, list):
        return len(summaries)
    return None


def _payload_algorithm(payload: Mapping[str, object]) -> str:
    hyperparameters = dict(payload.get("hyperparameters") or {})
    return str(payload.get("algorithm") or hyperparameters.get("algorithm_family") or "").lower()


def _infer_rollout_threads(payload: Optional[Mapping[str, object]]) -> int:
    if not payload:
        return 1
    hyperparameters = dict(payload.get("hyperparameters") or {})
    for key in ("n_rollout_threads", "rollout_threads", "happo_n_rollout_threads"):
        value = _as_int(hyperparameters.get(key))
        if value is not None and value > 0:
            return value
    return 1


def read_job_launcher_complete_marker(output_dir: Path) -> Optional[Dict[str, object]]:
    output_dir = Path(output_dir)
    for rel in (f"{DATA_DIR_NAME}/{JOB_LAUNCHER_COMPLETE_MARKER}", JOB_LAUNCHER_COMPLETE_MARKER):
        path = output_dir / rel
        if not path.is_file():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def write_job_launcher_complete_marker(
    output_dir: Path,
    *,
    algorithm: str,
    scenario: Optional[str],
    target_episodes: int,
    episodes_completed: int,
    source: str,
) -> Path:
    """Write-once marker: a mistaken retrain cannot downgrade a finished job."""
    output_dir = Path(output_dir)
    data_dir = output_dir / DATA_DIR_NAME
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / JOB_LAUNCHER_COMPLETE_MARKER
    if path.is_file():
        return path
    payload = {
        "algorithm": algorithm.upper(),
        "scenario": scenario,
        "target_episodes": int(target_episodes),
        "episodes_completed": int(episodes_completed),
        "source": source,
        "marked_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_mirrors([path, output_dir / JOB_LAUNCHER_COMPLETE_MARKER], payload)
    fsync_file(path)
    return path


def _timeseries_global_step_episode_denominator(
    *,
    episode_time_steps: int,
    rollout_threads: int = 1,
    algorithm: str = "",
) -> int:
    """Steps per adapter episode index stored in timeseries.csv / live_progress.json.

    HAPPO persists rank-0 CityLearn steps only (``global_step += 1`` per env step in the
  recording worker). Parallel ``n_rollout_threads`` does **not** multiply the stored
    ``global_step`` or episode index. Dividing by ``episode_time_steps * rollout_threads``
    under-counts finished HAPPO episodes and leaves completed jobs stuck at ``target-1``.
    """
    episode_time_steps = max(1, int(episode_time_steps))
    if str(algorithm or "").lower() == "happo":
        return episode_time_steps
    rollout_threads = max(1, int(rollout_threads))
    if rollout_threads > 1:
        return episode_time_steps * rollout_threads
    return episode_time_steps


def infer_completed_episodes_from_timeseries_global_step(
    output_dir: Path,
    *,
    episode_time_steps: int,
    rollout_threads: int = 1,
    algorithm: str = "",
) -> int:
    """Episode count from max ``global_step`` in timeseries.csv.

    ``global_step`` is a 0-indexed step counter written *before* its post-step
    increment, so the final step of episode index ``E`` is stored as
    ``(E+1)*denom - 1``. Counting ``(max_gs + 1) // denom`` therefore yields ``E+1``
    exactly when that final step was logged, and ``E`` while the episode is still in
    progress. The old ``max_gs // denom`` under-counted finished runs by one and left
    HAPPO/MASAC/MATD3 stuck at ``target-1`` (the 49/50 resume loop). HAPPO never writes
    an ``all_done=True`` boundary row, so this step-budget rule (not the flag) is the
    reliable completion signal.
    """
    path = Path(output_dir) / DATA_DIR_NAME / "timeseries.csv"
    if not path.is_file():
        return 0
    try:
        rows = read_csv_rows(path)
    except Exception:
        return 0
    if not rows:
        return 0
    max_gs = max(_as_int(row.get("global_step")) or 0 for row in rows)
    if max_gs <= 0:
        return 0
    denom = _timeseries_global_step_episode_denominator(
        episode_time_steps=episode_time_steps,
        rollout_threads=rollout_threads,
        algorithm=algorithm,
    )
    return max(0, int((max_gs + 1) // denom))


def _adapter_completed_episode_count_from_artifacts(
    output_dir: Path,
    *,
    live_progress: Optional[Mapping[str, object]] = None,
) -> int:
    """``completed_episode_count`` from live_progress or persisted results audit."""
    if live_progress is None:
        live_progress = read_live_progress_json(output_dir)
    if live_progress:
        counted = _as_int(live_progress.get("completed_episode_count"))
        if counted is not None and counted > 0:
            return int(counted)
    payload = read_job_results_json(output_dir)
    if not payload:
        return 0
    audit = dict(payload.get("artifact_audit") or {})
    report_source = dict(audit.get("report_source") or {})
    counted = _as_int(report_source.get("completed_episode_count"))
    if counted is not None and counted > 0:
        return int(counted)
    counted = _as_int(payload.get("completed_episode_count"))
    return int(counted) if counted is not None and counted > 0 else 0


def infer_happo_verified_completed_episodes_from_timeseries(
    data_dir: Path,
    *,
    episode_time_steps: int,
) -> int:
    """HAPPO episodes with a full per-episode row count (ignores lone ``all_done`` tails)."""
    episode_time_steps = max(1, int(episode_time_steps))
    path = Path(data_dir) / "timeseries.csv"
    if not path.is_file():
        return 0
    try:
        rows = read_csv_rows(path)
    except Exception:
        return 0
    counts: Dict[int, int] = {}
    for row in rows:
        ep = _as_int(row.get("episode"))
        if ep is None:
            continue
        counts[ep] = counts.get(ep, 0) + 1
    complete = {ep for ep, c in counts.items() if c >= episode_time_steps}
    return (max(complete) + 1) if complete else 0


def infer_trustable_completed_episodes(
    output_dir: Path,
    *,
    algorithm: str,
    episode_time_steps: int,
    rollout_threads: int = 1,
    live_progress: Optional[Mapping[str, object]] = None,
) -> int:
    """Episode count safe for skip/resume decisions (never inflated by CSV row heuristics)."""
    output_dir = Path(output_dir)
    algo = algorithm.lower()
    rollout_threads = max(1, int(rollout_threads))
    episode_time_steps = max(1, int(episode_time_steps))

    completed_gs = infer_completed_episodes_from_timeseries_global_step(
        output_dir,
        episode_time_steps=episode_time_steps,
        rollout_threads=rollout_threads,
        algorithm=algo,
    )
    if live_progress is None:
        live_progress = read_live_progress_json(output_dir)
    completed_live = 0
    if live_progress:
        completed_live = infer_completed_episodes_from_live_progress(
            live_progress,
            episode_time_steps=episode_time_steps,
            algorithm=algo,
            rollout_threads=rollout_threads,
        )
    completed_adapter = _adapter_completed_episode_count_from_artifacts(
        output_dir, live_progress=live_progress
    )

    if algo == "maac":
        _, maac_ep = find_maac_resume_checkpoint(output_dir / CHECKPOINT_DIR_NAME)
        return max(completed_gs, completed_live, int(maac_ep), completed_adapter)

    if algo == "happo":
        verified_csv = infer_happo_verified_completed_episodes_from_timeseries(
            output_dir / DATA_DIR_NAME,
            episode_time_steps=episode_time_steps,
        )
        reconciled = max(completed_gs, verified_csv, completed_adapter)
        if completed_gs > 0 and completed_live > reconciled + 1:
            # Stale live_progress.episode after preload_resume_artifacts without a Drive sync.
            return reconciled
        return max(reconciled, completed_live)

    if rollout_threads > 1:
        return max(completed_gs, completed_live, completed_adapter)

    completed_csv = infer_completed_episodes_from_timeseries_csv(
        output_dir / DATA_DIR_NAME,
        episode_time_steps=episode_time_steps,
    )
    return max(completed_gs, completed_live, completed_csv, completed_adapter)


def _checkpoint_exists_for_algorithm(output_dir: Path, algorithm: str) -> bool:
    algo = algorithm.lower()
    checkpoints_dir = Path(output_dir) / CHECKPOINT_DIR_NAME
    if algo == "happo":
        return find_harl_actor_checkpoint_dir(checkpoints_dir) is not None
    if algo == "masac":
        return find_masac_checkpoint_bundle(checkpoints_dir / "models") is not None
    if algo in {"matd3", "maddpg"}:
        return find_offpolicy_model_dir(checkpoints_dir / "offpolicy_run") is not None
    if algo == "maac":
        ckpt, _ = find_maac_resume_checkpoint(checkpoints_dir)
        return ckpt is not None
    return False


def _artifact_proves_job_complete(
    output_dir: Path,
    *,
    algorithm: str,
    target_episodes: int,
    episode_time_steps: int,
    rollout_threads: int = 1,
) -> bool:
    """Ground-truth completion from persisted training artifacts (not results.json claims)."""
    output_dir = Path(output_dir)
    target_episodes = max(1, int(target_episodes))
    episode_time_steps = max(1, int(episode_time_steps))
    rollout_threads = max(1, int(rollout_threads))
    algo = algorithm.lower()

    if algo == "maac":
        return max_maac_checkpoint_episode(output_dir) >= target_episodes

    inferred = infer_completed_episodes_from_timeseries_global_step(
        output_dir,
        episode_time_steps=episode_time_steps,
        rollout_threads=rollout_threads,
        algorithm=algo,
    )
    if algo == "happo":
        inferred = max(
            inferred,
            infer_happo_verified_completed_episodes_from_timeseries(
                output_dir / DATA_DIR_NAME,
                episode_time_steps=episode_time_steps,
            ),
            _adapter_completed_episode_count_from_artifacts(output_dir),
        )
    if inferred < target_episodes:
        return False
    return _checkpoint_exists_for_algorithm(output_dir, algo)


def job_launcher_completion_blockers(
    output_dir: Path,
    *,
    target_episodes: Optional[int] = None,
    rollout_threads: Optional[int] = None,
) -> List[str]:
    """Human-readable reasons why --skip-completed would NOT omit this job.

    ``rollout_threads`` overrides the value inferred from results.json so the preview
    (cell 2.1b / 7.1) and the resume plan interpret ``global_step`` with the same
    denominator; otherwise the diagnostic episode count could disagree with the plan.
    """
    output_dir = Path(output_dir)
    blockers: List[str] = []
    marker = read_job_launcher_complete_marker(output_dir)
    req_target = _as_int(target_episodes)

    if job_counts_as_launcher_complete(output_dir, target_episodes=req_target):
        return []

    if marker:
        m_target = _as_int(marker.get("target_episodes"))
        m_done = _as_int(marker.get("episodes_completed"))
        if req_target and m_done is not None and m_done >= req_target:
            return []
        blockers.append(
            f"marker present but episodes_completed={m_done} < target={req_target or m_target}"
        )

    payload = read_job_results_json(output_dir)
    if not payload:
        blockers.append("no results.json")
        return blockers

    if str(payload.get("status") or "").lower() == "completed_with_salvage":
        blockers.append("results.json status=completed_with_salvage")
    hyperparameters = dict(payload.get("hyperparameters") or {})
    if hyperparameters.get("run_completed_with_salvage"):
        blockers.append("hyperparameters.run_completed_with_salvage=true")
    if hyperparameters.get("run_error"):
        blockers.append(f"hyperparameters.run_error={hyperparameters.get('run_error')!r}")
    if hyperparameters.get("run_incomplete"):
        blockers.append("hyperparameters.run_incomplete=true")

    target = _resolve_job_target_episodes(payload, target_episodes=req_target)
    recorded = _recorded_episodes_from_results(payload)
    if target is not None and recorded is not None and int(recorded) < int(target):
        blockers.append(f"episodes_recorded={recorded} < target={target}")

    algo = _payload_algorithm(payload) or "unknown"
    episode_time_steps = (
        _as_int(payload.get("episode_time_steps"))
        or _as_int(hyperparameters.get("episode_time_steps"))
        or 8760
    )
    rollout_threads = (
        int(rollout_threads)
        if rollout_threads is not None and int(rollout_threads) > 0
        else _infer_rollout_threads(payload)
    )

    if algo == "maac" and target is not None:
        max_ckpt = max_maac_checkpoint_episode(output_dir)
        if 0 < max_ckpt < int(target):
            blockers.append(f"maac checkpoint_episode_{max_ckpt}.pt < target={target}")

    if target is not None and not _artifact_proves_job_complete(
        output_dir,
        algorithm=algo,
        target_episodes=int(target),
        episode_time_steps=int(episode_time_steps),
        rollout_threads=rollout_threads,
    ):
        inferred = infer_completed_episodes_from_timeseries_global_step(
            output_dir,
            episode_time_steps=int(episode_time_steps),
            rollout_threads=rollout_threads,
            algorithm=algo,
        )
        if algo == "happo":
            inferred = max(
                inferred,
                infer_happo_verified_completed_episodes_from_timeseries(
                    output_dir / DATA_DIR_NAME,
                    episode_time_steps=int(episode_time_steps),
                ),
                _adapter_completed_episode_count_from_artifacts(output_dir),
            )
        has_ckpt = _checkpoint_exists_for_algorithm(output_dir, algo)
        blockers.append(
            f"artifacts: inferred_episodes={inferred} (rollout_threads={rollout_threads}), "
            f"checkpoint={'yes' if has_ckpt else 'no'}, target={target}"
        )

    return blockers


def max_maac_checkpoint_episode(output_dir: Path) -> int:
    """Highest episode index backed by a real MAAC per-episode checkpoint.

    MAAC saves ``checkpoint_episode_{N}.pt`` after finishing episode N, so the maximum
    N on disk is ground truth for how many episodes were genuinely trained. Returns 0
    when no per-episode checkpoint exists (only ``model.pt``/``checkpoint_latest.pt``).
    """
    _, episode = find_maac_resume_checkpoint(Path(output_dir) / CHECKPOINT_DIR_NAME)
    return max(0, int(episode))


def _infer_algorithm_from_artifacts(output_dir: Path) -> str:
    live = read_live_progress_json(output_dir)
    if live and live.get("algorithm"):
        return str(live.get("algorithm") or "").lower()
    payload = read_job_results_json(output_dir)
    return _payload_algorithm(payload) if payload else ""


def job_counts_as_launcher_complete(
    output_dir: Path,
    *,
    target_episodes: Optional[int] = None,
) -> bool:
    """True only when a job genuinely finished all target episodes.

    Priority (grounded in persisted artifacts, never fragile row-count heuristics):

    1. ``job_launcher_complete.json`` write-once marker from a prior successful run.
    2. Clean ``results.json`` (no salvage/error/incomplete) with ``episodes_recorded>=target``.
    3. Artifact recovery when results.json was damaged by a mistaken retrain:
       - MAAC: ``checkpoint_episode_N.pt`` with N>=target.
       - HAPPO/MASAC/MATD3: max ``global_step`` in timeseries.csv proves >=target episodes
         (correct with n_rollout_threads>1) AND restorable checkpoints exist.
    """
    output_dir = Path(output_dir)
    req_target = _as_int(target_episodes)

    marker = read_job_launcher_complete_marker(output_dir)
    if marker:
        m_done = _as_int(marker.get("episodes_completed"))
        m_target = _as_int(marker.get("target_episodes"))
        need = req_target or m_target
        if m_done is not None and need is not None and int(m_done) >= int(need):
            return True

    payload = read_job_results_json(output_dir)
    hyperparameters = dict(payload.get("hyperparameters") or {}) if payload else {}
    algo = _payload_algorithm(payload) if payload else _infer_algorithm_from_artifacts(output_dir)
    episode_time_steps = (
        _as_int((payload or {}).get("episode_time_steps"))
        or _as_int(hyperparameters.get("episode_time_steps"))
        or 8760
    )
    rollout_threads = _infer_rollout_threads(payload)

    if payload:
        if str(payload.get("status") or "").lower() == "completed_with_salvage":
            pass  # fall through to artifact recovery
        elif hyperparameters.get("run_completed_with_salvage") or hyperparameters.get("run_error"):
            pass
        elif hyperparameters.get("run_incomplete"):
            pass
        else:
            target = _resolve_job_target_episodes(payload, target_episodes=req_target)
            recorded = _recorded_episodes_from_results(payload)

            results_ok = True
            if target is not None and recorded is not None and int(recorded) < int(target):
                results_ok = False
            audit = dict(payload.get("artifact_audit") or {})
            expected = audit.get("expected_episodes")
            if expected is not None and recorded is not None and int(recorded) < int(expected):
                results_ok = False
            if results_ok and algo == "maac" and target is not None:
                max_ckpt = max_maac_checkpoint_episode(output_dir)
                if 0 < max_ckpt < int(target):
                    results_ok = False

            if results_ok and target is not None and recorded is not None and int(recorded) >= int(target):
                write_job_launcher_complete_marker(
                    output_dir,
                    algorithm=algo or "UNKNOWN",
                    scenario=str(payload.get("scenario") or ""),
                    target_episodes=int(target),
                    episodes_completed=int(recorded),
                    source="results.json",
                )
                return True

    target = req_target or _resolve_job_target_episodes(payload or {}, target_episodes=req_target)
    if target is None or not algo:
        return False

    if _artifact_proves_job_complete(
        output_dir,
        algorithm=algo,
        target_episodes=int(target),
        episode_time_steps=int(episode_time_steps),
        rollout_threads=rollout_threads,
    ):
        inferred = infer_completed_episodes_from_timeseries_global_step(
            output_dir,
            episode_time_steps=int(episode_time_steps),
            rollout_threads=rollout_threads,
            algorithm=algo,
        )
        if algo == "happo":
            inferred = max(
                inferred,
                infer_happo_verified_completed_episodes_from_timeseries(
                    output_dir / DATA_DIR_NAME,
                    episode_time_steps=int(episode_time_steps),
                ),
                _adapter_completed_episode_count_from_artifacts(output_dir),
            )
        if algo == "maac":
            inferred = max(inferred, max_maac_checkpoint_episode(output_dir))
        write_job_launcher_complete_marker(
            output_dir,
            algorithm=algo,
            scenario=str((payload or {}).get("scenario") or ""),
            target_episodes=int(target),
            episodes_completed=max(int(inferred), int(target)),
            source="artifact_recovery",
        )
        return True

    return False


def read_live_progress_json(output_dir: Path) -> Optional[Dict[str, object]]:
    path = Path(output_dir) / "live_progress.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def infer_completed_episodes_from_live_progress(
    live_progress: Mapping[str, object],
    *,
    episode_time_steps: int,
    algorithm: str = "",
    rollout_threads: int = 1,
) -> int:
    """Episodes fully finished before an interrupt.

    For HAPPO the persisted ``episode`` field can be ahead of ``global_step`` after a
    resume preload; prefer ``global_step // episode_time_steps`` and
    ``completed_episode_count`` when present.
    """
    adapter_count = _as_int(live_progress.get("completed_episode_count"))
    global_step = _as_int(live_progress.get("global_step")) or 0
    if global_step <= 0 and (adapter_count is None or adapter_count <= 0):
        return 0

    algo = str(algorithm or "").lower()
    denom = _timeseries_global_step_episode_denominator(
        episode_time_steps=episode_time_steps,
        rollout_threads=rollout_threads,
        algorithm=algo,
    )
    # global_step is the 0-indexed pre-increment step counter (see
    # infer_completed_episodes_from_timeseries_global_step), so +1 makes the final
    # step of an episode count it as complete instead of under-counting by one.
    from_gs = max(0, int((global_step + 1) // denom)) if global_step > 0 else 0
    episode = _as_int(live_progress.get("episode")) or 0
    episode = max(0, min(int(episode), 10_000))

    if algo == "happo":
        if adapter_count is not None and adapter_count > 0:
            return max(from_gs, int(adapter_count))
        if episode > from_gs + 1:
            return from_gs
        return max(from_gs, episode)

    if adapter_count is not None and adapter_count > 0:
        return max(from_gs, int(adapter_count), episode)
    return max(from_gs, episode) if from_gs > 0 else episode


def infer_completed_episodes_from_timeseries_csv(
    data_dir: Path,
    *,
    episode_time_steps: int,
) -> int:
    """Completed-episode count from the persisted incremental timeseries.csv.

    live_progress.json is a single mutable snapshot that a buggy/old run can reset to
    ep1, but timeseries.csv is the append-only record of finished episodes. An episode
    is COMPLETE when it carries an all_done row or a full episode_time_steps row count.
    Returns max(completed_episode_index)+1 so numbering stays continuous.
    """
    episode_time_steps = max(1, int(episode_time_steps))
    path = Path(data_dir) / "timeseries.csv"
    if not path.is_file():
        return 0
    try:
        rows = read_csv_rows(path)
    except Exception:
        return 0

    counts: Dict[int, int] = {}
    done_eps: set = set()
    for row in rows:
        ep = _as_int(row.get("episode"))
        if ep is None:
            continue
        counts[ep] = counts.get(ep, 0) + 1
        flag = str(row.get("all_done", "")).strip().lower()
        if flag in {"true", "1", "1.0", "yes"}:
            done_eps.add(ep)

    complete = {ep for ep, c in counts.items() if c >= episode_time_steps} | done_eps
    return (max(complete) + 1) if complete else 0


def find_harl_actor_checkpoint_dir(checkpoints_dir: Path) -> Optional[Path]:
    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.is_dir():
        return None
    candidates = sorted(checkpoints_dir.rglob("actor_agent0.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].parent if candidates else None


def find_offpolicy_model_dir(checkpoints_dir: Path) -> Optional[Path]:
    checkpoints_dir = Path(checkpoints_dir)
    actor_files = sorted(checkpoints_dir.rglob("policy_0/actor.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not actor_files:
        actor_files = sorted(checkpoints_dir.rglob("actor.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not actor_files:
        return None
    policy_dir = actor_files[0].parent
    models_dir = policy_dir.parent
    return models_dir if models_dir.is_dir() else None


def find_masac_checkpoint_bundle(checkpoints_dir: Path) -> Optional[Dict[str, Path]]:
    checkpoints_dir = Path(checkpoints_dir)
    rnn_files = list(checkpoints_dir.rglob("*_rnn_net_params.pkl"))
    if not rnn_files:
        return None

    def _step_key(path: Path) -> int:
        stem = path.name.split("_", 1)[0]
        try:
            return int(stem)
        except ValueError:
            return -1

    rnn_path = max(rnn_files, key=_step_key)
    prefix = rnn_path.name.split("_", 1)[0] + "_"
    parent = rnn_path.parent
    qmix_path = parent / f"{prefix}qmix_net_params.pkl"
    policy_path = parent / f"{prefix}policy_net_params.pkl"
    if not qmix_path.is_file():
        return None
    bundle = {"rnn": rnn_path, "qmix": qmix_path}
    if policy_path.is_file():
        bundle["policy"] = policy_path
    return bundle


def find_maac_resume_checkpoint(checkpoints_dir: Path) -> Tuple[Optional[Path], int]:
    checkpoints_dir = Path(checkpoints_dir)
    numbered = []
    for path in checkpoints_dir.glob("checkpoint_episode_*.pt"):
        suffix = path.stem.replace("checkpoint_episode_", "")
        try:
            numbered.append((int(suffix), path))
        except ValueError:
            continue
    latest = checkpoints_dir / "checkpoint_latest.pt"
    if numbered:
        episode_no, path = max(numbered, key=lambda item: item[0])
        # A rolling intra-episode checkpoint carries warmer weights than the last
        # episode boundary. Load it when newer, but keep episode accounting at the
        # completed boundary so the resume target stays correct.
        if latest.is_file():
            try:
                if latest.stat().st_mtime > path.stat().st_mtime:
                    return latest, max(0, episode_no)
            except OSError:
                pass
        return path, max(0, episode_no)
    if latest.is_file():
        return latest, 0
    model_pt = checkpoints_dir / "model.pt"
    if model_pt.is_file():
        return model_pt, 0
    return None, 0


def discover_job_resume_plan(
    output_dir: Path,
    *,
    algorithm: str,
    target_episodes: int,
    episode_time_steps: int,
    rollout_threads: int = 1,
    allow_resume: bool = True,
) -> Dict[str, object]:
    """Plan intra-job resume from Drive/local artifacts when results.json is missing."""

    output_dir = Path(output_dir)
    target_episodes = max(1, int(target_episodes))
    episode_time_steps = max(1, int(episode_time_steps))
    rollout_threads = max(1, int(rollout_threads))
    checkpoints_dir = output_dir / CHECKPOINT_DIR_NAME

    plan: Dict[str, object] = {
        "active": False,
        "algorithm": algorithm.upper(),
        "target_episodes": target_episodes,
        "completed_episodes": 0,
        "remaining_episodes": target_episodes,
        "remaining_num_env_steps": target_episodes * episode_time_steps * rollout_threads,
        "model_dir": None,
        "maac_checkpoint": None,
        "maac_start_episode": 0,
        "note": "fresh_start",
    }

    if not allow_resume or job_counts_as_launcher_complete(output_dir, target_episodes=target_episodes):
        plan["note"] = "job_complete_or_resume_disabled"
        return plan

    algo = algorithm.lower()
    live = read_live_progress_json(output_dir)
    completed = infer_trustable_completed_episodes(
        output_dir,
        algorithm=algorithm,
        episode_time_steps=episode_time_steps,
        rollout_threads=rollout_threads,
        live_progress=live,
    )
    completed_gs = infer_completed_episodes_from_timeseries_global_step(
        output_dir,
        episode_time_steps=episode_time_steps,
        rollout_threads=rollout_threads,
        algorithm=algo,
    )
    completed_csv = infer_completed_episodes_from_timeseries_csv(
        output_dir / DATA_DIR_NAME,
        episode_time_steps=episode_time_steps,
    )

    model_dir: Optional[Path] = None
    maac_ckpt: Optional[Path] = None
    maac_start = 0

    if algo == "happo":
        model_dir = find_harl_actor_checkpoint_dir(checkpoints_dir)
    elif algo in {"matd3", "maddpg"}:
        model_dir = find_offpolicy_model_dir(checkpoints_dir / "offpolicy_run")
    elif algo == "masac":
        model_dir = None
        if find_masac_checkpoint_bundle(checkpoints_dir / "models"):
            model_dir = checkpoints_dir / "models"
    elif algo == "maac":
        maac_ckpt, maac_start = find_maac_resume_checkpoint(checkpoints_dir)
        completed = max(completed, maac_start)

    has_checkpoint = model_dir is not None or maac_ckpt is not None
    if not has_checkpoint:
        # Without restorable weights a "resume" would silently train fewer episodes
        # from a random init and corrupt the experiment. Restart fresh instead. This
        # also covers the Colab case where a crash lost checkpoints before Drive synced.
        plan["note"] = (
            "live_progress_without_weights_restart_fresh" if completed > 0 else "fresh_start"
        )
        return plan

    remaining = max(0, target_episodes - completed)
    if remaining <= 0:
        if _artifact_proves_job_complete(
            output_dir,
            algorithm=algo,
            target_episodes=target_episodes,
            episode_time_steps=episode_time_steps,
            rollout_threads=rollout_threads,
        ):
            plan["note"] = "episodes_complete_missing_results_json"
            plan["completed_episodes"] = max(completed, target_episodes)
            if job_counts_as_launcher_complete(output_dir, target_episodes=target_episodes):
                plan["note"] = "job_complete_or_resume_disabled"
            return plan
        # CSV episode-index heuristics can falsely claim completion for HAPPO; if
        # trustable counters still fall short, continue as an active resume instead.
        completed = max(completed_gs, completed)
        remaining = max(0, target_episodes - completed)

    if remaining <= 0:
        plan["note"] = "job_complete_or_resume_disabled"
        plan["completed_episodes"] = completed
        return plan

    plan.update(
        {
            "active": True,
            "completed_episodes": completed,
            "remaining_episodes": remaining,
            "remaining_num_env_steps": remaining * episode_time_steps * rollout_threads,
            "model_dir": str(model_dir) if model_dir else None,
            "maac_checkpoint": str(maac_ckpt) if maac_ckpt else None,
            "maac_start_episode": maac_start,
            "note": "resume_from_checkpoint",
            "completed_episodes_global_step": completed_gs,
            "completed_episodes_csv": completed_csv,
            "live_progress_episode": (live or {}).get("episode"),
            "live_progress_global_step": (live or {}).get("global_step"),
        }
    )
    return plan


def resolve_job_rollout_threads(
    output_dir: Path,
    algorithm: str,
    *,
    fallback: Optional[int] = None,
) -> int:
    """Rollout threads for skip/resume preview; prefer persisted hyperparameters."""
    output_dir = Path(output_dir)
    payload = read_job_results_json(output_dir)
    from_payload = _infer_rollout_threads(payload)
    if algorithm.lower() == "happo":
        if from_payload > 1:
            return from_payload
        if fallback is not None and int(fallback) > 0:
            return int(fallback)
    return max(1, from_payload)


def preview_job_launcher_decision(
    output_dir: Path,
    *,
    algorithm: str,
    target_episodes: int,
    episode_time_steps: int = 8760,
    rollout_threads: Optional[int] = None,
    allow_resume: bool = True,
) -> Dict[str, object]:
    """Mirror ``--skip-completed`` + intra-job resume for notebook cell 2.1b / launcher 7.2."""
    output_dir = Path(output_dir)
    algo = algorithm.lower()
    target_episodes = max(1, int(target_episodes))
    episode_time_steps = max(1, int(episode_time_steps))
    roll = resolve_job_rollout_threads(
        output_dir,
        algo,
        fallback=rollout_threads,
    )

    skip = job_counts_as_launcher_complete(output_dir, target_episodes=target_episodes)
    blockers: List[str] = (
        []
        if skip
        else job_launcher_completion_blockers(
            output_dir, target_episodes=target_episodes, rollout_threads=roll
        )
    )
    plan = discover_job_resume_plan(
        output_dir,
        algorithm=algo,
        target_episodes=target_episodes,
        episode_time_steps=episode_time_steps,
        rollout_threads=roll,
        allow_resume=allow_resume,
    )

    result: Dict[str, object] = {
        "algorithm": algorithm.upper(),
        "output_dir": str(output_dir),
        "target_episodes": target_episodes,
        "rollout_threads": roll,
        "skip": skip,
        "blockers": blockers,
        "plan": plan,
        "action": "run_fresh",
        "completed_episodes": 0,
        "remaining_episodes": target_episodes,
        "status_line": "PENDIENTE (fresh)",
        "launcher_line": "",
    }

    if skip:
        result.update(
            {
                "action": "skip",
                "completed_episodes": target_episodes,
                "remaining_episodes": 0,
                "status_line": f"OK COMPLETO {target_episodes}/{target_episodes} ep (se omite)",
                "launcher_line": "SKIP: existing results.json + job completo",
            }
        )
        return result

    name = algo.upper()
    if blockers:
        result["launcher_line"] = f"RUN: not skipping — {'; '.join(blockers)}"
    else:
        result["launcher_line"] = f"RUN {name}: fresh start"

    note = str(plan.get("note") or "")
    if plan.get("active"):
        completed = int(plan.get("completed_episodes") or 0)
        remaining = int(plan.get("remaining_episodes") or max(0, target_episodes - completed))
        parts: List[str] = []
        if algo == "maac" and plan.get("maac_checkpoint"):
            parts.append(Path(str(plan["maac_checkpoint"])).name)
        elif plan.get("model_dir"):
            parts.append(f"pesos:{Path(str(plan['model_dir'])).name}")
        gs = plan.get("completed_episodes_global_step")
        if gs is not None:
            parts.append(f"global_step={gs}")
        elif plan.get("completed_episodes_csv") is not None:
            parts.append(f"timeseries={plan['completed_episodes_csv']}")
        if plan.get("maac_start_episode") not in (None, 0):
            parts.append(f"ckpt_ep={plan['maac_start_episode']}")
        src = f" [{' ; '.join(parts)}]" if parts else ""
        result.update(
            {
                "action": "resume",
                "completed_episodes": completed,
                "remaining_episodes": remaining,
                "status_line": f"REANUDA ep {completed}/{target_episodes} (faltan {remaining}){src}",
            }
        )
        return result

    if note.startswith("live_progress_without_weights"):
        result.update(
            {
                "action": "restart_fresh",
                "status_line": "REINICIA fresh (hubo progreso pero NO hay pesos restaurables)",
            }
        )
        return result

    if blockers:
        result["status_line"] = (
            f"EJECUTA (launcher NO omitira): {'; '.join(blockers[:2])}"
        )
    return result


DEFAULT_REPORT_ALGORITHMS = ("happo", "masac", "matd3", "maac")
DEFAULT_REPORT_SCENARIOS = ("E1", "E2", "E3")


def build_jobs_resume_report(
    output_root: Path,
    *,
    target_episodes: int,
    algorithms: Sequence[str] = DEFAULT_REPORT_ALGORITHMS,
    scenarios: Sequence[str] = DEFAULT_REPORT_SCENARIOS,
    episode_time_steps: int = 8760,
    happo_rollout_threads: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, object]:
    """Per-job skip/resume preview for every algo x scenario under ``output_root``.

    Single source of truth for the notebook preview tables (cells 2.1b and 7.1 §4)
    so the launch flow never duplicates the loop. Each row mirrors exactly what
    ``--skip-completed`` + intra-job resume (cell 7.2) would do.
    """
    output_root = Path(output_root)
    algorithms = [str(a).lower() for a in algorithms]
    scenarios = [str(s) for s in scenarios]
    target_episodes = max(1, int(target_episodes))

    rows: List[Dict[str, object]] = []
    done = resume = pending = restart = 0
    episodes_done = 0

    for algo in algorithms:
        for scen in scenarios:
            run_dir = resolve_existing_job_run_dir(output_root, algo, scen, seed)
            if run_dir is None or not Path(run_dir).exists():
                rows.append(
                    {
                        "algorithm": algo,
                        "scenario": scen,
                        "action": "run_fresh",
                        "skip": False,
                        "completed_episodes": 0,
                        "status_line": "PENDIENTE (fresh)",
                        "launcher_line": "",
                    }
                )
                pending += 1
                continue

            dec = preview_job_launcher_decision(
                Path(run_dir),
                algorithm=algo,
                target_episodes=target_episodes,
                episode_time_steps=episode_time_steps,
                rollout_threads=happo_rollout_threads if algo == "happo" else None,
                allow_resume=True,
            )
            action = str(dec.get("action") or "")
            completed = int(dec.get("completed_episodes") or 0)
            if action == "skip":
                done += 1
                episodes_done += target_episodes
            elif action == "resume":
                resume += 1
                episodes_done += completed
            elif action == "restart_fresh":
                restart += 1
                pending += 1
            else:
                pending += 1
            rows.append(
                {
                    "algorithm": algo,
                    "scenario": scen,
                    "action": action,
                    "skip": bool(dec.get("skip")),
                    "completed_episodes": completed,
                    "status_line": str(dec.get("status_line") or ""),
                    "launcher_line": str(dec.get("launcher_line") or ""),
                }
            )

    episodes_target = len(algorithms) * len(scenarios) * target_episodes
    return {
        "output_root": str(output_root),
        "target_episodes": target_episodes,
        "jobs": rows,
        "completed": done,
        "resumable": resume,
        "pending": pending,
        "restart_fresh": restart,
        "episodes_done": episodes_done,
        "episodes_target": episodes_target,
        "progress_pct": (100.0 * episodes_done / episodes_target) if episodes_target else 0.0,
    }


def print_jobs_resume_report(
    report: Mapping[str, object],
    *,
    show_launcher_line: bool = True,
    show_footer_hint: bool = True,
) -> None:
    """Pretty-print :func:`build_jobs_resume_report` (cells 2.1b / 7.1 §4)."""
    output_root = str(report.get("output_root") or "")
    target = int(report.get("target_episodes") or 0)
    jobs = list(report.get("jobs") or [])
    print("=" * 70)
    print(f"  RUN: {Path(output_root).name}   (objetivo: {target} episodios/corrida)")
    print(f"  OUTPUT_ROOT: {output_root}")
    print("=" * 70)
    for row in jobs:
        print(f"  {str(row['algorithm']).upper():<6} {row['scenario']}  ->  {row['status_line']}")
        if show_launcher_line and not row.get("skip") and row.get("launcher_line"):
            print(f"           (7.2: {row['launcher_line']})")
    restart = int(report.get("restart_fresh") or 0)
    extra = f" (incl. {restart} reinicio-fresh)" if restart else ""
    print("-" * 70)
    print(
        f"  COMPLETOS={report.get('completed', 0)}  REANUDABLES={report.get('resumable', 0)}  "
        f"PENDIENTES={report.get('pending', 0)}{extra}  (total {len(jobs)})"
    )
    print(
        f"  Progreso global: ~{report.get('episodes_done', 0)}/{report.get('episodes_target', 0)} "
        f"episodios ({float(report.get('progress_pct') or 0.0):.1f}%) hacia el 100%"
    )
    print("=" * 70)
    if show_footer_hint:
        print("\n  Siguiente: ejecuta 6.1 -> 7.0 -> 7.1 -> 7.2 (no modifiques nada mas).")
        print("  Tras 7.1, vuelve a ejecutar esta celda 2.1b para confirmar HAPPO rollout_threads.")
        print(f"  7.2 usa --skip-completed (omite COMPLETOS) y resume intra-job (continua los")
        print(f"  REANUDABLES desde su ultimo checkpoint) hasta completar los {target} episodios.")


def write_job_resume_manifest(output_dir: Path, plan: Mapping[str, object]) -> Path:
    output_dir = Path(output_dir)
    data_dir = output_dir / DATA_DIR_NAME
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "job_resume_state.json"
    payload = dict(plan)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_masac_checkpoint_bundle(learner, bundle: Mapping[str, Path], *, use_cuda: bool) -> None:
    import torch

    map_location = "cuda:0" if use_cuda else "cpu"
    learner.eval_rnn.load_state_dict(torch.load(str(bundle["rnn"]), map_location=map_location))
    learner.eval_rnn_2.load_state_dict(torch.load(str(bundle["rnn"]), map_location=map_location))
    learner.eval_qmix_net.load_state_dict(torch.load(str(bundle["qmix"]), map_location=map_location))
    learner.eval_qmix_net_2.load_state_dict(torch.load(str(bundle["qmix"]), map_location=map_location))
    if bundle.get("policy") is not None:
        learner.agent.policy.load_state_dict(torch.load(str(bundle["policy"]), map_location=map_location))
    learner.target_rnn.load_state_dict(learner.eval_rnn.state_dict())
    learner.target_rnn_2.load_state_dict(learner.eval_rnn.state_dict())
    learner.target_qmix_net.load_state_dict(learner.eval_qmix_net.state_dict())
    learner.target_qmix_net_2.load_state_dict(learner.eval_qmix_net.state_dict())


def _episode_summaries(timeseries_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[int, List[Mapping[str, object]]] = {}

    for row in timeseries_rows:
        episode = row.get("episode")

        if episode is None:
            continue

        grouped.setdefault(int(episode), []).append(row)

    summaries = []

    for episode, rows in sorted(grouped.items()):
        reward_sums = [_as_float(row.get("reward_sum")) for row in rows]
        reward_means = [_as_float(row.get("reward_mean")) for row in rows]
        reward_sums = [value for value in reward_sums if value is not None]
        reward_means = [value for value in reward_means if value is not None]
        summaries.append({
            "episode": episode,
            "steps": len(rows),
            "reward_sum_total": None if not reward_sums else float(np.sum(reward_sums)),
            "reward_mean_average": None if not reward_means else float(np.mean(reward_means)),
            "first_global_step": rows[0].get("global_step"),
            "last_global_step": rows[-1].get("global_step"),
            "last_episode_step": rows[-1].get("episode_step"),
            "last_time_step": rows[-1].get("time_step"),
            "last_all_done": rows[-1].get("all_done"),
        })

    return summaries


def _sum_trace_columns(trace_rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> Dict[str, float]:
    totals = {column: 0.0 for column in columns}

    for row in trace_rows:
        for column in columns:
            value = _as_float(row.get(column))

            if value is not None:
                totals[column] += value

    return {column: float(value) for column, value in totals.items()}


def _trace_sampling_payload(adapter) -> Dict[str, object]:
    interval = getattr(adapter, "trace_record_interval", None) if adapter is not None else None
    detail = getattr(adapter, "trace_detail", None) if adapter is not None else None
    interval_int = _as_int(interval)
    sampled = bool(interval_int is not None and interval_int > 1)

    return {
        "trace_record_interval": interval_int,
        "trace_detail": detail,
        "trace_is_sampled": sampled,
        "trace_weighting_note": (
            "Trace-derived totals are sampled diagnostics, not full-run energy totals."
            if sampled
            else "Trace-derived totals use every recorded environment step."
        ),
    }


def _artifact_consistency_audit(
    *,
    report: Mapping[str, object],
    timeseries_rows: Sequence[Mapping[str, object]],
    trace_rows: Sequence[Mapping[str, object]],
    episode_summaries: Sequence[Mapping[str, object]],
    expected_episode_time_steps: Optional[int],
    expected_episodes: Optional[int],
) -> Dict[str, object]:
    report_source = dict(report.get("report_source", {}) or {})
    all_values = dict(report.get("all_values", {}) or {})
    completed_episode_rows = [
        row for row in timeseries_rows
        if bool(row.get("all_done"))
    ]
    last_row = dict(timeseries_rows[-1]) if timeseries_rows else {}
    expected_row_options: List[int] = []

    if expected_episode_time_steps and expected_episodes:
        expected_row_options = sorted({
            int(expected_episode_time_steps) * int(expected_episodes),
            max(int(expected_episode_time_steps) - 1, 1) * int(expected_episodes),
        })

    expected_rows = max(expected_row_options) if expected_row_options else None
    trace_totals = _sum_trace_columns(
        trace_rows,
        [
            "pv_generation_kwh",
            "pv_export_kwh",
            "ev_charge_kwh",
            "ev_v2g_export_kwh",
            "grid_import_kwh",
            "grid_export_kwh",
        ],
    )
    warnings: List[Dict[str, object]] = list(report.get("report_warnings", []) or [])

    if expected_row_options and len(timeseries_rows) not in expected_row_options:
        warnings.append({
            "severity": "warning",
            "code": "unexpected_timeseries_rows",
            "message": (
                f"Recorded {len(timeseries_rows)} timeseries rows; "
                f"expected one of {expected_row_options}."
            ),
        })

    if expected_episodes is not None and len(completed_episode_rows) < int(expected_episodes):
        warnings.append({
            "severity": "warning",
            "code": "incomplete_completed_episode_count",
            "message": f"Recorded {len(completed_episode_rows)} completed episodes; expected {expected_episodes}.",
        })

    if timeseries_rows and last_row.get("all_done") is not True and report_source.get("type") != "last_completed_episode_snapshot":
        warnings.append({
            "severity": "warning",
            "code": "last_row_not_done_without_snapshot",
            "message": "The last recorded row is not all_done and no completed snapshot was used for KPI reporting.",
        })

    if _as_float(all_values.get("pv_generation_total")) == 0.0 and trace_totals["pv_generation_kwh"] > 0.0:
        warnings.append({
            "severity": "warning",
            "code": "pv_report_zero_but_trace_positive",
            "message": "PV generation is positive in the training trace but zero in the objective report.",
        })

    if _as_float(all_values.get("ev_departure_count")) == 0.0 and (
        trace_totals["ev_charge_kwh"] > 0.0 or trace_totals["ev_v2g_export_kwh"] > 0.0
    ):
        warnings.append({
            "severity": "warning",
            "code": "ev_report_zero_but_trace_active",
            "message": "EV activity is present in the training trace but the objective report has zero departures.",
        })

    status = "ok"
    if any(item.get("severity") == "warning" for item in warnings):
        status = "warning"

    return {
        "status": status,
        "warnings": warnings,
        "report_source": report_source,
        "expected_episode_time_steps": expected_episode_time_steps,
        "expected_episodes": expected_episodes,
        "expected_timeseries_rows": expected_rows,
        "expected_timeseries_row_options": expected_row_options,
        "timeseries_rows": len(timeseries_rows),
        "trace_rows": len(trace_rows),
        "completed_episode_count_from_timeseries": len(completed_episode_rows),
        "episode_summaries": list(episode_summaries),
        "last_timeseries_row": last_row,
        "training_trace_totals": trace_totals,
        "objective_report_values": {
            "pv_generation_total": all_values.get("pv_generation_total"),
            "pv_export_total": all_values.get("pv_export_total"),
            "ev_charge_total": all_values.get("ev_charge_total"),
            "ev_departure_count": all_values.get("ev_departure_count"),
            "ev_departure_success_rate": all_values.get("ev_departure_success_rate"),
            "grid_import_control": all_values.get("grid_import_control"),
            "grid_import_baseline": all_values.get("grid_import_baseline"),
            "electricity_cost": all_values.get("electricity_cost"),
            "carbon_emissions": all_values.get("carbon_emissions"),
            "ramping_average": all_values.get("ramping_average"),
        },
    }


def _objective_kpi_rows(report: Mapping[str, object]) -> List[Dict[str, object]]:
    axes = report.get("objective_axis_kpis", {})
    rows: List[Dict[str, object]] = []

    if not isinstance(axes, Mapping):
        return rows

    for axis_code, axis in sorted(axes.items()):
        if not isinstance(axis, Mapping):
            continue

        kpis = axis.get("kpis", {})

        if not isinstance(kpis, Mapping):
            continue

        for kpi_name, payload in sorted(kpis.items()):
            payload = payload if isinstance(payload, Mapping) else {}
            trace = payload.get("trace", {})
            trace = trace if isinstance(trace, Mapping) else {}
            comparison = payload.get("comparison", {})
            comparison = comparison if isinstance(comparison, Mapping) else {}
            rows.append({
                "axis": axis_code,
                "axis_name": axis.get("name", axis_code),
                "kpi": kpi_name,
                "value": payload.get("value"),
                "source": trace.get("source"),
                "lower_is_better": trace.get("lower_is_better"),
                "baseline": comparison.get("baseline"),
                "delta_vs_baseline": comparison.get("delta_vs_baseline"),
                "improved_vs_baseline": comparison.get("improved_vs_baseline"),
                "available": comparison.get("available"),
            })

    return rows


def _axis_comparison_rows(report: Mapping[str, object]) -> List[Dict[str, object]]:
    axes = report.get("objective_axis_kpis", {})
    rows: List[Dict[str, object]] = []

    if not isinstance(axes, Mapping):
        return rows

    for axis_code, axis in sorted(axes.items()):
        if not isinstance(axis, Mapping):
            continue

        comparison = axis.get("baseline_comparison", {})
        comparison = comparison if isinstance(comparison, Mapping) else {}
        rows.append({
            "axis": axis_code,
            "axis_name": axis.get("name", axis_code),
            "comparable_kpis": comparison.get("comparable_kpis"),
            "improved_kpis": comparison.get("improved_kpis"),
            "not_improved_kpis": comparison.get("not_improved_kpis"),
        })

    return rows


def _core_kpi_rows(report: Mapping[str, object]) -> List[Dict[str, object]]:
    axis_kpis = report.get("axis_kpis", {})

    if not isinstance(axis_kpis, Mapping):
        return []

    core_names = [
        "peak_average",
        "ramping_average",
        "one_minus_load_factor_average",
        "pv_self_consumption_ratio",
        "battery_throughput_total",
        "ev_charge_total",
        "ev_v2g_export_total",
        "carbon_emissions",
        "carbon_emissions_control",
        "carbon_emissions_delta",
        "electricity_cost",
        "electricity_cost_control",
        "electricity_cost_delta",
        "price_signal_deviation",
    ]
    rows = []

    for name in core_names:
        if name in axis_kpis:
            rows.append({"kpi": name, "value": axis_kpis.get(name)})

    return rows


def _numeric_column(rows: Sequence[Mapping[str, object]], key: str) -> List[float]:
    values: List[float] = []

    for row in rows:
        value = _as_float(row.get(key))
        values.append(np.nan if value is None else value)

    return values


def _valid_pairs(rows: Sequence[Mapping[str, object]], x_key: str, y_key: str) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []

    for row in rows:
        x_value = _as_float(row.get(x_key))
        y_value = _as_float(row.get(y_key))

        if x_value is not None and y_value is not None:
            points.append((x_value, y_value))

    return points


def _rolling_mean(values: Sequence[float], window: int) -> List[float]:
    output: List[float] = []

    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = [value for value in values[start: idx + 1] if np.isfinite(value)]
        output.append(float(np.mean(chunk)) if chunk else np.nan)

    return output


def _training_efficiency_rows(timeseries_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[int, List[Mapping[str, object]]] = {}

    for row in timeseries_rows:
        episode = row.get("episode")

        if episode is None:
            continue

        grouped.setdefault(int(episode), []).append(row)

    output: List[Dict[str, object]] = []

    for episode, rows in sorted(grouped.items()):
        reward = [_as_float(row.get("reward_sum")) for row in rows]
        net_energy = [_as_float(row.get("district_net_electricity_consumption")) for row in rows]
        cost = [_as_float(row.get("district_net_electricity_consumption_cost")) for row in rows]
        emissions = [_as_float(row.get("district_net_electricity_consumption_emission")) for row in rows]
        reward = [value for value in reward if value is not None]
        net_energy = [value for value in net_energy if value is not None]
        cost = [value for value in cost if value is not None]
        emissions = [value for value in emissions if value is not None]
        reward_total = float(np.sum(reward)) if reward else None
        energy_import_total = float(np.sum([max(value, 0.0) for value in net_energy])) if net_energy else None
        abs_energy_total = float(np.sum(np.abs(net_energy))) if net_energy else None
        cost_total = float(np.sum(cost)) if cost else None
        emission_total = float(np.sum(emissions)) if emissions else None
        output.append({
            "episode": episode,
            "steps": len(rows),
            "return_total": reward_total,
            "grid_import_total": energy_import_total,
            "absolute_net_energy_total": abs_energy_total,
            "electricity_cost_total": cost_total,
            "carbon_emissions_total": emission_total,
            "return_per_grid_import": _safe_ratio(reward_total, energy_import_total),
            "return_per_cost": _safe_ratio(reward_total, cost_total),
            "return_per_kgco2": _safe_ratio(reward_total, emission_total),
        })

    return output


def _exploration_rows(trace_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[int, List[Mapping[str, object]]] = {}

    for row in trace_rows:
        episode = row.get("episode")

        if episode is None:
            continue

        grouped.setdefault(int(episode), []).append(row)

    output: List[Dict[str, object]] = []

    for episode, rows in sorted(grouped.items()):
        action_l2 = [_as_float(row.get("action_l2")) for row in rows]
        action_mean = [_as_float(row.get("action_mean")) for row in rows]
        reward = [_as_float(row.get("reward")) for row in rows]
        action_l2 = [value for value in action_l2 if value is not None]
        action_mean = [value for value in action_mean if value is not None]
        reward = [value for value in reward if value is not None]
        output.append({
            "episode": episode,
            "agent_steps": len(rows),
            "action_l2_mean": None if not action_l2 else float(np.mean(action_l2)),
            "action_l2_std": None if not action_l2 else float(np.std(action_l2)),
            "action_mean_average": None if not action_mean else float(np.mean(action_mean)),
            "action_abs_mean_average": None if not action_mean else float(np.mean(np.abs(action_mean))),
            "agent_reward_mean": None if not reward else float(np.mean(reward)),
            "agent_reward_total": None if not reward else float(np.sum(reward)),
        })

    return output


def _agent_reward_rows(trace_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Mapping[str, object]]] = {}

    for row in trace_rows:
        agent = row.get("agent")

        if agent is None:
            continue

        grouped.setdefault(str(agent), []).append(row)

    output: List[Dict[str, object]] = []

    for agent, rows in sorted(grouped.items()):
        rewards = [_as_float(row.get("reward")) for row in rows]
        action_l2 = [_as_float(row.get("action_l2")) for row in rows]
        rewards = [value for value in rewards if value is not None]
        action_l2 = [value for value in action_l2 if value is not None]
        output.append({
            "agent": agent,
            "agent_steps": len(rows),
            "reward_total": None if not rewards else float(np.sum(rewards)),
            "reward_mean": None if not rewards else float(np.mean(rewards)),
            "reward_min": None if not rewards else float(np.min(rewards)),
            "reward_max": None if not rewards else float(np.max(rewards)),
            "action_l2_mean": None if not action_l2 else float(np.mean(action_l2)),
        })

    return output


BUILDING_DETAIL_KPI_COLUMNS: Mapping[str, str] = {
    "building_energy_grid_total_import_control_kwh": "grid_import_control_kwh",
    "building_energy_grid_total_import_baseline_kwh": "grid_import_baseline_kwh",
    "building_energy_grid_total_import_delta_kwh": "grid_import_delta_kwh",
    "building_energy_grid_daily_average_import_control_kwh": "grid_import_daily_average_control_kwh",
    "building_energy_grid_daily_average_import_baseline_kwh": "grid_import_daily_average_baseline_kwh",
    "building_energy_grid_daily_average_import_delta_kwh": "grid_import_daily_average_delta_kwh",
    "building_energy_grid_total_export_control_kwh": "grid_export_control_kwh",
    "building_energy_grid_total_export_baseline_kwh": "grid_export_baseline_kwh",
    "building_energy_grid_total_export_delta_kwh": "grid_export_delta_kwh",
    "building_energy_grid_daily_average_export_control_kwh": "grid_export_daily_average_control_kwh",
    "building_energy_grid_daily_average_export_baseline_kwh": "grid_export_daily_average_baseline_kwh",
    "building_energy_grid_daily_average_export_delta_kwh": "grid_export_daily_average_delta_kwh",
    "building_energy_grid_total_net_exchange_control_kwh": "net_exchange_control_kwh",
    "building_energy_grid_total_net_exchange_baseline_kwh": "net_exchange_baseline_kwh",
    "building_energy_grid_total_net_exchange_delta_kwh": "net_exchange_delta_kwh",
    "building_energy_grid_ratio_to_baseline_import_total_ratio": "grid_import_ratio_to_baseline",
    "building_energy_grid_ratio_to_baseline_export_total_ratio": "grid_export_ratio_to_baseline",
    "building_energy_grid_ratio_to_baseline_net_exchange_total_ratio": "net_exchange_ratio_to_baseline",
    "building_cost_total_control_eur": "electricity_cost_control_eur",
    "building_cost_total_baseline_eur": "electricity_cost_baseline_eur",
    "building_cost_total_delta_eur": "electricity_cost_delta_eur",
    "building_cost_ratio_to_baseline_total_ratio": "electricity_cost_ratio_to_baseline",
    "building_emissions_total_control_kgco2": "carbon_emissions_control_kgco2",
    "building_emissions_total_baseline_kgco2": "carbon_emissions_baseline_kgco2",
    "building_emissions_total_delta_kgco2": "carbon_emissions_delta_kgco2",
    "building_emissions_ratio_to_baseline_total_ratio": "carbon_emissions_ratio_to_baseline",
    "building_solar_self_consumption_total_generation_kwh": "pv_generation_total_kwh",
    "building_solar_self_consumption_total_export_kwh": "pv_export_total_kwh",
    "building_solar_self_consumption_daily_average_generation_kwh": "pv_generation_daily_average_kwh",
    "building_solar_self_consumption_daily_average_export_kwh": "pv_export_daily_average_kwh",
    "building_solar_self_consumption_ratio_self_consumption_ratio": "pv_self_consumption_ratio",
    "building_battery_total_charge_kwh": "battery_charge_total_kwh",
    "building_battery_total_discharge_kwh": "battery_discharge_total_kwh",
    "building_battery_total_throughput_kwh": "battery_throughput_total_kwh",
    "building_battery_health_equivalent_full_cycles_count": "battery_equivalent_full_cycles",
    "building_battery_health_capacity_fade_ratio": "battery_capacity_fade_ratio",
    "building_ev_events_departure_count": "ev_departure_count",
    "building_ev_events_departure_met_count": "ev_departure_met_count",
    "building_ev_events_departure_within_tolerance_count": "ev_departure_within_tolerance_count",
    "building_ev_performance_departure_success_ratio": "ev_departure_success_rate",
    "building_ev_performance_departure_within_tolerance_ratio": "ev_departure_within_tolerance_rate",
    "building_ev_performance_departure_soc_deficit_mean_ratio": "ev_departure_soc_deficit_mean",
    "building_ev_total_charge_kwh": "ev_charge_total_kwh",
    "building_ev_total_v2g_export_kwh": "ev_v2g_export_total_kwh",
    "building_electrical_service_phase_violations_energy_total_kwh": "electrical_service_violation_total_kwh",
    "building_electrical_service_phase_violations_event_count": "electrical_service_violation_time_step_count",
    "building_electrical_service_phase_imbalance_phase_average_ratio": "phase_imbalance_ratio_average",
    "building_equity_benefit_relative_percent": "equity_relative_benefit_percent",
    # Legacy evaluate() names are accepted so tests and older runs can still be summarized.
    "electricity_consumption_control_total_kwh": "grid_import_control_kwh",
    "electricity_consumption_baseline_total_kwh": "grid_import_baseline_kwh",
    "electricity_consumption_delta_total_kwh": "grid_import_delta_kwh",
    "electricity_export_control_total_kwh": "grid_export_control_kwh",
    "electricity_export_baseline_total_kwh": "grid_export_baseline_kwh",
    "electricity_export_delta_total_kwh": "grid_export_delta_kwh",
    "zero_net_energy_control_total_kwh": "net_exchange_control_kwh",
    "zero_net_energy_baseline_total_kwh": "net_exchange_baseline_kwh",
    "zero_net_energy_delta_total_kwh": "net_exchange_delta_kwh",
    "cost_control_total_eur": "electricity_cost_control_eur",
    "cost_baseline_total_eur": "electricity_cost_baseline_eur",
    "cost_delta_total_eur": "electricity_cost_delta_eur",
    "carbon_emissions_control_total_kgco2": "carbon_emissions_control_kgco2",
    "carbon_emissions_baseline_total_kgco2": "carbon_emissions_baseline_kgco2",
    "carbon_emissions_delta_total_kgco2": "carbon_emissions_delta_kgco2",
    "pv_generation_total_kwh": "pv_generation_total_kwh",
    "pv_export_total_kwh": "pv_export_total_kwh",
    "pv_self_consumption_ratio": "pv_self_consumption_ratio",
    "bess_charge_total_kwh": "battery_charge_total_kwh",
    "bess_discharge_total_kwh": "battery_discharge_total_kwh",
    "bess_throughput_total_kwh": "battery_throughput_total_kwh",
    "bess_equivalent_full_cycles": "battery_equivalent_full_cycles",
    "bess_capacity_fade_ratio": "battery_capacity_fade_ratio",
    "ev_charge_total_kwh": "ev_charge_total_kwh",
    "ev_v2g_export_total_kwh": "ev_v2g_export_total_kwh",
}

KEY_OBSERVATION_NAMES = {
    "net_electricity_consumption",
    "solar_generation",
    "electricity_pricing",
    "carbon_intensity",
    "electrical_storage_soc",
    "cooling_storage_soc",
    "heating_storage_soc",
    "dhw_storage_soc",
    "cooling_demand",
    "heating_demand",
    "dhw_demand",
    "non_shiftable_load",
    "outdoor_dry_bulb_temperature",
    "occupant_count",
}


def _sort_agent_key(agent: object) -> Tuple[str, int, str]:
    text = str(agent)
    suffix = text.rsplit("_", 1)[-1]

    try:
        number = int(suffix)
    except ValueError:
        number = 0

    prefix = text[: -len(suffix)] if suffix else text
    return prefix, number, text


def _slug_name(name: object) -> str:
    text = str(name).strip().lower()
    output = "".join(character if character.isalnum() else "_" for character in text)
    output = "_".join(part for part in output.split("_") if part)
    return output or "value"


def _space_bound(space, attribute: str, index: int) -> Optional[float]:
    try:
        values = np.asarray(getattr(space, attribute), dtype=float).reshape(-1)
    except (AttributeError, TypeError, ValueError):
        return None

    if index >= values.size:
        return None

    return _as_float(values[index])


def _dataframe_like_rows(frame) -> List[Dict[str, object]]:
    if frame is None:
        return []

    if hasattr(frame, "to_dict"):
        try:
            return [dict(row) for row in frame.to_dict(orient="records")]
        except TypeError:
            pass

    if isinstance(frame, Sequence) and not isinstance(frame, (str, bytes)):
        rows = []
        for row in frame:
            if isinstance(row, Mapping):
                rows.append(dict(row))
        return rows

    return []


def _citylearn_kpi_frame_rows(candidate) -> List[Dict[str, object]]:
    adapter = _resolve_adapter(candidate)
    snapshot_rows = getattr(adapter, "last_completed_kpi_frame_rows", None) if adapter is not None else None

    if snapshot_rows:
        return [dict(row) for row in snapshot_rows]

    objective_env = _resolve_objective_env(candidate)

    if objective_env is None:
        return []

    frame = None

    if hasattr(objective_env, "get_kpi_frame"):
        frame = objective_env.get_kpi_frame()
    elif hasattr(objective_env, "evaluate_v2"):
        frame = objective_env.evaluate_v2()
    elif hasattr(objective_env, "env") and hasattr(objective_env.env, "evaluate_v2"):
        frame = objective_env.env.evaluate_v2()

    return _dataframe_like_rows(frame)


def _core_citylearn_env(candidate):
    objective_env = _resolve_objective_env(candidate)

    if objective_env is None:
        return None

    core = getattr(objective_env, "env", objective_env)
    return getattr(core, "unwrapped", core)


def _building_schema_rows(candidate, adapter=None) -> List[Dict[str, object]]:
    objective_env = _resolve_objective_env(candidate)
    core_env = _core_citylearn_env(candidate)

    if objective_env is None or core_env is None:
        return []

    agents = list(getattr(adapter, "agents", [])) or list(getattr(objective_env, "possible_agents", []))
    action_names = getattr(core_env, "action_names", []) or []
    observation_names = getattr(core_env, "observation_names", []) or []
    rows: List[Dict[str, object]] = []

    for agent_index, agent in enumerate(agents):
        for variable_type, all_names, space_getter in [
            ("action", action_names, getattr(objective_env, "action_space", None)),
            ("observation", observation_names, getattr(objective_env, "observation_space", None)),
        ]:
            names = list(all_names[agent_index]) if agent_index < len(all_names) else []

            if not names:
                dim = int(getattr(adapter, f"{variable_type}_dims", {}).get(agent, 0)) if adapter is not None else 0
                names = [f"{variable_type}_{idx}" for idx in range(dim)]

            space = space_getter(agent) if callable(space_getter) else None

            for variable_index, variable_name in enumerate(names):
                rows.append({
                    "agent": agent,
                    "agent_index": agent_index,
                    "variable_type": variable_type,
                    "variable_index": variable_index,
                    "variable_name": variable_name,
                    "csv_column": f"{variable_type}__{_slug_name(variable_name)}",
                    "space_low": _space_bound(space, "low", variable_index) if space is not None else None,
                    "space_high": _space_bound(space, "high", variable_index) if space is not None else None,
                })

    return rows


def _trace_agent_summary_rows(trace_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Mapping[str, object]]] = {}

    for row in trace_rows:
        agent = row.get("agent")

        if agent is None:
            continue

        grouped.setdefault(str(agent), []).append(row)

    output: List[Dict[str, object]] = []

    for agent, rows in sorted(grouped.items(), key=lambda item: _sort_agent_key(item[0])):
        summary: Dict[str, object] = {
            "agent": agent,
            "agent_steps": len(rows),
            "agent_index": rows[0].get("agent_index"),
        }

        for source_key, prefix in [
            ("reward", "team_reward"),
            ("individual_reward", "individual_reward"),
            ("action_l2", "action_l2"),
            ("action_mean", "action_mean"),
            ("observation_l2", "observation_l2"),
            ("observation_mean", "observation_mean"),
            ("grid_import_kwh", "grid_import"),
            ("grid_export_kwh", "grid_export"),
            ("net_electricity_consumption_kwh", "net_electricity_consumption"),
            ("net_electricity_consumption_cost", "electricity_cost"),
            ("net_electricity_consumption_emission", "carbon_emissions"),
        ]:
            values = [_as_float(row.get(source_key)) for row in rows]
            values = [value for value in values if value is not None]

            if not values:
                continue

            summary[f"{prefix}_mean"] = float(np.mean(values))
            summary[f"{prefix}_min"] = float(np.min(values))
            summary[f"{prefix}_max"] = float(np.max(values))
            summary[f"{prefix}_total"] = float(np.sum(values))

        output.append(summary)

    return output


def _building_kpi_summary_rows(kpi_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    output_by_building: Dict[str, Dict[str, object]] = {}

    for row in kpi_rows:
        level = str(row.get("level", "")).lower()

        if level != "building":
            continue

        building = str(row.get("name", ""))

        if not building:
            continue

        cost_function = str(row.get("cost_function", ""))
        column = BUILDING_DETAIL_KPI_COLUMNS.get(cost_function)

        if column is None:
            continue

        output_by_building.setdefault(building, {"agent": building})[column] = _as_float(row.get("value"))

    for building, row in output_by_building.items():
        grid_import = _as_float(row.get("grid_import_control_kwh"))
        grid_export = _as_float(row.get("grid_export_control_kwh"))

        if grid_import is not None and grid_export is not None:
            row["grid_import_minus_export_control_kwh"] = grid_import - grid_export

            if grid_import > grid_export + 1.0e-9:
                row["grid_role_control"] = "net_importer"
            elif grid_export > grid_import + 1.0e-9:
                row["grid_role_control"] = "net_exporter"
            else:
                row["grid_role_control"] = "balanced"

    return [
        output_by_building[name]
        for name in sorted(output_by_building, key=_sort_agent_key)
    ]


def _building_behavior_summary_rows(
    *,
    trace_rows: Sequence[Mapping[str, object]],
    kpi_rows: Sequence[Mapping[str, object]],
    schema_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    output: Dict[str, Dict[str, object]] = {}

    for row in _building_kpi_summary_rows(kpi_rows):
        output[str(row["agent"])] = dict(row)

    for row in _trace_agent_summary_rows(trace_rows):
        agent = str(row["agent"])
        target = output.setdefault(agent, {"agent": agent})
        target.update(row)

    for row in schema_rows:
        agent = str(row.get("agent", ""))

        if not agent:
            continue

        target = output.setdefault(agent, {"agent": agent})
        variable_type = str(row.get("variable_type"))
        count_key = f"{variable_type}_dim"
        target[count_key] = int(target.get(count_key, 0)) + 1
        target.setdefault("agent_index", row.get("agent_index"))

    return [
        output[name]
        for name in sorted(output, key=_sort_agent_key)
    ]


def _schema_action_names(schema_rows: Sequence[Mapping[str, object]]) -> Dict[str, Dict[int, str]]:
    output: Dict[str, Dict[int, str]] = {}

    for row in schema_rows:
        if row.get("variable_type") != "action":
            continue

        agent = str(row.get("agent"))
        index = row.get("variable_index")

        if index is None:
            continue

        output.setdefault(agent, {})[int(index)] = str(row.get("variable_name"))

    return output


def _building_trace_sample_rows(
    trace_rows: Sequence[Mapping[str, object]],
    schema_rows: Sequence[Mapping[str, object]],
    *,
    max_rows: int = 120,
) -> List[Dict[str, object]]:
    if not trace_rows:
        return []

    if len(trace_rows) <= max_rows:
        selected = list(trace_rows)
    else:
        head_count = max_rows // 2
        tail_count = max_rows - head_count
        selected = list(trace_rows[:head_count]) + list(trace_rows[-tail_count:])

    action_names = _schema_action_names(schema_rows)
    rows: List[Dict[str, object]] = []

    for row in selected:
        agent = str(row.get("agent"))
        output = dict(row)

        for index, action_name in action_names.get(agent, {}).items():
            value = row.get(f"action_{index}")

            if value is None:
                continue

            column = f"action__{_slug_name(action_name)}"
            if column in output:
                column = f"{column}_{index}"
            output[column] = value

        rows.append(output)

    return rows


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or abs(denominator) < 1e-12:
        return None

    return float(numerator / denominator)


def _save_line_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> Optional[Dict[str, object]]:
    points = []

    for row in rows:
        step = _as_float(row.get("global_step"))
        reward_sum = _as_float(row.get("reward_sum"))
        reward_mean = _as_float(row.get("reward_mean"))

        if step is not None and (reward_sum is not None or reward_mean is not None):
            points.append((step, reward_sum, reward_mean))

    if not points:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [item[0] for item in points]
    reward_sums = [np.nan if item[1] is None else item[1] for item in points]
    reward_means = [np.nan if item[2] is None else item[2] for item in points]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, reward_sums, label="reward_sum", linewidth=1.8)
    ax.plot(steps, reward_means, label="reward_mean", linewidth=1.6)
    ax.set_xlabel("global_step")
    ax.set_ylabel("reward")
    ax.set_title("Training reward trace")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": str(path), "kind": "line_plot", "name": path.name}


def _save_convergence_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> Optional[Dict[str, object]]:
    points = _valid_pairs(rows, "global_step", "reward_sum")

    if not points:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [item[0] for item in points]
    rewards = [item[1] for item in points]
    cumulative_returns = np.cumsum(rewards)
    window = max(2, min(100, int(len(rewards) / 10) or 2))
    rolling_rewards = _rolling_mean(rewards, window)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(steps, rewards, color="#406d96", linewidth=1.1, label="step_reward_sum")
    axes[0].plot(steps, rolling_rewards, color="#c46b40", linewidth=1.8, label=f"rolling_mean_{window}")
    axes[0].set_ylabel("reward")
    axes[0].set_title("MADRL convergence and learning reward")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(steps, cumulative_returns, color="#3b8c6e", linewidth=1.6, label="cumulative_return")
    axes[1].set_xlabel("global_step")
    axes[1].set_ylabel("return")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": str(path), "kind": "convergence_plot", "name": path.name}


def _save_episode_plot(path: Path, episode_summaries: Sequence[Mapping[str, object]]) -> Optional[Dict[str, object]]:
    rows = [
        row for row in episode_summaries
        if _as_float(row.get("reward_sum_total")) is not None
    ]

    if not rows:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    episodes = [int(row["episode"]) for row in rows]
    reward_totals = [_as_float(row.get("reward_sum_total")) for row in rows]
    reward_means = [_as_float(row.get("reward_mean_average")) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(episodes, reward_totals, color="#4d7c8a", label="reward_sum_total")
    if any(value is not None for value in reward_means):
        ax.plot(
            episodes,
            [np.nan if value is None else value for value in reward_means],
            color="#c76d3a",
            marker="o",
            linewidth=1.8,
            label="reward_mean_average",
        )
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.set_title("Episode reward summary")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": str(path), "kind": "bar_line_plot", "name": path.name}


def _save_learning_efficiency_plot(
    path: Path,
    efficiency_rows: Sequence[Mapping[str, object]],
) -> Optional[Dict[str, object]]:
    rows = [row for row in efficiency_rows if _as_float(row.get("return_total")) is not None]

    if not rows:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    episodes = [int(row["episode"]) for row in rows]
    return_total = [_as_float(row.get("return_total")) or 0.0 for row in rows]
    cost_total = [_as_float(row.get("electricity_cost_total")) for row in rows]
    emission_total = [_as_float(row.get("carbon_emissions_total")) for row in rows]
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(episodes, return_total, marker="o", linewidth=1.8, color="#406d96", label="return_total")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("return")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    if any(value is not None for value in cost_total):
        ax2.plot(
            episodes,
            [np.nan if value is None else value for value in cost_total],
            marker="s",
            linewidth=1.4,
            color="#c46b40",
            label="electricity_cost_total",
        )
    if any(value is not None for value in emission_total):
        ax2.plot(
            episodes,
            [np.nan if value is None else value for value in emission_total],
            marker="^",
            linewidth=1.4,
            color="#3b8c6e",
            label="carbon_emissions_total",
        )
    ax2.set_ylabel("cost / kgCO2")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    ax1.set_title("Learning efficiency by episode")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": str(path), "kind": "efficiency_plot", "name": path.name}


def _save_axis_comparison_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> Optional[Dict[str, object]]:
    filtered = [
        row for row in rows
        if _as_float(row.get("improved_kpis")) is not None
        and _as_float(row.get("not_improved_kpis")) is not None
    ]

    if not filtered:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row.get("axis")) for row in filtered]
    improved = [_as_float(row.get("improved_kpis")) or 0.0 for row in filtered]
    not_improved = [_as_float(row.get("not_improved_kpis")) or 0.0 for row in filtered]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.18, improved, width=0.36, label="improved_kpis", color="#3b8c6e")
    ax.bar(x + 0.18, not_improved, width=0.36, label="not_improved_kpis", color="#b85c5a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("KPI count")
    ax.set_title("Baseline comparison by objective axis")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": str(path), "kind": "bar_plot", "name": path.name}


def _save_baseline_gain_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> Optional[Dict[str, object]]:
    filtered = [
        row for row in rows
        if bool(row.get("available")) and _as_float(row.get("delta_vs_baseline")) is not None
    ]

    if not filtered:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    filtered = sorted(filtered, key=lambda row: (str(row.get("axis")), str(row.get("kpi"))))
    labels = [f"{row.get('axis')}:{row.get('kpi')}" for row in filtered]
    values = [_as_float(row.get("delta_vs_baseline")) or 0.0 for row in filtered]
    colors = ["#3b8c6e" if bool(row.get("improved_vs_baseline")) else "#b85c5a" for row in filtered]
    height = max(5.0, min(14.0, 0.3 * len(labels)))
    fig, ax = plt.subplots(figsize=(11, height))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors)
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("control - baseline")
    ax.set_title("KPI gain/loss versus CityLearn v2 baseline")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": str(path), "kind": "baseline_gain_plot", "name": path.name}


def _save_axis_kpi_plot(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    axis_code: str,
) -> Optional[Dict[str, object]]:
    filtered = [
        row for row in rows
        if str(row.get("axis")) == axis_code and _as_float(row.get("value")) is not None
    ]

    if not filtered:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row.get("kpi")) for row in filtered]
    values = [_as_float(row.get("value")) or 0.0 for row in filtered]
    colors = [
        "#3b8c6e" if bool(row.get("improved_vs_baseline")) else "#5d6f99"
        for row in filtered
    ]
    height = max(4.5, min(12.0, 0.32 * len(labels)))
    fig, ax = plt.subplots(figsize=(10, height))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("value")
    ax.set_title(f"{axis_code} objective KPI profile")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": str(path), "kind": "axis_kpi_plot", "name": path.name}


def _save_core_kpi_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> Optional[Dict[str, object]]:
    filtered = [
        row for row in rows
        if _as_float(row.get("value")) is not None
    ]

    if not filtered:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row.get("kpi")) for row in filtered]
    values = [_as_float(row.get("value")) or 0.0 for row in filtered]
    height = max(4.0, min(9.0, 0.42 * len(labels)))
    fig, ax = plt.subplots(figsize=(10, height))
    y = np.arange(len(labels))
    ax.barh(y, values, color="#5d6f99")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("value")
    ax.set_title("Core CityLearn v3 objective KPIs")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": str(path), "kind": "horizontal_bar_plot", "name": path.name}


def _save_citylearn_v2_timeseries_plot(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> Optional[Dict[str, object]]:
    if not rows:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    steps = _numeric_column(rows, "global_step")
    if not any(np.isfinite(value) for value in steps):
        return None

    series = [
        ("district_net_electricity_consumption", "#406d96", "kWh"),
        ("district_net_electricity_consumption_without_storage", "#8a7a4d", "kWh"),
        ("district_net_electricity_consumption_cost", "#c46b40", "cost"),
        ("district_net_electricity_consumption_emission", "#3b8c6e", "kgCO2"),
        ("electricity_price_mean", "#7d5da6", "price"),
        ("carbon_intensity_mean", "#555555", "kgCO2/kWh"),
    ]
    available = [
        (key, color, ylabel, _numeric_column(rows, key))
        for key, color, ylabel in series
    ]
    available = [
        item for item in available
        if any(np.isfinite(value) for value in item[3])
    ]

    if not available:
        return None

    fig, axes = plt.subplots(len(available), 1, figsize=(11, max(3.0, 2.0 * len(available))), sharex=True)
    if len(available) == 1:
        axes = [axes]

    for ax, (key, color, ylabel, values) in zip(axes, available):
        ax.plot(steps, values, color=color, linewidth=1.1)
        ax.set_ylabel(ylabel)
        ax.set_title(key)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("global_step")
    fig.suptitle("CityLearn v2 district time-series signals", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": str(path), "kind": "citylearn_v2_timeseries_plot", "name": path.name}


def _save_exploration_plot(
    path: Path,
    trace_rows: Sequence[Mapping[str, object]],
) -> Optional[Dict[str, object]]:
    grouped: Dict[int, List[float]] = {}

    for row in trace_rows:
        step = row.get("global_step")
        action_l2 = _as_float(row.get("action_l2"))

        if step is None or action_l2 is None:
            continue

        grouped.setdefault(int(step), []).append(action_l2)

    if not grouped:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    steps = sorted(grouped)
    means = [float(np.mean(grouped[step])) for step in steps]
    stds = [float(np.std(grouped[step])) for step in steps]
    upper = [mean + std for mean, std in zip(means, stds)]
    lower = [mean - std for mean, std in zip(means, stds)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, means, color="#406d96", linewidth=1.8, label="mean_action_l2")
    ax.fill_between(steps, lower, upper, color="#406d96", alpha=0.18, label="+/- std")
    ax.set_xlabel("global_step")
    ax.set_ylabel("action_l2")
    ax.set_title("Exploration and policy action magnitude")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": str(path), "kind": "exploration_plot", "name": path.name}


def _save_agent_reward_plot(
    path: Path,
    agent_rows: Sequence[Mapping[str, object]],
) -> Optional[Dict[str, object]]:
    filtered = [row for row in agent_rows if _as_float(row.get("reward_total")) is not None]

    if not filtered:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row.get("agent")) for row in filtered]
    values = [_as_float(row.get("reward_total")) or 0.0 for row in filtered]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels))
    ax.bar(x, values, color="#5d6f99")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("reward_total")
    ax.set_title("Agent-level reward contribution")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": str(path), "kind": "agent_reward_plot", "name": path.name}


def _write_training_figures_and_tables(
    *,
    dirs: Mapping[str, Path],
    report: Mapping[str, object],
    timeseries_rows: Sequence[Mapping[str, object]],
    trace_rows: Sequence[Mapping[str, object]],
    episode_summaries: Sequence[Mapping[str, object]],
    checkpoints: Sequence[Mapping[str, object]],
    extra_tables: Optional[Mapping[str, Sequence[Mapping[str, object]]]] = None,
) -> Dict[str, object]:
    figures_dir = dirs["figures"]
    tables_dir = dirs["tables"]
    figures: List[Dict[str, object]] = []
    tables: List[Dict[str, object]] = []

    objective_rows = _objective_kpi_rows(report)
    axis_rows = _axis_comparison_rows(report)
    core_rows = _core_kpi_rows(report)
    efficiency_rows = _training_efficiency_rows(timeseries_rows)
    exploration_rows = _exploration_rows(trace_rows)
    agent_rows = _agent_reward_rows(trace_rows)
    table_specs = [
        ("episode_summary", list(episode_summaries)),
        ("objective_kpis", objective_rows),
        ("axis_baseline_comparison", axis_rows),
        ("core_kpis", core_rows),
        ("training_efficiency", efficiency_rows),
        ("exploration_summary", exploration_rows),
        ("agent_reward_summary", agent_rows),
        ("checkpoint_inventory", list(checkpoints)),
    ]
    table_specs.extend((name, list(rows)) for name, rows in (extra_tables or {}).items())

    for table_name, rows in table_specs:
        csv_path = tables_dir / f"{table_name}.csv"
        md_path = tables_dir / f"{table_name}.md"
        write_csv(csv_path, rows)
        _write_markdown_table(md_path, rows)
        tables.extend([
            {"path": str(csv_path), "kind": "csv_table", "name": csv_path.name, "rows": len(rows)},
            {"path": str(md_path), "kind": "markdown_table", "name": md_path.name, "rows": len(rows)},
        ])

    figure_specs = [
        lambda: _save_line_plot(figures_dir / "reward_timeseries.png", timeseries_rows),
        lambda: _save_convergence_plot(figures_dir / "convergence_returns.png", timeseries_rows),
        lambda: _save_episode_plot(figures_dir / "episode_reward_summary.png", episode_summaries),
        lambda: _save_learning_efficiency_plot(figures_dir / "learning_efficiency.png", efficiency_rows),
        lambda: _save_citylearn_v2_timeseries_plot(figures_dir / "citylearn_v2_district_timeseries.png", timeseries_rows),
        lambda: _save_exploration_plot(figures_dir / "exploration_action_l2.png", trace_rows),
        lambda: _save_agent_reward_plot(figures_dir / "agent_reward_contribution.png", agent_rows),
        lambda: _save_axis_comparison_plot(figures_dir / "axis_baseline_comparison.png", axis_rows),
        lambda: _save_baseline_gain_plot(figures_dir / "baseline_gain_by_kpi.png", objective_rows),
        lambda: _save_core_kpi_plot(figures_dir / "core_kpis.png", core_rows),
        lambda: _save_axis_kpi_plot(figures_dir / "OE1_flexibility_kpis.png", objective_rows, "OE1"),
        lambda: _save_axis_kpi_plot(figures_dir / "OE2_co2_kpis.png", objective_rows, "OE2"),
        lambda: _save_axis_kpi_plot(figures_dir / "OE3_cost_kpis.png", objective_rows, "OE3"),
    ]
    figure_errors = []

    for create_figure in figure_specs:
        try:
            figure = create_figure()
        except Exception as exc:  # pragma: no cover - defensive plotting fallback
            figure_errors.append(str(exc))
            continue

        if figure is not None:
            figures.append(figure)

    manifest = {
        "figures_dir": str(figures_dir),
        "tables_dir": str(tables_dir),
        "figure_count": len(figures),
        "table_count": len(tables),
        "figures": figures,
        "tables": tables,
        "figure_errors": figure_errors,
    }
    write_json(figures_dir / "figures_manifest.json", manifest)
    return manifest


def _write_statistical_comparison_artifacts(
    output_dir: Path,
    algorithm: str,
    scenario: str,
    results_path: Path,
    timeseries_path: Path,
    trace_path: Path,
    include_trace: bool = True,
) -> Path:
    """Copy key run artifacts to OutputRoot/statistical_comparison/ with standardized names.

    Produces result_{algo}_{scenario}.json, timeseries_{algo}_{scenario}.csv and
    trace_{algo}_{scenario}.csv for direct use in cross-algorithm statistical tests.
    """
    algo_tag = f"{algorithm.lower()}_{scenario}"
    comparison_dir = output_dir.parent.parent / "statistical_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    artifact_pairs = [
        (results_path, f"result_{algo_tag}.json"),
        (timeseries_path, f"timeseries_{algo_tag}.csv"),
    ]
    if include_trace:
        artifact_pairs.append((trace_path, f"trace_{algo_tag}.csv"))

    for src, dest_name in artifact_pairs:
        if src is None or not src.exists():
            continue

        shutil.copyfile(src, comparison_dir / dest_name)

    return comparison_dir


def write_minimal_results_json(
    *,
    output_dir: Path,
    algorithm: str,
    backend: str,
    args,
    hyperparameters: Optional[Mapping[str, object]] = None,
    report: Optional[Mapping[str, object]] = None,
    error: Optional[BaseException] = None,
) -> Path:
    """Guarantee a valid results.json so a salvaged job counts as complete/skippable.

    Used only when the standard artifact writer itself fails. Without a results.json the
    launcher treats the job as failed and a relaunch would discard all trained progress.
    """

    output_dir = Path(output_dir)
    data_dir = output_dir / DATA_DIR_NAME
    data_dir.mkdir(parents=True, exist_ok=True)
    axis_metrics = None
    if isinstance(report, Mapping):
        axis_metrics = report.get("project_axis_metrics")
    payload = {
        "algorithm": algorithm,
        "backend": backend,
        "scenario": getattr(args, "scenario", None),
        "seed": getattr(args, "seed", None),
        "episode_time_steps": getattr(args, "episode_time_steps", None),
        "output_dir": str(output_dir),
        "status": "completed_with_salvage",
        "salvage_reason": (f"{type(error).__name__}: {error}" if error is not None else "artifact_writer_failed"),
        "hyperparameters": dict(hyperparameters or {}),
        "project_axis_metrics": axis_metrics,
    }
    results_path = data_dir / "results.json"
    _write_json_mirrors([results_path, output_dir / "results.json"], payload)
    fsync_file(results_path)
    fsync_file(output_dir / "results.json")
    flush_filesystem_buffers()
    print(f"[{algorithm.lower()}] wrote minimal salvage results.json -> {results_path}", flush=True)
    return results_path


def write_training_artifacts(
    *,
    output_dir: Path,
    algorithm: str,
    backend: str,
    args,
    report: Mapping[str, object],
    candidate=None,
    hyperparameters: Optional[Mapping[str, object]] = None,
    extra: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Write standardized technical outputs for a MADRL launcher."""

    dirs = ensure_artifact_layout(output_dir)
    data_dir = dirs["data"]
    adapter = _resolve_adapter(candidate)
    trace_sampling = _trace_sampling_payload(adapter)
    artifact_profile = str(getattr(args, "artifact_profile", "full") or "full").strip().lower()
    if artifact_profile not in {"full", "efficient", "minimal"}:
        artifact_profile = "full"

    write_root_timeseries = artifact_profile in {"full", "efficient"}
    write_root_trace = artifact_profile == "full"
    write_root_detail_tables = artifact_profile == "full"
    include_statistical_trace = artifact_profile == "full"

    timeseries_rows = list(getattr(adapter, "timeseries_records", [])) if adapter is not None else []
    trace_rows = list(getattr(adapter, "trace_records", [])) if adapter is not None else []
    episode_summaries = _episode_summaries(timeseries_rows)
    adapter_completed = int(getattr(adapter, "completed_episode_count", 0) or 0) if adapter else 0
    episodes_recorded = max(len(episode_summaries), adapter_completed)
    citylearn_kpi_frame_rows = _citylearn_kpi_frame_rows(candidate)
    expected_episode_time_steps = _as_int(getattr(args, "episode_time_steps", None))
    expected_episodes = _as_int((hyperparameters or {}).get("episodes"))
    artifact_audit = _artifact_consistency_audit(
        report=report,
        timeseries_rows=timeseries_rows,
        trace_rows=trace_rows,
        episode_summaries=episode_summaries,
        expected_episode_time_steps=expected_episode_time_steps,
        expected_episodes=expected_episodes,
    )
    building_schema_rows = _building_schema_rows(candidate, adapter=adapter)
    building_summary_rows = _building_behavior_summary_rows(
        trace_rows=trace_rows,
        kpi_rows=citylearn_kpi_frame_rows,
        schema_rows=building_schema_rows,
    )
    building_trace_sample_rows = _building_trace_sample_rows(trace_rows, building_schema_rows)
    building_detail_tables = {
        "building_behavior_summary": building_summary_rows,
        "building_kpis": [
            row for row in citylearn_kpi_frame_rows
            if str(row.get("level", "")).lower() == "building"
        ],
        "building_observation_action_schema": building_schema_rows,
        "building_trace_sample": building_trace_sample_rows,
    }

    timeseries_path = data_dir / "timeseries.csv"
    trace_path = data_dir / "trace.csv"
    root_timeseries_path = output_dir / "timeseries.csv"
    root_trace_path = output_dir / "trace.csv"
    timeseries_outputs = [timeseries_path]
    trace_outputs = [trace_path]
    if write_root_timeseries:
        timeseries_outputs.append(root_timeseries_path)
    if write_root_trace:
        trace_outputs.append(root_trace_path)

    _write_csv_mirrors(timeseries_outputs, timeseries_rows)
    _write_csv_mirrors(trace_outputs, trace_rows)

    building_detail_paths: Dict[str, Dict[str, object]] = {}
    for table_name, rows in building_detail_tables.items():
        data_path = data_dir / f"{table_name}.csv"
        root_path = output_dir / f"{table_name}.csv"
        detail_outputs = [data_path]
        if write_root_detail_tables:
            detail_outputs.append(root_path)
        _write_csv_mirrors(detail_outputs, rows)
        path_payload = {
            "rows": len(rows),
            "csv": str(data_path),
        }
        if write_root_detail_tables:
            path_payload["csv_root"] = str(root_path)
        building_detail_paths[table_name] = path_payload

    checkpoints = _checkpoint_files(output_dir, dirs["checkpoints"])
    normalization = dict(getattr(adapter, "normalization_metadata", {}) or {})
    checkpoint_manifest = {
        "algorithm": algorithm,
        "backend": backend,
        "checkpoint_dir": str(dirs["checkpoints"]),
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "hyperparameters": dict(hyperparameters or {}),
        "normalization": normalization,
    }
    checkpoint_manifest_path = data_dir / "checkpoint_manifest.json"
    root_checkpoint_manifest_path = output_dir / "checkpoint_manifest.json"
    _write_json_mirrors([checkpoint_manifest_path, root_checkpoint_manifest_path], checkpoint_manifest)
    artifact_audit_path = data_dir / "artifact_audit.json"
    root_artifact_audit_path = output_dir / "artifact_audit.json"
    _write_json_mirrors([artifact_audit_path, root_artifact_audit_path], artifact_audit)
    figures_manifest = _write_training_figures_and_tables(
        dirs=dirs,
        report=report,
        timeseries_rows=timeseries_rows,
        trace_rows=trace_rows,
        episode_summaries=episode_summaries,
        checkpoints=checkpoints,
        extra_tables=building_detail_tables,
    )

    results = {
        "algorithm": algorithm,
        "backend": backend,
        "scenario": args.scenario,
        "seed": args.seed,
        "episode_time_steps": args.episode_time_steps,
        "output_dir": str(output_dir),
        "artifact_layout": _artifact_layout_payload(dirs),
        "artifact_profile": artifact_profile,
        "artifact_write_policy": {
            "root_timeseries_csv": write_root_timeseries,
            "root_trace_csv": write_root_trace,
            "root_building_detail_csv": write_root_detail_tables,
            "statistical_comparison_trace_csv": include_statistical_trace,
            **trace_sampling,
        },
        "episodes_recorded": episodes_recorded,
        "timeseries_rows": len(timeseries_rows),
        "trace_rows": len(trace_rows),
        "timeseries_csv": str(timeseries_path),
        "timeseries_csv_root": str(root_timeseries_path) if write_root_timeseries else None,
        "trace_csv": str(trace_path),
        "trace_csv_root": str(root_trace_path) if write_root_trace else None,
        "checkpoint_manifest": str(checkpoint_manifest_path),
        "checkpoint_manifest_root": str(root_checkpoint_manifest_path),
        "checkpoint_count": len(checkpoints),
        "building_detail": building_detail_paths,
        "building_count": len(building_summary_rows),
        "episode_summaries": episode_summaries,
        "hyperparameters": dict(hyperparameters or {}),
        "normalization": normalization,
        "artifact_audit": artifact_audit,
        "artifact_audit_json": str(artifact_audit_path),
        "artifact_audit_json_root": str(root_artifact_audit_path),
        "figures_manifest": str(dirs["figures"] / "figures_manifest.json"),
        "figures": figures_manifest,
        "project_axis_metrics": report["project_axis_metrics"],
        "citylearn_v3_report": report,
    }

    if extra:
        results.update(dict(extra))

    results_path = data_dir / "results.json"
    root_results_path = output_dir / "results.json"
    hyperparameters = dict(hyperparameters or {})
    is_salvage = bool(
        hyperparameters.get("run_completed_with_salvage")
        or hyperparameters.get("run_error")
    )
    target_episodes = (
        _as_int(hyperparameters.get("target_episodes"))
        or _as_int(hyperparameters.get("episodes"))
        or _as_int(getattr(args, "episodes", None))
    )
    recorded_episodes = episodes_recorded
    adapter_completed = _as_int(
        (report.get("report_source") or {}).get("completed_episode_count")  # type: ignore[union-attr]
    )
    if adapter_completed is not None and adapter_completed > recorded_episodes:
        recorded_episodes = int(adapter_completed)

    if is_salvage and job_counts_as_launcher_complete(
        output_dir,
        target_episodes=target_episodes,
    ):
        salvage_path = data_dir / "results_salvage.json"
        root_salvage_path = output_dir / "results_salvage.json"
        _write_json_mirrors([salvage_path, root_salvage_path], results)
        fsync_file(salvage_path)
        print(
            f"[{algorithm.lower()}] preserved canonical results.json; "
            f"wrote salvage snapshot -> {salvage_path}",
            flush=True,
        )
    else:
        _write_json_mirrors([results_path, root_results_path], results)
        fsync_file(results_path)
        fsync_file(root_results_path)
        if (
            not is_salvage
            and target_episodes is not None
            and recorded_episodes >= int(target_episodes)
        ):
            write_job_launcher_complete_marker(
                output_dir,
                algorithm=algorithm,
                scenario=str(getattr(args, "scenario", "") or ""),
                target_episodes=int(target_episodes),
                episodes_completed=int(recorded_episodes),
                source="write_training_artifacts",
            )
    flush_filesystem_buffers()
    comparison_dir = _write_statistical_comparison_artifacts(
        output_dir=output_dir,
        algorithm=algorithm,
        scenario=str(getattr(args, "scenario", "unknown")),
        results_path=root_results_path,
        timeseries_path=root_timeseries_path,
        trace_path=root_trace_path,
        include_trace=include_statistical_trace,
    )
    return {
        "artifact_layout": _artifact_layout_payload(dirs),
        "artifact_profile": artifact_profile,
        "artifact_write_policy": results["artifact_write_policy"],
        "results_json": str(results_path),
        "results_json_root": str(root_results_path),
        "timeseries_csv": str(timeseries_path),
        "timeseries_csv_root": str(root_timeseries_path) if write_root_timeseries else None,
        "trace_csv": str(trace_path),
        "trace_csv_root": str(root_trace_path) if write_root_trace else None,
        "building_detail": building_detail_paths,
        "checkpoint_manifest": str(checkpoint_manifest_path),
        "checkpoint_manifest_root": str(root_checkpoint_manifest_path),
        "artifact_audit": artifact_audit,
        "artifact_audit_json": str(artifact_audit_path),
        "artifact_audit_json_root": str(root_artifact_audit_path),
        "figures_manifest": str(dirs["figures"] / "figures_manifest.json"),
        "figures_dir": str(dirs["figures"]),
        "tables_dir": str(dirs["tables"]),
        "figure_count": figures_manifest["figure_count"],
        "table_count": figures_manifest["table_count"],
        "checkpoint_count": len(checkpoints),
        "timeseries_rows": len(timeseries_rows),
        "trace_rows": len(trace_rows),
        "statistical_comparison_dir": str(comparison_dir),
    }


def write_training_summary(output_dir: Path, summary: Mapping[str, object]) -> Dict[str, str]:
    dirs = ensure_artifact_layout(output_dir)
    payload = dict(summary)
    payload.setdefault("artifact_layout", _artifact_layout_payload(dirs))
    root_path = output_dir / "training_summary.json"
    data_path = dirs["data"] / "training_summary.json"
    _write_json_mirrors([data_path, root_path], payload)
    return {
        "training_summary_json": str(data_path),
        "training_summary_json_root": str(root_path),
    }


def _pad(values: Sequence[float], target_dim: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    output = np.zeros(target_dim, dtype=np.float32)
    output[: values.size] = values
    return output


def build_discrete_action_table(
    *,
    action_bins: int,
    action_dim: int,
    mode: str = "axis",
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Build a backend-compatible discrete action table for CityLearn actions.

    ``cartesian`` enumerates every simultaneous actuator combination and grows as
    action_bins ** action_dim. ``axis`` keeps one no-op action plus one-axis
    actuator moves, which grows linearly and is suitable for discrete MADRL
    backends that require a single Discrete action id per agent.
    """

    action_bins = int(action_bins)
    action_dim = int(action_dim)
    mode = str(mode or "axis").strip().lower()

    if action_bins < 2:
        raise ValueError("action_bins must be >= 2 for discrete CityLearn actions.")

    if action_dim < 1:
        table = np.zeros((1, 0), dtype=np.float32)
        return table, {
            "mode": mode,
            "action_bins": action_bins,
            "max_action_dim": action_dim,
            "n_discrete_actions": 1,
            "cartesian_action_count": 1,
        }

    action_values = np.linspace(-1.0, 1.0, action_bins, dtype=np.float32)
    cartesian_action_count = int(action_bins ** action_dim)

    if mode in {"cartesian", "full"}:
        table = np.asarray(
            list(itertools.product(action_values, repeat=action_dim)),
            dtype=np.float32,
        )
        canonical_mode = "cartesian"
    elif mode in {"axis", "axiswise", "compact", "linear"}:
        rows = [np.zeros(action_dim, dtype=np.float32)]

        for dim in range(action_dim):
            for value in action_values:
                if np.isclose(float(value), 0.0):
                    continue

                row = np.zeros(action_dim, dtype=np.float32)
                row[dim] = value
                rows.append(row)

        table = np.asarray(rows, dtype=np.float32)
        canonical_mode = "axis"
    elif mode in {"shared", "scalar"}:
        rows = [np.full(action_dim, value, dtype=np.float32) for value in action_values]

        if not any(np.allclose(row, 0.0) for row in rows):
            rows.insert(0, np.zeros(action_dim, dtype=np.float32))

        table = np.asarray(rows, dtype=np.float32)
        canonical_mode = "shared"
    else:
        raise ValueError(
            "Unsupported discrete_action_mode: "
            f"{mode}. Use axis, shared or cartesian."
        )

    metadata = {
        "mode": canonical_mode,
        "action_bins": action_bins,
        "max_action_dim": action_dim,
        "n_discrete_actions": int(table.shape[0]),
        "cartesian_action_count": cartesian_action_count,
        "growth": "linear" if canonical_mode == "axis" else ("constant" if canonical_mode == "shared" else "exponential"),
        "note": (
            "Axis mode avoids the cartesian explosion from MultiDiscrete-style "
            "joint actuator combinations while preserving a single Discrete "
            "action id required by the external MASAC/MAAC backends."
        )
        if canonical_mode == "axis"
        else "",
    }
    return table, metadata


@contextlib.contextmanager
def _csv_read_cache(schema_path: Optional[str]):
    """Monkey-patch pd.read_csv with a two-level cache during CityLearnEnv init.

    CityLearn's loading.py reads weather.csv, carbon_intensity.csv and pricing.csv
    once per building (17 times each). This context manager intercepts pd.read_csv,
    serves repeated reads from an in-process dict, and persists the full cache to a
    pickle keyed by schema SHA256 so that subsequent training runs skip all CSV I/O.
    Cache TTL is 24 hours; any schema change (new SHA256) invalidates the cache.
    """
    import pandas as pd

    cache_key = None
    mem_cache: dict = {}

    if schema_path:
        try:
            schema_bytes = Path(schema_path).read_bytes()
            cache_key = hashlib.sha256(schema_bytes).hexdigest()[:20]
            cache_file = _CSV_CACHE_DIR / f"citylearn_csv_{cache_key}.pkl"
            meta_file = cache_file.with_suffix(".meta.json")
            if cache_file.exists() and meta_file.exists():
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(meta.get("ts", "1970-01-01T00:00:00+00:00"))).total_seconds() / 3600
                if meta.get("key") == cache_key and age_h <= 24:
                    with cache_file.open("rb") as fh:
                        mem_cache = pickle.load(fh)
        except Exception:
            cache_key = None
            mem_cache = {}

    original_read_csv = pd.read_csv

    def _cached(filepath_or_buffer, *args, **kwargs):
        fp = str(filepath_or_buffer) if isinstance(filepath_or_buffer, (str, Path)) else None
        if fp and fp in mem_cache:
            return mem_cache[fp].copy()
        df = original_read_csv(filepath_or_buffer, *args, **kwargs)
        if fp:
            mem_cache[fp] = df
        return df

    pd.read_csv = _cached
    try:
        yield
    finally:
        pd.read_csv = original_read_csv
        if cache_key and mem_cache:
            try:
                cache_file = _CSV_CACHE_DIR / f"citylearn_csv_{cache_key}.pkl"
                meta_file = cache_file.with_suffix(".meta.json")
                _CSV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                with cache_file.open("wb") as fh:
                    pickle.dump(mem_cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
                meta_file.write_text(
                    json.dumps({"key": cache_key, "ts": datetime.now(timezone.utc).isoformat()}),
                    encoding="utf-8",
                )
            except Exception:
                pass


def _build_ev_sim_index(core) -> Dict[str, Dict[int, Tuple[float, float]]]:
    """Build {ev_id: {t: (state, soc_arrival)}} from all charger simulations.

    Replaces the O(n_evs × n_buildings × n_chargers) nested scan in
    CityLearn's ``simulate_unconnected_ev_soc`` with a one-time O(T × C)
    build and O(1) per-step per-EV lookup.  Only call after env init.
    """
    ev_presence: Dict[str, Dict[int, Tuple[float, float]]] = {}
    _nan = float("nan")
    for building in getattr(core, "buildings", []):
        for charger in getattr(building, "electric_vehicle_chargers", None) or []:
            sim = charger.charger_simulation
            ev_id_arr = sim.electric_vehicle_id
            state_arr = sim.electric_vehicle_charger_state
            soc_arr = sim.electric_vehicle_estimated_soc_arrival
            n_ev = len(ev_id_arr)
            n_st = len(state_arr)
            n_soc = len(soc_arr)
            for t_c in range(n_ev):
                ev_id = ev_id_arr[t_c]
                if not isinstance(ev_id, str) or not ev_id or ev_id == "NONE":
                    continue
                state_v = float(state_arr[t_c]) if t_c < n_st else _nan
                soc_v = float(soc_arr[t_c]) if t_c < n_soc else _nan
                if ev_id not in ev_presence:
                    ev_presence[ev_id] = {}
                ev_presence[ev_id][t_c] = (state_v, soc_v)
    return ev_presence


def _install_ev_sim_fast_patch(core) -> bool:
    """Monkey-patch runtime_service.simulate_unconnected_ev_soc with O(1) version.

    Returns True if patch was installed, False if env has no EV runtime to patch.
    """
    runtime = getattr(core, "_runtime_service", None)
    if runtime is None or not hasattr(runtime, "simulate_unconnected_ev_soc"):
        return False
    ev_index = _build_ev_sim_index(core)
    _nan = float("nan")

    def _fast_simulate_unconnected_ev_soc(self):
        env = self.env
        rs = getattr(env, "_ev_drift_random_state", None)
        if rs is None:
            episode_idx = int(getattr(getattr(env, "episode_tracker", None), "episode", 0))
            rs = np.random.RandomState(int(env.random_seed) + episode_idx)
            env._ev_drift_random_state = rs
        t = env.time_step
        if t + 1 >= env.episode_tracker.episode_time_steps:
            return
        drift_std = self._ev_unconnected_drift_std(env.seconds_per_time_step)
        for ev in env.electric_vehicles:
            ev_id = ev.name
            ev_idx = ev_index.get(ev_id)
            found = False
            if ev_idx is not None:
                curr = ev_idx.get(t)
                nxt = ev_idx.get(t + 1)
                if curr is not None and curr[0] == 1:
                    found = True
                elif nxt is not None and nxt[0] == 1:
                    curr_state = curr[0] if curr is not None else _nan
                    if curr_state != 1:
                        found = True
                        is_incoming = curr is not None and curr[0] == 2
                        soc = curr[1] if is_incoming else nxt[1]
                        if np.isfinite(soc) and 0.0 <= soc <= 1.0:
                            ev.battery.force_set_soc(soc)
            if not found and t > 0:
                last_soc = float(ev.battery.soc[t - 1])
                variability = float(np.clip(rs.normal(1.0, drift_std), 0.6, 1.4))
                ev.battery.force_set_soc(float(np.clip(last_soc * variability, 0.0, 1.0)))

    runtime.simulate_unconnected_ev_soc = types.MethodType(_fast_simulate_unconnected_ev_soc, runtime)
    return True


class CityLearnV3BackendAdapter:
    """Common adapter around ``citylearn.v3`` for external backend launchers."""

    def __init__(
        self,
        *,
        schema_path: Optional[str] = None,
        scenario: Optional[str] = "E1",
        seed: int = 0,
        episode_time_steps: int = 4,
        action_bins: int = 3,
        discrete_action_mode: str = "axis",
        algorithm: str = "MADRL",
        live_progress_path: Optional[str] = None,
        live_progress_interval: int = 100,
        trace_record_interval: int = 1,
        trace_detail: str = "full",
        normalize_observations: bool = True,
        resume_completed_episodes: int = 0,
    ):
        ensure_project_paths()
        from citylearn.v3 import make_citylearn_v3_env, make_citylearn_v3_project_env

        self.normalize_observations = bool(normalize_observations)

        _resolved_schema = schema_path
        if not _resolved_schema:
            from citylearn.dec_pomdp import DEFAULT_17_BUILDING_EV_SCHEMA
            _resolved_schema = str(DEFAULT_17_BUILDING_EV_SCHEMA)

        with _csv_read_cache(_resolved_schema):
            if schema_path:
                self.env = make_citylearn_v3_env(
                    schema_path=schema_path,
                    scenario=scenario,
                    seed=seed,
                    episode_time_steps=episode_time_steps,
                    madrl_algorithm=algorithm,
                    normalize_observations=self.normalize_observations,
                )
            else:
                self.env = make_citylearn_v3_project_env(
                    scenario=scenario,
                    seed=seed,
                    episode_time_steps=episode_time_steps,
                    madrl_algorithm=algorithm,
                    normalize_observations=self.normalize_observations,
                )

        # Cache stable env traversal — identity never changes after init; avoids wrapper
        # traversal in every step-level hot path (_record_step, _write_live_progress, etc.)
        _obj = getattr(self.env, "env", getattr(self.env, "unwrapped", self.env))
        self._cached_objective_env = _obj
        self._cached_core_env = getattr(_obj, "unwrapped", _obj)

        self.scenario = scenario
        self.algorithm = str(algorithm).upper()
        self.seed_value = seed
        self.episode_time_steps = int(episode_time_steps)
        self.agents = list(self.env.possible_agents)
        self._agent_index_map: Dict[str, int] = {a: i for i, a in enumerate(self.agents)}
        self.n_agents = len(self.agents)
        self.num_agents = self.n_agents
        self.action_bins = int(action_bins)
        self.discrete_action_mode = str(discrete_action_mode or "axis").strip().lower()
        self.live_progress_path = Path(live_progress_path) if live_progress_path else None
        self.live_progress_interval = max(1, int(live_progress_interval))
        self.trace_record_interval = max(0, int(trace_record_interval))
        self.trace_detail = str(trace_detail or "full").strip().lower()
        if self.trace_detail not in {"full", "compact"}:
            self.trace_detail = "full"

        self._obs_spaces = {agent: self.env.observation_space(agent) for agent in self.agents}
        self._act_spaces = {agent: self.env.action_space(agent) for agent in self.agents}

        # Static EV-type code observations appended after CityLearn's own observations.
        # Encoding (already normalized [0,1]): moto_lineal=1/3, mototaxi=2/3, v2g=1.0
        # One value per charger socket in the building, read from schema hardware.ev_type.
        self._ev_type_codes_per_agent: Dict[str, np.ndarray] = {}
        try:
            import json as _json_mod
            _EV_TYPE_NORM: Dict[str, float] = {'moto_lineal': 1/3, 'mototaxi': 2/3, 'v2g': 1.0, 'camioneta': 1.0}
            _schema_data = _json_mod.loads(Path(_resolved_schema).read_text(encoding="utf-8"))
            for _agent in self.agents:
                _bdata = _schema_data.get("buildings", {}).get(_agent, {})
                _chargers = _bdata.get("chargers", {})
                _codes = [
                    _EV_TYPE_NORM.get(
                        _chargers[_k].get("hardware", {}).get("ev_type", "moto_lineal"), 1/3
                    )
                    for _k in _chargers.keys()
                ]
                self._ev_type_codes_per_agent[_agent] = np.array(_codes, dtype=np.float32)
        except Exception:
            for _agent in self.agents:
                self._ev_type_codes_per_agent[_agent] = np.zeros(0, dtype=np.float32)

        # CityLearn's own dims (before appending type codes)
        self._citylearn_observation_dims = {
            agent: int(space.shape[0])
            for agent, space in self._obs_spaces.items()
        }
        self.observation_dims = {
            agent: self._citylearn_observation_dims[agent] + len(self._ev_type_codes_per_agent.get(agent, []))
            for agent in self.agents
        }
        self.action_dims = {
            agent: int(space.shape[0])
            for agent, space in self._act_spaces.items()
        }
        self.observation_names_by_agent = self._variable_names_by_agent(
            "observation_names",
            self.observation_dims,
            "observation",
        )
        self.action_names_by_agent = self._variable_names_by_agent(
            "action_names",
            self.action_dims,
            "action",
        )
        self.max_observation_dim = max(self.observation_dims.values())
        self.max_action_dim = max(self.action_dims.values())
        self.state_dim = int(self.env.state_space.shape[0])
        self.padded_joint_observation_dim = self.n_agents * self.max_observation_dim

        obs_low = 0.0 if self.normalize_observations else -SPACE_BOUND
        obs_high = 1.0 if self.normalize_observations else SPACE_BOUND
        self.continuous_observation_space = [
            spaces.Box(
                low=obs_low,
                high=obs_high,
                shape=(self.max_observation_dim,),
                dtype=np.float32,
            )
            for _ in self.agents
        ]
        self.ctde_share_observation_space = [
            spaces.Box(low=obs_low, high=obs_high, shape=(self.state_dim,), dtype=np.float32)
            for _ in self.agents
        ]
        self.padded_share_observation_space = [
            spaces.Box(
                low=obs_low,
                high=obs_high,
                shape=(self.padded_joint_observation_dim,),
                dtype=np.float32,
            )
            for _ in self.agents
        ]
        self.continuous_action_space = [
            spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.max_action_dim,),
                dtype=np.float32,
            )
            for _ in self.agents
        ]

        self.discrete_action_table, self.discrete_action_metadata = build_discrete_action_table(
            action_bins=self.action_bins,
            action_dim=self.max_action_dim,
            mode=self.discrete_action_mode,
        )
        self.n_discrete_actions = int(self.discrete_action_table.shape[0])
        self.discrete_action_space = [
            spaces.Discrete(self.n_discrete_actions)
            for _ in self.agents
        ]
        self._live_progress_lock = threading.Lock()
        self._live_fps_last_step: Optional[int] = None
        self._live_fps_last_mono: Optional[float] = None
        self._live_fps_ema: Optional[float] = None

        self._last_observations: Dict[str, np.ndarray] = {}
        self.global_step = 0
        self.reset_count = 0
        self.trace_records: List[Dict[str, object]] = []
        self.timeseries_records: List[Dict[str, object]] = []
        # Resume-safe incremental persistence: flush each finished episode to
        # data/{timeseries,trace}.csv so an interrupted Colab run keeps every
        # already-trained episode on Drive, and a resumed run CONTINUES the same
        # files instead of overwriting them with only the new tail.
        _inc_data_dir = (
            self.live_progress_path.parent / DATA_DIR_NAME
            if self.live_progress_path is not None
            else None
        )
        self._incremental_ts_path = (_inc_data_dir / "timeseries.csv") if _inc_data_dir else None
        self._incremental_trace_path = (_inc_data_dir / "trace.csv") if _inc_data_dir else None
        self._ts_flushed_count = 0
        self._trace_flushed_count = 0
        self._ts_fieldnames: Optional[List[str]] = None
        self._trace_fieldnames: Optional[List[str]] = None
        self._incremental_enabled = self.live_progress_path is not None
        self.completed_episode_count = 0
        self.last_completed_episode: Optional[int] = None
        self.last_completed_global_step: Optional[int] = None
        self.last_completed_time_step: Optional[int] = None
        self.last_completed_objectives: Optional[Dict[str, object]] = None
        self.last_completed_kpi_frame_rows: List[Dict[str, object]] = []
        self.last_completed_snapshot_error: Optional[str] = None
        self._reset_reward_accumulators()
        self.reward_metadata = self._reward_metadata()
        self.normalization_metadata = self._normalization_metadata()
        # Install O(1) EV simulation lookup (replaces O(n_evs×n_buildings×n_chargers) scan)
        _install_ev_sim_fast_patch(self._cached_core_env)
        # Resume offset applied INSIDE the constructor so it also runs in HAPPO's
        # SubprocVecEnv worker processes (where the live_progress/CSV-writing adapter
        # actually lives). Single-process backends (MASAC/MATD3/MAAC) can also use it.
        if int(resume_completed_episodes or 0) > 0:
            self.preload_resume_artifacts(int(resume_completed_episodes))

    def seed(self, seed: int) -> None:
        self.seed_value = int(seed)

    def clear_records(self) -> None:
        self.global_step = 0
        self.reset_count = 0
        self.trace_records = []
        self.timeseries_records = []
        self._ts_flushed_count = 0
        self._trace_flushed_count = 0
        self.completed_episode_count = 0
        self.last_completed_episode = None
        self.last_completed_global_step = None
        self.last_completed_time_step = None
        self.last_completed_objectives = None
        self.last_completed_kpi_frame_rows = []
        self.last_completed_snapshot_error = None
        self._reset_reward_accumulators()

    def _reset_reward_accumulators(self) -> None:
        self._episode_reward_sums: Dict[int, float] = {}
        self._episode_reward_mean_sums: Dict[int, float] = {}
        self._episode_reward_counts: Dict[int, int] = {}
        self._episode_reward_mean_counts: Dict[int, int] = {}
        self._total_reward_sum = 0.0
        self._total_reward_mean_sum = 0.0
        self._total_reward_count = 0
        self._total_reward_mean_count = 0

    def _update_reward_accumulators(self, episode: int, timeseries_row: Mapping[str, object]) -> None:
        reward_sum = _as_float(timeseries_row.get("reward_sum"))
        reward_mean = _as_float(timeseries_row.get("reward_mean"))

        if reward_sum is None and reward_mean is None:
            return

        self._episode_reward_counts[episode] = self._episode_reward_counts.get(episode, 0) + 1
        self._total_reward_count += 1

        if reward_sum is not None:
            self._episode_reward_sums[episode] = self._episode_reward_sums.get(episode, 0.0) + reward_sum
            self._total_reward_sum += reward_sum

        if reward_mean is not None:
            self._episode_reward_mean_sums[episode] = self._episode_reward_mean_sums.get(episode, 0.0) + reward_mean
            self._episode_reward_mean_counts[episode] = self._episode_reward_mean_counts.get(episode, 0) + 1
            self._total_reward_mean_sum += reward_mean
            self._total_reward_mean_count += 1

    def reset(self) -> List[np.ndarray]:
        observations, _infos = self.env.reset(seed=self.seed_value)
        self.reset_count += 1
        self._last_observations = {
            agent: self._observation_array_for_agent(agent, observation)
            for agent, observation in observations.items()
        }
        return self.padded_observations(self._last_observations)

    def close(self) -> None:
        self.env.close()

    def preload_resume_artifacts(self, completed_episodes: int) -> Dict[str, int]:
        """Resume-safe preload: keep prior per-step rows for already COMPLETED episodes
        so the resumed job continues timeseries.csv/trace.csv instead of restarting them.

        Truncates any partial-episode tail (that episode is re-run from checkpoint),
        rewrites the incremental CSVs to the clean kept set, and advances global_step /
        reset_count so episode numbering stays continuous (episodes 0..N seamless).
        """
        summary = {"timeseries_rows": 0, "trace_rows": 0, "completed_episodes": 0}
        completed = max(0, int(completed_episodes))
        if not self._incremental_enabled or completed <= 0:
            return summary

        episode_length = max(int(self.episode_time_steps), 1)

        # timeseries: keep only fully-completed episodes (episode < completed) and
        # dedup by (episode, episode_step) keeping the first occurrence, so a buggy
        # prior run that appended duplicate low-episode rows cannot corrupt integrity.
        ts_rows = read_csv_rows(self._incremental_ts_path) if self._incremental_ts_path else []
        kept_ts = _dedup_rows_keep_first(
            [r for r in ts_rows if (_as_int(r.get("episode")) or 0) < completed],
            ("episode", "episode_step"),
        )
        if kept_ts:
            self.timeseries_records = list(kept_ts)
            self._ts_fieldnames = sorted({k for r in kept_ts for k in r.keys()})
            self._ts_flushed_count = len(kept_ts)
            write_csv(self._incremental_ts_path, kept_ts)  # clean truncation
            for row in kept_ts:
                ep = _as_int(row.get("episode"))
                if ep is not None:
                    self._update_reward_accumulators(int(ep), row)
            summary["timeseries_rows"] = len(kept_ts)

        # trace: same completed-episode filter + dedup (per agent within a step).
        trace_rows = read_csv_rows(self._incremental_trace_path) if self._incremental_trace_path else []
        kept_trace = _dedup_rows_keep_first(
            [r for r in trace_rows if (_as_int(r.get("episode")) or 0) < completed],
            ("episode", "episode_step", "agent"),
        )
        if kept_trace:
            self.trace_records = list(kept_trace)
            self._trace_fieldnames = sorted({k for r in kept_trace for k in r.keys()})
            self._trace_flushed_count = len(kept_trace)
            write_csv(self._incremental_trace_path, kept_trace)
            summary["trace_rows"] = len(kept_trace)

        # Continue numbering from the completed-episode boundary.
        self.global_step = completed * episode_length
        self.reset_count = completed
        self.completed_episode_count = completed
        summary["completed_episodes"] = completed
        if self.live_progress_path is not None:
            self._atomic_write_live_payload(
                {
                    "global_step": self.global_step,
                    "episode": completed,
                    "episode_step": 0,
                    "completed_episode_count": completed,
                    "scenario": self.scenario,
                    "algorithm": self.algorithm,
                    "episode_time_steps": episode_length,
                    "live_status": "resume_preload",
                    "live_status_updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        return summary

    def _incremental_flush_records(
        self,
        *,
        path: Optional[Path],
        all_records: List[Dict[str, object]],
        flushed_count: int,
        fieldnames_ref: Optional[List[str]],
    ) -> Tuple[int, Optional[List[str]]]:
        """Flush new records to an incremental CSV.

        - Stable schema (no new columns): fast append.
        - Schema growth (new columns appear): full atomic rewrite from memory so
          the on-disk file stays a valid, complete CSV for analysis.
        """
        if path is None or len(all_records) <= flushed_count:
            return flushed_count, fieldnames_ref
        new_rows = all_records[flushed_count:]
        new_keys = sorted({k for r in new_rows for k in r.keys()})
        if fieldnames_ref is None:
            fieldnames_ref = new_keys
            _append_csv_rows_stable_schema(path, new_rows, fieldnames_ref)
        elif set(new_keys) - set(fieldnames_ref):
            # Schema grew (e.g. trace_detail=full adds action_i columns mid-run).
            fieldnames_ref = sorted({k for r in all_records for k in r.keys()})
            write_csv(path, all_records)
        else:
            _append_csv_rows_stable_schema(path, new_rows, fieldnames_ref)
        return len(all_records), fieldnames_ref

    def _flush_incremental_artifacts(self) -> None:
        """Persist finished-episode rows to data/{timeseries,trace}.csv on Drive."""
        if not self._incremental_enabled:
            return
        self._ts_flushed_count, self._ts_fieldnames = self._incremental_flush_records(
            path=self._incremental_ts_path,
            all_records=self.timeseries_records,
            flushed_count=self._ts_flushed_count,
            fieldnames_ref=self._ts_fieldnames,
        )
        self._trace_flushed_count, self._trace_fieldnames = self._incremental_flush_records(
            path=self._incremental_trace_path,
            all_records=self.trace_records,
            flushed_count=self._trace_flushed_count,
            fieldnames_ref=self._trace_fieldnames,
        )

    def _reward_metadata(self) -> Dict[str, object]:
        reward_function = getattr(self._core_env(), "reward_function", None)
        metadata = getattr(reward_function, "metadata", None)

        if isinstance(metadata, Mapping):
            return dict(metadata)

        return {
            "function": reward_function.__class__.__name__ if reward_function is not None else None,
            "algorithm": self.algorithm,
            "scenario": self.scenario,
        }

    def _normalization_metadata(self) -> Dict[str, object]:
        return {
            "normalize_observations": self.normalize_observations,
            "observation_method": (
                "CityLearn NormalizedObservationWrapper plus adapter clipping to [0, 1]."
                if self.normalize_observations
                else "Raw CityLearn observations in physical/source units."
            ),
            "normalize_actions": False,
            "action_note": "CityLearn storage/EV actions already use the native control-space bounds exposed by action_space.",
        }

    def _variable_names_by_agent(
        self,
        attribute_name: str,
        dimensions: Mapping[str, int],
        fallback_prefix: str,
    ) -> Dict[str, List[str]]:
        objective_env = self._objective_env()
        core_env = self._core_env()
        source_env = objective_env if attribute_name == "observation_names" else core_env
        all_names = getattr(source_env, attribute_name, []) or getattr(core_env, attribute_name, []) or []
        output: Dict[str, List[str]] = {}

        for index, agent in enumerate(self.agents):
            names = list(all_names[index]) if index < len(all_names) else []
            dimension = int(dimensions.get(agent, len(names)))

            if len(names) < dimension:
                names.extend(f"{fallback_prefix}_{idx}" for idx in range(len(names), dimension))

            output[agent] = names[:dimension]

        return output

    def _observation_array_for_agent(self, agent: str, observation) -> np.ndarray:
        cl_dim = self._citylearn_observation_dims[agent]
        array = np.asarray(observation, dtype=np.float32).reshape(-1)[:cl_dim]

        if self.normalize_observations:
            array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
            array = np.clip(array, 0.0, 1.0)

        type_codes = self._ev_type_codes_per_agent.get(agent)
        if type_codes is not None and len(type_codes) > 0:
            array = np.concatenate([array, type_codes])

        return array.astype(np.float32)

    def _state_array(self) -> np.ndarray:
        state = np.asarray(self.env.state(), dtype=np.float32).reshape(-1)

        if self.normalize_observations:
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
            state = np.clip(state, 0.0, 1.0)

        return state.astype(np.float32)

    def padded_observations(self, observations: Mapping[str, np.ndarray]) -> List[np.ndarray]:
        return [
            _pad(self._observation_array_for_agent(agent, observations[agent]), self.max_observation_dim)
            for agent in self.agents
        ]

    def ctde_state(self) -> np.ndarray:
        return self._state_array()

    def repeated_ctde_state(self) -> List[np.ndarray]:
        state = self.ctde_state()
        return [state.copy() for _ in self.agents]

    def padded_joint_observation(self, observations: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(self.padded_observations(observations), dtype=np.float32)

    def repeated_padded_joint_observation(self, observations: Mapping[str, np.ndarray]) -> List[np.ndarray]:
        state = self.padded_joint_observation(observations)
        return [state.copy() for _ in self.agents]

    def _continuous_action_for_agent(self, agent: str, action: Sequence[float]) -> np.ndarray:
        dim = self.action_dims[agent]
        action = np.asarray(action, dtype=np.float32).reshape(-1)[:dim]
        space = self._act_spaces[agent]
        return np.clip(action, np.asarray(space.low).reshape(-1), np.asarray(space.high).reshape(-1)).astype(np.float32)

    def _discrete_index(self, action: object) -> int:
        array = np.asarray(action)

        if array.ndim == 0 or array.size == 1:
            return int(array.reshape(-1)[0])

        return int(np.argmax(array.reshape(-1)))

    def discrete_to_continuous(self, agent: str, action: object) -> np.ndarray:
        index = self._discrete_index(action) % self.n_discrete_actions
        return self._continuous_action_for_agent(agent, self.discrete_action_table[index])

    def step_continuous(self, actions: Sequence[Sequence[float]]):
        action_dict = {
            agent: self._continuous_action_for_agent(agent, actions[i])
            for i, agent in enumerate(self.agents)
        }
        return self._step(action_dict)

    def step_discrete(self, actions: Sequence[object]):
        action_dict = {
            agent: self.discrete_to_continuous(agent, actions[i])
            for i, agent in enumerate(self.agents)
        }
        return self._step(action_dict)

    def _step(self, action_dict: Mapping[str, np.ndarray]):
        self._clear_current_step_device_consumption()
        observations, rewards, terminations, truncations, infos = self.env.step(action_dict)
        self._last_observations = {
            agent: self._observation_array_for_agent(agent, observations[agent])
            for agent in self.agents
        }
        dones = {
            agent: bool(terminations.get(agent, False) or truncations.get(agent, False))
            for agent in self.agents
        }
        self._record_step(action_dict, observations, rewards, dones, infos)
        return self._last_observations, rewards, dones, infos

    def _clear_current_step_device_consumption(self) -> None:
        citylearn_env = self._core_env()

        for building in getattr(citylearn_env, "buildings", []) or []:
            for attribute in (
                "cooling_device",
                "heating_device",
                "dhw_device",
                "non_shiftable_load_device",
                "electrical_storage",
            ):
                self._set_device_electricity_consumption(getattr(building, attribute, None), 0.0)

            for charger in getattr(building, "electric_vehicle_chargers", []) or []:
                self._set_device_electricity_consumption(charger, 0.0)

            for washing_machine in getattr(building, "washing_machines", []) or []:
                self._set_device_electricity_consumption(washing_machine, 0.0)

    @staticmethod
    def _set_device_electricity_consumption(device, value: float) -> None:
        setter = getattr(device, "set_electricity_consumption", None)

        if setter is None:
            return

        try:
            setter(value, enforce_polarity=False)
        except TypeError:
            setter(value)

    def _objective_env(self):
        return self._cached_objective_env

    def _core_env(self):
        return self._cached_core_env

    def _building_for_agent(self, citylearn_env, agent: str):
        buildings = list(getattr(citylearn_env, "buildings", []) or [])
        agent_index = self._agent_index_map.get(agent, 0)
        return buildings[agent_index] if agent_index < len(buildings) else None

    def _building_step_metrics(self, building, time_step: int) -> Dict[str, object]:
        if building is None:
            return {}

        net = _series_value(building, "net_electricity_consumption", time_step)
        net_without_storage = _series_value(building, "net_electricity_consumption_without_storage", time_step)
        solar = _series_value(building, "solar_generation", time_step)
        storage = getattr(building, "electrical_storage", None)
        ev_consumption = 0.0
        ev_has_value = False

        for charger in getattr(building, "electric_vehicle_chargers", []) or []:
            value = _series_value(charger, "electricity_consumption", time_step)

            if value is not None:
                ev_has_value = True
                ev_consumption += value

        metrics = {
            "net_electricity_consumption_kwh": net,
            "grid_import_kwh": None if net is None else max(net, 0.0),
            "grid_export_kwh": None if net is None else max(-net, 0.0),
            "net_electricity_consumption_without_storage_kwh": net_without_storage,
            "net_electricity_consumption_cost": _series_value(building, "net_electricity_consumption_cost", time_step),
            "net_electricity_consumption_emission": _series_value(building, "net_electricity_consumption_emission", time_step),
            "solar_generation_raw_kwh": solar,
            "pv_generation_kwh": None if solar is None else max(-solar, 0.0),
            "pv_export_kwh": None if net is None or solar is None else min(max(-solar, 0.0), max(-net, 0.0)),
            "electrical_storage_soc": _series_value(storage, "soc", time_step) if storage is not None else None,
            "electrical_storage_energy_balance_kwh": _series_value(storage, "energy_balance", time_step) if storage is not None else None,
            "electrical_storage_electricity_consumption_kwh": _series_value(building, "electrical_storage_electricity_consumption", time_step),
            "ev_electricity_consumption_kwh": ev_consumption if ev_has_value else None,
            "ev_charge_kwh": max(ev_consumption, 0.0) if ev_has_value else None,
            "ev_v2g_export_kwh": max(-ev_consumption, 0.0) if ev_has_value else None,
            "electricity_price": _series_value(getattr(building, "pricing", None), "electricity_pricing", time_step),
            "carbon_intensity": _series_value(getattr(building, "carbon_intensity", None), "carbon_intensity", time_step),
        }
        return metrics

    def _named_action_values(self, agent: str, action: np.ndarray) -> Dict[str, object]:
        output: Dict[str, object] = {}

        for index, value in enumerate(action):
            if index >= len(self.action_names_by_agent.get(agent, [])):
                continue

            name = self.action_names_by_agent[agent][index]
            column = f"action__{_slug_name(name)}"
            if column in output:
                column = f"{column}_{index}"
            output[column] = _as_float(value)

        return output

    def _selected_observation_values(self, agent: str, observation: np.ndarray) -> Dict[str, object]:
        output: Dict[str, object] = {}
        names = self.observation_names_by_agent.get(agent, [])

        for index, value in enumerate(observation):
            if index >= len(names):
                continue

            name = names[index]
            normalized_name = str(name)
            if (
                normalized_name not in KEY_OBSERVATION_NAMES
                and "electric_vehicle" not in normalized_name
                and not normalized_name.startswith("charging_")
            ):
                continue

            column = f"observation__{_slug_name(normalized_name)}"
            if column in output:
                column = f"{column}_{index}"
            output[column] = _as_float(value)

        return output

    def _record_step(self, action_dict, observations, rewards, dones, infos) -> None:
        # Fast path for non-persisting rollout workers (HAPPO SubprocVecEnv rank>0:
        # live_progress_path is None -> _incremental_enabled is False). Their
        # timeseries/trace records are never flushed to disk nor read by the main
        # process, so all per-step bookkeeping here is dead work that steals CPU from
        # the single-thread env.step (the throughput bottleneck). Only advance
        # global_step to keep the worker's episode counter consistent. Convergence-
        # neutral: rewards returned to the runner come from env.step(), not from here.
        if not self._incremental_enabled:
            self.global_step += 1
            return

        episode_length = max(int(self.episode_time_steps), 1)
        episode = int(self.global_step // episode_length)
        episode_step = int(self.global_step % episode_length)
        citylearn_env = self._core_env()
        time_step = int(getattr(citylearn_env, "time_step", self.global_step))
        reward_values = [_as_float(rewards.get(agent)) for agent in self.agents]
        reward_values = [value for value in reward_values if value is not None]
        all_done = bool(all(dones.values())) if dones else False
        timeseries_row = {
            "global_step": self.global_step,
            "episode": episode,
            "episode_step": episode_step,
            "reset_count": self.reset_count,
            "time_step": time_step,
            "scenario": self.scenario,
            "algorithm": self.algorithm,
            "reward_function": self.reward_metadata.get("function"),
            "reward_profile": self.reward_metadata.get("profile", {}).get("profile_name")
            if isinstance(self.reward_metadata.get("profile"), Mapping)
            else self.reward_metadata.get("profile"),
            "reward_axis_weights": self.reward_metadata.get("axis_weights"),
            "reward_sum": None if not reward_values else float(np.sum(reward_values)),
            "reward_mean": None if not reward_values else float(np.mean(reward_values)),
            "all_done": all_done,
            "district_net_electricity_consumption": _district_current_scalar(citylearn_env, "net_electricity_consumption", time_step),
            "district_net_electricity_consumption_without_storage": _district_current_scalar(citylearn_env, "net_electricity_consumption_without_storage", time_step),
            "district_net_electricity_consumption_cost": _district_current_scalar(citylearn_env, "net_electricity_consumption_cost", time_step),
            "district_net_electricity_consumption_emission": _district_current_scalar(citylearn_env, "net_electricity_consumption_emission", time_step),
            "electricity_price_mean": _mean_current_building_signal(citylearn_env, "pricing", "electricity_pricing", time_step),
            "carbon_intensity_mean": _mean_current_building_signal(citylearn_env, "carbon_intensity", "carbon_intensity", time_step),
        }
        self.timeseries_records.append(timeseries_row)
        self._update_reward_accumulators(episode, timeseries_row)
        self._write_live_progress(timeseries_row)

        record_trace = (
            self.trace_record_interval > 0
            and (self.global_step % self.trace_record_interval == 0 or all_done)
        )

        if record_trace:
            state_stats = _compact_array_stats(self.ctde_state())
            for agent in self.agents:
                action = np.asarray(action_dict.get(agent, []), dtype=float).reshape(-1)
                observation = np.asarray(observations.get(agent, []), dtype=float).reshape(-1)
                action_stats = _compact_array_stats(action)
                observation_stats = _compact_array_stats(observation)
                row = {
                    "global_step": self.global_step,
                    "episode": episode,
                    "episode_step": episode_step,
                    "time_step": time_step,
                    "scenario": self.scenario,
                    "algorithm": self.algorithm,
                    "reward_function": self.reward_metadata.get("function"),
                    "reward_profile": timeseries_row.get("reward_profile"),
                    "agent": agent,
                    "agent_index": self._agent_index_map[agent],
                    "state_dim": int(self.state_dim),
                    "state_mean": state_stats["mean"],
                    "state_min": state_stats["min"],
                    "state_max": state_stats["max"],
                    "state_l2": state_stats["l2"],
                    "reward": _as_float(rewards.get(agent)),
                    "done": bool(dones.get(agent, False)),
                    "action_dim": int(action.size),
                    "action_mean": action_stats["mean"],
                    "action_min": action_stats["min"],
                    "action_max": action_stats["max"],
                    "action_l2": action_stats["l2"],
                    "observation_dim": int(observation.size),
                    "observation_mean": observation_stats["mean"],
                    "observation_min": observation_stats["min"],
                    "observation_max": observation_stats["max"],
                    "observation_l2": observation_stats["l2"],
                }
                row.update(self._building_step_metrics(self._building_for_agent(citylearn_env, agent), time_step))

                if self.trace_detail == "full":
                    row.update(self._named_action_values(agent, action))
                    row.update(self._selected_observation_values(agent, observation))

                    for idx, value in enumerate(action[: self.max_action_dim]):
                        row[f"action_{idx}"] = _as_float(value)

                info = dict(infos.get(agent, {})) if isinstance(infos, Mapping) else {}
                if "individual_reward" in info:
                    row["individual_reward"] = _as_float(info["individual_reward"])

                self.trace_records.append(row)

        if timeseries_row.get("all_done") is True:
            self._capture_completed_episode_snapshot(timeseries_row)
            self._flush_incremental_artifacts()

        self.global_step += 1

    def _capture_completed_episode_snapshot(self, timeseries_row: Mapping[str, object]) -> None:
        """Snapshot official KPIs before backend wrappers reset the environment."""

        self.completed_episode_count += 1
        self.last_completed_episode = _as_int(timeseries_row.get("episode"))
        self.last_completed_global_step = _as_int(timeseries_row.get("global_step"))
        self.last_completed_time_step = _as_int(timeseries_row.get("time_step"))

        try:
            from citylearn.v3.objectives import evaluate_objectives

            objective_env = self._objective_env()
            self.last_completed_objectives = dict(evaluate_objectives(objective_env))

            frame = None
            if hasattr(objective_env, "get_kpi_frame"):
                frame = objective_env.get_kpi_frame()
            elif hasattr(objective_env, "evaluate_v2"):
                frame = objective_env.evaluate_v2()
            elif hasattr(objective_env, "env") and hasattr(objective_env.env, "evaluate_v2"):
                frame = objective_env.env.evaluate_v2()

            self.last_completed_kpi_frame_rows = _dataframe_like_rows(frame)
            self.last_completed_snapshot_error = None
        except Exception as exc:  # pragma: no cover - defensive reporting fallback
            self.last_completed_snapshot_error = str(exc)

    def _write_live_progress(self, timeseries_row: Mapping[str, object]) -> None:
        if self.live_progress_path is None:
            return

        if int(timeseries_row["global_step"]) % self.live_progress_interval != 0:
            return

        episode = int(timeseries_row.get("episode", 0))
        episode_steps_recorded = self._episode_reward_counts.get(episode, 0)
        episode_return_cumulative = self._episode_reward_sums.get(episode)
        episode_reward_mean_sum = self._episode_reward_mean_sums.get(episode)
        episode_reward_mean_count = self._episode_reward_mean_counts.get(episode, 0)
        episode_reward_mean_cumulative = (
            None
            if episode_reward_mean_sum is None or episode_reward_mean_count == 0
            else float(episode_reward_mean_sum / episode_reward_mean_count)
        )
        total_reward_mean_cumulative = (
            None
            if self._total_reward_mean_count == 0
            else float(self._total_reward_mean_sum / self._total_reward_mean_count)
        )

        _reward_fn = getattr(self._core_env(), "reward_function", None)
        _breakdown = getattr(_reward_fn, "_last_component_breakdown", {})
        _comps = _breakdown.get("components") or []
        _cstats: Dict[str, object] = {}
        if _comps:
            for _k in ("flex", "carbon", "cost", "ev"):
                _vals = [c.get(_k, 0.0) for c in _comps if _k in c]
                if _vals:
                    _cstats[f"reward_component_{_k}_mean"] = float(np.mean(_vals))

        global_step = int(timeseries_row["global_step"])
        now_mono = time.monotonic()
        fps: Optional[float] = self._live_fps_ema
        if self._live_fps_last_step is not None and self._live_fps_last_mono is not None:
            step_delta = global_step - self._live_fps_last_step
            time_delta = now_mono - self._live_fps_last_mono
            if step_delta > 0 and time_delta > 0.05:
                instant_fps = step_delta / time_delta
                if self._live_fps_ema is None:
                    self._live_fps_ema = instant_fps
                else:
                    self._live_fps_ema = 0.65 * self._live_fps_ema + 0.35 * instant_fps
                fps = self._live_fps_ema
        self._live_fps_last_step = global_step
        self._live_fps_last_mono = now_mono

        payload = {
            "global_step": global_step,
            "episode": int(timeseries_row["episode"]),
            "episode_step": int(timeseries_row["episode_step"]),
            "time_step": int(timeseries_row["time_step"]),
            "completed_episode_count": int(self.completed_episode_count),
            "scenario": self.scenario,
            "algorithm": self.algorithm,
            "reward_function": self.reward_metadata.get("function"),
            "reward_profile": timeseries_row.get("reward_profile"),
            "reward_axis_weights": self.reward_metadata.get("axis_weights"),
            "instant_reward_sum": timeseries_row.get("reward_sum"),
            "instant_reward_mean": timeseries_row.get("reward_mean"),
            "reward_sum": timeseries_row.get("reward_sum"),
            "reward_mean": timeseries_row.get("reward_mean"),
            "reward_sum_semantics": "instant_step_sum_kept_for_backward_compatibility",
            "reward_mean_semantics": "instant_step_mean_kept_for_backward_compatibility",
            "episode_return_cumulative": episode_return_cumulative,
            "episode_reward_mean_cumulative": episode_reward_mean_cumulative,
            "episode_steps_recorded": episode_steps_recorded,
            "total_return_cumulative": float(self._total_reward_sum) if self._total_reward_count else None,
            "total_reward_mean_cumulative": total_reward_mean_cumulative,
            "total_steps_recorded": self._total_reward_count,
            "district_net_electricity_consumption": timeseries_row.get("district_net_electricity_consumption"),
            "district_net_electricity_consumption_cost": timeseries_row.get("district_net_electricity_consumption_cost"),
            "district_net_electricity_consumption_emission": timeseries_row.get("district_net_electricity_consumption_emission"),
            "electricity_price_mean": timeseries_row.get("electricity_price_mean"),
            "carbon_intensity_mean": timeseries_row.get("carbon_intensity_mean"),
            "reward_component_flex_mean": _cstats.get("reward_component_flex_mean"),
            "reward_component_carbon_mean": _cstats.get("reward_component_carbon_mean"),
            "reward_component_cost_mean": _cstats.get("reward_component_cost_mean"),
            "reward_component_ev_mean": _cstats.get("reward_component_ev_mean"),
            "reward_team_reward": _breakdown.get("team_reward"),
            "reward_district_import_kwh": _breakdown.get("district_import"),
            "fps": fps,
            "mean_return": episode_reward_mean_cumulative,
            "episode_time_steps": self.episode_time_steps,
            "live_status": "env_step",
            "live_status_updated_at": datetime.now(timezone.utc).isoformat(),
            "backend_training_active": False,
            "live_progress_semantics": (
                "global_step changes only when CityLearn advances the environment; "
                "heartbeat fields change while an external backend is updating neural networks."
            ),
        }
        self._atomic_write_live_payload(payload)

    def write_live_heartbeat(self, *, stage: str, note: Optional[str] = None) -> None:
        """Refresh live_progress.json while external backends train between env steps."""

        if self.live_progress_path is None:
            return

        now = datetime.now(timezone.utc).isoformat()
        with self._live_progress_lock:
            payload: Dict[str, object] = {}

            if self.live_progress_path.exists():
                try:
                    payload = json.loads(self.live_progress_path.read_text(encoding="utf-8"))
                except Exception:
                    payload = {}

            if not payload:
                episode_length = max(int(self.episode_time_steps), 1)
                last_recorded_step = max(int(self.global_step) - 1, 0)
                payload = {
                    "global_step": last_recorded_step,
                    "episode": int(last_recorded_step // episode_length),
                    "episode_step": int(last_recorded_step % episode_length),
                    "time_step": int(last_recorded_step % episode_length),
                    "scenario": self.scenario,
                    "algorithm": self.algorithm,
                    "reward_function": self.reward_metadata.get("function"),
                    "reward_profile": self.reward_metadata.get("profile", {}).get("profile_name")
                    if isinstance(self.reward_metadata.get("profile"), Mapping)
                    else self.reward_metadata.get("profile"),
                    "reward_axis_weights": self.reward_metadata.get("axis_weights"),
                }

            payload["live_status"] = str(stage)
            payload["live_status_updated_at"] = now
            payload["backend_training_active"] = True
            payload["backend_training_heartbeat_count"] = int(payload.get("backend_training_heartbeat_count") or 0) + 1
            payload["live_progress_semantics"] = (
                "global_step changes only when CityLearn advances the environment; "
                "heartbeat fields change while an external backend is updating neural networks."
            )
            if note:
                payload["live_status_note"] = str(note)

            self._write_live_payload_locked(payload)

    def _atomic_write_live_payload(self, payload: Mapping[str, object]) -> None:
        with self._live_progress_lock:
            self._write_live_payload_locked(payload)

    def _write_live_payload_locked(self, payload: Mapping[str, object]) -> None:
        self.live_progress_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.live_progress_path.with_name(
            f"{self.live_progress_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        fsync_file(tmp_path)

        try:
            tmp_path.replace(self.live_progress_path)
        except PermissionError:
            try:
                self.live_progress_path.unlink(missing_ok=True)
                tmp_path.replace(self.live_progress_path)
            except PermissionError:
                tmp_path.unlink(missing_ok=True)
                return
        fsync_file(self.live_progress_path)

    def kpi_summary(self) -> Dict[str, object]:
        return {
            "kpis": self.env.get_kpis(),
            "kpi_frame_shape": tuple(self.env.get_kpi_frame().shape),
        }


class CityLearnHARLEnv:
    """HARL ShareVecEnv-compatible CityLearn v3 adapter."""

    def __init__(self, **kwargs):
        self.adapter = CityLearnV3BackendAdapter(**kwargs)
        self.n_agents = self.adapter.n_agents
        self.observation_space = self.adapter.continuous_observation_space
        self.share_observation_space = self.adapter.ctde_share_observation_space
        self.action_space = self.adapter.continuous_action_space

    def seed(self, seed: int) -> None:
        self.adapter.seed(seed)

    def reset(self):
        obs = self.adapter.reset()
        return obs, self.adapter.repeated_ctde_state(), None

    def step(self, actions):
        observations, rewards, dones, infos = self.adapter.step_continuous(actions)
        obs = self.adapter.padded_observations(observations)
        info_list = [
            {"agent": agent, **dict(infos.get(agent, {}))}
            for agent in self.adapter.agents
        ]
        return (
            obs,
            self.adapter.repeated_ctde_state(),
            [[float(rewards[agent])] for agent in self.adapter.agents],
            [bool(dones[agent]) for agent in self.adapter.agents],
            info_list,
            None,
        )

    def close(self) -> None:
        self.adapter.close()


class CityLearnSMACDiscreteEnv:
    """SMAC-like discrete adapter for the MASAC/mSAC repository."""

    def __init__(self, **kwargs):
        self.adapter = CityLearnV3BackendAdapter(**kwargs)
        self.n_agents = self.adapter.n_agents
        self.n_actions = self.adapter.n_discrete_actions
        self.obs_shape = self.adapter.max_observation_dim
        self.state_shape = self.adapter.state_dim
        self.episode_limit = self.adapter.episode_time_steps
        self._last_obs = self.adapter.reset()

    def reset(self):
        self._last_obs = self.adapter.reset()
        return self._last_obs

    def get_obs(self):
        return self._last_obs

    def get_state(self):
        return self.adapter.ctde_state()

    def get_avail_agent_actions(self, agent_id: int):
        return np.ones(self.n_actions, dtype=np.float32)

    def get_env_info(self):
        return {
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "state_shape": self.state_shape,
            "obs_shape": self.obs_shape,
            "episode_limit": self.episode_limit,
        }

    def step(self, actions):
        observations, rewards, dones, infos = self.adapter.step_discrete(actions)
        self._last_obs = self.adapter.padded_observations(observations)
        reward = float(np.mean([rewards[agent] for agent in self.adapter.agents]))
        terminated = bool(all(dones.values())) if dones else False
        info = {"battle_won": False, "citylearn_infos": infos}
        return reward, terminated, info

    def save_replay(self):
        return None

    def close(self):
        self.adapter.close()


class CityLearnMAACVecEnv:
    """Single-thread MAAC-compatible discrete CityLearn v3 adapter."""

    def __init__(self, **kwargs):
        self.adapter = CityLearnV3BackendAdapter(**kwargs)
        self.num_envs = 1
        self.n_agents = self.adapter.n_agents
        self.observation_space = self.adapter.continuous_observation_space
        self.action_space = self.adapter.discrete_action_space

    def reset(self):
        return np.asarray([self.adapter.reset()], dtype=np.float32)

    def step(self, actions):
        env_actions = actions[0]
        observations, rewards, dones, infos = self.adapter.step_discrete(env_actions)
        obs = np.asarray([self.adapter.padded_observations(observations)], dtype=np.float32)
        reward_array = np.asarray(
            [[float(rewards[agent]) for agent in self.adapter.agents]],
            dtype=np.float32,
        )
        done_array = np.asarray(
            [[bool(dones[agent]) for agent in self.adapter.agents]],
            dtype=bool,
        )
        return obs, reward_array, done_array, [{"citylearn_infos": infos}]

    def close(self):
        self.adapter.close()


class CityLearnOffPolicyVecEnv:
    """Single-thread off-policy PyTorch MATD3-compatible CityLearn v3 adapter."""

    def __init__(self, **kwargs):
        self.adapter = CityLearnV3BackendAdapter(**kwargs)
        self.num_envs = 1
        self.num_agents = self.adapter.n_agents
        self.n_agents = self.adapter.n_agents
        self.observation_space = self.adapter.continuous_observation_space
        self.share_observation_space = self.adapter.padded_share_observation_space
        self.action_space = self.adapter.continuous_action_space

    def reset(self):
        return np.asarray([self.adapter.reset()], dtype=np.float32)

    def step(self, actions):
        observations, rewards, dones, infos = self.adapter.step_continuous(actions[0])
        obs = np.asarray([self.adapter.padded_observations(observations)], dtype=np.float32)
        reward_array = np.asarray(
            [[float(rewards[agent]) for agent in self.adapter.agents]],
            dtype=np.float32,
        )
        done_array = np.asarray(
            [[bool(dones[agent]) for agent in self.adapter.agents]],
            dtype=bool,
        )
        return obs, reward_array, done_array, [{"citylearn_infos": infos}]

    def close(self):
        self.adapter.close()


class NoOpLogger:
    def add_scalar(self, *args, **kwargs):
        return None

    def add_scalars(self, *args, **kwargs):
        return None

    def export_scalars_to_json(self, *args, **kwargs):
        return None

    def close(self):
        return None


class FiniteTensorBoardWriter:
    """SummaryWriter guard that skips non-finite scalar values and audits them."""

    def __init__(self, writer, audit_path: Optional[Path] = None):
        self.writer = writer
        self.audit_path = Path(audit_path) if audit_path is not None else None
        self.skipped_count = 0
        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def _scalar(self, value):
        if isinstance(value, np.ndarray):
            if value.size != 1:
                return None
            value = value.reshape(-1)[0]
        elif hasattr(value, "detach"):
            try:
                detached = value.detach()
                if hasattr(detached, "numel") and detached.numel() != 1:
                    return None
                value = detached.cpu().item()
            except Exception:
                return None

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None

        return numeric if np.isfinite(numeric) else None

    def _audit_skip(self, tag: str, global_step, value) -> None:
        self.skipped_count += 1
        if self.audit_path is None:
            return
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tag": str(tag),
            "global_step": global_step,
            "value_repr": repr(value),
            "reason": "non_finite_tensorboard_scalar",
        }
        with self.audit_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        clean_value = self._scalar(scalar_value)
        if clean_value is None:
            self._audit_skip(str(tag), global_step, scalar_value)
            return None
        return self.writer.add_scalar(tag, clean_value, global_step, *args, **kwargs)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, *args, **kwargs):
        clean_values = {}
        for key, value in dict(tag_scalar_dict or {}).items():
            clean_value = self._scalar(value)
            tag = f"{main_tag}/{key}"
            if clean_value is None:
                self._audit_skip(tag, global_step, value)
            else:
                clean_values[key] = clean_value
        if not clean_values:
            return None
        return self.writer.add_scalars(main_tag, clean_values, global_step, *args, **kwargs)

    def export_scalars_to_json(self, *args, **kwargs):
        return self.writer.export_scalars_to_json(*args, **kwargs)

    def close(self):
        return self.writer.close()


def install_finite_optimizer_step_guard(
    optimizer_specs: Sequence[Mapping[str, object]],
    audit_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Skip optimizer steps whose gradients contain NaN or Inf values."""

    import torch

    path = Path(audit_path) if audit_path is not None else None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)

    installed = 0
    seen_optimizers = set()

    def iter_named_parameters(spec: Mapping[str, object]):
        module = spec.get("module")
        if module is not None and hasattr(module, "named_parameters"):
            yield from module.named_parameters()
            return

        parameters = spec.get("parameters")
        if parameters is None:
            return
        for index, parameter in enumerate(list(parameters)):
            yield f"parameter_{index}", parameter

    def audit_skip(owner: str, bad_gradients: Sequence[str]) -> None:
        if path is None:
            return
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "owner": owner,
            "bad_gradients": list(bad_gradients),
            "reason": "non_finite_gradient_optimizer_step_skipped",
        }
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    for spec in optimizer_specs:
        optimizer = spec.get("optimizer")
        if optimizer is None:
            continue

        optimizer_id = id(optimizer)
        if optimizer_id in seen_optimizers or getattr(optimizer, "_citylearn_finite_guard_installed", False):
            continue

        owner = str(spec.get("owner") or f"optimizer_{installed}")
        parameters = list(iter_named_parameters(spec))
        if not parameters:
            continue

        seen_optimizers.add(optimizer_id)
        original_step = optimizer.step

        def guarded_step(
            *args,
            _owner=owner,
            _parameters=parameters,
            _optimizer=optimizer,
            _original_step=original_step,
            **kwargs,
        ):
            bad_gradients = []
            for name, parameter in _parameters:
                gradient = getattr(parameter, "grad", None)
                if gradient is None:
                    continue
                try:
                    is_finite = bool(torch.isfinite(gradient.detach()).all().item())
                except Exception as exc:
                    bad_gradients.append(f"{name}:finite_check_error={exc!r}")
                    continue
                if not is_finite:
                    bad_gradients.append(str(name))

            if bad_gradients:
                try:
                    _optimizer.zero_grad(set_to_none=True)
                except TypeError:
                    _optimizer.zero_grad()
                audit_skip(_owner, bad_gradients)
                return None

            return _original_step(*args, **kwargs)

        optimizer.step = guarded_step
        optimizer._citylearn_finite_guard_installed = True
        installed += 1

    return {
        "installed_optimizers": installed,
        "audit_path": str(path) if path is not None else None,
    }


def install_harl_finite_optimizer_step_guard(runner, audit_path: Optional[Path] = None) -> Dict[str, object]:
    """Skip HARL optimizer steps that contain non-finite gradients."""

    specs = []
    for agent_index, actor_policy in enumerate(getattr(runner, "actor", []) or []):
        specs.append(
            {
                "owner": f"actor_agent{agent_index}",
                "module": getattr(actor_policy, "actor", None),
                "optimizer": getattr(actor_policy, "actor_optimizer", None),
            }
        )

    critic_policy = getattr(runner, "critic", None)
    specs.append(
        {
            "owner": "critic",
            "module": getattr(critic_policy, "critic", None),
            "optimizer": getattr(critic_policy, "critic_optimizer", None),
        }
    )

    return install_finite_optimizer_step_guard(specs, audit_path)
