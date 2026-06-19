"""Notebook-friendly live monitor for Colab A100 CityLearn v3 MADRL runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence


ALGORITHMS = ("happo", "masac", "matd3", "maac")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return root / path


def resolve_output_root(root: Path, requested: str) -> Optional[Path]:
    if requested:
        return resolve_path(root, requested)

    for rel in ("outputs/latest_colab_output_root.txt", "outputs/latest_visible_training_output_root.txt"):
        path = root / rel
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return resolve_path(root, value)

    outputs = root / "outputs"
    if outputs.exists():
        candidates = [
            child
            for child in outputs.iterdir()
            if child.is_dir() and (child / "official_full_status.json").exists()
        ]
        if candidates:
            return max(candidates, key=lambda item: item.stat().st_mtime)

    return None


def path_for_job(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def run_text(cmd: Sequence[str], *, timeout: int = 10) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception as exc:
        return f"ERROR: {exc}"
    return (proc.stdout or proc.stderr or "").strip()


def file_age_seconds(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    return round(time.time() - path.stat().st_mtime, 1)


def result_artifact_exists(run_dir: Path) -> bool:
    return (run_dir / "data" / "results.json").exists() or (run_dir / "results.json").exists()


def job_state(status: Mapping[str, object], algorithm: str, scenario: str, root: Path) -> str:
    jobs = [
        job
        for job in status.get("jobs", [])
        if str(job.get("name")) == algorithm and str(job.get("scenario")) == scenario
    ]
    if jobs:
        job = jobs[-1]
        if job.get("completed_at") is None:
            return "running"
        if job.get("exit_code") == 0:
            return "skipped/done" if job.get("skipped") else "done"
        return "failed"

    output_root = path_for_job(root, str(status.get("output_root", "")))
    run_dir = output_root / algorithm / f"{scenario}_seed_{int(status.get('seed') or 0)}"
    if result_artifact_exists(run_dir):
        return "done/artifact"
    return "queued"


def active_job(status: Mapping[str, object], root: Path) -> Optional[Mapping[str, object]]:
    jobs = list(status.get("jobs", []))
    for job in jobs:
        if job.get("completed_at") is None and not job.get("planned_only"):
            return job

    latest_job = None
    latest_mtime = None
    for job in jobs:
        output_dir = job.get("output_dir")
        if not output_dir:
            continue
        progress = path_for_job(root, str(output_dir)) / "live_progress.json"
        if not progress.exists():
            continue
        mtime = progress.stat().st_mtime
        if latest_mtime is None or mtime > latest_mtime:
            latest_mtime = mtime
            latest_job = job
    return latest_job


def print_status(status: Mapping[str, object], root: Path) -> None:
    print(f"Estado global: {status.get('status')}")
    print(
        "Dataset: {} | Escenarios: {} | Seed: {} | Episodios: {} | Pasos: {}".format(
            status.get("dataset"),
            ",".join(status.get("scenarios", []) or [str(status.get("scenario"))]),
            status.get("seed"),
            status.get("episodes"),
            status.get("num_env_steps"),
        )
    )
    print(f"CUDA: {status.get('cuda')} | Torch: {status.get('torch')}")
    print("")
    print("Plan completo por eje y MADRL")
    for scenario in status.get("scenarios", []) or [status.get("scenario")]:
        states = [f"{algorithm}:{job_state(status, algorithm, str(scenario), root)}" for algorithm in ALGORITHMS]
        print(f"  {scenario}: " + " | ".join(states))


def print_gpu() -> None:
    if not shutil.which("nvidia-smi"):
        print("GPU: nvidia-smi no disponible")
        return
    print("")
    print("GPU")
    text = run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader",
        ]
    )
    print(text if text else "sin datos")
    apps = run_text(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    if apps and not apps.startswith("ERROR:"):
        print("Procesos GPU:")
        print(apps)


def print_progress(status: Mapping[str, object], root: Path) -> None:
    job = active_job(status, root)
    if not job:
        print("")
        print("Progreso vivo: sin job activo todavia.")
        return

    run_dir = path_for_job(root, str(job.get("output_dir")))
    progress_path = run_dir / "live_progress.json"
    progress = read_json(progress_path)
    print("")
    print("Progreso, metricas y recompensas")
    print(f"MADRL activo: {str(job.get('name')).upper()} | Escenario: {job.get('scenario')}")
    print(f"Directorio: {job.get('output_dir')}")

    if not progress:
        print("Progreso vivo aun no disponible; aparece despues del primer intervalo de pasos.")
        return

    total_steps = int(status.get("num_env_steps") or 0)
    episode_steps = int(status.get("episode_time_steps") or 0)
    episodes = int(status.get("episodes") or 0)
    global_step = int(progress.get("global_step") or 0)
    episode = int(progress.get("episode") or 0) + 1
    episode_step = int(progress.get("episode_step") or 0)
    pct = round(100.0 * global_step / total_steps, 2) if total_steps else 0.0
    ep_pct = round(100.0 * episode_step / episode_steps, 2) if episode_steps else 0.0
    weights = progress.get("reward_axis_weights") or {}

    print(
        "  episodio={}/{} paso_episodio={}/{} ({}%) paso_global={}/{} ({}%)".format(
            episode,
            episodes,
            episode_step,
            episode_steps,
            ep_pct,
            global_step,
            total_steps,
            pct,
        )
    )
    print(f"  global_step={global_step} time_step={progress.get('time_step')} live_status={progress.get('live_status')}")
    print(
        "  pesos: OE1_flex={} OE2_CO2={} OE3_costo={}".format(
            weights.get("flex"),
            weights.get("carbon"),
            weights.get("cost"),
        )
    )
    print(f"  reward_function={progress.get('reward_function')} profile={progress.get('reward_profile')}")
    print(
        "  instant_reward_sum={} instant_reward_mean={}".format(
            progress.get("instant_reward_sum"),
            progress.get("instant_reward_mean"),
        )
    )
    print(
        "  episode_return_cumulative={} episode_reward_mean_cumulative={} episode_steps={}".format(
            progress.get("episode_return_cumulative"),
            progress.get("episode_reward_mean_cumulative"),
            progress.get("episode_steps_recorded"),
        )
    )
    print(
        "  total_return_cumulative={} total_reward_mean_cumulative={} total_steps={}".format(
            progress.get("total_return_cumulative"),
            progress.get("total_reward_mean_cumulative"),
            progress.get("total_steps_recorded"),
        )
    )
    print(
        "  reward_components: flex={} carbon={} cost={} ev={} team={}".format(
            progress.get("reward_component_flex_mean"),
            progress.get("reward_component_carbon_mean"),
            progress.get("reward_component_cost_mean"),
            progress.get("reward_component_ev_mean"),
            progress.get("reward_team_reward"),
        )
    )
    print(
        "  energia_inst: cost={} co2={} net_load={} import_reward={}".format(
            progress.get("district_net_electricity_consumption_cost"),
            progress.get("district_net_electricity_consumption_emission"),
            progress.get("district_net_electricity_consumption"),
            progress.get("reward_district_import_kwh"),
        )
    )
    print(
        "  price_mean={} carbon_intensity_mean={}".format(
            progress.get("electricity_price_mean"),
            progress.get("carbon_intensity_mean"),
        )
    )
    age = file_age_seconds(progress_path)
    if age is not None:
        print(f"  live_progress: hace {age} s")


def print_artifacts(output_root: Path, limit: int = 12) -> None:
    print("")
    print("Artefactos recientes")
    if not output_root.exists():
        print("OutputRoot no existe todavia.")
        return
    names = {
        "results.json",
        "training_summary.json",
        "timeseries.csv",
        "trace.csv",
        "checkpoint_manifest.json",
        "figures_manifest.json",
        "episode_summary.csv",
        "live_progress.json",
    }
    files = [
        path
        for path in output_root.rglob("*")
        if path.is_file() and (path.name in names or path.suffix in {".pt", ".pth", ".ckpt"})
    ]
    files = sorted(files, key=lambda item: item.stat().st_mtime, reverse=True)[:limit]
    if not files:
        print("Sin artefactos todavia.")
        return
    for path in files:
        rel = path.relative_to(output_root)
        print(f"  {rel} | {path.stat().st_size / 1024.0:.1f} KB | {datetime.fromtimestamp(path.stat().st_mtime):%H:%M:%S}")


def print_logs(status: Mapping[str, object], log_tail: int) -> None:
    jobs = list(status.get("jobs", []))
    if not jobs:
        return
    print("")
    print("Logs recientes")
    for job in jobs[-4:]:
        log = Path(str(job.get("log") or ""))
        if not log.exists():
            continue
        print(f"--- {log.name} ---")
        lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines[-log_tail:]:
            if line.strip():
                print(f"  {line[:220]}")


def render_once(output_root: Path, log_tail: int) -> None:
    root = project_root()
    status_path = output_root / "official_full_status.json"
    status = read_json(status_path)
    print("=" * 72)
    print(f"CITYLEARN v3 MADRL - MONITOR COLAB A100 | {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 72)
    print(f"OutputRoot: {output_root}")

    if not status:
        print(f"No existe estado: {status_path}")
        return

    print_status(status, root)
    print_progress(status, root)
    print_gpu()
    print_artifacts(output_root)
    print_logs(status, log_tail)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--interval-seconds", default=30, type=int)
    parser.add_argument("--log-tail", default=16, type=int)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = project_root()
    output_root = resolve_output_root(root, args.output_root)
    if output_root is None:
        print("No se encontro OutputRoot. Ejecuta el launcher o pasa --output-root.")
        return 1

    while True:
        render_once(output_root, args.log_tail)
        if args.once:
            return 0
        time.sleep(max(5, int(args.interval_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
