"""Shared CityLearn v3 training launch utilities for external MADRL backends."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from gym import spaces


SCRIPT_PATH = Path(__file__).resolve()
CITYLEARN_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]
EXTERNAL_ROOT = PROJECT_ROOT / "external"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "citylearn_v3_madrl"
DATA_DIR_NAME = "data"
CHECKPOINT_DIR_NAME = "checkpoints"
FIGURES_DIR_NAME = "figures"
TABLES_DIR_NAME = "tables"


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


def resolve_output_dir(output_dir: Optional[str], algorithm: str, scenario: str, seed: int) -> Path:
    base = Path(output_dir) if output_dir else DEFAULT_OUTPUT_ROOT / algorithm.lower()
    path = base / f"{scenario}_seed_{seed}"
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
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: _csv_safe(row.get(key)) for key in fieldnames})


def _write_csv_mirrors(paths: Sequence[Path], rows: Sequence[Mapping[str, object]]) -> None:
    for path in paths:
        write_csv(path, rows)


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
        return candidate.adapter.env

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


def citylearn_v3_training_report(candidate) -> Dict[str, object]:
    """Return standardized CityLearn v2 KPI reporting for a launcher.

    The report is intentionally shared by HAPPO, MASAC, MATD3 and MAAC so each
    algorithm writes the same objective-axis KPIs. Project metrics are the
    three thesis axes: flexibility, CO2 emissions and costs.
    """

    from citylearn.v3.objectives import evaluate_objectives, objective_manifest

    objective_env = _resolve_objective_env(candidate)

    if objective_env is None:
        objectives = {
            "manifest": objective_manifest(),
            "axes": {},
            "project_axis_metrics": {},
            "axis_kpis": {},
            "supporting_values": {},
            "all_values": {},
            "kpi_frame_rows": 0,
        }
    else:
        objectives = evaluate_objectives(objective_env)

    return {
        "project_axis_metrics": objectives["project_axis_metrics"],
        "objective_axis_kpis": objectives["axes"],
        "axis_kpis": objectives["axis_kpis"],
        "supporting_values": objectives["supporting_values"],
        "all_values": objectives["all_values"],
        "kpi_frame_rows": objectives["kpi_frame_rows"],
        "objective_manifest": objectives["manifest"],
    }


def _checkpoint_files(output_dir: Path, checkpoint_dir: Optional[Path] = None) -> List[Dict[str, object]]:
    checkpoint_extensions = {".pt", ".pkl", ".pth", ".ckpt"}
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
        })

    return summaries


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


def _write_training_figures_and_tables(
    *,
    dirs: Mapping[str, Path],
    report: Mapping[str, object],
    timeseries_rows: Sequence[Mapping[str, object]],
    episode_summaries: Sequence[Mapping[str, object]],
    checkpoints: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    figures_dir = dirs["figures"]
    tables_dir = dirs["tables"]
    figures: List[Dict[str, object]] = []
    tables: List[Dict[str, object]] = []

    table_specs = [
        ("episode_summary", list(episode_summaries)),
        ("objective_kpis", _objective_kpi_rows(report)),
        ("axis_baseline_comparison", _axis_comparison_rows(report)),
        ("core_kpis", _core_kpi_rows(report)),
        ("checkpoint_inventory", list(checkpoints)),
    ]

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
        lambda: _save_episode_plot(figures_dir / "episode_reward_summary.png", episode_summaries),
        lambda: _save_axis_comparison_plot(figures_dir / "axis_baseline_comparison.png", table_specs[2][1]),
        lambda: _save_core_kpi_plot(figures_dir / "core_kpis.png", table_specs[3][1]),
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
    timeseries_rows = list(getattr(adapter, "timeseries_records", [])) if adapter is not None else []
    trace_rows = list(getattr(adapter, "trace_records", [])) if adapter is not None else []
    episode_summaries = _episode_summaries(timeseries_rows)

    timeseries_path = data_dir / "timeseries.csv"
    trace_path = data_dir / "trace.csv"
    root_timeseries_path = output_dir / "timeseries.csv"
    root_trace_path = output_dir / "trace.csv"
    _write_csv_mirrors([timeseries_path, root_timeseries_path], timeseries_rows)
    _write_csv_mirrors([trace_path, root_trace_path], trace_rows)

    checkpoints = _checkpoint_files(output_dir, dirs["checkpoints"])
    checkpoint_manifest = {
        "algorithm": algorithm,
        "backend": backend,
        "checkpoint_dir": str(dirs["checkpoints"]),
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "hyperparameters": dict(hyperparameters or {}),
    }
    checkpoint_manifest_path = data_dir / "checkpoint_manifest.json"
    root_checkpoint_manifest_path = output_dir / "checkpoint_manifest.json"
    _write_json_mirrors([checkpoint_manifest_path, root_checkpoint_manifest_path], checkpoint_manifest)
    figures_manifest = _write_training_figures_and_tables(
        dirs=dirs,
        report=report,
        timeseries_rows=timeseries_rows,
        episode_summaries=episode_summaries,
        checkpoints=checkpoints,
    )

    results = {
        "algorithm": algorithm,
        "backend": backend,
        "scenario": args.scenario,
        "seed": args.seed,
        "episode_time_steps": args.episode_time_steps,
        "output_dir": str(output_dir),
        "artifact_layout": _artifact_layout_payload(dirs),
        "episodes_recorded": len(episode_summaries),
        "timeseries_rows": len(timeseries_rows),
        "trace_rows": len(trace_rows),
        "timeseries_csv": str(timeseries_path),
        "timeseries_csv_root": str(root_timeseries_path),
        "trace_csv": str(trace_path),
        "trace_csv_root": str(root_trace_path),
        "checkpoint_manifest": str(checkpoint_manifest_path),
        "checkpoint_manifest_root": str(root_checkpoint_manifest_path),
        "checkpoint_count": len(checkpoints),
        "episode_summaries": episode_summaries,
        "hyperparameters": dict(hyperparameters or {}),
        "figures_manifest": str(dirs["figures"] / "figures_manifest.json"),
        "figures": figures_manifest,
        "project_axis_metrics": report["project_axis_metrics"],
        "citylearn_v3_report": report,
    }

    if extra:
        results.update(dict(extra))

    results_path = data_dir / "results.json"
    root_results_path = output_dir / "results.json"
    _write_json_mirrors([results_path, root_results_path], results)
    return {
        "artifact_layout": _artifact_layout_payload(dirs),
        "results_json": str(results_path),
        "results_json_root": str(root_results_path),
        "timeseries_csv": str(timeseries_path),
        "timeseries_csv_root": str(root_timeseries_path),
        "trace_csv": str(trace_path),
        "trace_csv_root": str(root_trace_path),
        "checkpoint_manifest": str(checkpoint_manifest_path),
        "checkpoint_manifest_root": str(root_checkpoint_manifest_path),
        "figures_manifest": str(dirs["figures"] / "figures_manifest.json"),
        "figures_dir": str(dirs["figures"]),
        "tables_dir": str(dirs["tables"]),
        "figure_count": figures_manifest["figure_count"],
        "table_count": figures_manifest["table_count"],
        "checkpoint_count": len(checkpoints),
        "timeseries_rows": len(timeseries_rows),
        "trace_rows": len(trace_rows),
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
    ):
        ensure_project_paths()
        from citylearn.v3 import make_citylearn_v3_env, make_citylearn_v3_project_env

        if schema_path:
            self.env = make_citylearn_v3_env(
                schema_path=schema_path,
                scenario=scenario,
                seed=seed,
                episode_time_steps=episode_time_steps,
            )
        else:
            self.env = make_citylearn_v3_project_env(
                scenario=scenario,
                seed=seed,
                episode_time_steps=episode_time_steps,
            )

        self.scenario = scenario
        self.seed_value = seed
        self.episode_time_steps = int(episode_time_steps)
        self.agents = list(self.env.possible_agents)
        self.n_agents = len(self.agents)
        self.num_agents = self.n_agents
        self.action_bins = int(action_bins)

        self._obs_spaces = {agent: self.env.observation_space(agent) for agent in self.agents}
        self._act_spaces = {agent: self.env.action_space(agent) for agent in self.agents}
        self.observation_dims = {
            agent: int(space.shape[0])
            for agent, space in self._obs_spaces.items()
        }
        self.action_dims = {
            agent: int(space.shape[0])
            for agent, space in self._act_spaces.items()
        }
        self.max_observation_dim = max(self.observation_dims.values())
        self.max_action_dim = max(self.action_dims.values())
        self.state_dim = int(self.env.state_space.shape[0])
        self.padded_joint_observation_dim = self.n_agents * self.max_observation_dim

        self.continuous_observation_space = [
            spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.max_observation_dim,),
                dtype=np.float32,
            )
            for _ in self.agents
        ]
        self.ctde_share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
            for _ in self.agents
        ]
        self.padded_share_observation_space = [
            spaces.Box(
                low=-np.inf,
                high=np.inf,
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

        action_values = np.linspace(-1.0, 1.0, self.action_bins, dtype=np.float32)
        self.discrete_action_table = np.asarray(
            list(itertools.product(action_values, repeat=self.max_action_dim)),
            dtype=np.float32,
        )
        self.n_discrete_actions = int(self.discrete_action_table.shape[0])
        self.discrete_action_space = [
            spaces.Discrete(self.n_discrete_actions)
            for _ in self.agents
        ]

        self._last_observations: Dict[str, np.ndarray] = {}
        self.global_step = 0
        self.reset_count = 0
        self.trace_records: List[Dict[str, object]] = []
        self.timeseries_records: List[Dict[str, object]] = []

    def seed(self, seed: int) -> None:
        self.seed_value = int(seed)

    def clear_records(self) -> None:
        self.global_step = 0
        self.reset_count = 0
        self.trace_records = []
        self.timeseries_records = []

    def reset(self) -> List[np.ndarray]:
        observations, _infos = self.env.reset(seed=self.seed_value)
        self.reset_count += 1
        self._last_observations = {
            agent: np.asarray(observation, dtype=np.float32)
            for agent, observation in observations.items()
        }
        return self.padded_observations(self._last_observations)

    def close(self) -> None:
        self.env.close()

    def padded_observations(self, observations: Mapping[str, np.ndarray]) -> List[np.ndarray]:
        return [
            _pad(observations[agent], self.max_observation_dim)
            for agent in self.agents
        ]

    def ctde_state(self) -> np.ndarray:
        return np.asarray(self.env.state(), dtype=np.float32).reshape(-1)

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
        observations, rewards, terminations, truncations, infos = self.env.step(action_dict)
        self._last_observations = {
            agent: np.asarray(observations[agent], dtype=np.float32)
            for agent in self.agents
        }
        dones = {
            agent: bool(terminations.get(agent, False) or truncations.get(agent, False))
            for agent in self.agents
        }
        self._record_step(action_dict, observations, rewards, dones, infos)
        return self._last_observations, rewards, dones, infos

    def _core_env(self):
        return getattr(self.env, "env", getattr(self.env, "unwrapped", self.env))

    def _record_step(self, action_dict, observations, rewards, dones, infos) -> None:
        episode_length = max(int(self.episode_time_steps), 1)
        episode = int(self.global_step // episode_length)
        episode_step = int(self.global_step % episode_length)
        citylearn_env = self._core_env()
        time_step = int(getattr(citylearn_env, "time_step", self.global_step))
        reward_values = [_as_float(rewards.get(agent)) for agent in self.agents]
        reward_values = [value for value in reward_values if value is not None]
        timeseries_row = {
            "global_step": self.global_step,
            "episode": episode,
            "episode_step": episode_step,
            "reset_count": self.reset_count,
            "time_step": time_step,
            "scenario": self.scenario,
            "reward_sum": None if not reward_values else float(np.sum(reward_values)),
            "reward_mean": None if not reward_values else float(np.mean(reward_values)),
            "all_done": bool(all(dones.values())) if dones else False,
            "district_net_electricity_consumption": _series_value(citylearn_env, "net_electricity_consumption", time_step),
            "district_net_electricity_consumption_without_storage": _series_value(citylearn_env, "net_electricity_consumption_without_storage", time_step),
            "district_net_electricity_consumption_cost": _series_value(citylearn_env, "net_electricity_consumption_cost", time_step),
            "district_net_electricity_consumption_emission": _series_value(citylearn_env, "net_electricity_consumption_emission", time_step),
            "electricity_price_mean": _mean_current_building_signal(citylearn_env, "pricing", "electricity_pricing", time_step),
            "carbon_intensity_mean": _mean_current_building_signal(citylearn_env, "carbon_intensity", "carbon_intensity", time_step),
        }
        self.timeseries_records.append(timeseries_row)

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
                "agent": agent,
                "agent_index": self.agents.index(agent),
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

            for idx, value in enumerate(action[: self.max_action_dim]):
                row[f"action_{idx}"] = _as_float(value)

            info = dict(infos.get(agent, {})) if isinstance(infos, Mapping) else {}
            if "individual_reward" in info:
                row["individual_reward"] = _as_float(info["individual_reward"])

            self.trace_records.append(row)

        self.global_step += 1

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
        terminated = bool(any(dones.values()))
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
