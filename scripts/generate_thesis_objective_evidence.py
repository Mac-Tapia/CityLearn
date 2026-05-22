"""Generate thesis-ready objective evidence from CityLearn v3 MADRL outputs.

The script scans completed or partial training runs and writes a compact
evidence package for the project-local thesis-plan and thesis-report skills.
It does not invent results: missing or incomplete evidence is marked as
pending/partial so the thesis text can distinguish implemented workflow from
validated quantitative findings.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SCRIPT_PATH = Path(__file__).resolve()
CITYLEARN_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(CITYLEARN_ROOT) not in sys.path:
    sys.path.insert(0, str(CITYLEARN_ROOT))


ALGORITHMS = ("happo", "masac", "matd3", "maac")
ALGORITHM_NAMES = tuple(algorithm.upper() for algorithm in ALGORITHMS)
SCENARIOS = ("E1", "E2", "E3")
STATISTICAL_ALPHA = 0.05
BOOTSTRAP_ITERATIONS = 2000
TABLE_NAMES = (
    "episode_summary",
    "objective_kpis",
    "axis_baseline_comparison",
    "core_kpis",
    "training_efficiency",
    "exploration_summary",
    "agent_reward_summary",
    "checkpoint_inventory",
)
FIGURE_NAMES = (
    "reward_timeseries.png",
    "convergence_returns.png",
    "episode_reward_summary.png",
    "learning_efficiency.png",
    "citylearn_v2_district_timeseries.png",
    "exploration_action_l2.png",
    "agent_reward_contribution.png",
    "axis_baseline_comparison.png",
    "baseline_gain_by_kpi.png",
    "core_kpis.png",
    "OE1_flexibility_kpis.png",
    "OE2_co2_kpis.png",
    "OE3_cost_kpis.png",
)

OBJECTIVE_DEFINITIONS = {
    "OE1": {
        "scenario": "E1",
        "short_name": "Flexibilidad energetica",
        "specific_objective": (
            "Demostrar que la capa CityLearn v3 MADRL propuesta permite evaluar y "
            "mejorar indicadores de flexibilidad energetica mediante coordinacion "
            "de edificios, almacenamiento, PV y EV/V2G."
        ),
        "dimension": "Flexibilidad energetica",
        "expected_result": (
            "Reduccion o caracterizacion favorable de picos, rampas, factor de "
            "carga, importacion de red, uso de almacenamiento y flexibilidad EV."
        ),
    },
    "OE2": {
        "scenario": "E2",
        "short_name": "Emisiones de CO2",
        "specific_objective": (
            "Demostrar que la capa CityLearn v3 MADRL propuesta permite evaluar y "
            "reducir indicadores de emisiones de CO2 mediante operacion sensible "
            "a intensidad de carbono."
        ),
        "dimension": "Emisiones de CO2",
        "expected_result": (
            "Reduccion o caracterizacion favorable de emisiones de CO2 de control "
            "frente a baseline y de deltas diarios/totales."
        ),
    },
    "OE3": {
        "scenario": "E3",
        "short_name": "Costos energeticos",
        "specific_objective": (
            "Demostrar que la capa CityLearn v3 MADRL propuesta permite evaluar y "
            "mejorar indicadores de costos energeticos bajo tarifas dinamicas."
        ),
        "dimension": "Costos energeticos",
        "expected_result": (
            "Reduccion o caracterizacion favorable del costo electrico, respuesta "
            "a precios y componentes de costo asociados a picos/ramping."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/thesis_objective_evidence",
        help="Directory where thesis evidence package will be written.",
    )
    parser.add_argument(
        "--output-root",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Training output root to scan. Repeatable. Defaults to official local "
            "and Colab Pro 50ep roots."
        ),
    )
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def parse_output_roots(items: Sequence[str]) -> Dict[str, Path]:
    if not items:
        items = [
            "official_local_5ep=outputs/citylearn_v3_madrl_official_full_cuda_v2",
            "colab_pro_50ep=outputs/citylearn_v3_madrl_colab_pro_50ep",
        ]

    output_roots: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --output-root value: {item!r}. Use LABEL=PATH.")
        label, raw_path = item.split("=", 1)
        output_roots[label.strip()] = (PROJECT_ROOT / raw_path.strip()).resolve()

    return output_roots


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys or ["empty"]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: normalize_cell(row.get(key)) for key in fieldnames})


def normalize_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def write_markdown_table(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys or ["empty"]

    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        values = [str(normalize_cell(row.get(key))).replace("|", "\\|").replace("\n", " ") for key in fieldnames]
        lines.append("| " + " | ".join(values) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "si"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def finite_values(values: Iterable[Any]) -> List[float]:
    output: List[float] = []
    for value in values:
        numeric = as_float(value)
        if numeric is not None and math.isfinite(numeric):
            output.append(float(numeric))
    return output


def mean_value(values: Sequence[float]) -> Optional[float]:
    values = finite_values(values)
    return None if not values else float(sum(values) / len(values))


def median_value(values: Sequence[float]) -> Optional[float]:
    values = sorted(finite_values(values))
    if not values:
        return None
    midpoint = len(values) // 2
    if len(values) % 2:
        return float(values[midpoint])
    return float((values[midpoint - 1] + values[midpoint]) / 2.0)


def sample_variance(values: Sequence[float]) -> Optional[float]:
    values = finite_values(values)
    if len(values) < 2:
        return None
    avg = float(sum(values) / len(values))
    return float(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def standard_deviation(values: Sequence[float]) -> Optional[float]:
    variance = sample_variance(values)
    if variance is None:
        return None
    return float(math.sqrt(max(variance, 0.0)))


def scipy_stats_module():
    try:
        from scipy import stats  # type: ignore

        return stats
    except Exception:
        return None


def rankdata_average(values: Sequence[float]) -> List[float]:
    indexed = sorted((float(value), index) for index, value in enumerate(values))
    ranks = [0.0] * len(indexed)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][0] == indexed[cursor][0]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for _, original_index in indexed[cursor:end]:
            ranks[original_index] = average_rank
        cursor = end
    return ranks


def kruskal_h_fallback(groups: Mapping[str, Sequence[float]]) -> Optional[float]:
    clean_groups = {name: finite_values(values) for name, values in groups.items()}
    clean_groups = {name: values for name, values in clean_groups.items() if values}
    if len(clean_groups) < 2:
        return None

    pooled: List[float] = []
    slices: Dict[str, tuple[int, int]] = {}
    for name, values in clean_groups.items():
        start = len(pooled)
        pooled.extend(values)
        slices[name] = (start, len(pooled))

    total_n = len(pooled)
    if total_n < 2:
        return None

    ranks = rankdata_average(pooled)
    numerator = 0.0
    for name, (start, end) in slices.items():
        group_ranks = ranks[start:end]
        numerator += (sum(group_ranks) ** 2) / len(clean_groups[name])

    h_statistic = (12.0 / (total_n * (total_n + 1.0))) * numerator - 3.0 * (total_n + 1.0)

    tie_counts: Dict[float, int] = {}
    for value in pooled:
        tie_counts[value] = tie_counts.get(value, 0) + 1
    tie_sum = sum(count ** 3 - count for count in tie_counts.values() if count > 1)
    if tie_sum and total_n > 1:
        correction = 1.0 - tie_sum / (total_n ** 3 - total_n)
        if correction > 0:
            h_statistic /= correction

    return float(max(h_statistic, 0.0))


def mann_whitney_u_fallback(group_a: Sequence[float], group_b: Sequence[float]) -> Optional[float]:
    a_values = finite_values(group_a)
    b_values = finite_values(group_b)
    if not a_values or not b_values:
        return None
    pooled = a_values + b_values
    ranks = rankdata_average(pooled)
    rank_sum_a = sum(ranks[: len(a_values)])
    u_a = rank_sum_a - len(a_values) * (len(a_values) + 1) / 2.0
    return float(u_a)


def brown_forsythe_w_fallback(groups: Mapping[str, Sequence[float]]) -> Optional[float]:
    clean_groups = {name: finite_values(values) for name, values in groups.items()}
    clean_groups = {name: values for name, values in clean_groups.items() if len(values) >= 2}
    group_count = len(clean_groups)
    total_n = sum(len(values) for values in clean_groups.values())
    if group_count < 2 or total_n <= group_count:
        return None

    deviations: Dict[str, List[float]] = {}
    for name, values in clean_groups.items():
        center = median_value(values)
        if center is None:
            continue
        deviations[name] = [abs(value - center) for value in values]

    all_deviations = [value for values in deviations.values() for value in values]
    grand_mean = mean_value(all_deviations)
    if grand_mean is None:
        return None

    numerator = 0.0
    denominator = 0.0
    for name, values in deviations.items():
        group_mean = mean_value(values)
        if group_mean is None:
            continue
        numerator += len(values) * (group_mean - grand_mean) ** 2
        denominator += sum((value - group_mean) ** 2 for value in values)

    if denominator <= 0.0:
        return 0.0
    return float(((total_n - group_count) / (group_count - 1)) * (numerator / denominator))


def stable_seed(*parts: object) -> int:
    seed = 0
    for char in "|".join(str(part) for part in parts):
        seed = (seed * 131 + ord(char)) % (2 ** 32)
    return seed


def bootstrap_mean_difference_ci(
    group_a: Sequence[float],
    group_b: Sequence[float],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    confidence: float = 0.95,
    rng_seed: int = 0,
) -> tuple[Optional[float], Optional[float]]:
    a_values = finite_values(group_a)
    b_values = finite_values(group_b)
    if not a_values or not b_values or iterations <= 0:
        return None, None

    rng = random.Random(rng_seed)
    diffs: List[float] = []
    for _ in range(iterations):
        sample_a = [a_values[rng.randrange(len(a_values))] for _ in a_values]
        sample_b = [b_values[rng.randrange(len(b_values))] for _ in b_values]
        mean_a = mean_value(sample_a)
        mean_b = mean_value(sample_b)
        if mean_a is not None and mean_b is not None:
            diffs.append(mean_a - mean_b)

    if not diffs:
        return None, None

    diffs.sort()
    alpha = max(0.0, min(1.0, 1.0 - confidence))
    low_index = int(math.floor((alpha / 2.0) * (len(diffs) - 1)))
    high_index = int(math.ceil((1.0 - alpha / 2.0) * (len(diffs) - 1)))
    return float(diffs[low_index]), float(diffs[high_index])


def cliffs_delta(group_a: Sequence[float], group_b: Sequence[float]) -> Optional[float]:
    a_values = finite_values(group_a)
    b_values = finite_values(group_b)
    if not a_values or not b_values:
        return None
    greater = 0
    lower = 0
    for value_a in a_values:
        for value_b in b_values:
            if value_a > value_b:
                greater += 1
            elif value_a < value_b:
                lower += 1
    return float((greater - lower) / (len(a_values) * len(b_values)))


def vargha_delaney_a12(group_a: Sequence[float], group_b: Sequence[float]) -> Optional[float]:
    delta = cliffs_delta(group_a, group_b)
    if delta is None:
        return None
    return float((delta + 1.0) / 2.0)


def cohen_d(group_a: Sequence[float], group_b: Sequence[float]) -> Optional[float]:
    a_values = finite_values(group_a)
    b_values = finite_values(group_b)
    if len(a_values) < 2 or len(b_values) < 2:
        return None
    var_a = sample_variance(a_values)
    var_b = sample_variance(b_values)
    mean_a = mean_value(a_values)
    mean_b = mean_value(b_values)
    if var_a is None or var_b is None or mean_a is None or mean_b is None:
        return None
    pooled = ((len(a_values) - 1) * var_a + (len(b_values) - 1) * var_b) / (len(a_values) + len(b_values) - 2)
    if pooled <= 0.0:
        return None
    return float((mean_a - mean_b) / math.sqrt(pooled))


def hedges_g(group_a: Sequence[float], group_b: Sequence[float]) -> Optional[float]:
    d_value = cohen_d(group_a, group_b)
    if d_value is None:
        return None
    degrees_of_freedom = len(finite_values(group_a)) + len(finite_values(group_b)) - 2
    if degrees_of_freedom <= 1:
        return d_value
    correction = 1.0 - (3.0 / (4.0 * degrees_of_freedom - 1.0))
    return float(d_value * correction)


def cliffs_delta_magnitude(delta: Optional[float]) -> str:
    if delta is None:
        return "no_calculable"
    absolute = abs(delta)
    if absolute < 0.147:
        return "negligible"
    if absolute < 0.33:
        return "pequeno"
    if absolute < 0.474:
        return "mediano"
    return "grande"


def objective_manifest() -> Dict[str, Any]:
    try:
        from citylearn.v3.objectives import objective_manifest as load_manifest

        return load_manifest()
    except Exception:
        return {"axes": {}, "axis_kpis": {}}


def run_dir(output_root: Path, algorithm: str, scenario: str, seed: int) -> Path:
    return output_root / algorithm / f"{scenario}_seed_{seed}"


def table_path(run_path: Path, table_name: str) -> Path:
    return run_path / "figures" / "tables" / f"{table_name}.csv"


def scan_runs(output_roots: Mapping[str, Path], seed: int) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for output_label, output_root in output_roots.items():
        launcher_status = read_json(output_root / "official_full_status.json")
        launcher_jobs: Dict[tuple, Mapping[str, Any]] = {}
        if isinstance(launcher_status, Mapping):
            for job in launcher_status.get("jobs", []):
                if not isinstance(job, Mapping):
                    continue
                key = (str(job.get("name", "")).lower(), str(job.get("scenario", "")))
                launcher_jobs[key] = job

        for algorithm in ALGORITHMS:
            for scenario in SCENARIOS:
                path = run_dir(output_root, algorithm, scenario, seed)
                if not path.exists():
                    continue
                launcher_job = launcher_jobs.get((algorithm, scenario), {})
                launcher_exit_code = launcher_job.get("exit_code", "") if launcher_job else ""
                launcher_job_known = bool(launcher_job)
                launcher_status_file = bool(launcher_status)
                if not launcher_status_file:
                    launcher_validation_status = "no_launcher_status"
                elif not launcher_job_known:
                    launcher_validation_status = "not_listed_in_current_launcher"
                elif str(launcher_exit_code) == "0":
                    launcher_validation_status = "completed_exit_0"
                elif launcher_exit_code in ("", None) and not launcher_job.get("completed_at"):
                    launcher_validation_status = "running_or_incomplete"
                else:
                    launcher_validation_status = "failed_or_interrupted"

                figures_manifest = read_json(path / "figures" / "figures_manifest.json")
                figures = figures_manifest.get("figures", []) if isinstance(figures_manifest, Mapping) else []
                figure_names = {item.get("name") for item in figures if isinstance(item, Mapping)}
                summary = read_json(path / "data" / "training_summary.json") or read_json(path / "training_summary.json")
                checkpoint_rows = read_csv(table_path(path, "checkpoint_inventory"))
                objective_rows = read_csv(table_path(path, "objective_kpis"))
                results_json = (path / "data" / "results.json").is_file()
                timeseries_csv = (path / "data" / "timeseries.csv").is_file()
                trace_csv = (path / "data" / "trace.csv").is_file()
                checkpoint_manifest = (path / "data" / "checkpoint_manifest.json").is_file()
                table_count = sum(1 for name in TABLE_NAMES if table_path(path, name).is_file())
                has_artifacts = any([
                    results_json,
                    bool(summary),
                    timeseries_csv,
                    trace_csv,
                    checkpoint_manifest,
                    bool(figures_manifest),
                    bool(objective_rows),
                    bool(checkpoint_rows),
                    bool(figures),
                    table_count > 0,
                ])
                if not has_artifacts:
                    continue
                inventory = {
                    "output_profile": output_label,
                    "algorithm": algorithm.upper(),
                    "scenario": scenario,
                    "seed": seed,
                    "run_path": str(path),
                    "launcher_status_file": launcher_status_file,
                    "launcher_status": launcher_status.get("status", "") if isinstance(launcher_status, Mapping) else "",
                    "launcher_job_known": launcher_job_known,
                    "launcher_exit_code": launcher_exit_code,
                    "launcher_started_at": launcher_job.get("started_at", "") if launcher_job else "",
                    "launcher_completed_at": launcher_job.get("completed_at", "") if launcher_job else "",
                    "launcher_validation_status": launcher_validation_status,
                    "results_json": results_json,
                    "training_summary": bool(summary),
                    "timeseries_csv": timeseries_csv,
                    "trace_csv": trace_csv,
                    "checkpoint_manifest": checkpoint_manifest,
                    "figures_manifest": bool(figures_manifest),
                    "objective_kpis_rows": len(objective_rows),
                    "checkpoint_rows": len(checkpoint_rows),
                    "figure_count": len(figures),
                    "table_count": table_count,
                    "missing_figures": ", ".join(name for name in FIGURE_NAMES if name not in figure_names),
                    "missing_tables": ", ".join(name for name in TABLE_NAMES if not table_path(path, name).is_file()),
                }
                runs.append(inventory)
    return runs


def run_is_usable_for_results(run: Mapping[str, Any]) -> bool:
    if not as_bool(run.get("launcher_status_file")):
        return True
    return (
        as_bool(run.get("launcher_job_known")) is True
        and str(run.get("launcher_exit_code", "")) == "0"
    )


def collect_table_rows(runs: Sequence[Mapping[str, Any]], table_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        if not run_is_usable_for_results(run):
            continue
        run_path_value = run.get("run_path")
        if not run_path_value:
            continue
        for row in read_csv(table_path(Path(str(run_path_value)), table_name)):
            enriched = dict(row)
            enriched.update({
                "output_profile": run["output_profile"],
                "algorithm": run["algorithm"],
                "scenario": run["scenario"],
                "seed": run["seed"],
                "run_path": run["run_path"],
            })
            rows.append(enriched)
    return rows


def kpi_manifest_rows(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    axes = manifest.get("axes", {})
    traces = manifest.get("axis_kpis", {})
    if not isinstance(axes, Mapping):
        return rows

    for axis_code, axis in axes.items():
        if not isinstance(axis, Mapping):
            continue
        for kpi in axis.get("kpis", []):
            trace = traces.get(kpi, {}) if isinstance(traces, Mapping) else {}
            trace = trace if isinstance(trace, Mapping) else {}
            rows.append({
                "axis": axis_code,
                "axis_name": axis.get("name", axis_code),
                "scenario": OBJECTIVE_DEFINITIONS.get(axis_code, {}).get("scenario", axis.get("scenario", "")),
                "specific_objective": OBJECTIVE_DEFINITIONS.get(axis_code, {}).get("specific_objective", ""),
                "dimension": OBJECTIVE_DEFINITIONS.get(axis_code, {}).get("dimension", axis.get("name", axis_code)),
                "kpi": kpi,
                "source": trace.get("source", ""),
                "lower_is_better": trace.get("lower_is_better", ""),
                "citylearn_v2_names": ", ".join(trace.get("citylearn_v2_names", [])),
                "note": trace.get("note", ""),
            })
    return rows


def compute_objective_compliance(
    *,
    manifest: Mapping[str, Any],
    run_inventory: Sequence[Mapping[str, Any]],
    objective_rows: Sequence[Mapping[str, Any]],
    axis_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    manifest_rows = kpi_manifest_rows(manifest)
    expected_kpis_by_axis = {
        axis: {row["kpi"] for row in manifest_rows if row["axis"] == axis}
        for axis in OBJECTIVE_DEFINITIONS
    }
    output: List[Dict[str, Any]] = []

    for axis_code, definition in OBJECTIVE_DEFINITIONS.items():
        scenario = definition["scenario"]
        focus_runs = [
            run for run in run_inventory
            if run.get("scenario") == scenario and run.get("output_profile") in {"official_local_5ep", "colab_pro_50ep"}
        ]
        usable_runs = [run for run in focus_runs if run_is_usable_for_results(run)]
        focus_objectives = [
            row for row in objective_rows
            if row.get("axis") == axis_code and row.get("scenario") == scenario
        ]
        focus_axis_rows = [
            row for row in axis_rows
            if row.get("axis") == axis_code and row.get("scenario") == scenario
        ]
        algorithms_with_kpis = sorted({row.get("algorithm") for row in focus_objectives if row.get("algorithm")})
        measured_kpis = {row.get("kpi") for row in focus_objectives if row.get("kpi")}
        expected_kpis = expected_kpis_by_axis.get(axis_code, set())
        improved_records = [row for row in focus_objectives if as_bool(row.get("improved_vs_baseline")) is True]
        not_improved_records = [row for row in focus_objectives if as_bool(row.get("improved_vs_baseline")) is False]
        comparable_records = [
            row for row in focus_objectives
            if as_bool(row.get("available")) is True or row.get("baseline") not in (None, "")
        ]
        complete_artifact_runs = [
            run for run in usable_runs
            if not run.get("missing_figures") and not run.get("missing_tables") and run.get("training_summary")
        ]

        if not focus_objectives:
            data_status = "pendiente_sin_resultados"
            compliance_status = "pendiente_de_entrenamiento"
        elif len(algorithms_with_kpis) < len(ALGORITHMS):
            data_status = "evidencia_parcial"
            compliance_status = "cumplimiento_parcial_por_consolidar" if improved_records else "pendiente_de_consolidacion"
        elif len(measured_kpis) < len(expected_kpis):
            data_status = "evidencia_kpi_incompleta"
            compliance_status = "cumplimiento_parcial_por_kpis" if improved_records else "pendiente_de_kpis"
        else:
            data_status = "evidencia_completa"
            if len(improved_records) > len(not_improved_records):
                compliance_status = "cumplimiento_cuantitativo_mayoritario"
            elif improved_records:
                compliance_status = "cumplimiento_cuantitativo_parcial"
            else:
                compliance_status = "no_demostrado_cuantitativamente"

        output.append({
            "axis": axis_code,
            "scenario": scenario,
            "specific_objective": definition["specific_objective"],
            "dimension": definition["dimension"],
            "expected_result": definition["expected_result"],
            "expected_algorithms": len(ALGORITHMS),
            "runs_found": len(focus_runs),
            "usable_result_runs": len(usable_runs),
            "complete_artifact_runs": len(complete_artifact_runs),
            "algorithms_with_kpis": ", ".join(algorithms_with_kpis),
            "expected_kpi_count": len(expected_kpis),
            "measured_kpi_count": len(measured_kpis),
            "comparable_kpi_records": len(comparable_records),
            "improved_kpi_records": len(improved_records),
            "not_improved_kpi_records": len(not_improved_records),
            "axis_comparison_rows": len(focus_axis_rows),
            "data_generation_status": data_status,
            "objective_compliance_status": compliance_status,
            "demonstration_statement": build_demonstration_statement(
                axis_code=axis_code,
                definition=definition,
                data_status=data_status,
                compliance_status=compliance_status,
                algorithms_with_kpis=algorithms_with_kpis,
                improved_count=len(improved_records),
                not_improved_count=len(not_improved_records),
            ),
        })

    return output


def build_demonstration_statement(
    *,
    axis_code: str,
    definition: Mapping[str, Any],
    data_status: str,
    compliance_status: str,
    algorithms_with_kpis: Sequence[str],
    improved_count: int,
    not_improved_count: int,
) -> str:
    if data_status == "pendiente_sin_resultados":
        return (
            f"{axis_code} ({definition['short_name']}) queda metodologicamente definido, "
            "pero la demostracion cuantitativa esta pendiente porque aun no existen "
            "objective_kpis.csv consolidados para el escenario objetivo."
        )

    return (
        f"{axis_code} ({definition['short_name']}) cuenta con evidencia en "
        f"{', '.join(algorithms_with_kpis) or 'sin algoritmos completos'}; "
        f"estado de datos: {data_status}; estado de cumplimiento: {compliance_status}; "
        f"KPIs mejorados: {improved_count}; KPIs no mejorados: {not_improved_count}. "
        "La interpretacion final debe usar estos registros y no inferir resultados no observados."
    )


def algorithm_kpi_score_rows(objective_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Build baseline-aware KPI scores for cross-algorithm statistics.

    The score is a signed relative gain against the baseline. Positive values
    are better for both lower-is-better and higher-is-better KPIs.
    """

    rows: List[Dict[str, Any]] = []
    for row in objective_rows:
        algorithm = str(row.get("algorithm", "")).upper()
        axis = str(row.get("axis", ""))
        scenario = str(row.get("scenario", ""))
        kpi = str(row.get("kpi", ""))
        value = as_float(row.get("value"))
        baseline = as_float(row.get("baseline"))
        lower_is_better = as_bool(row.get("lower_is_better"))
        available = as_bool(row.get("available"))

        if (
            algorithm not in ALGORITHM_NAMES
            or axis not in OBJECTIVE_DEFINITIONS
            or value is None
            or baseline is None
            or lower_is_better is None
            or available is not True
        ):
            continue

        raw_delta = value - baseline
        signed_absolute_gain = -raw_delta if lower_is_better else raw_delta
        denominator = abs(baseline)
        baseline_zero_normalization = False
        if denominator < 1.0e-12:
            denominator = max(abs(value), 1.0)
            baseline_zero_normalization = True

        signed_relative_gain = signed_absolute_gain / denominator
        rows.append({
            "scope": axis,
            "axis": axis,
            "scenario": scenario,
            "dimension": OBJECTIVE_DEFINITIONS[axis]["dimension"],
            "output_profile": row.get("output_profile", ""),
            "algorithm": algorithm,
            "kpi": kpi,
            "value": value,
            "baseline": baseline,
            "lower_is_better": lower_is_better,
            "raw_delta_value_minus_baseline": raw_delta,
            "signed_absolute_gain": signed_absolute_gain,
            "signed_relative_gain": signed_relative_gain,
            "improved_vs_baseline": as_bool(row.get("improved_vs_baseline")),
            "baseline_zero_normalization": baseline_zero_normalization,
            "score_rule": "positive_signed_relative_gain_is_better",
            "source": row.get("source", ""),
            "run_path": row.get("run_path", ""),
        })

    return rows


def statistical_scopes(score_rows: Sequence[Mapping[str, Any]]) -> List[str]:
    axes = [
        axis for axis in OBJECTIVE_DEFINITIONS
        if any(row.get("axis") == axis for row in score_rows)
    ]
    return axes + (["ALL"] if score_rows else [])


def score_groups(score_rows: Sequence[Mapping[str, Any]], scope: str) -> Dict[str, List[float]]:
    groups: Dict[str, List[float]] = {algorithm: [] for algorithm in ALGORITHM_NAMES}
    for row in score_rows:
        if scope != "ALL" and row.get("axis") != scope:
            continue
        algorithm = str(row.get("algorithm", "")).upper()
        if algorithm not in groups:
            continue
        value = as_float(row.get("signed_relative_gain"))
        if value is not None and math.isfinite(value):
            groups[algorithm].append(value)
    return groups


def group_summary_payload(groups: Mapping[str, Sequence[float]], reducer) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for algorithm in ALGORITHM_NAMES:
        values = finite_values(groups.get(algorithm, []))
        reduced = reducer(values) if values else None
        output[algorithm] = reduced
    return output


def best_algorithm_by_median(groups: Mapping[str, Sequence[float]]) -> tuple[str, Optional[float]]:
    medians = group_summary_payload(groups, median_value)
    valid = {algorithm: value for algorithm, value in medians.items() if value is not None}
    if not valid:
        return "", None
    algorithm = max(valid, key=lambda name: valid[name])
    return algorithm, valid[algorithm]


def statistical_omnibus_rows(score_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    stats = scipy_stats_module()

    for scope in statistical_scopes(score_rows):
        groups = score_groups(score_rows, scope)
        clean_groups = {name: values for name, values in groups.items() if values}
        scenario = "ALL" if scope == "ALL" else OBJECTIVE_DEFINITIONS[scope]["scenario"]
        dimension = "Todos los ejes" if scope == "ALL" else OBJECTIVE_DEFINITIONS[scope]["dimension"]
        kruskal_h = kruskal_h_fallback(clean_groups)
        kruskal_p = None
        kruskal_status = "fallback_statistic_without_p_value"
        brown_w = brown_forsythe_w_fallback(clean_groups)
        brown_p = None
        brown_status = "fallback_statistic_without_p_value"
        shapiro_results: Dict[str, Dict[str, Any]] = {
            algo: {"statistic": None, "p_value": None, "status": "not_computed"}
            for algo in ALGORITHM_NAMES
        }

        if len(clean_groups) >= 2 and stats is not None:
            try:
                result = stats.kruskal(*[clean_groups[name] for name in sorted(clean_groups)])
                kruskal_h = float(result.statistic)
                kruskal_p = float(result.pvalue)
                kruskal_status = "ok"
            except ValueError as exc:
                if "All numbers are identical" in str(exc):
                    kruskal_h = 0.0
                    kruskal_p = 1.0
                    kruskal_status = "all_values_identical"
                else:
                    kruskal_status = f"not_calculable: {exc}"
            except Exception as exc:
                kruskal_status = f"not_calculable: {exc}"

            try:
                result = stats.levene(*[clean_groups[name] for name in sorted(clean_groups)], center="median")
                brown_w = float(result.statistic)
                brown_p = float(result.pvalue)
                if not math.isfinite(brown_w) or not math.isfinite(brown_p):
                    brown_w = 0.0
                    brown_p = 1.0
                    brown_status = "all_deviations_identical"
                else:
                    brown_status = "ok"
            except Exception as exc:
                brown_status = f"not_calculable: {exc}"

            for algo_name in ALGORITHM_NAMES:
                sw_values = finite_values(clean_groups.get(algo_name, []))
                if len(sw_values) >= 3:
                    try:
                        sw_result = stats.shapiro(sw_values)
                        shapiro_results[algo_name] = {
                            "statistic": float(sw_result.statistic),
                            "p_value": float(sw_result.pvalue),
                            "status": "ok",
                        }
                    except Exception as exc:
                        shapiro_results[algo_name]["status"] = f"not_calculable: {exc}"
                elif sw_values:
                    shapiro_results[algo_name]["status"] = "insufficient_data_n<3"
                else:
                    shapiro_results[algo_name]["status"] = "no_data"

        normality_violated = any(
            r["p_value"] is not None and r["p_value"] < STATISTICAL_ALPHA
            for r in shapiro_results.values()
        )

        best_algorithm, best_median = best_algorithm_by_median(clean_groups)
        rows.append({
            "scope": scope,
            "scenario": scenario,
            "dimension": dimension,
            "method": (
                "Shapiro-Wilk normalidad por grupo; Kruskal-Wallis omnibus; "
                "Levene/Brown-Forsythe varianza homogenea"
            ),
            "alpha": STATISTICAL_ALPHA,
            "n_algorithms": len(clean_groups),
            "algorithms": ", ".join(name for name in ALGORITHM_NAMES if name in clean_groups),
            "n_total": sum(len(values) for values in clean_groups.values()),
            "group_n_json": json.dumps({name: len(clean_groups.get(name, [])) for name in ALGORITHM_NAMES}, sort_keys=True),
            "group_mean_signed_relative_gain_json": json.dumps(group_summary_payload(clean_groups, mean_value), sort_keys=True),
            "group_median_signed_relative_gain_json": json.dumps(group_summary_payload(clean_groups, median_value), sort_keys=True),
            "best_algorithm_by_median_gain": best_algorithm,
            "best_median_signed_relative_gain": best_median,
            "shapiro_wilk_statistic_HAPPO": shapiro_results["HAPPO"]["statistic"],
            "shapiro_wilk_p_value_HAPPO": shapiro_results["HAPPO"]["p_value"],
            "shapiro_wilk_normality_rejected_HAPPO": (
                None if shapiro_results["HAPPO"]["p_value"] is None
                else shapiro_results["HAPPO"]["p_value"] < STATISTICAL_ALPHA
            ),
            "shapiro_wilk_status_HAPPO": shapiro_results["HAPPO"]["status"],
            "shapiro_wilk_statistic_MASAC": shapiro_results["MASAC"]["statistic"],
            "shapiro_wilk_p_value_MASAC": shapiro_results["MASAC"]["p_value"],
            "shapiro_wilk_normality_rejected_MASAC": (
                None if shapiro_results["MASAC"]["p_value"] is None
                else shapiro_results["MASAC"]["p_value"] < STATISTICAL_ALPHA
            ),
            "shapiro_wilk_status_MASAC": shapiro_results["MASAC"]["status"],
            "shapiro_wilk_statistic_MATD3": shapiro_results["MATD3"]["statistic"],
            "shapiro_wilk_p_value_MATD3": shapiro_results["MATD3"]["p_value"],
            "shapiro_wilk_normality_rejected_MATD3": (
                None if shapiro_results["MATD3"]["p_value"] is None
                else shapiro_results["MATD3"]["p_value"] < STATISTICAL_ALPHA
            ),
            "shapiro_wilk_status_MATD3": shapiro_results["MATD3"]["status"],
            "shapiro_wilk_statistic_MAAC": shapiro_results["MAAC"]["statistic"],
            "shapiro_wilk_p_value_MAAC": shapiro_results["MAAC"]["p_value"],
            "shapiro_wilk_normality_rejected_MAAC": (
                None if shapiro_results["MAAC"]["p_value"] is None
                else shapiro_results["MAAC"]["p_value"] < STATISTICAL_ALPHA
            ),
            "shapiro_wilk_status_MAAC": shapiro_results["MAAC"]["status"],
            "normality_assumption_violated_any_group": normality_violated,
            "shapiro_wilk_note": (
                "SW p<0.05 rechaza normalidad; si algun grupo la rechaza, "
                "los tests no parametricos (Kruskal-Wallis, Wilcoxon, Mann-Whitney) quedan justificados."
            ),
            "kruskal_h_statistic": kruskal_h,
            "kruskal_p_value": kruskal_p,
            "kruskal_status": kruskal_status,
            "kruskal_significant_alpha_0_05": None if kruskal_p is None else kruskal_p < STATISTICAL_ALPHA,
            "brown_forsythe_w_statistic": brown_w,
            "brown_forsythe_p_value": brown_p,
            "brown_forsythe_status": brown_status,
            "variance_heterogeneity_alpha_0_05": None if brown_p is None else brown_p < STATISTICAL_ALPHA,
            "method_note": (
                "Samples are KPI-level signed relative gains against baseline; "
                "positive values favor the algorithm. Interpret p-values as exploratory "
                "when only one seed is available."
            ),
        })

    return rows


def pairwise_statistical_rows(
    score_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    stats = scipy_stats_module()

    for scope in statistical_scopes(score_rows):
        groups = score_groups(score_rows, scope)
        scenario = "ALL" if scope == "ALL" else OBJECTIVE_DEFINITIONS[scope]["scenario"]
        dimension = "Todos los ejes" if scope == "ALL" else OBJECTIVE_DEFINITIONS[scope]["dimension"]

        for algorithm_a, algorithm_b in itertools.combinations(ALGORITHM_NAMES, 2):
            values_a = finite_values(groups.get(algorithm_a, []))
            values_b = finite_values(groups.get(algorithm_b, []))
            mean_a = mean_value(values_a)
            mean_b = mean_value(values_b)
            median_a = median_value(values_a)
            median_b = median_value(values_b)
            u_statistic = mann_whitney_u_fallback(values_a, values_b)
            u_p_value = None
            u_status = "fallback_statistic_without_p_value"
            wilcoxon_statistic = None
            wilcoxon_p_value = None
            wilcoxon_status = "not_computed"

            if values_a and values_b and stats is not None:
                try:
                    result = stats.mannwhitneyu(values_a, values_b, alternative="two-sided", method="auto")
                    u_statistic = float(result.statistic)
                    u_p_value = float(result.pvalue)
                    u_status = "ok"
                except Exception as exc:
                    u_status = f"not_calculable: {exc}"

                n_paired = min(len(values_a), len(values_b))
                paired_a = values_a[:n_paired]
                paired_b = values_b[:n_paired]
                if n_paired >= 1 and any(a != b for a, b in zip(paired_a, paired_b)):
                    try:
                        wc_result = stats.wilcoxon(paired_a, paired_b, alternative="two-sided")
                        wilcoxon_statistic = float(wc_result.statistic)
                        wilcoxon_p_value = float(wc_result.pvalue)
                        wilcoxon_status = f"ok_n_paired={n_paired}"
                    except Exception as exc:
                        wilcoxon_status = f"not_calculable: {exc}"
                else:
                    wilcoxon_status = "all_differences_zero_or_empty"
            elif not values_a or not values_b:
                u_status = "insufficient_data"
                wilcoxon_status = "insufficient_data"

            ci_low, ci_high = bootstrap_mean_difference_ci(
                values_a,
                values_b,
                iterations=bootstrap_iterations,
                rng_seed=stable_seed(scope, algorithm_a, algorithm_b, "bootstrap"),
            )
            delta = cliffs_delta(values_a, values_b)
            a12 = vargha_delaney_a12(values_a, values_b)
            d_value = cohen_d(values_a, values_b)
            g_value = hedges_g(values_a, values_b)
            mean_difference = None if mean_a is None or mean_b is None else mean_a - mean_b
            median_difference = None if median_a is None or median_b is None else median_a - median_b
            if median_difference is None:
                better_by_median = ""
            elif median_difference > 0:
                better_by_median = algorithm_a
            elif median_difference < 0:
                better_by_median = algorithm_b
            else:
                better_by_median = "empate"

            rows.append({
                "scope": scope,
                "scenario": scenario,
                "dimension": dimension,
                "algorithm_a": algorithm_a,
                "algorithm_b": algorithm_b,
                "n_a": len(values_a),
                "n_b": len(values_b),
                "mean_a": mean_a,
                "mean_b": mean_b,
                "median_a": median_a,
                "median_b": median_b,
                "mean_difference_a_minus_b": mean_difference,
                "median_difference_a_minus_b": median_difference,
                "mann_whitney_u": u_statistic,
                "mann_whitney_p_value": u_p_value,
                "mann_whitney_status": u_status,
                "mann_whitney_significant_alpha_0_05": None if u_p_value is None else u_p_value < STATISTICAL_ALPHA,
                "wilcoxon_statistic": wilcoxon_statistic,
                "wilcoxon_p_value": wilcoxon_p_value,
                "wilcoxon_status": wilcoxon_status,
                "wilcoxon_significant_alpha_0_05": None if wilcoxon_p_value is None else wilcoxon_p_value < STATISTICAL_ALPHA,
                "cliffs_delta": delta,
                "cliffs_delta_magnitude": cliffs_delta_magnitude(delta),
                "vargha_delaney_a12": a12,
                "cohen_d": d_value,
                "hedges_g": g_value,
                "bootstrap_mean_diff_ci_low": ci_low,
                "bootstrap_mean_diff_ci_high": ci_high,
                "bootstrap_iterations": bootstrap_iterations,
                "better_by_median": better_by_median,
                "effect_direction_note": (
                    "Positive differences, Cliff's delta, Cohen d and Hedges g favor algorithm_a; "
                    "A12 > 0.5 favors algorithm_a."
                ),
            })

    return rows


def enrich_objective_compliance_with_statistics(
    objective_compliance: Sequence[Mapping[str, Any]],
    omnibus_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    omnibus_by_scope = {str(row.get("scope")): row for row in omnibus_rows}
    enriched: List[Dict[str, Any]] = []

    for row in objective_compliance:
        axis = str(row.get("axis", ""))
        stats_row = omnibus_by_scope.get(axis, {})
        p_value = as_float(stats_row.get("kruskal_p_value"))
        variance_p_value = as_float(stats_row.get("brown_forsythe_p_value"))
        significant = as_bool(stats_row.get("kruskal_significant_alpha_0_05"))
        best_algorithm = str(stats_row.get("best_algorithm_by_median_gain", ""))

        if not stats_row:
            interpretation = "Analisis estadistico no calculable por falta de scores KPI-normalizados."
        elif significant is True:
            interpretation = (
                f"Kruskal-Wallis detecta diferencias globales entre algoritmos MADRL en {axis}; "
                f"el mejor por mediana de ganancia relativa KPI-normalizada es {best_algorithm}."
            )
        else:
            interpretation = (
                f"Kruskal-Wallis no detecta diferencias globales significativas en {axis} con alpha=0.05; "
                f"el ranking KPI observado se conserva como evidencia descriptiva, con {best_algorithm or 'sin algoritmo dominante'} "
                "por mediana de ganancia relativa."
            )

        output = dict(row)
        output.update({
            "statistical_method": (
                "Shapiro-Wilk normalidad por grupo; Kruskal-Wallis omnibus; "
                "Mann-Whitney U + Wilcoxon signed-rank por pares; "
                "Cliff's delta; Vargha-Delaney A12; Cohen d; Hedges g; "
                "Levene/Brown-Forsythe; bootstrap CI 95%"
            ),
            "statistical_score_unit": "signed_relative_gain_vs_baseline_positive_is_better",
            "statistical_best_algorithm_by_median_gain": best_algorithm,
            "kruskal_p_value": p_value,
            "kruskal_significant_alpha_0_05": significant,
            "brown_forsythe_p_value": variance_p_value,
            "variance_heterogeneity_alpha_0_05": as_bool(stats_row.get("variance_heterogeneity_alpha_0_05")),
            "statistical_interpretation": interpretation,
            "statistical_evidence_files": (
                "scores_kpi_algoritmo_madrl.csv; analisis_estadistico_madrl.csv; "
                "comparaciones_por_pares_madrl.csv; hipotesis_estadisticas_madrl.csv"
            ),
            "statistical_limitation": (
                "Contrastes exploratorios sobre KPIs normalizados de una corrida por algoritmo; "
                "no sustituyen replicacion por multiples semillas."
            ),
        })
        enriched.append(output)

    return enriched


def statistical_hypothesis_rows(
    objective_compliance: Sequence[Mapping[str, Any]],
    omnibus_rows: Sequence[Mapping[str, Any]],
    pairwise_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    omnibus_by_scope = {str(row.get("scope")): row for row in omnibus_rows}
    rows: List[Dict[str, Any]] = []

    for axis_code, definition in OBJECTIVE_DEFINITIONS.items():
        compliance = next((row for row in objective_compliance if row.get("axis") == axis_code), {})
        omnibus = omnibus_by_scope.get(axis_code, {})
        best_algorithm = str(omnibus.get("best_algorithm_by_median_gain", ""))
        best_pairs = [
            row for row in pairwise_rows
            if row.get("scope") == axis_code
            and (row.get("algorithm_a") == best_algorithm or row.get("algorithm_b") == best_algorithm)
        ]
        significant_pairs_mwu = [
            row for row in best_pairs
            if as_bool(row.get("mann_whitney_significant_alpha_0_05")) is True
        ]
        significant_pairs_wilcoxon = [
            row for row in best_pairs
            if as_bool(row.get("wilcoxon_significant_alpha_0_05")) is True
        ]
        rows.append({
            "axis": axis_code,
            "scenario": definition["scenario"],
            "dimension": definition["dimension"],
            "hypothesis_or_expected_result": definition["expected_result"],
            "observed_compliance_status": compliance.get("objective_compliance_status", ""),
            "observed_demonstration_statement": compliance.get("demonstration_statement", ""),
            "statistical_best_algorithm_by_median_gain": best_algorithm,
            "normality_assumption_violated_any_group": omnibus.get("normality_assumption_violated_any_group"),
            "shapiro_wilk_p_value_HAPPO": omnibus.get("shapiro_wilk_p_value_HAPPO"),
            "shapiro_wilk_p_value_MASAC": omnibus.get("shapiro_wilk_p_value_MASAC"),
            "shapiro_wilk_p_value_MATD3": omnibus.get("shapiro_wilk_p_value_MATD3"),
            "shapiro_wilk_p_value_MAAC": omnibus.get("shapiro_wilk_p_value_MAAC"),
            "kruskal_h_statistic": omnibus.get("kruskal_h_statistic"),
            "kruskal_p_value": omnibus.get("kruskal_p_value"),
            "kruskal_significant_alpha_0_05": omnibus.get("kruskal_significant_alpha_0_05"),
            "brown_forsythe_w_statistic": omnibus.get("brown_forsythe_w_statistic"),
            "brown_forsythe_p_value": omnibus.get("brown_forsythe_p_value"),
            "variance_heterogeneity_alpha_0_05": omnibus.get("variance_heterogeneity_alpha_0_05"),
            "pairwise_comparisons_with_best": len(best_pairs),
            "significant_pairwise_mwu_with_best": len(significant_pairs_mwu),
            "significant_pairwise_wilcoxon_with_best": len(significant_pairs_wilcoxon),
            "effect_sizes_reported": "Cliff's delta, Vargha-Delaney A12, Cohen d, Hedges g",
            "bootstrap_ci_reported": "95% CI for mean signed-relative-gain difference",
            "decision_rule": (
                "Use KPI compliance as primary thesis evidence; use non-parametric tests and "
                "effect sizes as statistical support. Do not infer unobserved results."
            ),
            "evidence_files": (
                "objetivos_especificos_cumplimiento.csv; matriz_resultados_madrl.csv; "
                "analisis_estadistico_madrl.csv; comparaciones_por_pares_madrl.csv"
            ),
            "limitations": (
                "Single seed and KPI-level samples; formal confirmation requires more seeds "
                "or independent experimental repetitions."
            ),
        })

    overall = omnibus_by_scope.get("ALL", {})
    rows.append({
        "axis": "OG",
        "scenario": "ALL",
        "dimension": "Gestion coordinada integral",
        "hypothesis_or_expected_result": "Determinar el mejor MADRL en los tres ejes integrados.",
        "observed_compliance_status": "ranking_integrado_por_kpis",
        "observed_demonstration_statement": "La evidencia descriptiva integrada se calcula sobre los KPIs comparables de OE1, OE2 y OE3.",
        "statistical_best_algorithm_by_median_gain": overall.get("best_algorithm_by_median_gain", ""),
        "kruskal_h_statistic": overall.get("kruskal_h_statistic"),
        "kruskal_p_value": overall.get("kruskal_p_value"),
        "kruskal_significant_alpha_0_05": overall.get("kruskal_significant_alpha_0_05"),
        "brown_forsythe_w_statistic": overall.get("brown_forsythe_w_statistic"),
        "brown_forsythe_p_value": overall.get("brown_forsythe_p_value"),
        "variance_heterogeneity_alpha_0_05": overall.get("variance_heterogeneity_alpha_0_05"),
        "pairwise_comparisons_with_best": len([
            row for row in pairwise_rows
            if row.get("scope") == "ALL"
            and (
                row.get("algorithm_a") == overall.get("best_algorithm_by_median_gain")
                or row.get("algorithm_b") == overall.get("best_algorithm_by_median_gain")
            )
        ]),
        "significant_pairwise_mwu_with_best": len([
            row for row in pairwise_rows
            if row.get("scope") == "ALL"
            and as_bool(row.get("mann_whitney_significant_alpha_0_05")) is True
            and (
                row.get("algorithm_a") == overall.get("best_algorithm_by_median_gain")
                or row.get("algorithm_b") == overall.get("best_algorithm_by_median_gain")
            )
        ]),
        "significant_pairwise_wilcoxon_with_best": len([
            row for row in pairwise_rows
            if row.get("scope") == "ALL"
            and as_bool(row.get("wilcoxon_significant_alpha_0_05")) is True
            and (
                row.get("algorithm_a") == overall.get("best_algorithm_by_median_gain")
                or row.get("algorithm_b") == overall.get("best_algorithm_by_median_gain")
            )
        ]),
        "normality_assumption_violated_any_group": overall.get("normality_assumption_violated_any_group"),
        "effect_sizes_reported": "Cliff's delta, Vargha-Delaney A12, Cohen d, Hedges g",
        "bootstrap_ci_reported": "95% CI for mean signed-relative-gain difference",
        "decision_rule": "Use integrated KPI ranking as primary O.G. evidence and omnibus/pairwise tests as support.",
        "evidence_files": "analisis_estadistico_madrl.csv; comparaciones_por_pares_madrl.csv",
        "limitations": "Single seed and KPI-level samples; formal confirmation requires more seeds.",
    })
    return rows


def methodological_matrix_rows(manifest_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in manifest_rows:
        rows.append({
            "variable": "Desempeno del despacho optimo bajo restricciones electricas y operacion segura",
            "dimension": row["dimension"],
            "indicator": row["kpi"],
            "objective_axis": row["axis"],
            "scenario": row["scenario"],
            "measurement_source": row["source"],
            "instrument": "CityLearn v2 evaluate_v2 + CityLearn v3 proposed MADRL artifact tables",
            "technique": (
                "Simulacion computacional, entrenamiento MADRL, comparacion contra baseline, "
                "analisis de KPIs, Shapiro-Wilk normalidad, Kruskal-Wallis, "
                "Mann-Whitney U, Wilcoxon signed-rank, Cliff's delta, "
                "Vargha-Delaney A12, Cohen d, Hedges g, Levene/Brown-Forsythe y bootstrap CI"
            ),
            "scale": "Numerica continua o ratio segun KPI",
            "interpretation_rule": "Usar lower_is_better y delta_vs_baseline cuando exista baseline disponible",
        })
    return rows


def consistency_matrix_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for axis_code, definition in OBJECTIVE_DEFINITIONS.items():
        rows.append({
            "specific_problem": (
                f"Como evaluar el eje {definition['dimension']} del despacho inteligente "
                "mediante CityLearn v3 propuesto y MADRL colaborativo?"
            ),
            "specific_objective": definition["specific_objective"],
            "hypothesis_or_expected_result": definition["expected_result"],
            "variable": "Desempeno del despacho optimo bajo restricciones electricas y operacion segura",
            "dimension": definition["dimension"],
            "scenario": definition["scenario"],
            "method": "Simulacion computacional no experimental con CityLearn v2 y capa CityLearn v3 propuesta",
            "technique": (
                "Entrenamiento MADRL CTDE, extraccion de KPIs, comparacion contra baseline, "
                "Shapiro-Wilk normalidad, Kruskal-Wallis, Mann-Whitney U, "
                "Wilcoxon signed-rank y contrastes no parametricos con tamanos de efecto"
            ),
            "instrument": (
                "Scripts train_citylearn_v3_*.py, objective_kpis.csv, axis_baseline_comparison.csv, "
                "figures_manifest.json, analisis_estadistico_madrl.csv, comparaciones_por_pares_madrl.csv"
            ),
        })
    return rows


def backend_rows() -> List[Dict[str, Any]]:
    return [
        {
            "backend": "HAPPO",
            "source": "external/HARL",
            "training_script": "CityLearn/scripts/train_citylearn_v3_happo.py",
            "ctde_role": "Critico centralizado con observacion compartida; actor local por edificio",
            "action_space": "Continuo",
            "evidence_outputs": "training_summary.json, timeseries.csv, trace.csv, objective_kpis.csv, checkpoints",
        },
        {
            "backend": "MASAC",
            "source": "external/MARL/src",
            "training_script": "CityLearn/scripts/train_citylearn_v3_masac.py",
            "ctde_role": "Estado global estilo SMAC mediante get_state(); politica local discretizada",
            "action_space": "Accion continua discretizada",
            "evidence_outputs": "training_summary.json, timeseries.csv, trace.csv, objective_kpis.csv, checkpoints",
        },
        {
            "backend": "MATD3",
            "source": "external/off-policy",
            "training_script": "CityLearn/scripts/train_citylearn_v3_matd3.py",
            "ctde_role": "Criticos centralizados con observaciones/acciones conjuntas; actores locales continuos",
            "action_space": "Continuo",
            "evidence_outputs": "training_summary.json, timeseries.csv, trace.csv, objective_kpis.csv, checkpoints",
        },
        {
            "backend": "MAAC",
            "source": "external/MAAC",
            "training_script": "CityLearn/scripts/train_citylearn_v3_maac.py",
            "ctde_role": "Critico de atencion multiagente; politicas locales discretizadas",
            "action_space": "Accion continua discretizada",
            "evidence_outputs": "training_summary.json, timeseries.csv, trace.csv, objective_kpis.csv, checkpoints",
        },
    ]


def architecture_rows() -> List[Dict[str, Any]]:
    return [
        {
            "component": "CityLearn v2 base",
            "path": "CityLearn/",
            "purpose": "Simulador, dataset, dinamica fisica, edificios, DERs, EVs y KPIs evaluate_v2",
            "thesis_role": "Entorno base existente",
        },
        {
            "component": "CityLearn v3 propuesto",
            "path": "CityLearn/citylearn/v3/",
            "purpose": "Capa experimental para Dec-POMDP, CTDE, objetivos OE1/OE2/OE3 y wrappers MADRL",
            "thesis_role": "Extension experimental propuesta por la investigacion",
        },
        {
            "component": "Adaptador de entrenamiento",
            "path": "CityLearn/scripts/citylearn_v3_training_common.py",
            "purpose": "Estandariza registros, artefactos, KPIs, figuras y tablas",
            "thesis_role": "Instrumento de recoleccion y procesamiento de datos",
        },
        {
            "component": "Scripts MADRL",
            "path": "CityLearn/scripts/train_citylearn_v3_*.py",
            "purpose": "Ejecutan HAPPO, MASAC, MATD3 y MAAC sobre CityLearn v3 propuesto",
            "thesis_role": "Intervencion computacional experimental",
        },
        {
            "component": "Paquete de evidencia de tesis",
            "path": "CityLearn/scripts/generate_thesis_objective_evidence.py",
            "purpose": "Consolida evidencia por objetivo especifico y genera feeds para skills de tesis",
            "thesis_role": "Puente entre resultados experimentales y redaccion academica",
        },
    ]


def seai_rows() -> List[Dict[str, Any]]:
    return [
        {
            "aspect": "Pertinencia territorial",
            "evidence": "El marco metodologico conserva SEAI Iquitos como caso de aplicabilidad; los resultados simulados no sustituyen mediciones reales del sistema.",
            "status": "aplicable_metodologicamente",
        },
        {
            "aspect": "Flexibilidad energetica",
            "evidence": "OE1 usa KPIs de picos, ramping, factor de carga, almacenamiento, PV y EV/V2G como proxies de flexibilidad.",
            "status": "medible_por_simulacion",
        },
        {
            "aspect": "Emisiones de CO2",
            "evidence": "OE2 usa intensidad de carbono y emisiones control/baseline/delta para evaluar operacion carbon-aware.",
            "status": "medible_por_simulacion",
        },
        {
            "aspect": "Costos energeticos",
            "evidence": "OE3 usa costos, precio dinamico, picos y desviacion de senal tarifaria para evaluar eficiencia economica.",
            "status": "medible_por_simulacion",
        },
        {
            "aspect": "Limitacion",
            "evidence": "La transferencia a SEAI Iquitos requiere calibracion con datos operativos reales y restricciones electricas especificas.",
            "status": "pendiente_de_validacion_operativa_real",
        },
    ]


def citylearn_v3_rows() -> List[Dict[str, Any]]:
    return [
        {
            "section": "Definicion",
            "content": "CityLearn v3 propuesto es una capa experimental de tesis sobre CityLearn v2; no se declara como version oficial del paquete CityLearn.",
            "evidence": "CityLearn/citylearn/v3/; ESTRATEGIA_3PILARES_MADRL.md; README.md",
            "thesis_use": "Delimitar el aporte tecnico de la investigacion.",
        },
        {
            "section": "Modelo formal",
            "content": "El problema se formula como Dec-POMDP cooperativo con observaciones locales por edificio y estado global para CTDE.",
            "evidence": "CityLearn/citylearn/v3/environment.py; CityLearn/citylearn/v3/objectives.py",
            "thesis_use": "Sustentar modelo matematico, agentes, estados, observaciones y acciones.",
        },
        {
            "section": "Objetivos especificos",
            "content": "OE1 flexibilidad energetica, OE2 emisiones de CO2 y OE3 costos energeticos se separan por escenarios E1/E2/E3.",
            "evidence": "objective_manifest_snapshot.json; objetivos_especificos_cumplimiento.csv",
            "thesis_use": "Alinear el experimento con los tres objetivos especificos.",
        },
        {
            "section": "Instrumentacion",
            "content": (
                "La capa de entrenamiento exporta resumenes, trazas, series de tiempo, KPIs, "
                "comparacion contra baseline, figuras, checkpoints y matrices estadisticas "
                "MADRL multialgoritmo."
            ),
            "evidence": "CityLearn/scripts/citylearn_v3_training_common.py; run_artifact_inventory.csv; analisis_estadistico_madrl.csv",
            "thesis_use": "Definir tecnicas e instrumentos de recoleccion de datos.",
        },
        {
            "section": "No invencion de resultados",
            "content": "Los estados pendiente_sin_resultados y evidencia_parcial impiden afirmar resultados cuantitativos aun no generados.",
            "evidence": "resumen_evidencia_tesis.md; thesis_skill_feed.json",
            "thesis_use": "Control de calidad metodologico para plan e informe.",
        },
    ]


def marllib_rows() -> List[Dict[str, Any]]:
    return [
        {
            "section": "Rol en la tesis",
            "content": "MARLlib se conserva como marco de referencia para comparar terminologia, patrones CTDE y organizacion de algoritmos multiagente.",
            "evidence": "tools/skills/madrl-citylearn-thesis-plan/references/module-a-plan-literature.md",
            "status": "referencia_metodologica",
        },
        {
            "section": "Backends implementados",
            "content": "El proyecto ejecuta HAPPO, MASAC, MATD3 y MAAC mediante scripts propios conectados a la capa CityLearn v3 propuesta.",
            "evidence": "Backends_MADRL.csv; CityLearn/scripts/train_citylearn_v3_*.py",
            "status": "implementado_en_scripts",
        },
        {
            "section": "Limite",
            "content": "No se afirma que MARLlib sea el motor de ejecucion de todos los entrenamientos si el script usa HARL, MARL/src, off-policy o MAAC externos.",
            "evidence": "Backends_MADRL.csv",
            "status": "delimitacion_tecnica",
        },
    ]


def co2_cost_rows(manifest_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for axis_code in ("OE2", "OE3"):
        definition = OBJECTIVE_DEFINITIONS[axis_code]
        kpis = sorted({str(row.get("kpi", "")) for row in manifest_rows if row.get("axis") == axis_code and row.get("kpi")})
        rows.append({
            "axis": axis_code,
            "scenario": definition["scenario"],
            "dimension": definition["dimension"],
            "expected_result": definition["expected_result"],
            "kpi_count": len(kpis),
            "kpis": ", ".join(kpis),
            "evidence_source": "matriz_kpis_tesis.csv; objective_kpis.csv por corrida cuando el entrenamiento exista",
            "status": "definido_metodologicamente",
        })
    return rows


def dataset_code_rows(output_roots: Mapping[str, Path]) -> List[Dict[str, Any]]:
    rows = [
        {
            "source_type": "Codigo base",
            "name": "CityLearn v2 local",
            "path": "CityLearn/",
            "role": "Simulador base y evaluacion evaluate_v2.",
            "verification_status": "presente_en_repositorio",
        },
        {
            "source_type": "Codigo propuesto",
            "name": "CityLearn v3 propuesto",
            "path": "CityLearn/citylearn/v3/",
            "role": "Capa experimental MADRL, objetivos, wrappers y exportacion de evidencia.",
            "verification_status": "presente_en_repositorio",
        },
        {
            "source_type": "Scripts de entrenamiento",
            "name": "HAPPO, MASAC, MATD3, MAAC",
            "path": "CityLearn/scripts/train_citylearn_v3_*.py",
            "role": "Generacion de resultados por algoritmo, escenario y semilla.",
            "verification_status": "presente_en_repositorio",
        },
    ]
    for label, path in output_roots.items():
        rows.append({
            "source_type": "Salida experimental",
            "name": label,
            "path": str(path),
            "role": "Fuente de tablas, figuras, checkpoints y matrices de evaluacion.",
            "verification_status": "detectado" if path.exists() else "pendiente_de_generacion",
        })
    return rows


def executive_summary_rows(
    objective_compliance: Sequence[Mapping[str, Any]],
    run_inventory: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows = [
        {
            "section": "Proposito",
            "content": "Consolidar evidencia tecnica para alimentar los skills locales de plan de tesis e informe de tesis.",
            "evidence": "thesis_skill_feed.json",
            "status": "generado",
        },
        {
            "section": "Inventario experimental",
            "content": f"Corridas detectadas: {len(run_inventory)}.",
            "evidence": "run_artifact_inventory.csv",
            "status": "generado",
        },
    ]
    for row in objective_compliance:
        rows.append({
            "section": row.get("axis", ""),
            "content": row.get("demonstration_statement", ""),
            "evidence": "objetivos_especificos_cumplimiento.csv; matriz_resultados_madrl.csv",
            "status": row.get("objective_compliance_status", ""),
        })
    return rows


def write_table_bundle(output_dir: Path, name: str, rows: Sequence[Mapping[str, Any]]) -> None:
    write_csv(output_dir / f"{name}.csv", rows)
    write_markdown_table(output_dir / f"{name}.md", rows)


def write_summary_markdown(
    path: Path,
    *,
    objective_compliance: Sequence[Mapping[str, Any]],
    run_inventory: Sequence[Mapping[str, Any]],
    statistical_omnibus: Sequence[Mapping[str, Any]],
    table_names: Sequence[str],
) -> None:
    lines = [
        "# Paquete de evidencia para plan e informe de tesis",
        "",
        f"Generado: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Estado por objetivo especifico",
        "",
    ]
    for row in objective_compliance:
        lines.extend([
            f"### {row['axis']} - {row['dimension']}",
            "",
            f"- Escenario: `{row['scenario']}`",
            f"- Estado de datos: `{row['data_generation_status']}`",
            f"- Estado de cumplimiento: `{row['objective_compliance_status']}`",
            f"- Algoritmos con KPIs: {row['algorithms_with_kpis'] or 'pendiente'}",
            f"- KPIs medidos/esperados: {row['measured_kpi_count']}/{row['expected_kpi_count']}",
            f"- KPIs mejorados/no mejorados: {row['improved_kpi_records']}/{row['not_improved_kpi_records']}",
            f"- Mejor algoritmo por mediana estadistica: {row.get('statistical_best_algorithm_by_median_gain') or 'no calculable'}",
            f"- Kruskal-Wallis p-value: {row.get('kruskal_p_value') if row.get('kruskal_p_value') not in (None, '') else 'no calculable'}",
            "",
            row["demonstration_statement"],
            "",
            row.get("statistical_interpretation", ""),
            "",
        ])

    if statistical_omnibus:
        lines.extend([
            "## Analisis estadistico MADRL",
            "",
            "Los contrastes se calculan sobre `signed_relative_gain` por KPI comparable; valores positivos favorecen al algoritmo frente al baseline.",
            "",
        ])
        for row in statistical_omnibus:
            lines.append(
                f"- `{row.get('scope')}`: Kruskal p={row.get('kruskal_p_value')}; "
                f"Brown-Forsythe p={row.get('brown_forsythe_p_value')}; "
                f"mejor mediana={row.get('best_algorithm_by_median_gain') or 'no calculable'}."
            )
        lines.extend([
            "",
            "Estos p-values son apoyo exploratorio cuando solo existe una semilla por algoritmo; la evidencia primaria sigue siendo la matriz KPI/baseline.",
            "",
        ])

    lines.extend([
        "## Productos para skills locales",
        "",
        "Archivos CSV y MD generados con nombres alineados a los workbooks de `madrl-citylearn-thesis-plan` y `madrl-citylearn-thesis-integrated`:",
        "",
    ])
    for table_name in table_names:
        lines.append(f"- `{table_name}.csv` / `{table_name}.md`")

    lines.extend([
        "",
        "## Inventario de corridas",
        "",
        f"- Corridas detectadas: {len(run_inventory)}",
        "- Si existe `official_full_status.json`, los artefactos se cuentan como evidencia cuantitativa solo cuando el job del launcher figura con `exit_code=0`.",
        "- Este paquete marca resultados pendientes cuando faltan `objective_kpis.csv`, tablas, figuras o entrenamientos completos.",
        "- No debe usarse para afirmar resultados cuantitativos que no figuren en las matrices generadas.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_roots = parse_output_roots(args.output_root)
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = objective_manifest()
    run_inventory = scan_runs(output_roots, args.seed)
    objective_rows = collect_table_rows(run_inventory, "objective_kpis")
    axis_rows = collect_table_rows(run_inventory, "axis_baseline_comparison")
    manifest_rows = kpi_manifest_rows(manifest)
    objective_compliance = compute_objective_compliance(
        manifest=manifest,
        run_inventory=run_inventory,
        objective_rows=objective_rows,
        axis_rows=axis_rows,
    )
    statistical_score_rows = algorithm_kpi_score_rows(objective_rows)
    statistical_omnibus = statistical_omnibus_rows(statistical_score_rows)
    statistical_pairwise = pairwise_statistical_rows(statistical_score_rows)
    objective_compliance = enrich_objective_compliance_with_statistics(
        objective_compliance,
        statistical_omnibus,
    )
    statistical_hypotheses = statistical_hypothesis_rows(
        objective_compliance,
        statistical_omnibus,
        statistical_pairwise,
    )
    methodology_rows = methodological_matrix_rows(manifest_rows)
    consistency_rows = consistency_matrix_rows()
    backends = backend_rows()
    architecture = architecture_rows()
    seai = seai_rows()
    citylearn_v3 = citylearn_v3_rows()
    marllib = marllib_rows()
    co2_cost = co2_cost_rows(manifest_rows)
    dataset_code = dataset_code_rows(output_roots)
    executive_summary = executive_summary_rows(objective_compliance, run_inventory)

    tables = {
        "Resumen_ejecutivo": executive_summary,
        "run_artifact_inventory": run_inventory,
        "objetivos_especificos_cumplimiento": objective_compliance,
        "KPIs_y_metricas": manifest_rows,
        "Matriz_KPIs": manifest_rows,
        "matriz_kpis_tesis": manifest_rows,
        "matriz_resultados_madrl": objective_rows,
        "matriz_baseline_por_eje": axis_rows,
        "scores_kpi_algoritmo_madrl": statistical_score_rows,
        "analisis_estadistico_madrl": statistical_omnibus,
        "comparaciones_por_pares_madrl": statistical_pairwise,
        "hipotesis_estadisticas_madrl": statistical_hypotheses,
        "matriz_operacionalizacion_variables": methodology_rows,
        "Marco_metodologico_MADRL": methodology_rows,
        "CityLearn_v3_Propuesto": citylearn_v3,
        "matriz_consistencia_objetivos": consistency_rows,
        "Backends_MADRL": backends,
        "MARLlib_Integracion": marllib,
        "CityLearn_CO2_Costos": co2_cost,
        "Datasets_y_codigo": dataset_code,
        "Arquitectura_Propuesta": architecture,
        "Aplicabilidad_SEAI_Iquitos": seai,
    }

    for name, rows in tables.items():
        write_table_bundle(output_dir, name, rows)

    skill_feed = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "output_roots": {label: str(path) for label, path in output_roots.items()},
        "objective_compliance": objective_compliance,
        "matrices": {name: str(output_dir / f"{name}.csv") for name in tables},
        "required_skill_products_present": sorted(tables),
        "thesis_skill_targets": [
            "madrl-citylearn-thesis-plan",
            "madrl-citylearn-thesis-integrated",
        ],
        "non_invention_rule": (
            "Use generated quantitative values only when present in CSV/JSON. "
            "Mark missing values as resultados por validar or pendiente de entrenamiento."
        ),
    }
    write_json(output_dir / "thesis_skill_feed.json", skill_feed)
    write_json(output_dir / "objective_manifest_snapshot.json", manifest)
    write_summary_markdown(
        output_dir / "resumen_evidencia_tesis.md",
        objective_compliance=objective_compliance,
        run_inventory=run_inventory,
        statistical_omnibus=statistical_omnibus,
        table_names=sorted(tables),
    )

    print(json.dumps({
        "output_dir": str(output_dir),
        "run_count": len(run_inventory),
        "objective_rows": len(objective_rows),
        "statistical_score_rows": len(statistical_score_rows),
        "statistical_omnibus_rows": len(statistical_omnibus),
        "statistical_pairwise_rows": len(statistical_pairwise),
        "objective_compliance": objective_compliance,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
