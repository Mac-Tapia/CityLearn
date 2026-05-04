"""Build a master comparison between CityLearn v2 agents and v3 MADRL runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_V2_ROOT = Path("outputs/citylearn_v2_original_benchmark")
DEFAULT_V3_ROOT = Path("outputs/citylearn_v3_madrl_official_full_cuda_v2")
DEFAULT_OUTPUT = Path("outputs/comparison_citylearn_v2_vs_v3_madrl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", default=str(DEFAULT_V2_ROOT))
    parser.add_argument("--v3-root", default=str(DEFAULT_V3_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--scenario", default="E3")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--weights",
        default="OE1=0.34,OE2=0.33,OE3=0.33",
        help="Axis weights used for global ranking.",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _candidate_run_dirs(root: Path, scenario: str, seed: int) -> List[Path]:
    suffix = f"{scenario}_seed_{seed}"
    if not root.exists():
        return []

    return [
        path
        for path in root.rglob(suffix)
        if path.is_dir()
    ]


def _load_objective_table(run_dir: Path, *, family: str) -> pd.DataFrame:
    table_path = run_dir / "figures" / "tables" / "objective_kpis.csv"
    table = _read_csv(table_path)

    if table.empty:
        return table

    method = run_dir.parent.name
    table["family"] = family
    table["method"] = method.upper() if family == "citylearn_v3_madrl" else method
    table["run_dir"] = str(run_dir)
    table["table_path"] = str(table_path)
    return table


def load_all(v2_root: Path, v3_root: Path, *, scenario: str, seed: int) -> pd.DataFrame:
    tables: List[pd.DataFrame] = []

    for run_dir in _candidate_run_dirs(v2_root, scenario, seed):
        table = _load_objective_table(run_dir, family="citylearn_v2_original")
        if not table.empty:
            tables.append(table)

    for run_dir in _candidate_run_dirs(v3_root, scenario, seed):
        table = _load_objective_table(run_dir, family="citylearn_v3_madrl")
        if not table.empty:
            tables.append(table)

    if not tables:
        return pd.DataFrame()

    df = pd.concat(tables, ignore_index=True)

    for column in ["value", "baseline", "delta_vs_baseline"]:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "improved_vs_baseline" in df:
        df["improved_vs_baseline"] = df["improved_vs_baseline"].map(_to_bool)

    if "lower_is_better" in df:
        df["lower_is_better"] = df["lower_is_better"].map(_to_bool)

    return df


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan

    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_weights(spec: str) -> Dict[str, float]:
    output: Dict[str, float] = {}

    for item in spec.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        output[key.strip()] = float(value)

    total = sum(output.values()) or 1.0
    return {key: value / total for key, value in output.items()}


def _normalized_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []

    for kpi, group in df.groupby("kpi", dropna=False):
        values = pd.to_numeric(group["value"], errors="coerce")
        valid = values.notna()

        if valid.sum() == 0:
            temp = group.copy()
            temp["normalized_score"] = np.nan
            rows.append(temp)
            continue

        low = float(values[valid].min())
        high = float(values[valid].max())
        span = high - low
        lower_is_better = bool(group["lower_is_better"].dropna().iloc[0]) if group["lower_is_better"].notna().any() else True
        temp = group.copy()

        if span == 0.0:
            temp["normalized_score"] = 1.0
        elif lower_is_better:
            temp["normalized_score"] = (high - values) / span
        else:
            temp["normalized_score"] = (values - low) / span

        rows.append(temp)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_rankings(df: pd.DataFrame, weights: Mapping[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = _normalized_scores(df)

    if scored.empty:
        return pd.DataFrame(), pd.DataFrame()

    axis_rank = (
        scored.groupby(["family", "method", "axis"], dropna=False)
        .agg(
            normalized_score=("normalized_score", "mean"),
            available_kpis=("value", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            improved_kpis=("improved_vs_baseline", lambda s: int((s == True).sum())),
            total_kpis=("kpi", "count"),
        )
        .reset_index()
    )
    axis_rank["axis_weight"] = axis_rank["axis"].map(weights).fillna(0.0)
    axis_rank["weighted_score"] = axis_rank["normalized_score"] * axis_rank["axis_weight"]
    axis_rank["axis_rank"] = axis_rank.groupby("axis")["normalized_score"].rank(ascending=False, method="dense")

    global_rank = (
        axis_rank.groupby(["family", "method"], dropna=False)
        .agg(
            global_score=("weighted_score", "sum"),
            mean_axis_score=("normalized_score", "mean"),
            improved_kpis=("improved_kpis", "sum"),
            total_kpis=("total_kpis", "sum"),
            available_kpis=("available_kpis", "sum"),
        )
        .reset_index()
        .sort_values(["global_score", "improved_kpis"], ascending=[False, False])
    )
    global_rank["global_rank"] = range(1, len(global_rank) + 1)
    return axis_rank, global_rank


def _write_markdown_table(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_markdown(index=False), encoding="utf-8")


def _plot_axis(df: pd.DataFrame, output_dir: Path, axis: str) -> Optional[Path]:
    subset = df[df["axis"] == axis].copy()

    if subset.empty:
        return None

    summary = (
        subset.groupby(["family", "method"], dropna=False)
        .agg(score=("normalized_score", "mean"))
        .reset_index()
        .sort_values("score", ascending=True)
    )

    labels = [f"{row.method}\n{row.family.replace('citylearn_', '')}" for row in summary.itertuples()]
    colors = ["#2f7d6d" if family == "citylearn_v3_madrl" else "#6f7fa8" for family in summary["family"]]
    fig, ax = plt.subplots(figsize=(11, max(4, 0.45 * len(summary))))
    ax.barh(labels, summary["score"], color=colors)
    ax.set_xlabel("normalized KPI score (higher is better)")
    ax.set_title(f"{axis} comparison: CityLearn v2 original vs CityLearn v3 MADRL")
    fig.tight_layout()
    path = output_dir / f"{axis}_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_heatmap(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    subset = df[np.isfinite(pd.to_numeric(df["delta_vs_baseline"], errors="coerce"))].copy()

    if subset.empty:
        return None

    pivot = subset.pivot_table(
        index=["family", "method"],
        columns="kpi",
        values="delta_vs_baseline",
        aggfunc="first",
    )
    if pivot.empty:
        return None

    clipped = pivot.clip(lower=pivot.quantile(0.05), upper=pivot.quantile(0.95), axis=1)
    fig, ax = plt.subplots(figsize=(14, max(5, 0.5 * len(pivot))))
    image = ax.imshow(clipped.fillna(0.0).values, aspect="auto", cmap="coolwarm")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([f"{idx[1]} ({idx[0].replace('citylearn_', '')})" for idx in pivot.index])
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    ax.set_title("Delta vs CityLearn v2 baseline by KPI")
    fig.colorbar(image, ax=ax, shrink=0.75)
    fig.tight_layout()
    path = output_dir / "baseline_gain_heatmap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _summary_json(df: pd.DataFrame, axis_rank: pd.DataFrame, global_rank: pd.DataFrame, weights: Mapping[str, float]) -> Dict[str, object]:
    best_by_axis = {}

    if not axis_rank.empty:
        for axis, group in axis_rank.sort_values("normalized_score", ascending=False).groupby("axis"):
            row = group.iloc[0]
            best_by_axis[axis] = {
                "family": row["family"],
                "method": row["method"],
                "score": float(row["normalized_score"]),
            }

    best_global = None
    if not global_rank.empty:
        row = global_rank.iloc[0]
        best_global = {
            "family": row["family"],
            "method": row["method"],
            "score": float(row["global_score"]),
        }

    return {
        "weights": dict(weights),
        "rows": int(len(df)),
        "families": sorted(df["family"].dropna().unique().tolist()) if not df.empty else [],
        "methods": sorted(df["method"].dropna().unique().tolist()) if not df.empty else [],
        "best_by_axis": best_by_axis,
        "best_global": best_global,
    }


def main() -> int:
    args = parse_args()
    v2_root = Path(args.v2_root)
    v3_root = Path(args.v3_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = parse_weights(args.weights)
    df = load_all(v2_root, v3_root, scenario=args.scenario, seed=args.seed)

    if df.empty:
        raise SystemExit(
            "No objective_kpis.csv files found. Run benchmark_citylearn_v2_agents.py "
            "and ensure MADRL v3 runs have completed artifacts."
        )

    scored = _normalized_scores(df)
    axis_rank, global_rank = build_rankings(df, weights)

    df.to_csv(output_dir / "master_kpi_comparison.csv", index=False)
    scored.to_csv(output_dir / "master_kpi_comparison_scored.csv", index=False)
    axis_rank.to_csv(output_dir / "ranking_by_axis.csv", index=False)
    global_rank.to_csv(output_dir / "ranking_global_weighted.csv", index=False)
    _write_markdown_table(output_dir / "master_kpi_comparison.md", df)
    _write_markdown_table(output_dir / "ranking_by_axis.md", axis_rank)
    _write_markdown_table(output_dir / "ranking_global_weighted.md", global_rank)

    figures = []
    for axis in sorted(df["axis"].dropna().unique()):
        path = _plot_axis(scored, output_dir, axis)
        if path is not None:
            figures.append(str(path))
    heatmap = _plot_heatmap(df, output_dir)
    if heatmap is not None:
        figures.append(str(heatmap))

    summary = _summary_json(df, axis_rank, global_rank, weights)
    summary["figures"] = figures
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(output_dir / "comparison_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
