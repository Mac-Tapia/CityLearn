"""Regenerate CityLearn v3 MADRL figures and tables from existing run data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from citylearn_v3_training_common import (  # noqa: E402
    _checkpoint_files,
    _episode_summaries,
    _write_training_figures_and_tables,
    ensure_artifact_layout,
)


def _read_csv(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        return []

    with path.open("r", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


def regenerate(run_dir: Path) -> Dict[str, object]:
    dirs = ensure_artifact_layout(run_dir)
    data_dir = dirs["data"]
    results = _read_json(data_dir / "results.json") or _read_json(run_dir / "results.json")
    report = results.get("citylearn_v3_report", results)
    report = report if isinstance(report, dict) else {}
    timeseries_rows = _read_csv(data_dir / "timeseries.csv") or _read_csv(run_dir / "timeseries.csv")
    trace_rows = _read_csv(data_dir / "trace.csv") or _read_csv(run_dir / "trace.csv")
    episode_summaries = _episode_summaries(timeseries_rows)
    checkpoints = _checkpoint_files(run_dir, dirs["checkpoints"])
    return _write_training_figures_and_tables(
        dirs=dirs,
        report=report,
        timeseries_rows=timeseries_rows,
        trace_rows=trace_rows,
        episode_summaries=episode_summaries,
        checkpoints=checkpoints,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="+", help="MADRL run directory, e.g. outputs/.../happo/E3_seed_0")
    args = parser.parse_args()

    for run_dir_text in args.run_dir:
        run_dir = Path(run_dir_text)
        manifest = regenerate(run_dir)
        print(json.dumps({
            "run_dir": str(run_dir),
            "figure_count": manifest.get("figure_count"),
            "table_count": manifest.get("table_count"),
            "figures_manifest": str(run_dir / "figures" / "figures_manifest.json"),
        }, indent=2))


if __name__ == "__main__":
    main()
