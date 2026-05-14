"""Generate per-building detail reports for saved CityLearn v3 MADRL runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from citylearn_v3_training_common import (
    CityLearnV3BackendAdapter,
    _as_float,
    _building_behavior_summary_rows,
    _building_schema_rows,
    _building_trace_sample_rows,
    _citylearn_kpi_frame_rows,
    _write_markdown_table,
    ensure_project_paths,
    write_csv,
    write_json,
)


class _Candidate:
    def __init__(self, adapter: CityLearnV3BackendAdapter):
        self.adapter = adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", help="Single run directory containing results.json and trace.csv.")
    source.add_argument("--output-root", help="Root containing many run directories, e.g. official_full_cuda_v2.")
    parser.add_argument("--report-dir", help="Output directory for a single --run-dir report.")
    parser.add_argument(
        "--replay-actions",
        action="store_true",
        help="Replay saved trace actions to regenerate physical per-building KPIs/import/export.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional replay/debug limit in environment steps. Full KPIs require all steps.",
    )
    parser.add_argument(
        "--replay-episode",
        default=None,
        help="Optional episode index to replay, or 'last' for the last episode in trace.csv.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=120,
        help="Rows to keep in building_trace_sample.csv.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=2500,
        help="Replay progress interval in environment steps.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _run_dirs_from_root(root: Path) -> List[Path]:
    output: List[Path] = []

    for results_path in sorted(root.rglob("results.json")):
        run_dir = results_path.parent

        if (run_dir / "trace.csv").is_file():
            output.append(run_dir)

    return output


def _metadata(results: Mapping[str, object], run_dir: Path) -> Dict[str, object]:
    algorithm = str(results.get("algorithm") or run_dir.parent.name).upper()
    scenario = str(results.get("scenario") or run_dir.name.split("_seed_", 1)[0])
    seed = int(results.get("seed") or 0)
    episode_time_steps = int(results.get("episode_time_steps") or 8760)
    hyperparameters = results.get("hyperparameters", {})
    hyperparameters = hyperparameters if isinstance(hyperparameters, Mapping) else {}
    action_bins = int(hyperparameters.get("action_bins") or 3)
    return {
        "algorithm": algorithm,
        "scenario": scenario,
        "seed": seed,
        "episode_time_steps": episode_time_steps,
        "action_bins": action_bins,
    }


def _make_adapter(meta: Mapping[str, object], live_progress_path: Optional[Path] = None) -> CityLearnV3BackendAdapter:
    return CityLearnV3BackendAdapter(
        scenario=str(meta["scenario"]),
        seed=int(meta["seed"]),
        episode_time_steps=int(meta["episode_time_steps"]),
        action_bins=int(meta["action_bins"]),
        algorithm=str(meta["algorithm"]),
        live_progress_path=str(live_progress_path) if live_progress_path is not None else None,
    )


def _action_from_trace(row: Mapping[str, object], dim: int) -> np.ndarray:
    values = []

    for index in range(dim):
        value = _as_float(row.get(f"action_{index}"))
        values.append(0.0 if value is None else value)

    return np.asarray(values, dtype=np.float32)


def _trace_step_groups(trace_path: Path) -> Iterable[List[Dict[str, object]]]:
    with trace_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        current_step = None
        group: List[Dict[str, object]] = []

        for row in reader:
            step = row.get("global_step")

            if current_step is None:
                current_step = step

            if step != current_step:
                yield group
                group = []
                current_step = step

            group.append(dict(row))

        if group:
            yield group


def _last_episode(trace_path: Path) -> Optional[int]:
    last = None

    with trace_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            value = _as_float(row.get("episode"))

            if value is not None:
                last = int(value)

    return last


def _resolve_replay_episode(trace_path: Path, replay_episode: Optional[str]) -> Optional[int]:
    if replay_episode is None or replay_episode == "":
        return None

    if str(replay_episode).strip().lower() == "last":
        return _last_episode(trace_path)

    return int(replay_episode)


def _replay_trace(
    *,
    adapter: CityLearnV3BackendAdapter,
    trace_path: Path,
    max_steps: Optional[int],
    progress_interval: int,
    replay_episode: Optional[str],
) -> None:
    current_episode = None
    replayed_steps = 0
    target_episode = _resolve_replay_episode(trace_path, replay_episode)

    for group in _trace_step_groups(trace_path):
        if not group:
            continue

        episode = int(_as_float(group[0].get("episode")) or 0)

        if target_episode is not None and episode != target_episode:
            if current_episode is not None and episode > target_episode:
                break
            continue

        if current_episode != episode:
            adapter.reset()
            current_episode = episode

        row_by_agent = {str(row.get("agent")): row for row in group}
        actions = [
            _action_from_trace(row_by_agent.get(agent, {}), adapter.action_dims[agent])
            for agent in adapter.agents
        ]
        adapter.step_continuous(actions)
        replayed_steps += 1

        if progress_interval > 0 and replayed_steps % progress_interval == 0:
            print(f"replayed_steps={replayed_steps} episode={episode}")

        if max_steps is not None and replayed_steps >= max_steps:
            break


def _write_report_tables(report_dir: Path, tables: Mapping[str, Sequence[Mapping[str, object]]]) -> Dict[str, object]:
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_tables = {}

    for name, rows in tables.items():
        csv_path = report_dir / f"{name}.csv"
        md_path = report_dir / f"{name}.md"
        write_csv(csv_path, rows)
        _write_markdown_table(md_path, rows)
        manifest_tables[name] = {
            "rows": len(rows),
            "csv": str(csv_path),
            "markdown": str(md_path),
        }

    return manifest_tables


def generate_report(
    *,
    run_dir: Path,
    report_dir: Optional[Path],
    replay_actions: bool,
    max_steps: Optional[int],
    replay_episode: Optional[str],
    sample_rows: int,
    progress_interval: int,
) -> Dict[str, object]:
    results_path = run_dir / "results.json"
    trace_path = run_dir / "trace.csv"

    if not results_path.is_file():
        raise FileNotFoundError(f"Missing results.json: {results_path}")

    if not trace_path.is_file():
        raise FileNotFoundError(f"Missing trace.csv: {trace_path}")

    results = _read_json(results_path)
    meta = _metadata(results, run_dir)
    adapter = _make_adapter(meta)
    candidate = _Candidate(adapter)
    schema_rows = _building_schema_rows(candidate, adapter=adapter)

    if replay_actions:
        adapter.clear_records()
        _replay_trace(
            adapter=adapter,
            trace_path=trace_path,
            max_steps=max_steps,
            progress_interval=progress_interval,
            replay_episode=replay_episode,
        )
        trace_rows = list(adapter.trace_records)
        kpi_rows = _citylearn_kpi_frame_rows(candidate)
    else:
        trace_rows = _read_csv_rows(trace_path)
        kpi_rows = []

    building_summary_rows = _building_behavior_summary_rows(
        trace_rows=trace_rows,
        kpi_rows=kpi_rows,
        schema_rows=schema_rows,
    )
    sample_rows_payload = _building_trace_sample_rows(trace_rows, schema_rows, max_rows=max(1, sample_rows))
    output_dir = report_dir or (run_dir / "building_detail_report")
    tables = {
        "building_behavior_summary": building_summary_rows,
        "building_kpis": [
            row for row in kpi_rows
            if str(row.get("level", "")).lower() == "building"
        ],
        "building_observation_action_schema": schema_rows,
        "building_trace_sample": sample_rows_payload,
    }
    table_manifest = _write_report_tables(output_dir, tables)
    manifest = {
        "run_dir": str(run_dir),
        "report_dir": str(output_dir),
        "source_trace": str(trace_path),
        "replay_actions": replay_actions,
        "max_steps": max_steps,
        "replay_episode": replay_episode,
        "metadata": meta,
        "building_count": len(building_summary_rows),
        "tables": table_manifest,
    }
    write_json(output_dir / "report_manifest.json", manifest)
    adapter.close()
    return manifest


def main() -> int:
    args = parse_args()
    ensure_project_paths()

    if args.run_dir:
        manifest = generate_report(
            run_dir=Path(args.run_dir),
            report_dir=Path(args.report_dir) if args.report_dir else None,
            replay_actions=args.replay_actions,
            max_steps=args.max_steps,
            replay_episode=args.replay_episode,
            sample_rows=args.sample_rows,
            progress_interval=args.progress_interval,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    root = Path(args.output_root)
    manifests = []

    for run_dir in _run_dirs_from_root(root):
        manifests.append(generate_report(
            run_dir=run_dir,
            report_dir=None,
            replay_actions=args.replay_actions,
            max_steps=args.max_steps,
            replay_episode=args.replay_episode,
            sample_rows=args.sample_rows,
            progress_interval=args.progress_interval,
        ))

    summary = {
        "output_root": str(root),
        "run_count": len(manifests),
        "reports": [
            {
                "run_dir": manifest["run_dir"],
                "report_dir": manifest["report_dir"],
                "building_count": manifest["building_count"],
            }
            for manifest in manifests
        ],
    }
    write_json(root / "building_detail_reports_manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
