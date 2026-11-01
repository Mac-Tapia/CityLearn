#!/usr/bin/env python
"""Launch the canonical CityLearn v3 MADRL multi-seed campaign (4 x 3 x 12).

Design (matches ``citylearn.v3.config`` / Cap. 6):
  algorithms = HAPPO, MASAC, MATD3, MAAC
  scenarios  = E1, E2, E3
  seeds      = 0..11  (N_SEEDS=12)
  episodes   = 50

Each seed invokes ``colab_a100_official_launcher.py`` with ``--seed N`` and
``--skip-completed`` so interrupted campaigns resume safely. Layout uses the
existing ``job_run_dir_for_launcher`` convention (seed 0 → ``E1``; seed k>0 →
``E1_sk``).

Usage (repo root):

  python CityLearn/scripts/launch_citylearn_v3_multiseed_campaign.py --dry-run
  python CityLearn/scripts/launch_citylearn_v3_multiseed_campaign.py \\
      --output-root outputs/madrl_v3_multiseed_YYYYMMDD --seeds 0,1,2 --dry-run
  python CityLearn/scripts/launch_citylearn_v3_multiseed_campaign.py \\
      --output-root /content/drive/.../madrl_v3_multiseed --execute
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / "CityLearn" / "scripts" / "colab_a100_official_launcher.py"
ALGORITHMS = ("HAPPO", "MASAC", "MATD3", "MAAC")
SCENARIOS = ("E1", "E2", "E3")
N_SEEDS_DEFAULT = 12


def _parse_seeds(raw: Optional[str], n_seeds: int) -> List[int]:
    if raw is None or str(raw).strip() == "":
        return list(range(int(n_seeds)))
    out: List[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _parse_csv_upper(raw: Optional[str], default: Sequence[str]) -> List[str]:
    if raw is None or str(raw).strip() == "":
        return [str(x).upper() for x in default]
    return [p.strip().upper() for p in str(raw).split(",") if p.strip()]


def build_plan(
    *,
    seeds: Sequence[int],
    algorithms: Sequence[str],
    scenarios: Sequence[str],
    episodes: int,
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    exp_id = 1
    for seed in seeds:
        for algo in algorithms:
            for scen in scenarios:
                jobs.append(
                    {
                        "id": exp_id,
                        "seed": int(seed),
                        "algorithm": algo,
                        "scenario": scen,
                        "n_episodes": int(episodes),
                    }
                )
                exp_id += 1
    return jobs


def launcher_cmd(
    *,
    output_root: Path,
    seed: int,
    episodes: int,
    schema_path: Optional[Path],
    extra_args: Sequence[str],
    dry_run: bool,
) -> List[str]:
    cmd = [
        sys.executable,
        str(LAUNCHER),
        "--scenario",
        "ALL",
        "--seed",
        str(int(seed)),
        "--episodes",
        str(int(episodes)),
        "--output-root",
        str(output_root),
        "--skip-completed",
        "--start-from-algorithm",
        "happo",
    ]
    if schema_path is not None:
        cmd.extend(["--schema-path", str(schema_path)])
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend(list(extra_args))
    return cmd


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=str(REPO / "outputs" / f"madrl_v3_multiseed_{datetime.now().strftime('%Y%m%d')}"),
        help="Campaign output root (per-seed job dirs created underneath).",
    )
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS_DEFAULT)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma list or ranges (e.g. 0,1,2 or 0-11). Default: 0..n_seeds-1.",
    )
    parser.add_argument("--algorithms", default="HAPPO,MASAC,MATD3,MAAC")
    parser.add_argument("--scenarios", default="E1,E2,E3")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--schema-path", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write campaign_manifest.json and print launcher commands; pass --dry-run to launcher.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually invoke the official launcher once per seed (GPU/Colab).",
    )
    parser.add_argument(
        "--extra-launcher-arg",
        action="append",
        default=[],
        help="Extra arg forwarded to colab_a100_official_launcher (repeatable).",
    )
    args = parser.parse_args(argv)

    if not args.dry_run and not args.execute:
        # Safe default: planning only.
        args.dry_run = True

    seeds = _parse_seeds(args.seeds, args.n_seeds)
    algorithms = _parse_csv_upper(args.algorithms, ALGORITHMS)
    scenarios = _parse_csv_upper(args.scenarios, SCENARIOS)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    schema = Path(args.schema_path).resolve() if args.schema_path else None

    plan = build_plan(
        seeds=seeds,
        algorithms=algorithms,
        scenarios=scenarios,
        episodes=args.episodes,
    )
    seed_batches = sorted({int(j["seed"]) for j in plan})

    manifest: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaign": "citylearn_v3_madrl_multiseed",
        "n_seeds": len(seed_batches),
        "seeds": seed_batches,
        "algorithms": algorithms,
        "scenarios": scenarios,
        "episodes": int(args.episodes),
        "expected_jobs": len(plan),
        "formula": "4 algorithms x 3 scenarios x N seeds",
        "output_root": str(output_root),
        "layout_note": "seed 0 -> ALGO/E*; seed k>0 -> ALGO/E*_sk (normalize_scenario_dir)",
        "mode": "execute" if args.execute and not args.dry_run else "dry_run",
        "jobs": plan,
        "seed_runs": [],
    }

    print(
        f"[multiseed] plan={len(plan)} jobs | seeds={seed_batches} | "
        f"episodes={args.episodes} | root={output_root}"
    )

    rc_all = 0
    for seed in seed_batches:
        cmd = launcher_cmd(
            output_root=output_root,
            seed=seed,
            episodes=args.episodes,
            schema_path=schema,
            extra_args=args.extra_launcher_arg,
            dry_run=bool(args.dry_run) or not args.execute,
        )
        entry: Dict[str, Any] = {
            "seed": int(seed),
            "command": cmd,
            "status": "planned",
            "returncode": None,
        }
        print(f"[multiseed] seed={seed} -> {' '.join(cmd)}")
        if args.execute and not args.dry_run:
            proc = subprocess.run(cmd, cwd=str(REPO))
            entry["returncode"] = int(proc.returncode)
            entry["status"] = "ok" if proc.returncode == 0 else "failed"
            if proc.returncode != 0:
                rc_all = proc.returncode
        else:
            entry["status"] = "dry_run"
        manifest["seed_runs"].append(entry)

    manifest_path = output_root / "campaign_manifest.json"
    write_manifest(manifest_path, manifest)
    print(f"[multiseed] wrote {manifest_path}")
    return int(rc_all)


if __name__ == "__main__":
    raise SystemExit(main())
