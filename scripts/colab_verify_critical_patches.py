"""Verify MAAC/launcher critical patches after Colab git sync.

Invoked from madrl_citylearn_v3_tutorial.ipynb cells 1.2 and 1.2b so checks always
read synced files on disk — not stale notebook cell source in the Colab kernel.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List

BROKEN_IMPORT_RE = re.compile(
    r"from\s+citylearn_v3_training_common\s+import[^\n]*\bresolve_status_path\b"
)


def verify_critical_patches(repo: Path) -> List[str]:
    problems: List[str] = []

    maac_sac = repo / "external/MAAC/algorithms/attention_sac.py"
    train_common = repo / "CityLearn/scripts/citylearn_v3_training_common.py"
    train_maac = repo / "CityLearn/scripts/train_citylearn_v3_maac.py"
    launcher = repo / "CityLearn/scripts/colab_a100_official_launcher.py"

    for path in (maac_sac, train_common, train_maac, launcher):
        if not path.is_file():
            problems.append(f"Falta archivo: {path}")

    if problems:
        return problems

    maac_src = maac_sac.read_text(encoding="utf-8")
    if "_sync_optimizer_state" not in maac_src:
        problems.append("MAAC sin _sync_optimizer_state (fix cuda/cpu Adam)")

    common_src = train_common.read_text(encoding="utf-8")
    if "def job_counts_as_launcher_complete" not in common_src:
        problems.append("training_common sin job_counts_as_launcher_complete")
    if "def preview_job_launcher_decision" not in common_src:
        problems.append(
            "training_common sin preview_job_launcher_decision (celdas 2.1b/7.1)"
        )

    maac_train_src = train_maac.read_text(encoding="utf-8")
    if "job_counts_as_launcher_complete" not in maac_train_src:
        problems.append("train_citylearn_v3_maac sin guardia de corrida completa")

    launcher_src = launcher.read_text(encoding="utf-8")
    if "def completed_artifact_exists" not in launcher_src:
        problems.append("launcher sin completed_artifact_exists")
    if "job_counts_as_launcher_complete" not in launcher_src:
        problems.append("launcher sin job_counts_as_launcher_complete")
    if BROKEN_IMPORT_RE.search(launcher_src):
        problems.append(
            "launcher importa resolve_status_path inexistente (ImportError)"
        )

    return problems


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        required=True,
        help="Repo root (e.g. /content/MADRLCitytleranflexresdr)",
    )
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve()

    problems = verify_critical_patches(repo)
    if problems:
        print(
            "Parches críticos ausentes tras sync (re-ejecuta celda 1.2):\n  - "
            + "\n  - ".join(problems),
            file=sys.stderr,
        )
        return 1

    print(
        "[OK] parches verificados: MAAC cuda-sync + validación de corrida "
        "+ preview_job_launcher_decision + ImportError launcher"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
