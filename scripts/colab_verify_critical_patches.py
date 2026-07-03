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
    if "map_location" not in maac_src or "weights_only=False" not in maac_src:
        problems.append("MAAC init_from_save sin map_location/weights_only=False (PyTorch 2.6+)")

    common_src = train_common.read_text(encoding="utf-8")
    if "def job_counts_as_launcher_complete" not in common_src:
        problems.append("training_common sin job_counts_as_launcher_complete")
    if "def preview_job_launcher_decision" not in common_src:
        problems.append(
            "training_common sin preview_job_launcher_decision (celdas 2.1b/7.1)"
        )
    if "unwrap_citylearn_core_env" not in common_src:
        problems.append("training_common sin unwrap_citylearn_core_env en adapter init")

    maac_train_src = train_maac.read_text(encoding="utf-8")
    if "job_counts_as_launcher_complete" not in maac_train_src:
        problems.append("train_citylearn_v3_maac sin guardia de corrida completa")
    if "_save_maac_checkpoint" not in maac_train_src:
        problems.append("train_citylearn_v3_maac sin guard de checkpoint atomico (Drive FUSE)")

    launcher_src = launcher.read_text(encoding="utf-8")
    if "def completed_artifact_exists" not in launcher_src:
        problems.append("launcher sin completed_artifact_exists")
    if "job_counts_as_launcher_complete" not in launcher_src:
        problems.append("launcher sin job_counts_as_launcher_complete")
    if BROKEN_IMPORT_RE.search(launcher_src):
        problems.append(
            "launcher importa resolve_status_path inexistente (ImportError)"
        )

    madrl_kpis = repo / "CityLearn/citylearn/madrl_kpis.py"
    dec_pomdp = repo / "CityLearn/citylearn/dec_pomdp.py"
    harl_wrappers = repo / "external/HARL/harl/envs/env_wrappers.py"
    regenerate = repo / "CityLearn/scripts/regenerate_happo_kpis.py"
    for path in (madrl_kpis, dec_pomdp, harl_wrappers, regenerate):
        if not path.is_file():
            problems.append(f"Falta archivo: {path}")

    if madrl_kpis.is_file():
        kpis_src = madrl_kpis.read_text(encoding="utf-8")
        if "def unwrap_citylearn_core_env" not in kpis_src:
            problems.append("madrl_kpis sin unwrap_citylearn_core_env (fix VecEnvWrapper)")

    if dec_pomdp.is_file():
        dec_src = dec_pomdp.read_text(encoding="utf-8")
        if "self.env.unwrapped" in dec_src:
            problems.append("dec_pomdp aún usa self.env.unwrapped (NameError HARL)")
        if "unwrap_citylearn_core_env" not in dec_src:
            problems.append("dec_pomdp sin unwrap_citylearn_core_env")

    if harl_wrappers.is_file():
        harl_src = harl_wrappers.read_text(encoding="utf-8")
        if "class VecEnvWrapper" not in harl_src:
            problems.append("HARL env_wrappers sin stub VecEnvWrapper")

    prepare = repo / "CityLearn/scripts/prepare_happo_colab_resume.py"
    if not prepare.is_file():
        problems.append("Falta prepare_happo_colab_resume.py (resume HAPPO 49→50)")
    if regenerate.is_file() and "preflight" not in regenerate.read_text(encoding="utf-8"):
        problems.append("regenerate_happo_kpis sin preflight de checkpoints")

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
        "+ preview_job_launcher_decision + ImportError launcher "
        "+ unwrap_citylearn_core_env (HAPPO KPIs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
