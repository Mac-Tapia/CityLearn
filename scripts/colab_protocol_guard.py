"""Hard guard against legacy Colab 9+3 layout and Drive-sourced training scripts."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

PROTOCOL_ID = "two_phase_happo_masac_v3"
COLAB_CODE_ROOT = "/content/MADRLCitytleranflexresdr"
LEGACY_DRIVE_ROOT = "/content/drive/MyDrive/MADRL_CityLearn_v3/MADRLCitytleranflexresdr"

LEGACY_MARKERS = (
    "FASE 1: HAPPO + MATD3",
    "run_two_phase_jobs",
    "TWO_PHASE_LIGHT",
    "algo_sequential",
)

LAUNCHER_REQUIRED = (
    "LAUNCHER_PROTOCOL_ID",
    PROTOCOL_ID,
    "run_two_phase_happo_masac_jobs",
    "TWO_PHASE_ORDER",
    "TWO_PHASE_P1_HM",
    "TWO_PHASE_P2_HM",
)

MONITOR_REQUIRED = (
    "MONITOR_PROTOCOL_ID",
    PROTOCOL_ID,
    "FOUR_PHASE_ORDER",
    "TWO_PHASE_P1",
    "TWO_PHASE_P2",
)


def is_colab_runtime() -> bool:
    return os.path.isdir("/content")


def assert_not_legacy_path(path: Path, *, role: str) -> None:
    resolved = str(path.resolve())
    if "MADRL_CityLearn_v3" in resolved:
        raise RuntimeError(
            f"{role} apunta al clone legacy en Drive: {path}\n"
            f"El codigo debe vivir solo en {COLAB_CODE_ROOT} (celda 1.2)."
        )
    if is_colab_runtime() and not resolved.startswith(COLAB_CODE_ROOT):
        raise RuntimeError(
            f"{role} fuera del repo Colab canonico: {path}\n"
            f"Esperado prefijo {COLAB_CODE_ROOT}. Ejecuta celda 1.2 (hard sync)."
        )


def validate_launcher_source(text: str, *, path: str = "") -> None:
    missing = [s for s in LAUNCHER_REQUIRED if s not in text]
    legacy = [s for s in LEGACY_MARKERS if s in text]
    if missing or legacy:
        lines = [f"Launcher invalido{(' (' + path + ')') if path else ''}."]
        if missing:
            lines.append(f"  Faltan: {missing}")
        if legacy:
            lines.append(f"  Layout legacy detectado: {legacy}")
        lines.append("  Ejecuta celda 1.2 (git reset --hard) y vuelve a 6.1 -> 7.1.")
        raise RuntimeError("\n".join(lines))


def validate_monitor_source(text: str, *, path: str = "") -> None:
    missing = [s for s in MONITOR_REQUIRED if s not in text]
    legacy = [s for s in LEGACY_MARKERS if s in text]
    if missing or legacy:
        lines = [f"Monitor invalido{(' (' + path + ')') if path else ''}."]
        if missing:
            lines.append(f"  Faltan: {missing}")
        if legacy:
            lines.append(f"  Layout legacy detectado: {legacy}")
        lines.append("  Ejecuta celda 1.2 (git reset --hard) y vuelve a 6.1 -> 7.1.")
        raise RuntimeError("\n".join(lines))


def validate_repo(repo: Path) -> None:
    repo = repo.resolve()
    launcher = repo / "CityLearn/scripts/colab_a100_official_launcher.py"
    monitor = repo / "CityLearn/scripts/colab_a100_live_monitor.py"
    for script in (launcher, monitor):
        if not script.is_file():
            raise FileNotFoundError(f"Script requerido no encontrado: {script}")
        assert_not_legacy_path(script, role=script.name)
    validate_launcher_source(launcher.read_text(encoding="utf-8"), path=str(launcher))
    validate_monitor_source(monitor.read_text(encoding="utf-8"), path=str(monitor))


def quarantine_legacy_drive_scripts(*, dry_run: bool = False) -> List[str]:
    """Rename legacy Drive CityLearn/scripts so stale 9+3 code cannot be imported."""
    actions: List[str] = []
    legacy_scripts = Path(LEGACY_DRIVE_ROOT) / "CityLearn/scripts"
    if not legacy_scripts.is_dir():
        return actions

    launcher = legacy_scripts / "colab_a100_official_launcher.py"
    should_quarantine = True
    if launcher.is_file():
        text = launcher.read_text(encoding="utf-8")
        if PROTOCOL_ID in text and not any(m in text for m in LEGACY_MARKERS):
            should_quarantine = False

    if not should_quarantine:
        return actions

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = legacy_scripts.parent / f"scripts.legacy_quarantine_{stamp}"
    actions.append(f"quarantine {legacy_scripts} -> {dest}")
    if not dry_run:
        legacy_scripts.rename(dest)
    return actions


def sniff_monitor_stdout(stdout: str) -> None:
    """Fail fast if monitor output looks like legacy 9+3 layout."""
    bad_phrases = (
        "FASE 1: HAPPO + MATD3",
        "FASE 2: MASAC (3 jobs",
        "En espera de inicio: delay=600",
    )
    hits = [p for p in bad_phrases if p in stdout]
    if hits:
        raise RuntimeError(
            "Monitor legacy en ejecucion (layout 9+3):\n"
            + "\n".join(f"  - {h}" for h in hits)
            + "\nDeten el entrenamiento, ejecuta 1.2 -> 1.5 -> 6.1 -> 7.1."
        )
    if f"protocol={PROTOCOL_ID}" not in stdout:
        raise RuntimeError(
            f"Monitor sin linea protocol={PROTOCOL_ID}. "
            "Colab esta usando scripts antiguos; ejecuta celda 1.2."
        )


def validate_parallelization_strategy(strategy: str, *, parallelization: Optional[dict] = None) -> None:
    """Validate launcher manifest strategy for two_phase_happo_masac (incl. dynamic backfill)."""

    strategy = str(strategy or "")
    par = dict(parallelization or {})
    dynamic_backfill = bool(par.get("dynamic_backfill", True))
    if dynamic_backfill:
        required = ("dynamic_backfill", "HAPPO+MASAC", "cap=6")
        missing = [token for token in required if token not in strategy]
        if missing:
            raise RuntimeError(
                f"strategy dynamic_backfill invalida: faltan {missing} en {strategy!r}"
            )
    else:
        required = ("Phase1=HAPPO+MASAC", "6 parallel")
        missing = [token for token in required if token not in strategy]
        if "no stagger" not in strategy.lower():
            missing.append("no stagger")
        if missing:
            raise RuntimeError(f"strategy secuencial invalida: faltan {missing} en {strategy!r}")
    for algo in ("HAPPO", "MASAC", "MATD3", "MAAC"):
        if algo not in strategy.upper():
            raise RuntimeError(f"strategy sin algoritmo {algo}: {strategy!r}")


def validate_dry_run_status(status: dict) -> None:
    """Validate official_full_status.json from launcher --dry-run."""

    if status.get("status") != "dry_run":
        raise RuntimeError(f"status esperado dry_run, obtuvo {status.get('status')!r}")
    if status.get("execution") != "two_phase_happo_masac":
        raise RuntimeError(
            f"execution={status.get('execution')!r} — falta --execution-mode two_phase_happo_masac"
        )
    par = dict(status.get("parallelization") or {})
    validate_parallelization_strategy(par.get("strategy", ""), parallelization=par)
    # Acepta cualquier numero de episodios (prueba rapida o produccion). Solo se
    # exige que haya al menos 1 episodio; a100_ready es informativo (50 ep oficial).
    episodes = int(status.get("episodes") or 0)
    if episodes < 1:
        raise RuntimeError(f"episodes debe ser >= 1, obtuvo {episodes}")
    jobs = list(status.get("jobs") or [])
    if len(jobs) != 12:
        raise RuntimeError(f"se esperaban 12 jobs, obtuvo {len(jobs)}")
    if any(float(j.get("startup_delay_seconds") or 0) > 0 for j in jobs):
        raise RuntimeError("stagger detectado (startup_delay_seconds > 0) — layout legacy")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    verify = sub.add_parser("verify-repo", help="Validate launcher/monitor under repo root")
    verify.add_argument("--repo", default=COLAB_CODE_ROOT)

    quarantine = sub.add_parser(
        "quarantine-legacy-drive",
        help="Rename legacy Drive CityLearn/scripts if it contains 9+3 layout",
    )
    quarantine.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "verify-repo":
        validate_repo(Path(args.repo))
        print(f"[protocol-guard] OK repo={args.repo} protocol={PROTOCOL_ID}")
        return 0

    if args.cmd == "quarantine-legacy-drive":
        actions = quarantine_legacy_drive_scripts(dry_run=args.dry_run)
        for action in actions:
            print(f"[protocol-guard] {action}")
        if not actions:
            print("[protocol-guard] nada que cuarentenar (sin clone legacy o ya actualizado)")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
