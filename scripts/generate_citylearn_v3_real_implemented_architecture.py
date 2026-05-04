"""Generate the real implemented CityLearn v3 MADRL architecture blueprint."""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs"
PDF_REAL = DOCS_DIR / "PLANO_REAL_IMPLEMENTADO_CITYLEARN_V3_MADRL.pdf"
PNG_REAL = DOCS_DIR / "PLANO_REAL_IMPLEMENTADO_CITYLEARN_V3_MADRL.png"
PDF_INTEGRATED = DOCS_DIR / "PLANO_INTEGRADO_CITYLEARN_V3_MADRL.pdf"
PNG_INTEGRATED = DOCS_DIR / "PLANO_INTEGRADO_CITYLEARN_V3_MADRL.png"

BLUE = "#102a43"
INK = "#17212b"
MUTED = "#52606d"
DATA = "#2563eb"
CORE = "#0f766e"
ADAPTER = "#7c3aed"
TRAIN = "#b45309"
OUT = "#15803d"
WARN = "#b91c1c"
COMPAT = "#475569"
BG = "#fbfdff"
GRID = "#cbd5e1"


def wrap(text: str, width: int) -> str:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return "\n".join(lines)


def box(ax, x, y, w, h, title, body, color, *, title_size=11.2, body_size=7.7, width=30):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.26,rounding_size=0.65",
        linewidth=1.45,
        edgecolor=color,
        facecolor=BG,
    )
    ax.add_patch(patch)
    ax.text(x + 0.7, y + h - 0.9, title, fontsize=title_size, weight="bold", color=color, va="top")
    ax.text(x + 0.7, y + h - 3.1, wrap(body, width), fontsize=body_size, color=INK, va="top", linespacing=1.12)
    return patch


def arrow(ax, p1, p2, color=BLUE, label=None, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.55,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if label:
        ax.text(
            (p1[0] + p2[0]) / 2,
            (p1[1] + p2[1]) / 2 + 0.7,
            label,
            fontsize=7.2,
            color=color,
            ha="center",
            weight="bold",
        )


def section(ax, x, y, text, color):
    ax.add_patch(Rectangle((x, y), 24, 2.0, facecolor=color, edgecolor=color))
    ax.text(x + 12, y + 1.03, text, fontsize=9.8, color="white", weight="bold", ha="center", va="center")


def table(ax, x, y, w, h, headers, rows, col_widths, *, title=None):
    if title:
        ax.text(x, y + h + 1.4, title, fontsize=12, color=BLUE, weight="bold", va="bottom")
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=BLUE, linewidth=1.2))
    header_h = 3.2
    ax.add_patch(Rectangle((x, y + h - header_h), w, header_h, facecolor="#eaf2ff", edgecolor=BLUE, linewidth=0.8))
    col_x = [x]
    for cw in col_widths:
        col_x.append(col_x[-1] + w * cw)
    for cx in col_x[1:-1]:
        ax.plot([cx, cx], [y, y + h], color=GRID, linewidth=0.8)
    ax.plot([x, x + w], [y + h - header_h, y + h - header_h], color=BLUE, linewidth=0.9)
    for i, header in enumerate(headers):
        ax.text((col_x[i] + col_x[i + 1]) / 2, y + h - 1.8, header, fontsize=7.8, color=BLUE, weight="bold", ha="center", va="center")
    row_h = (h - header_h) / len(rows)
    for r, row in enumerate(rows):
        top = y + h - header_h - r * row_h
        ax.plot([x, x + w], [top - row_h, top - row_h], color=GRID, linewidth=0.65)
        for c, value in enumerate(row):
            ax.text(
                col_x[c] + 0.45,
                top - row_h / 2,
                wrap(str(value), 24 if c != 2 else 32),
                fontsize=6.7,
                color=INK,
                ha="left",
                va="center",
                linespacing=1.05,
            )


def main() -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(36, 20), dpi=190)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 86)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 150, 86, facecolor="white", edgecolor="none"))
    ax.add_patch(Rectangle((1, 1), 148, 84, fill=False, edgecolor=BLUE, linewidth=2.2))

    ax.text(4, 82.8, "Arquitectura real implementada del proyecto CityLearn v3 MADRL", fontsize=24, color=BLUE, weight="bold", va="top")
    ax.text(
        4,
        79.8,
        "Este plano muestra solo el flujo activo y los componentes existentes en el repositorio: rutas reales, scripts reales, backends usados, escenarios ejecutados y artefactos generados.",
        fontsize=10.8,
        color=MUTED,
        va="top",
    )

    # Stage labels.
    for x, label in [(5, "INICIO"), (24, "CITYLEARN v2"), (45, "CAPA v3"), (70, "4 MADRL"), (101, "3 EJES"), (126, "SALIDAS"), (143, "FIN")]:
        ax.plot([x, x], [8, 76], color=GRID, linewidth=0.8, linestyle="--")
        ax.text(x, 76.6, label, fontsize=8.6, color=MUTED, weight="bold", ha="center")

    section(ax, 4, 73, "FLUJO ACTIVO IMPLEMENTADO", DATA)

    # Top-level flow.
    box(ax, 4, 61, 18, 10, "1. Entrada real", "CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json\n17 edificios + EV/V2G + PV + baterias.", DATA, width=25)
    box(ax, 25, 61, 18, 10, "2. Nucleo CityLearn v2", "CityLearn/citylearn\nSimulador base conservado: edificios, DERs, EV, precios, carbono, confort y evaluacion.", CORE, width=25)
    box(ax, 46, 61, 19, 10, "3. Capa CityLearn v3", "CityLearn/citylearn/v3\nenvironment.py, config.py, objectives.py, backends.py.\nExpone objetivos y entorno v3.", CORE, width=27)
    box(ax, 68, 61, 22, 10, "4. Adaptador comun", "CityLearn/scripts/citylearn_v3_training_common.py\nCityLearnV3BackendAdapter + wrappers por backend + artefactos.", ADAPTER, width=31)
    box(ax, 93, 61, 22, 10, "5. Launcher oficial", "CityLearn/scripts/launch_citylearn_v3_official_training.ps1\n-Scenario ALL ejecuta E1, E2 y E3.", TRAIN, width=31)
    box(ax, 118, 61, 26, 10, "6. Salida canonica", "outputs/citylearn_v3_madrl_official_full_cuda_v2/{madrl}/{E*_seed_0}/\nlogs, checkpoints, data, figures, tables.", OUT, width=36)

    for p1, p2, color in [((22, 66), (25, 66), DATA), ((43, 66), (46, 66), CORE), ((65, 66), (68, 66), CORE), ((90, 66), (93, 66), ADAPTER), ((115, 66), (118, 66), TRAIN)]:
        arrow(ax, p1, p2, color)

    # Actual MADRL connection table.
    rows = [
        ("HAPPO", "train_citylearn_v3_happo.py", "CityLearnHARLEnv", "external/HARL", "happo/E*_seed_0"),
        ("MASAC", "train_citylearn_v3_masac.py", "CityLearnSMACDiscreteEnv", "external/MARL/src", "masac/E*_seed_0"),
        ("MATD3", "train_citylearn_v3_matd3.py", "CityLearnOffPolicyVecEnv", "external/off-policy", "matd3/E*_seed_0"),
        ("MAAC", "train_citylearn_v3_maac.py", "CityLearnMAACVecEnv", "external/MAAC", "maac/E*_seed_0"),
    ]
    table(
        ax,
        4,
        38.7,
        86,
        17,
        ["MADRL", "Script activo", "Wrapper real", "Backend usado", "Salida"],
        rows,
        [0.12, 0.25, 0.23, 0.20, 0.20],
        title="Conexion real de los 4 MADRL con CityLearn v3",
    )

    # Scenarios and KPIs.
    box(ax, 94, 48.8, 15, 7.5, "E1 / OE1", "Flexibilidad energetica\nscenario_manager.py: RTP, sin outages.\nKPIs: peak_average, ramping_average, one_minus_load_factor_average, EV/bateria/PV.", OUT, width=22)
    box(ax, 111, 48.8, 15, 7.5, "E2 / OE2", "Emisiones CO2\nscenario_manager.py: carbon-aware con outages.\nKPIs: carbon_emissions, control, baseline, delta.", OUT, width=22)
    box(ax, 128, 48.8, 15, 7.5, "E3 / OE3", "Costos energeticos\nscenario_manager.py: RTP + outages.\nKPIs: electricity_cost, cost_peak_average, price_signal_deviation.", OUT, width=22)
    arrow(ax, (104.5, 61), (101.5, 56.3), TRAIN, label="ALL -> E1")
    arrow(ax, (104.5, 61), (118.5, 56.3), TRAIN, label="ALL -> E2")
    arrow(ax, (104.5, 61), (135.5, 56.3), TRAIN, label="ALL -> E3")

    # Artifacts / live.
    box(ax, 94, 36.7, 22, 8, "Monitor real", "monitor_citylearn_v3_official_training.ps1\nMuestra matriz E1/E2/E3 x 4 MADRL, GPU, global_step, episodio, reward, costo, CO2.", TRAIN, width=31)
    box(ax, 119, 36.7, 25, 8, "Artefactos reales por corrida", "live_progress.json, results.json, training_summary.json, timeseries.csv, trace.csv, checkpoint_manifest.json, figures_manifest.json.", OUT, width=35)
    arrow(ax, (115, 41), (119, 41), OUT)

    # Evaluation / benchmark.
    section(ax, 4, 34, "EVALUACION Y CIERRE IMPLEMENTADOS", OUT)
    box(ax, 4, 22.5, 27, 9, "Figuras por entrenamiento", "write_training_artifacts() genera tablas CSV y figuras PNG: rewards, convergencia, resumen por episodio, KPIs por eje y baseline_gain_by_kpi.", OUT, width=39)
    box(ax, 34, 22.5, 27, 9, "Benchmark CityLearn v2", "CityLearn/scripts/benchmark_citylearn_v2_agents.py\nEjecuta agentes originales v2 con el mismo dataset para linea base.", WARN, width=39)
    box(ax, 64, 22.5, 28, 9, "Comparador v2 vs v3", "CityLearn/scripts/compare_citylearn_v2_vs_v3_madrl.py\nLee resultados v2 y v3, calcula diferencias, mejora porcentual y ranking.", WARN, width=40)
    box(ax, 95, 22.5, 23, 9, "Documentacion", "ESTRATEGIA_3PILARES_MADRL.md\ndocs/*.pdf\ndocs/PLAN_TESIS_MADRL_CITYLEARN_V3.docx", BLUE, width=33)
    box(ax, 121, 22.5, 23, 9, "Fin real del flujo", "Resultados cuantitativos y visuales por eje, por MADRL y contra CityLearn v2 para sustento de tesis.", BLUE, width=33)
    for p1, p2, color in [((31, 27), (34, 27), WARN), ((61, 27), (64, 27), WARN), ((92, 27), (95, 27), BLUE), ((118, 27), (121, 27), BLUE)]:
        arrow(ax, p1, p2, color)

    # Compatibility but not active.
    section(ax, 4, 18.5, "COMPONENTES EXISTENTES NO USADOS EN EL LAUNCHER OFICIAL ACTUAL", COMPAT)
    box(
        ax,
        4,
        9.5,
        43,
        7,
        "Compatibilidad MARLlib",
        "Existe CityLearn/citylearn/v3/marllib_env.py y external/MARLlib. Es una capa disponible de compatibilidad, pero el launcher oficial actual no la usa para las 12 corridas.",
        COMPAT,
        width=63,
        title_size=10.5,
    )
    box(
        ax,
        50,
        9.5,
        43,
        7,
        "MATD3implementation",
        "Existe external/MATD3implementation como repositorio clonado/fuente. La ruta activa para MATD3 Python 3.9 en el entrenamiento oficial es external/off-policy.",
        COMPAT,
        width=63,
        title_size=10.5,
    )
    box(
        ax,
        96,
        9.5,
        48,
        7,
        "Lectura correcta del plano",
        "Si un bloque no esta conectado por flecha al launcher oficial, no es parte activa de la corrida actual. La ruta activa esta en la franja superior y en la tabla de conexion MADRL.",
        COMPAT,
        width=70,
        title_size=10.5,
    )

    ax.text(4, 5.5, "Estado de ejecucion esperado: launch_citylearn_v3_official_training.ps1 -Scenario ALL -Episodes 5 -Cuda -> 12 jobs secuenciales: E1/E2/E3 x HAPPO/MASAC/MATD3/MAAC.", fontsize=9.2, color=MUTED)
    ax.text(4, 3.7, "Trazabilidad: cada resultado final se rastrea a dataset, escenario, script MADRL, wrapper, backend, seed, checkpoint, timeseries/trace, KPIs y comparador v2 vs v3.", fontsize=9.2, color=MUTED)

    fig.savefig(PNG_REAL, bbox_inches="tight", dpi=230)
    fig.savefig(PDF_REAL, bbox_inches="tight")
    plt.close(fig)

    shutil.copyfile(PDF_REAL, PDF_INTEGRATED)
    shutil.copyfile(PNG_REAL, PNG_INTEGRATED)

    print(PDF_REAL)
    print(PNG_REAL)
    print(PDF_INTEGRATED)
    print(PNG_INTEGRATED)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
