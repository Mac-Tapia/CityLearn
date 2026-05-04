"""Generate a professional CityLearn v3 MADRL architecture and workflow PDF."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "docs"
PDF_PATH = OUTPUT_DIR / "ARQUITECTURA_FLUJO_CITYLEARN_V3_MADRL.pdf"
PNG_ARCH_PATH = OUTPUT_DIR / "ARQUITECTURA_CITYLEARN_V3_MADRL.png"
PNG_FLOW_PATH = OUTPUT_DIR / "FLUJO_TRABAJO_CITYLEARN_V3_MADRL.png"

BLUEPRINT = "#102a43"
PANEL = "#f7fbff"
INK = "#102033"
MUTED = "#52606d"
ACCENT = "#2f80ed"
GREEN = "#2f855a"
AMBER = "#b7791f"
RED = "#c53030"
PURPLE = "#6b46c1"
GRAY = "#d9e2ec"


def setup_ax(title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(23.4, 16.5), dpi=180)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.add_patch(Rectangle((0, 0), 100, 70, facecolor="white", edgecolor="none"))
    ax.add_patch(Rectangle((1, 1), 98, 68, fill=False, edgecolor=BLUEPRINT, linewidth=2.2))
    ax.text(3, 66.8, title, fontsize=23, weight="bold", color=BLUEPRINT, va="top")
    ax.text(3, 63.8, subtitle, fontsize=12.5, color=MUTED, va="top")
    return fig, ax


def wrapped(text: str, width: int) -> str:
    lines = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return "\n".join(lines)


def box(
    ax,
    xy,
    wh,
    title: str,
    body: str,
    *,
    color=ACCENT,
    fill=PANEL,
    title_size=12.5,
    body_size=9.3,
    wrap=30,
):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.35,rounding_size=0.9",
        linewidth=1.8,
        edgecolor=color,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + 1.2, y + h - 1.6, title, fontsize=title_size, weight="bold", color=color, va="top")
    ax.text(x + 1.2, y + h - 5.0, wrapped(body, wrap), fontsize=body_size, color=INK, va="top", linespacing=1.18)
    return patch


def arrow(ax, start, end, *, color=BLUEPRINT, label: str | None = None, rad=0.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 1.0, label, fontsize=8.8, color=color, ha="center", va="bottom", weight="bold")


def legend(ax, x, y):
    items = [
        (ACCENT, "Simulacion / entorno"),
        (PURPLE, "MADRL CTDE"),
        (GREEN, "KPIs y metricas"),
        (AMBER, "Artefactos"),
        (RED, "Benchmark v2 vs v3"),
    ]
    ax.text(x, y + 3, "Leyenda", fontsize=11.5, weight="bold", color=BLUEPRINT)
    for i, (color, label) in enumerate(items):
        ax.add_patch(Rectangle((x, y - i * 2.5), 1.6, 1.2, facecolor=color, edgecolor=color))
        ax.text(x + 2.3, y + 0.05 - i * 2.5, label, fontsize=9.5, color=INK, va="bottom")


def page_architecture(pdf: PdfPages):
    fig, ax = setup_ax(
        "Arquitectura profesional CityLearn v3 MADRL",
        "Proyecto: Multi-agente de aprendizaje por refuerzo profundo para flexibilidad energetica, emisiones de carbono y eficiencia economica en comunidades inteligentes.",
    )

    box(
        ax,
        (4, 49),
        (20, 11),
        "Dataset oficial",
        "citylearn_challenge_2022_phase_all_plus_evs\n17 edificios + EV/V2G + PV + baterias + precios + carbono + confort.",
        color=ACCENT,
        wrap=26,
    )
    box(
        ax,
        (29, 49),
        (21, 11),
        "CityLearn v2 base",
        "Simulador fisico, DERs, KPIs v2, evaluacion original y agentes base. Se conserva como nucleo del entorno.",
        color=ACCENT,
        wrap=27,
    )
    box(
        ax,
        (55, 49),
        (19, 11),
        "Capa CityLearn v3",
        "Adaptadores Dec-POMDP, PettingZoo-like, CTDE, registros live_progress, trace, timeseries y artefactos.",
        color=ACCENT,
        wrap=26,
    )
    box(
        ax,
        (79, 49),
        (17, 11),
        "Ejecucion GPU",
        "Python 3.9, PyTorch CUDA, entrenamiento secuencial para 8 GB VRAM, monitor PowerShell.",
        color=ACCENT,
        wrap=23,
    )

    arrow(ax, (24, 54.5), (29, 54.5))
    arrow(ax, (50, 54.5), (55, 54.5))
    arrow(ax, (74, 54.5), (79, 54.5))

    ax.text(4, 45.2, "Formulacion multiagente", fontsize=15, weight="bold", color=BLUEPRINT)
    box(
        ax,
        (4, 34),
        (29, 9),
        "Dec-POMDP descentralizado",
        "Cada edificio/EV actua como agente con observacion local y ejecucion descentralizada. El distrito define estado global para entrenamiento centralizado.",
        color=PURPLE,
        wrap=40,
    )
    box(
        ax,
        (36, 34),
        (28, 9),
        "CTDE",
        "Criticos/valores centralizados durante entrenamiento; politicas ejecutables por agente al desplegar. Compatible con HAPPO, MASAC, MATD3 y MAAC.",
        color=PURPLE,
        wrap=39,
    )
    box(
        ax,
        (67, 34),
        (29, 9),
        "Multiobjetivo",
        "Recompensa y reportes orientados a tres ejes: flexibilidad, carbono y costos; KPIs de CityLearn v2 preservados.",
        color=PURPLE,
        wrap=39,
    )
    arrow(ax, (33, 38.5), (36, 38.5), color=PURPLE)
    arrow(ax, (64, 38.5), (67, 38.5), color=PURPLE)

    ax.text(4, 30.2, "Backends oficiales integrados", fontsize=15, weight="bold", color=BLUEPRINT)
    algorithms = [
        ("HAPPO", "external/HARL\nOn-policy, CTDE, politicas por agente, checkpoints HARL.", (4, 19), PURPLE),
        ("MASAC", "external/MARL\nSAC multiagente discreto, buffer, exploracion entropica.", (28, 19), PURPLE),
        ("MATD3", "external/off-policy\nActores continuos, criticos twin, replay buffer.", (52, 19), PURPLE),
        ("MAAC", "external/MAAC\nCritico con atencion, politicas discretas multiagente.", (76, 19), PURPLE),
    ]
    for title, body, pos, color in algorithms:
        box(ax, pos, (20, 8), title, body, color=color, wrap=25, title_size=12.8, body_size=9.1)

    ax.text(4, 14.6, "Ejes del proyecto y KPIs principales", fontsize=15, weight="bold", color=BLUEPRINT)
    box(
        ax,
        (4, 5),
        (28, 8),
        "E1 Flexibilidad energetica",
        "Desplazar cargas y aprovechar almacenamiento, baterias, EVs y autoconsumo. KPIs: peak_average, ramping_average, 1-load_factor_average y KPIs v2 asociados.",
        color=GREEN,
        wrap=40,
    )
    box(
        ax,
        (36, 5),
        (28, 8),
        "E2 Emisiones de CO2",
        "Reducir huella ambiental del distrito y evitar importaciones en alta intensidad de carbono. KPIs: carbon_emissions_total, diferencia vs baseline y KPIs v2 de carbono.",
        color=GREEN,
        wrap=40,
    )
    box(
        ax,
        (68, 5),
        (28, 8),
        "E3 Costos energeticos",
        "Optimizar gasto, reducir picos y aprovechar tarifas dinamicas. KPIs: electricity_cost, peak_average, ramping_average y KPIs v2 economicos.",
        color=GREEN,
        wrap=40,
    )

    legend(ax, 80, 29)
    pdf.savefig(fig, bbox_inches="tight")
    fig.savefig(PNG_ARCH_PATH, bbox_inches="tight", dpi=220)
    plt.close(fig)


def page_workflow(pdf: PdfPages):
    fig, ax = setup_ax(
        "Flujo de trabajo oficial de entrenamiento, evaluacion y comparacion",
        "Secuencia reproducible para lanzar los 4 MADRL sobre los tres ejes del proyecto y comparar CityLearn v2 contra CityLearn v3.",
    )

    steps = [
        ("1. Preparacion", "Validar Python 3.9, PyTorch CUDA, dataset 17 edificios + EV, dependencias externas y paths.", (4, 53), ACCENT),
        ("2. Limpieza", "Detener procesos previos y limpiar outputs/checkpoints para entrenamiento desde cero.", (28, 53), AMBER),
        ("3. Launcher ALL", "launch_citylearn_v3_official_training.ps1 -Scenario ALL -Episodes 5 -Cuda.", (52, 53), ACCENT),
        ("4. Monitor vivo", "PowerShell muestra GPU, job activo, progreso, reward_sum, reward_mean, costo, CO2 y artefactos.", (76, 53), ACCENT),
    ]
    for title, body, pos, color in steps:
        box(ax, pos, (20, 9), title, body, color=color, wrap=27)
    for start_x in [24, 48, 72]:
        arrow(ax, (start_x, 57.5), (start_x + 4, 57.5), color=BLUEPRINT)

    ax.text(4, 47.8, "Matriz oficial de ejecuciones secuenciales", fontsize=15, weight="bold", color=BLUEPRINT)
    y0 = 43
    ax.add_patch(Rectangle((4, 24), 92, 20, facecolor="#fbfdff", edgecolor=BLUEPRINT, linewidth=1.4))
    columns = ["Eje / Escenario", "HAPPO", "MASAC", "MATD3", "MAAC", "Salida esperada"]
    xs = [6, 24, 39, 54, 69, 82]
    for x, col in zip(xs, columns):
        ax.text(x, y0, col, fontsize=10.5, weight="bold", color=BLUEPRINT, ha="center")
    rows = [
        ("E1 Flexibilidad", "happo/E1_seed_0", "masac/E1_seed_0", "matd3/E1_seed_0", "maac/E1_seed_0", "KPIs flex + figuras"),
        ("E2 CO2", "happo/E2_seed_0", "masac/E2_seed_0", "matd3/E2_seed_0", "maac/E2_seed_0", "KPIs CO2 + figuras"),
        ("E3 Costos", "happo/E3_seed_0", "masac/E3_seed_0", "matd3/E3_seed_0", "maac/E3_seed_0", "KPIs costo + figuras"),
    ]
    for r, row in enumerate(rows):
        y = y0 - 4.3 * (r + 1)
        ax.plot([5, 95], [y + 2.1, y + 2.1], color=GRAY, linewidth=1)
        for x, value in zip(xs, row):
            ax.text(x, y, value, fontsize=8.5, color=INK, ha="center", va="center")

    box(
        ax,
        (4, 12),
        (21, 8),
        "Artefactos por corrida",
        "results.json, training_summary.json, timeseries.csv, trace.csv, live_progress.json, checkpoint_manifest.json.",
        color=AMBER,
        wrap=28,
    )
    box(
        ax,
        (29, 12),
        (21, 8),
        "Figuras y tablas",
        "reward_timeseries, convergence, episode summary, KPI tables, objective_kpis y comparativas.",
        color=AMBER,
        wrap=29,
    )
    box(
        ax,
        (54, 12),
        (19, 8),
        "Baseline v2",
        "Agentes originales CityLearn v2: RBC/SAC/MARLISA u otros disponibles, evaluados con el mismo dataset.",
        color=RED,
        wrap=27,
    )
    box(
        ax,
        (77, 12),
        (19, 8),
        "Comparador maestro",
        "compare_citylearn_v2_vs_v3_madrl.py calcula delta, mejora porcentual y ranking por eje/MADRL.",
        color=RED,
        wrap=27,
    )
    arrow(ax, (25, 16), (29, 16), color=AMBER)
    arrow(ax, (50, 16), (54, 16), color=RED)
    arrow(ax, (73, 16), (77, 16), color=RED)

    box(
        ax,
        (4, 3.8),
        (92, 5),
        "Criterio de exito del proyecto",
        "Demostrar que CityLearn v3 con MADRL CTDE multiobjetivo mejora o caracteriza de forma superior la flexibilidad, las emisiones de CO2 y los costos frente a la linea base CityLearn v2, usando los mismos datos y KPIs oficiales.",
        color=GREEN,
        wrap=135,
        title_size=12.8,
        body_size=9.5,
    )

    pdf.savefig(fig, bbox_inches="tight")
    fig.savefig(PNG_FLOW_PATH, bbox_inches="tight", dpi=220)
    plt.close(fig)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with PdfPages(PDF_PATH) as pdf:
        page_architecture(pdf)
        page_workflow(pdf)

    print(PDF_PATH)
    print(PNG_ARCH_PATH)
    print(PNG_FLOW_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
