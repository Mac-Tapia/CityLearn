"""Generate a single integrated start-to-end CityLearn v3 MADRL blueprint."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "docs"
PDF_PATH = OUTPUT_DIR / "PLANO_INTEGRADO_CITYLEARN_V3_MADRL.pdf"
PNG_PATH = OUTPUT_DIR / "PLANO_INTEGRADO_CITYLEARN_V3_MADRL.png"

BLUE = "#12355b"
INK = "#17212b"
MUTED = "#546a7b"
BG = "#fbfdff"
DATA = "#2f80ed"
ENV = "#0f766e"
MADRL = "#6b46c1"
TRAIN = "#b7791f"
EVAL = "#2f855a"
COMPARE = "#c53030"
DOC = "#334e68"
LINE = "#bcccdc"


def wrap(text: str, width: int) -> str:
    lines = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return "\n".join(lines)


def box(ax, x, y, w, h, title, body, color, *, title_size=11.8, body_size=8.6, width=28):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.28,rounding_size=0.75",
        linewidth=1.55,
        edgecolor=color,
        facecolor=BG,
    )
    ax.add_patch(patch)
    ax.text(x + 0.8, y + h - 1.1, title, fontsize=title_size, weight="bold", color=color, va="top")
    ax.text(x + 0.8, y + h - 3.7, wrap(body, width), fontsize=body_size, color=INK, va="top", linespacing=1.15)
    return patch


def arrow(ax, p1, p2, color=BLUE, label=None, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.7,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if label:
        ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.9, label, fontsize=7.7, color=color, ha="center", weight="bold")


def lane_label(ax, y, text, color):
    ax.add_patch(Rectangle((1.5, y - 0.8), 12.5, 1.8, facecolor=color, edgecolor=color))
    ax.text(7.75, y + 0.1, text, color="white", fontsize=9.5, weight="bold", ha="center", va="center")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(33.1, 18.6), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 78)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 140, 78, facecolor="white", edgecolor="none"))
    ax.add_patch(Rectangle((1, 1), 138, 76, fill=False, edgecolor=BLUE, linewidth=2.2))

    ax.text(4, 74.5, "Plano integrado y sincronizado del proyecto CityLearn v3 MADRL", fontsize=23, weight="bold", color=BLUE, va="top")
    ax.text(
        4,
        71.6,
        "Inicio -> datos CityLearn v2 -> adecuacion CityLearn v3 -> entrenamiento MADRL CTDE -> artefactos -> evaluacion -> benchmark v2 vs v3 -> resultados de tesis.",
        fontsize=11.5,
        color=MUTED,
        va="top",
    )

    lane_label(ax, 64, "ENTRADAS", DATA)
    lane_label(ax, 52, "ENTORNO", ENV)
    lane_label(ax, 40, "MADRL", MADRL)
    lane_label(ax, 28, "ENTRENAMIENTO", TRAIN)
    lane_label(ax, 16, "EVALUACION", EVAL)
    lane_label(ax, 7, "CIERRE", DOC)

    # Vertical stage separators.
    stages = [
        (5, "INICIO"),
        (23, "BASE V2"),
        (43, "CAPA V3"),
        (65, "4 MADRL"),
        (88, "3 EJES"),
        (109, "SALIDAS"),
        (128, "FIN"),
    ]
    for x, label in stages:
        ax.plot([x, x], [5, 68], color=LINE, linewidth=0.9, linestyle="--")
        ax.text(x, 68.6, label, fontsize=9, weight="bold", color=MUTED, ha="center")

    # Main flow.
    box(ax, 4, 57.5, 17, 10, "1. Inicio del estudio", "Problema: gestion coordinada de comunidad inteligente con flexibilidad, CO2 y costos.", DATA, width=24)
    box(ax, 24, 57.5, 17, 10, "2. Datos oficiales", "Dataset citylearn_challenge_2022_phase_all_plus_evs: 17 edificios + EV/V2G + PV + baterias.", DATA, width=24)
    box(ax, 44, 57.5, 17, 10, "3. KPIs v2", "Se preservan KPIs CityLearn v2 para evaluacion comparable y trazable.", DATA, width=24)
    box(ax, 64, 57.5, 17, 10, "4. Configuracion", "Python 3.9, PyTorch CUDA, seeds, episodios, GPU local y salidas canonicas.", DATA, width=24)
    arrow(ax, (21, 62.5), (24, 62.5), DATA)
    arrow(ax, (41, 62.5), (44, 62.5), DATA)
    arrow(ax, (61, 62.5), (64, 62.5), DATA)

    box(ax, 24, 45.5, 17, 9, "CityLearn v2", "Simulador base: fisica, DERs, EVs, confort, precios, carbono y agentes originales.", ENV, width=24)
    box(ax, 44, 45.5, 17, 9, "CityLearn v3", "Capa agregada: Dec-POMDP, adaptadores multiagente, estado CTDE y registros.", ENV, width=24)
    box(ax, 64, 45.5, 17, 9, "MARLlib/CTDE", "Interfaz compatible para entrenamiento centralizado y ejecucion descentralizada.", ENV, width=24)
    arrow(ax, (32.5, 57.5), (32.5, 54.5), ENV)
    arrow(ax, (41, 50), (44, 50), ENV)
    arrow(ax, (61, 50), (64, 50), ENV)

    box(ax, 44, 33.5, 13.5, 9, "HAPPO", "HARL\non-policy, CTDE, politicas por agente.", MADRL, width=19)
    box(ax, 59, 33.5, 13.5, 9, "MASAC", "MARL/mSAC\nentropia, acciones discretas, buffer.", MADRL, width=19)
    box(ax, 74, 33.5, 13.5, 9, "MATD3", "off-policy\nactores continuos, twin critics.", MADRL, width=19)
    box(ax, 89, 33.5, 13.5, 9, "MAAC", "attention critic\ncoordinacion multiagente.", MADRL, width=19)
    arrow(ax, (72.5, 45.5), (72.5, 42.5), MADRL)

    box(ax, 84, 57.5, 16, 10, "E1 Flexibilidad", "Aumentar capacidad de desplazar cargas y usar almacenamiento, EVs y autoconsumo.", EVAL, width=22)
    box(ax, 102, 57.5, 16, 10, "E2 CO2", "Reducir huella ambiental y evitar importacion en horas de alta intensidad de carbono.", EVAL, width=22)
    box(ax, 120, 57.5, 16, 10, "E3 Costos", "Reducir gasto energetico, picos y aprovechar tarifas dinamicas.", EVAL, width=22)
    arrow(ax, (81, 62.5), (84, 62.5), EVAL)
    arrow(ax, (100, 62.5), (102, 62.5), EVAL)
    arrow(ax, (118, 62.5), (120, 62.5), EVAL)

    box(ax, 84, 21.5, 16, 9, "Launcher ALL", "12 ejecuciones secuenciales:\nE1/E2/E3 x HAPPO/MASAC/MATD3/MAAC.", TRAIN, width=22)
    box(ax, 102, 21.5, 16, 9, "Monitor vivo", "GPU, job activo, global_step, episodio, reward, costo, CO2 y carga neta.", TRAIN, width=22)
    box(ax, 120, 21.5, 16, 9, "Checkpoints", "Modelos, manifests y rutas por algoritmo/eje/seed.", TRAIN, width=22)
    arrow(ax, (102.5, 38), (92, 30.5), TRAIN, rad=-0.2)
    arrow(ax, (100, 26), (102, 26), TRAIN)
    arrow(ax, (118, 26), (120, 26), TRAIN)

    box(ax, 84, 9.5, 16, 9, "Artefactos", "results.json, training_summary.json, timeseries.csv, trace.csv, figures y tables.", EVAL, width=22)
    box(ax, 102, 9.5, 16, 9, "Benchmark v2", "Agentes originales CityLearn v2 con mismo dataset y KPIs.", COMPARE, width=22)
    box(ax, 120, 9.5, 16, 9, "Comparador final", "Delta, mejora %, ranking por eje, graficas y cuadros v2 vs v3.", COMPARE, width=22)
    arrow(ax, (128, 21.5), (128, 18.5), EVAL)
    arrow(ax, (100, 14), (102, 14), COMPARE)
    arrow(ax, (118, 14), (120, 14), COMPARE)

    box(
        ax,
        4,
        3.2,
        74,
        7,
        "Trazabilidad cientifica",
        "Todo resultado final debe poder rastrearse a: dataset oficial, escenario/eje, algoritmo MADRL, hiperparametros, seed, checkpoint, KPIs CityLearn v2, metricas por eje y comparacion contra baseline.",
        DOC,
        width=112,
        title_size=12.5,
        body_size=9,
    )
    box(
        ax,
        82,
        3.2,
        54,
        7,
        "Fin del flujo",
        "Salida final para tesis: evidencia cuantitativa y visual de si CityLearn v3 MADRL mejora flexibilidad, reduce CO2 y optimiza costos frente a CityLearn v2.",
        DOC,
        width=78,
        title_size=12.5,
        body_size=9,
    )

    # Cross-lane synchronization arrows.
    arrow(ax, (52.5, 45.5), (50.5, 42.5), MADRL, label="observacion local")
    arrow(ax, (72.5, 45.5), (82.5, 42.5), MADRL, label="estado global CTDE")
    arrow(ax, (92, 57.5), (92, 30.5), TRAIN, label="E1")
    arrow(ax, (110, 57.5), (110, 30.5), TRAIN, label="E2")
    arrow(ax, (128, 57.5), (128, 30.5), TRAIN, label="E3")

    ax.text(5, 12, "Lectura del plano: seguir las flechas desde INICIO hasta FIN. Las lineas verticales separan etapas y las franjas horizontales separan responsabilidades.", fontsize=9.5, color=MUTED)

    fig.savefig(PNG_PATH, bbox_inches="tight", dpi=220)
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)

    print(PDF_PATH)
    print(PNG_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
