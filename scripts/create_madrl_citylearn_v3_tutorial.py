"""Create the CityLearn v3 MADRL tutorial notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "examples" / "madrl_citylearn_v3_tutorial.ipynb"
PROJECT_TITLE = (
    "MULTI-AGENTE DE APRENDIZAJE POR REFUERZO PROFUNDO PARA GESTIÓN "
    "COORDINADA DE FLEXIBILIDAD ENERGÉTICA, EMISIONES DE CARBONO Y "
    "EFICIENCIA ECONÓMICA EN COMUNIDADES INTELIGENTES"
)


cells = []


def md(source: str) -> None:
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    })


def code(source: str) -> None:
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    })


def _comment_for_code_line(line: str) -> str:
    stripped = line.strip()

    if stripped.startswith("!"):
        return "Ejecuta un comando de shell/notebook para preparar o verificar el entorno."
    if stripped.startswith(("import ", "from ")):
        return "Importa dependencias necesarias para esta seccion."
    if stripped.startswith("def "):
        name = stripped.split("def ", 1)[1].split("(", 1)[0]
        return f"Define la funcion auxiliar `{name}`."
    if stripped.startswith("class "):
        name = stripped.split("class ", 1)[1].split("(", 1)[0].split(":", 1)[0]
        return f"Define la clase `{name}`."
    if stripped.startswith("return "):
        return "Retorna el resultado calculado por la funcion."
    if stripped.startswith("if "):
        return "Evalua una condicion antes de continuar el flujo."
    if stripped.startswith("elif "):
        return "Evalua una condicion alternativa."
    if stripped == "else:":
        return "Ejecuta la rama alternativa cuando la condicion previa no se cumple."
    if stripped.startswith("for "):
        return "Itera sobre una coleccion de elementos."
    if stripped.startswith("while "):
        return "Mantiene un ciclo mientras la condicion sea verdadera."
    if stripped == "try:":
        return "Inicia un bloque protegido para capturar errores controlados."
    if stripped.startswith("except "):
        return "Captura una excepcion para manejarla sin detener todo el notebook."
    if stripped == "finally:":
        return "Ejecuta acciones finales independientemente del resultado anterior."
    if stripped.startswith("with "):
        return "Abre un contexto administrado para usar recursos de forma segura."
    if stripped.startswith("raise "):
        return "Interrumpe la ejecucion con un error explicito si falla una condicion critica."
    if stripped.startswith("print("):
        return "Muestra informacion de seguimiento para el usuario."
    if stripped.startswith("display("):
        return "Renderiza una tabla, figura o Markdown dentro del notebook."
    if stripped.startswith(("plt.", "fig,", "fig ", "ax.", "axes")):
        return "Configura o dibuja una visualizacion."
    if stripped.startswith(("subprocess.", "command = ", "MONITOR_COMMAND", "OFFICIAL_COMMAND")):
        return "Construye o ejecuta comandos externos del flujo de trabajo."
    if stripped.startswith(("]", ")", "}", "],", "),", "},")):
        return "Cierra la estructura de datos o llamada definida arriba."
    if stripped.startswith(("'", '"')) and ("--" in stripped or ".py" in stripped or "powershell" in stripped.lower()):
        return "Declara un argumento o componente del comando ejecutable."
    if "=" in stripped and not stripped.startswith(("==", "!=", ">=", "<=")):
        name = stripped.split("=", 1)[0].strip().split(" ")[-1]
        return f"Configura o actualiza `{name}`."

    return "Ejecuta una instruccion necesaria para esta celda."


def _comment_code_source(source: list[str]) -> list[str]:
    commented: list[str] = []
    previous_nonempty_was_comment = False

    for raw_line in source:
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped == "":
            commented.append(raw_line)
            previous_nonempty_was_comment = False
            continue

        if stripped.startswith("#"):
            commented.append(raw_line)
            previous_nonempty_was_comment = True
            continue

        indent = line[: len(line) - len(line.lstrip())]
        if not previous_nonempty_was_comment:
            commented.append(f"{indent}# {_comment_for_code_line(line)}\n")

        commented.append(raw_line)
        previous_nonempty_was_comment = False

    return commented


def comment_notebook_code_cells(cells_payload: list[dict]) -> None:
    for cell in cells_payload:
        if cell.get("cell_type") != "code":
            continue

        cell["source"] = _comment_code_source(cell.get("source", []))


md("""
<a href="https://colab.research.google.com/github/Mac-Tapia/CityLearn/blob/citylearn-v3-madrl/examples/madrl_citylearn_v3_tutorial.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
""")

md(f"""
# {PROJECT_TITLE}

## Tutorial CityLearn v3 MADRL sobre CityLearn v2

Este notebook sigue la estructura pedagogica del tutorial original de CityLearn: contexto, datos, preprocesamiento, entorno, KPIs, visualizacion, control, entrenamiento, evaluacion, ajuste y siguientes pasos. La diferencia es que aqui el controlador ya no es un unico agente RL, sino un sistema **MADRL colaborativo** con **Dec-POMDP**, **CTDE** y cuatro backends oficiales: **HAPPO**, **MASAC**, **MATD3** y **MAAC**.

La idea central del proyecto es conservar **CityLearn v2** como simulador, dataset, fisica y fuente oficial de KPIs, agregando una capa **CityLearn v3** para entrenamiento multiagente profundo y evaluacion multiobjetivo.
""")

md("""
# Glossary

- **CityLearn v2**: simulador base usado para edificios, baterias, PV, EVs, tarifas, intensidad de carbono y KPIs `evaluate_v2`.
- **CityLearn v3 MADRL**: capa experimental de este proyecto que adapta CityLearn v2 a Dec-POMDP, CTDE, backends MADRL oficiales y reportes por ejes.
- **MADRL**: aprendizaje por refuerzo profundo multiagente.
- **Dec-POMDP**: juego de Markov parcialmente observable descentralizado; cada edificio observa localmente y actua localmente.
- **CTDE**: entrenamiento centralizado y ejecucion descentralizada; el critico puede usar estado global durante entrenamiento, pero cada actor ejecuta con informacion local.
- **HAPPO**: actor-critic multiagente de HARL, usado con critico centralizado.
- **MASAC**: variante multiagente de Soft Actor-Critic usada sobre un estado global estilo SMAC.
- **MATD3**: TD3 multiagente con critico centralizado y actores continuos por agente.
- **MAAC**: Actor-Attention-Critic multiagente con critico de atencion.
- **OE1**: flexibilidad energetica.
- **OE2**: emisiones de CO2.
- **OE3**: costos energeticos.
- **Baseline CityLearn v2**: referencia usada por CityLearn para calcular ratios, deltas y comparaciones de KPIs.
""")

md("""
<a name="overview"></a>

# Overview

El tutorial original de CityLearn enseña como pasar de datos y reglas de control a agentes RL que modifican acciones de almacenamiento. Este notebook conserva esa ruta de aprendizaje, pero cambia el foco hacia comunidades inteligentes donde cada edificio es un agente coordinado.

El flujo de trabajo sera:

1. Revisar el objetivo cientifico y los ejes de evaluacion.
2. Cargar el dataset CityLearn v2 con 17 edificios + EV.
3. Inspeccionar clima, precios, carbono y archivos de edificios.
4. Construir el entorno Dec-POMDP de CityLearn v3.
5. Validar observaciones locales, acciones locales y estado global CTDE.
6. Evaluar KPIs CityLearn v2 y KPIs del proyecto por OE1/OE2/OE3.
7. Revisar los cuatro backends MADRL oficiales.
8. Ejecutar entrenamientos cortos o lanzar entrenamiento oficial.
9. Analizar `results.json`, `timeseries.csv`, `trace.csv`, checkpoints, figuras y tablas.
10. Comparar algoritmos contra la linea base CityLearn v2.
""")

md("""
## Arquitectura y flujo renderizables del proyecto

Los cambios recientes del proyecto quedaron documentados en un plano y en un Markdown maestro listo para renderizar con Mermaid:

- `docs/ARQUITECTURA_Y_FLUJO_TRABAJO_CITYLEARN_V3_MADRL.md`
- `docs/PLANO_REAL_IMPLEMENTADO_CITYLEARN_V3_MADRL.pdf`
- `docs/PLANO_INTEGRADO_CITYLEARN_V3_MADRL.pdf`

El flujo real implementado es: dataset oficial -> CityLearn v2 -> capa CityLearn v3 -> adaptador comun Dec-POMDP/CTDE -> cuatro scripts MADRL -> launcher `-Scenario ALL` -> artefactos por eje -> benchmark CityLearn v2 -> comparador v2 vs v3.
""")

md("""
## Contributions

Este proyecto aporta una integracion reproducible para estudiar control coordinado en comunidades de edificios:

- Mantiene CityLearn v2 como fuente oficial de datos, dinamica fisica y KPIs.
- Expone cada edificio como agente descentralizado.
- Incluye EVs dentro de los espacios de accion/observacion de los edificios.
- Permite CTDE con estado global durante entrenamiento y ejecucion local por edificio.
- Integra cuatro backends MADRL oficiales sin implementar algoritmos dentro de `citylearn.agents`.
- Genera artefactos tecnicos comparables: checkpoints, resultados JSON, series temporales, trazas por agente, figuras y tablas.
- Ordena la evaluacion en tres ejes: flexibilidad, CO2 y costos.
""")

md("""
## Learning Outcomes

Al finalizar este notebook deberias poder:

- Explicar por que CityLearn v3 sigue usando CityLearn v2 como entorno de entrenamiento.
- Identificar agentes, observaciones, acciones y estado global en un Dec-POMDP.
- Distinguir KPIs CityLearn v2 de los ejes de evaluacion del proyecto.
- Ejecutar validaciones de estructura antes de entrenar.
- Lanzar entrenamientos cortos para HAPPO, MASAC, MATD3 y MAAC.
- Leer los artefactos generados por cada MADRL.
- Interpretar graficas de convergencia, exploracion, eficiencia, recompensas, returns y comparacion con baseline.
""")

md("""
<a name="climate-impact"></a>

# Climate Impact

Las comunidades inteligentes pueden desplazar cargas, usar almacenamiento y coordinar EVs para reducir importaciones en horas criticas. Esta coordinacion tiene tres impactos medibles:

- **Flexibilidad energetica**: reducir picos, rampas y dependencia de importacion desde red.
- **Emisiones de CO2**: evitar consumo en horas con alta intensidad de carbono.
- **Eficiencia economica**: reducir costo total, aprovechar tarifas dinamicas y limitar demanda pico.

El aporte MADRL consiste en aprender politicas coordinadas para muchos edificios sin exigir que cada edificio observe todo el distrito en ejecucion.
""")

md("""
## Diagnóstico de la realidad

El problema energético de edificios no es marginal. UNEP y GlobalABC reportan que, en 2022, edificios y construcción concentraron cerca del 34% de la demanda energética global y 37% de las emisiones de CO2 relacionadas con energía y procesos, además de una brecha creciente frente a la trayectoria necesaria de descarbonización (United Nations Environment Programme & Global Alliance for Buildings and Construction, 2024). La IEA estima que la operación de edificios representa alrededor del 30% del consumo final de energía y 26% de emisiones energéticas globales; también advierte que el sector debe acelerar eficiencia, electrificación, resiliencia y reducción de emisiones para alinearse con el escenario Net Zero (International Energy Agency, 2023).

En comunidades urbanas, la penetración de PV, baterías, bombas de calor, vehículos eléctricos y tarifas dinámicas aumenta la flexibilidad disponible, pero también incrementa la complejidad de coordinación. CityLearn fue propuesto precisamente para estandarizar investigación en RL/MARL para respuesta de demanda y gestión energética urbana, en un contexto donde la integración de renovables, almacenamiento y EVs introduce nuevos desafíos operativos para la red (Vázquez-Canteli et al., 2020). CityLearn v2 amplía ese marco hacia comunidades grid-interactive, resilientes, ocupante-céntricas y carbon-aware con DERs, V2G y confort térmico (Nweye et al., 2025).
""")

md("""
## Descripción problemática

La problemática de tesis se puede resumir así: una comunidad con 17 edificios y EVs dispone de recursos flexibles, pero las decisiones locales no coordinadas pueden aumentar picos, rampas, importaciones en horas de alta intensidad de carbono y costos bajo tarifas dinámicas. Un controlador centralizado puro puede explotar información global, pero escala mal, reduce privacidad y no representa adecuadamente la ejecución real de edificios heterogéneos. Un controlador independiente por edificio preserva descentralización, pero puede sufrir no estacionariedad, pobre asignación de crédito y decisiones incompatibles con el objetivo distrital.

Por ello se adopta un **Dec-POMDP colaborativo con CTDE**: durante entrenamiento se permite usar estado global para estabilizar críticos y aprendizaje; durante ejecución, cada edificio actúa con su observación local. Esta formulación sigue la motivación clásica de Dec-POMDP para control descentralizado bajo incertidumbre (Bernstein et al., 2002) y la tradición MARL moderna de entrenamiento centralizado con políticas descentralizadas (Lowe et al., 2017; Rashid et al., 2018).

La hipótesis operativa del proyecto es que los cuatro MADRL oficiales pueden aprender políticas colaborativas que mejoren, respecto a la línea base CityLearn v2, uno o más de los ejes: **OE1 flexibilidad energética**, **OE2 emisiones de CO2** y **OE3 eficiencia económica**. La comparación final no se debe basar solo en reward: debe usar KPIs CityLearn v2, series técnicas, trazas por agente, checkpoints y figuras por eje.
""")

md("""
<a name="target-audience"></a>

# Target Audience

Este notebook esta pensado para investigadores en energia e IA, estudiantes de RL/MARL, usuarios de CityLearn que necesitan reproducir experimentos con multiples edificios y EV, y evaluadores de tesis que necesitan ver claramente datos, algoritmos, KPIs y artefactos.
""")

md("""
<a name="prereqs"></a>

# Prerequisites

Se recomienda tener Python 3.9, el entorno `.venv39-citylearn-v3`, PyTorch con CUDA si se va a entrenar en GPU, repositorios externos bajo `external/`, y conocimientos basicos de RL, actor-critic, SAC/TD3/PPO y evaluacion energetica.

Los entrenamientos completos pueden tardar bastante. Las celdas de entrenamiento incluyen banderas para evitar ejecutar procesos largos por accidente.
""")

md("""
<a name="background"></a>

# Background

## Grid-Interactive Efficient Buildings and Energy Flexibility

Los edificios interactivos con la red pueden modificar su consumo neto usando almacenamiento electrico, almacenamiento termico, PV, EVs y control de cargas. La flexibilidad se observa en la forma de la curva agregada: picos, rampas, factor de carga, autoconsumo y exportacion.

## Carbon-Aware District Operation

Cuando existe una serie de intensidad de carbono, la politica puede aprender a desplazar importaciones hacia horas menos intensivas en CO2. Por eso el segundo eje no se trata como metrica secundaria sino como objetivo completo.

## Economic Efficiency

La eficiencia economica combina costo de energia, precios dinamicos, reduccion de picos y respuesta a senales tarifarias. Una politica puede reducir costo sin necesariamente reducir emisiones; por eso los tres ejes se reportan por separado.
""")

md("""
## Marco teórico y estado del arte relacionado

### CityLearn y comunidades grid-interactive

CityLearn nace como entorno estandarizado para investigar RL/MARL en respuesta de demanda y gestión energética urbana, buscando facilitar comparación y replicabilidad entre algoritmos (Vázquez-Canteli et al., 2020). CityLearn v2 amplía el alcance hacia comunidades con DERs, EV/V2G, resiliencia, confort y control carbon-aware, lo que lo hace adecuado para evaluar flexibilidad, emisiones y costos en un mismo simulador (Nweye et al., 2025).

### Dec-POMDP, CTDE y coordinación multiagente

El Dec-POMDP formaliza decisiones secuenciales descentralizadas con observabilidad parcial, donde cada agente dispone de información local y el equipo comparte un objetivo global (Bernstein et al., 2002). En MARL profundo, CTDE se usa para reducir no estacionariedad durante entrenamiento sin exigir información global en ejecución. MADDPG introdujo un esquema actor-critic con políticas locales y críticos que pueden observar acciones/observaciones de otros agentes (Lowe et al., 2017). QMIX mostró otra ruta CTDE: aprender un valor conjunto centralizado factorizable en utilidades por agente para ejecución descentralizada (Rashid et al., 2018).

### MADRL para energía, demanda flexible y EVs

La literatura reciente en smart grids y edificios muestra que MARL es pertinente cuando existen muchos recursos distribuidos, información local, privacidad, tarifas dinámicas y necesidad de coordinación. En tesis de maestría, González Rotger (2021) aplicó MARL a HVAC en BEMS y reportó trade-offs entre energía, confort y calidad de aire. Fonseca (2023) estudió integración de activos flexibles, EVs, V2G y comunidades energéticas con MADRL multiobjetivo. Dong (2022) combinó predicción de picos y MARL para gestión de DERs en smart grids con entrenamiento centralizado y ejecución distribuida. En tesis doctoral, Almannouny (2025) abordó pricing dinámico y respuesta de demanda integrada con DRL para sistemas multi-energía.

### Relación con los tres ejes de este proyecto

- **OE1 flexibilidad energética**: se conecta con reducción de picos/rampas, autoconsumo, baterías, EV/V2G y balance comunitario.
- **OE2 emisiones de CO2**: se conecta con control carbon-aware e importaciones en horas de alta intensidad de carbono.
- **OE3 eficiencia económica**: se conecta con precios dinámicos, costos de electricidad, respuesta de demanda y reducción de demanda pico.

El marco teórico justifica que el proyecto no trate estos ejes como métricas aisladas. Son objetivos parcialmente conflictivos, por lo que deben reportarse por separado y compararse con baseline mediante KPIs CityLearn v2.
""")

md("""
### KPIs por eje de investigación

Cada eje tiene un conjunto explícito de KPIs. Los ejes son las **métricas científicas del proyecto**; los KPIs son las variables CityLearn v2/v3 usadas para medir cada eje contra la línea base.

| Eje | Objetivo | KPIs incluidos |
|---|---|---|
| **OE1 Flexibilidad energética** | Aumentar desplazamiento de carga, aprovechamiento de baterías, EVs/V2G, PV, autoconsumo e intercambio comunitario. | `grid_import`, `grid_import_control`, `grid_import_baseline`, `grid_import_delta`, `zero_net_energy`, `net_exchange_control`, `net_exchange_baseline`, `net_exchange_delta`, `grid_export_ratio`, `grid_export_control`, `grid_export_baseline`, `grid_export_delta`, `peak_average`, `ramping_average`, `one_minus_load_factor_average`, `pv_generation_total`, `pv_generation_daily_average`, `pv_export_total`, `pv_export_daily_average`, `pv_self_consumption_ratio`, `community_local_traded_total`, `community_local_traded_daily_average`, `community_import_share`, `battery_charge_total`, `battery_discharge_total`, `battery_throughput_total`, `battery_equivalent_full_cycles`, `battery_capacity_fade_ratio`, `ev_departure_count`, `ev_departure_met_count`, `ev_departure_within_tolerance_count`, `ev_departure_success_rate`, `ev_departure_within_tolerance_rate`, `ev_departure_soc_deficit_mean`, `ev_charge_total`, `ev_v2g_export_total`. |
| **OE2 Emisiones de CO2** | Reducir la huella ambiental del distrito y evitar importaciones en horas de alta intensidad de carbono. | `carbon_emissions`, `carbon_emissions_control`, `carbon_emissions_baseline`, `carbon_emissions_delta`, `carbon_emissions_daily_average_control`, `carbon_emissions_daily_average_baseline`, `carbon_emissions_daily_average_delta`. |
| **OE3 Costos energéticos** | Optimizar gasto energético, reducir picos con efecto económico y aprovechar tarifas dinámicas. | `electricity_cost`, `electricity_cost_control`, `electricity_cost_baseline`, `electricity_cost_delta`, `electricity_cost_daily_average_control`, `electricity_cost_daily_average_baseline`, `electricity_cost_daily_average_delta`, `cost_peak_average`, `cost_ramping_average`, `cost_one_minus_load_factor_average`, `price_signal_deviation`. |

`price_signal_deviation` se mantiene como KPI derivado del proyecto porque no es una salida nativa de `evaluate_v2` en este código; se calcula desde importación neta distrital y `electricity_pricing`. Todos los demás KPIs provienen de CityLearn v2 o de agregaciones trazables sobre sus series.
""")

md("""
## Control Theories for Smart Communities

En el tutorial original, el usuario pasa de reglas RBC a Q-learning y SAC. En este proyecto el salto conceptual es hacia control multiagente:

- **RBC**: reglas fijas, utiles como linea base y diagnostico.
- **RL centralizado**: un agente decide todas las acciones; puede escalar mal con muchos edificios.
- **MARL/MADRL descentralizado**: cada edificio decide su accion.
- **CTDE**: durante entrenamiento se permite informacion global para estabilizar el aprendizaje; en ejecucion cada actor usa observacion local.
""")

md("""
## Reinforcement Learning and MADRL for CityLearn

Cada paso del entorno entrega observaciones locales por edificio, acciones continuas por edificio y recompensas. La formulacion Dec-POMDP usada aqui es:

- agentes: `Building_1`, ..., `Building_17`;
- observacion local: variables CityLearn v2 habilitadas para cada edificio;
- accion local: almacenamiento, EV y otros actuadores disponibles del edificio;
- estado global CTDE: concatenacion de observaciones locales;
- recompensa v3: `CityLearnV3MADRLRewardFunction`, con pesos por eje y perfil por algoritmo;
- agregacion colaborativa Dec-POMDP: `team_mean` por defecto despues de calcular la recompensa v3;
- evaluacion: KPIs CityLearn v2 y reporte CityLearn v3 por ejes.
""")

md("""
### Reward Function v3: pesos por eje y perfil por MADRL

La recompensa usada en los entrenamientos CityLearn v3 no usa los pesos base heredados de `MARL` como criterio principal. Los scripts `train_citylearn_v3_*.py` fuerzan `CityLearnV3MADRLRewardFunction`, que combina:

- pesos multiobjetivo por escenario `E1/E2/E3`;
- multiplicadores especificos por MADRL;
- componente EV/V2G separado para SoC, restricciones de carga, autoconsumo y uso de excedentes;
- mezcla local/equipo mediante `team_reward_ratio`.

Esto separa claramente reward de entrenamiento, agregacion colaborativa Dec-POMDP y KPIs CityLearn v2 de evaluacion final.
""")

code("""
from citylearn.reward_function import (
    CITYLEARN_V3_AXIS_REWARD_WEIGHTS,
    CITYLEARN_V3_MADRL_REWARD_PROFILES,
)

axis_reward_table = pd.DataFrame.from_dict(CITYLEARN_V3_AXIS_REWARD_WEIGHTS, orient='index')
axis_reward_table.index.name = 'scenario'
display(axis_reward_table)

profile_rows = []
for algorithm, profile in CITYLEARN_V3_MADRL_REWARD_PROFILES.items():
    if algorithm == 'MADRL':
        continue
    multipliers = profile['axis_weight_multipliers']
    profile_rows.append({
        'algorithm': algorithm,
        'profile_name': profile['profile_name'],
        'flex_multiplier': multipliers['flex'],
        'carbon_multiplier': multipliers['carbon'],
        'cost_multiplier': multipliers['cost'],
        'team_reward_ratio': profile['team_reward_ratio'],
        'ev_weight': profile['ev_weight'],
        'reward_scale': profile['reward_scale'],
        'peak_weight': profile['peak_weight'],
        'ramp_weight': profile['ramp_weight'],
    })

reward_profile_table = pd.DataFrame(profile_rows)
display(reward_profile_table)
""")

code("""
def effective_axis_weights(scenario: str, algorithm: str) -> Dict[str, float]:
    base = CITYLEARN_V3_AXIS_REWARD_WEIGHTS[scenario]
    multipliers = CITYLEARN_V3_MADRL_REWARD_PROFILES[algorithm]['axis_weight_multipliers']
    weighted = {
        key: base[key] * multipliers[key]
        for key in ['flex', 'carbon', 'cost']
    }
    total = sum(weighted.values())
    return {
        key: weighted[key] / total
        for key in weighted
    }


effective_rows = []
for algorithm in ['HAPPO', 'MASAC', 'MATD3', 'MAAC']:
    for scenario in SCENARIOS:
        row = {
            'algorithm': algorithm,
            'scenario': scenario,
            **effective_axis_weights(scenario, algorithm),
        }
        effective_rows.append(row)

effective_reward_table = pd.DataFrame(effective_rows)
display(effective_reward_table)
""")

md("""
## CityLearn

CityLearn v3 no reemplaza CityLearn v2. Lo envuelve.

- CityLearn v2 conserva datasets, fisica, evaluacion y API base.
- CityLearn v3 agrega adaptadores Dec-POMDP, reportes multiobjetivo, backends oficiales y estructura de artefactos.
- Los algoritmos MADRL viven en `external/`, no dentro de `citylearn.agents`.
""")

md("""
### Environment

El entorno principal es `CityLearnDecPOMDPEnv`, compatible con la idea de `ParallelEnv`: cada agente recibe su observacion y devuelve su accion. La propiedad `state()` da el estado global usado por algoritmos CTDE.
""")

md("""
### Control

| Algoritmo | Entrenamiento | Ejecucion |
|---|---|---|
| HAPPO | critico centralizado HARL | actor por edificio |
| MASAC | estado global estilo SMAC | accion discreta mapeada a CityLearn |
| MATD3 | critico con observaciones/acciones conjuntas | actor continuo por edificio |
| MAAC | critico de atencion multiagente | politica local por edificio |
""")

md("""
### Backends oficiales, GitHub y papers

| Algoritmo | Paper base | GitHub oficial / backend usado | Rol en este proyecto |
|---|---|---|---|
| HAPPO | Zhong et al. (2024), *Heterogeneous-Agent Reinforcement Learning*, JMLR. https://jmlr.org/papers/v25/23-0488.html | HARL: https://github.com/PKU-MARL/HARL | Backend principal para HAPPO con políticas heterogéneas y critic centralizado. |
| MASAC / mSAC | Pu et al. (2021), *Decomposed Soft Actor-Critic Method for Cooperative Multi-Agent Reinforcement Learning*. https://arxiv.org/abs/2104.06655 | MARL: https://github.com/puyuan1996/MARL | Backend paper-repository para mSAC/MASAC cooperativo. |
| MATD3 | Ackermann et al. (2019), *Reducing Overestimation Bias in Multi-Agent Domains Using Double Centralized Critics*. https://arxiv.org/abs/1910.01465 | Repositorio original: https://github.com/JohannesAck/MATD3implementation; backend PyTorch usado: https://github.com/marlbenchmark/off-policy | El repo original es referencia oficial; para Python 3.9 se usa backend PyTorch source-backed. |
| MAAC | Iqbal y Sha (2019), *Actor-Attention-Critic for Multi-Agent Reinforcement Learning*. https://arxiv.org/abs/1810.02912 | MAAC: https://github.com/shariqiqbal2810/MAAC | Backend original con crítico de atención multiagente. |
| MARLlib | Hu et al. (2023), *MARLlib: A Scalable and Efficient Multi-agent Reinforcement Learning Library*. https://arxiv.org/abs/2210.13708 | MARLlib: https://github.com/Replicable-MARL/MARLlib | Framework MARL adicional para registro del entorno `citylearn_v3`. |

Regla metodológica: ningún algoritmo MADRL se reimplementa dentro de `citylearn.agents`. Cada entrenamiento debe llamar el backend externo correspondiente y registrar en `results.json` el backend, hiperparámetros, checkpoints y artefactos de evaluación.
""")

md("""
### Datasets

El proyecto puede usar cualquier `schema.json` CityLearn v2 compatible. El caso de tesis usa `citylearn_challenge_2022_phase_all_plus_evs` con 17 edificios y EVs.
""")

md("""
### Other Environments

MARLlib queda registrado mediante un adaptador `citylearn_v3`. Los backends oficiales se conservan como fuentes externas:

- `external/HARL`
- `external/MARL`
- `external/off-policy`
- `external/MAAC`
- `external/MARLlib`
- `external/MATD3implementation` como referencia legacy del paper MATD3
""")

md("""
## Other References

Consulta tambien `ESTRATEGIA_3PILARES_MADRL.md`, `CityLearn/CITYLEARN_V3_MADRL.md`, `external/backends.lock.json` y los manifiestos de figuras en `outputs/<experimento>/<madrl>/<escenario>_seed_<seed>/figures/`.
""")

md("""
# Hands-On Experiments

Las siguientes celdas estan disenadas para ejecutarse dentro del repositorio completo `MADRLCitytleranflexresdr`. Algunas son de inspeccion rapida y otras lanzan entrenamientos. Por defecto, las celdas largas quedan protegidas con banderas booleanas.
""")

md("""
<a name="software-requirements"></a>

# Software Requirements

Primero verificamos la version de Python. El proyecto fue preparado para Python 3.9.
""")

code("""
!python --version
""")

md("""
Si estas en una maquina nueva, instala dependencias desde el entorno preparado del proyecto. En este repositorio ya se usa `.venv39-citylearn-v3`; en Colab tendrias que clonar el repositorio con submodulos y recrear el entorno.
""")

code("""
# Ejemplo local, no ejecutar si el entorno ya esta preparado:
# !python -m pip install -e ../CityLearn pytest matplotlib pandas numpy
# !python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
""")

md("""
Importamos librerias comunes y configuramos rutas. La funcion `find_project_root` permite ejecutar el notebook desde la raiz del proyecto o desde `CityLearn/examples`.
""")

code("""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display


def find_project_root(start: Optional[Path] = None) -> Path:
    start = Path.cwd() if start is None else Path(start).resolve()
    candidates = [start, *start.parents]

    for candidate in candidates:
        if (candidate / 'CityLearn').exists() and (candidate / 'external').exists():
            return candidate

    for candidate in candidates:
        if (candidate / 'citylearn').exists() and (candidate / 'examples').exists():
            return candidate.parent if candidate.name == 'CityLearn' else candidate

    return start


PROJECT_ROOT = find_project_root()
CITYLEARN_ROOT = PROJECT_ROOT / 'CityLearn' if (PROJECT_ROOT / 'CityLearn').exists() else PROJECT_ROOT
EXTERNAL_ROOT = PROJECT_ROOT / 'external'
SCRIPTS_DIR = CITYLEARN_ROOT / 'scripts'

for path in [PROJECT_ROOT, CITYLEARN_ROOT, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

print('PROJECT_ROOT =', PROJECT_ROOT)
print('CITYLEARN_ROOT =', CITYLEARN_ROOT)
print('EXTERNAL_ROOT =', EXTERNAL_ROOT)
""")

md("""
Aqui incluimos ajustes globales para el resto del notebook. Para tutorial se usa un horizonte corto; para tesis se usa `8760` pasos por episodio.
""")

code("""
plt.rcParams['figure.figsize'] = (10, 4)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.25
pd.set_option('display.max_columns', 120)

RANDOM_SEED = 0
SCENARIOS = ['E1', 'E2', 'E3']
SCENARIO = 'E1'  # escenario corto para celdas interactivas del tutorial
TUTORIAL_ALGORITHM = 'HAPPO'  # perfil reward v3 usado en celdas interactivas
OFFICIAL_SCENARIO = 'ALL'  # ejecuta E1, E2 y E3 en el launcher oficial
TUTORIAL_EPISODE_TIME_STEPS = 24
OFFICIAL_EPISODE_TIME_STEPS = 8760
ALGORITHMS = ['happo', 'masac', 'matd3', 'maac']
""")

md("""
<a name="data-description"></a>

# Dataset Description

## Loading the Data

El dataset de tesis es una variante CityLearn v2 con 17 edificios y EVs. Se carga desde su `schema.json`, igual que en el tutorial original se carga el dataset base de CityLearn.
""")

code("""
from citylearn.data import DataSet
from citylearn.dec_pomdp import DEFAULT_17_BUILDING_EV_SCHEMA

DATASET_NAME = 'citylearn_challenge_2022_phase_all_plus_evs'
SCHEMA_PATH = Path(DEFAULT_17_BUILDING_EV_SCHEMA)

schema = json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))
print('Dataset:', DATASET_NAME)
print('Schema:', SCHEMA_PATH)
print('Simulation start:', schema.get('simulation_start_time_step'))
print('Simulation end:', schema.get('simulation_end_time_step'))
print('Buildings:', len(schema.get('buildings', {})))
""")

md("""
Podemos listar los datasets disponibles en CityLearn. El dataset de tesis puede no aparecer si es una extension local del proyecto, pero su `schema.json` esta dentro de `CityLearn/data/datasets`.
""")

code("""
try:
    display(pd.Series(sorted(DataSet.get_names()), name='dataset').to_frame())
except Exception as exc:
    print('Could not list DataSet names:', exc)
""")

md("""
### Preview a Building Data File

Inspeccionamos el primer edificio incluido. En CityLearn v2 cada edificio apunta a archivos CSV de cargas, clima, precios e intensidad de carbono.
""")

code("""
def included_buildings(schema: Mapping[str, object]) -> List[str]:
    buildings = schema.get('buildings', {})
    return [name for name, payload in buildings.items() if payload.get('include', True)]


def resolve_dataset_file(schema_path: Path, schema: Mapping[str, object], filename: str) -> Path:
    root = Path(schema.get('root_directory') or schema_path.parent)
    if not root.is_absolute():
        direct = schema_path.parent / root
        root = direct if direct.exists() else schema_path.parent
    return root / filename


building_names = included_buildings(schema)
building_name = building_names[0]
building_schema = schema['buildings'][building_name]
print('Selected building:', building_name)
print('Building keys:', sorted(building_schema.keys()))

building_file = resolve_dataset_file(SCHEMA_PATH, schema, building_schema['energy_simulation'])
building_data = pd.read_csv(building_file)
display(building_data.head())
display(building_data.describe(include='all').T.head(20))
""")

md("""
El archivo del edificio permite revisar cargas no desplazables, condiciones interiores, refrigeracion/calefaccion y otros perfiles. Estas variables son la base fisica que los agentes no deben inventar: el entrenamiento MADRL opera sobre la misma simulacion CityLearn v2.
""")

code("""
columns = [col for col in building_data.columns if any(token in col.lower() for token in ['load', 'cooling', 'heating', 'temperature'])]
columns = columns[:4]
fig, axes = plt.subplots(len(columns), 1, figsize=(11, max(3, 2.2 * len(columns))), sharex=True)
if len(columns) == 1:
    axes = [axes]
for ax, column in zip(axes, columns):
    ax.plot(building_data[column].iloc[:168].values, linewidth=1.2)
    ax.set_title(column)
    ax.set_xlabel('hour')
fig.tight_layout()
plt.show()
""")

md("""
### Preview Weather File

El clima afecta cargas termicas y produccion PV. Revisarlo ayuda a entender por que el aprendizaje debe generalizar por episodios y semillas.
""")

code("""
weather_file = resolve_dataset_file(SCHEMA_PATH, schema, building_schema['weather'])
weather_data = pd.read_csv(weather_file)
display(weather_data.head())
display(weather_data.describe(include='all').T.head(20))

weather_columns = [col for col in weather_data.columns if any(token in col.lower() for token in ['temperature', 'humidity', 'solar', 'radiation'])][:4]
fig, axes = plt.subplots(len(weather_columns), 1, figsize=(11, max(3, 2.2 * len(weather_columns))), sharex=True)
if len(weather_columns) == 1:
    axes = [axes]
for ax, column in zip(axes, weather_columns):
    ax.plot(weather_data[column].iloc[:168].values, linewidth=1.2)
    ax.set_title(column)
fig.tight_layout()
plt.show()
""")

md("""
### Preview Electricity Price Data

El eje OE3 depende de costos y tarifas dinamicas. Esta serie se usa para evaluar si la politica desplaza consumo hacia horas economicamente favorables.
""")

code("""
pricing_file = resolve_dataset_file(SCHEMA_PATH, schema, building_schema['pricing'])
pricing_data = pd.read_csv(pricing_file)
display(pricing_data.head())
pricing_column = pricing_data.columns[0]

fig, ax = plt.subplots(figsize=(11, 2.8))
ax.plot(pricing_data[pricing_column].iloc[:168].values, color='tab:purple', linewidth=1.2)
ax.set_title(f'Electricity price: {pricing_column}')
ax.set_xlabel('hour')
plt.show()
""")

md("""
### Preview Carbon Intensity Data

El eje OE2 usa intensidad de carbono para calcular emisiones y comparar control contra baseline. Si el dataset expone esta serie, CityLearn v2 calcula KPIs de carbono.
""")

code("""
carbon_filename = building_schema.get('carbon_intensity')
if carbon_filename:
    carbon_file = resolve_dataset_file(SCHEMA_PATH, schema, carbon_filename)
    carbon_data = pd.read_csv(carbon_file)
    display(carbon_data.head())
    carbon_column = carbon_data.columns[0]
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.plot(carbon_data[carbon_column].iloc[:168].values, color='tab:green', linewidth=1.2)
    ax.set_title(f'Carbon intensity: {carbon_column}')
    ax.set_xlabel('hour')
    plt.show()
else:
    print('This building does not define a carbon_intensity file.')
""")

md("""
## Data Preprocessing

Igual que el tutorial original, podemos modificar una copia del `schema` para seleccionar edificios, periodo de simulacion y observaciones. Para la tesis, el entrenamiento oficial usa los 17 edificios y el horizonte completo.
""")

code("""
def set_schema_buildings(schema: Dict[str, object], count: int) -> Dict[str, object]:
    output = json.loads(json.dumps(schema))
    names = list(output.get('buildings', {}).keys())
    selected = set(names[:count])
    for name, payload in output.get('buildings', {}).items():
        payload['include'] = name in selected
    return output


def set_schema_simulation_period(schema: Dict[str, object], start: int, end: int) -> Dict[str, object]:
    output = json.loads(json.dumps(schema))
    output['simulation_start_time_step'] = int(start)
    output['simulation_end_time_step'] = int(end)
    return output


def active_observation_names(schema: Mapping[str, object]) -> List[str]:
    observations = schema.get('observations', {})
    return [name for name, payload in observations.items() if payload.get('active', False)]


def active_action_names(schema: Mapping[str, object]) -> List[str]:
    actions = schema.get('actions', {})
    return [name for name, payload in actions.items() if payload.get('active', False)]

print('Active observations:', active_observation_names(schema)[:25])
print('Active actions:', active_action_names(schema))
""")

md("""
### Setting your Random Seed

La semilla controla inicializaciones, escenarios aleatorios y reproducibilidad. En experimentos formales se deben ejecutar varias semillas por algoritmo.
""")

code("""
np.random.seed(RANDOM_SEED)
print('RANDOM_SEED =', RANDOM_SEED)
""")

md("""
### Setting the Buildings, Time Periods and Observations to use in Simulations from the Schema

Para una ejecucion pedagogica se puede usar un periodo corto. Para resultados oficiales se usa todo el ano y todos los edificios.
""")

code("""
tutorial_schema = set_schema_buildings(schema, count=17)
tutorial_schema = set_schema_simulation_period(tutorial_schema, start=0, end=TUTORIAL_EPISODE_TIME_STEPS - 1)
print('Tutorial buildings:', len(included_buildings(tutorial_schema)))
print('Tutorial time steps:', tutorial_schema['simulation_start_time_step'], 'to', tutorial_schema['simulation_end_time_step'])
print('Official time steps:', 0, 'to', OFFICIAL_EPISODE_TIME_STEPS - 1)
""")

md("""
# Initialize a CityLearn v3 Dec-POMDP Environment

La celda siguiente crea el entorno del proyecto. Internamente se usa `CityLearnEnv` de CityLearn v2, pero la interfaz externa es multiagente.
""")

code("""
from citylearn.v3 import (
    describe_environment,
    evaluate_objectives,
    make_citylearn_v3_project_env,
    objective_manifest,
)

env = make_citylearn_v3_project_env(
    scenario=SCENARIO,
    seed=RANDOM_SEED,
    episode_time_steps=TUTORIAL_EPISODE_TIME_STEPS,
    madrl_algorithm=TUTORIAL_ALGORITHM,
)

description = describe_environment(env)
display(pd.Series(description).to_frame('value'))
display(pd.Series(description['reward_metadata']).to_frame('reward_metadata'))
""")

md("""
El objeto `env` conserva el simulador CityLearn v2 en `env.env`, y expone propiedades multiagente como `possible_agents`, `observation_space(agent)`, `action_space(agent)` y `state()`.
""")

code("""
print('Number of agents:', env.num_agents)
print('First agents:', env.possible_agents[:5])

space_rows = []
for agent in env.possible_agents:
    space_rows.append({
        'agent': agent,
        'observation_dim': int(env.observation_space(agent).shape[0]),
        'action_dim': int(env.action_space(agent).shape[0]),
    })

display(pd.DataFrame(space_rows))
print('CTDE state dimension:', env.state_space.shape)
""")

md("""
Ejecutamos unos pasos con acciones cero para confirmar la forma de observaciones, recompensas, terminaciones y estado global. Este no es entrenamiento: es una prueba funcional del Dec-POMDP.
""")

code("""
observations, infos = env.reset(seed=RANDOM_SEED)
print('Observation keys:', list(observations)[:5])
print('Initial CTDE state shape:', env.state().shape)

for step in range(3):
    actions = {
        agent: np.zeros(env.action_space(agent).shape, dtype=np.float32)
        for agent in env.agents
    }
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(f'step={step}', 'reward_mean=', np.mean(list(rewards.values())), 'active_agents=', len(env.agents))
""")

md("""
# Key Performance Indicators for Evaluation

CityLearn v2 produce KPIs de evaluacion. CityLearn v3 organiza esos KPIs en tres ejes de tesis. Esta separacion evita confundir metricas de simulacion con objetivos cientificos.
""")

code("""
objective_info = objective_manifest()
for axis, payload in objective_info['axes'].items():
    print(axis, '-', payload['name'])
    print(' ', payload['statement'])
    print(' ', 'kpi_count =', len(payload['kpis']))

axis_kpi_rows = []
for axis, payload in objective_info['axes'].items():
    for kpi in payload['kpis']:
        trace = objective_info['axis_kpis'].get(kpi, {})
        axis_kpi_rows.append({
            'axis': axis,
            'axis_name': payload['name'],
            'scenario': payload['scenario'],
            'kpi': kpi,
            'source': trace.get('source'),
            'lower_is_better': trace.get('lower_is_better'),
            'citylearn_v2_names': ', '.join(trace.get('citylearn_v2_names', [])),
            'note': trace.get('note', ''),
        })

axis_kpi_manifest = pd.DataFrame(axis_kpi_rows)
display(axis_kpi_manifest)
""")

code("""
report = evaluate_objectives(env)
print('KPI frame rows:', report.get('kpi_frame_rows'))
display(pd.DataFrame([
    {'kpi': name, 'value': value}
    for name, value in report.get('axis_kpis', {}).items()
]).head(30))
""")

md("""
# Convenience Functions to Display Simulation Results

El tutorial original define funciones de visualizacion para KPIs, perfiles de carga y baterias. Aqui definimos funciones equivalentes para los artefactos MADRL: resultados, series temporales, trazas por agente y figuras generadas.
""")

code("""
def run_dir(output_root: Path, algorithm: str, scenario: str = SCENARIO, seed: int = RANDOM_SEED) -> Path:
    return output_root / algorithm.lower() / f'{scenario}_seed_{seed}'


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def load_run_results(path: Path) -> Dict[str, object]:
    return load_json(path / 'data' / 'results.json') or load_json(path / 'results.json')


def load_run_timeseries(path: Path) -> pd.DataFrame:
    csv_path = path / 'data' / 'timeseries.csv'
    if not csv_path.is_file():
        csv_path = path / 'timeseries.csv'
    return pd.read_csv(csv_path) if csv_path.is_file() else pd.DataFrame()


def load_run_trace(path: Path) -> pd.DataFrame:
    csv_path = path / 'data' / 'trace.csv'
    if not csv_path.is_file():
        csv_path = path / 'trace.csv'
    return pd.read_csv(csv_path) if csv_path.is_file() else pd.DataFrame()


def load_objective_kpis(path: Path) -> pd.DataFrame:
    table_path = path / 'figures' / 'tables' / 'objective_kpis.csv'
    return pd.read_csv(table_path) if table_path.is_file() else pd.DataFrame()
""")

code("""
def plot_axis_kpis(results: Mapping[str, object], axis: str) -> plt.Figure:
    report = results.get('citylearn_v3_report', results)
    axis_payload = report.get('objective_axis_kpis', {}).get(axis, {})
    rows = []
    for name, payload in axis_payload.get('kpis', {}).items():
        value = payload.get('value') if isinstance(payload, Mapping) else None
        if value is not None:
            rows.append({'kpi': name, 'value': value})
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, max(3, 0.3 * len(data))))
    if not data.empty:
        ax.barh(data['kpi'], data['value'])
        ax.set_title(f'{axis} KPI profile')
        ax.set_xlabel('value')
    return fig


def plot_district_timeseries(timeseries: pd.DataFrame) -> plt.Figure:
    columns = [
        'district_net_electricity_consumption',
        'district_net_electricity_consumption_without_storage',
        'district_net_electricity_consumption_cost',
        'district_net_electricity_consumption_emission',
        'electricity_price_mean',
        'carbon_intensity_mean',
    ]
    columns = [column for column in columns if column in timeseries.columns]
    fig, axes = plt.subplots(len(columns), 1, figsize=(11, max(3, 2 * len(columns))), sharex=True)
    if len(columns) == 1:
        axes = [axes]
    for ax, column in zip(axes, columns):
        ax.plot(timeseries['global_step'], timeseries[column], linewidth=1.1)
        ax.set_title(column)
    axes[-1].set_xlabel('global_step')
    fig.tight_layout()
    return fig


def display_generated_figures(path: Path, names: Optional[Sequence[str]] = None) -> None:
    manifest_path = path / 'figures' / 'figures_manifest.json'
    manifest = load_json(manifest_path)
    figures = manifest.get('figures', [])
    if names is not None:
        figures = [item for item in figures if item.get('name') in set(names)]
    for item in figures:
        display(Markdown(f"### {item.get('name')}"))
        display(Image(filename=item['path']))
""")

md("""
# Build your Baseline Validation

Para comparacion formal, CityLearn v2 calcula KPIs contra una linea base. En CityLearn v3 no se redefine esa linea base; se reutilizan los valores `baseline`, `control`, `delta` y ratios que provienen de `evaluate_v2`.
""")

code("""
VALIDATE_OBJECTIVES = False
VALIDATION_OUTPUT = PROJECT_ROOT / 'outputs' / 'citylearn_v3_objective_validation_notebook'

if VALIDATE_OBJECTIVES:
    command = [
        sys.executable,
        '-B',
        str(SCRIPTS_DIR / 'validate_citylearn_v3_objectives.py'),
        '--scenario', SCENARIO,
        '--seed', str(RANDOM_SEED),
        '--episode-time-steps', str(TUTORIAL_EPISODE_TIME_STEPS),
        '--include-citylearn-v2-test-agents',
        '--output-dir', str(VALIDATION_OUTPUT),
    ]
    print('Running:', ' '.join(map(str, command)))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
else:
    print('Set VALIDATE_OBJECTIVES=True to run the validation script.')
""")

md("""
# An Introduction to MADRL Algorithms as Adaptive Controllers

En lugar de construir un RBC interactivo o un agente Q-learning tabular, este proyecto usa cuatro algoritmos MADRL profundos respaldados por repositorios oficiales. Antes de entrenar, revisamos el manifiesto de backends.
""")

code("""
from citylearn.v3.backends import citylearn_v3_backend_manifest

backend_manifest = citylearn_v3_backend_manifest()
print('Version layer:', backend_manifest['version_layer'])
print('Simulator:', backend_manifest['simulator'])
display(pd.DataFrame.from_dict(backend_manifest['backends'], orient='index'))
display(pd.Series(backend_manifest['dec_pomdp']).to_frame('value'))
""")

md("""
## Dec-POMDP and CTDE Contract

Cada algoritmo debe cumplir el mismo contrato experimental aunque sus redes internas sean distintas. La comparacion entre algoritmos solo es defendible si el entorno, dataset, horizonte, semillas y KPIs permanecen constantes.
""")

code("""
ctde_contract = pd.DataFrame([
    {
        'algorithm': 'HAPPO',
        'centralized_training': 'HARL centralized critic/share_observation_space',
        'decentralized_execution': 'local actor per building',
        'action_type': 'continuous',
    },
    {
        'algorithm': 'MASAC',
        'centralized_training': 'SMAC-style global state get_state()',
        'decentralized_execution': 'local discrete policy mapped to CityLearn action',
        'action_type': 'discretized continuous action table',
    },
    {
        'algorithm': 'MATD3',
        'centralized_training': 'joint observations/actions in centralized critic',
        'decentralized_execution': 'local continuous actor per building',
        'action_type': 'continuous',
    },
    {
        'algorithm': 'MAAC',
        'centralized_training': 'attention critic over agents',
        'decentralized_execution': 'local discrete policy mapped to CityLearn action',
        'action_type': 'discretized continuous action table',
    },
])
display(ctde_contract)
""")

md("""
## Validate Reward Profiles

Esta validacion confirma que `E1/E2/E3` usan perfiles de recompensa MADRL propios del proyecto CityLearn v3, que los pesos efectivos suman 1 por eje y que no se esta usando la recompensa base MARL como criterio principal.
""")

code("""
REWARD_PROFILE_VALIDATION = PROJECT_ROOT / 'outputs' / 'citylearn_v3_reward_profile_validation.json'
VALIDATE_REWARD_PROFILES = False

if VALIDATE_REWARD_PROFILES:
    command = [
        sys.executable,
        '-B',
        str(SCRIPTS_DIR / 'validate_citylearn_v3_reward_profiles.py'),
        '--output',
        str(REWARD_PROFILE_VALIDATION),
    ]
    print('Running:', ' '.join(map(str, command)))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)

if REWARD_PROFILE_VALIDATION.is_file():
    reward_validation = load_json(REWARD_PROFILE_VALIDATION)
    print('status:', reward_validation['status'])
    display(pd.DataFrame(reward_validation['rows']))
else:
    print('No reward profile validation JSON found yet.')
""")

md("""
# Optimize MADRL Controllers

Los entrenamientos se ejecutan con scripts separados por algoritmo. En el flujo oficial actual se usa `-Scenario ALL`, que ejecuta secuencialmente los tres ejes `E1`, `E2` y `E3` para cada MADRL. Todos escriben la misma estructura de artefactos:

```text
outputs/<experimento>/<madrl>/<escenario>_seed_<seed>/
  data/
  checkpoints/
  figures/
  live_progress.json
  results.json
  training_summary.json
  timeseries.csv
  trace.csv
```

Por seguridad, las celdas de entrenamiento no se ejecutan automaticamente.
""")

code("""
def train_command(
    algorithm: str,
    output_root: Path,
    episode_time_steps: int,
    episodes: int,
    cuda: bool = True,
    scenario: str = SCENARIO,
) -> List[str]:
    script = SCRIPTS_DIR / f'train_citylearn_v3_{algorithm}.py'
    num_env_steps = episode_time_steps * episodes
    command = [
        sys.executable,
        '-B',
        str(script),
        '--scenario', scenario,
        '--seed', str(RANDOM_SEED),
        '--episode-time-steps', str(episode_time_steps),
        '--episodes', str(episodes),
        '--output-dir', str(output_root / algorithm),
    ]
    if algorithm == 'happo':
        command.extend([
            '--num-env-steps', str(num_env_steps),
            '--hidden-size', '384',
            '--torch-threads', '12',
            '--n-rollout-threads', '1',
            '--log-interval', '1',
            '--eval-interval', '1',
            '--live-progress-interval', '250',
        ])
    elif algorithm == 'masac':
        command.extend([
            '--action-bins', '3',
            '--buffer-size', '8',
            '--critic-batch-size', '2',
            '--critic-train-steps', '2',
            '--actor-sample-times', '8',
            '--rnn-hidden-dim', '128',
            '--qmix-hidden-dim', '64',
            '--hyper-hidden-dim', '128',
            '--live-progress-interval', '250',
        ])
    elif algorithm == 'matd3':
        command.extend([
            '--num-env-steps', str(num_env_steps),
            '--batch-size', '512',
            '--buffer-size', '50000',
            '--hidden-size', '384',
            '--train-interval', '100',
            '--num-random-episodes', '1',
            '--live-progress-interval', '250',
        ])
    elif algorithm == 'maac':
        command.extend([
            '--action-bins', '3',
            '--batch-size', '512',
            '--buffer-length', '200000',
            '--steps-per-update', '250',
            '--num-updates', '8',
            '--hidden-size', '384',
            '--attend-heads', '4',
            '--pi-lr', '0.0003',
            '--q-lr', '0.001',
            '--tau', '0.005',
            '--gamma', '0.99',
            '--live-progress-interval', '250',
        ])
    if cuda:
        command.append('--cuda')
    return command


SMOKE_OUTPUT = PROJECT_ROOT / 'outputs' / 'citylearn_v3_notebook_smoke'
for algorithm in ALGORITHMS:
    print(algorithm.upper())
    print(' '.join(map(str, train_command(algorithm, SMOKE_OUTPUT, episode_time_steps=4, episodes=1))))
""")

md("""
## Train

Esta celda ejecuta entrenamientos minimos. Usala para validar que los cuatro backends arrancan, no para obtener resultados cientificos.
""")

code("""
RUN_SMOKE_TRAINING = False

if RUN_SMOKE_TRAINING:
    for algorithm in ALGORITHMS:
        command = train_command(algorithm, SMOKE_OUTPUT, episode_time_steps=4, episodes=1, cuda=True)
        print('Running', algorithm.upper())
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
else:
    print('Set RUN_SMOKE_TRAINING=True to train a tiny run for all algorithms.')
""")

md("""
## Perfil GPU local y limite real de MASAC

El lanzamiento oficial vigente usa un perfil **GPU-tuned conservador** para la RTX 4060 Laptop de 8 GB: redes de 384 unidades en HAPPO/MATD3/MAAC, lotes `512` en MATD3/MAAC, `live_progress_interval=250`, y MASAC con `buffer_size=2`, `critic_batch_size=1`, `critic_train_steps=1`, `actor_sample_times=5`, `rnn_hidden_dim=64`, `qmix_hidden_dim=32` e `hyper_hidden_dim=64`.

En MASAC puede verse memoria GPU alta y utilizacion baja. Esto no significa que CUDA este fallando: el backend oficial alterna entre simulacion secuencial del entorno CityLearn para 17 edificios + EV y actualizaciones PyTorch. Durante el rollout, el cuello de botella es CPU/Python/CityLearn; la GPU se activa mas durante las actualizaciones de red. Subir mas los lotes en la GPU local no necesariamente acelera, porque con 8 GB de VRAM aumenta el costo por actualizacion y el riesgo de quedarse sin memoria; por eso MASAC queda con un perfil estable de 8 GB sin cambiar sus pesos multiobjetivo ni los KPIs del proyecto.

## Official Full Training

El entrenamiento oficial usa 17 edificios + EV, `-Scenario ALL`, 8760 pasos por episodio, 5 episodios y perfil GPU-tuned local conservador. El lanzador ejecuta 12 trabajos secuenciales: `E1/E2/E3 x HAPPO/MASAC/MATD3/MAAC`. Esta ejecucion secuencial reduce conflictos de memoria GPU y deja salidas separadas por eje:

```text
outputs/citylearn_v3_madrl_official_full_cuda_v2/
  happo/E1_seed_0
  happo/E2_seed_0
  happo/E3_seed_0
  masac/E1_seed_0
  ...
```
""")

code("""
OFFICIAL_OUTPUT = PROJECT_ROOT / 'outputs' / 'citylearn_v3_madrl_official_full_cuda_v2'
OFFICIAL_COMMAND = [
    'powershell.exe',
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-File', str(SCRIPTS_DIR / 'launch_citylearn_v3_official_training.ps1'),
    '-Scenario', OFFICIAL_SCENARIO,
    '-Seed', str(RANDOM_SEED),
    '-EpisodeTimeSteps', str(OFFICIAL_EPISODE_TIME_STEPS),
    '-Episodes', '5',
    '-OutputRoot', str(OFFICIAL_OUTPUT),
    '-TorchThreads', '12',
    '-Cuda',
]
print(' '.join(map(str, OFFICIAL_COMMAND)))
""")

code("""
RUN_OFFICIAL_TRAINING = False

if RUN_OFFICIAL_TRAINING:
    subprocess.Popen(OFFICIAL_COMMAND, cwd=PROJECT_ROOT)
    print('Official training launched in background.')
else:
    print('Set RUN_OFFICIAL_TRAINING=True only when you intend to launch the long official run.')
""")

md("""
## Monitor visual local

El proyecto incluye un monitor PowerShell que muestra la matriz `E1/E2/E3 x HAPPO/MASAC/MATD3/MAAC`, uso GPU, proceso activo, `global_step`, episodio, recompensa instantanea, retorno acumulado, funcion reward, perfil MADRL, pesos activos, costo, CO2, carga neta y artefactos recientes.
""")

code("""
MONITOR_COMMAND = [
    'powershell.exe',
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-File', str(SCRIPTS_DIR / 'monitor_citylearn_v3_official_training.ps1'),
    '-OutputRoot', str(OFFICIAL_OUTPUT),
    '-IntervalSeconds', '5',
    '-LogTail', '20',
]
print(' '.join(map(str, MONITOR_COMMAND)))
""")

md("""
## Google Colab GPU A100/T4 Training Cell

Esta celda permite ejecutar el entrenamiento en Google Colab usando GPU. Esta pensada para Colab Pro/Pro+ con **A100** cuando este disponible; tambien funciona con T4/V100, pero sera mas lento. En Colab Linux no se usa el launcher PowerShell; por eso la celda ejecuta directamente los cuatro scripts `train_citylearn_v3_*.py` para cada escenario `E1`, `E2` y `E3`.

La celda esta apagada por defecto. Cambia `RUN_COLAB_GPU_TRAINING = True` solo cuando el repositorio ya este clonado con submodulos y las dependencias instaladas.
""")

code("""
RUN_COLAB_GPU_TRAINING = False
COLAB_OUTPUT = PROJECT_ROOT / 'outputs' / 'citylearn_v3_madrl_colab_gpu'
COLAB_EPISODES = 5
COLAB_EPISODE_TIME_STEPS = 8760
COLAB_SCENARIOS = ['E1', 'E2', 'E3']
COLAB_ALGORITHMS = ['happo', 'masac', 'matd3', 'maac']


def is_google_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False


def colab_madrl_command(algorithm: str, scenario: str) -> List[str]:
    script = SCRIPTS_DIR / f'train_citylearn_v3_{algorithm}.py'
    num_env_steps = COLAB_EPISODE_TIME_STEPS * COLAB_EPISODES
    command = [
        sys.executable,
        '-B',
        str(script),
        '--scenario', scenario,
        '--seed', str(RANDOM_SEED),
        '--episode-time-steps', str(COLAB_EPISODE_TIME_STEPS),
        '--episodes', str(COLAB_EPISODES),
        '--output-dir', str(COLAB_OUTPUT / algorithm),
        '--cuda',
    ]

    if algorithm == 'happo':
        command.extend([
            '--num-env-steps', str(num_env_steps),
            '--hidden-size', '384',
            '--torch-threads', '12',
            '--n-rollout-threads', '1',
            '--log-interval', '1',
            '--eval-interval', '1',
            '--live-progress-interval', '250',
        ])
    elif algorithm == 'masac':
        command.extend([
            '--action-bins', '3',
            '--buffer-size', '8',
            '--critic-batch-size', '2',
            '--critic-train-steps', '2',
            '--actor-sample-times', '8',
            '--rnn-hidden-dim', '128',
            '--qmix-hidden-dim', '64',
            '--hyper-hidden-dim', '128',
            '--live-progress-interval', '250',
        ])
    elif algorithm == 'matd3':
        command.extend([
            '--num-env-steps', str(num_env_steps),
            '--batch-size', '512',
            '--buffer-size', '50000',
            '--hidden-size', '384',
            '--train-interval', '100',
            '--num-random-episodes', '1',
            '--live-progress-interval', '250',
        ])
    elif algorithm == 'maac':
        command.extend([
            '--action-bins', '3',
            '--batch-size', '512',
            '--buffer-length', '200000',
            '--steps-per-update', '250',
            '--num-updates', '8',
            '--hidden-size', '384',
            '--attend-heads', '4',
            '--pi-lr', '0.0003',
            '--q-lr', '0.001',
            '--tau', '0.005',
            '--gamma', '0.99',
            '--live-progress-interval', '250',
        ])
    else:
        raise ValueError(f'Unknown algorithm: {algorithm}')

    return command


if RUN_COLAB_GPU_TRAINING:
    import torch

    if not is_google_colab():
        print('Aviso: esta celda tambien puede correr localmente, pero fue preparada para Google Colab.')

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA no esta disponible. En Colab ve a Runtime > Change runtime type > GPU.')

    device_name = torch.cuda.get_device_name(0)
    print('CUDA device:', device_name)
    if 'A100' not in device_name.upper():
        print('Aviso: no parece ser A100. El entrenamiento puede funcionar, pero sera mas lento.')

    for scenario in COLAB_SCENARIOS:
        for algorithm in COLAB_ALGORITHMS:
            command = colab_madrl_command(algorithm, scenario)
            print('\\n===', algorithm.upper(), scenario, '===')
            print(' '.join(map(str, command)))
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)

    print('Colab GPU training completed:', COLAB_OUTPUT)
else:
    print('Set RUN_COLAB_GPU_TRAINING=True only in Google Colab with GPU enabled.')
    print('Recommended runtime: Colab Pro/Pro+ A100. T4/V100 also works with longer runtime.')
""")

md("""
# Evaluate the Episode Rewards for MADRL Algorithms

Las recompensas no sustituyen los KPIs. Sirven para diagnosticar aprendizaje, convergencia, exploracion y estabilidad. Una politica con recompensa creciente aun puede fallar en CO2 o costo si la recompensa no captura bien esos objetivos.
""")

code("""
def available_run_dirs(output_root: Path, scenarios: Sequence[str] = SCENARIOS) -> Dict[str, Path]:
    runs: Dict[str, Path] = {}
    for algorithm in ALGORITHMS:
        for scenario in scenarios:
            path = run_dir(output_root, algorithm, scenario=scenario)
            if path.exists():
                runs[f'{algorithm}_{scenario}'] = path
    return runs


runs = available_run_dirs(OFFICIAL_OUTPUT)
print('Available runs:')
for run_name, path in runs.items():
    print(run_name, '->', path)
""")

code("""
reward_rows = []
for run_name, path in runs.items():
    timeseries = load_run_timeseries(path)
    if timeseries.empty or 'reward_sum' not in timeseries.columns:
        continue
    grouped = timeseries.groupby('episode', as_index=False).agg(
        reward_total=('reward_sum', 'sum'),
        reward_mean=('reward_mean', 'mean'),
        steps=('global_step', 'count'),
    )
    grouped['run'] = run_name.upper()
    reward_rows.append(grouped)

if reward_rows:
    reward_table = pd.concat(reward_rows, ignore_index=True)
    display(reward_table)
    fig, ax = plt.subplots(figsize=(10, 4))
    for run_name, group in reward_table.groupby('run'):
        ax.plot(group['episode'], group['reward_total'], marker='o', label=run_name)
    ax.set_title('Episode returns by MADRL run')
    ax.set_xlabel('episode')
    ax.set_ylabel('reward_total')
    ax.legend()
    plt.show()
else:
    print('No reward tables available yet.')
""")

md("""
# Analyze Training Artifacts

Cada MADRL produce `data/results.json`, `data/timeseries.csv`, `data/trace.csv`, `data/checkpoint_manifest.json` y una carpeta `figures/` con graficas y tablas.
""")

code("""
REGENERATE_FIGURES = False

if REGENERATE_FIGURES:
    for run_name, path in runs.items():
        command = [sys.executable, '-B', str(SCRIPTS_DIR / 'regenerate_citylearn_v3_figures.py'), str(path)]
        print('Regenerating figures for', run_name.upper())
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
else:
    print('Set REGENERATE_FIGURES=True to rebuild figures/tables from saved CSV/JSON artifacts.')
""")

md("""
Las figuras obligatorias cubren rendimiento, eficiencia, convergencia, comparacion con baseline, evolucion de entrenamiento, exploracion, aprendizaje, recompensas, returns, ganancias y perfiles por eje.
""")

code("""
if runs:
    first_run_name, first_path = next(iter(runs.items()))
    print('Showing figures for:', first_run_name.upper())
    display_generated_figures(first_path, names=[
        'reward_timeseries.png',
        'convergence_returns.png',
        'learning_efficiency.png',
        'citylearn_v2_district_timeseries.png',
        'exploration_action_l2.png',
        'baseline_gain_by_kpi.png',
    ])
else:
    print('No run directories found yet. Train or point OFFICIAL_OUTPUT to an existing run root.')
""")

md("""
# Compare MADRL Algorithms Against Baseline

La comparacion debe hacerse por eje. No se recomienda mezclar todos los KPIs en un unico numero sin explicar pesos y normalizacion. Primero miramos KPIs por algoritmo; despues se puede aplicar TOPSIS o ranking ponderado.
""")

code("""
all_kpi_tables = []
for run_name, path in runs.items():
    table = load_objective_kpis(path)
    if not table.empty:
        table['run'] = run_name.upper()
        table['algorithm'] = run_name.split('_')[0].upper()
        table['scenario'] = run_name.split('_')[1].upper() if '_' in run_name else SCENARIO
        all_kpi_tables.append(table)

if all_kpi_tables:
    kpi_comparison = pd.concat(all_kpi_tables, ignore_index=True)
    display(kpi_comparison.head(20))
else:
    kpi_comparison = pd.DataFrame()
    print('No objective_kpis.csv tables available yet.')
""")

code("""
def plot_algorithm_kpi_comparison(kpi_table: pd.DataFrame, axis: str, kpis: Sequence[str]) -> Optional[plt.Figure]:
    if kpi_table.empty:
        return None
    subset = kpi_table[(kpi_table['axis'] == axis) & (kpi_table['kpi'].isin(kpis))].copy()
    if subset.empty:
        return None
    pivot = subset.pivot_table(index='kpi', columns='algorithm', values='value', aggfunc='first')
    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(pivot))))
    pivot.plot(kind='barh', ax=ax)
    ax.set_title(f'{axis} algorithm KPI comparison')
    ax.set_xlabel('value')
    fig.tight_layout()
    return fig

if not kpi_comparison.empty:
    manifest = objective_manifest()
    for axis, payload in manifest['axes'].items():
        fig = plot_algorithm_kpi_comparison(kpi_comparison, axis, payload['kpis'])
        if fig is None:
            print(f'No KPI values available for {axis}.')
            continue
        plt.show()
""")

md("""
# Tune your MADRL Experiment

Como en el tutorial original se ajusta SAC, aqui se ajusta la configuracion experimental MADRL: horizonte de entrenamiento, semillas, escenario E1/E2/E3, pesos multiobjetivo, arquitectura, frecuencia de actualizacion, tamano de buffer, tasas de aprendizaje, discretizacion de acciones, cabezas de atencion y retardo de politica.
""")

code("""
from citylearn.v3.config import CityLearnV3ExperimentConfig

config = CityLearnV3ExperimentConfig()
print('Algorithms:', config.algorithms)
print('Scenarios:', config.scenarios)
print('Seeds:', config.seeds)
print('Episode time steps:', config.episode_time_steps)
print('Reward function:', config.reward_function)
display(pd.Series(config.hyperparameters).to_frame('value'))
display(axis_reward_table)
display(reward_profile_table)
""")

code("""
official_hyperparameters = pd.DataFrame([
    {
        'algorithm': 'HAPPO',
        'main_parameters': 'hidden_size=256, share_param=False, n_rollout_threads=1',
        'training': 'episode_length=8760, num_env_steps=43800, log_interval=1, eval_interval=1',
        'optimizer': 'HARL official defaults: lr=5e-4, critic_lr=5e-4, gamma=0.99, gae_lambda=0.95, clip=0.2',
    },
    {
        'algorithm': 'MASAC',
        'main_parameters': 'action_bins=3, critic_batch_size=1, buffer_size=2',
        'training': 'episodes=5, n_epoch=5, n_episodes=1, episode_limit=8760',
        'optimizer': 'paper backend defaults plus CityLearn v3 CTDE state_shape',
    },
    {
        'algorithm': 'MATD3',
        'main_parameters': 'batch_size=256, buffer_size=10000, hidden_size=256',
        'training': 'num_env_steps=43800, train_interval=100, num_random_episodes=1',
        'optimizer': 'off-policy defaults: lr=5e-4, gamma=0.99, tau=0.005, target_noise=0.2',
    },
    {
        'algorithm': 'MAAC',
        'main_parameters': 'batch_size=256, buffer_length=100000, hidden_size=256, attend_heads=4',
        'training': 'steps_per_update=100, num_updates=4, episodes=5',
        'optimizer': 'pi_lr=3e-4, q_lr=1e-3, tau=0.005, gamma=0.99, reward_scale=100',
    },
])
display(official_hyperparameters)
""")

md("""
## Set Environment, Agent and Reward Function

Para resultados de tesis no basta una corrida. Una matriz minima defendible usa los cuatro algoritmos, tres escenarios, varias semillas, mismo dataset, mismo horizonte, mismos KPIs y comparacion contra baseline CityLearn v2.
""")

code("""
experiment_matrix = pd.MultiIndex.from_product(
    [config.algorithms, config.scenarios, config.seeds[:3]],
    names=['algorithm', 'scenario', 'seed'],
).to_frame(index=False)
print('Example experiment count with first 3 seeds:', len(experiment_matrix))
display(experiment_matrix.head(20))
""")

md("""
## Submit

En lugar de una celda de envio a leaderboard, este proyecto usa trazabilidad local/GitHub: codigo, `backends.lock.json`, `official_full_manifest.json`, `official_full_status.json`, checkpoints, JSON/CSV/PNG/MD por corrida.
""")

code("""
status_path = OFFICIAL_OUTPUT / 'official_full_status.json'
manifest_path = OFFICIAL_OUTPUT / 'official_full_manifest.json'

if status_path.is_file():
    status = load_json(status_path)
    display(pd.Series({
        'status': status.get('status'),
        'dataset': status.get('dataset'),
        'scenario': status.get('scenario'),
        'scenarios': ', '.join(status.get('scenarios', [])) if isinstance(status.get('scenarios'), list) else status.get('scenarios'),
        'episodes': status.get('episodes'),
        'episode_time_steps': status.get('episode_time_steps'),
        'num_env_steps': status.get('num_env_steps'),
        'torch': status.get('torch'),
        'cuda': status.get('cuda'),
        'output_root': status.get('output_root'),
    }).to_frame('value'))

    jobs = status.get('jobs', [])
    if jobs:
        display(pd.DataFrame(jobs)[['scenario', 'name', 'started_at', 'completed_at', 'exit_code', 'output_dir']])
else:
    print('No official status file found at', status_path)
""")

md("""
# Referencias seleccionadas en formato APA 7

Ackermann, J., Gabler, V., Osa, T., & Sugiyama, M. (2019). *Reducing overestimation bias in multi-agent domains using double centralized critics*. arXiv. https://doi.org/10.48550/arXiv.1910.01465

Almannouny, G. A. (2025). *Intelligent dynamic pricing and integrated demand response for multi-energy systems using deep reinforcement learning* [Doctoral dissertation, University of Glasgow]. Enlighten Theses. https://doi.org/10.5525/gla.thesis.85367

Bernstein, D. S., Givan, R., Immerman, N., & Zilberstein, S. (2002). The complexity of decentralized control of Markov decision processes. *Mathematics of Operations Research, 27*(4), 819-840. https://doi.org/10.1287/moor.27.4.819.297

Dong, J. (2022). *Peak load ensemble prediction and multi-agent reinforcement learning for DER demand response management in smart grids* [Master's thesis, Lakehead University]. Knowledge Commons. https://knowledgecommons.lakeheadu.ca/handle/2453/4944

Fonseca, T. C. C. (2023). *A multi-agent reinforcement learning approach to integrate flexible assets into energy communities* [Master's thesis, Instituto Superior de Engenharia do Porto]. Repositório Científico do Instituto Politécnico do Porto. http://hdl.handle.net/10400.22/24068

González Rotger, C. (2021). *Multi-agent reinforcement learning applied to heating, ventilation, and air conditioning in a building energy management system* [Master's thesis, Universitat de les Illes Balears]. http://hdl.handle.net/11201/158415

Hu, S., Zhong, Y., Gao, M., Wang, W., Dong, H., Liang, X., Li, Z., Chang, X., & Yang, Y. (2023). *MARLlib: A scalable and efficient multi-agent reinforcement learning library*. arXiv. https://arxiv.org/abs/2210.13708

International Energy Agency. (2023). *Buildings*. https://www.iea.org/energy-system/buildings

Iqbal, S., & Sha, F. (2019). *Actor-attention-critic for multi-agent reinforcement learning*. arXiv. https://doi.org/10.48550/arXiv.1810.02912

Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. *Advances in Neural Information Processing Systems, 30*. https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments

Nweye, K., Kaspar, K., Buscemi, G., Fonseca, T., Pinto, G., Ghose, D., Duddukuru, S., Pratapa, P., Li, H., Mohammadi, J., Lino Ferreira, L., Hong, T., Ouf, M., Capozzoli, A., & Nagy, Z. (2025). CityLearn v2: Energy-flexible, resilient, occupant-centric, and carbon-aware management of grid-interactive communities. *Journal of Building Performance Simulation, 18*(1), 17-38. https://doi.org/10.1080/19401493.2024.2418813

Pu, Y., Wang, S., Yang, R., Yao, X., & Li, B. (2021). *Decomposed soft actor-critic method for cooperative multi-agent reinforcement learning*. arXiv. https://doi.org/10.48550/arXiv.2104.06655

Rashid, T., Samvelyan, M., Schroeder de Witt, C., Farquhar, G., Foerster, J., & Whiteson, S. (2018). *QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning*. arXiv. https://doi.org/10.48550/arXiv.1803.11485

United Nations Environment Programme & Global Alliance for Buildings and Construction. (2024). *Global status report for buildings and construction*. https://www.unep.org/resources/report/global-status-report-buildings-and-construction

Vázquez-Canteli, J. R., Dey, S., Henze, G., & Nagy, Z. (2020). *CityLearn: Standardizing research in multi-agent reinforcement learning for demand response and urban energy management*. arXiv. https://doi.org/10.48550/arXiv.2012.10504

Zhong, Y., Kuba, J. G., Feng, X., Hu, S., Ji, J., & Yang, Y. (2024). Heterogeneous-agent reinforcement learning. *Journal of Machine Learning Research, 25*(32), 1-67. https://jmlr.org/papers/v25/23-0488.html
""")

md("""
# Next Steps

1. Reiniciar el entrenamiento oficial secuencial despues de limpiar salidas creadas con rewards anteriores.
2. Regenerar figuras si alguna corrida fue creada antes de `CityLearnV3MADRLRewardFunction`.
3. Consolidar `objective_kpis.csv` de HAPPO, MASAC, MATD3 y MAAC.
4. Crear tablas comparativas por OE1/OE2/OE3.
5. Aplicar ranking TOPSIS o ponderado con pesos justificados.
6. Revisar estabilidad por semilla y no solo una corrida.
7. Reportar limitaciones: costo computacional, discretizacion de algunos backends y dependencia de calidad de recompensa.
""")

md("""
## Other Ideas

- Ejecutar E1, E2 y E3 por separado para observar especializacion de politicas.
- Comparar `team_mean`, `team_sum`, `individual` y recompensa mixta.
- Incorporar analisis de equidad entre edificios.
- Evaluar sensibilidad a EV penetration y disponibilidad PV.
- Generar curvas de Pareto entre flexibilidad, CO2 y costo.
- Usar MARLlib para experimentos adicionales con el mismo adaptador `citylearn_v3`.
""")


comment_notebook_code_cells(cells)

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.9",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(NOTEBOOK_PATH)
print("cells", len(cells))
