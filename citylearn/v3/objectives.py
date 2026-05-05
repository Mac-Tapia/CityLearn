"""Objective and KPI reporting for CityLearn v3 MADRL experiments.

The project metrics are the three thesis axes:

* OE1: energy flexibility.
* OE2: CO2 emissions.
* OE3: energy costs.

CityLearn v2 ``evaluate_v2`` KPIs are used directly when they exist.
``price_signal_deviation`` is marked as a derived project KPI because it is not
exported natively by ``evaluate_v2`` in this codebase.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from citylearn.madrl_kpis import (
    CITYLEARN_V2_KPI_GROUPS,
    evaluate_citylearn_v2_kpi_frame,
    extract_citylearn_v2_kpis,
)


@dataclass(frozen=True)
class ObjectiveAxis:
    code: str
    name: str
    statement: str
    # ``scenario`` is kept for backward-compatible artifacts. ``axis_scenario``
    # is the explicit name used in reports to avoid confusing the objective
    # label with the scenario of a specific MADRL run.
    scenario: str
    kpis: tuple[str, ...]


@dataclass(frozen=True)
class KPITrace:
    name: str
    objective: str
    source: str
    lower_is_better: bool
    citylearn_v2_names: tuple[str, ...] = ()
    note: str = ""


OE1_FLEXIBILITY_KPIS: tuple[str, ...] = (
    "grid_import",
    "grid_import_control",
    "grid_import_baseline",
    "grid_import_delta",
    "zero_net_energy",
    "net_exchange_control",
    "net_exchange_baseline",
    "net_exchange_delta",
    "grid_export_ratio",
    "grid_export_control",
    "grid_export_baseline",
    "grid_export_delta",
    "peak_average",
    "ramping_average",
    "one_minus_load_factor_average",
    "pv_generation_total",
    "pv_generation_daily_average",
    "pv_export_total",
    "pv_export_daily_average",
    "pv_self_consumption_ratio",
    "community_local_traded_total",
    "community_local_traded_daily_average",
    "community_import_share",
    "battery_charge_total",
    "battery_discharge_total",
    "battery_throughput_total",
    "battery_equivalent_full_cycles",
    "battery_capacity_fade_ratio",
    "ev_departure_count",
    "ev_departure_met_count",
    "ev_departure_within_tolerance_count",
    "ev_departure_success_rate",
    "ev_departure_within_tolerance_rate",
    "ev_departure_soc_deficit_mean",
    "ev_charge_total",
    "ev_v2g_export_total",
)

OE2_EMISSIONS_KPIS: tuple[str, ...] = (
    "carbon_emissions",
    "carbon_emissions_control",
    "carbon_emissions_baseline",
    "carbon_emissions_delta",
    "carbon_emissions_daily_average_control",
    "carbon_emissions_daily_average_baseline",
    "carbon_emissions_daily_average_delta",
)

OE3_COST_KPIS: tuple[str, ...] = (
    "electricity_cost",
    "electricity_cost_control",
    "electricity_cost_baseline",
    "electricity_cost_delta",
    "electricity_cost_daily_average_control",
    "electricity_cost_daily_average_baseline",
    "electricity_cost_daily_average_delta",
    "cost_peak_average",
    "cost_ramping_average",
    "cost_one_minus_load_factor_average",
    "price_signal_deviation",
)


OBJECTIVE_AXES: Mapping[str, ObjectiveAxis] = {
    "OE1": ObjectiveAxis(
        code="OE1",
        name="Flexibilidad energetica",
        statement=(
            "Aumentar la capacidad de desplazar cargas y aprovechar "
            "almacenamiento, EVs y autoconsumo en comunidades de edificios "
            "interactivos con la red electrica."
        ),
        scenario="E1",
        kpis=OE1_FLEXIBILITY_KPIS,
    ),
    "OE2": ObjectiveAxis(
        code="OE2",
        name="Emisiones de CO2",
        statement=(
            "Reducir la huella ambiental del distrito, minimizando "
            "importaciones en horas de alta intensidad de carbono."
        ),
        scenario="E2",
        kpis=OE2_EMISSIONS_KPIS,
    ),
    "OE3": ObjectiveAxis(
        code="OE3",
        name="Costos energeticos",
        statement=(
            "Optimizar el gasto energetico, reduciendo picos de demanda y "
            "aprovechando tarifas dinamicas."
        ),
        scenario="E3",
        kpis=OE3_COST_KPIS,
    ),
}

PROJECT_AXIS_METRICS: tuple[str, ...] = tuple(OBJECTIVE_AXES.keys())

HIGHER_IS_BETTER: set[str] = {
    "pv_generation_total",
    "pv_generation_daily_average",
    "pv_self_consumption_ratio",
    "community_local_traded_total",
    "community_local_traded_daily_average",
    "community_import_share",
    "battery_charge_total",
    "battery_discharge_total",
    "battery_throughput_total",
    "battery_equivalent_full_cycles",
    "ev_departure_met_count",
    "ev_departure_within_tolerance_count",
    "ev_departure_success_rate",
    "ev_departure_within_tolerance_rate",
    "ev_charge_total",
    "ev_v2g_export_total",
}

RATIO_TO_BASELINE_KPIS: set[str] = {
    "grid_import",
    "zero_net_energy",
    "grid_export_ratio",
    "peak_average",
    "ramping_average",
    "one_minus_load_factor_average",
    "carbon_emissions",
    "electricity_cost",
    "cost_peak_average",
    "cost_ramping_average",
    "cost_one_minus_load_factor_average",
}

KPI_NOTES: Mapping[str, str] = {
    "grid_import": "CityLearn v2 district import ratio to baseline.",
    "zero_net_energy": "CityLearn v2 district net exchange ratio to baseline.",
    "pv_self_consumption_ratio": "CityLearn v2 PV self-consumption ratio for autoconsumption.",
    "battery_throughput_total": "CityLearn v2 district battery throughput in kWh.",
    "ev_v2g_export_total": "CityLearn v2 EV vehicle-to-grid export in kWh.",
    "carbon_emissions": "CityLearn v2 district CO2 ratio to baseline.",
    "carbon_emissions_control": "Total district greenhouse gas emissions under control in kgCO2.",
    "carbon_emissions_baseline": "Total district greenhouse gas emissions for the CityLearn v2 baseline in kgCO2.",
    "carbon_emissions_delta": "Control minus baseline total district greenhouse gas emissions in kgCO2.",
    "electricity_cost": "CityLearn v2 normalized district cost ratio to baseline.",
    "electricity_cost_control": "Total district electricity cost under control in EUR.",
    "electricity_cost_baseline": "Total district electricity cost for the CityLearn v2 baseline in EUR.",
    "electricity_cost_delta": "Control minus baseline total district electricity cost in EUR.",
    "price_signal_deviation": (
        "Not a native evaluate_v2 KPI in this codebase; derived from district "
        "net import and the electricity_pricing signal."
    ),
}


def _v2_name_lookup() -> Dict[str, tuple[str, ...]]:
    output: Dict[str, tuple[str, ...]] = {}

    for group in CITYLEARN_V2_KPI_GROUPS.values():
        output.update(group)

    return output


def _axis_for_kpi(name: str) -> str:
    for code, axis in OBJECTIVE_AXES.items():
        if name in axis.kpis:
            return code

    return "project_axis_metric"


def _make_kpi_traces() -> Dict[str, KPITrace]:
    citylearn_v2_names = _v2_name_lookup()
    traces: Dict[str, KPITrace] = {}

    for axis in OBJECTIVE_AXES.values():
        for name in axis.kpis:
            source = (
                "derived_from_citylearn_v2_timeseries"
                if name == "price_signal_deviation"
                else "citylearn_v2.evaluate_v2"
            )
            traces[name] = KPITrace(
                name=name,
                objective=axis.code,
                source=source,
                lower_is_better=name not in HIGHER_IS_BETTER,
                citylearn_v2_names=citylearn_v2_names.get(name, ()),
                note=KPI_NOTES.get(name, ""),
            )

    return traces


KPI_TRACES: Mapping[str, KPITrace] = _make_kpi_traces()


def objective_manifest() -> Dict[str, object]:
    """Return a serializable manifest of objectives and KPI provenance."""

    axes = {}
    for code, axis in OBJECTIVE_AXES.items():
        payload = asdict(axis)
        payload["axis_scenario"] = axis.scenario
        payload["scenario_semantics"] = "objective_axis_label"
        axes[code] = payload

    return {
        "axes": axes,
        "project_axis_metrics": PROJECT_AXIS_METRICS,
        "axis_kpis": {name: asdict(trace) for name, trace in KPI_TRACES.items()},
        "metrics": {
            "project_axis_metrics": PROJECT_AXIS_METRICS,
            "note": "The project metrics are the three objective axes OE1, OE2 and OE3.",
        },
        "multiobjective": {
            "method": "weighted_scalar_report_with_topsis_ready_inputs",
            "project_metrics": PROJECT_AXIS_METRICS,
            "lower_is_better": {
                name: trace.lower_is_better
                for name, trace in KPI_TRACES.items()
            },
        },
    }


def _unwrap_env(env):
    if hasattr(env, "env") and hasattr(env.env, "unwrapped"):
        return env.env.unwrapped

    return getattr(env, "unwrapped", env)


def _safe_float(value) -> Optional[float]:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(output):
        return None

    return output


def _value_from_frame(frame: pd.DataFrame, names: Iterable[str]) -> Optional[float]:
    if frame.empty or not {"cost_function", "value"}.issubset(frame.columns):
        return None

    rows = frame[frame["cost_function"].isin(tuple(names))]
    if rows.empty:
        return None

    if {"level", "name"}.issubset(rows.columns):
        district_rows = rows[(rows["level"] == "district") | (rows["name"] == "District")]
        rows = district_rows if not district_rows.empty else rows

    values = [_safe_float(value) for value in rows["value"]]
    values = [value for value in values if value is not None]
    return None if not values else float(np.mean(values))


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or abs(denominator) <= 1e-12:
        return None

    return float(numerator / denominator)


def _series(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _district_price_signal(env, length: int) -> Optional[np.ndarray]:
    price_series = []

    for building in getattr(env, "buildings", []):
        pricing = getattr(building, "pricing", None)
        prices = getattr(pricing, "electricity_pricing", None)

        if prices is None:
            continue

        price_series.append(np.asarray(prices, dtype=float).reshape(-1)[:length])

    if not price_series:
        return None

    min_len = min(len(series) for series in price_series)
    if min_len == 0:
        return None

    return np.nanmean([series[:min_len] for series in price_series], axis=0)


def price_signal_deviation(load: Sequence[float], price: Sequence[float]) -> Optional[float]:
    """Return deviation from an inverse-price demand-response shape.

    Load is clipped to grid import, normalized to [0, 1], and compared with
    ``1 - normalized_price``. Lower values indicate better response to high
    price periods. ``None`` means the price signal has no usable variation.
    """

    load_array = np.clip(_series(load), 0.0, None)
    price_array = _series(price)
    length = min(load_array.size, price_array.size)

    if length == 0:
        return None

    load_array = load_array[:length]
    price_array = price_array[:length]
    price_span = float(price_array.max() - price_array.min())

    if price_span <= 1e-12:
        return None

    load_span = float(load_array.max() - load_array.min())
    load_norm = np.zeros_like(load_array) if load_span <= 1e-12 else (load_array - load_array.min()) / load_span
    price_norm = (price_array - price_array.min()) / price_span
    target_load_shape = 1.0 - price_norm

    return float(np.mean(np.abs(load_norm - target_load_shape)))


def derived_price_signal_kpis(env) -> Dict[str, Optional[float]]:
    """Compute the derived dynamic-tariff KPI from CityLearn v2 time series."""

    control_load = _series(getattr(env, "net_electricity_consumption", []))
    if control_load.size == 0:
        return {}

    price = _district_price_signal(env, control_load.size)
    if price is None:
        return {}

    baseline_load = _series(
        getattr(
            env,
            "net_electricity_consumption_without_storage_and_partial_load",
            getattr(env, "net_electricity_consumption_without_storage", []),
        )
    )
    control = price_signal_deviation(control_load, price)
    baseline = price_signal_deviation(baseline_load, price)
    delta = None if control is None or baseline is None else float(control - baseline)

    return {
        "price_signal_deviation": control,
        "price_signal_deviation_baseline": baseline,
        "price_signal_deviation_delta": delta,
        "price_signal_deviation_ratio": _safe_div(control, baseline),
    }


def derived_price_signal_metrics(env) -> Dict[str, Optional[float]]:
    """Backward-compatible alias for earlier validation scripts."""

    return derived_price_signal_kpis(env)


def _comparison(value: Optional[float], trace: KPITrace, *, baseline: Optional[float]) -> Dict[str, object]:
    if value is None:
        return {
            "available": False,
            "baseline": baseline,
            "delta_vs_baseline": None,
            "improved_vs_baseline": None,
        }

    if baseline is None:
        return {
            "available": True,
            "baseline": None,
            "delta_vs_baseline": None,
            "improved_vs_baseline": None,
        }

    delta = float(value - baseline)
    improved = delta < 0.0 if trace.lower_is_better else delta > 0.0

    return {
        "available": True,
        "baseline": baseline,
        "delta_vs_baseline": delta,
        "improved_vs_baseline": bool(improved),
    }


def _paired_baseline_name(name: str) -> Optional[str]:
    suffix_pairs = (
        ("_daily_average_control", "_daily_average_baseline"),
        ("_control", "_baseline"),
    )

    for control_suffix, baseline_suffix in suffix_pairs:
        if name.endswith(control_suffix):
            return f"{name[:-len(control_suffix)]}{baseline_suffix}"

    return None


def _baseline_for(name: str, values: Mapping[str, float], trace: KPITrace) -> Optional[float]:
    if name == "price_signal_deviation":
        return values.get("price_signal_deviation_baseline")

    paired_name = _paired_baseline_name(name)
    if paired_name is not None:
        return values.get(paired_name)

    if name.endswith("_delta"):
        return 0.0

    if name in RATIO_TO_BASELINE_KPIS:
        return 1.0

    return None


def _axis_baseline_summary(axis: ObjectiveAxis, value_report: Mapping[str, Mapping[str, object]]) -> Dict[str, int]:
    comparisons = [
        value_report[kpi]["comparison"]
        for kpi in axis.kpis
        if value_report[kpi]["comparison"]["improved_vs_baseline"] is not None
    ]
    improved = sum(1 for comparison in comparisons if comparison["improved_vs_baseline"] is True)

    return {
        "comparable_kpis": len(comparisons),
        "improved_kpis": improved,
        "not_improved_kpis": len(comparisons) - improved,
    }


def evaluate_objectives(env) -> Dict[str, object]:
    """Evaluate thesis objectives for a completed or partial CityLearn episode."""

    citylearn_env = _unwrap_env(env)
    frame = evaluate_citylearn_v2_kpi_frame(citylearn_env)
    axis_kpis = extract_citylearn_v2_kpis(frame)
    derived_price_values = {
        key: value
        for key, value in derived_price_signal_kpis(citylearn_env).items()
        if value is not None
    }

    if "price_signal_deviation" in derived_price_values:
        axis_kpis["price_signal_deviation"] = derived_price_values["price_signal_deviation"]

    supporting_values = {
        key: value
        for key, value in derived_price_values.items()
        if key != "price_signal_deviation"
    }
    all_values = {**axis_kpis, **supporting_values}
    value_report: Dict[str, Dict[str, object]] = {}

    for name, trace in KPI_TRACES.items():
        value = all_values.get(name)
        baseline = _baseline_for(name, all_values, trace)
        value_report[name] = {
            "value": value,
            "trace": asdict(trace),
            "comparison": _comparison(value, trace, baseline=baseline),
        }

    axes = {}

    for code, axis in OBJECTIVE_AXES.items():
        axis_payload = asdict(axis)
        axis_payload["axis_scenario"] = axis.scenario
        axis_payload["scenario_semantics"] = "objective_axis_label"
        axes[code] = {
            **axis_payload,
            "kpis": {
                kpi: value_report[kpi]
                for kpi in axis.kpis
            },
            "baseline_comparison": _axis_baseline_summary(axis, value_report),
        }

    reported_axis_kpis = {
        name: all_values.get(name)
        for name in KPI_TRACES
    }

    return {
        "manifest": objective_manifest(),
        "axes": axes,
        "project_axis_metrics": axes,
        "axis_kpis": reported_axis_kpis,
        "supporting_values": supporting_values,
        "all_values": all_values,
        "kpi_frame_rows": int(len(frame)),
    }
