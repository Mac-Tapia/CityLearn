"""CityLearn v2 KPI mapping for MADRL experiments.

The thesis-level MADRL experiments evaluate performance through three project
axes: energy flexibility, CO2 emissions and energy costs. This module
centralizes the exact KPI names exported by ``CityLearnEnv.evaluate_v2`` and
keeps legacy names only as compatibility fallbacks for older outputs.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


DISTRICT = "District"


def unwrap_citylearn_core_env(env):
    """Reach CityLearn core without calling broken HARL ``ShareVecEnv.unwrapped``."""
    if env is None:
        return env
    seen: set[int] = set()
    current = env
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "evaluate_v2"):
            return current
        nested = getattr(current, "env", None)
        if nested is not None and nested is not current:
            current = nested
            continue
        try:
            unwrapped = getattr(current, "unwrapped", current)
        except NameError:
            unwrapped = current
        if unwrapped is current:
            break
        current = unwrapped
    return current


CITYLEARN_V2_KPI_GROUPS: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "flexibility": {
        "grid_import": (
            "district_energy_grid_ratio_to_baseline_import_total_ratio",
        ),
        "grid_import_control": (
            "district_energy_grid_total_import_control_kwh",
        ),
        "grid_import_baseline": (
            "district_energy_grid_total_import_baseline_kwh",
        ),
        "grid_import_delta": (
            "district_energy_grid_total_import_delta_kwh",
            "district_energy_grid_daily_average_import_delta_kwh",
        ),
        "zero_net_energy": (
            "district_energy_grid_ratio_to_baseline_net_exchange_total_ratio",
        ),
        "net_exchange_control": (
            "district_energy_grid_total_net_exchange_control_kwh",
        ),
        "net_exchange_baseline": (
            "district_energy_grid_total_net_exchange_baseline_kwh",
        ),
        "net_exchange_delta": (
            "district_energy_grid_total_net_exchange_delta_kwh",
            "district_energy_grid_daily_average_net_exchange_delta_kwh",
        ),
        "grid_export_ratio": (
            "district_energy_grid_ratio_to_baseline_export_total_ratio",
        ),
        "grid_export_control": (
            "district_energy_grid_total_export_control_kwh",
        ),
        "grid_export_baseline": (
            "district_energy_grid_total_export_baseline_kwh",
        ),
        "grid_export_delta": (
            "district_energy_grid_total_export_delta_kwh",
            "district_energy_grid_daily_average_export_delta_kwh",
        ),
        "peak_average": (
            "district_energy_grid_shape_quality_peak_all_time_average_to_baseline_ratio",
            "district_energy_grid_shape_quality_peak_daily_average_to_baseline_ratio",
        ),
        "ramping_average": (
            "district_energy_grid_shape_quality_ramping_average_to_baseline_ratio",
        ),
        "one_minus_load_factor_average": (
            "district_energy_grid_shape_quality_load_factor_penalty_daily_average_to_baseline_ratio",
            "district_energy_grid_shape_quality_load_factor_penalty_monthly_average_to_baseline_ratio",
        ),
        "pv_generation_total": (
            "district_solar_self_consumption_total_generation_kwh",
        ),
        "pv_generation_daily_average": (
            "district_solar_self_consumption_daily_average_generation_kwh",
        ),
        "pv_export_total": (
            "district_solar_self_consumption_total_export_kwh",
        ),
        "pv_export_daily_average": (
            "district_solar_self_consumption_daily_average_export_kwh",
        ),
        "pv_self_consumption_ratio": (
            "district_solar_self_consumption_ratio_self_consumption_ratio",
        ),
        "community_local_traded_total": (
            "district_energy_grid_community_market_local_traded_total_kwh",
        ),
        "community_local_traded_daily_average": (
            "district_energy_grid_community_market_local_traded_daily_average_kwh",
        ),
        "community_import_share": (
            "district_solar_self_consumption_community_market_import_share_ratio",
        ),
        "battery_charge_total": (
            "district_battery_total_charge_kwh",
        ),
        "battery_discharge_total": (
            "district_battery_total_discharge_kwh",
        ),
        "battery_throughput_total": (
            "district_battery_total_throughput_kwh",
        ),
        "battery_equivalent_full_cycles": (
            "district_battery_health_equivalent_full_cycles_count",
        ),
        "battery_capacity_fade_ratio": (
            "district_battery_health_capacity_fade_ratio",
        ),
        "ev_departure_count": (
            "district_ev_events_departure_count",
        ),
        "ev_departure_met_count": (
            "district_ev_events_departure_met_count",
        ),
        "ev_departure_within_tolerance_count": (
            "district_ev_events_departure_within_tolerance_count",
        ),
        "ev_departure_success_rate": (
            "district_ev_performance_departure_success_ratio",
        ),
        "ev_departure_within_tolerance_rate": (
            "district_ev_performance_departure_within_tolerance_ratio",
        ),
        "ev_departure_soc_deficit_mean": (
            "district_ev_performance_departure_soc_deficit_mean_ratio",
        ),
        "ev_charge_total": (
            "district_ev_total_charge_kwh",
        ),
        "ev_v2g_export_total": (
            "district_ev_total_v2g_export_kwh",
        ),
    },
    "emissions": {
        "carbon_emissions": (
            "district_emissions_ratio_to_baseline_total_ratio",
        ),
        "carbon_emissions_control": (
            "district_emissions_total_control_kgco2",
        ),
        "carbon_emissions_baseline": (
            "district_emissions_total_baseline_kgco2",
        ),
        "carbon_emissions_delta": (
            "district_emissions_total_delta_kgco2",
            "district_emissions_daily_average_delta_kgco2",
        ),
        "carbon_emissions_daily_average_control": (
            "district_emissions_daily_average_control_kgco2",
        ),
        "carbon_emissions_daily_average_baseline": (
            "district_emissions_daily_average_baseline_kgco2",
        ),
        "carbon_emissions_daily_average_delta": (
            "district_emissions_daily_average_delta_kgco2",
        ),
    },
    "costs": {
        "electricity_cost": (
            "district_cost_ratio_to_baseline_total_ratio",
        ),
        "electricity_cost_control": (
            "district_cost_total_control_eur",
        ),
        "electricity_cost_baseline": (
            "district_cost_total_baseline_eur",
        ),
        "electricity_cost_delta": (
            "district_cost_total_delta_eur",
            "district_cost_daily_average_delta_eur",
        ),
        "electricity_cost_daily_average_control": (
            "district_cost_daily_average_control_eur",
        ),
        "electricity_cost_daily_average_baseline": (
            "district_cost_daily_average_baseline_eur",
        ),
        "electricity_cost_daily_average_delta": (
            "district_cost_daily_average_delta_eur",
        ),
        "cost_peak_average": (
            "district_energy_grid_shape_quality_peak_all_time_average_to_baseline_ratio",
            "district_energy_grid_shape_quality_peak_daily_average_to_baseline_ratio",
        ),
        "cost_ramping_average": (
            "district_energy_grid_shape_quality_ramping_average_to_baseline_ratio",
        ),
        "cost_one_minus_load_factor_average": (
            "district_energy_grid_shape_quality_load_factor_penalty_daily_average_to_baseline_ratio",
            "district_energy_grid_shape_quality_load_factor_penalty_monthly_average_to_baseline_ratio",
        ),
        "price_signal_deviation": (
            "district_demand_response_price_signal_deviation_ratio",
        ),
    },
}


CITYLEARN_V2_METRIC_GROUPS: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    # Backward-compatible hook. The project metrics are now the three objective
    # axes, so there are no non-axis CityLearn v2 metrics in the v3 report.
}


LEGACY_VALUE_FALLBACKS: Mapping[str, tuple[str, ...]] = {
    "grid_import": ("electricity_consumption_total",),
    "grid_import_control": ("electricity_consumption_control_total_kwh",),
    "grid_import_baseline": ("electricity_consumption_baseline_total_kwh",),
    "grid_import_delta": (
        "electricity_consumption_delta_total_kwh",
        "electricity_consumption_delta_daily_average_kwh",
    ),
    "zero_net_energy": ("zero_net_energy",),
    "net_exchange_control": ("zero_net_energy_control_total_kwh",),
    "net_exchange_baseline": ("zero_net_energy_baseline_total_kwh",),
    "net_exchange_delta": (
        "zero_net_energy_delta_total_kwh",
        "zero_net_energy_delta_daily_average_kwh",
    ),
    "grid_export_control": ("electricity_export_control_total_kwh",),
    "grid_export_baseline": ("electricity_export_baseline_total_kwh",),
    "grid_export_delta": (
        "electricity_export_delta_total_kwh",
        "electricity_export_delta_daily_average_kwh",
    ),
    "peak_average": ("all_time_peak_average", "daily_peak_average"),
    "ramping_average": ("ramping_average",),
    "one_minus_load_factor_average": (
        "daily_one_minus_load_factor_average",
        "monthly_one_minus_load_factor_average",
    ),
    "cost_peak_average": ("all_time_peak_average", "daily_peak_average"),
    "cost_ramping_average": ("ramping_average",),
    "cost_one_minus_load_factor_average": (
        "daily_one_minus_load_factor_average",
        "monthly_one_minus_load_factor_average",
    ),
    "pv_generation_total": ("pv_generation_total_kwh",),
    "pv_generation_daily_average": ("pv_generation_daily_average_kwh",),
    "pv_export_total": ("pv_export_total_kwh",),
    "pv_export_daily_average": ("pv_export_daily_average_kwh",),
    "pv_self_consumption_ratio": ("pv_self_consumption_ratio",),
    "battery_charge_total": ("bess_charge_total_kwh",),
    "battery_discharge_total": ("bess_discharge_total_kwh",),
    "battery_throughput_total": ("bess_throughput_total_kwh",),
    "battery_equivalent_full_cycles": ("bess_equivalent_full_cycles",),
    "battery_capacity_fade_ratio": ("bess_capacity_fade_ratio",),
    "ev_departure_count": ("ev_departure_events_count",),
    "ev_departure_met_count": ("ev_departure_met_events_count",),
    "ev_departure_within_tolerance_count": ("ev_departure_within_tolerance_events_count",),
    "ev_departure_success_rate": ("ev_departure_success_rate",),
    "ev_departure_within_tolerance_rate": ("ev_departure_within_tolerance_rate",),
    "ev_departure_soc_deficit_mean": ("ev_departure_soc_deficit_mean",),
    "ev_charge_total": ("ev_charge_total_kwh",),
    "ev_v2g_export_total": ("ev_v2g_export_total_kwh",),
    "carbon_emissions": ("carbon_emissions_total",),
    "carbon_emissions_control": ("carbon_emissions_control_total_kgco2",),
    "carbon_emissions_baseline": ("carbon_emissions_baseline_total_kgco2",),
    "carbon_emissions_delta": (
        "carbon_emissions_delta_total_kgco2",
        "carbon_emissions_delta_daily_average_kgco2",
    ),
    "carbon_emissions_daily_average_control": ("carbon_emissions_control_daily_average_kgco2",),
    "carbon_emissions_daily_average_baseline": ("carbon_emissions_baseline_daily_average_kgco2",),
    "carbon_emissions_daily_average_delta": ("carbon_emissions_delta_daily_average_kgco2",),
    "electricity_cost": ("cost_total",),
    "electricity_cost_control": ("cost_control_total_eur",),
    "electricity_cost_baseline": ("cost_baseline_total_eur",),
    "electricity_cost_delta": (
        "cost_delta_total_eur",
        "cost_delta_daily_average_eur",
    ),
    "electricity_cost_daily_average_control": ("cost_control_daily_average_eur",),
    "electricity_cost_daily_average_baseline": ("cost_baseline_daily_average_eur",),
    "electricity_cost_daily_average_delta": ("cost_delta_daily_average_eur",),
}


def _finite_mean(values: Iterable[float]) -> Optional[float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]

    if array.size == 0:
        return None

    return float(array.mean())


def _extract_value(df: pd.DataFrame, names: Iterable[str], *, prefer_district: bool = True) -> Optional[float]:
    if df.empty or "cost_function" not in df.columns or "value" not in df.columns:
        return None

    rows = df[df["cost_function"].isin(tuple(names))]

    if prefer_district and {"level", "name"}.issubset(rows.columns):
        district_rows = rows[(rows["level"] == "district") | (rows["name"] == DISTRICT)]
        rows = district_rows if not district_rows.empty else rows

    if rows.empty:
        return None

    return _finite_mean(rows["value"].to_numpy(dtype=float))


def _extract_group_values(df: pd.DataFrame, groups: Mapping[str, Mapping[str, tuple[str, ...]]]) -> Dict[str, float]:
    output: Dict[str, float] = {}

    for group in groups.values():
        for logical_name, v2_names in group.items():
            value = _extract_value(df, v2_names)

            if value is None:
                value = _extract_value(df, LEGACY_VALUE_FALLBACKS.get(logical_name, ()))

            if value is not None:
                output[logical_name] = value

    return output


def extract_citylearn_v2_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Return only thesis objective KPIs from a CityLearn KPI table.

    Missing KPIs are omitted rather than synthesized. This is intentional:
    thesis reporting should only use KPIs that CityLearn actually exports or
    explicitly documented project-derived KPIs.
    """

    return _extract_group_values(df, CITYLEARN_V2_KPI_GROUPS)


def extract_citylearn_v2_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Return non-axis CityLearn v2 performance metrics.

    Kept for compatibility with earlier scripts. The current thesis contract
    treats the three objective axes as the project metrics, so this returns an
    empty mapping unless a future non-axis group is explicitly added.
    """

    return _extract_group_values(df, CITYLEARN_V2_METRIC_GROUPS)


def evaluate_citylearn_v2_kpis(env) -> Dict[str, float]:
    """Evaluate CityLearn v2 KPIs when available, with legacy fallback."""

    if hasattr(env, "evaluate_v2"):
        frame = env.evaluate_v2()
        kpis = extract_citylearn_v2_kpis(frame)

        if kpis:
            return kpis

    if hasattr(env, "evaluate"):
        return extract_citylearn_v2_kpis(env.evaluate())

    return {}


def evaluate_citylearn_v2_kpi_frame(env) -> pd.DataFrame:
    """Return the complete CityLearn KPI table without dropping any metric."""

    if hasattr(env, "evaluate_v2"):
        return env.evaluate_v2()

    if hasattr(env, "evaluate"):
        return env.evaluate()

    return pd.DataFrame(columns=["cost_function", "value", "name", "level"])


def evaluate_citylearn_v2_all_kpis(env) -> Dict[str, Dict[str, float]]:
    """Return all CityLearn KPI values grouped by entity name.

    This keeps all current and future CityLearn v2 KPI names instead of
    restricting evaluation to the thesis summary indicators.
    """

    frame = evaluate_citylearn_v2_kpi_frame(env)

    if frame.empty or not {"name", "cost_function", "value"}.issubset(frame.columns):
        return {}

    output: Dict[str, Dict[str, float]] = {}

    for _, row in frame.iterrows():
        entity = str(row["name"])
        cost_function = str(row["cost_function"])

        try:
            value = float(row["value"])
        except (TypeError, ValueError):
            continue

        output.setdefault(entity, {})[cost_function] = value

    return output
