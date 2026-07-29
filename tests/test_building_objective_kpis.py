"""Building-level OE KPI extraction from evaluate_v2 frames."""

from __future__ import annotations

import pandas as pd

from citylearn.madrl_kpis import extract_citylearn_v2_building_kpis
from citylearn.v3.objectives import building_objective_kpi_rows


def test_extract_building_oe_kpis_maps_evaluate_v2_twins():
    frame = pd.DataFrame(
        [
            {
                "level": "district",
                "name": "District",
                "cost_function": "district_energy_grid_total_import_control_kwh",
                "value": 100.0,
            },
            {
                "level": "building",
                "name": "Building_1",
                "cost_function": "building_energy_grid_total_import_control_kwh",
                "value": 10.0,
            },
            {
                "level": "building",
                "name": "Building_1",
                "cost_function": "building_emissions_total_control_kgco2",
                "value": 5.0,
            },
            {
                "level": "building",
                "name": "Building_2",
                "cost_function": "building_cost_total_control_eur",
                "value": 3.5,
            },
        ]
    )

    per_building = extract_citylearn_v2_building_kpis(frame)
    assert set(per_building) == {"Building_1", "Building_2"}
    assert per_building["Building_1"]["grid_import_control"] == 10.0
    assert per_building["Building_1"]["carbon_emissions_control"] == 5.0
    assert per_building["Building_2"]["electricity_cost_control"] == 3.5

    rows = building_objective_kpi_rows(frame)
    assert {row["axis"] for row in rows} >= {"OE1", "OE2", "OE3"}
    assert any(row["building"] == "Building_1" and row["kpi"] == "grid_import_control" for row in rows)
    assert any(row["building"] == "Building_2" and row["kpi"] == "electricity_cost_control" for row in rows)
