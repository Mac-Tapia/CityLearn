import numpy as np
import pytest

from citylearn.reward_function import (
    CITYLEARN_V3_AXIS_PRIORITY_ORDER,
    CITYLEARN_V3_AXIS_REWARD_WEIGHTS,
    CITYLEARN_V3_HIERARCHY_LEVELS,
    CityLearnV3MADRLRewardFunction,
)


EXPECTED_WEIGHTS = {
    "E1": {"flex": 0.60, "carbon": 0.15, "cost": 0.25},
    "E2": {"flex": 0.25, "carbon": 0.60, "cost": 0.15},
    "E3": {"flex": 0.15, "carbon": 0.25, "cost": 0.60},
}


def test_hierarchical_weighted_sum_is_strong_and_cyclically_balanced():
    assert CITYLEARN_V3_AXIS_REWARD_WEIGHTS == EXPECTED_WEIGHTS
    assert CITYLEARN_V3_HIERARCHY_LEVELS == {
        "primary": 0.60,
        "secondary": 0.25,
        "residual": 0.15,
    }

    levels = sorted(CITYLEARN_V3_HIERARCHY_LEVELS.values())
    for scenario, weights in EXPECTED_WEIGHTS.items():
        assert sum(weights.values()) == pytest.approx(1.0)
        order = tuple(sorted(weights, key=weights.get, reverse=True))
        assert order == CITYLEARN_V3_AXIS_PRIORITY_ORDER[scenario]
        assert weights[order[0]] > weights[order[1]] + weights[order[2]]

    for axis in ("flex", "carbon", "cost"):
        assert sorted(weights[axis] for weights in EXPECTED_WEIGHTS.values()) == levels


@pytest.mark.parametrize("algorithm", ("HAPPO", "MASAC", "MATD3", "MAAC"))
@pytest.mark.parametrize("scenario", ("E1", "E2", "E3"))
def test_reward_metadata_exposes_compensatory_hierarchy(algorithm, scenario):
    reward = CityLearnV3MADRLRewardFunction(
        {"central_agent": False},
        algorithm=algorithm,
        scenario=scenario,
    )

    assert reward.metadata["axis_weights"] == EXPECTED_WEIGHTS[scenario]
    assert reward.metadata["weighting_method"] == "compensatory_hierarchical_weighted_sum"
    assert reward.metadata["axis_priority_order"] == list(CITYLEARN_V3_AXIS_PRIORITY_ORDER[scenario])
    assert reward.metadata["profile"]["profile_name"] == f"{algorithm.lower()}_unified_comparable_v4"


def test_effective_axis_contributions_are_auditable_and_exclude_service_terms():
    reward = CityLearnV3MADRLRewardFunction(
        {"central_agent": False},
        algorithm="HAPPO",
        scenario="E1",
    )
    observation = {
        "net_electricity_consumption": 10.0,
        "electricity_pricing": 0.20,
        "carbon_intensity": 0.35,
        "electrical_storage_soc": 0.50,
    }

    value = reward.calculate([observation])[0]
    breakdown = reward._last_component_breakdown
    component = breakdown["components"][0]

    weighted_axis_sum = sum(
        component[key]
        for key in ("weighted_flex", "weighted_carbon", "weighted_cost")
    )
    effective_share_sum = sum(
        component[key]
        for key in ("effective_flex_share", "effective_carbon_share", "effective_cost_share")
    )

    assert value == pytest.approx(weighted_axis_sum + component["ev"] + component["bess_cycle"])
    assert effective_share_sum == pytest.approx(1.0)
    assert np.isfinite(list(component.values())).all()
    assert breakdown["effective_share_scope"] == "absolute_weighted_axis_terms_excluding_ev_and_bess"
