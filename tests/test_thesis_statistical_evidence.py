import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_thesis_objective_evidence import (  # noqa: E402
    ALGORITHM_NAMES,
    algorithm_kpi_score_rows,
    cliffs_delta,
    pairwise_statistical_rows,
    statistical_omnibus_rows,
    vargha_delaney_a12,
)


def _objective_row(algorithm, kpi, value, baseline=100.0, axis="OE3", scenario="E3"):
    return {
        "axis": axis,
        "scenario": scenario,
        "algorithm": algorithm,
        "kpi": kpi,
        "value": value,
        "baseline": baseline,
        "available": True,
        "lower_is_better": True,
        "improved_vs_baseline": value < baseline,
        "source": "test",
    }


def test_statistical_scores_cover_all_madrl_algorithms_not_happo_only():
    rows = []
    values = {
        "HAPPO": [90.0, 95.0, 92.0],
        "MASAC": [110.0, 105.0, 102.0],
        "MATD3": [130.0, 120.0, 115.0],
        "MAAC": [80.0, 70.0, 60.0],
    }
    for algorithm, kpi_values in values.items():
        for index, value in enumerate(kpi_values):
            rows.append(_objective_row(algorithm, f"kpi_{index}", value))

    score_rows = algorithm_kpi_score_rows(rows)
    assert {row["algorithm"] for row in score_rows} == set(ALGORITHM_NAMES)

    omnibus = statistical_omnibus_rows(score_rows)
    assert {row["scope"] for row in omnibus} == {"OE3", "ALL"}
    oe3 = next(row for row in omnibus if row["scope"] == "OE3")
    assert oe3["n_algorithms"] == 4
    assert oe3["best_algorithm_by_median_gain"] == "MAAC"
    assert oe3["kruskal_h_statistic"] is not None
    assert oe3["brown_forsythe_w_statistic"] is not None


def test_pairwise_effect_sizes_and_bootstrap_ci_are_oriented_to_algorithm_a():
    score_rows = algorithm_kpi_score_rows([
        _objective_row("MAAC", "kpi_1", 80.0),
        _objective_row("MAAC", "kpi_2", 70.0),
        _objective_row("MAAC", "kpi_3", 60.0),
        _objective_row("HAPPO", "kpi_1", 90.0),
        _objective_row("HAPPO", "kpi_2", 95.0),
        _objective_row("HAPPO", "kpi_3", 92.0),
    ])

    pairwise = pairwise_statistical_rows(score_rows, bootstrap_iterations=200)
    comparison = next(
        row for row in pairwise
        if row["scope"] == "OE3"
        and row["algorithm_a"] == "HAPPO"
        and row["algorithm_b"] == "MAAC"
    )

    assert comparison["better_by_median"] == "MAAC"
    assert comparison["cliffs_delta"] < 0.0
    assert comparison["vargha_delaney_a12"] < 0.5
    assert comparison["hedges_g"] < 0.0
    assert comparison["bootstrap_mean_diff_ci_low"] <= comparison["bootstrap_mean_diff_ci_high"]


def test_effect_size_helpers_match_expected_limits():
    assert cliffs_delta([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert vargha_delaney_a12([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert cliffs_delta([1.0, 2.0], [3.0, 4.0]) == -1.0
    assert vargha_delaney_a12([1.0, 2.0], [3.0, 4.0]) == 0.0
