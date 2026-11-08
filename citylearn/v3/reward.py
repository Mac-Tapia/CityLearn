"""CityLearn v3 MADRL thesis reward (moved from citylearn.reward_function, 2026-11-01).

Canonical dotted path: ``citylearn.v3.reward.CityLearnV3MADRLRewardFunction``.
``citylearn.reward_function`` keeps a lazy alias to this same class object so
older dotted paths and checkpoints keep resolving. Base CityLearn rewards
(MARL, SolarPenaltyReward, ...) stay in ``citylearn.reward_function``.

Profile fresh_kpi_aligned_v6_net_billing (owner decision 2026-11-01): PV surplus
is stored in the BESS or exported to the public grid, never wasted, and the
export enters the three axes consistently with configs/evaluation/kpis.yaml:
cost and CO2 on building net consumption (export credited at the purchase price
and displacing thermal generation), flexibility on the bidirectional district
net power (|P| peak, |dP| ramp). The store-or-export decision is left to the
weighted E1/E2/E3 compromise under the team_mean common payoff.
"""
from typing import Any, List, Mapping, Tuple, Union

import numpy as np

from citylearn.data import ZERO_DIVISION_PLACEHOLDER
from citylearn.reward_function import Electric_Vehicles_Reward_Function

CITYLEARN_V3_HIERARCHY_LEVELS = {
    "primary": 0.60,
    "secondary": 0.25,
    "residual": 0.15,
}

# Compensatory hierarchy: the primary objective has a strict majority of the
# normalized axis reward, while the other objectives remain active.  This is a
# weighted-sum preference hierarchy, not lexicographic/pre-emptive optimization.
# The cyclic order is balanced: every axis receives each hierarchy level once
# across E1/E2/E3, so no thesis objective has greater aggregate exposure.
CITYLEARN_V3_AXIS_PRIORITY_ORDER = {
    "E1": ("flex", "cost", "carbon"),
    "E2": ("carbon", "flex", "cost"),
    "E3": ("cost", "carbon", "flex"),
}


def _build_hierarchical_axis_reward_weights() -> Mapping[str, Mapping[str, float]]:
    axes = {"flex", "carbon", "cost"}
    levels = CITYLEARN_V3_HIERARCHY_LEVELS

    if not levels["primary"] > levels["secondary"] > levels["residual"] > 0.0:
        raise ValueError("CityLearn v3 hierarchy must satisfy primary > secondary > residual > 0")
    if levels["primary"] <= levels["secondary"] + levels["residual"]:
        raise ValueError("CityLearn v3 primary hierarchy weight must exceed the non-primary sum")
    if not np.isclose(sum(levels.values()), 1.0, atol=1.0e-12):
        raise ValueError("CityLearn v3 hierarchy levels must sum to 1.0")

    weights = {}
    for scenario, order in CITYLEARN_V3_AXIS_PRIORITY_ORDER.items():
        if len(order) != 3 or set(order) != axes:
            raise ValueError(f"Invalid CityLearn v3 axis priority order for {scenario}: {order}")
        weights[scenario] = {
            order[0]: levels["primary"],
            order[1]: levels["secondary"],
            order[2]: levels["residual"],
        }

    for axis in axes:
        assigned = sorted(weights[scenario][axis] for scenario in weights)
        if not np.allclose(assigned, sorted(levels.values()), atol=1.0e-12):
            raise ValueError(f"CityLearn v3 hierarchy is not cyclically balanced for axis {axis}")

    return weights


CITYLEARN_V3_AXIS_REWARD_WEIGHTS = _build_hierarchical_axis_reward_weights()


def _catalog_digest(catalog_path) -> str:
    """sha256 of a scenario catalog plus the objective policies file it references.

    Same digest as ``uc3m.application.training.fresh_campaign.catalog_sha256``.
    """
    import hashlib
    from pathlib import Path

    import yaml

    catalog_path = Path(catalog_path)
    payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    policies = catalog_path.parent / payload["objective_policies_file"]
    digest = hashlib.sha256()
    for path in (catalog_path, policies):
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def scenario_catalog_axis_weights(scenario: str, catalog_path=None):
    """Axis weights of ``scenario`` read from the project scenario catalog.

    The catalog (``UC3M_SCENARIO_CATALOG`` when ``catalog_path`` is None) is the
    single source of the E1/E2/E3 weights (configs/objective_policies.yaml) or a
    generated sensitivity catalog. Returns ``(weights, source)`` or ``None`` when
    no catalog is configured. A configured but unreadable catalog raises: the
    reward never silently falls back to another table.
    """
    import os
    from pathlib import Path

    path = catalog_path or os.environ.get("UC3M_SCENARIO_CATALOG")
    if not path:
        return None
    from uc3m.application.configuration import load_scenario_catalog

    definition = load_scenario_catalog(Path(path))[str(scenario).upper()]
    weights = definition.objective_policy.as_dict()["weights"]
    return (
        {
            "flex": float(weights["flexibility"]),
            "carbon": float(weights["co2_emissions"]),
            "cost": float(weights["energy_cost"]),
        },
        {
            "kind": "scenario_catalog",
            "path": str(path),
            "sha256": _catalog_digest(path),
            "objective_policy_id": definition.objective_policy.policy_id,
        },
    )


# v6 — fresh_kpi_aligned_v6_net_billing (owner decision 2026-11-01): as v5, but the
#   export enters the three axes (net billing at the purchase price, CO2 displaced
#   by injected PV, bidirectional |P| peak); signed terms are bounded symmetrically.
# v5 — fresh_kpi_aligned_v5 (audit 2026-11-01, authorized by the owner):
# · no internal team mixing: the reward returns local terms and the Dec-POMDP
#   wrapper applies reward_aggregation="team_mean", so every agent receives the
#   same common-payoff team reward (the former 0.70/0.30 mixing had no effect
#   under team_mean and is removed; passing team_reward_ratio is an error);
# · flexibility on the coincident district net power: convex (squared) peak and
#   linear ramp, normalized per building by peak/ramp references;
# · CO2 = building import x carbon intensity, cost = building import x price,
#   linear and without offsets or export credits, as the KPI catalog
#   (configs/evaluation/kpis.yaml) defines them;
# · EV service/violation and BESS cycling are penalties only, outside the
#   weighted three-axis vector; the base EV self-production/self-consumption
#   energy incentives are not used;
# · every normalized term is bounded (term_bound for axes, 1.0 for constraint
#   terms) and its saturation recorded.
# v7 — thesis_indicators_v7 (owner 2026-11-01: «implementar todos los KPIs en el
#   entrenamiento»): one dense term per thesis indicator (tesis Tabla 2.1, ~548-600;
#   cap2 §2.2.8), grouped in its axis with declared subweights that sum 1; terms whose
#   formula or parameter is not defined in the thesis are declared pending_owner with
#   weight 0 (never invented). See CITYLEARN_V3_REWARD_TERMS.
CITYLEARN_V3_REWARD_PROFILE_VERSION = "thesis_indicators_v7"

# term -> (axis, thesis indicator / CityLearn KPI, source, status). Subweights per axis
# live in the profile (flex_subweights, carbon_subweights, cost_subweights).
CITYLEARN_V3_REWARD_TERMS = {
    "daily_peak": ("flex", "peak_average", "tesis Tabla 2.1 D-VD.1 Dim1 (~553); exact daily increment of max|P|", "defined"),
    "ramp": ("flex", "ramping_average", "tesis D-VD.1 Dim2 (~558); cap2 §2.2.8 beta1*(P(t)-P(t-1))^2", "defined"),
    "load_factor": ("flex", "one_minus_load_factor_average", "tesis D-VD.1 Dim3 (~563); exact daily increment of 1-mean|P|/max|P|", "defined"),
    "dso_excess": ("flex", "dso_excess", "cap2.md:345-349 r^(3) beta2 term max(0, P_com - P_lim_DSO)^2; P_lim_DSO has no value in the repository", "pending_owner"),
    "carbon_net": ("carbon", "carbon_emissions_total", "tesis D-VD.2 Dim1 (~568); cap2 §2.2.8 xi*max(0,P)-xi_disp*max(0,-P), xi_disp = xi (net billing)", "defined"),
    "carbon_delta": ("carbon", "carbon_emissions_delta", "tesis D-VD.2 Dim2 (~573); per step (P - P_baseline)*xi, baseline = CityLearn WITHOUT_STORAGE_BUT_WITH_PV", "defined"),
    "carbon_ci_import": ("carbon", "ci_weighted_consumption", "tesis D-VD.2 Dim3 (~578): CI-weighted consumption, import weighted by hourly CI, max(0,P)*xi", "defined"),
    "cost_net": ("cost", "electricity_cost_total", "tesis D-VD.3 Dim1 (~583); cap2 §2.2.8 p_imp*max(0,P)-p_exp*max(0,-P), p_exp = p_imp (net billing)", "defined"),
    "cost_delta": ("cost", "electricity_cost_delta", "tesis D-VD.3 Dim2 (~588); per step (P - P_baseline)*price", "defined"),
    "cost_peak_billing": ("cost", "peak_billing", "tesis D-VD.3 Dim3 (~593, 974); SPEC-016 D-V6-8 / D04: in a block hour p_pot*max(0, P - month peak so far in 18:00-23:00), p_pot from bill_tariff_<b>.csv (Osinergmin, D02); monthly sum = p_pot*block peak; 15-min window = 1 hourly step", "defined"),
    "cost_price_signal": ("cost", "price_signal_deviation", "tesis D-VD.3 Dim3 (~593); CityLearn v3 defines it per episode (objectives.py:360), no per-step form", "pending_owner"),
}
_CITYLEARN_V3_BASE_PROFILE = {
    "axis_weight_multipliers": {"flex": 1.00, "carbon": 1.00, "cost": 1.00},
    "ev_weight": 0.25,
    "reward_scale": 1.00,
    # Within-axis shares of the flexibility term (sum 1).
    # Subweights inside each axis (sum 1, identical for the 4 MADRL and E1/E2/E3); the
    # thesis gives no beta values, so defined terms share the axis equally and
    # pending_owner terms carry 0 until the owner fixes their formula or parameter.
    "flex_subweights": {"daily_peak": 1.0 / 3.0, "ramp": 1.0 / 3.0, "load_factor": 1.0 / 3.0, "dso_excess": 0.0},
    "carbon_subweights": {"carbon_net": 1.0 / 3.0, "carbon_delta": 1.0 / 3.0, "carbon_ci_import": 1.0 / 3.0},
    # Owner 2026-11-07 (D04): the cost axis split in thirds, as flex and carbon. Without a bill tariff in the
    # schema the demand term is not applicable and its third goes back to cost_net and cost_delta (1/2 each).
    "cost_subweights": {"cost_net": 1.0 / 3.0, "cost_delta": 1.0 / 3.0, "cost_peak_billing": 1.0 / 3.0,
                        "cost_price_signal": 0.0},
    "dso_limit_kw": None,
    # Per-building references used to normalize district and building terms.
    "peak_reference_kw": 25.0,
    "ramp_reference_kw": 15.0,
    "import_reference_kwh": 20.0,
    "term_bound": 4.0,
    "bess_cycle_weight": 0.10,
    "bess_cycle_scale": 0.05,
    "ev_violation_reference_kwh": 10.0,
    "ev_soc_tolerance": 0.05,
    "ev_soc_critical_deficit": 0.25,
    "ev_urgency_hours": 8.0,
    "ev_departure_deficit_weight": 0.70,
    "ev_urgency_deficit_weight": 0.30,
    "ev_idle_deficit_weight": 0.25,
}
CITYLEARN_V3_MADRL_REWARD_PROFILES = {
    algorithm: dict(
        _CITYLEARN_V3_BASE_PROFILE,
        axis_weight_multipliers=dict(_CITYLEARN_V3_BASE_PROFILE["axis_weight_multipliers"]),
        profile_name=(
            f"{algorithm.lower()}_{CITYLEARN_V3_REWARD_PROFILE_VERSION}"
            if algorithm != "MADRL"
            else f"generic_citylearn_v3_madrl_{CITYLEARN_V3_REWARD_PROFILE_VERSION}"
        ),
    )
    for algorithm in ("HAPPO", "MASAC", "MATD3", "MAAC", "MADRL")
}
_REMOVED_PROFILE_KEYS = {"team_reward_ratio"}


class CityLearnV3MADRLRewardFunction(Electric_Vehicles_Reward_Function):
    """CityLearn v3 thesis reward for MADRL training.

    The original CityLearn v2 schema keeps the EV-aware MARL reward for
    backward compatibility. This reward is used only by the v3 MADRL adapters:
    a weighted three-axis vector (flexibility, CO2, cost) aligned with the
    confirmatory KPIs plus separate EV/BESS constraint penalties. Axis weights
    come from the scenario catalog (``UC3M_SCENARIO_CATALOG``) unless passed
    explicitly; the embedded table is a legacy fallback only.
    """

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        algorithm: str = None,
        scenario: str = None,
        axis_weights: Mapping[str, float] = None,
        profile: Mapping[str, Any] = None,
        ev_weights: Mapping[str, float] = None,
        price_reference: float = 0.20,
        carbon_reference: float = 0.35,
    ):
        super().__init__(env_metadata, weights=ev_weights)
        self.algorithm = str(algorithm or "MADRL").upper()
        self.scenario = str(scenario or "E1").upper()
        self.price_reference = float(price_reference)
        self.carbon_reference = float(carbon_reference)
        self.profile = self._build_profile(profile)
        self.axis_weights_source = {}
        self.axis_weights = self._build_axis_weights(axis_weights)
        self._last_district_power = None
        self._last_component_breakdown = {}
        self._last_bess_socs: dict = {}
        self._reset_day()

    def _reset_day(self):
        self._day_max = 0.0
        self._day_sum = 0.0
        self._day_steps = 0
        self._day_lf = 0.0
        self._day_step_count = 0
        self._last_hour = None

    def _steps_per_day(self) -> int:
        seconds = self._safe_float((self.env_metadata or {}).get("seconds_per_time_step"), 3600.0) or 3600.0
        return max(1, int(round(86400.0 / seconds)))

    def _new_day(self, observations) -> bool:
        hour = self._safe_float(observations[0].get("hour"), default=np.nan) if observations else np.nan
        if np.isfinite(hour):
            new = self._last_hour is None or hour < self._last_hour
            self._last_hour = hour
            return new
        new = self._day_step_count % self._steps_per_day() == 0
        self._day_step_count += 1
        return new

    @staticmethod
    def _without_term(weights: Mapping[str, float], term: str) -> Mapping[str, float]:
        """Weights of an axis with a not-applicable term removed and the rest rescaled to the same total."""
        total = float(sum(weights.values()))
        kept = {k: float(v) for k, v in weights.items() if k != term}
        rest = float(sum(kept.values()))
        if rest <= ZERO_DIVISION_PLACEHOLDER:
            return {**kept, term: 0.0}
        return {**{k: v * total / rest for k, v in kept.items()}, term: 0.0}

    def reset(self):
        self._reset_day()
        self._last_district_power = None
        self._last_component_breakdown = {}
        self._last_bess_socs = {}

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "function": self.__class__.__name__,
            "algorithm": self.algorithm,
            "scenario": self.scenario,
            "axis_weights": dict(self.axis_weights),
            "axis_weights_source": dict(self.axis_weights_source),
            "weighting_method": "compensatory_hierarchical_weighted_sum",
            "hierarchy_levels": dict(CITYLEARN_V3_HIERARCHY_LEVELS),
            "axis_priority_order": list(CITYLEARN_V3_AXIS_PRIORITY_ORDER.get(self.scenario, ())),
            "profile": dict(self.profile),
            "aggregation": "local_terms; team payoff applied by CityLearnDecPOMDPEnv.reward_aggregation",
            "base_reward": "CityLearnV3MADRLRewardFunction",
            "not_using_marl_base_weights": True,
        }

    def _build_profile(self, profile: Mapping[str, Any] = None) -> Mapping[str, Any]:
        base = dict(
            CITYLEARN_V3_MADRL_REWARD_PROFILES.get(
                self.algorithm,
                CITYLEARN_V3_MADRL_REWARD_PROFILES["MADRL"],
            )
        )
        base["axis_weight_multipliers"] = dict(base.get("axis_weight_multipliers", {}))

        if profile:
            removed = _REMOVED_PROFILE_KEYS & set(profile)
            if removed:
                raise ValueError(
                    f"reward profile keys {sorted(removed)} were removed: the team payoff is applied "
                    "by the Dec-POMDP reward_aggregation, not mixed inside the reward"
                )
            for key, value in profile.items():
                if key == "axis_weight_multipliers":
                    merged = dict(base.get("axis_weight_multipliers", {}))
                    merged.update(dict(value))
                    base[key] = merged
                else:
                    base[key] = value

        return base

    def _build_axis_weights(self, axis_weights: Mapping[str, float] = None) -> Mapping[str, float]:
        if axis_weights:
            raw_weights = dict(axis_weights)
            self.axis_weights_source = {"kind": "explicit_argument"}
        else:
            resolved = scenario_catalog_axis_weights(self.scenario)
            if resolved is not None:
                raw_weights, self.axis_weights_source = resolved
            else:
                raw_weights = dict(
                    CITYLEARN_V3_AXIS_REWARD_WEIGHTS.get(
                        self.scenario,
                        {"flex": 1.0 / 3.0, "carbon": 1.0 / 3.0, "cost": 1.0 / 3.0},
                    )
                )
                self.axis_weights_source = {"kind": "embedded_legacy_table"}
        multipliers = dict(self.profile.get("axis_weight_multipliers", {}))
        weighted = {
            key: max(0.0, float(raw_weights.get(key, 0.0)) * float(multipliers.get(key, 1.0)))
            for key in ("flex", "carbon", "cost")
        }
        total = sum(weighted.values())

        if total <= ZERO_DIVISION_PLACEHOLDER:
            return {"flex": 1.0 / 3.0, "carbon": 1.0 / 3.0, "cost": 1.0 / 3.0}
        if np.isclose(total, 1.0, rtol=0.0, atol=1.0e-12):
            return weighted

        return {key: value / total for key, value in weighted.items()}

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            output = float(value)
        except (TypeError, ValueError):
            return default

        return output if np.isfinite(output) else default

    @staticmethod
    def _bounded(value: float, bound: float) -> Tuple[float, float]:
        """Clip a non-negative normalized term to ``bound``; returns (value, saturated flag)."""
        value = max(float(value), 0.0)
        return (float(bound), 1.0) if value > bound else (value, 0.0)

    @staticmethod
    def _bounded_signed(value: float, bound: float) -> Tuple[float, float]:
        """Clip a signed normalized term to [-bound, bound]; returns (value, saturated flag)."""
        value = float(value)
        if abs(value) > bound:
            return (float(np.copysign(bound, value)), 1.0)
        return (value, 0.0)

    def _ev_term(self, observation: Mapping[str, Union[int, float, dict]]) -> float:
        """EV constraint penalty in [-1, 0] (service deficit plus charging violation)."""
        return self._ev_terms(observation)["ev_raw"]

    def _ev_terms(self, observation: Mapping[str, Union[int, float, dict]]) -> Mapping[str, float]:
        violation = max(0.0, self._safe_float(observation.get("charging_constraint_violation_kwh"), 0.0))
        reference = max(
            self._safe_float(self.profile.get("ev_violation_reference_kwh"), 10.0), ZERO_DIVISION_PLACEHOLDER
        )
        norm_violation = violation * self.charging_constraint_penalty_coefficient / reference
        bounded_violation, saturated = self._bounded(norm_violation, 1.0)
        service = self._ev_service_constraint_term(observation)
        return {
            "raw_ev_violation_kwh": violation,
            "norm_ev_violation": norm_violation,
            "bounded_ev_violation": bounded_violation,
            "sat_ev_violation": saturated,
            "ev_service": service,
            "ev_raw": float(np.clip(service - bounded_violation, -1.0, 0.0)),
        }

    def _ev_service_constraint_term(self, observation: Mapping[str, Union[int, float, dict]]) -> float:
        """Penalize unmet EV service obligations independently of energy KPIs."""

        ev_chargers: Mapping[str, Mapping[str, Any]] = observation.get("electric_vehicles_chargers_dict", {})
        if not ev_chargers:
            return 0.0

        tolerance = max(self._safe_float(self.profile.get("ev_soc_tolerance"), 0.05), ZERO_DIVISION_PLACEHOLDER)
        critical_deficit = max(
            self._safe_float(self.profile.get("ev_soc_critical_deficit"), 0.25),
            tolerance,
        )
        urgency_hours = max(self._safe_float(self.profile.get("ev_urgency_hours"), 4.0), 0.0)
        departure_weight = max(self._safe_float(self.profile.get("ev_departure_deficit_weight"), 0.55), 0.0)
        urgency_weight = max(self._safe_float(self.profile.get("ev_urgency_deficit_weight"), 0.30), 0.0)
        idle_weight = max(self._safe_float(self.profile.get("ev_idle_deficit_weight"), 0.15), 0.0)

        departure_terms = []
        urgency_terms = []
        idle_terms = []

        for data in ev_chargers.values():
            if not data.get("connected", False):
                continue

            soc_now = self._safe_float(data.get("battery_soc"), default=np.nan)
            required_soc = self._safe_float(data.get("required_soc"), default=np.nan)
            if not np.isfinite(soc_now) or not np.isfinite(required_soc):
                continue

            deficit = max(0.0, required_soc - soc_now)
            if deficit <= tolerance:
                continue

            hours_until_departure = max(self._safe_float(data.get("hours_until_departure"), 0.0), 0.0)
            last_charged_kwh = self._safe_float(data.get("last_charged_kwh"), 0.0)
            deficit_ratio = float(np.clip((deficit - tolerance) / critical_deficit, 0.0, 1.0))
            urgency_factor = 0.0

            if hours_until_departure <= urgency_hours:
                urgency_factor = float((urgency_hours - hours_until_departure + 1.0) / (urgency_hours + 1.0))
                urgency_terms.append(deficit_ratio * urgency_factor)

            if hours_until_departure <= ZERO_DIVISION_PLACEHOLDER:
                departure_terms.append(deficit_ratio)

            if urgency_factor > 0.0 and last_charged_kwh <= ZERO_DIVISION_PLACEHOLDER:
                idle_terms.append(deficit_ratio * urgency_factor)

        def mean_or_zero(values: List[float]) -> float:
            return 0.0 if not values else float(np.mean(values))

        penalty = (
            departure_weight * mean_or_zero(departure_terms)
            + urgency_weight * mean_or_zero(urgency_terms)
            + idle_weight * mean_or_zero(idle_terms)
        )
        return -float(np.clip(penalty, 0.0, 1.0))

    def calculate(self, observations: List[Mapping[str, Union[int, float, dict]]]) -> List[float]:
        profile = self.profile
        net_values = [self._safe_float(o.get("net_electricity_consumption")) for o in observations]
        n_buildings = max(len(observations), 1)
        # Coincident district net power: the KPI sums building power before the peak/ramp.
        district_power = float(np.sum(net_values))
        district_ramp = 0.0 if self._last_district_power is None else abs(district_power - self._last_district_power)
        self._last_district_power = district_power

        term_bound = max(float(profile.get("term_bound", 4.0)), ZERO_DIVISION_PLACEHOLDER)
        peak_reference = max(float(profile.get("peak_reference_kw", 25.0)), ZERO_DIVISION_PLACEHOLDER)
        ramp_reference = max(float(profile.get("ramp_reference_kw", 15.0)), ZERO_DIVISION_PLACEHOLDER)
        import_reference = max(float(profile.get("import_reference_kwh", 20.0)), ZERO_DIVISION_PLACEHOLDER)
        ev_weight = float(profile.get("ev_weight", 0.25))
        reward_scale = float(profile.get("reward_scale", 1.0))
        bess_cycle_weight = float(profile.get("bess_cycle_weight", 0.0))
        bess_cycle_scale = max(float(profile.get("bess_cycle_scale", 0.05)), ZERO_DIVISION_PLACEHOLDER)
        flex_w = dict(profile.get("flex_subweights") or {})
        carbon_w = dict(profile.get("carbon_subweights") or {})
        cost_w = dict(profile.get("cost_subweights") or {})

        # Shared flexibility terms on the bidirectional district power |P| (identical for
        # every building). Daily peak and 1-LF are exact increments of the daily KPI value,
        # so their sum over a calendar day equals the KPI of that day.
        if self._new_day(observations):
            self._day_max = 0.0
            self._day_sum = 0.0
            self._day_steps = 0
            self._day_lf = 0.0
        magnitude = abs(district_power)
        day_max = max(self._day_max, magnitude)
        peak_increment = day_max - self._day_max
        self._day_sum += magnitude
        self._day_steps += 1
        day_lf = 0.0 if day_max <= ZERO_DIVISION_PLACEHOLDER else 1.0 - (self._day_sum / self._day_steps) / day_max
        lf_increment = day_lf - self._day_lf
        self._day_max, self._day_lf = day_max, day_lf
        dso_limit = profile.get("dso_limit_kw")
        dso_excess = 0.0 if dso_limit is None else max(0.0, district_power - float(dso_limit))
        flex_terms = {
            "daily_peak": (peak_increment, peak_increment / (n_buildings * peak_reference)),
            "ramp": (district_ramp, (district_ramp / (n_buildings * ramp_reference)) ** 2),
            "load_factor": (lf_increment, lf_increment),
            "dso_excess": (dso_excess, (dso_excess / (n_buildings * peak_reference)) ** 2),
        }
        shared = {}
        flex_component = 0.0
        for name, (raw, norm) in flex_terms.items():
            bounded, saturated = self._bounded_signed(norm, term_bound)
            shared.update({f"raw_{name}": float(raw), f"norm_{name}": float(norm),
                           f"bounded_{name}": bounded, f"sat_{name}": saturated})
            flex_component -= float(flex_w.get(name, 0.0)) * bounded

        rewards = []
        components = []

        for i, (observation, net_value) in enumerate(zip(observations, net_values)):
            import_kwh = max(net_value, 0.0)
            price = max(self._safe_float(observation.get("electricity_pricing")), 0.0)
            carbon = max(self._safe_float(observation.get("carbon_intensity")), 0.0)
            # Per-step CityLearn baseline of evaluate_v2 (WITHOUT_STORAGE_BUT_WITH_PV):
            # net minus storage and EV charger consumption of the same step.
            baseline_net = net_value - sum(
                self._safe_float(observation.get(key), 0.0)
                for key in ("cooling_storage_electricity_consumption", "heating_storage_electricity_consumption",
                            "dhw_storage_electricity_consumption", "electrical_storage_electricity_consumption")
            ) - sum(self._safe_float(d.get("last_charged_kwh"), 0.0)
                    for d in (observation.get("electric_vehicles_chargers_dict") or {}).values())
            carbon_scale = import_reference * self.carbon_reference
            cost_scale = import_reference * self.price_reference
            # SPEC-016 D-V6-8: demand charge of the step, from the bill state of the v3 layer (citylearn.v3.bill).
            demand_rate = observation.get("bill_demand_usd_kw_month")
            bill_applicable = demand_rate is not None
            peak_billing, peak_increment_kw = 0.0, 0.0
            if bill_applicable and self._safe_float(observation.get("bill_in_peak_block")) > 0.5:
                month_peak = self._safe_float(observation.get("bill_month_block_peak_kw"))
                peak_increment_kw = max(0.0, net_value - month_peak)
                peak_billing = self._safe_float(demand_rate) * peak_increment_kw
            step_cost_w = cost_w if bill_applicable else self._without_term(cost_w, "cost_peak_billing")
            # Net billing: export credited at the purchase price and displacing thermal generation.
            axis_terms = {
                "carbon_net": (net_value * carbon, net_value * carbon / carbon_scale, carbon_w),
                "carbon_delta": ((net_value - baseline_net) * carbon,
                                 (net_value - baseline_net) * carbon / carbon_scale, carbon_w),
                "carbon_ci_import": (import_kwh * carbon, import_kwh * carbon / carbon_scale, carbon_w),
                "cost_net": (net_value * price, net_value * price / cost_scale, step_cost_w),
                "cost_delta": ((net_value - baseline_net) * price,
                               (net_value - baseline_net) * price / cost_scale, step_cost_w),
                # P-V6-24: normalized as daily_peak, kW increment / peak_reference_kw (the USD value / the generic
                # cost scale would saturate above about 1.2 kW and make the term binary).
                "cost_peak_billing": (peak_billing, peak_increment_kw / peak_reference, step_cost_w),
                "cost_price_signal": (0.0, 0.0, step_cost_w),
            }
            building_terms = {}
            carbon_component = 0.0
            cost_component = 0.0
            for name, (raw, norm, weights) in axis_terms.items():
                bounded, saturated = self._bounded_signed(norm, term_bound)
                building_terms.update({f"raw_{name}": float(raw), f"norm_{name}": float(norm),
                                       f"bounded_{name}": bounded, f"sat_{name}": saturated})
                if name.startswith("carbon"):
                    carbon_component -= float(weights.get(name, 0.0)) * bounded
                else:
                    cost_component -= float(weights.get(name, 0.0)) * bounded

            ev_terms = self._ev_terms(observation)
            ev_component = ev_weight * ev_terms["ev_raw"]

            # BESS cycling constraint: -w * min(|soc_t - soc_{t-1}| / scale, 1).
            delta_soc = 0.0
            if bess_cycle_weight > ZERO_DIVISION_PLACEHOLDER:
                bess_soc_now = self._safe_float(observation.get("electrical_storage_soc"), default=np.nan)
                bess_soc_prev = self._last_bess_socs.get(i, np.nan)
                if np.isfinite(bess_soc_now) and np.isfinite(bess_soc_prev):
                    delta_soc = abs(bess_soc_now - bess_soc_prev)
                if np.isfinite(bess_soc_now):
                    self._last_bess_socs[i] = bess_soc_now
            norm_bess = delta_soc / bess_cycle_scale
            bounded_bess, sat_bess = self._bounded(norm_bess, 1.0)
            bess_cycle_component = -bess_cycle_weight * bounded_bess

            axis_contributions = {
                "flex": self.axis_weights["flex"] * flex_component,
                "carbon": self.axis_weights["carbon"] * carbon_component,
                "cost": self.axis_weights["cost"] * cost_component,
            }
            axis_absolute_total = float(sum(abs(value) for value in axis_contributions.values()))
            axis_effective_shares = {
                key: (
                    abs(value) / axis_absolute_total
                    if axis_absolute_total > ZERO_DIVISION_PLACEHOLDER
                    else 0.0
                )
                for key, value in axis_contributions.items()
            }

            local_reward = reward_scale * (
                sum(axis_contributions.values())
                + ev_component
                + bess_cycle_component
            )
            rewards.append(float(local_reward))
            components.append({
                "net": net_value,
                "baseline_net": baseline_net,
                "price": price,
                "carbon_intensity": carbon,
                "import_kwh": import_kwh,
                "raw_district_power_kw": district_power,
                **shared,
                **building_terms,
                "raw_ev_violation_kwh": ev_terms["raw_ev_violation_kwh"],
                "raw_bess_delta_soc": delta_soc,
                "cost_peak_billing_applicable": bool(bill_applicable),
                "norm_ev_violation": ev_terms["norm_ev_violation"],
                "norm_bess_cycle": norm_bess,
                "bounded_ev_violation": ev_terms["bounded_ev_violation"],
                "bounded_bess_cycle": bounded_bess,
                "sat_ev_violation": ev_terms["sat_ev_violation"],
                "sat_bess_cycle": sat_bess,
                "ev_service": ev_terms["ev_service"],
                "flex": flex_component,
                "carbon": carbon_component,
                "cost": cost_component,
                "weighted_flex": axis_contributions["flex"],
                "weighted_carbon": axis_contributions["carbon"],
                "weighted_cost": axis_contributions["cost"],
                "effective_flex_share": axis_effective_shares["flex"],
                "effective_carbon_share": axis_effective_shares["carbon"],
                "effective_cost_share": axis_effective_shares["cost"],
                "ev": ev_component,
                "bess_cycle": bess_cycle_component,
                "local_reward": float(local_reward),
            })

        team_reward = float(np.mean(rewards)) if rewards else 0.0
        self._last_component_breakdown = {
            "algorithm": self.algorithm,
            "scenario": self.scenario,
            "axis_weights": dict(self.axis_weights),
            "axis_weights_source": dict(self.axis_weights_source),
            "weighting_method": "compensatory_hierarchical_weighted_sum",
            "effective_share_scope": "absolute_weighted_axis_terms_excluding_ev_and_bess",
            "aggregation": "local_terms; team payoff applied by CityLearnDecPOMDPEnv.reward_aggregation",
            "profile_name": self.profile.get("profile_name"),
            "reward_scale": reward_scale,
            "district": {
                "net_power_kw": district_power,
                "ramp_kw": district_ramp,
                "import_kwh": float(sum(max(v, 0.0) for v in net_values)),
            },
            "district_import": float(sum(max(v, 0.0) for v in net_values)),
            "district_ramp": district_ramp,
            "team_reward": team_reward,
            "components": components,
        }

        if self.central_agent:
            return [float(np.sum(rewards))]

        return [float(value) for value in rewards]
