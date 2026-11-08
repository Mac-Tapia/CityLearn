"""ISO 13790 simple hourly method (5R1C, Annex C) for the CityLearn v3 layer (SPEC-016 D-V6-52, FR-016.4; air conditioning controllable within the comfort limit, D-V6-2).

The building is one thermal zone with three temperature nodes (air, central surface, mass) and five conductances:

- ``h_ve``: ventilation, from the outdoor-air flows of the project norms (EM.030, NTS 110);
- ``h_tr_w``: windows and ``h_tr_em``: opaque envelope to the mass node (ISO 13790 eq. C.3), together the envelope
  conductance ``envelope_w_m2k`` x floor area, the share of windows ``WINDOW_SHARE`` declared as an assumption;
- ``h_tr_is = 3.45 x A_t`` and ``h_tr_ms = 9.1 x A_m`` with ``A_t = 4.5 x A_f`` (ISO 13790 eqs. 9 and 64, 7.2.2.2);
- ``c_m`` and ``A_m`` from the mass class of ISO 13790 Table 12.

Gains (W): internal ``phi_int`` (half to the air node, eq. C.1) and solar ``phi_sol`` = aperture x horizontal
irradiance; ``phi_hc`` is the heating (+) or cooling (-) power delivered to the air node. One step is one hour.
The ideal load that holds a set point is found as in ISO 13790 C.4, with the two evaluations at 0 and 10 W/m2.

The envelope conductance and the solar aperture are not in the inventory; uc3m.application.dataset.thermal_calibration
fits them per building against the bill-anchored cooling (D-V6-18) and validates them with ASHRAE Guideline 14.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

# ISO 13790 Table 12: mass class -> (A_m / A_f, C_m / A_f in J/(m2 K)).
MASS_CLASSES = {
    'very_light': (2.5, 80_000.0),
    'light': (2.5, 110_000.0),
    'medium': (2.5, 165_000.0),
    'heavy': (3.0, 260_000.0),
    'very_heavy': (3.5, 370_000.0),
}
LAMBDA_AT = 4.5          # A_t / A_f, ISO 13790 7.2.2.2
H_IS = 3.45              # W/(m2 K), ISO 13790 eq. 9
H_MS = 9.1               # W/(m2 K), ISO 13790 eq. 64
WINDOW_SHARE = 0.3       # assumption: share of the envelope conductance through windows (no inventory value)
SECONDS_PER_STEP = 3600.0


@dataclass(frozen=True)
class Parameters:
    a_f: float
    a_m: float
    a_t: float
    c_m: float
    h_ve: float
    h_tr_w: float
    h_tr_em: float
    h_tr_is: float
    h_tr_ms: float
    solar_aperture_m2: float


@dataclass(frozen=True)
class ThermalState:
    theta_m: float


@dataclass(frozen=True)
class StepResult:
    theta_air: float
    theta_s: float
    theta_m: float
    state: ThermalState


def iso13790_parameters(*, floor_area_m2: float, envelope_w_m2k: float, solar_aperture_fraction: float,
                        ventilation_w_k: float, mass_class: str = 'medium',
                        heat_capacity_factor: float = 1.0) -> Parameters:
    """``heat_capacity_factor`` scales C_m of the Table 12 class (identified on measured data; 1 = the class)."""
    if floor_area_m2 <= 0:
        raise ValueError(f'floor area must be positive, got {floor_area_m2}')
    if envelope_w_m2k <= 0 or solar_aperture_fraction < 0 or ventilation_w_k < 0:
        raise ValueError('envelope conductance must be positive; solar aperture and ventilation non-negative')
    if heat_capacity_factor <= 0:
        raise ValueError(f'heat capacity factor must be positive, got {heat_capacity_factor}')
    if mass_class not in MASS_CLASSES:
        raise ValueError(f'unknown mass class {mass_class!r}; ISO 13790 Table 12 has {sorted(MASS_CLASSES)}')
    am_ratio, cm_per_m2 = MASS_CLASSES[mass_class]
    a_f = float(floor_area_m2)
    a_m, a_t = am_ratio*a_f, LAMBDA_AT*a_f
    h_tr_ms = H_MS*a_m
    envelope = float(envelope_w_m2k)*a_f
    h_tr_w = WINDOW_SHARE*envelope
    h_tr_op = envelope - h_tr_w
    if h_tr_op >= h_tr_ms:
        raise ValueError(f'opaque conductance {h_tr_op:.0f} W/K must stay below h_tr_ms {h_tr_ms:.0f} W/K (eq. C.3)')
    h_tr_em = 1.0/(1.0/h_tr_op - 1.0/h_tr_ms)
    return Parameters(a_f=a_f, a_m=a_m, a_t=a_t, c_m=cm_per_m2*a_f*float(heat_capacity_factor), h_ve=max(float(ventilation_w_k), 1e-6),
                      h_tr_w=h_tr_w, h_tr_em=h_tr_em, h_tr_is=H_IS*a_t, h_tr_ms=h_tr_ms,
                      solar_aperture_m2=float(solar_aperture_fraction)*a_f)


def step(p: Parameters, state: ThermalState, *, theta_e: float, phi_int_w: float, phi_sol_w: float,
         phi_hc_w: float) -> StepResult:
    """One hour of ISO 13790 Annex C (eqs. C.1 to C.11), supply air at the outdoor temperature."""
    theta_sup = theta_e
    phi_ia = 0.5*phi_int_w
    shared = 0.5*phi_int_w + phi_sol_w
    phi_m = p.a_m/p.a_t*shared
    phi_st = (1.0 - p.a_m/p.a_t - p.h_tr_w/(H_MS*p.a_t))*shared
    h_tr_1 = 1.0/(1.0/p.h_ve + 1.0/p.h_tr_is)
    h_tr_2 = h_tr_1 + p.h_tr_w
    h_tr_3 = 1.0/(1.0/h_tr_2 + 1.0/p.h_tr_ms)
    phi_mtot = phi_m + p.h_tr_em*theta_e + h_tr_3/h_tr_2*(
        phi_st + p.h_tr_w*theta_e + h_tr_1*((phi_ia + phi_hc_w)/p.h_ve + theta_sup))
    cm = p.c_m/SECONDS_PER_STEP
    theta_m_t = ((state.theta_m*(cm - 0.5*(h_tr_3 + p.h_tr_em)) + phi_mtot)
                 / (cm + 0.5*(h_tr_3 + p.h_tr_em)))
    theta_m = 0.5*(theta_m_t + state.theta_m)
    theta_s = ((p.h_tr_ms*theta_m + phi_st + p.h_tr_w*theta_e + h_tr_1*(theta_sup + (phi_ia + phi_hc_w)/p.h_ve))
               / (p.h_tr_ms + p.h_tr_w + h_tr_1))
    theta_air = (p.h_tr_is*theta_s + p.h_ve*theta_sup + phi_ia + phi_hc_w)/(p.h_tr_is + p.h_ve)
    return StepResult(theta_air=theta_air, theta_s=theta_s, theta_m=theta_m, state=ThermalState(theta_m=theta_m_t))


def ideal_load_w(p: Parameters, state: ThermalState, *, theta_e: float, phi_int_w: float, phi_sol_w: float,
                 set_point: float) -> float:
    """Power on the air node that brings it to the set point (ISO 13790 C.4: evaluations at 0 and 10 W/m2).
    Negative is cooling; 0 when the free-floating air is already at or below a cooling set point."""
    kwargs = dict(theta_e=theta_e, phi_int_w=phi_int_w, phi_sol_w=phi_sol_w)
    free = step(p, state, phi_hc_w=0.0, **kwargs).theta_air
    if free <= set_point:
        return 0.0
    probe = 10.0*p.a_f
    heated = step(p, state, phi_hc_w=probe, **kwargs).theta_air
    return probe*(set_point - free)/(heated - free)


# --- Gains, humidity and the ideal cooling series shared by the calibration and the simulation ------------------

AIR_DENSITY_KG_M3 = 1.2          # assumption, as configs/dataset/hvac_norms.yaml air_density_kg_m3
AIR_CP_J_KG_K = 1006.0           # dry air, ASHRAE Fundamentals ch. 1 (the 1.006 kJ/kg K of the enthalpy eq. 32)
H_FG_J_KG = 2_501_000.0          # latent heat at 0 C of the same enthalpy equation
OCCUPANT_SENSIBLE_W = 75.0       # assumption: ASHRAE Fundamentals ch. 18 Table 1 typical, split of hvac_norms' 130 W
OCCUPANT_LATENT_W = 55.0


@dataclass(frozen=True)
class ThermalInputs:
    """Hourly drivers of one building: outdoor air, horizontal irradiance, electric internal gains and occupants."""
    theta_e: np.ndarray
    rh_e: np.ndarray
    irradiance_w_m2: np.ndarray
    electric_gain_w: np.ndarray
    occupants: np.ndarray
    pressure_kpa: float


def humidity_ratio(t_c: float, rh_pct: float, pressure_kpa: float) -> float:
    """kg water / kg dry air, Magnus saturation pressure as uc3m hvac_capacity (Alduchov and Eskridge 1996)."""
    vapour = float(rh_pct)/100.0*0.61094*math.exp(17.625*float(t_c)/(float(t_c) + 243.04))
    return 0.621945*vapour/(float(pressure_kpa) - vapour)


def ventilation_conductance_w_k(ventilation_m3_s: float) -> float:
    return AIR_DENSITY_KG_M3*AIR_CP_J_KG_K*float(ventilation_m3_s)


def gains_w(p: Parameters, inputs: ThermalInputs, h: int):
    """(internal, solar) gains of hour h: all electricity of the base load becomes heat, plus the occupants."""
    phi_int = float(inputs.electric_gain_w[h]) + float(inputs.occupants[h])*OCCUPANT_SENSIBLE_W
    return phi_int, p.solar_aperture_m2*float(inputs.irradiance_w_m2[h])


def latent_scalar_w(theta_e: float, rh_e: float, occupants: float, *, pressure_kpa: float, ventilation_m3_s: float,
                    w_in: float) -> float:
    """Latent load of an occupied hour: outdoor air dried to the indoor humidity, plus the occupants."""
    if float(occupants) <= 0.0:
        return 0.0
    w_out = humidity_ratio(theta_e, rh_e, pressure_kpa)
    return (AIR_DENSITY_KG_M3*float(ventilation_m3_s)*H_FG_J_KG*max(0.0, w_out - w_in)
            + float(occupants)*OCCUPANT_LATENT_W)


def latent_w(inputs: ThermalInputs, h: int, *, ventilation_m3_s: float, w_in: float) -> float:
    return latent_scalar_w(float(inputs.theta_e[h]), float(inputs.rh_e[h]), float(inputs.occupants[h]),
                           pressure_kpa=inputs.pressure_kpa, ventilation_m3_s=ventilation_m3_s, w_in=w_in)


class Network:
    """Coefficients of eqs. C.1 to C.11 that do not change between hours; the step is affine in phi_hc."""

    def __init__(self, p: Parameters):
        self.p = p
        self.k_m = p.a_m/p.a_t
        self.k_st = 1.0 - p.a_m/p.a_t - p.h_tr_w/(H_MS*p.a_t)
        self.h1 = 1.0/(1.0/p.h_ve + 1.0/p.h_tr_is)
        self.h2 = self.h1 + p.h_tr_w
        self.h3 = 1.0/(1.0/self.h2 + 1.0/p.h_tr_ms)
        self.cm = p.c_m/SECONDS_PER_STEP
        self.s = 0.5*(self.h3 + p.h_tr_em)

    def air(self, theta_m_prev, theta_e, phi_int, phi_sol, phi_hc):
        """(air temperature, next mass temperature theta_m,t) of one hour."""
        p = self.p
        phi_ia = 0.5*phi_int
        shared = 0.5*phi_int + phi_sol
        phi_st = self.k_st*shared
        inj = (phi_ia + phi_hc)/p.h_ve + theta_e
        phi_mtot = self.k_m*shared + p.h_tr_em*theta_e + self.h3/self.h2*(phi_st + p.h_tr_w*theta_e + self.h1*inj)
        theta_m_t = (theta_m_prev*(self.cm - self.s) + phi_mtot)/(self.cm + self.s)
        theta_m = 0.5*(theta_m_t + theta_m_prev)
        theta_s = (p.h_tr_ms*theta_m + phi_st + p.h_tr_w*theta_e + self.h1*inj)/(p.h_tr_ms + p.h_tr_w + self.h1)
        theta_air = (p.h_tr_is*theta_s + p.h_ve*theta_e + phi_ia + phi_hc)/(p.h_tr_is + p.h_ve)
        return theta_air, theta_m_t


NO_COOLING_W = 1.0   # D-V6-61: below this the sensible cooling is the rounding of the CSV and of the float32 series


def open_windows(p: Parameters, *, ventilation_m3_s: float, natural_ventilation_m3_s: float) -> Optional['Network']:
    """D-V6-61: the same zone with the extra outdoor air of natural ventilation; None without it."""
    if not natural_ventilation_m3_s > 0.0:
        return None
    return Network(replace(p, h_ve=ventilation_conductance_w_k(ventilation_m3_s + natural_ventilation_m3_s)))


def _no_cooling(net: 'Network', opened: Optional['Network'], theta_m, theta_e, phi_int, phi_sol):
    """Air and mass of an hour without cooling. People open the windows only if that cools the air (D-V6-61)."""
    air, mass = net.air(theta_m, theta_e, phi_int, phi_sol, 0.0)
    if opened is not None and air > theta_e:
        return opened.air(theta_m, theta_e, phi_int, phi_sol, 0.0)
    return air, mass


def ideal_cooling_series(p: Parameters, inputs: ThermalInputs, *, set_point: float, ventilation_m3_s: float,
                         w_in: float, theta_m0: float = None, natural_ventilation_m3_s: float = 0.0) -> dict:
    """Ideal sensible cooling (W, positive) that holds the set point in occupied hours, free-floating otherwise,
    the latent load (W) and the air temperature: ideal_load_w + step hour by hour, without the object overhead.
    ``set_point`` is one value or one per hour (a measured indoor temperature in the CU-BEMS validation)."""
    net = Network(p)
    opened = open_windows(p, ventilation_m3_s=ventilation_m3_s, natural_ventilation_m3_s=natural_ventilation_m3_s)
    n = len(inputs.theta_e)
    points = np.broadcast_to(np.asarray(set_point, dtype=float), (n,))
    sensible, latent, theta_air = np.zeros(n), np.zeros(n), np.zeros(n)
    theta_m = float(points[0] if theta_m0 is None else theta_m0)
    probe = 10.0*p.a_f
    for h in range(n):
        theta_e = float(inputs.theta_e[h])
        phi_int, phi_sol = gains_w(p, inputs, h)
        free, free_m = net.air(theta_m, theta_e, phi_int, phi_sol, 0.0)
        load = 0.0
        if float(inputs.occupants[h]) > 0.0 and free > points[h]:
            heated, _ = net.air(theta_m, theta_e, phi_int, phi_sol, probe)
            load = probe*(points[h] - free)/(heated - free)
            theta_air[h], theta_m = net.air(theta_m, theta_e, phi_int, phi_sol, load)
        else:
            theta_air[h], theta_m = _no_cooling(net, opened, theta_m, theta_e, phi_int, phi_sol)
        sensible[h] = -load
        latent[h] = latent_w(inputs, h, ventilation_m3_s=ventilation_m3_s, w_in=w_in)
    return {'sensible_w': sensible, 'latent_w': latent, 'theta_air': theta_air}



def simulate_delivered(p: Parameters, inputs: ThermalInputs, delivered_kwh, *, ventilation_m3_s: float, w_in: float,
                       theta_m0: float, natural_ventilation_m3_s: float = 0.0) -> np.ndarray:
    """Air temperature under a delivered cooling series (thermal kWh per hour): the latent load is covered first and
    the rest is sensible, exactly as ISO13790Building.update_indoor_dry_bulb_temperature does at run time."""
    net = Network(p)
    opened = open_windows(p, ventilation_m3_s=ventilation_m3_s, natural_ventilation_m3_s=natural_ventilation_m3_s)
    theta_m = float(theta_m0)
    out = np.zeros(len(inputs.theta_e))
    for h in range(len(out)):
        phi_int, phi_sol = gains_w(p, inputs, h)
        sensible = max(0.0, float(delivered_kwh[h])*1000.0 - latent_w(inputs, h, ventilation_m3_s=ventilation_m3_s,
                                                                         w_in=w_in))
        if sensible <= NO_COOLING_W:
            out[h], theta_m = _no_cooling(net, opened, theta_m, float(inputs.theta_e[h]), phi_int, phi_sol)
        else:
            out[h], theta_m = net.air(theta_m, float(inputs.theta_e[h]), phi_int, phi_sol, -sensible)
    return out

# --- CityLearn building with the 5R1C dynamics (the air conditioning is an action) ------------------------------

from citylearn.building import DynamicsBuilding  # noqa: E402
from citylearn.dynamics import Dynamics  # noqa: E402

DEFAULT_INDOOR_RH_PCT = 50.0     # NTS 110/119 Table 6 midpoint of 45-60 % (D-V6-17), as hvac_norms indoor_design


class ISO13790Dynamics(Dynamics):
    """Parameters and mass-node state of one building (schema ``dynamics.attributes``)."""

    def __init__(self, *, floor_area_m2: float, envelope_w_m2k: float, solar_aperture_fraction: float,
                 ventilation_m3_s: float, mass_class: str = 'medium', set_point: float = 24.0,
                 pressure_kpa: float = 101.325, indoor_rh_pct: float = DEFAULT_INDOOR_RH_PCT,
                 electric_gain_fraction: float = 1.0, heat_capacity_factor: float = 1.0,
                 natural_ventilation_m3_s: float = 0.0, **_: object):
        super().__init__()
        self.params = iso13790_parameters(floor_area_m2=floor_area_m2, envelope_w_m2k=envelope_w_m2k,
                                          solar_aperture_fraction=solar_aperture_fraction,
                                          ventilation_w_k=ventilation_conductance_w_k(ventilation_m3_s),
                                          mass_class=mass_class, heat_capacity_factor=heat_capacity_factor)
        self.network = Network(self.params)
        self.ventilation_m3_s = float(ventilation_m3_s)
        self.natural_ventilation_m3_s = float(natural_ventilation_m3_s)
        self.opened_network = open_windows(self.params, ventilation_m3_s=self.ventilation_m3_s,
                                           natural_ventilation_m3_s=self.natural_ventilation_m3_s)
        self.set_point = float(set_point)
        self.pressure_kpa = float(pressure_kpa)
        self.w_in = humidity_ratio(self.set_point, indoor_rh_pct, self.pressure_kpa)
        self.electric_gain_fraction = float(electric_gain_fraction)
        self.reset()

    def reset(self):
        self.theta_m = self.set_point


class ISO13790Building(DynamicsBuilding):
    """Building whose indoor temperature follows ISO 13790 5R1C; ``cooling_device`` action = fraction of the
    device's nominal electric power. Delivered cooling covers the latent load first, the rest is sensible."""

    def __init__(self, *args, dynamics: ISO13790Dynamics, **kwargs):
        super().__init__(*args, dynamics=dynamics, **kwargs)
        self.dynamics: ISO13790Dynamics

    def _update_dynamics_input(self):
        """The 5R1C model reads its drivers directly at the current time step."""

    def update_cooling_demand(self, action: float):
        if ('cooling_device' in self.active_actions or 'cooling_or_heating_device' in self.active_actions) \
                and self.simulate_dynamics:
            if self.energy_simulation.hvac_mode[self.time_step] in [1, 3]:
                electric_power = max(0.0, float(action))*self.cooling_device.nominal_power\
                    *self.algorithm_action_based_time_step_hours_ratio
                demand = self.cooling_device.get_max_output_power(
                    self.weather.outdoor_dry_bulb_temperature[self.time_step], heating=False,
                    max_electric_power=electric_power)
            else:
                demand = 0.0
            stored = np.float32(demand)              # the series is float32: never round above the device output
            if float(stored) > demand:
                stored = np.nextafter(stored, np.float32(0.0))
            self.energy_simulation.cooling_demand[self.time_step] = stored

    def update_heating_demand(self, action: float):
        """Iquitos buildings have no heating (SPEC-016); the heating demand stays at its data value."""

    def update_indoor_dry_bulb_temperature(self):
        d, t = self.dynamics, self.time_step
        weather, sim = self.weather, self.energy_simulation
        theta_e = float(weather.outdoor_dry_bulb_temperature[t])
        occupants = float(sim.occupant_count[t]) if sim.occupant_count is not None else 0.0
        phi_int = float(sim.non_shiftable_load[t])*1000.0*d.electric_gain_fraction + occupants*OCCUPANT_SENSIBLE_W
        phi_sol = d.params.solar_aperture_m2*float(weather.diffuse_solar_irradiance[t]
                                                    + weather.direct_solar_irradiance[t])
        latent = latent_scalar_w(theta_e, float(weather.outdoor_relative_humidity[t]), occupants,
                                 pressure_kpa=d.pressure_kpa, ventilation_m3_s=d.ventilation_m3_s, w_in=d.w_in)
        sensible = max(0.0, float(sim.cooling_demand[t])*1000.0 - latent)
        if sensible <= NO_COOLING_W:
            theta_air, d.theta_m = _no_cooling(d.network, d.opened_network, d.theta_m, theta_e, phi_int, phi_sol)
        else:
            theta_air, d.theta_m = d.network.air(d.theta_m, theta_e, phi_int, phi_sol, -sensible)
        sim.indoor_dry_bulb_temperature[t] = theta_air


def as_billed_cooling_action(building) -> float:
    """Cooling action that delivers the data's uncontrolled cooling at the building's current step: the fraction
    of the nominal electric power whose output equals ``cooling_demand_without_control`` (no-control reference,
    baselines and the dataset gates). 0 for a building without a controllable cooling device."""
    t = building.time_step
    demand = float(building.energy_simulation.cooling_demand_without_control[t])
    device = building.cooling_device
    full = float(device.get_max_output_power(building.weather.outdoor_dry_bulb_temperature[t], heating=False,
                                             max_electric_power=device.nominal_power))
    return 0.0 if full <= 0.0 else min(1.0, max(0.0, demand/full))
