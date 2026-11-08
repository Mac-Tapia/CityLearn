"""Calibration of the ISO 13790 5R1C model per building and its ASHRAE Guideline 14 validation (SPEC-016 D-V6-52).

Two parameters the inventory does not give are fitted on the training hours: the envelope conductance per floor
area (W/m2K) and the effective solar aperture (fraction of the floor area). The target is the bill-anchored hourly
cooling of the dataset (D-V6-18, thermal kWh). The fit is validated out of sample with the ASHRAE Guideline 14
criteria for calibrated simulation (monthly NMBE +-5 %, CV(RMSE) <= 15 %; hourly NMBE +-10 %, CV(RMSE) <= 30 %).
There is no measured indoor temperature in the project: the validation is against the bill-anchored cooling.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from citylearn.v3 import thermal as th

CRITERIA = {'monthly': {'nmbe_pct': 5.0, 'cvrmse_pct': 15.0},
            'hourly': {'nmbe_pct': 10.0, 'cvrmse_pct': 30.0}}
BOUNDS = {'envelope_w_m2k': (0.05, 25.0), 'solar_aperture_fraction': (0.0, 0.5)}
START = (1.5, 0.05)
FIELDS = ('theta_e', 'rh_e', 'irradiance_w_m2', 'electric_gain_w', 'occupants')


def ashrae_g14(measured, simulated, *, n_params: int = 1) -> dict:
    """NMBE and CV(RMSE) in % of the measured mean, with n - p degrees of freedom (ASHRAE Guideline 14)."""
    m, s = np.asarray(measured, dtype=float), np.asarray(simulated, dtype=float)
    dof = len(m) - n_params
    mean = float(m.mean()) if len(m) else 0.0
    if dof <= 0 or mean <= 0:
        return {'nmbe_pct': float('inf'), 'cvrmse_pct': float('inf'), 'n': int(len(m))}
    return {'nmbe_pct': float(100.0*(m - s).sum()/(dof*mean)),
            'cvrmse_pct': float(100.0*np.sqrt(((m - s)**2).sum()/dof)/mean), 'n': int(len(m))}


def simulate_cooling_kwh(*, inputs: th.ThermalInputs, envelope_w_m2k: float, solar_aperture_fraction: float,
                         floor_area_m2: float, ventilation_m3_s: float, mass_class: str, set_point: float,
                         w_in: float, heat_capacity_factor: float = 1.0) -> np.ndarray:
    """Thermal kWh per hour (sensible + latent) of the ideal load at the set point in occupied hours."""
    p = th.iso13790_parameters(floor_area_m2=floor_area_m2, envelope_w_m2k=envelope_w_m2k,
                               solar_aperture_fraction=solar_aperture_fraction,
                               ventilation_w_k=th.ventilation_conductance_w_k(ventilation_m3_s), mass_class=mass_class,
                               heat_capacity_factor=heat_capacity_factor)
    series = th.ideal_cooling_series(p, inputs, set_point=set_point, ventilation_m3_s=ventilation_m3_s, w_in=w_in)
    return (series['sensible_w'] + series['latent_w'])/1000.0


def criteria(target, simulated, month, mask) -> dict:
    target, simulated, month, mask = (np.asarray(a) for a in (target, simulated, month, mask))
    hourly = ashrae_g14(target[mask], simulated[mask])
    labels = np.unique(month[mask])
    monthly = ashrae_g14([target[mask & (month == m)].sum() for m in labels],
                         [simulated[mask & (month == m)].sum() for m in labels])
    passed = all(abs(stats['nmbe_pct']) <= CRITERIA[k]['nmbe_pct'] and stats['cvrmse_pct'] <= CRITERIA[k]['cvrmse_pct']
                 for k, stats in (('hourly', hourly), ('monthly', monthly)))
    return {'hourly': hourly, 'monthly': monthly, 'passed': bool(passed)}


@dataclass(frozen=True)
class CalibrationResult:
    envelope_w_m2k: float
    solar_aperture_fraction: float
    validation: dict = field(default_factory=dict)
    simulated_kwh: np.ndarray = None


def calibrate(*, target_kwh, inputs: th.ThermalInputs, month, train, floor_area_m2: float, ventilation_m3_s: float,
              mass_class: str, set_point: float, w_in: float) -> CalibrationResult:
    """Least squares on the training hours; the validation is on the out-of-sample hours (``~train``)."""
    from scipy.optimize import least_squares

    target = np.asarray(target_kwh, dtype=float)
    month, train = np.asarray(month), np.asarray(train, dtype=bool)
    scale = max(float(target[train].mean()), 1e-9)
    kwargs = dict(floor_area_m2=floor_area_m2, ventilation_m3_s=ventilation_m3_s, mass_class=mass_class,
                  set_point=set_point, w_in=w_in)
    last = int(train.nonzero()[0].max()) + 1            # simulate only up to the end of the training hours
    cut = th.ThermalInputs(*(np.asarray(getattr(inputs, f))[:last] for f in FIELDS), pressure_kpa=inputs.pressure_kpa)

    def residuals(x):
        sim = simulate_cooling_kwh(inputs=cut, envelope_w_m2k=x[0], solar_aperture_fraction=x[1], **kwargs)
        return (sim - target[:last])[train[:last]]/scale

    low, high = zip(*(BOUNDS[k] for k in ('envelope_w_m2k', 'solar_aperture_fraction')))
    fit = least_squares(residuals, x0=np.array(START), bounds=(np.array(low), np.array(high)), x_scale='jac')
    envelope, aperture = (float(v) for v in fit.x)
    simulated = simulate_cooling_kwh(inputs=inputs, envelope_w_m2k=envelope, solar_aperture_fraction=aperture,
                                     **kwargs)
    validation = {'train': criteria(target, simulated, month, train),
                  'test': criteria(target, simulated, month, ~train) if (~train).any() else None,
                  'criteria': CRITERIA}
    validation['passed'] = bool(validation['test'] is not None and validation['test']['passed'])
    return CalibrationResult(envelope_w_m2k=envelope, solar_aperture_fraction=aperture, validation=validation,
                             simulated_kwh=simulated)


FRACTION_BOUNDS = (0.005, 1.0)
FRACTION_START = 0.3


@dataclass(frozen=True)
class ZoneCalibration:
    """D-V6-53: the conditioned zone of a building, as a fraction of its roofed area, with its 5R1C parameters."""
    envelope_w_m2k: float
    solar_aperture_fraction: float
    conditioned_fraction: float
    statistics: dict = field(default_factory=dict)


def zone_inputs(inputs: th.ThermalInputs, fraction: float, last: int = None) -> th.ThermalInputs:
    """Drivers of the conditioned zone: its share of the electric gains and of the occupants."""
    cut = slice(None, last)
    return th.ThermalInputs(theta_e=np.asarray(inputs.theta_e)[cut], rh_e=np.asarray(inputs.rh_e)[cut],
                            irradiance_w_m2=np.asarray(inputs.irradiance_w_m2)[cut],
                            electric_gain_w=np.asarray(inputs.electric_gain_w)[cut]*fraction,
                            occupants=np.asarray(inputs.occupants)[cut]*fraction, pressure_kpa=inputs.pressure_kpa)


def calibrate_conditioned_zone(*, target_kwh, inputs: th.ThermalInputs, month, train, floor_area_m2: float,
                               ventilation_m3_s: float, mass_class: str, set_point: float,
                               w_in: float, heat_capacity_factor: float = 1.0) -> ZoneCalibration:
    """Fit envelope, solar aperture and conditioned fraction on the training hours (D-V6-53). The ASHRAE
    Guideline 14 statistics against the target are reported, not used as a gate: the target is a model."""
    from scipy.optimize import least_squares

    target = np.asarray(target_kwh, dtype=float)
    month, train = np.asarray(month), np.asarray(train, dtype=bool)
    scale = max(float(target[train].mean()), 1e-9)
    last = int(train.nonzero()[0].max()) + 1

    def simulate(x, upto=None):
        envelope, aperture, fraction = x
        return simulate_cooling_kwh(inputs=zone_inputs(inputs, fraction, upto), envelope_w_m2k=envelope,
                                    solar_aperture_fraction=aperture, floor_area_m2=floor_area_m2*fraction,
                                    ventilation_m3_s=ventilation_m3_s*fraction, mass_class=mass_class,
                                    set_point=set_point, w_in=w_in, heat_capacity_factor=heat_capacity_factor)

    low = np.array([BOUNDS['envelope_w_m2k'][0], BOUNDS['solar_aperture_fraction'][0], FRACTION_BOUNDS[0]])
    high = np.array([BOUNDS['envelope_w_m2k'][1], BOUNDS['solar_aperture_fraction'][1], FRACTION_BOUNDS[1]])
    fit = least_squares(lambda x: (simulate(x, last) - target[:last])[train[:last]]/scale,
                        x0=np.array([*START, FRACTION_START]), bounds=(low, high), x_scale='jac')
    simulated = simulate(fit.x)
    statistics = {'train': criteria(target, simulated, month, train),
                  'test': criteria(target, simulated, month, ~train) if (~train).any() else None,
                  'criteria': CRITERIA, 'gate': False,
                  'at_bound': [name for name, v, lo, hi in zip(('envelope_w_m2k', 'solar_aperture_fraction',
                                                                 'conditioned_fraction'), fit.x, low, high)
                               if np.isclose(v, lo) or np.isclose(v, hi)]}
    envelope, aperture, fraction = (float(v) for v in fit.x)
    return ZoneCalibration(envelope_w_m2k=envelope, solar_aperture_fraction=aperture, conditioned_fraction=fraction,
                           statistics=statistics)
