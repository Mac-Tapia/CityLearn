"""Bill state of each building for the v3 reward and observations (SPEC-016 D-V6-8, FR-016.7 and FR-016.10).

The dataset ships one ``bill_tariff_Building_<b>.csv`` per building (columns ``energy_usd_kwh``,
``demand_usd_kw_month``, ``peak_block``), referenced from the schema as ``buildings.<key>.bill_tariff`` and
aligned row by row with the simulation time step. The hourly bill is

    cost(t) = P(t) x energy rate + demand rate x max(0, P(t) - peak of the month so far in the block)

(``uc3m.application.dataset.bill_tariff.incremental_bill_cost`` is the oracle). This module keeps the peak of the
month in the 18:00-23:00 block for every building, in one place: the reward reads it to price the demand charge
of the step and the Dec-POMDP appends it to each agent's observation, so the reward stays Markovian in what the
agent sees.

Rules:
- the peak resets to 0 at the first step of each month and at the start of each episode (a non-consecutive
  time step); an episode that starts mid-month therefore starts without the earlier peak of that month (declared);
- a schema where only some buildings declare ``bill_tariff`` is an error; with none, the tracker is absent and the
  demand term is not applicable (the reward keeps its other cost terms).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

COLUMNS = ('energy_usd_kwh', 'demand_usd_kw_month', 'peak_block')
FIELDS = ('bill_month_block_peak_kw', 'bill_in_peak_block', 'bill_demand_usd_kw_month')


def _base(env):
    inner = env
    while not hasattr(inner, 'buildings') and hasattr(inner, 'env'):
        inner = inner.env
    return inner


def _read(path: Path) -> Dict[str, np.ndarray]:
    with open(path, newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        missing = set(COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f'{path.name}: bill tariff columns {sorted(missing)} are missing')
        rows = list(reader)
    table = {c: np.array([float(r[c]) for r in rows], dtype=float) for c in COLUMNS}
    if not np.all(np.isfinite(table['energy_usd_kwh'])) or not np.all(np.isfinite(table['demand_usd_kw_month'])):
        raise ValueError(f'{path.name}: a bill tariff rate is missing')
    return table


class BillTracker:
    """Tariff rows and the month peak in the block of every building, indexed like ``env.buildings``."""

    def __init__(self, env, tables: List[Dict[str, np.ndarray]]):
        self.base = _base(env)
        self.tables = tables
        self.months = [np.asarray(b.energy_simulation.month, dtype=int) for b in self.base.buildings]
        for building, table, months in zip(self.base.buildings, tables, self.months):
            if len(table['peak_block']) < len(months):
                raise ValueError(f'{building.name}: bill tariff has {len(table["peak_block"])} rows, '
                                 f'the simulation {len(months)}')
        # Bound of the peak observation: CityLearn's own estimate of the building's net consumption limit.
        self.peak_bound = [max(float(b.estimate_observation_space_limits(include_all=True)[1]
                                     .get('net_electricity_consumption', 1.0)), 1.0) for b in self.base.buildings]
        self.peak = [0.0]*len(tables)
        self.month: List[Optional[int]] = [None]*len(tables)
        self.last_time_step: Optional[int] = None

    @classmethod
    def from_env(cls, env) -> Optional['BillTracker']:
        base = _base(env)
        schema = getattr(base, 'schema', None)
        if not isinstance(schema, dict):
            return None
        declared = schema.get('buildings') or {}
        names = [b.name for b in base.buildings]
        refs = {n: (declared.get(n) or {}).get('bill_tariff') for n in names}
        if not any(refs.values()):
            return None
        missing = [n for n, ref in refs.items() if not ref]
        if missing:
            raise ValueError(f'bill_tariff declared for some buildings but not for {missing}')
        root = Path(getattr(base, 'root_directory', None) or schema.get('root_directory') or '.')
        return cls(env, [_read(root/refs[n]) for n in names])

    def reset(self):
        """Start of an episode: no peak carried over, even when the next episode is contiguous."""
        self.peak = [0.0]*len(self.tables)
        self.month = [None]*len(self.tables)
        self.last_time_step = None

    def _sync(self, t: int):
        """New episode (non-consecutive step) resets every peak; a new month resets that building's peak."""
        if self.last_time_step is None or t != self.last_time_step + 1:
            self.peak = [0.0]*len(self.tables)
            self.month = [None]*len(self.tables)
        for i, months in enumerate(self.months):
            month = int(months[t])
            if self.month[i] != month:
                self.peak[i], self.month[i] = 0.0, month
        self.last_time_step = t

    def fields(self, index: int, t: int) -> Dict[str, float]:
        """Bill state seen at step t, before its power is known (the peak of earlier hours of the month)."""
        table = self.tables[index]
        same_month = (self.month[index] == int(self.months[index][t])
                      and self.last_time_step is not None and t == self.last_time_step + 1)
        return {'bill_month_block_peak_kw': float(self.peak[index]) if same_month else 0.0,
                'bill_in_peak_block': float(table['peak_block'][t]),
                'bill_demand_usd_kw_month': float(table['demand_usd_kw_month'][t])}

    def reward_fields(self, t: int) -> List[Dict[str, float]]:
        """Fields the reward of step t reads; syncs month and episode first (the only mutation besides update)."""
        self._sync(t)
        return [{'bill_month_block_peak_kw': float(self.peak[i]),
                 'bill_in_peak_block': float(table['peak_block'][t]),
                 'bill_demand_usd_kw_month': float(table['demand_usd_kw_month'][t])}
                for i, table in enumerate(self.tables)]

    def update(self, t: int, net_kw: List[float]):
        """After the reward of step t: the block peak of the month takes this hour's power."""
        for i, power in enumerate(net_kw):
            if self.tables[i]['peak_block'][t] and power > self.peak[i]:
                self.peak[i] = float(power)

    def bounds(self, index: int, normalized: bool):
        if normalized:
            return [1.0, 1.0, 1.0]
        return [self.peak_bound[index], 1.0, float(np.max(self.tables[index]['demand_usd_kw_month']))]

    def observe(self, index: int, t: int, normalized: bool) -> List[float]:
        row = self.fields(index, t)
        values = [row[f] for f in FIELDS]
        if normalized:
            high = self.bounds(index, False)
            values = [min(max(v/max(h, 1e-9), 0.0), 1.0) for v, h in zip(values, high)]
        return values


def tracker_for(env) -> Optional[BillTracker]:
    """The env's tracker, created once and attached to the core env (shared by runtime, reward and Dec-POMDP)."""
    base = _base(env)
    if not hasattr(base, '_bill_tracker'):
        base._bill_tracker = BillTracker.from_env(base)
    return base._bill_tracker
