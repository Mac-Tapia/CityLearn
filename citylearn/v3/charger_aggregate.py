"""Charger observations aggregated per vehicle class (SPEC-016 D-V6-49), for every MADRL of the v3 layer.

With one block of observations per socket, an iquitos_v6 agent observed up to 355 values and MASAC no longer
fitted an 80 GB GPU. Each socket stays its own action; the observation keeps every non-charger value of the
building and replaces the seven values of each socket by five per vehicle class of the building:

- ``ev_<class>_occupied_fraction``: sockets with a connected vehicle / sockets of the class;
- ``ev_<class>_soc_mean``: mean SOC of the connected vehicles (0 when none);
- ``ev_<class>_pending_kwh``: energy still to charge, sum of max(0, required SOC - SOC) x battery capacity;
- ``ev_<class>_next_departure_h``: hours to the first departure among the connected (0 when none);
- ``ev_<class>_incoming_fraction``: sockets with an incoming vehicle / sockets of the class.

The values come from the chargers and vehicles themselves (physical SOC and capacity), not from the possibly
normalized observation vector; with normalized observations they are scaled to [0, 1] by their bounds.
Enabled by schema ``electric_vehicle_observation: class_aggregate`` or when every charger declares
``hardware.vehicle_class`` (written by tools/dataset/write_ev_v6.py).
"""
from __future__ import annotations

from typing import Dict, List, Mapping

import numpy as np
from gymnasium import spaces

FEATURES = ('occupied_fraction', 'soc_mean', 'pending_kwh', 'next_departure_h', 'incoming_fraction')
DEPARTURE_BOUND_H = 24.0


def _unwrapped(env):
    inner = env
    while hasattr(inner, 'env') and not hasattr(inner, 'buildings'):
        inner = inner.env
    return getattr(env, 'unwrapped', inner) if not hasattr(inner, 'buildings') else inner


def enabled(env) -> bool:
    base = _unwrapped(env)
    schema = getattr(base, 'schema', None)
    if not isinstance(schema, dict):
        return False
    if schema.get('electric_vehicle_observation') == 'class_aggregate':
        return True
    chargers = [c for b in (schema.get('buildings') or {}).values() for c in (b.get('chargers') or {}).values()]
    return bool(chargers) and all((c.get('hardware') or {}).get('vehicle_class') for c in chargers)


class ChargerClassAggregator:
    """Per agent: indices of the non-charger observations and the class groups of its sockets."""

    def __init__(self, env, *, normalized: bool):
        self.env = env
        self.base = _unwrapped(env)
        self.normalized = bool(normalized)
        schema_buildings = self.base.schema.get('buildings') or {}
        names = env.observation_names
        self.keep: List[np.ndarray] = []
        self.groups: List[Dict[str, list]] = []
        self.capacity: List[Dict[str, float]] = []
        for i, building in enumerate(self.base.buildings):
            row = names[i]
            self.keep.append(np.array([j for j, n in enumerate(row) if 'charger' not in n], dtype=int))
            declared = (schema_buildings.get(building.name) or {}).get('chargers') or {}
            groups: Dict[str, list] = {}
            for charger in building.electric_vehicle_chargers or []:
                hardware = (declared.get(charger.charger_id) or {}).get('hardware') or {}
                kind = hardware.get('vehicle_class') or charger.charger_id.split('_')[1]
                groups.setdefault(kind, []).append(charger)
            self.groups.append(groups)
            evs = {ev.name: ev for ev in getattr(self.base, 'electric_vehicles', [])}
            prefix = f"B{int(building.name.split('_')[-1]):02d}_" if building.name.split('_')[-1].isdigit() else ''
            caps = {}
            for kind in groups:
                own = [float(ev.battery.capacity) for name, ev in evs.items() if name.startswith(f'{prefix}{kind}_')]
                caps[kind] = max(own) if own else max((float(ev.battery.capacity) for ev in evs.values()), default=1.0)
            self.capacity.append(caps)

    def names(self, index: int) -> List[str]:
        row = self.env.observation_names[index]
        base = [row[j] for j in self.keep[index]]
        return base + [f'ev_{kind}_{f}' for kind in self.groups[index] for f in FEATURES]

    def space(self, index: int, space: spaces.Box) -> spaces.Box:
        low = list(np.asarray(space.low, dtype=np.float32)[self.keep[index]])
        high = list(np.asarray(space.high, dtype=np.float32)[self.keep[index]])
        for kind, chargers in self.groups[index].items():
            bounds = (1.0, 1.0, len(chargers)*self.capacity[index][kind], DEPARTURE_BOUND_H, 1.0)
            low += [0.0]*len(FEATURES)
            high += [1.0]*len(FEATURES) if self.normalized else list(bounds)
        return spaces.Box(low=np.array(low, dtype=np.float32), high=np.array(high, dtype=np.float32), dtype=np.float32)

    def _features(self, index: int, kind: str, chargers: list) -> List[float]:
        t = self.base.time_step
        connected, incoming, socs, pending, departures = 0, 0, [], 0.0, []
        for charger in chargers:
            sim = charger.charger_simulation
            state = sim.electric_vehicle_charger_state[t] if t < len(sim.electric_vehicle_charger_state) else 3
            ev = getattr(charger, 'connected_electric_vehicle', None)
            if state == 1 and ev is not None:
                connected += 1
                soc = float(ev.battery.soc[t]) if t < len(ev.battery.soc) else float(ev.battery.soc[-1])
                socs.append(soc)
                required = float(sim.electric_vehicle_required_soc_departure[t])
                pending += max(0.0, required - soc)*float(ev.battery.capacity)
                departures.append(float(max(sim.electric_vehicle_departure_time[t], 0)))
            elif state == 2:
                incoming += 1
        n = max(len(chargers), 1)
        values = [connected/n, float(np.mean(socs)) if socs else 0.0, pending,
                  min(min(departures), DEPARTURE_BOUND_H) if departures else 0.0, incoming/n]
        if self.normalized:
            values[2] = values[2]/max(n*self.capacity[index][kind], 1e-9)
            values[3] = values[3]/DEPARTURE_BOUND_H
        return values

    def transform(self, index: int, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        extra: List[float] = []
        for kind, chargers in self.groups[index].items():
            extra += self._features(index, kind, chargers)
        return np.concatenate([vector[self.keep[index]], np.asarray(extra, dtype=np.float32)]).astype(np.float32)


def aggregator_for(env) -> 'ChargerClassAggregator | None':
    if not enabled(env):
        return None
    from citylearn.wrappers import NormalizedObservationWrapper
    normalized, inner = False, env
    while inner is not None:
        if isinstance(inner, NormalizedObservationWrapper):
            normalized = True
            break
        inner = getattr(inner, 'env', None)
    return ChargerClassAggregator(env, normalized=normalized)


def mapping_names(env, aggregator) -> Mapping[int, List[str]]:
    return {i: aggregator.names(i) for i in range(len(aggregator.keep))}
