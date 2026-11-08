"""Common action grid of the 4 MADRL in the CityLearn v3 layer (SPEC-015 D-M3-7, D-M3-9 and D-M3-10; one head per equipment, P-M3-2).

Every equipment action (BESS, each charger socket, the air conditioning) takes one of ``LEVELS`` evenly spaced
values over its own range: -1, -0.5, 0, 0.5 and 1 for a [-1, 1] action, 0 to 1 in quarters for the air conditioning.
HAPPO and MATD3 keep their continuous output and ``round`` sends the nearest level to the environment; MASAC and
MAAC choose one level per equipment (a multi-discrete head, one branch per equipment, ``decode``). Agents with fewer
equipments than the largest one get padded branches whose only available level is ``PAD_LEVEL`` (``avail_mask``),
so backends that need one action size for all agents can share it.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

LEVELS = 5            # D-M3-10 (owner, 2026-11-08): 11 levels did not fit MASAC on an 80 GB GPU
PAD_LEVEL = 0


class ActionGrid:
    def __init__(self, action_spaces: Sequence):
        self.low: List[np.ndarray] = [np.asarray(s.low, dtype=np.float64).reshape(-1) for s in action_spaces]
        self.high: List[np.ndarray] = [np.asarray(s.high, dtype=np.float64).reshape(-1) for s in action_spaces]
        self.branches = [len(lo) for lo in self.low]
        self.max_branches = max(self.branches) if self.branches else 0
        self._levels = [np.linspace(lo, hi, LEVELS, axis=1) for lo, hi in zip(self.low, self.high)]

    @property
    def n_actions(self) -> int:
        """Size of the flat multi-discrete head shared by all agents (max branches x levels)."""
        return self.max_branches*LEVELS

    def levels(self, agent: int, equipment: int) -> np.ndarray:
        return self._levels[agent][equipment]

    def round(self, agent: int, values) -> np.ndarray:
        """Nearest level of each equipment to a continuous output (clipped to the range first)."""
        values = np.clip(np.asarray(values, dtype=np.float64).reshape(-1), self.low[agent], self.high[agent])
        table = self._levels[agent]
        return table[np.arange(len(values)), np.abs(table - values[:, None]).argmin(axis=1)]

    def decode(self, agent: int, indices) -> np.ndarray:
        """Level index per branch -> action values of the agent's equipments (padded branches ignored)."""
        indices = np.asarray(indices, dtype=int).reshape(-1)
        k = self.branches[agent]
        if len(indices) not in (k, self.max_branches):
            raise ValueError(f'agent {agent}: {len(indices)} indices for {k} equipments')
        real = indices[:k]
        if real.min(initial=0) < 0 or real.max(initial=0) >= LEVELS:
            raise ValueError(f'agent {agent}: level indices must be in [0, {LEVELS})')
        return self._levels[agent][np.arange(k), real]

    def avail_mask(self, agent: int) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[:self.branches[agent]*LEVELS] = True
        for branch in range(self.branches[agent], self.max_branches):
            mask[branch*LEVELS + PAD_LEVEL] = True
        return mask

    def onehot(self, agent: int, indices) -> np.ndarray:
        indices = np.asarray(indices, dtype=int).reshape(-1)
        out = np.zeros(self.n_actions, dtype=np.float32)
        for branch in range(self.max_branches):
            level = indices[branch] if branch < len(indices) else PAD_LEVEL
            out[branch*LEVELS + int(level)] = 1.0
        return out

    def indices_from_onehot(self, onehot) -> np.ndarray:
        return np.asarray(onehot, dtype=np.float32).reshape(self.max_branches, LEVELS).argmax(axis=1)

    def describe(self) -> dict:
        return {'levels': LEVELS, 'pad_level': PAD_LEVEL, 'branches_by_agent': list(self.branches),
                'max_branches': self.max_branches, 'n_actions': self.n_actions,
                'decision': 'SPEC-015 D-M3-7, D-M3-9, D-M3-10'}
