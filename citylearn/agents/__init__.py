"""CityLearn agents.

MADRL algorithms are intentionally not implemented in ``citylearn.agents``.
CityLearn v3 uses the official upstream repositories recorded in
``citylearn.official_madrl`` and exposes environment adapters through
``citylearn.v3``.
"""

from .base import Agent
from .rbc import RBC
from .rlc import RLC
from .q_learning import TabularQLearning
from .sac import SAC
from .marlisa import MARLISA

QLearning = TabularQLearning
_MADRL_NAMES = {
    'HAPPO',
    'HAPPOController',
    'MASAC',
    'MASACController',
    'MATD3',
    'MATD3Controller',
    'MAAC',
    'MAACController',
}


def __getattr__(name):
    if name in _MADRL_NAMES:
        raise ImportError(
            f"{name} is not implemented locally in citylearn.agents. "
            "Use citylearn.v3 adapters with the official repositories listed "
            "in citylearn.official_madrl."
        )

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'Agent',
    'RBC',
    'RLC',
    'QLearning',
    'TabularQLearning',
    'SAC',
    'MARLISA',
]
