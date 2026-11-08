# Decisions implemented here (governance/decision_traceability.yaml):
#   P-017-1: declaration of the built-in backends
"""Universal MADRL backend contract of the CityLearn v3 layer (SPEC-017 FR-017.3).

What the layer guarantees to any MADRL backend:
- one agent per building, each with its own Box observation and action space;
- local observations and the CTDE global state (concatenated local observations);
- rewards per agent, or aggregated for a team, from the v3 reward.

A backend declares its action interface ('continuous' or 'discrete_masked'), its reward kind ('per_agent'
or 'team'), whether it uses the global state, and the import path of its adapter. HAPPO, MASAC, MATD3 and
MAAC are built in with their native adapters (CityLearn/scripts/citylearn_v3_training_common.py); a new
backend calls ``register_backend`` and needs no edit of the layer.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional

import numpy as np

ACTION_INTERFACES = ("continuous", "discrete_masked")
REWARD_KINDS = ("per_agent", "team")
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
_COMMON = "citylearn_v3_training_common"


@dataclass(frozen=True)
class MADRLBackend:
    name: str
    action_interface: str
    reward: str
    uses_global_state: bool
    adapter: str
    policy_factory: Optional[Callable] = None
    source: str = ""
    adapter_kwargs: Mapping = field(default_factory=dict)
    adapter_step: Optional[Callable] = None   # (adapter_env, rng) -> reward of one step, in the backend's format


@dataclass(frozen=True)
class ContractReport:
    passed: bool
    failures: List[str] = field(default_factory=list)
    n_agents: int = 0
    observation_dims: Mapping[str, int] = field(default_factory=dict)
    action_dims: Mapping[str, int] = field(default_factory=dict)
    state_dim: int = 0


@dataclass(frozen=True)
class SmokeResult:
    backend: str
    steps: int
    agents: int
    total_reward: float


_REGISTRY: Dict[str, MADRLBackend] = {}


def register_backend(spec: MADRLBackend) -> None:
    if spec.action_interface not in ACTION_INTERFACES:
        raise ValueError(f"{spec.name}: unknown action interface {spec.action_interface!r}")
    if spec.reward not in REWARD_KINDS:
        raise ValueError(f"{spec.name}: unknown reward kind {spec.reward!r}")
    if ":" not in spec.adapter:
        raise ValueError(f"{spec.name}: adapter must be 'module:Class', got {spec.adapter!r}")
    if spec.name in _REGISTRY:
        raise ValueError(f"backend {spec.name} is already registered")
    _REGISTRY[spec.name] = spec


def unregister_backend(name: str) -> None:
    _REGISTRY.pop(name, None)


def registered_backends() -> Dict[str, MADRLBackend]:
    return dict(_REGISTRY)


def resolve_adapter(name: str):
    module_name, class_name = _REGISTRY[name].adapter.split(":")
    if module_name == _COMMON and str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    return getattr(importlib.import_module(module_name), class_name)


def _dim(space) -> int:
    return int(np.prod(space.shape))


def check_contract(env) -> ContractReport:
    """Check on a Dec-POMDP env the guarantees every backend relies on (resets and steps it once)."""
    failures: List[str] = []
    agents = list(env.possible_agents)
    obs_dims = {a: _dim(env.observation_space(a)) for a in agents}
    act_dims = {a: _dim(env.action_space(a)) for a in agents}
    if not agents:
        failures.append("no agents")
    observations, _ = env.reset()
    for agent in agents:
        if agent not in observations:
            failures.append(f"{agent}: missing observation after reset")
        elif np.asarray(observations[agent]).reshape(-1).shape[0] != obs_dims[agent]:
            failures.append(f"{agent}: observation size differs from its space")
    state = np.asarray(env.state()).reshape(-1)
    if state.shape[0] != sum(obs_dims.values()):
        failures.append(f"state size {state.shape[0]} != sum of local observations {sum(obs_dims.values())}")
    zero = {a: np.zeros(act_dims[a], dtype=np.float32) for a in agents}
    result = env.step(zero)
    if len(result) != 5:
        failures.append(f"step returned {len(result)} values, expected 5")
    else:
        rewards = result[1]
        if set(rewards) != set(agents) or not all(np.isfinite(float(r)) for r in rewards.values()):
            failures.append("rewards must be finite and given for every agent")
    return ContractReport(passed=not failures, failures=failures, n_agents=len(agents),
                          observation_dims=obs_dims, action_dims=act_dims, state_dim=int(state.shape[0]))


def run_smoke_episode(env, name: str, *, steps: int, seed: int = 0) -> SmokeResult:
    """Short episode of a registered backend's policy on the Dec-POMDP env."""
    spec = _REGISTRY[name]
    if spec.policy_factory is None:
        raise ValueError(f"{name} has no policy factory for a Dec-POMDP smoke episode")
    act = spec.policy_factory(env, seed)
    observations, _ = env.reset(seed=seed)
    total, done = 0.0, 0
    for _ in range(steps):
        if not env.agents:
            break
        observations, rewards, _, _, _ = env.step(act(observations))
        total += float(sum(rewards.values()))
        done += 1
    return SmokeResult(backend=name, steps=done, agents=len(env.possible_agents), total_reward=total)


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: List[str]
    contract: Optional[ContractReport]
    backends: Mapping[str, Mapping[str, float]]


def _uniform(rng, space):
    return rng.uniform(np.asarray(space.low, dtype=float), np.asarray(space.high, dtype=float)).astype(np.float32)


def _step_harl(env, rng):
    env.reset()
    out = env.step(np.asarray([_uniform(rng, s) for s in env.action_space]))
    return float(np.sum(out[2]))


def _step_offpolicy(env, rng):
    env.reset()
    out = env.step([[_uniform(rng, s) for s in env.action_space]])
    return float(np.sum(out[1]))


def _step_smac(env, rng):
    env.reset()
    actions = [int(rng.choice(np.flatnonzero(env.get_avail_agent_actions(i)))) for i in range(env.n_agents)]
    return float(env.step(actions)[0])


def _step_maac(env, rng):
    env.reset()
    out = env.step([[np.eye(s.n)[int(rng.integers(s.n))] for s in env.action_space]])
    return float(np.sum(out[1]))


def operational_gate(schema_path, *, backends: Optional[List[str]] = None, steps: int = 2, seed: int = 0,
                     scenario: str = "E1", episode_time_steps: int = 24) -> GateResult:
    """FR-017.4: a dataset is operational only if v3 loads it, meets the contract and every backend steps.

    Each step resets the adapter, so ``steps`` counts independent one-step checks of the backend's own
    adapter. Any exception becomes a failure: the gate never raises and never passes by omission.
    """
    from citylearn.v3.environment import make_citylearn_v3_env

    failures: List[str] = []
    contract = None
    try:
        env = make_citylearn_v3_env(schema_path=schema_path, scenario=scenario, seed=seed,
                                    episode_time_steps=episode_time_steps)
        try:
            contract = check_contract(env)
            failures += [f"contract: {f}" for f in contract.failures]
        finally:
            env.close()
    except Exception as error:  # noqa: BLE001 - the gate reports every failure
        failures.append(f"v3 env: {type(error).__name__}: {error}")
        return GateResult(False, failures, None, {})

    results: Dict[str, Dict[str, float]] = {}
    for name in backends or sorted(_REGISTRY):
        spec = _REGISTRY[name]
        if spec.adapter_step is None:
            continue
        rng = np.random.default_rng(seed)
        try:
            adapter = resolve_adapter(name)(schema_path=str(schema_path), scenario=scenario, seed=seed,
                                            episode_time_steps=episode_time_steps, algorithm=name,
                                            **dict(spec.adapter_kwargs))
            try:
                rewards = [spec.adapter_step(adapter, rng) for _ in range(steps)]
            finally:
                close = getattr(adapter, "close", None)
                if close:
                    close()
            total = float(np.sum(rewards))
            if not np.isfinite(total):
                failures.append(f"{name}: non-finite reward")
            results[name] = {"steps": len(rewards), "reward": total}
        except Exception as error:  # noqa: BLE001
            failures.append(f"{name}: {type(error).__name__}: {error}")
    return GateResult(passed=not failures, failures=failures, contract=contract, backends=results)


_DISCRETE = {"action_bins": 3, "discrete_action_mode": "axis"}

for _spec in (
    MADRLBackend("HAPPO", "continuous", "per_agent", True, f"{_COMMON}:CityLearnHARLEnv",
                 source="external/HARL (Zhong et al.)", adapter_step=_step_harl),
    MADRLBackend("MASAC", "discrete_masked", "team", True, f"{_COMMON}:CityLearnSMACDiscreteEnv",
                 source="external/MARL", adapter_kwargs=_DISCRETE, adapter_step=_step_smac),
    MADRLBackend("MATD3", "continuous", "per_agent", True, f"{_COMMON}:CityLearnOffPolicyVecEnv",
                 source="external/off-policy", adapter_step=_step_offpolicy),
    MADRLBackend("MAAC", "discrete_masked", "per_agent", True, f"{_COMMON}:CityLearnMAACVecEnv",
                 source="external/MAAC (Iqbal and Sha)", adapter_kwargs=_DISCRETE, adapter_step=_step_maac),
):
    register_backend(_spec)
