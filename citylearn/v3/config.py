"""Configuration objects for CityLearn v3 MADRL experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

from citylearn.dec_pomdp import DEFAULT_17_BUILDING_EV_SCHEMA


@dataclass(frozen=True)
class CityLearnV3ExperimentConfig:
    """Configuration for a CityLearn v3 MADRL experiment.

    Defaults point to this project's 17-building + EV thesis dataset, but the
    v3 layer is intentionally generic for any CityLearn v2 schema. Algorithm
    internals must come from the official repositories recorded in
    ``citylearn.official_madrl``.
    """

    schema_path: Path = DEFAULT_17_BUILDING_EV_SCHEMA
    algorithms: Tuple[str, ...] = ("HAPPO", "MASAC", "MATD3", "MAAC")
    scenarios: Tuple[str, ...] = ("E1", "E2", "E3")
    seeds: Tuple[int, ...] = tuple(range(10))
    episode_time_steps: int = 8760
    reward_aggregation: str = "team_mean"
    reward_function: str = "citylearn.reward_function.CityLearnV3MADRLRewardFunction"
    central_agent: bool = False
    backend_policy: str = "official_repository"
    use_citylearn_v2_kpis: bool = True
    multiobjective_method: str = "TOPSIS"
    marl_framework: str = "MARLlib"
    expose_all_citylearn_v2_kpis: bool = True
    citylearn_kwargs: Dict[str, object] = field(default_factory=dict)
    model: Dict[str, object] = field(
        default_factory=lambda: {
            "hidden_sizes": [256, 256],
            "core_arch": "mlp",
            "recurrent_option": ["gru", "lstm"],
        }
    )
    hyperparameters: Dict[str, object] = field(
        default_factory=lambda: {
            "actor_lr": 3e-4,
            "critic_lr": 1e-3,
            "gamma": 0.99,
            "batch_size": 256,
            "replay_buffer_size": 1_000_000,
            "tau": 0.005,
            "ppo_clip": 0.2,
            "matd3_policy_delay": 2,
            "maac_attention_heads": 4,
            "masac_alpha": "auto",
        }
    )

    @property
    def experiment_count(self) -> int:
        return len(self.algorithms) * len(self.scenarios) * len(self.seeds)

    def for_smoke_test(self, *, episode_time_steps: int = 4) -> "CityLearnV3ExperimentConfig":
        """Return a tiny config that preserves the same schema and interfaces."""

        return CityLearnV3ExperimentConfig(
            schema_path=self.schema_path,
            algorithms=self.algorithms,
            scenarios=("E1",),
            seeds=(0,),
            episode_time_steps=episode_time_steps,
            reward_aggregation=self.reward_aggregation,
            central_agent=self.central_agent,
            backend_policy=self.backend_policy,
            use_citylearn_v2_kpis=self.use_citylearn_v2_kpis,
            multiobjective_method=self.multiobjective_method,
            marl_framework=self.marl_framework,
            expose_all_citylearn_v2_kpis=self.expose_all_citylearn_v2_kpis,
            citylearn_kwargs=self.citylearn_kwargs,
            model=self.model,
            hyperparameters=self.hyperparameters,
        )
