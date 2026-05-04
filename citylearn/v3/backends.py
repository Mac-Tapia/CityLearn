"""CityLearn v3 official backend manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

from citylearn.official_madrl import OFFICIAL_MADRL_SOURCES, official_backend_status
from citylearn.v3.config import CityLearnV3ExperimentConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_ROOT = PROJECT_ROOT / "external"
BACKENDS_LOCK = EXTERNAL_ROOT / "backends.lock.json"


def _load_backend_lock() -> Dict[str, object]:
    if not BACKENDS_LOCK.exists():
        return {}

    with BACKENDS_LOCK.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data.get("backends", {})


def citylearn_v3_backend_manifest(
    config: Optional[CityLearnV3ExperimentConfig] = None,
    *,
    algorithms: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """Return the source-backed training manifest for CityLearn v3."""

    config = CityLearnV3ExperimentConfig() if config is None else config
    selected = tuple(algorithm.upper() for algorithm in (algorithms or config.algorithms))
    status = official_backend_status((*selected, "MARLLIB"), external_root=EXTERNAL_ROOT)
    locked_backends = _load_backend_lock()

    return {
        "version_layer": "citylearn-v3-madrl",
        "simulator": "citylearn-v2",
        "schema_path": str(config.schema_path),
        "dec_pomdp": {
            "central_agent": config.central_agent,
            "reward_aggregation": config.reward_aggregation,
            "reward_function": config.reward_function,
            "reward_policy": "axis_and_algorithm_specific_citylearn_v3_reward",
            "state": "concatenated_local_observations_for_ctde",
            "agents": "CityLearn buildings with EV chargers embedded in building action spaces",
        },
        "evaluation": {
            "kpis": "CityLearn evaluate_v2",
            "multiobjective_method": config.multiobjective_method,
        },
        "experiment_count": config.experiment_count,
        "backends": status,
        "locked_commits": locked_backends,
        "external_root": str(EXTERNAL_ROOT),
        "sources": {
            name: {
                "paper": OFFICIAL_MADRL_SOURCES[name].paper_url,
                "repository": OFFICIAL_MADRL_SOURCES[name].repository_url,
            }
            for name in selected
        },
    }
