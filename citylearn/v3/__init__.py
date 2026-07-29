"""CityLearn v3 MADRL layer.

CityLearn v3 is an experimental thesis layer on top of the CityLearn v2
simulator. It keeps CityLearn v2 as the training environment and adds the
Dec-POMDP, KPI, scenario and official-backend integration needed for MADRL
experiments.
"""

from citylearn.v3.config import CityLearnV3ExperimentConfig, N_SEEDS, SEEDS
from citylearn.v3.environment import describe_environment, make_citylearn_v3_env, make_citylearn_v3_project_env
from citylearn.v3.marllib_env import CityLearnV3MARLlibEnv, register_citylearn_v3_marllib_env
from citylearn.v3.objectives import (
    PROJECT_AXIS_METRICS,
    evaluate_objectives,
    objective_manifest,
)

__version__ = "3.0.0-madrl"

__all__ = [
    "CityLearnV3ExperimentConfig",
    "CityLearnV3MARLlibEnv",
    "N_SEEDS",
    "PROJECT_AXIS_METRICS",
    "SEEDS",
    "describe_environment",
    "evaluate_objectives",
    "make_citylearn_v3_env",
    "make_citylearn_v3_project_env",
    "objective_manifest",
    "register_citylearn_v3_marllib_env",
]
