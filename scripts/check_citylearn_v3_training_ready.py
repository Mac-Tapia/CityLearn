"""Check CityLearn v3 MADRL training readiness.

The check is intentionally split between Python 3.9-ready components and
legacy official sources. MATD3's official repository is present, but its
training entry point targets TensorFlow 1.x APIs that are not available in a
native Python 3.9 stack.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
CITYLEARN_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]
EXTERNAL_ROOT = PROJECT_ROOT / "external"


def _prepend(path: Path) -> None:
    path_text = str(path)

    if path.exists() and path_text not in sys.path:
        sys.path.insert(0, path_text)


_prepend(CITYLEARN_ROOT)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.integer, np.floating)):
        return value.item()

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]

    return value


def _record(results: Dict[str, Dict[str, Any]], name: str, fn: Callable[[], Any]) -> None:
    try:
        results[name] = {"ok": True, "detail": _jsonable(fn())}
    except Exception as exc:  # pragma: no cover - diagnostic script.
        results[name] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _pip_check() -> str:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    output = (completed.stdout + completed.stderr).strip()

    if completed.returncode != 0:
        raise RuntimeError(output)

    return output


def _citylearn_v3_smoke() -> Dict[str, Any]:
    from citylearn.v3 import describe_environment, make_citylearn_v3_project_env

    env = make_citylearn_v3_project_env(episode_time_steps=4)

    try:
        observations, infos = env.reset(seed=0)
        actions = {
            agent: np.zeros(env.action_space(agent).shape, dtype=np.float32)
            for agent in env.possible_agents
        }
        next_observations, rewards, terminations, truncations, step_infos = env.step(actions)
        kpi_frame = env.get_kpi_frame()

        return {
            "environment": describe_environment(env),
            "initial_observation_agents": sorted(observations),
            "next_observation_agents": sorted(next_observations),
            "info_agents": sorted(infos),
            "step_info_agents": sorted(step_infos),
            "reward_agents": sorted(rewards),
            "terminated": any(terminations.values()),
            "truncated": any(truncations.values()),
            "kpi_frame_shape": kpi_frame.shape,
        }
    finally:
        env.close()


def _marllib_import() -> str:
    _prepend(EXTERNAL_ROOT / "MARLlib")
    module = importlib.import_module("marllib")
    importlib.import_module("marllib.marl")
    return str(Path(module.__file__).resolve())


def _marllib_registration() -> str:
    from citylearn.v3 import register_citylearn_v3_marllib_env

    _prepend(EXTERNAL_ROOT / "MARLlib")
    return register_citylearn_v3_marllib_env()


def _harl_happo_import() -> Dict[str, Any]:
    _prepend(EXTERNAL_ROOT / "HARL")
    module = importlib.import_module("harl.algorithms.actors.happo")
    runners = importlib.import_module("harl.runners")

    return {
        "class": getattr(module, "HAPPO").__name__,
        "has_runner_registry": hasattr(runners, "RUNNER_REGISTRY"),
    }


def _masac_import() -> str:
    _prepend(EXTERNAL_ROOT / "MARL" / "src")
    module = importlib.import_module("runner_msac")
    return str(Path(module.__file__).resolve())


def _maac_import() -> str:
    _prepend(EXTERNAL_ROOT / "MAAC")
    sys.modules.pop("utils", None)
    module = importlib.import_module("algorithms.attention_sac")
    return getattr(module, "AttentionSAC").__name__


def _matd3_source_present() -> Dict[str, Any]:
    root = EXTERNAL_ROOT / "MATD3implementation"
    train_path = root / "matd3" / "train.py"

    if not train_path.exists():
        raise FileNotFoundError(train_path)

    return {
        "path": root,
        "train_entrypoint": train_path,
        "legacy_note": "Official training entry point imports tensorflow.contrib and requires TensorFlow 1.x.",
    }


def _matd3_legacy_training_import() -> str:
    _prepend(EXTERNAL_ROOT / "MATD3implementation" / "matd3")
    module = importlib.import_module("train")
    return str(Path(module.__file__).resolve())


def _matd3_pytorch_offpolicy_import() -> Dict[str, str]:
    _prepend(EXTERNAL_ROOT / "off-policy")
    module = importlib.import_module("offpolicy.algorithms.matd3.matd3")
    recurrent_module = importlib.import_module("offpolicy.algorithms.r_matd3.r_matd3")
    policy_module = importlib.import_module("offpolicy.algorithms.matd3.algorithm.MATD3Policy")

    return {
        "backend": "marlbenchmark/off-policy",
        "matd3_class": getattr(module, "MATD3").__name__,
        "rmatd3_class": getattr(recurrent_module, "R_MATD3").__name__,
        "policy_class": getattr(policy_module, "MATD3Policy").__name__,
        "path": str((EXTERNAL_ROOT / "off-policy").resolve()),
    }


def _versions() -> Dict[str, str]:
    packages = [
        "citylearn",
        "ray",
        "gym",
        "gymnasium",
        "numpy",
        "torch",
        "pettingzoo",
        "supersuit",
    ]
    versions = {"python": sys.version}

    for package in packages:
        module = importlib.import_module(package)
        versions[package] = str(getattr(module, "__version__", "unknown"))

    return versions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if any Python 3.9 training-critical check fails.",
    )
    args = parser.parse_args()

    checks: Dict[str, Dict[str, Any]] = {}
    _record(checks, "versions", _versions)
    _record(checks, "pip_check", _pip_check)
    _record(checks, "citylearn_v3_project_smoke", _citylearn_v3_smoke)
    _record(checks, "ray_rllib_import", lambda: importlib.import_module("ray.rllib").__name__)
    _record(checks, "marllib_import", _marllib_import)
    _record(checks, "marllib_citylearn_v3_registration", _marllib_registration)
    _record(checks, "harl_happo_import", _harl_happo_import)
    _record(checks, "masac_import", _masac_import)
    _record(checks, "maac_import", _maac_import)
    _record(checks, "matd3_source_present", _matd3_source_present)
    _record(checks, "matd3_legacy_tf1_training_import", _matd3_legacy_training_import)
    _record(checks, "matd3_pytorch_offpolicy_import", _matd3_pytorch_offpolicy_import)

    python39_required = [
        "pip_check",
        "citylearn_v3_project_smoke",
        "ray_rllib_import",
        "marllib_import",
        "marllib_citylearn_v3_registration",
        "harl_happo_import",
        "masac_import",
        "maac_import",
        "matd3_source_present",
        "matd3_pytorch_offpolicy_import",
    ]
    python39_core_ready = all(checks[name]["ok"] for name in python39_required)
    four_algorithm_python39_training_imports_ready = (
        checks["harl_happo_import"]["ok"]
        and checks["masac_import"]["ok"]
        and checks["maac_import"]["ok"]
        and checks["matd3_pytorch_offpolicy_import"]["ok"]
    )

    report = {
        "project_root": str(PROJECT_ROOT),
        "python39_core_ready": python39_core_ready,
        "four_algorithm_sources_present": all(
            checks[name]["ok"]
            for name in [
                "harl_happo_import",
                "masac_import",
                "maac_import",
                "matd3_pytorch_offpolicy_import",
            ]
        ),
        "matd3_pytorch_backend_ready": checks["matd3_pytorch_offpolicy_import"]["ok"],
        "four_algorithm_python39_training_imports_ready": four_algorithm_python39_training_imports_ready,
        "matd3_legacy_tf1_required": (
            checks["matd3_source_present"]["ok"]
            and not checks["matd3_legacy_tf1_training_import"]["ok"]
        ),
        "checks": checks,
    }

    print(json.dumps(report, indent=2, sort_keys=True))

    if args.strict and not python39_core_ready:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
