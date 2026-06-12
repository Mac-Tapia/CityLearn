"""Check CityLearn v3 MADRL training readiness.

The check is intentionally split between Python 3.9-ready components and
legacy official sources. MATD3's official repository is present, but its
training entry point targets TensorFlow 1.x APIs that are not available in a
native Python 3.9 stack.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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


def _citylearn_v3_smoke(schema_path: Optional[str] = None, scenario: Optional[str] = "E1") -> Dict[str, Any]:
    from citylearn.v3 import describe_environment, make_citylearn_v3_env, make_citylearn_v3_project_env

    if schema_path:
        env = make_citylearn_v3_env(
            schema_path=schema_path,
            scenario=scenario,
            seed=0,
            episode_time_steps=4,
        )
    else:
        env = make_citylearn_v3_project_env(
            scenario=scenario,
            seed=0,
            episode_time_steps=4,
        )

    try:
        observations, infos = env.reset(seed=0)
        actions = {
            agent: np.zeros(env.action_space(agent).shape, dtype=np.float32)
            for agent in env.possible_agents
        }
        next_observations, rewards, terminations, truncations, step_infos = env.step(actions)
        kpi_frame = env.get_kpi_frame()

        return {
            "schema_path": schema_path,
            "scenario": scenario,
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


def _schema_dataset_integrity(schema_path: Optional[str] = None) -> Dict[str, Any]:
    from citylearn.dec_pomdp import DEFAULT_17_BUILDING_EV_SCHEMA, resolve_citylearn_schema_path

    resolved = Path(resolve_citylearn_schema_path(schema_path or DEFAULT_17_BUILDING_EV_SCHEMA))
    if not resolved.exists():
        raise FileNotFoundError(resolved)

    with resolved.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    dataset_dir = resolved.parent
    expected_rows = int(schema["simulation_end_time_step"]) - int(schema["simulation_start_time_step"]) + 1
    building_columns = [
        "month",
        "hour",
        "day_type",
        "daylight_savings_status",
        "indoor_dry_bulb_temperature",
        "average_unmet_cooling_setpoint_difference",
        "indoor_relative_humidity",
        "non_shiftable_load",
        "dhw_demand",
        "cooling_demand",
        "heating_demand",
        "solar_generation",
    ]
    charger_columns = [
        "electric_vehicle_charger_state",
        "electric_vehicle_id",
        "electric_vehicle_departure_time",
        "electric_vehicle_required_soc_departure",
        "electric_vehicle_estimated_arrival_time",
        "electric_vehicle_estimated_soc_arrival",
    ]
    pricing_columns = [
        "electricity_pricing",
        "electricity_pricing_predicted_1",
        "electricity_pricing_predicted_2",
        "electricity_pricing_predicted_3",
    ]

    def csv_header_and_rows(path: Path) -> tuple[list[str], int]:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            return header, sum(1 for _ in reader)

    buildings = {
        name: building
        for name, building in schema.get("buildings", {}).items()
        if building.get("include", True)
    }
    ev_defs = schema.get("electric_vehicles_def", {})
    errors: list[str] = []
    charger_count = 0
    v2g_charger_count = 0
    charger_active_session_count = 0
    pricing_files: set[str] = set()

    for building_name, building in sorted(buildings.items()):
        energy_file = dataset_dir / str(building.get("energy_simulation", ""))
        if not energy_file.exists():
            errors.append(f"{building_name}: missing energy_simulation {energy_file.name}")
        else:
            header, rows = csv_header_and_rows(energy_file)
            if header != building_columns:
                errors.append(f"{building_name}: invalid building CSV columns")
            if rows != expected_rows:
                errors.append(f"{building_name}: rows={rows}, expected={expected_rows}")

        if "cooling_device" not in building:
            errors.append(f"{building_name}: missing cooling_device")
        if "electrical_storage" not in building:
            errors.append(f"{building_name}: missing electrical_storage")
        if "pv" not in building:
            errors.append(f"{building_name}: missing pv")

        pricing_file = building.get("pricing")
        if pricing_file is None:
            errors.append(f"{building_name}: missing pricing file")
        else:
            pricing_files.add(str(pricing_file))

        for charger_name, charger in sorted((building.get("chargers") or {}).items()):
            charger_count += 1
            charger_file = dataset_dir / str(charger.get("charger_simulation", ""))
            attrs = charger.get("attributes", {})
            max_discharge = float(attrs.get("max_discharging_power", 0.0) or 0.0)
            if max_discharge > 0.0:
                v2g_charger_count += 1

            if not charger_file.exists():
                errors.append(f"{building_name}/{charger_name}: missing charger CSV {charger_file.name}")
                continue

            with charger_file.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != charger_columns:
                    errors.append(f"{building_name}/{charger_name}: invalid charger CSV columns")
                    continue

                rows = 0
                states: set[int] = set()
                ev_ids: set[str] = set()
                for row in reader:
                    rows += 1
                    state_text = str(row.get("electric_vehicle_charger_state", "")).strip()
                    try:
                        state = int(float(state_text))
                    except ValueError:
                        errors.append(f"{building_name}/{charger_name}: invalid charger state {state_text!r}")
                        continue
                    states.add(state)
                    if state == 1:
                        charger_active_session_count += 1
                    ev_id = str(row.get("electric_vehicle_id", "")).strip()
                    if ev_id and ev_id.lower() not in {"nan", "none", "null"}:
                        ev_ids.add(ev_id)

                if rows != expected_rows:
                    errors.append(f"{building_name}/{charger_name}: rows={rows}, expected={expected_rows}")
                if not states.issubset({1, 2, 3}):
                    errors.append(f"{building_name}/{charger_name}: invalid states={sorted(states)}")
                for ev_id in ev_ids:
                    if ev_id not in ev_defs:
                        errors.append(f"{building_name}/{charger_name}: EV id {ev_id} missing from electric_vehicles_def")

    pricing_ranges: dict[str, dict[str, float]] = {}
    for pricing_name in sorted(pricing_files):
        pricing_path = dataset_dir / pricing_name
        if not pricing_path.exists():
            errors.append(f"pricing: missing {pricing_name}")
            continue

        values: list[float] = []
        with pricing_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != pricing_columns:
                errors.append(f"pricing: invalid columns in {pricing_name}")
                continue

            rows = 0
            for row in reader:
                rows += 1
                for column in pricing_columns:
                    try:
                        value = float(row[column])
                    except (KeyError, ValueError):
                        errors.append(f"pricing: invalid value in {pricing_name}:{column}")
                        value = float("nan")
                    values.append(value)

            if rows != expected_rows:
                errors.append(f"pricing: rows={rows}, expected={expected_rows} in {pricing_name}")

        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            errors.append(f"pricing: no finite values in {pricing_name}")
            continue
        if float(finite_values.min()) < 0.0:
            errors.append(f"pricing: negative values in {pricing_name}")
        if float(finite_values.max()) <= 0.0:
            errors.append(f"pricing: all-zero values in {pricing_name}")

        pricing_ranges[pricing_name] = {
            "min": float(finite_values.min()),
            "max": float(finite_values.max()),
            "mean": float(finite_values.mean()),
        }

    observations = schema.get("observations", {})
    actions = schema.get("actions", {})
    active_ev_observations = [
        name for name, config in observations.items()
        if "electric_vehicle" in name and isinstance(config, dict) and config.get("active")
    ]
    active_ev_actions = [
        name for name, config in actions.items()
        if "electric_vehicle" in name and isinstance(config, dict) and config.get("active")
    ]

    if not active_ev_observations:
        errors.append("schema: no active EV observations")
    if not active_ev_actions:
        errors.append("schema: no active EV actions")
    if charger_count == 0:
        errors.append("schema: no building chargers")
    if charger_active_session_count == 0:
        errors.append("charger CSVs: no active EV sessions")

    if errors:
        raise RuntimeError("; ".join(errors[:20]))

    return {
        "schema_path": str(resolved),
        "dataset_dir": str(dataset_dir),
        "expected_rows_per_timeseries": expected_rows,
        "included_buildings": len(buildings),
        "chargers": charger_count,
        "v2g_chargers": v2g_charger_count,
        "electric_vehicle_definitions": len(ev_defs),
        "active_ev_session_rows": charger_active_session_count,
        "active_ev_observations": active_ev_observations,
        "active_ev_actions": active_ev_actions,
        "building_csv_structure": "12-column CityLearn standard",
        "pricing_files": sorted(pricing_files),
        "pricing_ranges": pricing_ranges,
    }


def _citylearn_v3_training_adapter_normalization_smoke(
    schema_path: Optional[str] = None,
    scenario: Optional[str] = "E1",
) -> Dict[str, Any]:
    from citylearn_v3_training_common import CityLearnV3BackendAdapter

    adapter = CityLearnV3BackendAdapter(
        schema_path=schema_path,
        scenario=scenario,
        seed=0,
        episode_time_steps=4,
        algorithm="MATD3",
        normalize_observations=True,
    )

    try:
        observations = adapter.reset()
        values = np.concatenate([observation.reshape(-1) for observation in observations])
        state = adapter.ctde_state()
        finite_values = values[np.isfinite(values)]
        finite_state = state[np.isfinite(state)]
        observation_min = float(finite_values.min()) if finite_values.size else None
        observation_max = float(finite_values.max()) if finite_values.size else None
        state_min = float(finite_state.min()) if finite_state.size else None
        state_max = float(finite_state.max()) if finite_state.size else None
        has_ev_observation_names = any(
            "electric_vehicle" in name
            for names in adapter.observation_names_by_agent.values()
            for name in names
        )

        if observation_min is None or observation_min < 0.0 or observation_max is None or observation_max > 1.0:
            raise RuntimeError(
                f"Normalized training observations outside [0, 1]: min={observation_min}, max={observation_max}"
            )

        if state_min is None or state_min < 0.0 or state_max is None or state_max > 1.0:
            raise RuntimeError(
                f"Normalized CTDE state outside [0, 1]: min={state_min}, max={state_max}"
            )

        return {
            "schema_path": schema_path,
            "scenario": scenario,
            "normalization": adapter.normalization_metadata,
            "agents": adapter.agents,
            "num_agents": adapter.n_agents,
            "max_observation_dim": adapter.max_observation_dim,
            "state_dim": adapter.state_dim,
            "observation_min": observation_min,
            "observation_max": observation_max,
            "state_min": state_min,
            "state_max": state_max,
            "has_ev_observation_names": has_ev_observation_names,
        }
    finally:
        adapter.close()


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
        help="Return non-zero if any active Python 3.9 training-critical check fails.",
    )
    parser.add_argument(
        "--schema-path",
        default=None,
        help="Optional CityLearn schema path to smoke-test instead of the project default dataset.",
    )
    parser.add_argument(
        "--scenario",
        default="E1",
        help="Scenario used by the CityLearn v3 smoke check.",
    )
    parser.add_argument(
        "--require-pip-check",
        action="store_true",
        help="Make `python -m pip check` a strict blocker. By default it is reported only.",
    )
    args = parser.parse_args()

    checks: Dict[str, Dict[str, Any]] = {}
    _record(checks, "versions", _versions)
    _record(checks, "pip_check", _pip_check)
    _record(checks, "citylearn_v3_schema_dataset_integrity", lambda: _schema_dataset_integrity(args.schema_path))
    _record(checks, "citylearn_v3_dataset_smoke", lambda: _citylearn_v3_smoke(args.schema_path, args.scenario))
    _record(
        checks,
        "citylearn_v3_training_adapter_normalization_smoke",
        lambda: _citylearn_v3_training_adapter_normalization_smoke(args.schema_path, args.scenario),
    )
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
        "citylearn_v3_dataset_smoke",
        "citylearn_v3_schema_dataset_integrity",
        "citylearn_v3_training_adapter_normalization_smoke",
        "ray_rllib_import",
        "marllib_import",
        "marllib_citylearn_v3_registration",
        "harl_happo_import",
        "masac_import",
        "maac_import",
        "matd3_source_present",
        "matd3_pytorch_offpolicy_import",
    ]

    if args.require_pip_check:
        python39_required.insert(0, "pip_check")

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
        "matd3_legacy_tf1_required": False,
        "matd3_legacy_tf1_available": checks["matd3_legacy_tf1_training_import"]["ok"],
        "matd3_legacy_tf1_note": "Legacy TensorFlow 1.x MATD3 source is kept for reference; active training uses the PyTorch off-policy backend.",
        "pip_metadata_consistent": checks["pip_check"]["ok"],
        "pip_check_required": bool(args.require_pip_check),
        "checks": checks,
    }

    print(json.dumps(report, indent=2, sort_keys=True))

    if args.strict and not python39_core_ready:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
