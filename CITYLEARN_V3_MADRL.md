# CityLearn v3 MADRL Layer

CityLearn v3 in this project is an experimental MADRL layer on top of the
existing CityLearn v2 simulator. CityLearn v2 remains the source of truth for
all datasets, building physics, EV chargers, observations, actions, rewards and
`evaluate_v2` KPIs.

The goal is not to replace or limit CityLearn v2. The goal is to evolve the
current project from MARL support to source-backed MADRL support while
preserving every current CityLearn v2 dataset and KPI surface. Under this
design, CityLearn v3 is the research extension that makes CityLearn usable for
decentralized, collaborative, scalable and multi-objective MADRL studies.

## Scientific Contribution

This layer can be treated as a scientific contribution because it provides a
reproducible bridge between:

- the CityLearn v2 building-energy simulator;
- any current CityLearn v2 schema/dataset as a decentralized energy community;
- Dec-POMDP/CTDE training interfaces;
- complete CityLearn v2 KPI reporting, plus thesis KPI summaries for
  flexibility, CO2 emissions and energy costs;
- official MADRL algorithm sources instead of locally invented implementations.

Future studies can reuse the same environment contract and swap algorithms,
datasets, reward aggregation, scenario definitions or KPI weighting without
changing CityLearn v2 internals.

## Architecture

- `citylearn.citylearn.CityLearnEnv`: unchanged CityLearn v2 simulator.
- `citylearn.dec_pomdp.CityLearnDecPOMDPEnv`: generic PettingZoo ParallelEnv
  wrapper for any CityLearn v2 schema as a decentralized partially observable
  Markov game.
- `citylearn.v3`: research-facing package that builds generic CityLearn v2
  Dec-POMDP environments and exposes MARLlib/RLlib-compatible adapters.
- `external/`: official upstream repositories cloned as source-backed MADRL
  backends. These repositories are part of the project as pinned scientific
  dependencies, not duplicated algorithm code inside `citylearn.agents`.

## Official Sources

- HAPPO: `external/HARL`
- MASAC/mSAC: `external/MARL`
- MATD3: `external/MATD3implementation`
- MAAC: `external/MAAC`
- MARLlib framework: `external/MARLlib`

There are no local `citylearn.agents.*` MADRL implementations. Thesis-grade
training must use the official upstream sources through the v3 adapters or an
isolated compatible environment.

## Naming Boundary

- `citylearn.__version__` remains the upstream CityLearn v2 package version so
  dataset resolution and existing APIs keep working.
- `citylearn.v3.__version__` identifies the MADRL research layer.
- A future distributable release can promote this layer to a formal CityLearn
  v3 package once training backends, dependency locks and experiment pipelines
  are fully reproducible.

## Dataset Scope

The v3 environment factory is generic:

```python
from citylearn.v3 import make_citylearn_v3_env

env = make_citylearn_v3_env(
    schema_path="data/datasets/baeda_3dem/schema.json",
    episode_time_steps=24,
)
```

This project's thesis default remains 17 buildings + EV:

```python
from citylearn.v3 import make_citylearn_v3_project_env

env = make_citylearn_v3_project_env(scenario="E1", seed=0)
```

## Project Default Environment

The project default v3 environment uses:

- Schema: `citylearn_challenge_2022_phase_all_plus_evs/schema.json`
- Agents: 17 CityLearn buildings
- EV: charger actions/observations embedded in the building spaces
- Reward: collaborative team mean by default
- CTDE state: concatenated local observations
- KPIs: full CityLearn v2 `evaluate_v2` table, plus thesis summary extraction

## Thesis Objective KPIs

The objective manifest is implemented in `citylearn.v3.objectives` and is
validated by `CityLearn/scripts/validate_citylearn_v3_objectives.py`.

- OE1, energy flexibility: load-shape KPIs (`peak_average`,
  `ramping_average`, `one_minus_load_factor_average`), grid import/export and
  net-exchange KPIs, PV/autoconsumption KPIs, battery KPIs and EV KPIs.
- OE2, CO2 emissions: `carbon_emissions`, total control/baseline/delta
  emissions in kgCO2, and daily-average control/baseline/delta emissions.
- OE3, energy costs: `electricity_cost`, total and daily-average
  control/baseline/delta costs in EUR, demand peak/ramping/load-factor cost
  support KPIs, and `price_signal_deviation`.

Project metrics are the three thesis axes: OE1, OE2 and OE3. CO2 is no longer
an additional metric; it is the complete second objective axis. The
`axis_kpis` section contains the KPI values for the three axes, and every KPI
report includes a baseline comparison when CityLearn v2 exposes the required
baseline/control/delta values.

`electricity_cost` and `carbon_emissions` are CityLearn v2 ratios to baseline.
Raw totals are exposed separately as `*_control`, `*_baseline` and `*_delta`
to avoid mixing normalized ratios with EUR or kgCO2.

`price_signal_deviation` is not a native CityLearn `evaluate_v2` KPI in this
codebase. It is explicitly marked as a derived project KPI and is computed from
CityLearn v2 district net import and `electricity_pricing` time series.

Run the objective validation report:

```powershell
python -B CityLearn\scripts\validate_citylearn_v3_objectives.py `
  --scenario E3 `
  --episode-time-steps 4 `
  --include-citylearn-v2-test-agents `
  --output-dir outputs\citylearn_v3_madrl_validation
```

## Smoke Usage

```python
from citylearn.v3 import CityLearnV3ExperimentConfig, make_citylearn_v3_project_env

config = CityLearnV3ExperimentConfig().for_smoke_test()
env = make_citylearn_v3_project_env(config, scenario="E1", seed=0)
obs, infos = env.reset()
state = env.state()
all_citylearn_v2_kpis = env.get_all_kpis()
thesis_kpi_summary = env.get_kpis()
env.close()
```

## Python 3.9 Training Environment

The validated local training environment is:

- Virtual environment: `.venv39-citylearn-v3`
- Python: `3.9.25`
- CityLearn: editable install from `CityLearn/`
- MARLlib/RLlib stack: `ray==1.8.0`, `gym==0.20.0`
- CityLearn compatibility stack: `gymnasium==0.28.1`, `numpy==1.23.5`

Recreate it from PowerShell:

```powershell
.\CityLearn\scripts\setup_citylearn_v3_training_env.ps1
```

Activate and validate it:

```powershell
.\.venv39-citylearn-v3\Scripts\Activate.ps1
python -B CityLearn\scripts\check_citylearn_v3_training_ready.py --strict
python -B CityLearn\scripts\run_citylearn_v3_env_smoke.py --episode-time-steps 4 --steps 3
```

The readiness check verifies:

- `pip check` has no broken requirements.
- CityLearn v3 builds the 17-building + EV Dec-POMDP.
- CTDE global state and local decentralized observations/actions are exposed.
- Full CityLearn v2 KPI tables are available.
- MARLlib imports and registers `citylearn_v3`.
- HAPPO, MASAC and MAAC official sources import in Python 3.9.
- MATD3 PyTorch imports through the `marlbenchmark/off-policy` backend.
- MATD3 official source is present, but its training entry point imports
  `tensorflow.contrib`, so official MATD3 training requires a separate legacy
  TensorFlow 1.x environment or a documented compatibility port.

## MATD3 PyTorch Backend

The original MATD3 author repository remains pinned as
`external/MATD3implementation`. It is the paper source, but it targets Python
3.6, TensorFlow 1.x and Gym 0.10.

For Python 3.9 PyTorch training, this project also pins:

- Repository: `https://github.com/marlbenchmark/off-policy`
- Local path: `external/off-policy`
- Branch: `release`
- Commit: `41fd5eb46d12df2847e1c2e29842997ff2c24998`
- Classes verified in the Python 3.9 environment:
  - `offpolicy.algorithms.matd3.matd3.MATD3`
  - `offpolicy.algorithms.r_matd3.r_matd3.R_MATD3`
  - `offpolicy.algorithms.matd3.algorithm.MATD3Policy`

This backend is source-backed PyTorch MATD3/RMATD3. It must be documented as a
compatible PyTorch reference backend, not as the original author repository.

## MARLlib Adapter

`citylearn.v3.CityLearnV3MARLlibEnv` exposes a Ray/RLlib-style multi-agent
environment. Because the 17-building EV dataset has heterogeneous observation
and action dimensions, the adapter pads observations/actions to the maximum
local dimension and slices actions back before stepping CityLearn.

MARLlib itself is cloned under `external/MARLlib`. Its upstream setup pins
Ray/RLlib 1.8.0. The project environment keeps that Ray version and uses
`numpy==1.23.5` so CityLearn/Gymnasium remain functional under Python 3.9.
