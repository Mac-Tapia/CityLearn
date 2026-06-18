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

- Schema: `citylearn_iquitos_2023_2025/schema.json`
- Agents: 17 real Iquitos buildings
- EV: 185 Mode 3 charger loadpoints embedded in the building spaces; 31
  camioneta loadpoints are V2G bidirectional and non-camioneta EVs remain
  charge-only
- Reward: collaborative team mean by default
- CTDE state: concatenated local observations
- KPIs: full CityLearn v2 `evaluate_v2` table, plus thesis summary extraction

## Iquitos Dataset Distillation

The active thesis dataset is generated from real building inputs in
`CityLearn/data/buildingcsv/` and integrated into
`CityLearn/data/datasets/citylearn_iquitos_2023_2025/`.

### Real building parameters (updated 2026-06-04)

`building.csv` provides the authoritative building inventory for all 17 buildings:
real official names, exact roof areas, CityLearn use types, large cooling systems
(Chiller Water-Cooled, Multi-Chiller Plant, Clinical Chiller + HEPA,
DataCenter Precision AC, Industrial Mobile Vessel AC, Scientific Ultra-Freezers -80C),
estimated split AC unit counts, and predominant vehicle types.

Key area corrections from building.csv vs previous estimates:

| ID | Building | Area m2 | Cooling system |
|---|---|---:|---|
| B05 | Hotel Plaza S.A. | 1,141.89 | Commercial Kitchen Cold Rooms |
| B09 | Gobierno Regional COER | 4,479.67 | DataCenter Precision AC (N+1) |
| B10 | Gobierno Regional de Loreto | 14,295.73 | Duct Central Split System |
| B11 | Hospital Regional de Loreto | 42,649.33 | Clinical Chiller + HEPA + Blood Bank |
| B12 | Seguro Social EsSalud | 18,197.48 | Medical Archive AC System |
| B14 | Autoridad Portuaria Nacional | 17,761.00 | Splits autonomos |
| B15 | DREL Colegio Nacional | 9,889.92 | Splits autonomos |
| B16 | SIMA Iquitos S.R.Ltda | 10,294.00 | Industrial Mobile Vessel AC |
| B17 | Asociacion Civil Selva Amazonica | 1,611.23 | Scientific Ultra-Freezers -80C |

`cooling_peak` is estimated as `split_units * 3.5 kW` plus large system capacity.
COP is assigned per real cooling system type (Chiller=4.5, Multi-Chiller=5.0,
Precision AC=2.0, Ultra-Freezers=0.8, splits=2.8).

### Distillation from monthly measurements

- `B_02.csv` through `B_17.csv` provide monthly measured meter inputs.
  `Building_1.csv` is preserved because there is no matching `buildingcsv`
  source for B_01. All building CSVs keep the 12-column, 26,304-row structure.
- `tools/distill_building_loads.py` converts monthly measurements into hourly
  CityLearn loads by calendar-aware transformations, not arbitrary synthesis.
  `EnergiaActivaHoraPunta` and `EnergiaActivaFueraPunta` are the physical kWh
  source; `totalEnergiaActiva` is used as fallback only when the peak/off-peak
  split is missing.
- The distilled `non_shiftable_load` is residual:
  `NSL = E_medido_mes - cooling_demand/COP - dhw_demand/COP`.
  Monthly balance delta is guaranteed < 0.1%. EV, BESS and PV are control/DER
  assets and are not subtracted from historical building meter energy.
- `TotalFacturado` and `Tarifa` calibrate `pricing.csv` via
  `C_mes = p_punta * E_punta + p_fuera * E_fuera`. Output is the
  CityLearn-compatible hourly `electricity_pricing` plus 1/2/3-hour forecasts.
- Missing months are forecasted with `calendar_month_mean_overlap_scaled` and
  documented in `tools/dataset_docs/distillation_report.csv`.
- `tools/generate_iquitos_dataset.py` synchronizes names, areas, PV sizing
  (pvlib SAPM, SunPower SPR-315E), BESS sizing (Hesse 2017 method), EV charger
  profiles (50 files) and carbon intensity (0.671-0.790 kgCO2/kWh, RAGEI 2019).
- `dhw_demand` is non-zero only for B05 (Hotel, 614 kWh/day), B11 (Hospital,
  1200 kWh/day) and B12 (EsSalud, 780 kWh/day). All other buildings have
  dhw_demand=0 because tropical Iquitos (28-38 degC) does not require domestic
  hot water heating in commercial buildings without dedicated DHW devices.

Full pipeline documentation: `docs/dataset_construction_pipeline.md`.

The validated environment exposes 17 agents, EV actions/observations,
`state_dim=1856`, 31 bidirectional V2G EV actions and full CityLearn v2 KPI
tables.

## Training Observation Normalization

Training launchers normalize observations before they enter the MADRL backend.
This step is applied at the environment-wrapper layer, not by modifying the
dataset CSV files. The Iquitos dataset keeps physical units such as kWh, kgCO2,
prices, SOC percentages and charger states so CityLearn simulation, KPIs and
audits remain traceable to source data.

The common training adapter uses CityLearn's `NormalizedObservationWrapper` by
default. Temporal observations such as `month`, `day_type` and `hour` are
encoded cyclically, then active observations are min-max scaled to `[0, 1]`.
For the 17-building Iquitos EV schema, EV observations/actions remain exposed
and the validated CTDE state dimension is `state_dim=1856`.

Use `--raw-observations` only to reproduce legacy raw-input runs. The default
training behavior is normalized input:

```powershell
python -B CityLearn\scripts\train_citylearn_v3_matd3.py `
  --schema-path CityLearn\data\datasets\citylearn_iquitos_2023_2025\schema.json `
  --scenario E1
```

## Training Artifact Traceability

Completed MADRL runs use `data/` as the canonical artifact boundary. For each
`<OutputRoot>/<algorithm>/<scenario>_seed_<seed>/` directory, final evidence is
accepted from:

- `data/results.json`
- `data/training_summary.json`
- `data/artifact_audit.json`
- `data/checkpoint_manifest.json`
- `data/timeseries.csv`
- `data/trace.csv`
- `checkpoints/`
- `figures/figures_manifest.json`
- `figures/tables/`

Root-level mirrors such as `results.json`, `timeseries.csv` or
`checkpoint_manifest.json` are disabled by default to avoid duplicate or stale
traceability. They are available only through `--legacy-root-artifacts` for
legacy consumers. Cross-run copies under `statistical_comparison/` are also
disabled by default and require `--statistical-comparison-artifacts`.

`live_progress.json` is transient runtime state. The training writer removes it
after final artifacts are written, and the official monitor exits when
`official_full_status.json` reaches `completed` unless launched with
`-KeepOpenOnComplete`.

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
python -B CityLearn\scripts\check_citylearn_v3_training_ready.py `
  --strict `
  --schema-path CityLearn\data\datasets\citylearn_iquitos_2023_2025\schema.json `
  --scenario E1
python -B CityLearn\scripts\run_citylearn_v3_env_smoke.py `
  --schema-path CityLearn\data\datasets\citylearn_iquitos_2023_2025\schema.json `
  --scenario E1 `
  --episode-time-steps 4 `
  --steps 3
```

The readiness check verifies:

- CityLearn v3 builds the Iquitos 17-building + EV Dec-POMDP.
- The dataset exposes exactly 31 camioneta V2G loadpoints and 154 charge-only
  EV loadpoints.
- CTDE global state and local decentralized observations/actions are exposed.
- Full CityLearn v2 KPI tables are available.
- MARLlib imports and registers `citylearn_v3`.
- HAPPO, MASAC and MAAC official sources import in Python 3.9.
- MATD3 PyTorch imports through the `marlbenchmark/off-policy` backend.
- MATD3 legacy TensorFlow 1.x source is present as reference only; active
  training uses the PyTorch off-policy backend.
- `pip check` is reported as metadata consistency only unless
  `--require-pip-check` is explicitly requested. The validated stack keeps
  `numpy==1.23.5` for Ray/CityLearn compatibility.

## Training Launch Boundary

Full training must not be launched as part of file/documentation validation.
Use smoke checks first. The official 12-job chain is launched only after an
explicit user confirmation:

The launchers activate the project environment before spawning training jobs:
`VIRTUAL_ENV` is set to `.venv39-citylearn-v3`, that environment's `Scripts`
directory is prepended to `PATH`, and `PYTHONPATH` is set to the project root
plus `CityLearn/`. Each manifest records the resolved `python.exe`,
`virtual_env` and `pythonpath` under `active_project_environment`.

MASAC and MAAC use the official discrete-action backends, while CityLearn
buildings expose multi-dimensional continuous control actions. The adapter maps
each discrete policy output to a compact one-axis CityLearn action basis by
default (`--discrete-action-mode axis`). This preserves the official backend
interface without enumerating the exponential cartesian product of all actuator
bins, which is not memory-tractable for the 17-building EV schema. HAPPO and
MATD3 remain continuous-action backends.

Verify the launch environment without starting training:

```powershell
powershell -ExecutionPolicy Bypass -File CityLearn\scripts\launch_citylearn_v3_official_training.ps1 `
  -Scenario E1 `
  -EpisodeTimeSteps 4 `
  -Episodes 1 `
  -SchemaPath CityLearn\data\datasets\citylearn_iquitos_2023_2025\schema.json `
  -OutputRoot outputs\citylearn_v3_env_dryrun `
  -DryRun
```

```powershell
powershell -ExecutionPolicy Bypass -File CityLearn\scripts\launch_citylearn_v3_official_training.ps1 `
  -Scenario ALL `
  -Seed 0 `
  -EpisodeTimeSteps 8760 `
  -Episodes 5 `
  -SchemaPath CityLearn\data\datasets\citylearn_iquitos_2023_2025\schema.json `
  -OutputRoot outputs\citylearn_v3_madrl_iquitos_official_full_cuda_v1 `
  -TorchThreads 12 `
  -LiveProgressInterval 250 `
  -LiveOutput `
  -Cuda
```

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
