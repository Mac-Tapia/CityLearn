param(
    [string]$Scenario = "ALL",
    [int]$Seed = 0,
    [int]$EpisodeTimeSteps = 8760,
    [int]$Episodes = 5,
    [string]$OutputRoot = "outputs\citylearn_v3_madrl_iquitos",
    [int]$TorchThreads = 12,
    [int]$LiveProgressInterval = 250,
    [switch]$Cuda = $true,
    [switch]$LiveOutput
)

$ErrorActionPreference = "Stop"

$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptPath "..\..")
$Python = Join-Path $ProjectRoot ".venv39-citylearn-v3\Scripts\python.exe"
$OutputRootPath = Join-Path $ProjectRoot $OutputRoot
$LogDir = Join-Path $OutputRootPath "logs"
$ManifestPath = Join-Path $OutputRootPath "iquitos_manifest.json"
$StatusPath = Join-Path $OutputRootPath "iquitos_status.json"
$TrainingConfigJson = "CityLearn\configs\citylearn_v3_madrl_training.json"
$SchemaPath = "CityLearn\data\datasets\citylearn_iquitos_2023_2025\schema.json"
$DatasetName = "citylearn_iquitos_2023_2025"
$NumEnvSteps = $EpisodeTimeSteps * $Episodes
$CudaArgs = if ($Cuda) { @("--cuda") } else { @() }
$ScenarioList = if ($Scenario.ToUpperInvariant() -in @("ALL", "TODOS", "3EJES")) {
    @("E1", "E2", "E3")
}
else {
    @($Scenario.ToUpperInvariant())
}

foreach ($scenarioName in $ScenarioList) {
    if ($scenarioName -notin @("E1", "E2", "E3")) {
        throw "Unknown scenario: $scenarioName. Use E1, E2, E3 or ALL."
    }
}

New-Item -ItemType Directory -Force -Path $OutputRootPath | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $ProjectRoot

$jobs = @()
foreach ($scenarioName in $ScenarioList) {
    $jobs += [ordered]@{
        name = "happo"
        scenario = $scenarioName
        script = "CityLearn\scripts\train_citylearn_v3_happo.py"
        args = @(
            "--schema-path", $SchemaPath,
            "--scenario", $scenarioName,
            "--seed", "$Seed",
            "--episode-time-steps", "$EpisodeTimeSteps",
            "--episodes", "$Episodes",
            "--num-env-steps", "$NumEnvSteps",
            "--hidden-size", "384",
            "--torch-threads", "$TorchThreads",
            "--n-rollout-threads", "1",
            "--log-interval", "1",
            "--eval-interval", "1",
            "--live-progress-interval", "$LiveProgressInterval"
        ) + $CudaArgs + @(
            "--output-dir", (Join-Path $OutputRoot "happo")
        )
    }
    $jobs += [ordered]@{
        name = "masac"
        scenario = $scenarioName
        script = "CityLearn\scripts\train_citylearn_v3_masac.py"
        args = @(
            "--schema-path", $SchemaPath,
            "--scenario", $scenarioName,
            "--seed", "$Seed",
            "--episode-time-steps", "$EpisodeTimeSteps",
            "--episodes", "$Episodes",
            "--action-bins", "3",
            "--buffer-size", "2",
            "--critic-batch-size", "1",
            "--critic-train-steps", "1",
            "--actor-sample-times", "5",
            "--rnn-hidden-dim", "64",
            "--qmix-hidden-dim", "32",
            "--hyper-hidden-dim", "64",
            "--live-progress-interval", "$LiveProgressInterval"
        ) + $CudaArgs + @(
            "--output-dir", (Join-Path $OutputRoot "masac")
        )
    }
    $jobs += [ordered]@{
        name = "matd3"
        scenario = $scenarioName
        script = "CityLearn\scripts\train_citylearn_v3_matd3.py"
        args = @(
            "--schema-path", $SchemaPath,
            "--scenario", $scenarioName,
            "--seed", "$Seed",
            "--episode-time-steps", "$EpisodeTimeSteps",
            "--episodes", "$Episodes",
            "--num-env-steps", "$NumEnvSteps",
            "--batch-size", "512",
            "--buffer-size", "50000",
            "--hidden-size", "384",
            "--train-interval", "100",
            "--num-random-episodes", "1",
            "--live-progress-interval", "$LiveProgressInterval"
        ) + $CudaArgs + @(
            "--output-dir", (Join-Path $OutputRoot "matd3")
        )
    }
    $jobs += [ordered]@{
        name = "maac"
        scenario = $scenarioName
        script = "CityLearn\scripts\train_citylearn_v3_maac.py"
        args = @(
            "--schema-path", $SchemaPath,
            "--scenario", $scenarioName,
            "--seed", "$Seed",
            "--episode-time-steps", "$EpisodeTimeSteps",
            "--episodes", "$Episodes",
            "--action-bins", "3",
            "--batch-size", "512",
            "--buffer-length", "200000",
            "--steps-per-update", "250",
            "--num-updates", "8",
            "--hidden-size", "384",
            "--attend-heads", "4",
            "--pi-lr", "0.0003",
            "--q-lr", "0.001",
            "--tau", "0.005",
            "--gamma", "0.99",
            "--live-progress-interval", "$LiveProgressInterval"
        ) + $CudaArgs + @(
            "--output-dir", (Join-Path $OutputRoot "maac")
        )
    }
}

$manifest = [ordered]@{
    started_at = (Get-Date).ToString("o")
    completed_at = $null
    status = "running"
    dataset = $DatasetName
    schema_path = $SchemaPath
    scenario = $Scenario
    scenarios = $ScenarioList
    seed = $Seed
    episode_time_steps = $EpisodeTimeSteps
    episodes = $Episodes
    num_env_steps = $NumEnvSteps
    torch = "torch 2.8.0+cu126"
    cuda = [bool]$Cuda
    execution = "sequential"
    dataset_info = [ordered]@{
        name = $DatasetName
        buildings = 17
        total_hours = 26304
        years = @(2023, 2024, 2025)
        location = "Iquitos, Loreto, Peru"
        lat = -3.7491
        lon = -73.2538
        grid_type = "sistema_aislado_diesel_electro_oriente"
        ev_chargers = 50
    }
    gpu_optimization = [ordered]@{
        enabled = $true
        live_progress_interval = $LiveProgressInterval
        strategy = "larger batches, grouped updates, reduced live-progress IO"
        note = "Environment execution sequential for reproducible Dec-POMDP rollouts."
    }
    training_config = [ordered]@{
        json = $TrainingConfigJson
        schema_version = 2
    }
    reward = [ordered]@{
        function = "citylearn.reward_function.CityLearnV3MADRLRewardFunction"
        aggregation = "team_mean"
        not_using_marl_base_weights = $true
    }
    output_root = $OutputRoot
    jobs = @()
}

$manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $ManifestPath -Encoding UTF8
$manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $StatusPath -Encoding UTF8

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  CityLearn v3 MADRL — Dataset: citylearn_iquitos_2023_2025" -ForegroundColor Cyan
Write-Host "  17 edificios | 26 304 horas | 3 anos (2023-2025)" -ForegroundColor Cyan
Write-Host "  Escenarios: $($ScenarioList -join ', ') | Seed: $Seed" -ForegroundColor Cyan
Write-Host "  Salida: $OutputRoot" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

foreach ($job in $jobs) {
    $logPath = Join-Path $LogDir "$($job.scenario)_$($job.name).log"
    $errPath = Join-Path $LogDir "$($job.scenario)_$($job.name).stderr.log"
    $commandArgs = @("-B", $job.script) + $job.args
    $startedAt = Get-Date

    Write-Host ">>> [$($job.scenario)] $($job.name.ToUpper()) iniciando..." -ForegroundColor Yellow

    $jobRecord = [ordered]@{
        name = $job.name
        scenario = $job.scenario
        script = $job.script
        dataset = $DatasetName
        schema_path = $SchemaPath
        started_at = $startedAt.ToString("o")
        completed_at = $null
        exit_code = $null
        log = $logPath
        stderr_log = $errPath
        output_dir = Join-Path $OutputRoot "$($job.name)\$($job.scenario)_seed_$Seed"
        command = "$Python " + ($commandArgs -join " ")
    }

    $manifest.jobs += $jobRecord
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $StatusPath -Encoding UTF8

    if ($LiveOutput) {
        Push-Location $ProjectRoot
        try {
            Write-Host ""
            Write-Host "=== CityLearn v3 MADRL Iquitos: $($job.name.ToUpper()) | $($job.scenario) ===" -ForegroundColor Cyan
            Write-Host "$Python $($commandArgs -join ' ')" -ForegroundColor DarkGray
            & $Python @commandArgs 2>&1 | Tee-Object -FilePath $logPath
            $exitCode = $LASTEXITCODE
        }
        finally {
            Pop-Location
        }

        if (-not (Test-Path $errPath)) {
            New-Item -ItemType File -Force -Path $errPath | Out-Null
        }
    }
    else {
        $process = Start-Process `
            -FilePath $Python `
            -ArgumentList $commandArgs `
            -WorkingDirectory $ProjectRoot `
            -RedirectStandardOutput $logPath `
            -RedirectStandardError $errPath `
            -WindowStyle Hidden `
            -Wait `
            -PassThru
        $exitCode = $process.ExitCode
    }
    $completedAt = Get-Date
    $duration = ($completedAt - $startedAt).TotalMinutes

    $jobRecord.completed_at = $completedAt.ToString("o")
    $jobRecord.exit_code = $exitCode
    $jobRecord.duration_minutes = [math]::Round($duration, 2)
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $StatusPath -Encoding UTF8

    if ($exitCode -ne 0) {
        Write-Host "!!! [$($job.scenario)] $($job.name.ToUpper()) FALLO (exit=$exitCode) — ver $errPath" -ForegroundColor Red
        $manifest.status = "failed"
        $manifest.completed_at = $completedAt.ToString("o")
        $manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $ManifestPath -Encoding UTF8
        $manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $StatusPath -Encoding UTF8
        exit $exitCode
    }

    Write-Host "    [$($job.scenario)] $($job.name.ToUpper()) completado en $([math]::Round($duration,1)) min" -ForegroundColor Green
}

$manifest.status = "completed"
$manifest.completed_at = (Get-Date).ToString("o")
$manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $ManifestPath -Encoding UTF8
$manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $StatusPath -Encoding UTF8

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  ENTRENAMIENTO IQUITOS COMPLETADO" -ForegroundColor Green
Write-Host "  Resultados en: $OutputRootPath" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
