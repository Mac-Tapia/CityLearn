param(
    [string]$Scenario = "ALL",
    [int]$Seed = 0,
    [int]$EpisodeTimeSteps = 8760,
    [int]$Episodes = 5,
    [string]$OutputRoot = "outputs\citylearn_v3_madrl_official_full_cuda_v2",
    [int]$TorchThreads = 12,
    [switch]$Cuda = $true,
    [switch]$LiveOutput
)

$ErrorActionPreference = "Stop"

$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptPath "..\..")
$Python = Join-Path $ProjectRoot ".venv39-citylearn-v3\Scripts\python.exe"
$OutputRootPath = Join-Path $ProjectRoot $OutputRoot
$LogDir = Join-Path $OutputRootPath "logs"
$ManifestPath = Join-Path $OutputRootPath "official_full_manifest.json"
$StatusPath = Join-Path $OutputRootPath "official_full_status.json"
$TrainingConfigYaml = "CityLearn\configs\citylearn_v3_madrl_training.yaml"
$TrainingConfigJson = "CityLearn\configs\citylearn_v3_madrl_training.json"
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
            "--scenario", $scenarioName,
            "--seed", "$Seed",
            "--episode-time-steps", "$EpisodeTimeSteps",
            "--episodes", "$Episodes",
            "--num-env-steps", "$NumEnvSteps",
            "--hidden-size", "256",
            "--torch-threads", "$TorchThreads"
        ) + $CudaArgs + @(
            "--output-dir", (Join-Path $OutputRoot "happo")
        )
    }
    $jobs += [ordered]@{
        name = "masac"
        scenario = $scenarioName
        script = "CityLearn\scripts\train_citylearn_v3_masac.py"
        args = @(
            "--scenario", $scenarioName,
            "--seed", "$Seed",
            "--episode-time-steps", "$EpisodeTimeSteps",
            "--episodes", "$Episodes",
            "--action-bins", "3",
            "--buffer-size", "2"
        ) + $CudaArgs + @(
            "--output-dir", (Join-Path $OutputRoot "masac")
        )
    }
    $jobs += [ordered]@{
        name = "matd3"
        scenario = $scenarioName
        script = "CityLearn\scripts\train_citylearn_v3_matd3.py"
        args = @(
            "--scenario", $scenarioName,
            "--seed", "$Seed",
            "--episode-time-steps", "$EpisodeTimeSteps",
            "--episodes", "$Episodes",
            "--num-env-steps", "$NumEnvSteps",
            "--batch-size", "256",
            "--buffer-size", "10000",
            "--hidden-size", "256",
            "--train-interval", "100",
            "--num-random-episodes", "1"
        ) + $CudaArgs + @(
            "--output-dir", (Join-Path $OutputRoot "matd3")
        )
    }
    $jobs += [ordered]@{
        name = "maac"
        scenario = $scenarioName
        script = "CityLearn\scripts\train_citylearn_v3_maac.py"
        args = @(
            "--scenario", $scenarioName,
            "--seed", "$Seed",
            "--episode-time-steps", "$EpisodeTimeSteps",
            "--episodes", "$Episodes",
            "--action-bins", "3",
            "--batch-size", "256",
            "--buffer-length", "100000",
            "--steps-per-update", "100",
            "--num-updates", "4",
            "--hidden-size", "256",
            "--attend-heads", "4",
            "--pi-lr", "0.0003",
            "--q-lr", "0.001",
            "--tau", "0.005",
            "--gamma", "0.99"
        ) + $CudaArgs + @(
            "--output-dir", (Join-Path $OutputRoot "maac")
        )
    }
}

$manifest = [ordered]@{
    started_at = (Get-Date).ToString("o")
    completed_at = $null
    status = "running"
    dataset = "citylearn_challenge_2022_phase_all_plus_evs"
    schema_path = "CityLearn\data\datasets\citylearn_challenge_2022_phase_all_plus_evs\schema.json"
    scenario = $Scenario
    scenarios = $ScenarioList
    seed = $Seed
    episode_time_steps = $EpisodeTimeSteps
    episodes = $Episodes
    num_env_steps = $NumEnvSteps
    torch = "torch 2.8.0+cu126"
    cuda = [bool]$Cuda
    execution = "sequential"
    training_config = [ordered]@{
        yaml = $TrainingConfigYaml
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

foreach ($job in $jobs) {
    $logPath = Join-Path $LogDir "$($job.scenario)_$($job.name).log"
    $errPath = Join-Path $LogDir "$($job.scenario)_$($job.name).stderr.log"
    $commandArgs = @("-B", $job.script) + $job.args
    $startedAt = Get-Date

    $jobRecord = [ordered]@{
        name = $job.name
        scenario = $job.scenario
        script = $job.script
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
            Write-Host "=== CityLearn v3 MADRL: $($job.name.ToUpper()) | $($job.scenario) ===" -ForegroundColor Cyan
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

    $jobRecord.completed_at = $completedAt.ToString("o")
    $jobRecord.exit_code = $exitCode
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $StatusPath -Encoding UTF8

    if ($exitCode -ne 0) {
        $manifest.status = "failed"
        $manifest.completed_at = $completedAt.ToString("o")
        $manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $ManifestPath -Encoding UTF8
        $manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $StatusPath -Encoding UTF8
        exit $exitCode
    }
}

$manifest.status = "completed"
$manifest.completed_at = (Get-Date).ToString("o")
$manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $ManifestPath -Encoding UTF8
$manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $StatusPath -Encoding UTF8
