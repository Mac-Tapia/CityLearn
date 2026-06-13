param(
    [string]$Scenario = "ALL",
    [int]$Seed = 0,
    [int]$EpisodeTimeSteps = 8760,
    [int]$Episodes = 5,
    [string]$OutputRoot = "outputs\citylearn_v3_madrl_iquitos",
    [int]$TorchThreads = 12,
    [int]$LiveProgressInterval = 250,
    [ValidateSet("full", "efficient", "minimal")]
    [string]$ArtifactProfile = "efficient",
    [int]$TraceRecordInterval = 10,
    [ValidateSet("full", "compact")]
    [string]$TraceDetail = "compact",
    [bool]$ParallelScenarios = $true,
    [ValidateRange(1, 16)]
    [int]$MaxConcurrentScenarioJobs = 2,
    [ValidateRange(1, 16)]
    [int]$MaxConcurrentHeavyJobs = 1,
    [ValidateSet("local4060_fast", "local4060", "balanced", "conservative", "aws")]
    [string]$GpuProfile = "local4060_fast",
    [double]$MaxGpuVramGib = 0,
    [double]$GpuVramReserveGib = 1.5,
    [ValidateRange(0.0, 1.0)]
    [double]$CudaMemoryFraction = 0,
    [switch]$AllowGpuOversubscription,
    [switch]$Cuda = $true,
    [switch]$LiveOutput,
    [switch]$DryRun,
    [switch]$SkipCompleted
)

$ErrorActionPreference = "Stop"

function Get-DedicatedCudaGpuInfo {
    try {
        $line = & nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits 2>$null |
            Select-Object -First 1
        if ([string]::IsNullOrWhiteSpace($line)) {
            return $null
        }

        $parts = @($line -split "," | ForEach-Object { $_.Trim() })
        if ($parts.Count -lt 4) {
            return $null
        }

        $totalMiB = [double]$parts[1]
        $freeMiB = [double]$parts[2]
        return [pscustomobject]@{
            name = $parts[0]
            memory_total_mib = $totalMiB
            memory_free_mib = $freeMiB
            dedicated_vram_gib = [math]::Round($totalMiB / 1024.0, 2)
            free_vram_gib = [math]::Round($freeMiB / 1024.0, 2)
            driver_version = $parts[3]
            source = "nvidia-smi dedicated memory, not Windows shared GPU memory"
        }
    }
    catch {
        return $null
    }
}

$DedicatedGpuInfo = if ($Cuda) { Get-DedicatedCudaGpuInfo } else { $null }
$DetectedDedicatedVramGib = if ($null -ne $DedicatedGpuInfo) { [double]$DedicatedGpuInfo.dedicated_vram_gib } else { $null }
$EffectiveMaxGpuVramGib = $null
if ($Cuda) {
    if ($MaxGpuVramGib -gt 0) {
        $EffectiveMaxGpuVramGib = [double]$MaxGpuVramGib
        if ($null -ne $DetectedDedicatedVramGib) {
            $EffectiveMaxGpuVramGib = [Math]::Min($EffectiveMaxGpuVramGib, $DetectedDedicatedVramGib)
        }
    }
    elseif ($null -ne $DetectedDedicatedVramGib) {
        $EffectiveMaxGpuVramGib = $DetectedDedicatedVramGib
    }
}

$EffectiveCudaMemoryFraction = $null
if ($Cuda) {
    if ($CudaMemoryFraction -gt 0) {
        $EffectiveCudaMemoryFraction = [Math]::Min(1.0, [Math]::Max(0.1, [double]$CudaMemoryFraction))
    }
    elseif ($null -ne $DetectedDedicatedVramGib -and $DetectedDedicatedVramGib -gt 0) {
        $usableGpuVramGib = [Math]::Max(1.0, ([double]$EffectiveMaxGpuVramGib - [double]$GpuVramReserveGib))
        $EffectiveCudaMemoryFraction = [Math]::Round([Math]::Min(0.92, [Math]::Max(0.25, $usableGpuVramGib / $DetectedDedicatedVramGib)), 3)
    }
}

$IsLocal8GbGpu = [bool]($Cuda -and $null -ne $DetectedDedicatedVramGib -and $DetectedDedicatedVramGib -le 8.5)
$Local8GbConcurrencyAdjusted = $false
$Local8GbConcurrencyNote = $null
if ($IsLocal8GbGpu -and -not $AllowGpuOversubscription) {
    if ($MaxConcurrentScenarioJobs -gt 2) {
        $MaxConcurrentScenarioJobs = 2
        $Local8GbConcurrencyAdjusted = $true
    }
    if ($MaxConcurrentHeavyJobs -gt 1) {
        $MaxConcurrentHeavyJobs = 1
        $Local8GbConcurrencyAdjusted = $true
    }
    $Local8GbConcurrencyNote = "8GB VRAM profile allows up to 2 concurrent scenario jobs and keeps MASAC/MAAC heavy stages at 1; visible monitoring is parallel-safe, while LiveOutput=true is sequential debug display."
}

if ($ArtifactProfile -eq "full") {
    if (-not $PSBoundParameters.ContainsKey("TraceRecordInterval")) { $TraceRecordInterval = 1 }
    if (-not $PSBoundParameters.ContainsKey("TraceDetail")) { $TraceDetail = "full" }
}
else {
    if (-not $PSBoundParameters.ContainsKey("TraceRecordInterval")) { $TraceRecordInterval = 10 }
    if (-not $PSBoundParameters.ContainsKey("TraceDetail")) { $TraceDetail = "compact" }
}

$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptPath "..\..")).Path
$VenvRoot = Join-Path $ProjectRoot ".venv39-citylearn-v3"
$VenvScripts = Join-Path $VenvRoot "Scripts"
$Python = Join-Path $VenvScripts "python.exe"
$OutputRootPath = Join-Path $ProjectRoot $OutputRoot
$LogDir = Join-Path $OutputRootPath "logs"
$ManifestPath = Join-Path $OutputRootPath "iquitos_manifest.json"
$StatusPath = Join-Path $OutputRootPath "iquitos_status.json"
$TrainingConfigJson = "CityLearn\configs\citylearn_v3_madrl_training.json"
$SchemaPath = "CityLearn\data\datasets\citylearn_iquitos_2023_2025\schema.json"
$DatasetName = "citylearn_iquitos_2023_2025"
$NumEnvSteps = $EpisodeTimeSteps * $Episodes
$CudaArgs = if ($Cuda) { @("--cuda") } else { @() }
$ArtifactArgs = @(
    "--artifact-profile", "$ArtifactProfile",
    "--trace-record-interval", "$TraceRecordInterval",
    "--trace-detail", "$TraceDetail"
)

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python environment not found: $Python"
}

$VenvRoot = (Resolve-Path -LiteralPath $VenvRoot).Path
$VenvScripts = (Resolve-Path -LiteralPath $VenvScripts).Path
$Python = (Resolve-Path -LiteralPath $Python).Path
$env:VIRTUAL_ENV = $VenvRoot
$pathEntries = @($VenvScripts) + @(
    ($env:Path -split [System.IO.Path]::PathSeparator) |
        Where-Object { $_ -and ($_.TrimEnd('\') -ine $VenvScripts.TrimEnd('\')) }
)
$env:Path = ($pathEntries -join [System.IO.Path]::PathSeparator)
$env:PYTHONPATH = @(
    $ProjectRoot
    (Join-Path $ProjectRoot "CityLearn")
) -join [System.IO.Path]::PathSeparator
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
if ($Cuda) {
    $isWindowsOs = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
        [System.Runtime.InteropServices.OSPlatform]::Windows
    )
    $defaultCudaAllocConf = if ($isWindowsOs) { "max_split_size_mb:128" } else { "expandable_segments:True,max_split_size_mb:128" }

    if ([string]::IsNullOrWhiteSpace($env:PYTORCH_CUDA_ALLOC_CONF)) {
        $env:PYTORCH_CUDA_ALLOC_CONF = $defaultCudaAllocConf
    }
    elseif ($isWindowsOs -and $env:PYTORCH_CUDA_ALLOC_CONF -match "expandable_segments") {
        $parts = @(
            $env:PYTORCH_CUDA_ALLOC_CONF -split "," |
                ForEach-Object { $_.Trim() } |
                Where-Object { $_ -and ($_ -notmatch "^expandable_segments") }
        )
        $env:PYTORCH_CUDA_ALLOC_CONF = if ($parts.Count -gt 0) { $parts -join "," } else { "max_split_size_mb:128" }
    }
}

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

$EffectiveParallelScenarios = [bool]$ParallelScenarios -and (-not [bool]$LiveOutput) -and ($ScenarioList.Count -gt 1)
if ([bool]$ParallelScenarios -and [bool]$LiveOutput -and $ScenarioList.Count -gt 1) {
    Write-Host "ParallelScenarios requested, but LiveOutput requires sequential display. Running sequential live mode." -ForegroundColor Yellow
}

$MasacMaxReplayBufferGib = if ($IsLocal8GbGpu) { 3.5 } else { 8 }
$HappoHiddenSize = if ($IsLocal8GbGpu) { 256 } else { 384 }
$Matd3BatchSize = if ($IsLocal8GbGpu) { 256 } else { 512 }
$Matd3BufferSize = if ($IsLocal8GbGpu) { 4096 } else { 50000 }
$Matd3HiddenSize = if ($IsLocal8GbGpu) { 256 } else { 384 }
$MaacBatchSize = if ($IsLocal8GbGpu) { 256 } else { 512 }
$MaacBufferLength = if ($IsLocal8GbGpu) { 50000 } else { 200000 }
$MaacHiddenSize = if ($IsLocal8GbGpu) { 256 } else { 384 }
$MaacNumUpdates = if ($IsLocal8GbGpu) { 4 } else { 8 }
$CudaMemoryArgs = if ($null -ne $EffectiveCudaMemoryFraction) {
    @("--cuda-memory-fraction", "$EffectiveCudaMemoryFraction")
}
else {
    @()
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
            "--hidden-size", "$HappoHiddenSize",
            "--torch-threads", "$TorchThreads",
            "--n-rollout-threads", "1",
            "--log-interval", "1",
            "--eval-interval", "1",
            "--live-progress-interval", "$LiveProgressInterval",
            "--gpu-profile", "$GpuProfile"
        ) + $CudaArgs + $CudaMemoryArgs + $ArtifactArgs + @(
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
            "--discrete-action-mode", "axis",
            "--max-replay-buffer-gib", "$MasacMaxReplayBufferGib",
            "--buffer-size", "2",
            "--critic-batch-size", "1",
            "--critic-train-steps", "1",
            "--actor-sample-times", "5",
            "--rnn-hidden-dim", "64",
            "--qmix-hidden-dim", "32",
            "--hyper-hidden-dim", "64",
            "--live-progress-interval", "$LiveProgressInterval",
            "--gpu-profile", "$GpuProfile"
        ) + $CudaArgs + $CudaMemoryArgs + $ArtifactArgs + @(
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
            "--batch-size", "$Matd3BatchSize",
            "--buffer-size", "$Matd3BufferSize",
            "--hidden-size", "$Matd3HiddenSize",
            "--train-interval", "100",
            "--num-random-episodes", "1",
            "--live-progress-interval", "$LiveProgressInterval",
            "--gpu-profile", "$GpuProfile"
        ) + $CudaArgs + $CudaMemoryArgs + $ArtifactArgs + @(
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
            "--discrete-action-mode", "axis",
            "--max-discrete-actions", "512",
            "--batch-size", "$MaacBatchSize",
            "--buffer-length", "$MaacBufferLength",
            "--steps-per-update", "250",
            "--num-updates", "$MaacNumUpdates",
            "--hidden-size", "$MaacHiddenSize",
            "--attend-heads", "4",
            "--pi-lr", "0.0003",
            "--q-lr", "0.001",
            "--tau", "0.005",
            "--gamma", "0.99",
            "--live-progress-interval", "$LiveProgressInterval",
            "--gpu-profile", "$GpuProfile"
        ) + $CudaArgs + $CudaMemoryArgs + $ArtifactArgs + @(
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
    execution = if ($EffectiveParallelScenarios) { "parallel_scenarios_by_algorithm" } else { "sequential" }
    parallelization = [ordered]@{
        requested = [bool]$ParallelScenarios
        effective = [bool]$EffectiveParallelScenarios
        max_concurrent_scenario_jobs = $MaxConcurrentScenarioJobs
        max_concurrent_heavy_jobs = $MaxConcurrentHeavyJobs
        heavy_algorithms = @("masac", "maac")
        strategy = "Run the same algorithm across scenarios in parallel; cap MASAC/MAAC lower because replay buffers and attention updates are heavier."
        local_8gb_concurrency_adjusted = [bool]$Local8GbConcurrencyAdjusted
        local_8gb_concurrency_note = $Local8GbConcurrencyNote
        disabled_reason = if (-not [bool]$ParallelScenarios) { "not_requested" } elseif ([bool]$LiveOutput) { "live_output_requires_sequential_display" } elseif ($ScenarioList.Count -le 1) { "single_scenario" } else { $null }
    }
    active_project_environment = [ordered]@{
        python = $Python
        virtual_env = $env:VIRTUAL_ENV
        pythonpath = $env:PYTHONPATH
    }
    dataset_info = [ordered]@{
        name = $DatasetName
        buildings = 17
        total_hours = 26304
        years = @(2023, 2024, 2025)
        location = "Iquitos, Loreto, Peru"
        lat = -3.7491
        lon = -73.2538
        grid_type = "sistema_aislado_diesel_electro_oriente"
        ev_chargers = 185
    }
    gpu_optimization = [ordered]@{
        enabled = $true
        profile = $GpuProfile
        detected_gpu = if ($null -ne $DedicatedGpuInfo) { "$($DedicatedGpuInfo.name), $($DedicatedGpuInfo.memory_total_mib) MiB dedicated, driver $($DedicatedGpuInfo.driver_version)" } else { $null }
        dedicated_gpu = $DedicatedGpuInfo
        max_gpu_vram_gib_requested = $MaxGpuVramGib
        max_gpu_vram_gib_effective = $EffectiveMaxGpuVramGib
        gpu_vram_reserve_gib = $GpuVramReserveGib
        cuda_memory_fraction_requested = $CudaMemoryFraction
        cuda_memory_fraction_effective = $EffectiveCudaMemoryFraction
        local_8gb_safety_mode = $IsLocal8GbGpu
        allow_gpu_oversubscription = [bool]$AllowGpuOversubscription
        cuda_device_order = $env:CUDA_DEVICE_ORDER
        pytorch_cuda_alloc_conf = $env:PYTORCH_CUDA_ALLOC_CONF
        live_progress_interval = $LiveProgressInterval
        strategy = if ($IsLocal8GbGpu) { "8GB dedicated VRAM profile: up to 2 scenario jobs, MASAC/MAAC heavy stages capped at 1, capped per-process memory, smaller buffers." } else { "larger batches, grouped updates, reduced live-progress IO" }
        note = "Environment execution sequential for reproducible Dec-POMDP rollouts."
    }
    algorithm_resource_limits = [ordered]@{
        happo_hidden_size = $HappoHiddenSize
        masac_max_replay_buffer_gib = $MasacMaxReplayBufferGib
        matd3_batch_size = $Matd3BatchSize
        matd3_buffer_size = $Matd3BufferSize
        matd3_hidden_size = $Matd3HiddenSize
        maac_batch_size = $MaacBatchSize
        maac_buffer_length = $MaacBufferLength
        maac_hidden_size = $MaacHiddenSize
        maac_num_updates = $MaacNumUpdates
    }
    training_config = [ordered]@{
        json = $TrainingConfigJson
        schema_version = 2
    }
    artifact_optimization = [ordered]@{
        profile = $ArtifactProfile
        trace_record_interval = $TraceRecordInterval
        trace_detail = $TraceDetail
        root_trace_csv = ($ArtifactProfile -eq "full")
        statistical_trace_copy = ($ArtifactProfile -eq "full")
        note = "efficient/minimal reduce per-agent trace generation and avoid duplicate heavy trace CSV mirrors."
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
Write-Host "  CityLearn v3 MADRL - Dataset: citylearn_iquitos_2023_2025" -ForegroundColor Cyan
Write-Host "  17 edificios | 26 304 horas | 3 anos (2023-2025)" -ForegroundColor Cyan
Write-Host "  Escenarios: $($ScenarioList -join ', ') | Seed: $Seed" -ForegroundColor Cyan
Write-Host "  Salida: $OutputRoot" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

if ($DryRun) {
    $manifest.status = "dry_run"
    $manifest.completed_at = (Get-Date).ToString("o")
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $ManifestPath -Encoding UTF8
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $StatusPath -Encoding UTF8
    Write-Host "Dry run completed. Training was not started." -ForegroundColor Yellow
    Write-Host "Python: $Python" -ForegroundColor Cyan
    Write-Host "VIRTUAL_ENV: $env:VIRTUAL_ENV" -ForegroundColor Cyan
    Write-Host "Manifest: $ManifestPath"
    exit 0
}

function Resolve-TrainingExitCode {
    param(
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$Process,

        [Parameter(Mandatory = $true)]
        [string]$JobOutputDir
    )

    $Process.WaitForExit()
    $Process.Refresh()
    $exitCode = $Process.ExitCode

    if ($null -eq $exitCode) {
        $resultsPath = Join-Path $ProjectRoot (Join-Path $JobOutputDir "data\results.json")
        if (Test-Path -LiteralPath $resultsPath) {
            return 0
        }

        Write-Warning "Process ExitCode was empty after completion and results.json was not found: $resultsPath"
        return 1
    }

    return [int]$exitCode
}

function Wait-TrainingRam {
    param(
        [string]$Label,
        [double]$MinFreeGB = 1.5
    )

    $ramCheckInterval = 30
    $ramChecks = 0
    do {
        $freeGB = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 2)
        if ($freeGB -lt $MinFreeGB) {
            if ($ramChecks -eq 0) {
                Write-Host ""
                Write-Host "  RAM libre insuficiente para ${Label}: ${freeGB} GB < ${MinFreeGB} GB requeridos." -ForegroundColor Yellow
                Write-Host "  Reintentando cada ${ramCheckInterval}s..." -ForegroundColor DarkGray
            }
            $ramChecks++
            Start-Sleep -Seconds $ramCheckInterval
        }
    } while ($freeGB -lt $MinFreeGB)

    if ($ramChecks -gt 0) {
        Write-Host "  RAM libre OK: ${freeGB} GB - continuando con ${Label}." -ForegroundColor Green
    }
}

function Add-SkippedTrainingJobRecord {
    param(
        [Parameter(Mandatory = $true)]
        $Job
    )

    $jobOutputDir = Join-Path $OutputRoot "$($Job.name)\$($Job.scenario)_seed_$Seed"
    $skippedRecord = [ordered]@{
        name = $Job.name
        scenario = $Job.scenario
        script = $Job.script
        dataset = $DatasetName
        schema_path = $SchemaPath
        started_at = "skipped"
        completed_at = (Get-Date).ToString("o")
        exit_code = 0
        log = $null
        stderr_log = $null
        output_dir = $jobOutputDir
        skipped = $true
    }
    $script:manifest.jobs += $skippedRecord
    $script:manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $StatusPath -Encoding UTF8
}

function Test-TrainingJobCompleted {
    param(
        [Parameter(Mandatory = $true)]
        $Job
    )

    $jobOutputDir = Join-Path $OutputRoot "$($Job.name)\$($Job.scenario)_seed_$Seed"
    $jobResultsPath = Join-Path $ProjectRoot (Join-Path $jobOutputDir "data\results.json")
    return (Test-Path -LiteralPath $jobResultsPath)
}

function Start-ParallelTrainingJob {
    param(
        [Parameter(Mandatory = $true)]
        $Job
    )

    $label = "$($Job.name.ToUpper())/$($Job.scenario)"
    Wait-TrainingRam -Label $label

    $logPath = Join-Path $LogDir "$($Job.scenario)_$($Job.name).log"
    $errPath = Join-Path $LogDir "$($Job.scenario)_$($Job.name).stderr.log"
    $commandArgs = @("-B", $Job.script) + $Job.args
    $startedAt = Get-Date

    $jobRecord = [ordered]@{
        name = $Job.name
        scenario = $Job.scenario
        script = $Job.script
        dataset = $DatasetName
        schema_path = $SchemaPath
        started_at = $startedAt.ToString("o")
        completed_at = $null
        exit_code = $null
        log = $logPath
        stderr_log = $errPath
        output_dir = Join-Path $OutputRoot "$($Job.name)\$($Job.scenario)_seed_$Seed"
        command = "$Python " + ($commandArgs -join " ")
        parallel_stage = $Job.name
    }

    $script:manifest.jobs += $jobRecord
    $script:manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $StatusPath -Encoding UTF8

    $env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = "1"
    $env:PYTHONUNBUFFERED = "1"
    $process = Start-Process `
        -FilePath $Python `
        -ArgumentList $commandArgs `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $logPath `
        -RedirectStandardError $errPath `
        -WindowStyle Hidden `
        -PassThru

    Write-Host ("  START {0,-9} PID={1} log={2}" -f $label, $process.Id, $logPath) -ForegroundColor Cyan

    return [pscustomobject]@{
        Job = $Job
        Record = $jobRecord
        Process = $process
        StartedAt = $startedAt
        LogPath = $logPath
        ErrPath = $errPath
    }
}

function Complete-ParallelTrainingJob {
    param(
        [Parameter(Mandatory = $true)]
        $Run
    )

    $exitCode = Resolve-TrainingExitCode -Process $Run.Process -JobOutputDir $Run.Record.output_dir
    $completedAt = Get-Date
    $duration = ($completedAt - $Run.StartedAt).TotalMinutes
    $Run.Record.completed_at = $completedAt.ToString("o")
    $Run.Record.exit_code = $exitCode
    $Run.Record.duration_minutes = [math]::Round($duration, 2)
    $script:manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $StatusPath -Encoding UTF8

    if ($exitCode -eq 0) {
        Write-Host ("  DONE  {0,-9} exit=0 duration={1:N1} min" -f "$($Run.Job.name.ToUpper())/$($Run.Job.scenario)", $duration) -ForegroundColor Green
    }
    else {
        Write-Host ("  FAIL  {0,-9} exit={1} stderr={2}" -f "$($Run.Job.name.ToUpper())/$($Run.Job.scenario)", $exitCode, $Run.ErrPath) -ForegroundColor Red
    }

    return [int]$exitCode
}

function Invoke-ParallelScenarioStage {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Algorithm,

        [Parameter(Mandatory = $true)]
        [array]$StageJobs,

        [Parameter(Mandatory = $true)]
        [int]$MaxConcurrent
    )

    if ($StageJobs.Count -eq 0) {
        return
    }

    Write-Host ""
    Write-Host ("=== Parallel stage: {0} | jobs={1} | max_concurrent={2} ===" -f $Algorithm.ToUpper(), $StageJobs.Count, $MaxConcurrent) -ForegroundColor Magenta
    $pending = [System.Collections.Queue]::new()
    foreach ($stageJob in $StageJobs) {
        $pending.Enqueue($stageJob)
    }
    $running = @()

    while ($pending.Count -gt 0 -or $running.Count -gt 0) {
        while ($pending.Count -gt 0 -and $running.Count -lt $MaxConcurrent) {
            $nextJob = $pending.Dequeue()
            if ($SkipCompleted -and (Test-TrainingJobCompleted -Job $nextJob)) {
                Write-Host ("  SKIP  {0}/{1} already has data/results.json" -f $nextJob.name.ToUpper(), $nextJob.scenario) -ForegroundColor Yellow
                Add-SkippedTrainingJobRecord -Job $nextJob
                continue
            }

            $running += Start-ParallelTrainingJob -Job $nextJob
        }

        if ($running.Count -eq 0) {
            continue
        }

        Start-Sleep -Seconds 5
        $stillRunning = @()
        foreach ($run in $running) {
            try { $run.Process.Refresh() } catch {}
            if ($run.Process.HasExited) {
                $exitCode = Complete-ParallelTrainingJob -Run $run
                if ($exitCode -ne 0) {
                    $script:manifest.status = "failed"
                    $script:manifest.completed_at = (Get-Date).ToString("o")
                    $script:manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $ManifestPath -Encoding UTF8
                    $script:manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $StatusPath -Encoding UTF8
                    exit $exitCode
                }
            }
            else {
                $stillRunning += $run
            }
        }
        $running = $stillRunning
    }
}

if ($EffectiveParallelScenarios) {
    Write-Host ""
    Write-Host "Running optimized parallel-scenario schedule. Use -LiveOutput for sequential rich display." -ForegroundColor Green
    foreach ($algorithmName in @("happo", "masac", "matd3", "maac")) {
        $stageJobs = @($jobs | Where-Object { $_.name -eq $algorithmName })
        $stageMaxConcurrent = if ($algorithmName -in @("masac", "maac")) {
            [Math]::Min($MaxConcurrentScenarioJobs, $MaxConcurrentHeavyJobs)
        }
        else {
            $MaxConcurrentScenarioJobs
        }
        Invoke-ParallelScenarioStage -Algorithm $algorithmName -StageJobs $stageJobs -MaxConcurrent $stageMaxConcurrent
    }

    $manifest.status = "completed"
    $manifest.completed_at = (Get-Date).ToString("o")
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $ManifestPath -Encoding UTF8
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $StatusPath -Encoding UTF8
    exit 0
}

foreach ($job in $jobs) {
    if ($SkipCompleted -and (Test-TrainingJobCompleted -Job $job)) {
        Write-Host ""
        Write-Host "=== SKIP (ya completado): $($job.name.ToUpper()) | $($job.scenario) ===" -ForegroundColor Yellow
        Add-SkippedTrainingJobRecord -Job $job
        continue
    }

    Wait-TrainingRam -Label "$($job.name.ToUpper())/$($job.scenario)"

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
        Write-Host "!!! [$($job.scenario)] $($job.name.ToUpper()) FALLO (exit=$exitCode) - ver $errPath" -ForegroundColor Red
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
