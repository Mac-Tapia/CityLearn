param(
    [string]$Scenario = "ALL",
    [int]$Seed = 0,
    [int]$EpisodeTimeSteps = 8760,
    [int]$Episodes = 5,
    [string]$OutputRoot = "outputs\citylearn_v3_madrl_iquitos_official_full_cuda_v1",
    [string]$SchemaPath = "CityLearn\data\datasets\citylearn_iquitos_2023_2025\schema.json",
    [int]$TorchThreads = 12,
    [int]$LiveProgressInterval = 250,
    # -StartFromAlgorithm: skip all algorithm stages before this one.
    # Valid values: happo (default, run all), masac, matd3, maac.
    # Use to resume after a partial failure without re-running completed algorithms.
    [ValidateSet("happo", "masac", "matd3", "maac")]
    [string]$StartFromAlgorithm = "happo",
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
    [int]$HappoHiddenSize = 384,
    [int]$HappoLiveHeartbeatSeconds = 30,
    [double]$MasacMaxReplayBufferGib = 8,
    [int]$MasacBufferSize = 20,
    [int]$MasacCriticBatchSize = 64,
    [int]$MasacCriticTrainSteps = 1,
    [int]$MasacActorSampleTimes = 5,
    [int]$MasacRnnHiddenDim = 256,
    [int]$MasacQmixHiddenDim = 128,
    [int]$MasacHyperHiddenDim = 256,
    [int]$MasacLiveHeartbeatSeconds = 30,
    [int]$Matd3BatchSize = 256,
    [int]$Matd3BufferSize = 4096,
    [int]$Matd3HiddenSize = 256,
    [int]$Matd3TrainInterval = 100,
    [int]$Matd3LiveHeartbeatSeconds = 30,
    [int]$MaacBatchSize = 256,
    [int]$MaacBufferLength = 50000,
    [int]$MaacHiddenSize = 256,
    [int]$MaacStepsPerUpdate = 250,
    [int]$MaacNumUpdates = 8,
    [int]$MaacLiveHeartbeatSeconds = 30,
    [switch]$Cuda = $true,
    [switch]$LiveOutput,
    [switch]$DryRun,
    [switch]$SkipCompleted
)

$ErrorActionPreference = "Stop"

function Clear-TrainingHost {
    try {
        Clear-Host
    }
    catch {
        Write-Host ""
    }
}

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

if ($GpuProfile -eq "local4060_fast") {
    if (-not $PSBoundParameters.ContainsKey("TorchThreads")) { $TorchThreads = 8 }
    if (-not $PSBoundParameters.ContainsKey("LiveProgressInterval")) { $LiveProgressInterval = 1000 }
    if (-not $PSBoundParameters.ContainsKey("HappoHiddenSize")) { $HappoHiddenSize = 256 }
    if (-not $PSBoundParameters.ContainsKey("MasacMaxReplayBufferGib")) { $MasacMaxReplayBufferGib = if ($IsLocal8GbGpu) { 3.0 } else { 8 } }
    if (-not $PSBoundParameters.ContainsKey("MasacBufferSize")) { $MasacBufferSize = if ($IsLocal8GbGpu) { 2 } else { 20 } }
    if (-not $PSBoundParameters.ContainsKey("MasacCriticBatchSize")) { $MasacCriticBatchSize = if ($IsLocal8GbGpu) { 1 } else { 64 } }
    if (-not $PSBoundParameters.ContainsKey("MasacCriticTrainSteps")) { $MasacCriticTrainSteps = 1 }
    if (-not $PSBoundParameters.ContainsKey("MasacActorSampleTimes")) { $MasacActorSampleTimes = if ($IsLocal8GbGpu) { 2 } else { 4 } }
    if (-not $PSBoundParameters.ContainsKey("MasacRnnHiddenDim")) { $MasacRnnHiddenDim = if ($IsLocal8GbGpu) { 64 } else { 256 } }
    if (-not $PSBoundParameters.ContainsKey("MasacQmixHiddenDim")) { $MasacQmixHiddenDim = if ($IsLocal8GbGpu) { 32 } else { 128 } }
    if (-not $PSBoundParameters.ContainsKey("MasacHyperHiddenDim")) { $MasacHyperHiddenDim = if ($IsLocal8GbGpu) { 64 } else { 256 } }
    if (-not $PSBoundParameters.ContainsKey("Matd3BatchSize")) { $Matd3BatchSize = 256 }
    if (-not $PSBoundParameters.ContainsKey("Matd3BufferSize")) { $Matd3BufferSize = 4096 }
    if (-not $PSBoundParameters.ContainsKey("Matd3HiddenSize")) { $Matd3HiddenSize = 256 }
    if (-not $PSBoundParameters.ContainsKey("Matd3TrainInterval")) { $Matd3TrainInterval = 100 }
    if (-not $PSBoundParameters.ContainsKey("MaacBatchSize")) { $MaacBatchSize = 256 }
    if (-not $PSBoundParameters.ContainsKey("MaacBufferLength")) { $MaacBufferLength = 50000 }
    if (-not $PSBoundParameters.ContainsKey("MaacHiddenSize")) { $MaacHiddenSize = 256 }
    if (-not $PSBoundParameters.ContainsKey("MaacNumUpdates")) { $MaacNumUpdates = 4 }
}
elseif ($GpuProfile -eq "local4060") {
    if (-not $PSBoundParameters.ContainsKey("TorchThreads")) { $TorchThreads = 12 }
    if (-not $PSBoundParameters.ContainsKey("LiveProgressInterval")) { $LiveProgressInterval = 500 }
    if (-not $PSBoundParameters.ContainsKey("HappoHiddenSize")) { $HappoHiddenSize = 512 }
    if (-not $PSBoundParameters.ContainsKey("MasacMaxReplayBufferGib")) { $MasacMaxReplayBufferGib = if ($IsLocal8GbGpu) { 4 } else { 7 } }
    if (-not $PSBoundParameters.ContainsKey("MasacBufferSize")) { $MasacBufferSize = 20 }
    if (-not $PSBoundParameters.ContainsKey("MasacCriticBatchSize")) { $MasacCriticBatchSize = 64 }
    if (-not $PSBoundParameters.ContainsKey("MasacCriticTrainSteps")) { $MasacCriticTrainSteps = 2 }
    if (-not $PSBoundParameters.ContainsKey("MasacActorSampleTimes")) { $MasacActorSampleTimes = 8 }
    if (-not $PSBoundParameters.ContainsKey("MasacRnnHiddenDim")) { $MasacRnnHiddenDim = 256 }
    if (-not $PSBoundParameters.ContainsKey("MasacQmixHiddenDim")) { $MasacQmixHiddenDim = 128 }
    if (-not $PSBoundParameters.ContainsKey("MasacHyperHiddenDim")) { $MasacHyperHiddenDim = 256 }
    if (-not $PSBoundParameters.ContainsKey("Matd3BatchSize")) { $Matd3BatchSize = 1024 }
    if (-not $PSBoundParameters.ContainsKey("Matd3BufferSize")) { $Matd3BufferSize = 8192 }
    if (-not $PSBoundParameters.ContainsKey("Matd3HiddenSize")) { $Matd3HiddenSize = 384 }
    if (-not $PSBoundParameters.ContainsKey("Matd3TrainInterval")) { $Matd3TrainInterval = 50 }
    if (-not $PSBoundParameters.ContainsKey("MaacBatchSize")) { $MaacBatchSize = 256 }
    if (-not $PSBoundParameters.ContainsKey("MaacBufferLength")) { $MaacBufferLength = 50000 }
    if (-not $PSBoundParameters.ContainsKey("MaacHiddenSize")) { $MaacHiddenSize = 256 }
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
$ManifestPath = Join-Path $OutputRootPath "official_full_manifest.json"
$StatusPath = Join-Path $OutputRootPath "official_full_status.json"
$TrainingConfigYaml = "CityLearn\configs\citylearn_v3_madrl_training.yaml"
$TrainingConfigJson = "CityLearn\configs\citylearn_v3_madrl_training.json"
$NumEnvSteps = $EpisodeTimeSteps * $Episodes
$CudaArgs = if ($Cuda) { @("--cuda") } else { @() }
$ArtifactArgs = @(
    "--artifact-profile", "$ArtifactProfile",
    "--trace-record-interval", "$TraceRecordInterval",
    "--trace-detail", "$TraceDetail"
)
$CudaMemoryArgs = if ($null -ne $EffectiveCudaMemoryFraction) {
    @("--cuda-memory-fraction", "$EffectiveCudaMemoryFraction")
}
else {
    @()
}
$SchemaPathInput = $SchemaPath.Trim()

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

$GpuInfo = if ($null -ne $DedicatedGpuInfo) {
    "$($DedicatedGpuInfo.name), $($DedicatedGpuInfo.memory_total_mib) MiB dedicated, driver $($DedicatedGpuInfo.driver_version)"
}
else {
    $null
}

if ([string]::IsNullOrWhiteSpace($SchemaPathInput)) {
    throw "SchemaPath cannot be empty."
}

$SchemaPathFull = if ([System.IO.Path]::IsPathRooted($SchemaPathInput)) {
    $SchemaPathInput
}
else {
    Join-Path $ProjectRoot $SchemaPathInput
}

if (-not (Test-Path -LiteralPath $SchemaPathFull)) {
    throw "SchemaPath not found: $SchemaPathFull"
}

$SchemaPathResolved = (Resolve-Path -LiteralPath $SchemaPathFull).Path
$SchemaPathForArgs = if ([System.IO.Path]::IsPathRooted($SchemaPathInput)) {
    $SchemaPathResolved
}
else {
    $SchemaPathInput
}
$DatasetName = Split-Path -Leaf (Split-Path -Parent $SchemaPathResolved)

$ReadinessScript = Join-Path $ProjectRoot "tools\check_training_dataset_ready.py"
if (-not (Test-Path -LiteralPath $ReadinessScript)) {
    throw "Dataset readiness gate not found: $ReadinessScript"
}

$ReadinessManifest = Join-Path $ProjectRoot "outputs\dataset_audit\training_dataset_ready_manifest.json"
$DatasetDirResolved = Split-Path -Parent $SchemaPathResolved

# Check if a fresh manifest already exists (written in the last 6 hours with status=ready).
# If so, skip the full gate check (which loads all 17-building CSVs via CityLearnEnv).
# The CSV integrity was validated by the audit pipeline; the env will be tested when the
# first training run starts.
$skipGate = $false
if (Test-Path -LiteralPath $ReadinessManifest) {
    try {
        $manifestData = Get-Content -LiteralPath $ReadinessManifest -Raw | ConvertFrom-Json
        $generatedAt = [datetime]::Parse($manifestData.generated_at).ToUniversalTime()
        $ageHours = ([datetime]::UtcNow - $generatedAt).TotalHours
        if ($ageHours -le 6 -and $manifestData.status -eq "ready") {
            $skipGate = $true
            Write-Host "Gate check: manifest fresco ($([math]::Round($ageHours,1))h) con status=ready. Saltando recarga de CityLearnEnv." -ForegroundColor Green
        }
    } catch {
        $skipGate = $false
    }
}

if (-not $skipGate) {
    Write-Host "Validating raw CityLearn dataset before MADRL normalization..." -ForegroundColor Cyan
    # --skip-citylearn-load evita instanciar CityLearnEnv 3 veces desde CSV.
    # El env se valida cuando arranque el primer run de entrenamiento.
    & $Python -B "tools\check_training_dataset_ready.py" `
        --dataset-dir $DatasetDirResolved `
        --buildingcsv-dir "CityLearn\data\buildingcsv" `
        --audit-dir "outputs\dataset_audit" `
        --manifest-out "outputs\dataset_audit\training_dataset_ready_manifest.json" `
        --skip-citylearn-load
    $readinessExit = $LASTEXITCODE
    if ($readinessExit -ne 0) {
        throw "Dataset is not ready for training normalization. See: $ReadinessManifest"
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

$MaxConcurrentScenarioJobs = [Math]::Max(1, [int]$MaxConcurrentScenarioJobs)
$MaxConcurrentHeavyJobs = [Math]::Max(1, [int]$MaxConcurrentHeavyJobs)
$EffectiveParallelScenarios = [bool]$ParallelScenarios -and (-not [bool]$LiveOutput) -and ($ScenarioList.Count -gt 1)
if ([bool]$ParallelScenarios -and [bool]$LiveOutput -and $ScenarioList.Count -gt 1) {
    Write-Host "ParallelScenarios requested, but LiveOutput requires sequential rich display. Running sequential live mode." -ForegroundColor Yellow
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
            "--schema-path", $SchemaPathForArgs,
            "--seed", "$Seed",
            "--episode-time-steps", "$EpisodeTimeSteps",
            "--episodes", "$Episodes",
            "--num-env-steps", "$NumEnvSteps",
            "--hidden-size", "$HappoHiddenSize",
            "--gamma", "0.9999",
            "--torch-threads", "$TorchThreads",
            "--n-rollout-threads", "1",
            "--log-interval", "1",
            "--eval-interval", "1",
            "--live-progress-interval", "$LiveProgressInterval",
            "--live-heartbeat-seconds", "$HappoLiveHeartbeatSeconds",
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
            "--scenario", $scenarioName,
            "--schema-path", $SchemaPathForArgs,
            "--seed", "$Seed",
            "--episode-time-steps", "$EpisodeTimeSteps",
            "--episodes", "$Episodes",
            "--action-bins", "3",
            "--discrete-action-mode", "axis",
            "--max-replay-buffer-gib", "$MasacMaxReplayBufferGib",
            "--buffer-size", "$MasacBufferSize",
            "--critic-batch-size", "$MasacCriticBatchSize",
            "--critic-train-steps", "$MasacCriticTrainSteps",
            "--actor-sample-times", "$MasacActorSampleTimes",
            "--grad-norm-clip", "1.0",
            "--rnn-hidden-dim", "$MasacRnnHiddenDim",
            "--qmix-hidden-dim", "$MasacQmixHiddenDim",
            "--hyper-hidden-dim", "$MasacHyperHiddenDim",
            "--gamma", "0.9999",
            "--torch-threads", "$TorchThreads",
            "--live-heartbeat-seconds", "$MasacLiveHeartbeatSeconds",
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
            "--scenario", $scenarioName,
            "--schema-path", $SchemaPathForArgs,
            "--seed", "$Seed",
            "--episode-time-steps", "$EpisodeTimeSteps",
            "--episodes", "$Episodes",
            "--num-env-steps", "$NumEnvSteps",
            "--batch-size", "$Matd3BatchSize",
            "--buffer-size", "$Matd3BufferSize",
            "--hidden-size", "$Matd3HiddenSize",
            "--gamma", "0.9999",
            "--max-grad-norm", "1.0",
            "--train-interval", "$Matd3TrainInterval",
            "--num-random-episodes", "1",
            "--torch-threads", "$TorchThreads",
            "--live-progress-interval", "$LiveProgressInterval",
            "--live-heartbeat-seconds", "$Matd3LiveHeartbeatSeconds",
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
            "--scenario", $scenarioName,
            "--schema-path", $SchemaPathForArgs,
            "--seed", "$Seed",
            "--episode-time-steps", "$EpisodeTimeSteps",
            "--episodes", "$Episodes",
            "--action-bins", "3",
            "--discrete-action-mode", "axis",
            "--max-discrete-actions", "512",
            "--batch-size", "$MaacBatchSize",
            "--buffer-length", "$MaacBufferLength",
            "--steps-per-update", "$MaacStepsPerUpdate",
            "--num-updates", "$MaacNumUpdates",
            "--hidden-size", "$MaacHiddenSize",
            "--attend-heads", "4",
            "--pi-lr", "0.0003",
            "--q-lr", "0.001",
            "--tau", "0.005",
            "--gamma", "0.9999",
            "--torch-threads", "$TorchThreads",
            "--live-progress-interval", "$LiveProgressInterval",
            "--live-heartbeat-seconds", "$MaacLiveHeartbeatSeconds",
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
    schema_path = $SchemaPathForArgs
    schema_path_resolved = $SchemaPathResolved
    scenario = $Scenario
    scenarios = $ScenarioList
    seed = $Seed
    episode_time_steps = $EpisodeTimeSteps
    episodes = $Episodes
    num_env_steps = $NumEnvSteps
    torch = "torch 2.8.0+cu126"
    cuda = [bool]$Cuda
    algorithm_family = "MADRL"
    execution = if ($EffectiveParallelScenarios) { "parallel_scenarios_by_algorithm" } else { "sequential" }
    parallelization = [ordered]@{
        requested = [bool]$ParallelScenarios
        effective = [bool]$EffectiveParallelScenarios
        max_concurrent_scenario_jobs = $MaxConcurrentScenarioJobs
        max_concurrent_heavy_jobs = $MaxConcurrentHeavyJobs
        heavy_algorithms = @("masac", "maac")
        strategy = "Run the same MADRL algorithm across scenarios concurrently, while keeping MASAC/MAAC limited for memory stability."
        local_8gb_concurrency_adjusted = [bool]$Local8GbConcurrencyAdjusted
        local_8gb_concurrency_note = $Local8GbConcurrencyNote
        disabled_reason = if (-not [bool]$ParallelScenarios) { "not_requested" } elseif ([bool]$LiveOutput) { "live_output_requires_sequential_display" } elseif ($ScenarioList.Count -le 1) { "single_scenario" } else { $null }
    }
    active_project_environment = [ordered]@{
        python = $Python
        virtual_env = $env:VIRTUAL_ENV
        pythonpath = $env:PYTHONPATH
    }
    gpu_optimization = [ordered]@{
        enabled = $true
        profile = $GpuProfile
        detected_gpu = $GpuInfo
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
        strategy = if ($GpuProfile -eq "local4060_fast") { "RTX 4060 fast local profile: TF32-enabled Torch runtime, lighter network/update sizes, lower logging IO" } else { "RTX 4060 local profile: TF32-enabled Torch runtime, larger MADRL network/update sizes, grouped updates, reduced live-progress IO" }
        note = "CityLearn environment stepping remains sequential to preserve episode accounting. GPU is used by the four MADRL neural backends."
    }
    algorithm_resource_limits = [ordered]@{
        happo_hidden_size = $HappoHiddenSize
        happo_live_heartbeat_seconds = $HappoLiveHeartbeatSeconds
        masac_max_replay_buffer_gib = $MasacMaxReplayBufferGib
        masac_buffer_size = $MasacBufferSize
        masac_critic_batch_size = $MasacCriticBatchSize
        masac_critic_train_steps = $MasacCriticTrainSteps
        masac_actor_sample_times = $MasacActorSampleTimes
        masac_rnn_hidden_dim = $MasacRnnHiddenDim
        masac_qmix_hidden_dim = $MasacQmixHiddenDim
        masac_hyper_hidden_dim = $MasacHyperHiddenDim
        masac_live_heartbeat_seconds = $MasacLiveHeartbeatSeconds
        matd3_batch_size = $Matd3BatchSize
        matd3_buffer_size = $Matd3BufferSize
        matd3_hidden_size = $Matd3HiddenSize
        matd3_train_interval = $Matd3TrainInterval
        matd3_live_heartbeat_seconds = $Matd3LiveHeartbeatSeconds
        maac_batch_size = $MaacBatchSize
        maac_buffer_length = $MaacBufferLength
        maac_hidden_size = $MaacHiddenSize
        maac_steps_per_update = $MaacStepsPerUpdate
        maac_num_updates = $MaacNumUpdates
        maac_live_heartbeat_seconds = $MaacLiveHeartbeatSeconds
    }
    training_config = [ordered]@{
        yaml = $TrainingConfigYaml
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

if ($DryRun) {
    $manifest.jobs = @()
    foreach ($job in $jobs) {
        $commandArgs = @("-B", $job.script) + $job.args
        $manifest.jobs += [ordered]@{
            name = $job.name
            scenario = $job.scenario
            script = $job.script
            started_at = $null
            completed_at = $null
            exit_code = $null
            output_dir = Join-Path $OutputRoot "$($job.name)\$($job.scenario)_seed_$Seed"
            command = "$Python " + ($commandArgs -join " ")
            planned_only = $true
        }
    }
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

function Get-TrainingProcessSnapshot {
    param(
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$LauncherProcess,

        [Parameter(Mandatory = $true)]
        [datetime]$StartedAt
    )

    $windowStart = $StartedAt.AddSeconds(-5)
    $windowEnd = (Get-Date).AddSeconds(5)
    $candidates = @()

    try {
        $candidates = @(
            Get-Process -Name python -ErrorAction SilentlyContinue |
                Where-Object {
                    try {
                        $_.StartTime -ge $windowStart -and $_.StartTime -le $windowEnd
                    }
                    catch {
                        $false
                    }
                }
        )
    }
    catch {
        $candidates = @()
    }

    try { $LauncherProcess.Refresh() } catch {}
    $launcherCandidate = $candidates | Where-Object { $_.Id -eq $LauncherProcess.Id } | Select-Object -First 1
    if ($null -eq $launcherCandidate) {
        $launcherCandidate = $LauncherProcess
    }

    $activeCandidate = $candidates |
        Sort-Object -Property @{ Expression = { if ($null -ne $_.CPU) { $_.CPU } else { 0 } }; Descending = $true } |
        Select-Object -First 1

    if ($null -eq $activeCandidate) {
        $activeCandidate = $launcherCandidate
    }

    [pscustomobject]@{
        LauncherPid = [int]$LauncherProcess.Id
        LauncherCpu = if ($null -ne $launcherCandidate.CPU) { [double]$launcherCandidate.CPU } else { 0.0 }
        ActivePid = if ($null -ne $activeCandidate) { [int]$activeCandidate.Id } else { [int]$LauncherProcess.Id }
        ActiveCpu = if ($null -ne $activeCandidate -and $null -ne $activeCandidate.CPU) { [double]$activeCandidate.CPU } else { 0.0 }
        ActivePath = if ($null -ne $activeCandidate) { try { [string]$activeCandidate.Path } catch { "" } } else { "" }
        HasWorker = ($null -ne $activeCandidate -and $activeCandidate.Id -ne $LauncherProcess.Id)
    }
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
        name         = $Job.name
        scenario     = $Job.scenario
        script       = $Job.script
        started_at   = "skipped"
        completed_at = (Get-Date).ToString("o")
        exit_code    = 0
        log          = $null
        stderr_log   = $null
        output_dir   = $jobOutputDir
        skipped      = $true
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
    $algorithmOrder = @("happo", "masac", "matd3", "maac")
    $startIdx = $algorithmOrder.IndexOf($StartFromAlgorithm.ToLower())
    foreach ($algorithmName in $algorithmOrder) {
        $thisIdx = $algorithmOrder.IndexOf($algorithmName)
        if ($thisIdx -lt $startIdx) {
            Write-Host "  SKIP  $($algorithmName.ToUpper()) (StartFromAlgorithm=$StartFromAlgorithm)" -ForegroundColor DarkGray
            continue
        }
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
    # ── Skip if already completed ─────────────────────────────────────────────
    if ($SkipCompleted) {
        $jobOutputDir = Join-Path $OutputRoot "$($job.name)\$($job.scenario)_seed_$Seed"
        $jobResultsPath = Join-Path $ProjectRoot (Join-Path $jobOutputDir "data\results.json")
        if (Test-Path -LiteralPath $jobResultsPath) {
            Write-Host ""
            Write-Host "=== SKIP (ya completado): $($job.name.ToUpper()) | $($job.scenario) ===" -ForegroundColor Yellow
            Write-Host "    results.json encontrado: $jobResultsPath" -ForegroundColor DarkGray
            Add-SkippedTrainingJobRecord -Job $job
            continue
        }
    }

    # ── RAM guard: esperar hasta tener al menos 1.5 GB libres ────────────────
    Wait-TrainingRam -Label "$($job.name.ToUpper())/$($job.scenario)"

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
            Write-Host ("=== CityLearn v3 MADRL: {0} / {1} ===" -f $job.name.ToUpper(), $job.scenario) -ForegroundColor Cyan
            Write-Host "$Python $($commandArgs -join ' ')" -ForegroundColor DarkGray

            # Previene forrtl: error (200) al cerrar la ventana de consola
            $env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = "1"
            # Fuerza flush inmediato de stdout/stderr a los archivos de log
            $env:PYTHONUNBUFFERED = "1"

            $process = Start-Process `
                -FilePath $Python `
                -ArgumentList $commandArgs `
                -WorkingDirectory $ProjectRoot `
                -RedirectStandardOutput $logPath `
                -RedirectStandardError $errPath `
                -WindowStyle Hidden `
                -PassThru

            $liveProgressPath = Join-Path $ProjectRoot (Join-Path $jobRecord.output_dir "live_progress.json")
            $episodeSummaryPath = Join-Path $ProjectRoot (Join-Path $jobRecord.output_dir "figures\tables\episode_summary.csv")

            # Extrae hiperparametros relevantes del comando
            $argsStr = $commandArgs -join " "
            $episodesArg    = if ($argsStr -match '--episodes\s+(\S+)')         { $Matches[1] } else { "?" }
            $stepsArg       = if ($argsStr -match '--episode-time-steps\s+(\S+)') { $Matches[1] } else { $EpisodeTimeSteps }
            $seedArg        = if ($argsStr -match '--seed\s+(\S+)')              { $Matches[1] } else { $Seed }
            $hiddenArg      = if ($argsStr -match '--hidden-size\s+(\S+)')       { $Matches[1] } else { "-" }
            $batchArg       = if ($argsStr -match '--batch-size\s+(\S+)')        { $Matches[1] } else { "-" }
            $bufferArg      = if ($argsStr -match '--buffer[-_](?:size|length)\s+(\S+)') { $Matches[1] } else { "-" }
            $lrArg          = if ($argsStr -match '--pi-lr\s+(\S+)')             { $Matches[1] } else { "-" }

            while (-not $process.HasExited) {
                $process.Refresh()
                $processSnapshot = Get-TrainingProcessSnapshot -LauncherProcess $process -StartedAt $startedAt

                Clear-TrainingHost

                # ── Encabezado ───────────────────────────────────────────────────────
                $hora = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
                Write-Host "================================================================================" -ForegroundColor Cyan
                Write-Host "  CITYLEARN v3 MADRL  -  ENTRENAMIENTO OFICIAL EN VIVO  -  $hora" -ForegroundColor Cyan
                Write-Host "================================================================================" -ForegroundColor Cyan
                Write-Host ("  Dataset   : {0}" -f $DatasetName)
                Write-Host ("  Algoritmo : {0,-8}  Escenario : {1,-4}  PID lanzador : {2}" -f $job.name.ToUpper(), $job.scenario, $process.Id)
                Write-Host ("  Proceso   : PID activo {0}  CPU={1:N1}s  Worker={2}" -f $processSnapshot.ActivePid, $processSnapshot.ActiveCpu, $processSnapshot.HasWorker)
                if ($job.name -eq "masac") {
                    Write-Host "  Backend   : MASAC/MSAC externo con mezclador QMIX discreto; el job sigue siendo MASAC MADRL." -ForegroundColor DarkGray
                }
                Write-Host ("  OutputDir : {0}" -f $jobRecord.output_dir)
                Write-Host ""

                # ── Cadena de corridas: completadas / actual / pendientes ──────────
                Write-Host "  [CADENA DE CORRIDAS]" -ForegroundColor Magenta
                $allJobs = @()
                foreach ($sc in $ScenarioList) {
                    foreach ($alg in @("happo","masac","matd3","maac")) {
                        $allJobs += "$($alg.ToUpper())/$sc"
                    }
                }
                $currentTag = "$($job.name.ToUpper())/$($job.scenario)"
                $reached    = $false
                $chainLine  = "  "
                foreach ($tag in $allJobs) {
                    if ($tag -eq $currentTag) {
                        $chainLine += "[>$tag<] "
                        $reached = $true
                    } elseif (-not $reached) {
                        $chainLine += "[OK:$tag] "
                    } else {
                        $chainLine += "[$tag] "
                    }
                }
                Write-Host $chainLine -ForegroundColor White
                Write-Host ""

                # ── Hiperparametros ────────────────────────────────────────────────
                Write-Host "  [HIPERPARAMETROS]" -ForegroundColor Yellow
                Write-Host ("    Episodios={0}  Pasos/ep={1}  Seed={2}  CUDA={3}  Threads={4}" -f `
                    $episodesArg, $stepsArg, $seedArg, [bool]$Cuda, $TorchThreads)
                $hpLine = "   "
                if ($hiddenArg -ne "-") { $hpLine += " hidden={0}" -f $hiddenArg }
                if ($batchArg  -ne "-") { $hpLine += " batch={0}"  -f $batchArg  }
                if ($bufferArg -ne "-") { $hpLine += " buffer={0}" -f $bufferArg }
                if ($lrArg     -ne "-") { $hpLine += " pi_lr={0}"  -f $lrArg     }
                if ($hpLine.Trim()) { Write-Host $hpLine }
                Write-Host ""

                if (Test-Path -LiteralPath $liveProgressPath) {
                    try {
                        $p = Get-Content -LiteralPath $liveProgressPath -Raw | ConvertFrom-Json
                        $w = $p.reward_axis_weights

                        # ── Pesos multiobjetivo ───────────────────────────────────
                        Write-Host "  [PESOS MULTIOBJETIVO  (OE.1 flex / OE.2 CO2 / OE.3 cost)]" -ForegroundColor Yellow
                        Write-Host ("    OE.1 flex = {0:F3}   OE.2 CO2 = {1:F3}   OE.3 cost = {2:F3}" -f `
                            $w.flex, $w.carbon, $w.cost) -ForegroundColor White
                        Write-Host ("    Funcion: {0}   Perfil: {1}" -f $p.reward_function, $p.reward_profile)
                        Write-Host ""

                        # ── Progreso con barra ────────────────────────────────────
                        $totalPasos  = [int]$episodesArg * [int]$stepsArg
                        $pasoActual  = [int]$p.global_step
                        $pct         = if ($totalPasos -gt 0) { [math]::Round(100.0 * $pasoActual / $totalPasos, 1) } else { 0 }
                        $barWidth    = 50
                        $filled      = [math]::Round($barWidth * $pct / 100)
                        $bar         = "#" * $filled + "-" * ($barWidth - $filled)
                        Write-Host "  [PROGRESO]" -ForegroundColor Yellow
                        Write-Host ("    [{0}] {1,5:F1}%" -f $bar, $pct) -ForegroundColor Cyan
                        Write-Host ("    Episodio  : {0} / {1}     Paso ep   : {2} / {3}" -f `
                            $p.episode, ([int]$episodesArg - 1), $p.episode_step, $stepsArg)
                        Write-Host ("    Paso glob : {0} / {1}     Time step : {2}" -f `
                            $pasoActual, $totalPasos, $p.time_step)
                        Write-Host ""

                        # ── Ganancias / Recompensas por componente ────────────────
                        Write-Host "  [GANANCIAS POR COMPONENTE  (reward por eje)]" -ForegroundColor Yellow
                        $rFlex = $p.reward_component_flex_mean
                        $rCO2  = $p.reward_component_carbon_mean
                        $rCost = $p.reward_component_cost_mean
                        $rEV   = $p.reward_component_ev_mean
                        $rTeam = $p.reward_team_reward
                        $rInst = $p.instant_reward_mean
                        Write-Host ("    OE.1 flex_mean = {0,10:F6}   (ganancia flexibilidad)" -f $rFlex)    -ForegroundColor $(if ($rFlex -ge 0) {"Green"} else {"Red"})
                        Write-Host ("    OE.2 co2_mean  = {0,10:F6}   (ganancia reduccion CO2)" -f $rCO2)    -ForegroundColor $(if ($rCO2  -ge 0) {"Green"} else {"Red"})
                        Write-Host ("    OE.3 cost_mean = {0,10:F6}   (ganancia reduccion costo)" -f $rCost)  -ForegroundColor $(if ($rCost -ge 0) {"Green"} else {"Red"})
                        Write-Host ("    EV_mean        = {0,10:F6}   (ganancia gestion EV)" -f $rEV)         -ForegroundColor $(if ($rEV   -ge 0) {"Green"} else {"Red"})
                        Write-Host ("    team_reward    = {0,10:F6}   (recompensa cooperativa equipo)" -f $rTeam)
                        Write-Host ("    instant_mean   = {0,10:F6}   (recompensa instante actual)" -f $rInst) -ForegroundColor Cyan
                        Write-Host ""

                        # ── Metricas acumuladas ───────────────────────────────────
                        Write-Host "  [METRICAS ACUMULADAS]" -ForegroundColor Yellow
                        Write-Host ("    retorno_acum_ep  = {0,12:F4}   recomp_media_ep  = {1,10:F6}" -f `
                            $p.episode_return_cumulative, $p.episode_reward_mean_cumulative)
                        Write-Host ("    retorno_acum_tot = {0,12:F4}   recomp_media_tot = {1,10:F6}" -f `
                            $p.total_return_cumulative,   $p.total_reward_mean_cumulative)
                        Write-Host ""

                        # ── KPIs energeticos del distrito ─────────────────────────
                        Write-Host "  [KPIs ENERGETICOS  (escenario $($job.scenario))]" -ForegroundColor Yellow
                        Write-Host ("    Intensidad CO2      = {0,8:F4} kgCO2/kWh" -f $p.carbon_intensity_mean)
                        Write-Host ("    Precio electricidad = {0,8:F4} $/kWh" -f $p.electricity_price_mean)
                        Write-Host ("    Consumo neto dist.  = {0,10:F2} kWh" -f $p.district_net_electricity_consumption)
                        Write-Host ("    Costo neto dist.    = {0,10:F2} $" -f $p.district_net_electricity_consumption_cost)
                        Write-Host ("    Emisiones dist.     = {0,10:F2} kgCO2" -f $p.district_net_electricity_consumption_emission)
                        Write-Host ""

                        $ts = (Get-Item -LiteralPath $liveProgressPath).LastWriteTime
                        $progressAgeSeconds = [math]::Round(((Get-Date) - $ts).TotalSeconds, 1)
                        $progressColor = if ($progressAgeSeconds -gt [math]::Max(120, $LiveProgressInterval * 2)) { "Yellow" } else { "DarkGray" }
                        Write-Host ("  live_progress actualizado: {0}  (hace {1:N1} s)" -f $ts, $progressAgeSeconds) -ForegroundColor $progressColor
                        if ($progressAgeSeconds -gt [math]::Max(120, $LiveProgressInterval * 2)) {
                            Write-Host "  Aviso: proceso activo sin pasos nuevos escritos; MASAC puede estar entrenando critic/actor entre episodios." -ForegroundColor Yellow
                        }
                    }
                    catch {
                        Write-Host "  Leyendo live_progress.json..." -ForegroundColor Yellow
                    }
                }
                else {
                    Write-Host "  Inicializando entorno y agente; aun no hay live_progress.json." -ForegroundColor Yellow
                }

                # ── Historial por episodio ─────────────────────────────────────────
                if (Test-Path -LiteralPath $episodeSummaryPath) {
                    try {
                        $epRows = Import-Csv -LiteralPath $episodeSummaryPath
                        if ($epRows.Count -gt 0) {
                            Write-Host "  [HISTORIAL POR EPISODIO]" -ForegroundColor Yellow
                            Write-Host ("    {0,-4} {1,-10} {2,-10} {3,-14} {4,-12} {5}" -f `
                                "EP", "PASO_INI", "PASO_FIN", "RETORNO_TOTAL", "MEDIA_EP", "PASOS")
                            foreach ($row in $epRows) {
                                $nPasos = [int]$row.last_global_step - [int]$row.first_global_step
                                Write-Host ("    {0,-4} {1,-10} {2,-10} {3,-14} {4,-12} {5}" -f `
                                    $row.episode, $row.first_global_step, $row.last_global_step,
                                    ([double]$row.reward_sum_total).ToString("F2"),
                                    ([double]$row.reward_mean_average).ToString("F6"),
                                    $nPasos)
                            }
                            Write-Host ""
                        }
                    }
                    catch {}
                }

                # ── GPU ───────────────────────────────────────────────────────────
                Write-Host "  [GPU]" -ForegroundColor Yellow
                try { nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>$null } catch {}

                # ── Log filtrado ──────────────────────────────────────────────────
                Write-Host ""
                Write-Host "  [LOG - ultimas lineas significativas]" -ForegroundColor Yellow
                if (Test-Path -LiteralPath $logPath) {
                    Get-Content -LiteralPath $logPath -Tail 80 |
                        Where-Object {
                            $line = $_.TrimStart()
                            $line -notlike '*Box(*'              -and
                            $line -notlike '*1000000*'           -and
                            $line -notlike '*share_observation*' -and
                            $line -notlike '*observation_space*' -and
                            $line -notlike '*action_space*'      -and
                            $line -notlike '*float32*'           -and
                            $line -notlike '*dtype*'             -and
                            -not ($line -match '^\s*[\d\.e\+\-]+(\s+[\d\.e\+\-]+){3,}') -and
                            -not ($line -match '^\[[\d\.\s]+\]') -and
                            -not $line.Contains('], [')          -and
                            $line.Length -gt 3                   -and
                            $line.Length -lt 300
                        } |
                        Select-Object -Last 10 |
                        ForEach-Object { Write-Host ("    " + $_) }
                }

                Write-Host ""
                Write-Host "  Pantalla actualiza cada 5 s  |  Logs: $logPath" -ForegroundColor DarkGray
                Write-Host "  NO CIERRE esta ventana durante el entrenamiento." -ForegroundColor Red
                Start-Sleep -Seconds 5
            }

            $exitCode = Resolve-TrainingExitCode -Process $process -JobOutputDir $jobRecord.output_dir

            Clear-TrainingHost
            Write-Host "CITYLEARN V3 MADRL - JOB FINALIZADO" -ForegroundColor Cyan
            Write-Host ("Job: {0}/{1} | exit_code: {2}" -f $job.scenario, $job.name.ToUpper(), $exitCode)
            Write-Host ("Log completo: {0}" -f $logPath)
        }
        finally {
            Pop-Location
        }

        if (-not (Test-Path $errPath)) {
            New-Item -ItemType File -Force -Path $errPath | Out-Null
        }
    }
    else {
        $env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = "1"
        $process = Start-Process `
            -FilePath $Python `
            -ArgumentList $commandArgs `
            -WorkingDirectory $ProjectRoot `
            -RedirectStandardOutput $logPath `
            -RedirectStandardError $errPath `
            -WindowStyle Hidden `
            -Wait `
            -PassThru
        $exitCode = Resolve-TrainingExitCode -Process $process -JobOutputDir $jobRecord.output_dir
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
