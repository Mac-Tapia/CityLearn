param(
    [string]$Scenario = "ALL",
    [int]$Seed = 0,
    [int]$EpisodeTimeSteps = 8760,
    [int]$Episodes = 5,
    [string]$OutputRoot = "outputs\citylearn_v3_madrl_iquitos_official_full_cuda_v1",
    [string]$SchemaPath = "CityLearn\data\datasets\citylearn_iquitos_2023_2025\schema.json",
    [int]$TorchThreads = 12,
    [int]$LiveProgressInterval = 250,
    [ValidateSet("local4060", "balanced", "conservative", "aws")]
    [string]$GpuProfile = "local4060",
    [int]$HappoHiddenSize = 384,
    [int]$HappoLiveHeartbeatSeconds = 30,
    [double]$MasacMaxReplayBufferGib = 8,
    [int]$MasacBufferSize = 2,
    [int]$MasacCriticBatchSize = 1,
    [int]$MasacCriticTrainSteps = 1,
    [int]$MasacActorSampleTimes = 5,
    [int]$MasacRnnHiddenDim = 64,
    [int]$MasacQmixHiddenDim = 32,
    [int]$MasacHyperHiddenDim = 64,
    [int]$MasacLiveHeartbeatSeconds = 30,
    [int]$Matd3BatchSize = 256,
    [int]$Matd3BufferSize = 4096,
    [int]$Matd3HiddenSize = 256,
    [int]$Matd3TrainInterval = 100,
    [int]$Matd3LiveHeartbeatSeconds = 30,
    [int]$MaacBatchSize = 64,
    [int]$MaacBufferLength = 256,
    [int]$MaacHiddenSize = 128,
    [int]$MaacLiveHeartbeatSeconds = 30,
    [switch]$Cuda = $true,
    [switch]$LiveOutput,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ($GpuProfile -eq "local4060") {
    if (-not $PSBoundParameters.ContainsKey("TorchThreads")) { $TorchThreads = 12 }
    if (-not $PSBoundParameters.ContainsKey("LiveProgressInterval")) { $LiveProgressInterval = 500 }
    if (-not $PSBoundParameters.ContainsKey("HappoHiddenSize")) { $HappoHiddenSize = 512 }
    if (-not $PSBoundParameters.ContainsKey("MasacMaxReplayBufferGib")) { $MasacMaxReplayBufferGib = 7 }
    if (-not $PSBoundParameters.ContainsKey("MasacBufferSize")) { $MasacBufferSize = 4 }
    if (-not $PSBoundParameters.ContainsKey("MasacCriticBatchSize")) { $MasacCriticBatchSize = 2 }
    if (-not $PSBoundParameters.ContainsKey("MasacCriticTrainSteps")) { $MasacCriticTrainSteps = 2 }
    if (-not $PSBoundParameters.ContainsKey("MasacActorSampleTimes")) { $MasacActorSampleTimes = 8 }
    if (-not $PSBoundParameters.ContainsKey("MasacRnnHiddenDim")) { $MasacRnnHiddenDim = 96 }
    if (-not $PSBoundParameters.ContainsKey("MasacQmixHiddenDim")) { $MasacQmixHiddenDim = 64 }
    if (-not $PSBoundParameters.ContainsKey("MasacHyperHiddenDim")) { $MasacHyperHiddenDim = 96 }
    if (-not $PSBoundParameters.ContainsKey("Matd3BatchSize")) { $Matd3BatchSize = 512 }
    if (-not $PSBoundParameters.ContainsKey("Matd3BufferSize")) { $Matd3BufferSize = 8192 }
    if (-not $PSBoundParameters.ContainsKey("Matd3HiddenSize")) { $Matd3HiddenSize = 384 }
    if (-not $PSBoundParameters.ContainsKey("Matd3TrainInterval")) { $Matd3TrainInterval = 50 }
    if (-not $PSBoundParameters.ContainsKey("MaacBatchSize")) { $MaacBatchSize = 128 }
    if (-not $PSBoundParameters.ContainsKey("MaacBufferLength")) { $MaacBufferLength = 512 }
    if (-not $PSBoundParameters.ContainsKey("MaacHiddenSize")) { $MaacHiddenSize = 256 }
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

$GpuInfo = $null
try {
    $GpuInfo = (& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
}
catch {
    $GpuInfo = $null
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
Write-Host "Validating raw CityLearn dataset before MADRL normalization..." -ForegroundColor Cyan
& $Python -B "tools\check_training_dataset_ready.py" `
    --dataset-dir $DatasetDirResolved `
    --buildingcsv-dir "CityLearn\data\buildingcsv" `
    --audit-dir "outputs\dataset_audit" `
    --manifest-out "outputs\dataset_audit\training_dataset_ready_manifest.json"
$readinessExit = $LASTEXITCODE
if ($readinessExit -ne 0) {
    throw "Dataset is not ready for training normalization. See: $ReadinessManifest"
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
            "--torch-threads", "$TorchThreads",
            "--n-rollout-threads", "1",
            "--log-interval", "1",
            "--eval-interval", "1",
            "--live-progress-interval", "$LiveProgressInterval",
            "--live-heartbeat-seconds", "$HappoLiveHeartbeatSeconds",
            "--gpu-profile", "$GpuProfile"
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
            "--rnn-hidden-dim", "$MasacRnnHiddenDim",
            "--qmix-hidden-dim", "$MasacQmixHiddenDim",
            "--hyper-hidden-dim", "$MasacHyperHiddenDim",
            "--torch-threads", "$TorchThreads",
            "--live-heartbeat-seconds", "$MasacLiveHeartbeatSeconds",
            "--live-progress-interval", "$LiveProgressInterval",
            "--gpu-profile", "$GpuProfile"
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
            "--schema-path", $SchemaPathForArgs,
            "--seed", "$Seed",
            "--episode-time-steps", "$EpisodeTimeSteps",
            "--episodes", "$Episodes",
            "--num-env-steps", "$NumEnvSteps",
            "--batch-size", "$Matd3BatchSize",
            "--buffer-size", "$Matd3BufferSize",
            "--hidden-size", "$Matd3HiddenSize",
            "--train-interval", "$Matd3TrainInterval",
            "--num-random-episodes", "1",
            "--torch-threads", "$TorchThreads",
            "--live-progress-interval", "$LiveProgressInterval",
            "--live-heartbeat-seconds", "$Matd3LiveHeartbeatSeconds",
            "--gpu-profile", "$GpuProfile"
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
            "--schema-path", $SchemaPathForArgs,
            "--seed", "$Seed",
            "--episode-time-steps", "$EpisodeTimeSteps",
            "--episodes", "$Episodes",
            "--action-bins", "3",
            "--discrete-action-mode", "axis",
            "--max-discrete-actions", "512",
            "--batch-size", "$MaacBatchSize",
            "--buffer-length", "$MaacBufferLength",
            "--steps-per-update", "250",
            "--num-updates", "8",
            "--hidden-size", "$MaacHiddenSize",
            "--attend-heads", "4",
            "--pi-lr", "0.0003",
            "--q-lr", "0.001",
            "--tau", "0.005",
            "--gamma", "0.99",
            "--torch-threads", "$TorchThreads",
            "--live-progress-interval", "$LiveProgressInterval",
            "--live-heartbeat-seconds", "$MaacLiveHeartbeatSeconds",
            "--gpu-profile", "$GpuProfile"
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
    execution = "sequential"
    active_project_environment = [ordered]@{
        python = $Python
        virtual_env = $env:VIRTUAL_ENV
        pythonpath = $env:PYTHONPATH
    }
    gpu_optimization = [ordered]@{
        enabled = $true
        profile = $GpuProfile
        detected_gpu = $GpuInfo
        cuda_device_order = $env:CUDA_DEVICE_ORDER
        pytorch_cuda_alloc_conf = $env:PYTORCH_CUDA_ALLOC_CONF
        live_progress_interval = $LiveProgressInterval
        strategy = "RTX 4060 local profile: TF32-enabled Torch runtime, larger MADRL network/update sizes, grouped updates, reduced live-progress IO"
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
        maac_live_heartbeat_seconds = $MaacLiveHeartbeatSeconds
    }
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

                Clear-Host

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

            Clear-Host
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
