param(
    [string]$Scenario = "ALL",
    [int]$Seed = 0,
    [int]$EpisodeTimeSteps = 8760,
    [int]$Episodes = 5,
    [string]$OutputRoot = "outputs\citylearn_v3_madrl_iquitos_official_full_cuda_v1",
    [string]$SchemaPath = "CityLearn\data\datasets\citylearn_iquitos_2023_2025\schema.json",
    [int]$TorchThreads = 12,
    [int]$LiveProgressInterval = 250,
    [int]$HappoHiddenSize = 384,
    [int]$Matd3BatchSize = 256,
    [int]$Matd3BufferSize = 4096,
    [int]$Matd3HiddenSize = 256,
    [int]$Matd3TrainInterval = 100,
    [int]$MaacBatchSize = 64,
    [int]$MaacBufferLength = 256,
    [int]$MaacHiddenSize = 128,
    [switch]$Cuda = $true,
    [switch]$LiveOutput,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

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
            "--scenario", $scenarioName,
            "--schema-path", $SchemaPathForArgs,
            "--seed", "$Seed",
            "--episode-time-steps", "$EpisodeTimeSteps",
            "--episodes", "$Episodes",
            "--action-bins", "3",
            "--discrete-action-mode", "axis",
            "--max-replay-buffer-gib", "8",
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
    execution = "sequential"
    active_project_environment = [ordered]@{
        python = $Python
        virtual_env = $env:VIRTUAL_ENV
        pythonpath = $env:PYTHONPATH
    }
    gpu_optimization = [ordered]@{
        enabled = $true
        live_progress_interval = $LiveProgressInterval
        strategy = "larger batches, grouped updates, reduced live-progress IO"
        note = "Environment execution remains sequential to preserve CityLearn episode accounting and reproducible v2/v3 comparisons."
    }
    algorithm_resource_limits = [ordered]@{
        happo_hidden_size = $HappoHiddenSize
        matd3_batch_size = $Matd3BatchSize
        matd3_buffer_size = $Matd3BufferSize
        matd3_hidden_size = $Matd3HiddenSize
        matd3_train_interval = $Matd3TrainInterval
        maac_batch_size = $MaacBatchSize
        maac_buffer_length = $MaacBufferLength
        maac_hidden_size = $MaacHiddenSize
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
                -NoNewWindow `
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

                Clear-Host
                Write-Host "========================================================================" -ForegroundColor Cyan
                Write-Host "  CITYLEARN v3 MADRL - ENTRENAMIENTO EN VIVO" -ForegroundColor Cyan
                Write-Host "========================================================================" -ForegroundColor Cyan
                Write-Host ("  Dataset   : {0}" -f $DatasetName)
                Write-Host ("  Algoritmo : {0,-10}  Escenario : {1}" -f $job.name.ToUpper(), $job.scenario)
                Write-Host ("  OutputDir : {0}" -f $jobRecord.output_dir)
                Write-Host ("  PID       : {0,-10}  Hora      : {1}" -f $process.Id, (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"))
                Write-Host ""

                # Bloque de hiperparametros
                Write-Host "  [PARAMETROS]" -ForegroundColor Yellow
                Write-Host ("    episodios={0}  pasos/ep={1}  seed={2}  cuda={3}  torch_threads={4}" -f $episodesArg, $stepsArg, $seedArg, [bool]$Cuda, $TorchThreads)
                if ($hiddenArg -ne "-") { Write-Host ("    hidden_size={0}" -f $hiddenArg) -NoNewline }
                if ($batchArg  -ne "-") { Write-Host ("  batch_size={0}" -f $batchArg) -NoNewline }
                if ($bufferArg -ne "-") { Write-Host ("  buffer={0}" -f $bufferArg) -NoNewline }
                if ($lrArg     -ne "-") { Write-Host ("  pi_lr={0}" -f $lrArg) -NoNewline }
                Write-Host ""
                Write-Host ""

                if (Test-Path -LiteralPath $liveProgressPath) {
                    try {
                        $p = Get-Content -LiteralPath $liveProgressPath -Raw | ConvertFrom-Json
                        $w = $p.reward_axis_weights

                        # Pesos de objetivos (OE.1/OE.2/OE.3)
                        Write-Host "  [PESOS MULTIOBJETIVO]" -ForegroundColor Yellow
                        Write-Host ("    OE.1 flex={0:F3}  OE.2 co2={1:F3}  OE.3 cost={2:F3}    funcion={3}" -f $w.flex, $w.carbon, $w.cost, $p.reward_function)
                        Write-Host ""

                        # Progreso del entrenamiento
                        Write-Host "  [PROGRESO]" -ForegroundColor Yellow
                        Write-Host ("    Episodio  : {0} / {1}" -f $p.episode, ($episodesArg - 1))
                        Write-Host ("    Paso ep   : {0} / {1}" -f $p.episode_step, $stepsArg)
                        Write-Host ("    Paso glob : {0} / {1}" -f $p.global_step, ($episodesArg * $stepsArg))
                        Write-Host ("    Time step : {0}" -f $p.time_step)
                        Write-Host ""

                        # Metricas de recompensa
                        Write-Host "  [METRICAS DE RECOMPENSA]" -ForegroundColor Yellow
                        Write-Host ("    retorno_acum_ep   = {0,12:F4}" -f $p.episode_return_cumulative)
                        Write-Host ("    recomp_media_ep   = {0,12:F6}" -f $p.episode_reward_mean_cumulative)
                        Write-Host ("    retorno_acum_tot  = {0,12:F4}" -f $p.total_return_cumulative)
                        Write-Host ("    recomp_media_tot  = {0,12:F6}" -f $p.total_reward_mean_cumulative)
                        Write-Host ("    recomp_instante   = {0,12:F6}" -f $p.instant_reward_mean)
                        Write-Host ""

                        # KPIs energeticos
                        Write-Host "  [KPIs ENERGETICOS]" -ForegroundColor Yellow
                        Write-Host ("    intensidad_CO2     = {0,8:F4} kg/kWh" -f $p.carbon_intensity_mean)
                        Write-Host ("    precio_electricidad= {0,8:F4} $/kWh" -f $p.electricity_price_mean)
                        Write-Host ("    consumo_neto_dist  = {0,8:F1} kWh" -f $p.district_net_electricity_consumption)
                        Write-Host ""

                        $ts = (Get-Item -LiteralPath $liveProgressPath).LastWriteTime
                        Write-Host ("  live_progress actualizado: {0}" -f $ts) -ForegroundColor DarkGray
                    }
                    catch {
                        Write-Host "  Leyendo live_progress.json..." -ForegroundColor Yellow
                    }
                }
                else {
                    Write-Host "  Inicializando entorno y agente; aun no hay live_progress.json." -ForegroundColor Yellow
                }

                # Historial por episodio
                if (Test-Path -LiteralPath $episodeSummaryPath) {
                    try {
                        $epRows = Import-Csv -LiteralPath $episodeSummaryPath
                        if ($epRows.Count -gt 0) {
                            Write-Host "  [HISTORIAL POR EPISODIO]" -ForegroundColor Yellow
                            Write-Host ("    {0,-5} {1,-12} {2,-12} {3,-14} {4}" -f "EP", "PASO_INICIO", "PASO_FIN", "R_SUM_TOTAL", "R_MEAN_AVG")
                            foreach ($row in $epRows) {
                                Write-Host ("    {0,-5} {1,-12} {2,-12} {3,-14} {4}" -f `
                                    $row.episode, $row.first_global_step, $row.last_global_step,
                                    ([double]$row.reward_sum_total).ToString("F1"),
                                    ([double]$row.reward_mean_average).ToString("F6"))
                            }
                            Write-Host ""
                        }
                    }
                    catch {}
                }

                Write-Host "  [GPU]" -ForegroundColor Yellow
                try { nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>$null } catch {}

                Write-Host ""
                Write-Host "  [LOG - ultimas lineas]" -ForegroundColor Yellow
                if (Test-Path -LiteralPath $logPath) {
                    Get-Content -LiteralPath $logPath -Tail 20 |
                        Where-Object {
                            $line = $_.TrimStart()
                            $line -notlike '*Box(*' -and
                            $line -notlike '*1000000.*' -and
                            $line -notlike '*-1000000.*' -and
                            $line -notlike '*share_observation_space*' -and
                            $line -notlike '*observation_space*' -and
                            $line -notlike '*float32)*' -and
                            -not $line.Contains('], [') -and
                            $line -notlike '0.*' -and
                            $line -notlike '1.*'
                        } |
                        Select-Object -Last 8 |
                        ForEach-Object { Write-Host ("    " + $_) }
                }

                Write-Host ""
                Write-Host "  Pantalla se actualiza cada 5 s. Logs completos en: $logPath" -ForegroundColor DarkGray
                Write-Host "  AVISO: NO CIERRE esta ventana mientras el entrenamiento esta activo." -ForegroundColor Red
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
