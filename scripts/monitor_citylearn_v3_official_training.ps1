param(
    [string]$OutputRoot = "",
    [int]$IntervalSeconds = 30,
    [int]$LogTail = 20
)

$ErrorActionPreference = "Continue"

function Clear-MonitorHost {
    try {
        Clear-Host
    }
    catch {
        Write-Host ""
    }
}

$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptPath "..\..")

function Resolve-OutputRoot {
    param([string]$Requested)

    if (-not [string]::IsNullOrWhiteSpace($Requested)) {
        return $Requested.Trim()
    }

    $latestPath = Join-Path $ProjectRoot "outputs\latest_visible_training_output_root.txt"
    if (Test-Path -LiteralPath $latestPath) {
        $value = (Get-Content -LiteralPath $latestPath -Raw).Trim()
        if (-not [string]::IsNullOrWhiteSpace($value)) {
            return $value
        }
    }

    $candidate = Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "outputs") -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "official_full_status.json") } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if ($candidate) {
        return ("outputs\" + $candidate.Name)
    }

    return $null
}

$OutputRoot = Resolve-OutputRoot -Requested $OutputRoot
if (-not $OutputRoot) {
    Write-Host "No se encontro OutputRoot. Lanza entrenamiento o pasa -OutputRoot." -ForegroundColor Red
    exit 1
}

$OutputRootPath = if ([System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot } else { Join-Path $ProjectRoot $OutputRoot }
$StatusPath = Join-Path $OutputRootPath "official_full_status.json"
$LogDir = Join-Path $OutputRootPath "logs"

function Read-JsonFile {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return $null
    }

    try {
        return Get-Content $Path -Raw | ConvertFrom-Json
    }
    catch {
        Write-Host "No se pudo leer JSON: $Path" -ForegroundColor Yellow
        return $null
    }
}

function Get-FileAgeSeconds {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }

    try {
        return [math]::Round(((Get-Date) - (Get-Item -LiteralPath $Path).LastWriteTime).TotalSeconds, 1)
    }
    catch {
        return $null
    }
}

function Get-ActivePythonProcessSnapshot {
    param($StartedAt)

    try {
        $started = [datetime]$StartedAt
    }
    catch {
        return $null
    }

    $windowStart = $started.AddSeconds(-5)
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

    $activeCandidate = $candidates |
        Sort-Object -Property @{ Expression = { if ($null -ne $_.CPU) { $_.CPU } else { 0 } }; Descending = $true } |
        Select-Object -First 1

    if ($null -eq $activeCandidate) {
        return $null
    }

    $activePath = ""
    try {
        $activePath = [string]$activeCandidate.Path
    }
    catch {
        $activePath = ""
    }

    [pscustomobject]@{
        ActivePid = [int]$activeCandidate.Id
        ActiveCpu = if ($null -ne $activeCandidate.CPU) { [double]$activeCandidate.CPU } else { 0.0 }
        ActivePath = $activePath
        CandidateCount = [int]$candidates.Count
    }
}

function Show-Gpu {
    $nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidia) {
        return
    }

    Write-Host ""
    Write-Host "GPU" -ForegroundColor Cyan
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
}

function Show-Processes {
    Write-Host ""
    Write-Host "Procesos CityLearn v3 activos" -ForegroundColor Cyan
    $processes = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
        Where-Object { $_.CommandLine -like "*train_citylearn_v3*" } |
        Select-Object ProcessId, CreationDate, CommandLine

    if ($processes) {
        $processes | Format-Table -AutoSize
    }
    else {
        Write-Host "No hay procesos train_citylearn_v3 activos." -ForegroundColor Yellow
    }
}

function Get-TrainingRunDir {
    param(
        [string]$Algorithm,
        [string]$Scenario,
        [int]$Seed
    )

    return (Join-Path $OutputRootPath ("{0}\{1}" -f $Algorithm.ToUpperInvariant(), $Scenario))
}

function Test-TrainingResultsArtifact {
    param(
        [string]$Algorithm,
        [string]$Scenario,
        [int]$Seed
    )

    $runDir = Get-TrainingRunDir -Algorithm $Algorithm -Scenario $Scenario -Seed $Seed
    $candidates = @(
        (Join-Path $runDir "results.json"),
        (Join-Path $runDir "data\results.json")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $true
        }
    }

    return $false
}

function Get-AlgorithmPlanState {
    param(
        $Status,
        [string]$Algorithm,
        [string]$Scenario
    )

    $jobMatches = @($Status.jobs | Where-Object { $_.scenario -eq $Scenario -and $_.name -eq $Algorithm })
    if ($jobMatches.Count -gt 0) {
        $job = $jobMatches[-1]
        if ($null -eq $job.completed_at) {
            return "running"
        }
        if ($job.exit_code -eq 0) {
            if ($job.skipped) {
                return "skipped/done"
            }
            return "done"
        }
        return "failed"
    }

    if (Test-TrainingResultsArtifact -Algorithm $Algorithm -Scenario $Scenario -Seed ([int]$Status.seed)) {
        return "done/artifact"
    }

    $startFrom = [string]$Status.start_from_algorithm
    if (-not [string]::IsNullOrWhiteSpace($startFrom)) {
        $order = @("happo", "masac", "matd3", "maac")
        $algorithmIdx = $order.IndexOf($Algorithm.ToLowerInvariant())
        $startIdx = $order.IndexOf($startFrom.ToLowerInvariant())
        if ($algorithmIdx -ge 0 -and $startIdx -ge 0 -and $algorithmIdx -lt $startIdx) {
            return "skipped/no-record"
        }
    }

    return "queued"
}

function Show-Status {
    $status = Read-JsonFile -Path $StatusPath
    if ($null -eq $status) {
        Write-Host "No existe estado: $StatusPath" -ForegroundColor Yellow
        return
    }

    Write-Host "Estado global: $($status.status)" -ForegroundColor Green
    Write-Host "Dataset: $($status.dataset)"
    Write-Host "Escenario: $($status.scenario) | Seed: $($status.seed) | Episodios: $($status.episodes) | Pasos: $($status.num_env_steps)"
    Write-Host "CUDA: $($status.cuda) | Torch: $($status.torch)"

    Write-Host ""
    Write-Host "Plan completo por eje y MADRL" -ForegroundColor Cyan
    $algorithms = @("happo", "masac", "matd3", "maac")
    $scenarios = @()
    if ($status.scenarios) {
        $scenarios = @($status.scenarios)
    }
    elseif ($status.scenario -eq "ALL") {
        $scenarios = @("E1", "E2", "E3")
    }
    else {
        $scenarios = @($status.scenario)
    }

    foreach ($scenarioName in $scenarios) {
        $labels = @()
        foreach ($algorithmName in $algorithms) {
            $state = Get-AlgorithmPlanState -Status $status -Algorithm $algorithmName -Scenario $scenarioName
            $labels += ("{0}:{1}" -f $algorithmName, $state)
        }

        Write-Host ("{0}: {1}" -f $scenarioName, ($labels -join " | "))
    }

    Write-Host ""
    Write-Host "Jobs iniciados" -ForegroundColor Cyan
    foreach ($job in $status.jobs) {
        $state = if ($null -eq $job.completed_at) {
            "running"
        }
        elseif ($job.exit_code -eq 0 -and $job.skipped) {
            "skipped/done"
        }
        elseif ($job.exit_code -eq 0) {
            "completed"
        }
        else {
            "failed"
        }
        $scenarioName = if ($job.scenario) { $job.scenario } else { $status.scenario }
        Write-Host ("{0,-3} {1,-8} {2,-10} start={3} end={4} exit={5}" -f $scenarioName, $job.name, $state, $job.started_at, $job.completed_at, $job.exit_code)
    }
}

function Get-ActiveJob {
    $status = Read-JsonFile -Path $StatusPath
    if ($null -eq $status -or $null -eq $status.jobs) {
        return $null
    }

    $jobs = @($status.jobs)
    foreach ($job in $jobs) {
        if ($null -eq $job.completed_at) {
            return $job
        }
    }

    $latestProgressJob = $null
    $latestProgressTime = $null
    foreach ($job in $jobs) {
        $progressPath = Join-Path (Join-Path $ProjectRoot $job.output_dir) "live_progress.json"
        if (-not (Test-Path -LiteralPath $progressPath)) {
            continue
        }

        $progressTime = (Get-Item -LiteralPath $progressPath).LastWriteTime
        if ($null -eq $latestProgressTime -or $progressTime -gt $latestProgressTime) {
            $latestProgressTime = $progressTime
            $latestProgressJob = $job
        }
    }

    if ($null -ne $latestProgressJob) {
        return $latestProgressJob
    }

    if ($jobs.Count -gt 0) {
        return $jobs[$jobs.Count - 1]
    }

    return $null
}

function Get-NumericValue {
    param(
        $Value,
        [double]$Default = 0
    )

    if ($null -eq $Value -or $Value -eq "") {
        return $Default
    }

    try {
        return [double]$Value
    }
    catch {
        return $Default
    }
}

function Show-TrainingProgress {
    $job = Get-ActiveJob
    if ($null -eq $job) {
        return
    }

    Write-Host ""
    Write-Host "Progreso, metricas y recompensas" -ForegroundColor Cyan
    Write-Host "MADRL activo: $($job.name.ToUpper())"
    Write-Host "Directorio: $($job.output_dir)"

    $runDir = Join-Path $ProjectRoot $job.output_dir
    $resultsPath = Join-Path $runDir "results.json"
    $summaryPath = Join-Path $runDir "training_summary.json"
    $tracePath = Join-Path $runDir "trace.csv"
    $timeseriesPath = Join-Path $runDir "timeseries.csv"
    $liveProgressPath = Join-Path $runDir "live_progress.json"
    $checkpointManifestPath = Join-Path $runDir "checkpoint_manifest.json"
    $status = Read-JsonFile -Path $StatusPath

    if (-not (Test-Path -LiteralPath $resultsPath)) {
        $resultsPath = Join-Path $runDir "data\results.json"
    }
    if (-not (Test-Path -LiteralPath $summaryPath)) {
        $summaryPath = Join-Path $runDir "data\training_summary.json"
    }
    if (-not (Test-Path -LiteralPath $tracePath)) {
        $tracePath = Join-Path $runDir "data\trace.csv"
    }
    if (-not (Test-Path -LiteralPath $timeseriesPath)) {
        $timeseriesPath = Join-Path $runDir "data\timeseries.csv"
    }
    if (-not (Test-Path -LiteralPath $checkpointManifestPath)) {
        $checkpointManifestPath = Join-Path $runDir "data\checkpoint_manifest.json"
    }

    $summary = Read-JsonFile -Path $summaryPath
    if ($null -eq $summary) {
        $summary = Read-JsonFile -Path $resultsPath
    }

    if ($null -ne $summary) {
        if ($summary.hyperparameters) {
            Write-Host "Hiperparametros principales:" -ForegroundColor DarkCyan
            $summary.hyperparameters.PSObject.Properties |
                Select-Object -First 12 |
                ForEach-Object { Write-Host ("  {0}: {1}" -f $_.Name, $_.Value) }
        }

        if ($summary.project_axis_metrics) {
            Write-Host "KPIs por eje:" -ForegroundColor DarkCyan
            $summary.project_axis_metrics.PSObject.Properties |
                ForEach-Object {
                    $axis = $_.Name
                    Write-Host "  [$axis]"
                    $_.Value.PSObject.Properties |
                        Select-Object -First 8 |
                        ForEach-Object { Write-Host ("    {0}: {1}" -f $_.Name, $_.Value) }
                }
        }

        $artifactProfile = $null
        if ($summary.artifact_profile) {
            $artifactProfile = $summary.artifact_profile
        }
        elseif ($summary.artifacts -and $summary.artifacts.artifact_profile) {
            $artifactProfile = $summary.artifacts.artifact_profile
        }
        if ($artifactProfile) {
            Write-Host ("Artefactos: profile={0}" -f $artifactProfile) -ForegroundColor DarkCyan
            if ($summary.artifacts -and $summary.artifacts.artifact_write_policy) {
                $policy = $summary.artifacts.artifact_write_policy
                Write-Host ("  trace_interval={0} trace_detail={1} root_trace_csv={2}" -f $policy.trace_record_interval, $policy.trace_detail, $policy.root_trace_csv)
            }
        }
    }
    else {
        Write-Host "Aun no hay results/training_summary; se escriben al cerrar el MADRL activo." -ForegroundColor Yellow
    }

    $liveProgress = Read-JsonFile -Path $liveProgressPath
    if ($null -ne $liveProgress) {
        $instantRewardSum = $liveProgress.reward_sum
        $instantRewardMean = $liveProgress.reward_mean
        if ($null -ne $liveProgress.instant_reward_sum) {
            $instantRewardSum = $liveProgress.instant_reward_sum
        }
        if ($null -ne $liveProgress.instant_reward_mean) {
            $instantRewardMean = $liveProgress.instant_reward_mean
        }

        Write-Host "Progreso vivo:" -ForegroundColor DarkCyan
        $episodeTotal = [int](Get-NumericValue -Value $status.episode_time_steps -Default 0)
        $totalSteps = [int](Get-NumericValue -Value $status.num_env_steps -Default 0)
        $totalEpisodes = [int](Get-NumericValue -Value $status.episodes -Default 0)
        $episodeNumber = [int](Get-NumericValue -Value $liveProgress.episode -Default 0) + 1
        $episodeRecorded = [int](Get-NumericValue -Value $liveProgress.episode_steps_recorded -Default ((Get-NumericValue -Value $liveProgress.episode_step -Default 0) + 1))
        $globalRecorded = [int](Get-NumericValue -Value $liveProgress.total_steps_recorded -Default ((Get-NumericValue -Value $liveProgress.global_step -Default 0) + 1))
        $episodePct = if ($episodeTotal -gt 0) { [Math]::Round(100.0 * $episodeRecorded / $episodeTotal, 2) } else { 0 }
        $globalPct = if ($totalSteps -gt 0) { [Math]::Round(100.0 * $globalRecorded / $totalSteps, 2) } else { 0 }
        Write-Host ("  episodio={0}/{1} paso_episodio={2}/{3} ({4}%) paso_global={5}/{6} ({7}%)" -f $episodeNumber, $totalEpisodes, $episodeRecorded, $episodeTotal, $episodePct, $globalRecorded, $totalSteps, $globalPct)
        Write-Host ("  global_step={0} episode_step={1} time_step={2}" -f $liveProgress.global_step, $liveProgress.episode_step, $liveProgress.time_step)
        if ($liveProgress.reward_function) {
            $weights = $liveProgress.reward_axis_weights
            if ($weights) {
                Write-Host ("  multiobjetivo: OE1_flex={0} OE2_CO2={1} OE3_costo={2}" -f $weights.flex, $weights.carbon, $weights.cost)
            }
            Write-Host ("  reward_function={0} profile={1}" -f $liveProgress.reward_function, $liveProgress.reward_profile)
        }
        Write-Host ("  instant_reward_sum={0} instant_reward_mean={1}" -f $instantRewardSum, $instantRewardMean)
        if ($null -ne $liveProgress.episode_return_cumulative) {
            Write-Host ("  episode_return_cumulative={0} episode_reward_mean_cumulative={1} episode_steps={2}" -f $liveProgress.episode_return_cumulative, $liveProgress.episode_reward_mean_cumulative, $liveProgress.episode_steps_recorded)
            Write-Host ("  total_return_cumulative={0} total_reward_mean_cumulative={1} total_steps={2}" -f $liveProgress.total_return_cumulative, $liveProgress.total_reward_mean_cumulative, $liveProgress.total_steps_recorded)
        }
        else {
            Write-Host "  retornos acumulados: aun no disponibles para este proceso; apareceran al cargar el codigo nuevo en el siguiente MADRL/job." -ForegroundColor Yellow
        }
        Write-Host ("  energia_inst: cost={0} co2={1} net_load={2}" -f $liveProgress.district_net_electricity_consumption_cost, $liveProgress.district_net_electricity_consumption_emission, $liveProgress.district_net_electricity_consumption)
        if ($null -ne $liveProgress.reward_district_import_kwh) {
            Write-Host ("  reward_signal: district_import_kwh={0} team_reward={1}" -f $liveProgress.reward_district_import_kwh, $liveProgress.reward_team_reward)
        }
        Write-Host ("  price_mean={0} carbon_intensity_mean={1}" -f $liveProgress.electricity_price_mean, $liveProgress.carbon_intensity_mean)
    }
    else {
        Write-Host "Progreso vivo aun no disponible; aparece despues del primer intervalo de pasos." -ForegroundColor Yellow
    }

    if (Test-Path $tracePath) {
        Write-Host "Ultimos pasos/rewards en trace.csv:" -ForegroundColor DarkCyan
        Import-Csv $tracePath | Select-Object -Last 8 | Format-Table -AutoSize
    }
    elseif (Test-Path $timeseriesPath) {
        Write-Host "Ultimos registros en timeseries.csv:" -ForegroundColor DarkCyan
        Import-Csv $timeseriesPath | Select-Object -Last 8 | Format-Table -AutoSize
    }
    else {
        Write-Host "Trace/timeseries aun no disponibles para este MADRL." -ForegroundColor Yellow
    }

    if (Test-Path $checkpointManifestPath) {
        Write-Host "Checkpoint manifest:" -ForegroundColor DarkCyan
        $checkpointManifest = Read-JsonFile -Path $checkpointManifestPath
        if ($checkpointManifest) {
            $checkpointManifest.PSObject.Properties |
                Select-Object -First 10 |
                ForEach-Object { Write-Host ("  {0}: {1}" -f $_.Name, $_.Value) }
        }
    }

    $eventFiles = Get-ChildItem $runDir -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "events.out.tfevents*" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 3
    if ($eventFiles) {
        Write-Host "TensorBoard/eventos recientes:" -ForegroundColor DarkCyan
        $eventFiles | Select-Object FullName, Length, LastWriteTime | Format-Table -AutoSize
    }
}

function Show-Logs {
    if (-not (Test-Path $LogDir)) {
        return
    }

    Write-Host ""
    Write-Host "Logs recientes" -ForegroundColor Cyan
    Get-ChildItem $LogDir -File | Sort-Object LastWriteTime -Descending | Select-Object -First 4 |
        ForEach-Object {
            Write-Host ""
            Write-Host "--- $($_.Name) | $($_.Length) bytes | $($_.LastWriteTime) ---" -ForegroundColor DarkCyan
            if ($_.Length -gt 0) {
                $lines = Get-Content $_.FullName -Tail ([Math]::Max($LogTail * 8, 200)) |
                    Where-Object {
                        $line = $_.Trim()
                        # Filtra ruido de inicializacion: arrays numericos, Box(...), shapes, dtypes
                        $line -ne "" -and
                        $line -notmatch "^Box\(" -and
                        $line -notmatch "^\[?[\s\d\.\-e]+\]?,?" -and
                        $line -notmatch "^\([\d,\s]+\),?\s*(float|int|bool)" -and
                        $line -notmatch "float(32|64)\)" -and
                        $line -notmatch "^\s*[\d\.]+\s+[\d\.]+\s+[\d\.]" -and
                        $line -notmatch "share_observation_space|observation_space" -and
                        $line -notmatch "^[\s\[\]01\. ,eE\+\-]+$"
                    } |
                    Select-Object -Last $LogTail

                if ($lines) {
                    $lines | ForEach-Object {
                        if ($_.Length -gt 220) {
                            Write-Host ($_.Substring(0, 220) + " ...")
                        }
                        else {
                            Write-Host $_
                        }
                    }
                }
                else {
                    Write-Host "(solo salida de inicializacion de espacios; ocultada por el monitor)"
                }
            }
            else {
                Write-Host "(sin salida escrita todavía)"
            }
        }
}

function Show-Artifacts {
    Write-Host ""
    Write-Host "Artefactos y checkpoints" -ForegroundColor Cyan

    # Archivos de resultados/metricas
    $resultFiles = Get-ChildItem $OutputRootPath -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -in @("results.json","timeseries.csv","trace.csv","checkpoint_manifest.json",
                          "figures_manifest.json","episode_summary.csv","kpis.csv","live_progress.json") -or
            $_.Extension -in @(".pkl",".pt",".pth",".ckpt")
        } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 10

    if ($resultFiles) {
        $resultFiles | ForEach-Object {
            $rel = $_.FullName.Substring($OutputRootPath.Length).TrimStart("\")
            if ($rel.Length -gt 85) { $rel = "..." + $rel.Substring($rel.Length - 82) }
            [pscustomobject]@{
                Archivo  = $rel
                KB       = [math]::Round($_.Length / 1KB, 1)
                Hora     = $_.LastWriteTime.ToString("HH:mm:ss")
            }
        } | Format-Table -AutoSize
    }

    # Directorios de checkpoints
    $ckptDirs = Get-ChildItem $OutputRootPath -Recurse -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match "\\checkpoints\\" -or $_.Name -eq "checkpoints" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 6

    if ($ckptDirs) {
        Write-Host "  Checkpoints guardados:" -ForegroundColor DarkCyan
        $ckptDirs | ForEach-Object {
            $rel = $_.FullName.Substring($OutputRootPath.Length).TrimStart("\")
            $nFiles = (Get-ChildItem $_.FullName -File -ErrorAction SilentlyContinue).Count
            Write-Host ("    {0}  ({1} archivos, mod {2})" -f $rel, $nFiles, $_.LastWriteTime.ToString("HH:mm:ss"))
        }
    } else {
        Write-Host "  Checkpoints: aun no disponibles (se crean al finalizar episodio 1)"
    }
}

while ($true) {
    Clear-MonitorHost
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  CITYLEARN v3 MADRL - MONITOR EN TIEMPO REAL" -ForegroundColor Cyan
    Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')   Intervalo: ${IntervalSeconds}s" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  OutputRoot: $OutputRoot" -ForegroundColor DarkGray

    # Resumen rapido del job activo
    $statusNow = Read-JsonFile -Path $StatusPath
    if ($statusNow) {
        $activeJob = $statusNow.jobs | Where-Object { $null -eq $_.completed_at } | Select-Object -First 1
        if ($activeJob) {
            $lpPath = Join-Path $ProjectRoot (Join-Path $activeJob.output_dir "live_progress.json")
            $lp = Read-JsonFile -Path $lpPath
            if ($lp) {
                $totalSteps = [int]$statusNow.num_env_steps
                $pct = if ($totalSteps -gt 0) { [math]::Round($lp.global_step / $totalSteps * 100, 1) } else { 0 }
                $w = $lp.reward_axis_weights
                $progressAgeSeconds = Get-FileAgeSeconds -Path $lpPath
                $processSnapshot = Get-ActivePythonProcessSnapshot -StartedAt $activeJob.started_at
                Write-Host ""
                Write-Host ("  ACTIVO : {0,-8} {1}   Paso {2}/{3} ({4}%)" -f $activeJob.name.ToUpper(), $activeJob.scenario, $lp.global_step, $totalSteps, $pct) -ForegroundColor Green
                if ($processSnapshot) {
                    Write-Host ("  Proceso: PID activo={0}  CPU={1:N1}s  candidatos={2}" -f $processSnapshot.ActivePid, $processSnapshot.ActiveCpu, $processSnapshot.CandidateCount) -ForegroundColor Green
                }
                Write-Host ("  Ep {0}/{1}  PasoEp {2}/{3}   R_mean={4:F5}   Retorno={5:F1}" -f ($lp.episode+1), $statusNow.episodes, $lp.episode_step, $statusNow.episode_time_steps, $lp.episode_reward_mean_cumulative, $lp.episode_return_cumulative) -ForegroundColor Green
                Write-Host ("  Pesos  : OE1_flex={0:F3}  OE2_co2={1:F3}  OE3_cost={2:F3}" -f $w.flex, $w.carbon, $w.cost) -ForegroundColor Yellow
                $rewardImportText = "n/d"
                if ($null -ne $lp.reward_district_import_kwh) {
                    $rewardImportText = "{0:F1}" -f [double]$lp.reward_district_import_kwh
                }
                Write-Host ("  CO2={0:F4} kg/kWh   Precio={1:F4} $/kWh   Carga_neta_inst={2:F1} kWh   Import_reward={3} kWh" -f $lp.carbon_intensity_mean, $lp.electricity_price_mean, $lp.district_net_electricity_consumption, $rewardImportText) -ForegroundColor Yellow
                if ($null -ne $progressAgeSeconds) {
                    $ageColor = if ($progressAgeSeconds -gt [math]::Max(120, [int]$statusNow.live_progress_interval * 2)) { "Yellow" } else { "DarkGray" }
                    Write-Host ("  live_progress: hace {0:N1} s" -f $progressAgeSeconds) -ForegroundColor $ageColor
                    if ($progressAgeSeconds -gt [math]::Max(120, [int]$statusNow.live_progress_interval * 2)) {
                        Write-Host "  Aviso: hay proceso activo, pero no hay pasos nuevos escritos; MASAC puede estar actualizando redes entre episodios." -ForegroundColor Yellow
                    }
                }
            }
        }
    }

    Write-Host ""
    Show-Status
    Show-Gpu
    Show-TrainingProgress
    Show-Artifacts
    Show-Logs
    Write-Host ""
    Write-Host "  Actualiza cada $IntervalSeconds s. Presiona Ctrl+C para salir." -ForegroundColor DarkGray
    Start-Sleep -Seconds $IntervalSeconds
}
