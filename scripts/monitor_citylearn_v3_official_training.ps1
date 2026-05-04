param(
    [string]$OutputRoot = "outputs\citylearn_v3_madrl_official_full_cuda_v2",
    [int]$IntervalSeconds = 30,
    [int]$LogTail = 20
)

$ErrorActionPreference = "Continue"

$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptPath "..\..")
$OutputRootPath = Join-Path $ProjectRoot $OutputRoot
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
            $matches = @($status.jobs | Where-Object { $_.scenario -eq $scenarioName -and $_.name -eq $algorithmName })
            if ($matches.Count -eq 0) {
                $labels += ("{0}:queued" -f $algorithmName)
                continue
            }

            $job = $matches[-1]
            $state = if ($null -eq $job.completed_at) { "running" } elseif ($job.exit_code -eq 0) { "done" } else { "failed" }
            $labels += ("{0}:{1}" -f $algorithmName, $state)
        }

        Write-Host ("{0}: {1}" -f $scenarioName, ($labels -join " | "))
    }

    Write-Host ""
    Write-Host "Jobs iniciados" -ForegroundColor Cyan
    foreach ($job in $status.jobs) {
        $state = if ($null -eq $job.completed_at) { "running" } elseif ($job.exit_code -eq 0) { "completed" } else { "failed" }
        $scenarioName = if ($job.scenario) { $job.scenario } else { $status.scenario }
        Write-Host ("{0,-3} {1,-8} {2,-10} start={3} end={4} exit={5}" -f $scenarioName, $job.name, $state, $job.started_at, $job.completed_at, $job.exit_code)
    }
}

function Get-ActiveJob {
    $status = Read-JsonFile -Path $StatusPath
    if ($null -eq $status -or $null -eq $status.jobs) {
        return $null
    }

    foreach ($job in $status.jobs) {
        if ($null -eq $job.completed_at) {
            return $job
        }
    }

    if ($status.jobs.Count -gt 0) {
        return $status.jobs[$status.jobs.Count - 1]
    }

    return $null
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
        Write-Host ("  global_step={0} episode={1} episode_step={2} time_step={3}" -f $liveProgress.global_step, $liveProgress.episode, $liveProgress.episode_step, $liveProgress.time_step)
        Write-Host ("  instant_reward_sum={0} instant_reward_mean={1}" -f $instantRewardSum, $instantRewardMean)
        if ($null -ne $liveProgress.episode_return_cumulative) {
            Write-Host ("  episode_return_cumulative={0} episode_reward_mean_cumulative={1} episode_steps={2}" -f $liveProgress.episode_return_cumulative, $liveProgress.episode_reward_mean_cumulative, $liveProgress.episode_steps_recorded)
            Write-Host ("  total_return_cumulative={0} total_reward_mean_cumulative={1} total_steps={2}" -f $liveProgress.total_return_cumulative, $liveProgress.total_reward_mean_cumulative, $liveProgress.total_steps_recorded)
        }
        else {
            Write-Host "  retornos acumulados: aun no disponibles para este proceso; apareceran al cargar el codigo nuevo en el siguiente MADRL/job." -ForegroundColor Yellow
        }
        Write-Host ("  cost={0} co2={1} net_load={2}" -f $liveProgress.district_net_electricity_consumption_cost, $liveProgress.district_net_electricity_consumption_emission, $liveProgress.district_net_electricity_consumption)
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
                $lines = Get-Content $_.FullName -Tail ([Math]::Max($LogTail * 8, 80)) |
                    Where-Object {
                        $_ -notmatch "^\s*(Box\(|\[?-?1000000\.|1000000\.|inf\s|inf\]|inf,)"
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
    Write-Host "Artefactos recientes" -ForegroundColor Cyan
    $files = Get-ChildItem $OutputRootPath -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -in @("results.json", "timeseries.csv", "trace.csv", "checkpoint_manifest.json", "figures_manifest.json") -or $_.Extension -in @(".pkl", ".pt", ".pth", ".ckpt") } |
        Sort-Object LastWriteTime -Descending

    if (-not $files) {
        Write-Host "Aun no hay artefactos finales/checkpoints visibles."
        return
    }

    $files |
        Select-Object -First 8 |
        ForEach-Object {
            $relative = $_.FullName.Substring($OutputRootPath.Length).TrimStart("\")
            if ($relative.Length -gt 95) {
                $relative = "..." + $relative.Substring($relative.Length - 92)
            }

            [pscustomobject]@{
                Artifact = $relative
                KB = [Math]::Round($_.Length / 1KB, 1)
                Modified = $_.LastWriteTime.ToString("HH:mm:ss")
            }
        } |
        Format-Table -AutoSize

    Write-Host "Resumen de checkpoints por corrida:" -ForegroundColor DarkCyan
    $files |
        Where-Object { $_.FullName -match "\\checkpoints\\" -or $_.Extension -in @(".pkl", ".pt", ".pth", ".ckpt") } |
        ForEach-Object {
            $relative = $_.FullName.Substring($OutputRootPath.Length).TrimStart("\")
            $parts = $relative -split "\\"
            if ($parts.Count -ge 2) {
                "{0}\{1}" -f $parts[0], $parts[1]
            }
        } |
        Group-Object |
        Sort-Object Count -Descending |
        Select-Object -First 8 Name, Count |
        Format-Table -AutoSize
}

while ($true) {
    Clear-Host
    Write-Host "CityLearn v3 MADRL Monitor - $(Get-Date -Format o)" -ForegroundColor White
    Write-Host "OutputRoot: $OutputRoot" -ForegroundColor DarkGray
    Show-Status
    Show-Processes
    Show-Gpu
    Show-TrainingProgress
    Show-Artifacts
    Show-Logs
    Write-Host ""
    Write-Host "Actualiza cada $IntervalSeconds segundos. Presiona Ctrl+C para salir." -ForegroundColor DarkGray
    Start-Sleep -Seconds $IntervalSeconds
}
