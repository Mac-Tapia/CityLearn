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
        Where-Object {
            (Test-Path -LiteralPath (Join-Path $_.FullName "official_full_status.json")) -or
            (Test-Path -LiteralPath (Join-Path $_.FullName "iquitos_status.json"))
        } |
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
if (-not (Test-Path -LiteralPath $StatusPath)) {
    $StatusPath = Join-Path $OutputRootPath "iquitos_status.json"
}
$LogDir = Join-Path $OutputRootPath "logs"

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $null }
    try { return Get-Content $Path -Raw | ConvertFrom-Json }
    catch { Write-Host "No se pudo leer JSON: $Path" -ForegroundColor Yellow; return $null }
}

function Show-Gpu {
    $nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidia) { return }
    Write-Host ""
    Write-Host "GPU" -ForegroundColor Cyan
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
}

function Show-Processes {
    Write-Host ""
    Write-Host "Procesos CityLearn v3 Iquitos activos" -ForegroundColor Cyan
    $processes = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
        Where-Object { $_.CommandLine -like "*train_citylearn_v3*" -and $_.CommandLine -like "*iquitos*" } |
        Select-Object ProcessId, CreationDate, CommandLine
    if ($processes) { $processes | Format-Table -AutoSize }
    else { Write-Host "No hay procesos activos." -ForegroundColor Yellow }
}

function Show-Status {
    $status = Read-JsonFile -Path $StatusPath
    if ($null -eq $status) {
        Write-Host "Estado: archivo no encontrado en $StatusPath" -ForegroundColor Yellow
        return
    }
    Write-Host ""
    Write-Host "Dataset: $($status.dataset)  |  Escenarios: $($status.scenarios -join ', ')  |  Seed: $($status.seed)" -ForegroundColor White
    Write-Host "Estado: $($status.status)  |  Inicio: $($status.started_at)" -ForegroundColor White
    if ($status.artifact_optimization) {
        Write-Host ("Artefactos: profile={0} trace_interval={1} trace_detail={2}" -f $status.artifact_optimization.profile, $status.artifact_optimization.trace_record_interval, $status.artifact_optimization.trace_detail) -ForegroundColor DarkCyan
    }
    Write-Host ""

    if ($status.jobs) {
        Write-Host "Jobs:" -ForegroundColor Cyan
        foreach ($job in $status.jobs) {
            $dur = if ($job.duration_minutes) { "$($job.duration_minutes) min" } else { "en curso..." }
            $col = if ($job.exit_code -eq 0) { "Green" } elseif ($null -eq $job.exit_code) { "Yellow" } else { "Red" }
            $exitStr = if ($null -ne $job.exit_code) { "exit=$($job.exit_code)" } else { "corriendo" }
            Write-Host "  [$($job.scenario)] $($job.name.ToUpper().PadRight(6))  $exitStr  $dur" -ForegroundColor $col
        }
    }

    # Live progress files
    Write-Host ""
    Write-Host "Progreso en vivo:" -ForegroundColor Cyan
    $algos = @("happo", "masac", "matd3", "maac")
    $scenarios = @("E1", "E2", "E3")
    foreach ($algo in $algos) {
        foreach ($sc in $scenarios) {
            $progressPath = Join-Path $OutputRootPath "$($algo.ToUpper())\$sc\live_progress.json"
            if (Test-Path $progressPath) {
                try {
                    $prog = Get-Content $progressPath -Raw | ConvertFrom-Json
                    $step = if ($null -ne $prog.global_step) { $prog.global_step } else { $prog.env_step }
                    $total = if ($null -ne $prog.num_env_steps) { $prog.num_env_steps } else { $status.num_env_steps }
                    $pct = if ($total -gt 0) { [math]::Round(100 * $step / $total, 1) } else { "?" }
                    $rewardImport = if ($null -ne $prog.reward_district_import_kwh) { "{0:N1}" -f [double]$prog.reward_district_import_kwh } else { "n/d" }
                    Write-Host "  [$sc] $($algo.ToUpper().PadRight(6))  paso $step / $total  ($pct%)  import_reward=$rewardImport kWh" -ForegroundColor Gray
                }
                catch {}
            }
        }
    }
}

function Show-LogTails {
    Write-Host ""
    Write-Host "Logs recientes ($LogTail lineas):" -ForegroundColor Cyan
    $logFiles = Get-ChildItem -Path $LogDir -Filter "*.log" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notlike "*.stderr.log" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 2
    foreach ($lf in $logFiles) {
        Write-Host "  $($lf.Name):" -ForegroundColor DarkGray
        Get-Content $lf.FullName -Tail $LogTail | ForEach-Object { Write-Host "    $_" }
    }
}

Write-Host "Monitor CityLearn v3 Iquitos — $OutputRoot" -ForegroundColor Cyan
Write-Host "Intervalo: ${IntervalSeconds}s  |  Ctrl+C para detener" -ForegroundColor DarkGray

while ($true) {
    Clear-MonitorHost
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  CityLearn v3 MADRL Iquitos — $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan

    Show-Status
    Show-Gpu
    Show-Processes
    Show-LogTails

    $statusObj = Read-JsonFile -Path $StatusPath
    if ($statusObj -and $statusObj.status -in @("completed", "failed")) {
        Write-Host ""
        Write-Host "Entrenamiento $($statusObj.status.ToUpper()). Monitor finalizado." -ForegroundColor Green
        break
    }

    Start-Sleep -Seconds $IntervalSeconds
}
