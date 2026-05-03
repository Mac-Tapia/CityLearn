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
    Write-Host "Jobs" -ForegroundColor Cyan
    foreach ($job in $status.jobs) {
        $state = if ($null -eq $job.completed_at) { "running" } elseif ($job.exit_code -eq 0) { "completed" } else { "failed" }
        Write-Host ("{0,-8} {1,-10} start={2} end={3} exit={4}" -f $job.name, $state, $job.started_at, $job.completed_at, $job.exit_code)
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
                Get-Content $_.FullName -Tail $LogTail
            }
            else {
                Write-Host "(sin salida escrita todavía)"
            }
        }
}

function Show-Artifacts {
    Write-Host ""
    Write-Host "Artefactos recientes" -ForegroundColor Cyan
    Get-ChildItem $OutputRootPath -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -in @("results.json", "timeseries.csv", "trace.csv", "checkpoint_manifest.json", "figures_manifest.json") -or $_.Extension -in @(".pkl", ".pt", ".pth", ".ckpt") } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 12 FullName, Length, LastWriteTime |
        Format-Table -AutoSize
}

while ($true) {
    Clear-Host
    Write-Host "CityLearn v3 MADRL Monitor - $(Get-Date -Format o)" -ForegroundColor White
    Write-Host "OutputRoot: $OutputRoot" -ForegroundColor DarkGray
    Show-Status
    Show-Processes
    Show-Gpu
    Show-Artifacts
    Show-Logs
    Write-Host ""
    Write-Host "Actualiza cada $IntervalSeconds segundos. Presiona Ctrl+C para salir." -ForegroundColor DarkGray
    Start-Sleep -Seconds $IntervalSeconds
}
