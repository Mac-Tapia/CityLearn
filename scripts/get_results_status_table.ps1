param(
    [string]$OutputRootPath = "",
    [string]$Format = "console"  # console, json, markdown
)

<#
.SYNOPSIS
Generate a results status table showing which jobs have completed artifacts.

Shows: results.json, timeseries.csv, trace.csv per ALGO/ESCENARIO

Status symbols:
  ✅ = File exists (complete)
  ⏳ = Directory exists but files not yet written (in progress)
  ❌ = Directory not found or error
#>

$ErrorActionPreference = "Continue"

# Algorithms and scenarios
$Algos = @("HAPPO", "MASAC", "MATD3", "MAAC")
$Scenarios = @("E1", "E2", "E3")
$FileTypes = @("results.json", "timeseries.csv", "trace.csv")

# Symbols
$SymbolComplete = "✅"
$SymbolPending = "⏳"
$SymbolMissing = "❌"

function Get-ResultsStatus {
    param(
        [string]$OutputRoot,
        [string]$Algo,
        [string]$Scenario
    )
    
    # Job directory pattern: OutputRoot/ALGO_LOWER/ESCENARIO_LOWER_seed_0/
    $jobDir = Join-Path $OutputRoot -ChildPath ($Algo.ToLower()) |
              ForEach-Object { Join-Path $_ -ChildPath "${Scenario}_seed_0" }
    
    $status = @{
        algo = $Algo
        scenario = $Scenario
        job_dir = $jobDir
        exists = $false
        results_json = $SymbolMissing
        timeseries_csv = $SymbolMissing
        trace_csv = $SymbolMissing
    }
    
    if (-not (Test-Path -LiteralPath $jobDir)) {
        return $status
    }
    
    $status.exists = $true
    
    # Check for each file type
    foreach ($fileType in $FileTypes) {
        $filePath = Join-Path $jobDir -ChildPath $fileType
        
        if (Test-Path -LiteralPath $filePath) {
            $status[$fileType.Replace(".", "_")] = $SymbolComplete
        }
        else {
            # Directory exists but file not yet written = pending
            $status[$fileType.Replace(".", "_")] = $SymbolPending
        }
    }
    
    return $status
}

function Format-ConsoleTable {
    param([array]$Results)
    
    Write-Host ""
    Write-Host "📊 Resultados por job (results.json / timeseries.csv / trace.csv)" -ForegroundColor Cyan -BackgroundColor Black
    Write-Host ""
    Write-Host "ALGO/ESCENARIO       results.json      timeseries.csv      trace.csv" -ForegroundColor White
    Write-Host ("—" * 75) -ForegroundColor Gray
    
    foreach ($result in $Results) {
        $label = "{0}/{1}" -f $result.algo, $result.scenario
        $label = $label.PadRight(20)
        
        $line = "{0}{1}          {2}              {3}" -f `
            $label, `
            $result.results_json, `
            $result.timeseries_csv, `
            $result.trace_csv
        
        Write-Host $line
    }
    
    Write-Host ""
}

function Format-JsonOutput {
    param([array]$Results)
    
    return @{
        timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
        output_root = $OutputRootPath
        summary = @{
            total_jobs = $Results.Count
            completed_results = ($Results | Where-Object { $_.results_json -eq $SymbolComplete }).Count
            completed_timeseries = ($Results | Where-Object { $_.timeseries_csv -eq $SymbolComplete }).Count
            completed_trace = ($Results | Where-Object { $_.trace_csv -eq $SymbolComplete }).Count
        }
        results = $Results
    } | ConvertTo-Json -Depth 10
}

function Format-MarkdownTable {
    param([array]$Results)
    
    $md = @"
## 📊 Resultados por job

| ALGO/ESCENARIO | results.json | timeseries.csv | trace.csv |
|---|---|---|---|
"@
    
    foreach ($result in $Results) {
        $line = "| {0}/{1} | {2} | {3} | {4} |" -f `
            $result.algo, `
            $result.scenario, `
            $result.results_json, `
            $result.timeseries_csv, `
            $result.trace_csv
        $md += "`n$line"
    }
    
    return $md
}

# Main
if ([string]::IsNullOrWhiteSpace($OutputRootPath)) {
    Write-Host "Error: -OutputRootPath is required" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path -LiteralPath $OutputRootPath)) {
    Write-Host "Error: OutputRoot not found: $OutputRootPath" -ForegroundColor Red
    exit 1
}

# Collect results
$results = @()
foreach ($algo in $Algos) {
    foreach ($scenario in $Scenarios) {
        $result = Get-ResultsStatus -OutputRoot $OutputRootPath -Algo $algo -Scenario $scenario
        $results += $result
    }
}

# Output
switch ($Format.ToLower()) {
    "json" {
        Format-JsonOutput -Results $results
    }
    "markdown" {
        Format-MarkdownTable -Results $results
    }
    default {
        Format-ConsoleTable -Results $results
    }
}
