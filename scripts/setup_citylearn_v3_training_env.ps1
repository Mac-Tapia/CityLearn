param(
    [string]$VenvName = ".venv39-citylearn-v3"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$CityLearnRoot = Join-Path $ProjectRoot "CityLearn"
$VenvPath = Join-Path $ProjectRoot $VenvName
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install uv first or use an existing Python 3.9 interpreter."
}

uv python install 3.9
uv venv --python 3.9 $VenvPath

& $PythonPath -m ensurepip --upgrade
& $PythonPath -m pip install --force-reinstall "pip==21.3.1" "setuptools==65.5.0" "wheel==0.38.0"

# Fuente unica de dependencias: requirements.txt en la raiz del proyecto
# (incluye -e ./CityLearn, -e . y las versiones fijas de compatibilidad RL).
Push-Location $ProjectRoot
try {
    & $PythonPath -m pip install -r "requirements.txt"
}
finally {
    Pop-Location
}

& $PythonPath -m pip check
& $PythonPath -B (Join-Path $CityLearnRoot "scripts\check_citylearn_v3_training_ready.py") --strict
