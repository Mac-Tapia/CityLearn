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

& $PythonPath -m pip install -e $CityLearnRoot pytest dm-tree setproctitle absl-py tensorboardX matplotlib

# MARLlib 1.0.3 targets Ray/RLlib 1.8.0. CityLearn/Gymnasium need a newer
# NumPy than MARLlib's historical lock, so this project validates NumPy 1.23.5.
& $PythonPath -m pip install --force-reinstall `
    "ray[rllib]==1.8.0" `
    "gym==0.20.0" `
    "gymnasium==0.28.1" `
    "numpy==1.23.5" `
    "protobuf==3.20.3"

& $PythonPath -m pip install `
    "icecream==2.1.3" `
    "supersuit==3.2.0" `
    "pettingzoo==1.12.0" `
    "importlib-metadata>=6.0,<9" `
    "sphinx==7.4.7" `
    "nbsphinx==0.9.8"

& $PythonPath -m pip check
& $PythonPath -B (Join-Path $CityLearnRoot "scripts\check_citylearn_v3_training_ready.py") --strict
