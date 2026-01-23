<#
.SYNOPSIS
  PowerShell runner for run_experiment.py

.DESCRIPTION
  Activates venv if present and runs run_experiment.py with the provided config.

.EXAMPLE
  ./run_experiment.ps1 -Config configs\config_test.json -ExtraArgs "--some-flag"
#>

param(
  [string]$Config = "configs/config_test.json",
  [string]$ExtraArgs = ""
)

function Find-Python {
  if (Test-Path -Path "venv\Scripts\python.exe") { return Join-Path -Path (Get-Location) -ChildPath "venv\Scripts\python.exe" }
  $py = Get-Command python -ErrorAction SilentlyContinue
  if ($py) { return $py.Path }
  throw "python not found. Please install Python 3 or create a venv named 'venv'."
}

$python = Find-Python
Write-Host "Using python: $python"
Write-Host "Config: $Config"
if ($ExtraArgs) { Write-Host "Extra args: $ExtraArgs" }

 & $python run_experiment.py --config $Config $ExtraArgs
