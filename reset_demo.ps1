# PowerShell script to reset all demo models, hospitals, and storage
Write-Host "========================================================" -ForegroundColor Magenta
Write-Host "  Federated Learning System -- Demo Reset Utility" -ForegroundColor Magenta
Write-Host "========================================================" -ForegroundColor Magenta

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
$PythonCmd = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

& "$PythonCmd" "scripts/reset_demo.py" $args
