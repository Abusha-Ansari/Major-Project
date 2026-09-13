# PowerShell script to start the FL Backend Server and Desktop Client
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Starting Federated Learning Pneumonia Detection System" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Check virtual environment
$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
$VenvUvicorn = Join-Path $ScriptDir ".venv\Scripts\uvicorn.exe"

$PythonCmd = if (Test-Path $VenvPython) { $VenvPython } else { "python" }
$UvicornCmd = if (Test-Path $VenvUvicorn) { $VenvUvicorn } else { "uvicorn" }

Write-Host "[1/3] Launching FastAPI Backend Server on port 8000..." -ForegroundColor Yellow
Start-Process -FilePath "cmd.exe" -ArgumentList "/k", "title FL Server && `"$UvicornCmd`" src.api:app --host 127.0.0.1 --port 8000 --reload"

Write-Host "[2/3] Waiting for server to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

Write-Host "[3/3] Launching Hospital Desktop Client..." -ForegroundColor Yellow
Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "title FL Hospital Client && `"$PythonCmd`" src/hospital_desktop.py"

Write-Host "========================================================" -ForegroundColor Green
Write-Host "  Federated Learning System Started Successfully!" -ForegroundColor Green
Write-Host "  - Backend Server:   http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "  - Swagger API Docs: http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "  - Run '.\stop.ps1' or 'stop.bat' to terminate." -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
