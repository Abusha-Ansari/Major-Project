# PowerShell script to stop the FL Backend Server and Desktop Client
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Stopping Federated Learning System" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# 1. Kill process listening on port 8000
Write-Host "[1/2] Terminating process on port 8000 (Server)..." -ForegroundColor Yellow
$Port8000Processes = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
foreach ($PidToKill in $Port8000Processes) {
    if ($PidToKill) {
        Write-Host "Stopping Process ID: $PidToKill" -ForegroundColor DarkGray
        Stop-Process -Id $PidToKill -Force -ErrorAction SilentlyContinue
    }
}

# 2. Stop Python processes running hospital_desktop.py or uvicorn
Write-Host "[2/2] Terminating Hospital Desktop and Uvicorn instances..." -ForegroundColor Yellow
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { 
    $_.CommandLine -like "*hospital_desktop.py*" -or $_.CommandLine -like "*uvicorn*src.api:app*"
} | ForEach-Object {
    Write-Host "Stopping process: $($_.Name) (PID: $($_.ProcessId))" -ForegroundColor DarkGray
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}

Write-Host "========================================================" -ForegroundColor Green
Write-Host "  Federated Learning System Stopped." -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
