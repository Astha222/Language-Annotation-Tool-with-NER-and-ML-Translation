# Language Annotation Tool - Project Starter (Windows)
# Starts the backend (FastAPI, port 8000) and the frontend (Expo web, port 8081) in two new windows.
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Language Annotation Tool - Starting  " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check MongoDB
$mongo = Get-Service -Name "MongoDB" -ErrorAction SilentlyContinue
if ($null -eq $mongo) {
    Write-Host "[!!] No MongoDB Windows service found - start MongoDB yourself (e.g. docker run -d -p 27017:27017 mongo)" -ForegroundColor Yellow
} elseif ($mongo.Status -eq "Running") {
    Write-Host "[OK] MongoDB is Running" -ForegroundColor Green
} else {
    Write-Host "[!!] Starting MongoDB..." -ForegroundColor Red
    Start-Service MongoDB -ErrorAction SilentlyContinue
}

# Check PostgreSQL (not needed if POSTGRES_URL points at SQLite)
$pg = Get-Process -Name "postgres" -ErrorAction SilentlyContinue
if ($pg) {
    Write-Host "[OK] PostgreSQL is Running" -ForegroundColor Green
} else {
    Write-Host "[!!] PostgreSQL might not be running - check manually (ignore this if you use SQLite)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting Backend (FastAPI)..." -ForegroundColor Yellow

# Start Backend
$root = $PSScriptRoot
$activate = ""
# use a virtual environment if there is one (backend\.venv as in the README, or .venv at the project root)
foreach ($venv in "$root\backend\.venv", "$root\.venv") {
    if (Test-Path "$venv\Scripts\Activate.ps1") { $activate = "& '$venv\Scripts\Activate.ps1'; "; break }
}
$backendCmd = "cd '$root\backend'; ${activate}python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload"
Start-Process -FilePath "powershell" -ArgumentList "-NoExit -Command $backendCmd"

Start-Sleep -Seconds 4

Write-Host "Starting Frontend (Expo Web)..." -ForegroundColor Yellow

# Start Frontend
$frontendCmd = "cd '$root\frontend'; npx expo start --web --port 8081"
Start-Process -FilePath "powershell" -ArgumentList "-NoExit -Command $frontendCmd"

Start-Sleep -Seconds 5

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Project Started!                      " -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Frontend : http://localhost:8081      " -ForegroundColor White
Write-Host "  Backend  : http://localhost:8000      " -ForegroundColor White
Write-Host "  API Docs : http://localhost:8000/docs " -ForegroundColor White
Write-Host ""

Start-Sleep -Seconds 3
Start-Process "http://localhost:8081"
