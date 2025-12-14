# Start RAG Web UI - Launches both API and React servers
# Run this from the project root directory

Write-Host "=== Starting RAG System Web UI ===" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-Not (Test-Path ".\.venv")) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Run setup_web.ps1 first" -ForegroundColor Yellow
    exit 1
}

# Check if web/node_modules exists
if (-Not (Test-Path ".\web\node_modules")) {
    Write-Host "❌ React dependencies not installed!" -ForegroundColor Red
    Write-Host "Run setup_web.ps1 first or: cd web; npm install" -ForegroundColor Yellow
    exit 1
}

# Check MongoDB
Write-Host "Checking MongoDB..." -ForegroundColor Cyan
try {
    $mongoTest = .\.venv\Scripts\python.exe -c "from pymongo import MongoClient; MongoClient('mongodb://localhost:27017', serverSelectionTimeoutMS=2000).admin.command('ping'); print('OK')" 2>&1
    if ($mongoTest -like "*OK*") {
        Write-Host "✓ MongoDB is running" -ForegroundColor Green
    } else {
        throw "MongoDB not responding"
    }
} catch {
    Write-Host "⚠️  MongoDB may not be running!" -ForegroundColor Yellow
    Write-Host "API server may fail to connect" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host ""
Write-Host "Starting servers..." -ForegroundColor Cyan
Write-Host ""
Write-Host "API Server will be at: http://localhost:8000" -ForegroundColor White
Write-Host "Web UI will be at: http://localhost:3000" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop both servers" -ForegroundColor Yellow
Write-Host ""
Start-Sleep -Seconds 2

# Start API server in background
Write-Host "[API] Starting FastAPI server..." -ForegroundColor Blue
$apiJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    & .\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000
}

# Wait for API to start
Write-Host "[API] Waiting for server to initialize..." -ForegroundColor Blue
Start-Sleep -Seconds 3

# Check if API started successfully
$apiOutput = Receive-Job -Job $apiJob 2>&1
if ($apiJob.State -ne "Running") {
    Write-Host "❌ API server failed to start!" -ForegroundColor Red
    Write-Host "Error: $apiOutput" -ForegroundColor Red
    Remove-Job -Job $apiJob -Force
    exit 1
}
Write-Host "[API] ✓ Server started" -ForegroundColor Green

# Start React dev server in background
Write-Host "[Web] Starting React dev server..." -ForegroundColor Blue
$webJob = Start-Job -ScriptBlock {
    Set-Location "$using:PWD\web"
    npm start
}

Write-Host "[Web] Waiting for server to initialize..." -ForegroundColor Blue
Start-Sleep -Seconds 5

# Check if React started successfully
if ($webJob.State -ne "Running") {
    Write-Host "❌ React dev server failed to start!" -ForegroundColor Red
    Write-Host "Stopping API server..." -ForegroundColor Yellow
    Stop-Job -Job $apiJob
    Remove-Job -Job $apiJob -Force
    Remove-Job -Job $webJob -Force
    exit 1
}
Write-Host "[Web] ✓ Server started" -ForegroundColor Green

Write-Host ""
Write-Host "=== Both servers are running! ===" -ForegroundColor Green
Write-Host ""
Write-Host "📡 API: http://localhost:8000/docs (Swagger UI)" -ForegroundColor Cyan
Write-Host "🌐 Web: http://localhost:3000 (React UI)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Logs will appear below. Press Ctrl+C to stop." -ForegroundColor Yellow
Write-Host ""

# Stream logs from both jobs
try {
    while ($true) {
        $apiLogs = Receive-Job -Job $apiJob 2>&1 | Where-Object { $_ }
        if ($apiLogs) {
            $apiLogs | ForEach-Object { Write-Host "[API] $_" -ForegroundColor Blue }
        }
        
        $webLogs = Receive-Job -Job $webJob 2>&1 | Where-Object { $_ }
        if ($webLogs) {
            $webLogs | ForEach-Object { Write-Host "[Web] $_" -ForegroundColor Magenta }
        }
        
        Start-Sleep -Milliseconds 500
        
        # Check if jobs are still running
        if ($apiJob.State -ne "Running") {
            Write-Host "[API] Server stopped unexpectedly" -ForegroundColor Red
            break
        }
        if ($webJob.State -ne "Running") {
            Write-Host "[Web] Server stopped unexpectedly" -ForegroundColor Red
            break
        }
    }
} finally {
    Write-Host ""
    Write-Host "Stopping servers..." -ForegroundColor Yellow
    Stop-Job -Job $apiJob -ErrorAction SilentlyContinue
    Stop-Job -Job $webJob -ErrorAction SilentlyContinue
    Remove-Job -Job $apiJob -Force -ErrorAction SilentlyContinue
    Remove-Job -Job $webJob -Force -ErrorAction SilentlyContinue
    Write-Host "✓ Servers stopped" -ForegroundColor Green
}
