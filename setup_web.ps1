# Quick Setup Script for RAG Web UI
# Run this from the project root directory

Write-Host "=== RAG System Web UI Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-Not (Test-Path ".\.venv")) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Virtual environment found" -ForegroundColor Green

# Check if Node.js is installed
try {
    $nodeVersion = node --version
    Write-Host "✓ Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found!" -ForegroundColor Red
    Write-Host "Please install Node.js from https://nodejs.org/" -ForegroundColor Yellow
    exit 1
}

# Check if MongoDB is running
Write-Host ""
Write-Host "Checking MongoDB..." -ForegroundColor Cyan
try {
    $mongoTest = .\.venv\Scripts\python.exe -c "from pymongo import MongoClient; MongoClient('mongodb://localhost:27017', serverSelectionTimeoutMS=2000).admin.command('ping'); print('OK')" 2>&1
    if ($mongoTest -like "*OK*") {
        Write-Host "✓ MongoDB is running" -ForegroundColor Green
    } else {
        throw "MongoDB not responding"
    }
} catch {
    Write-Host "❌ MongoDB is not running!" -ForegroundColor Red
    Write-Host "Please start MongoDB: See docs/PROJECT_GUIDE.md" -ForegroundColor Yellow
    Write-Host "Continuing anyway..." -ForegroundColor Yellow
}

# Install web dependencies
Write-Host ""
Write-Host "Installing React dependencies..." -ForegroundColor Cyan
Set-Location web
npm install
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install npm packages" -ForegroundColor Red
    Set-Location ..
    exit 1
}
Write-Host "✓ React dependencies installed" -ForegroundColor Green
Set-Location ..

# Check if Python dependencies are installed
Write-Host ""
Write-Host "Checking Python dependencies..." -ForegroundColor Cyan
$missingDeps = @()

$requiredPackages = @("fastapi", "uvicorn", "pymongo", "faiss", "sentence_transformers", "llama_cpp")
foreach ($pkg in $requiredPackages) {
    # Attempt import; discard output, rely on exit code
    .\.venv\Scripts\python.exe -c "import $pkg" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        $missingDeps += $pkg
    }
}

if ($missingDeps.Count -gt 0) {
    Write-Host "❌ Missing Python packages: $($missingDeps -join ', ')" -ForegroundColor Red
    Write-Host "Install with: pip install fastapi uvicorn pymongo faiss-cpu sentence-transformers llama-cpp-python" -ForegroundColor Yellow
} else {
    Write-Host "✓ All Python dependencies installed" -ForegroundColor Green
}

# Summary
Write-Host ""
Write-Host "=== Setup Complete! ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start the RAG web UI:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Start API server (Terminal 1):" -ForegroundColor White
Write-Host "   .\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Start React dev server (Terminal 2):" -ForegroundColor White
Write-Host "   cd web" -ForegroundColor Gray
Write-Host "   npm start" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Open browser to: http://localhost:3000" -ForegroundColor White
Write-Host ""
Write-Host "See docs/WEB_UI_GUIDE.md for detailed instructions" -ForegroundColor Cyan

