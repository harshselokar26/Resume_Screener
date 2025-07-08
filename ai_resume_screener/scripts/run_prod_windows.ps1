# =============================================================================
# AI Resume Screener - Windows Production Server Script
# =============================================================================

param(
    [string]$ServerHost = "0.0.0.0",
    [int]$Port = 8000,
    [int]$Workers = 4,
    [string]$LogLevel = "warning",
    [switch]$AccessLog,
    [switch]$Check,
    [switch]$Validate,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Configuration
$VENV_NAME = "venv"
$TIMEOUT = 30
$KEEPALIVE = 5
$MAX_REQUESTS = 1000
$MAX_REQUESTS_JITTER = 100

function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    
    if ($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Warning "Running as Administrator. This is not recommended for production."
        $continue = Read-Host "Continue anyway? (y/N)"
        if ($continue -ne "y" -and $continue -ne "Y") {
            exit 1
        }
    }
}

function Test-VirtualEnvironment {
    if (-not (Test-Path $VENV_NAME)) {
        Write-Error-Custom "Virtual environment not found. Please run setup_windows.ps1 first."
        exit 1
    }
}

function Enable-VirtualEnvironment {
    $activateScript = Join-Path $VENV_NAME "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
        Write-Success "Virtual environment activated"
    }
    else {
        Write-Error-Custom "Virtual environment activation script not found"
        exit 1
    }
}

function Test-ProductionDependencies {
    Write-Status "Checking production dependencies..."
    
    $requiredPackages = @("fastapi", "uvicorn", "gunicorn")
    
    foreach ($package in $requiredPackages) {
        try {
            python -c "import $package" 2>$null
        }
        catch {
            Write-Error-Custom "Required package '$package' not found"
            Write-Status "Please install production requirements: pip install -r requirements\prod.txt"
            exit 1
        }
    }
    
    Write-Success "Production dependencies check passed"
}

function Import-Environment {
    if (Test-Path ".env") {
        Write-Success "Environment variables loaded"
    }
    else {
        Write-Error-Custom "No .env file found. Production requires environment configuration."
        exit 1
    }
    
    # Override environment for production
    $env:ENVIRONMENT = "production"
    $env:DEBUG = "false"
}

function Test-ProductionConfig {
    Write-Status "Validating production configuration..."
    
    # Check required environment variables
    $requiredVars = @("SECRET_KEY")
    
    foreach ($var in $requiredVars) {
        $value = [Environment]::GetEnvironmentVariable($var)
        if (-not $value) {
            Write-Error-Custom "Required environment variable '$var' is not set"
            exit 1
        }
    }
    
    # Check secret key strength
    $secretKey = [Environment]::GetEnvironmentVariable("SECRET_KEY")
    if ($secretKey.Length -lt 32) {
        Write-Error-Custom "SECRET_KEY must be at least 32 characters long"
        exit 1
    }
    
    # Validate other critical settings
    $debug = [Environment]::GetEnvironmentVariable("DEBUG")
    if ($debug -eq "true") {
        Write-Warning "DEBUG is enabled in production environment"
    }
    
    Write-Success "Production configuration validated"
}

function New-ProductionDirectories {
    Write-Status "Creating production directories..."
    
    $directories = @("logs", "uploads", "models", "tmp")
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -Path $dir -ItemType Directory -Force | Out-Null
            Write-Status "Created directory: $dir"
        }
    }
    
    Write-Success "Production directories created"
}

function Test-SystemResources {
    Write-Status "Checking system resources..."
    
    # Check available memory
    $totalMemory = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1MB
    $recommendedMemory = $Workers * 512  # 512MB per worker
    
    if ($totalMemory -lt $recommendedMemory) {
        Write-Warning "Available memory ($([math]::Round($totalMemory)) MB) may be insufficient for $Workers workers"
        Write-Warning "Recommended: $recommendedMemory MB"
    }
    
    # Check disk space
    $diskSpace = (Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'").FreeSpace / 1GB
    if ($diskSpace -lt 1) {
        Write-Warning "Low disk space available: $([math]::Round($diskSpace, 2)) GB"
    }
    
    Write-Success "System resources check completed"
}

function Set-ProductionOptimizations {
    Write-Status "Applying production optimizations..."
    
    # Set Python optimizations
    $env:PYTHONOPTIMIZE = "1"
    $env:PYTHONDONTWRITEBYTECODE = "1"
    
    Write-Success "Production optimizations applied"
}

function Start-ProductionServer {
    Write-Status "Starting production server..."
    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Green
    Write-Host "🚀 AI Resume Screener Production Server" -ForegroundColor Green
    Write-Host "==============================================" -ForegroundColor Green
    Write-Host "Host: $ServerHost"
    Write-Host "Port: $Port"
    Write-Host "Workers: $Workers"
    Write-Host "Environment: production"
    Write-Host "Log Level: $LogLevel"
    Write-Host "Timeout: ${TIMEOUT}s"
    Write-Host ""
    Write-Host "API Endpoint: http://$ServerHost`:$Port"
    Write-Host "Health Check: http://$ServerHost`:$Port/health"
    Write-Host ""
    Write-Host "Press Ctrl+C to stop the server"
    Write-Host "==============================================" -ForegroundColor Green
    Write-Host ""

    $accessLogFlag = if ($AccessLog) { "--access-log" } else { "" }
    
    # Start with Gunicorn (if available) or Uvicorn
    try {
        gunicorn app.main:app --bind "$ServerHost`:$Port" --workers $Workers --timeout $TIMEOUT --keep-alive $KEEPALIVE --max-requests $MAX_REQUESTS --max-requests-jitter $MAX_REQUESTS_JITTER --worker-class uvicorn.workers.UvicornWorker --log-level $LogLevel $accessLogFlag
    }
    catch {
        Write-Warning "Gunicorn not available, falling back to Uvicorn"
        uvicorn app.main:app --host $ServerHost --port $Port --log-level $LogLevel $accessLogFlag
    }
}

function Show-Help {
    Write-Host "AI Resume Screener Windows Production Server"
    Write-Host ""
    Write-Host "Usage: .\scripts\run_prod_windows.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -ServerHost HOST        Server host (default: 0.0.0.0)"
    Write-Host "  -Port PORT              Server port (default: 8000)"
    Write-Host "  -Workers WORKERS        Number of worker processes (default: 4)"
    Write-Host "  -LogLevel LEVEL         Log level (default: warning)"
    Write-Host "  -AccessLog              Enable access logging"
    Write-Host "  -Check                  Check production setup without starting"
    Write-Host "  -Validate               Validate configuration and exit"
    Write-Host "  -Help                   Show this help message"
    Write-Host ""
}

function Invoke-ProductionChecks {
    Write-Status "Running production readiness checks..."
    
    Test-Administrator
    Test-VirtualEnvironment
    Enable-VirtualEnvironment
    Test-ProductionDependencies
    Import-Environment
    Test-ProductionConfig
    New-ProductionDirectories
    Test-SystemResources
    Set-ProductionOptimizations
    
    Write-Success "Production readiness checks passed!"
}

# Handle parameters
if ($Help) {
    Show-Help
    exit 0
}

if ($Check) {
    Invoke-ProductionChecks
    exit 0
}

if ($Validate) {
    Invoke-ProductionChecks
    Write-Success "Production configuration is valid"
    exit 0
}

# Main execution
try {
    Invoke-ProductionChecks
    Start-ProductionServer
}
catch {
    Write-Error-Custom "Failed to start production server: $($_.Exception.Message)"
    exit 1
}
