# =============================================================================
# AI Resume Screener - Windows Development Server Script
# =============================================================================

param(
    [string]$ServerHost = "0.0.0.0",
    [int]$Port = 8000,
    [string]$LogLevel = "info",
    [switch]$NoReload,
    [switch]$Check,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Configuration
$VENV_NAME = "venv"

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

function Test-AppModule {
    if (-not (Test-Path "app\main.py")) {
        Write-Error-Custom "App module not found. Please ensure app\main.py exists."
        exit 1
    }
}

function New-ProjectDirectories {
    $directories = @("logs", "uploads", "models")
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -Path $dir -ItemType Directory -Force | Out-Null
            Write-Status "Created directory: $dir"
        }
    }
}

function Test-Dependencies {
    Write-Status "Checking dependencies..."
    
    try {
        python -c "import fastapi, uvicorn" 2>$null
        Write-Success "Dependencies check passed"
    }
    catch {
        Write-Error-Custom "Required dependencies not found. Please run: pip install -r requirements.txt"
        exit 1
    }
}

function Import-Environment {
    if (Test-Path ".env") {
        Write-Success "Environment variables loaded from .env"
    }
    else {
        Write-Warning "No .env file found. Using default configuration."
    }
}

function Test-Port {
    param([int]$PortNumber)
    
    try {
        $connection = Test-NetConnection -ComputerName localhost -Port $PortNumber -InformationLevel Quiet -WarningAction SilentlyContinue
        if ($connection) {
            Write-Warning "Port $PortNumber is already in use"
            $killProcess = Read-Host "Kill existing process and continue? (y/N)"
            if ($killProcess -eq "y" -or $killProcess -eq "Y") {
                $processes = Get-NetTCPConnection -LocalPort $PortNumber -ErrorAction SilentlyContinue
                foreach ($process in $processes) {
                    Stop-Process -Id $process.OwningProcess -Force
                }
                Write-Success "Existing process killed"
            }
            else {
                Write-Error-Custom "Cannot start server. Port $PortNumber is in use."
                exit 1
            }
        }
    }
    catch {
        # Port is available
    }
}

function Start-DevServer {
    Write-Status "Starting development server..."
    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Green
    Write-Host "🚀 AI Resume Screener Development Server" -ForegroundColor Green
    Write-Host "==============================================" -ForegroundColor Green
    Write-Host "Host: $ServerHost"
    Write-Host "Port: $Port"
    Write-Host "Environment: development"
    Write-Host "Hot Reload: $(-not $NoReload)"
    Write-Host "Log Level: $LogLevel"
    Write-Host ""
    Write-Host "API Documentation: http://$ServerHost`:$Port/docs"
    Write-Host "Alternative Docs: http://$ServerHost`:$Port/redoc"
    Write-Host "Health Check: http://$ServerHost`:$Port/health"
    Write-Host ""
    Write-Host "Press Ctrl+C to stop the server"
    Write-Host "==============================================" -ForegroundColor Green
    Write-Host ""

    $reloadFlag = if ($NoReload) { "" } else { "--reload" }
    
    uvicorn app.main:app --host $ServerHost --port $Port --log-level $LogLevel $reloadFlag --access-log --use-colors
}

function Show-Help {
    Write-Host "AI Resume Screener Windows Development Server"
    Write-Host ""
    Write-Host "Usage: .\scripts\run_dev_windows.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -ServerHost HOST   Server host (default: 0.0.0.0)"
    Write-Host "  -Port PORT         Server port (default: 8000)"
    Write-Host "  -LogLevel LEVEL    Log level (default: info)"
    Write-Host "  -NoReload          Disable auto-reload"
    Write-Host "  -Check             Check setup without starting server"
    Write-Host "  -Help              Show this help message"
    Write-Host ""
}

function Invoke-Checks {
    Write-Status "Running setup checks..."
    
    Test-VirtualEnvironment
    Enable-VirtualEnvironment
    Test-AppModule
    Test-Dependencies
    Import-Environment
    New-ProjectDirectories
    
    Write-Success "All checks passed!"
}

# Handle parameters
if ($Help) {
    Show-Help
    exit 0
}

if ($Check) {
    Invoke-Checks
    exit 0
}

# Main execution
try {
    Invoke-Checks
    Test-Port -PortNumber $Port
    Start-DevServer
}
catch {
    Write-Error-Custom "Failed to start development server: $($_.Exception.Message)"
    exit 1
}
# End of script