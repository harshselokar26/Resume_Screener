# =============================================================================
# AI Resume Screener - Windows PowerShell Setup Script
# =============================================================================
# This script sets up the development environment for the AI Resume Screener
# on Windows systems using PowerShell
# =============================================================================

param(
    [switch]$Dev,
    [switch]$Prod,
    [switch]$Clean,
    [switch]$Help
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Configuration
$PROJECT_NAME = "ai_resume_screener"
$PYTHON_VERSION = "3.9"
$VENV_NAME = "venv"
$REQUIREMENTS_FILE = "requirements.txt"

# Colors for output (using Write-Host with colors)
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

# Function to check if command exists
function Test-Command {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Function to check Python version
function Test-PythonVersion {
    Write-Status "Checking Python installation..."
    
    $pythonCmd = $null
    if (Test-Command "python") {
        $pythonCmd = "python"
    }
    elseif (Test-Command "python3") {
        $pythonCmd = "python3"
    }
    elseif (Test-Command "py") {
        $pythonCmd = "py"
    }
    else {
        Write-Error-Custom "Python is not installed or not in PATH. Please install Python $PYTHON_VERSION or higher."
        Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Cyan
        exit 1
    }

    # Check Python version
    $pythonVersion = & $pythonCmd --version 2>&1
    if ($pythonVersion -match "Python (\d+)\.(\d+)\.(\d+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 9)) {
            Write-Error-Custom "Python $PYTHON_VERSION or higher is required. Found: $pythonVersion"
            exit 1
        }
        
        Write-Success "Python $pythonVersion detected"
        return $pythonCmd
    }
    else {
        Write-Error-Custom "Could not determine Python version"
        exit 1
    }
}

# Function to create virtual environment
function New-VirtualEnvironment {
    param([string]$PythonCmd)
    
    Write-Status "Creating virtual environment..."
    
    if (Test-Path $VENV_NAME) {
        Write-Warning "Virtual environment already exists. Removing..."
        Remove-Item -Path $VENV_NAME -Recurse -Force
    }

    & $PythonCmd -m venv $VENV_NAME
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Failed to create virtual environment"
        exit 1
    }
    
    Write-Success "Virtual environment created: $VENV_NAME"
}

# Function to activate virtual environment
function Enable-VirtualEnvironment {
    Write-Status "Activating virtual environment..."
    
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

# Function to upgrade pip
function Update-Pip {
    Write-Status "Upgrading pip..."
    python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Failed to upgrade pip"
        exit 1
    }
    Write-Success "Pip upgraded successfully"
}

# Function to install dependencies
function Install-Dependencies {
    Write-Status "Installing Python dependencies..."
    
    if (Test-Path $REQUIREMENTS_FILE) {
        pip install -r $REQUIREMENTS_FILE
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Custom "Failed to install dependencies"
            exit 1
        }
        Write-Success "Dependencies installed successfully"
    }
    else {
        Write-Warning "Requirements file not found. Installing basic dependencies..."
        pip install fastapi uvicorn pydantic
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Custom "Failed to install basic dependencies"
            exit 1
        }
    }
}

# Function to install development dependencies
function Install-DevDependencies {
    Write-Status "Installing development dependencies..."
    
    $devRequirements = "requirements\dev.txt"
    if (Test-Path $devRequirements) {
        pip install -r $devRequirements
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Custom "Failed to install development dependencies"
            exit 1
        }
        Write-Success "Development dependencies installed"
    }
    else {
        Write-Warning "Development requirements file not found. Installing basic dev tools..."
        pip install pytest black flake8 mypy pre-commit
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Custom "Failed to install development tools"
            exit 1
        }
    }
}

# Function to create necessary directories
function New-ProjectDirectories {
    Write-Status "Creating necessary directories..."
    
    $directories = @(
        "uploads",
        "logs",
        "models",
        "data\sample_data\resumes",
        "data\sample_data\job_descriptions",
        "tests\fixtures"
    )

    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -Path $dir -ItemType Directory -Force | Out-Null
            Write-Success "Created directory: $dir"
        }
        else {
            Write-Warning "Directory already exists: $dir"
        }
    }
}

# Function to create environment file
function New-EnvironmentFile {
    Write-Status "Creating environment configuration file..."
    
    if (-not (Test-Path ".env")) {
        if (Test-Path ".env.example") {
            Copy-Item ".env.example" ".env"
            Write-Success "Environment file created from template"
        }
        else {
            $envContent = @"
# Environment Configuration
ENVIRONMENT=development
DEBUG=True

# Server Configuration
HOST=0.0.0.0
PORT=8000

# Security
SECRET_KEY=your-secret-key-here-change-in-production

# File Upload Configuration
MAX_FILE_SIZE=10485760
ALLOWED_FILE_TYPES=pdf,doc,docx
UPLOAD_DIR=uploads

# NLP Configuration
SPACY_MODEL=en_core_web_sm

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Database Configuration (Optional)
DATABASE_URL=sqlite:///./resume_screener.db

# CORS Configuration
ALLOWED_ORIGINS=*
ALLOWED_HOSTS=*
"@
            Set-Content -Path ".env" -Value $envContent
            Write-Success "Default environment file created"
        }
    }
    else {
        Write-Warning "Environment file already exists"
    }
}

# Function to download spaCy models
function Install-SpacyModels {
    Write-Status "Downloading spaCy models..."
    
    try {
        python scripts\download_models.py
        Write-Success "spaCy models downloaded successfully"
    }
    catch {
        Write-Warning "Could not download spaCy models automatically"
        Write-Status "Please run manually: python scripts\download_models.py"
    }
}

# Function to initialize git repository
function Initialize-GitRepository {
    Write-Status "Initializing Git repository..."
    
    if (-not (Test-Path ".git")) {
        git init
        if ($LASTEXITCODE -eq 0) {
            if (Test-Path ".gitignore") {
                git add .gitignore
                git commit -m "Initial commit: Add .gitignore"
                Write-Success "Git repository initialized"
            }
            else {
                Write-Warning "No .gitignore file found"
            }
        }
        else {
            Write-Warning "Git initialization failed"
        }
    }
    else {
        Write-Warning "Git repository already exists"
    }
}

# Function to install pre-commit hooks
function Install-PreCommitHooks {
    Write-Status "Installing pre-commit hooks..."
    
    if ((Test-Command "pre-commit") -and (Test-Path ".pre-commit-config.yaml")) {
        pre-commit install
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Pre-commit hooks installed"
        }
        else {
            Write-Warning "Failed to install pre-commit hooks"
        }
    }
    else {
        Write-Warning "Pre-commit not available or config file missing"
    }
}

# Function to run initial tests
function Invoke-InitialTests {
    Write-Status "Running initial tests..."
    
    if (Test-Command "pytest") {
        pytest tests\ -v --tb=short
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Initial tests completed successfully"
        }
        else {
            Write-Warning "Some tests failed"
        }
    }
    else {
        Write-Warning "Pytest not available. Skipping tests."
    }
}

# Function to display setup summary
function Show-SetupSummary {
    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Green
    Write-Host "🎉 Setup Complete!" -ForegroundColor Green
    Write-Host "==============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "1. Activate virtual environment: .\venv\Scripts\Activate.ps1"
    Write-Host "2. Update .env file with your configuration"
    Write-Host "3. Start development server: .\scripts\run_dev_windows.ps1"
    Write-Host "4. Visit: http://localhost:8000/docs"
    Write-Host ""
    Write-Host "Available scripts:"
    Write-Host "- .\scripts\run_dev_windows.ps1    - Start development server"
    Write-Host "- .\scripts\run_prod_windows.ps1   - Start production server"
    Write-Host "- python scripts\download_models.py - Download ML models"
    Write-Host ""
    Write-Host "For help, see README.md"
    Write-Host ""
}

# Function to show help
function Show-Help {
    Write-Host "AI Resume Screener Windows Setup Script"
    Write-Host ""
    Write-Host "Usage: .\scripts\setup_windows.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Help          Show this help message"
    Write-Host "  -Dev           Install development dependencies"
    Write-Host "  -Prod          Production setup (minimal dependencies)"
    Write-Host "  -Clean         Clean existing setup"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\scripts\setup_windows.ps1                # Basic setup"
    Write-Host "  .\scripts\setup_windows.ps1 -Dev           # Development setup"
    Write-Host "  .\scripts\setup_windows.ps1 -Prod          # Production setup"
    Write-Host "  .\scripts\setup_windows.ps1 -Clean         # Clean setup"
    Write-Host ""
}

# Function to clean existing setup
function Clear-Setup {
    Write-Status "Cleaning existing setup..."
    
    $itemsToRemove = @($VENV_NAME, ".env", "logs", "uploads", "models")
    
    foreach ($item in $itemsToRemove) {
        if (Test-Path $item) {
            Remove-Item -Path $item -Recurse -Force
            Write-Success "Removed: $item"
        }
    }
    
    Write-Success "Cleanup completed"
}

# Main setup function
function Start-Setup {
    Write-Host "==============================================" -ForegroundColor Blue
    Write-Host "🚀 AI Resume Screener Windows Setup" -ForegroundColor Blue
    Write-Host "==============================================" -ForegroundColor Blue
    Write-Host ""

    # Check system requirements
    Write-Status "Checking system requirements..."
    $pythonCmd = Test-PythonVersion

    # Setup virtual environment
    New-VirtualEnvironment -PythonCmd $pythonCmd
    Enable-VirtualEnvironment
    Update-Pip

    # Install dependencies
    Install-Dependencies
    
    # Install development dependencies if requested
    if ($Dev) {
        Install-DevDependencies
        Install-PreCommitHooks
    }

    # Create project structure
    New-ProjectDirectories
    New-EnvironmentFile

    # Download models
    $downloadModels = Read-Host "Download spaCy models now? (Y/n)"
    if ($downloadModels -ne "n" -and $downloadModels -ne "N") {
        Install-SpacyModels
    }

    # Git setup
    $initGit = Read-Host "Initialize Git repository? (Y/n)"
    if ($initGit -ne "n" -and $initGit -ne "N") {
        Initialize-GitRepository
    }

    # Run tests
    if (Test-Command "pytest") {
        $runTests = Read-Host "Run initial tests? (Y/n)"
        if ($runTests -ne "n" -and $runTests -ne "N") {
            Invoke-InitialTests
        }
    }

    # Display summary
    Show-SetupSummary
}

# Handle script parameters
if ($Help) {
    Show-Help
    exit 0
}

if ($Clean) {
    Clear-Setup
    exit 0
}

# Check execution policy
$executionPolicy = Get-ExecutionPolicy
if ($executionPolicy -eq "Restricted") {
    Write-Warning "PowerShell execution policy is restricted."
    Write-Host "Run this command as Administrator to allow script execution:"
    Write-Host "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Or run the script with:"
    Write-Host "PowerShell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1" -ForegroundColor Yellow
    exit 1
}

# Run main setup
try {
    Start-Setup
}
catch {
    Write-Error-Custom "Setup failed: $($_.Exception.Message)"
    exit 1
}
