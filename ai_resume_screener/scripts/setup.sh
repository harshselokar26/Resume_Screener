#!/bin/bash

# =============================================================================
# AI Resume Screener - Environment Setup Script
# =============================================================================
# This script sets up the development environment for the AI Resume Screener
# including Python virtual environment, dependencies, and initial configuration.
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="ai_resume_screener"
PYTHON_VERSION="3.9"
VENV_NAME="venv"
REQUIREMENTS_FILE="requirements.txt"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed. Please install Python $PYTHON_VERSION or higher."
        exit 1
    fi

    # Check Python version
    PYTHON_VER=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VER | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VER | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
        print_error "Python $PYTHON_VERSION or higher is required. Found: $PYTHON_VER"
        exit 1
    fi

    print_success "Python $PYTHON_VER detected"
}

# Function to create virtual environment
create_virtual_environment() {
    print_status "Creating virtual environment..."
    
    if [ -d "$VENV_NAME" ]; then
        print_warning "Virtual environment already exists. Removing..."
        rm -rf "$VENV_NAME"
    fi

    $PYTHON_CMD -m venv "$VENV_NAME"
    print_success "Virtual environment created: $VENV_NAME"
}

# Function to activate virtual environment
activate_virtual_environment() {
    print_status "Activating virtual environment..."
    
    if [ -f "$VENV_NAME/bin/activate" ]; then
        source "$VENV_NAME/bin/activate"
        print_success "Virtual environment activated"
    else
        print_error "Virtual environment activation script not found"
        exit 1
    fi
}

# Function to upgrade pip
upgrade_pip() {
    print_status "Upgrading pip..."
    pip install --upgrade pip
    print_success "Pip upgraded successfully"
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    if [ -f "$REQUIREMENTS_FILE" ]; then
        pip install -r "$REQUIREMENTS_FILE"
        print_success "Dependencies installed successfully"
    else
        print_warning "Requirements file not found. Installing basic dependencies..."
        pip install fastapi uvicorn pydantic
    fi
}

# Function to install development dependencies
install_dev_dependencies() {
    print_status "Installing development dependencies..."
    
    if [ -f "requirements/dev.txt" ]; then
        pip install -r requirements/dev.txt
        print_success "Development dependencies installed"
    else
        print_warning "Development requirements file not found. Installing basic dev tools..."
        pip install pytest black flake8 mypy pre-commit
    fi
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    directories=(
        "uploads"
        "logs"
        "models"
        "data/sample_data/resumes"
        "data/sample_data/job_descriptions"
        "tests/fixtures"
    )

    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created directory: $dir"
        else
            print_warning "Directory already exists: $dir"
        fi
    done
}

# Function to create environment file
create_env_file() {
    print_status "Creating environment configuration file..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp ".env.example" ".env"
            print_success "Environment file created from template"
        else
            cat > .env << EOF
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
EOF
            print_success "Default environment file created"
        fi
    else
        print_warning "Environment file already exists"
    fi
}

# Function to download spaCy models
download_spacy_models() {
    print_status "Downloading spaCy models..."
    
    if command_exists python; then
        python scripts/download_models.py
    else
        print_warning "Could not download spaCy models automatically"
        print_status "Please run: python scripts/download_models.py"
    fi
}

# Function to initialize git repository
init_git_repo() {
    print_status "Initializing Git repository..."
    
    if [ ! -d ".git" ]; then
        git init
        
        if [ -f ".gitignore" ]; then
            git add .gitignore
            git commit -m "Initial commit: Add .gitignore"
            print_success "Git repository initialized"
        else
            print_warning "No .gitignore file found"
        fi
    else
        print_warning "Git repository already exists"
    fi
}

# Function to install pre-commit hooks
install_pre_commit_hooks() {
    print_status "Installing pre-commit hooks..."
    
    if command_exists pre-commit && [ -f ".pre-commit-config.yaml" ]; then
        pre-commit install
        print_success "Pre-commit hooks installed"
    else
        print_warning "Pre-commit not available or config file missing"
    fi
}

# Function to run initial tests
run_initial_tests() {
    print_status "Running initial tests..."
    
    if command_exists pytest; then
        pytest tests/ -v --tb=short || print_warning "Some tests failed"
        print_success "Initial tests completed"
    else
        print_warning "Pytest not available. Skipping tests."
    fi
}

# Function to display setup summary
display_summary() {
    echo ""
    echo "=============================================="
    echo -e "${GREEN}🎉 Setup Complete!${NC}"
    echo "=============================================="
    echo ""
    echo "Next steps:"
    echo "1. Activate virtual environment: source $VENV_NAME/bin/activate"
    echo "2. Update .env file with your configuration"
    echo "3. Start development server: ./scripts/run_dev.sh"
    echo "4. Visit: http://localhost:8000/docs"
    echo ""
    echo "Available scripts:"
    echo "- ./scripts/run_dev.sh    - Start development server"
    echo "- ./scripts/run_prod.sh   - Start production server"
    echo "- ./scripts/download_models.py - Download ML models"
    echo ""
    echo "For help, see README.md or run: python -m app.main --help"
    echo ""
}

# Main setup function
main() {
    echo "=============================================="
    echo -e "${BLUE}🚀 AI Resume Screener Setup${NC}"
    echo "=============================================="
    echo ""

    # Check system requirements
    print_status "Checking system requirements..."
    check_python_version

    # Setup virtual environment
    create_virtual_environment
    activate_virtual_environment
    upgrade_pip

    # Install dependencies
    install_dependencies
    
    # Ask for development setup
    read -p "Install development dependencies? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_dev_dependencies
        install_pre_commit_hooks
    fi

    # Create project structure
    create_directories
    create_env_file

    # Download models
    read -p "Download spaCy models now? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        download_spacy_models
    fi

    # Git setup
    read -p "Initialize Git repository? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        init_git_repo
    fi

    # Run tests
    if command_exists pytest; then
        read -p "Run initial tests? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            run_initial_tests
        fi
    fi

    # Display summary
    display_summary
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "AI Resume Screener Setup Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --dev          Install development dependencies"
        echo "  --prod         Production setup (minimal dependencies)"
        echo "  --clean        Clean existing setup"
        echo ""
        exit 0
        ;;
    --dev)
        export SETUP_MODE="development"
        ;;
    --prod)
        export SETUP_MODE="production"
        ;;
    --clean)
        print_status "Cleaning existing setup..."
        rm -rf "$VENV_NAME" .env logs/ uploads/ models/
        print_success "Cleanup completed"
        exit 0
        ;;
esac

# Run main setup
main "$@"
