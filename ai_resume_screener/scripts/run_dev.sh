#!/bin/bash

# =============================================================================
# AI Resume Screener - Development Server Script
# =============================================================================
# This script starts the development server with hot reload and debugging
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
VENV_NAME="venv"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"
RELOAD="${RELOAD:-true}"

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

# Function to check if virtual environment exists
check_virtual_environment() {
    if [ ! -d "$VENV_NAME" ]; then
        print_error "Virtual environment not found. Please run setup.sh first."
        exit 1
    fi
}

# Function to activate virtual environment
activate_virtual_environment() {
    if [ -f "$VENV_NAME/bin/activate" ]; then
        source "$VENV_NAME/bin/activate"
        print_success "Virtual environment activated"
    else
        print_error "Virtual environment activation script not found"
        exit 1
    fi
}

# Function to check if app module exists
check_app_module() {
    if [ ! -f "app/main.py" ]; then
        print_error "App module not found. Please ensure app/main.py exists."
        exit 1
    fi
}

# Function to create necessary directories
create_directories() {
    directories=("logs" "uploads" "models")
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_status "Created directory: $dir"
        fi
    done
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! python -c "import fastapi, uvicorn" 2>/dev/null; then
        print_error "Required dependencies not found. Please run: pip install -r requirements.txt"
        exit 1
    fi
    
    print_success "Dependencies check passed"
}

# Function to load environment variables
load_environment() {
    if [ -f ".env" ]; then
        export $(cat .env | grep -v '^#' | xargs)
        print_success "Environment variables loaded from .env"
    else
        print_warning "No .env file found. Using default configuration."
    fi
}

# Function to check port availability
check_port() {
    if command -v lsof >/dev/null 2>&1; then
        if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null; then
            print_warning "Port $PORT is already in use"
            read -p "Kill existing process and continue? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                lsof -ti:$PORT | xargs kill -9
                print_success "Existing process killed"
            else
                print_error "Cannot start server. Port $PORT is in use."
                exit 1
            fi
        fi
    fi
}

# Function to start development server
start_dev_server() {
    print_status "Starting development server..."
    echo ""
    echo "=============================================="
    echo -e "${GREEN}🚀 AI Resume Screener Development Server${NC}"
    echo "=============================================="
    echo "Host: $HOST"
    echo "Port: $PORT"
    echo "Environment: development"
    echo "Hot Reload: enabled"
    echo "Log Level: $LOG_LEVEL"
    echo ""
    echo "API Documentation: http://$HOST:$PORT/docs"
    echo "Alternative Docs: http://$HOST:$PORT/redoc"
    echo "Health Check: http://$HOST:$PORT/health"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo "=============================================="
    echo ""

    # Start uvicorn with development settings
    uvicorn app.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --log-level "$LOG_LEVEL" \
        --access-log \
        --use-colors \
        --loop uvloop \
        --reload-dir app \
        --reload-dir data
}

# Function to handle cleanup on exit
cleanup() {
    echo ""
    print_status "Shutting down development server..."
    print_success "Server stopped successfully"
    exit 0
}

# Function to show help
show_help() {
    echo "AI Resume Screener Development Server"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h         Show this help message"
    echo "  --host HOST        Server host (default: 0.0.0.0)"
    echo "  --port PORT        Server port (default: 8000)"
    echo "  --log-level LEVEL  Log level (default: info)"
    echo "  --no-reload        Disable auto-reload"
    echo "  --check            Check setup without starting server"
    echo ""
    echo "Environment Variables:"
    echo "  HOST               Server host"
    echo "  PORT               Server port"
    echo "  LOG_LEVEL          Logging level"
    echo "  RELOAD             Enable/disable reload (true/false)"
    echo ""
    exit 0
}

# Function to run setup checks
run_checks() {
    print_status "Running setup checks..."
    
    check_virtual_environment
    activate_virtual_environment
    check_app_module
    check_dependencies
    load_environment
    create_directories
    
    print_success "All checks passed!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --no-reload)
            RELOAD="false"
            shift
            ;;
        --check)
            run_checks
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Main execution
main() {
    # Run all checks
    run_checks
    
    # Check port availability
    check_port
    
    # Start the development server
    start_dev_server
}

# Run main function
main "$@"
