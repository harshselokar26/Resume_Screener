#!/bin/bash

# =============================================================================
# AI Resume Screener - Production Server Script
# =============================================================================
# This script starts the production server with optimized settings
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Production Configuration
VENV_NAME="venv"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-4}"
LOG_LEVEL="${LOG_LEVEL:-warning}"
ACCESS_LOG="${ACCESS_LOG:-false}"
WORKER_CLASS="${WORKER_CLASS:-uvicorn.workers.UvicornWorker}"
TIMEOUT="${TIMEOUT:-30}"
KEEPALIVE="${KEEPALIVE:-5}"
MAX_REQUESTS="${MAX_REQUESTS:-1000}"
MAX_REQUESTS_JITTER="${MAX_REQUESTS_JITTER:-100}"

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

# Function to check if running as root
check_root() {
    if [ "$EUID" -eq 0 ]; then
        print_warning "Running as root is not recommended for production"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Function to check virtual environment
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

# Function to check production dependencies
check_production_dependencies() {
    print_status "Checking production dependencies..."
    
    required_packages=("fastapi" "uvicorn" "gunicorn")
    
    for package in "${required_packages[@]}"; do
        if ! python -c "import $package" 2>/dev/null; then
            print_error "Required package '$package' not found"
            print_status "Please install production requirements: pip install -r requirements/prod.txt"
            exit 1
        fi
    done
    
    print_success "Production dependencies check passed"
}

# Function to load environment variables
load_environment() {
    if [ -f ".env" ]; then
        export $(cat .env | grep -v '^#' | xargs)
        print_success "Environment variables loaded"
    else
        print_error "No .env file found. Production requires environment configuration."
        exit 1
    fi
    
    # Override environment for production
    export ENVIRONMENT="production"
    export DEBUG="false"
}

# Function to validate production configuration
validate_production_config() {
    print_status "Validating production configuration..."
    
    # Check required environment variables
    required_vars=("SECRET_KEY")
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            print_error "Required environment variable '$var' is not set"
            exit 1
        fi
    done
    
    # Check secret key strength
    if [ ${#SECRET_KEY} -lt 32 ]; then
        print_error "SECRET_KEY must be at least 32 characters long"
        exit 1
    fi
    
    # Validate other critical settings
    if [ "$DEBUG" = "true" ]; then
        print_warning "DEBUG is enabled in production environment"
    fi
    
    print_success "Production configuration validated"
}

# Function to create necessary directories with proper permissions
create_production_directories() {
    print_status "Creating production directories..."
    
    directories=("logs" "uploads" "models" "tmp")
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            chmod 755 "$dir"
            print_status "Created directory: $dir"
        fi
    done
    
    # Set proper permissions for log directory
    chmod 755 logs
    
    print_success "Production directories created"
}

# Function to check system resources
check_system_resources() {
    print_status "Checking system resources..."
    
    # Check available memory
    if command -v free >/dev/null 2>&1; then
        AVAILABLE_MEM=$(free -m | awk 'NR==2{printf "%.0f", $7}')
        RECOMMENDED_MEM=$((WORKERS * 512))  # 512MB per worker
        
        if [ "$AVAILABLE_MEM" -lt "$RECOMMENDED_MEM" ]; then
            print_warning "Available memory ($AVAILABLE_MEM MB) may be insufficient for $WORKERS workers"
            print_warning "Recommended: ${RECOMMENDED_MEM} MB"
        fi
    fi
    
    # Check disk space
    if command -v df >/dev/null 2>&1; then
        AVAILABLE_DISK=$(df . | awk 'NR==2 {print $4}')
        if [ "$AVAILABLE_DISK" -lt 1048576 ]; then  # Less than 1GB
            print_warning "Low disk space available: $(($AVAILABLE_DISK / 1024)) MB"
        fi
    fi
    
    print_success "System resources check completed"
}

# Function to optimize system settings
optimize_system_settings() {
    print_status "Applying production optimizations..."
    
    # Set Python optimizations
    export PYTHONOPTIMIZE=1
    export PYTHONDONTWRITEBYTECODE=1
    
    # Set uvloop if available
    if python -c "import uvloop" 2>/dev/null; then
        export UVLOOP_ENABLED=1
        print_success "uvloop optimization enabled"
    fi
    
    print_success "Production optimizations applied"
}

# Function to start production server with Gunicorn
start_production_server() {
    print_status "Starting production server..."
    echo ""
    echo "=============================================="
    echo -e "${GREEN}🚀 AI Resume Screener Production Server${NC}"
    echo "=============================================="
    echo "Host: $HOST"
    echo "Port: $PORT"
    echo "Workers: $WORKERS"
    echo "Worker Class: $WORKER_CLASS"
    echo "Environment: production"
    echo "Log Level: $LOG_LEVEL"
    echo "Timeout: ${TIMEOUT}s"
    echo ""
    echo "API Endpoint: http://$HOST:$PORT"
    echo "Health Check: http://$HOST:$PORT/health"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo "=============================================="
    echo ""

    # Create Gunicorn configuration
    cat > gunicorn.conf.py << EOF
# Gunicorn configuration for AI Resume Screener
import multiprocessing

# Server socket
bind = "$HOST:$PORT"
backlog = 2048

# Worker processes
workers = $WORKERS
worker_class = "$WORKER_CLASS"
worker_connections = 1000
timeout = $TIMEOUT
keepalive = $KEEPALIVE
max_requests = $MAX_REQUESTS
max_requests_jitter = $MAX_REQUESTS_JITTER

# Restart workers after this many requests, with up to jitter additional requests
preload_app = True

# Logging
loglevel = "$LOG_LEVEL"
accesslog = "logs/access.log" if $ACCESS_LOG else None
errorlog = "logs/error.log"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "ai_resume_screener"

# Server mechanics
daemon = False
pidfile = "tmp/gunicorn.pid"
user = None
group = None
tmp_upload_dir = "tmp"

# SSL (uncomment and configure for HTTPS)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"
EOF

    # Start Gunicorn
    exec gunicorn app.main:app -c gunicorn.conf.py
}

# Function to handle cleanup on exit
cleanup() {
    echo ""
    print_status "Shutting down production server..."
    
    # Remove PID file if it exists
    if [ -f "tmp/gunicorn.pid" ]; then
        rm -f tmp/gunicorn.pid
    fi
    
    # Remove Gunicorn config
    if [ -f "gunicorn.conf.py" ]; then
        rm -f gunicorn.conf.py
    fi
    
    print_success "Server stopped successfully"
    exit 0
}

# Function to show help
show_help() {
    echo "AI Resume Screener Production Server"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h              Show this help message"
    echo "  --host HOST             Server host (default: 0.0.0.0)"
    echo "  --port PORT             Server port (default: 8000)"
    echo "  --workers WORKERS       Number of worker processes (default: 4)"
    echo "  --log-level LEVEL       Log level (default: warning)"
    echo "  --timeout TIMEOUT       Worker timeout in seconds (default: 30)"
    echo "  --access-log            Enable access logging"
    echo "  --check                 Check production setup without starting"
    echo "  --validate              Validate configuration and exit"
    echo ""
    echo "Environment Variables:"
    echo "  HOST                    Server host"
    echo "  PORT                    Server port"
    echo "  WORKERS                 Number of workers"
    echo "  LOG_LEVEL               Logging level"
    echo "  SECRET_KEY              Application secret key (required)"
    echo ""
    exit 0
}

# Function to run production checks
run_production_checks() {
    print_status "Running production readiness checks..."
    
    check_root
    check_virtual_environment
    activate_virtual_environment
    check_production_dependencies
    load_environment
    validate_production_config
    create_production_directories
    check_system_resources
    optimize_system_settings
    
    print_success "Production readiness checks passed!"
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
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --access-log)
            ACCESS_LOG="true"
            shift
            ;;
        --check)
            run_production_checks
            exit 0
            ;;
        --validate)
            run_production_checks
            print_success "Production configuration is valid"
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
    # Run all production checks
    run_production_checks
    
    # Start the production server
    start_production_server
}

# Run main function
main "$@"
