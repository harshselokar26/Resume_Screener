# =============================================================================
# AI Resume Screener - Makefile
# =============================================================================
# Common development and deployment commands
# =============================================================================

# Variables
PYTHON := python
PIP := pip
DOCKER := docker
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := ai-resume-screener
VERSION := $(shell cat VERSION 2>/dev/null || echo "1.0.0")

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

# ===== HELP =====
.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)AI Resume Screener - Available Commands$(NC)"
	@echo "======================================"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ===== SETUP =====
.PHONY: setup
setup: ## Setup development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	./scripts/setup.sh --dev
	@echo "$(GREEN)✅ Setup completed!$(NC)"

.PHONY: install
install: ## Install dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✅ Dependencies installed!$(NC)"

.PHONY: install-dev
install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -r requirements/dev.txt
	@echo "$(GREEN)✅ Development dependencies installed!$(NC)"

.PHONY: models
models: ## Download required ML models
	@echo "$(BLUE)Downloading ML models...$(NC)"
	$(PYTHON) scripts/download_models.py
	@echo "$(GREEN)✅ Models downloaded!$(NC)"

# ===== DEVELOPMENT =====
.PHONY: dev
dev: ## Start development server
	@echo "$(BLUE)Starting development server...$(NC)"
	./scripts/run_dev.sh

.PHONY: dev-docker
dev-docker: ## Start development server with Docker
	@echo "$(BLUE)Starting development server with Docker...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.dev.yml up --build

.PHONY: shell
shell: ## Open Python shell with app context
	@echo "$(BLUE)Opening Python shell...$(NC)"
	$(PYTHON) -c "from app.main import app; import IPython; IPython.start_ipython(argv=[], user_ns=locals())"

# ===== TESTING =====
.PHONY: test
test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing

.PHONY: test-unit
test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/unit/ -v

.PHONY: test-integration
test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/integration/ -v

.PHONY: test-watch
test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	pytest-watch tests/ -- -v

.PHONY: coverage
coverage: ## Generate coverage report
	@echo "$(BLUE)Generating coverage report...$(NC)"
	pytest tests/ --cov=app --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)✅ Coverage report generated in htmlcov/$(NC)"

# ===== CODE QUALITY =====
.PHONY: lint
lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	flake8 app/ tests/
	mypy app/
	@echo "$(GREEN)✅ Linting completed!$(NC)"

.PHONY: format
format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	black app/ tests/
	isort app/ tests/
	@echo "$(GREEN)✅ Code formatted!$(NC)"

.PHONY: format-check
format-check: ## Check code formatting
	@echo "$(BLUE)Checking code formatting...$(NC)"
	black --check app/ tests/
	isort --check-only app/ tests/

.PHONY: security
security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	bandit -r app/
	safety check
	@echo "$(GREEN)✅ Security checks completed!$(NC)"

.PHONY: quality
quality: format-check lint security ## Run all quality checks

# ===== DATABASE =====
.PHONY: db-upgrade
db-upgrade: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	alembic upgrade head
	@echo "$(GREEN)✅ Database upgraded!$(NC)"

.PHONY: db-downgrade
db-downgrade: ## Rollback database migration
	@echo "$(BLUE)Rolling back database migration...$(NC)"
	alembic downgrade -1
	@echo "$(GREEN)✅ Database downgraded!$(NC)"

.PHONY: db-reset
db-reset: ## Reset database
	@echo "$(YELLOW)⚠️  Resetting database...$(NC)"
	$(DOCKER_COMPOSE) down -v
	$(DOCKER_COMPOSE) up -d db
	sleep 5
	alembic upgrade head
	@echo "$(GREEN)✅ Database reset!$(NC)"

# ===== DOCKER =====
.PHONY: build
build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	$(DOCKER) build -t $(PROJECT_NAME):$(VERSION) .
	$(DOCKER) tag $(PROJECT_NAME):$(VERSION) $(PROJECT_NAME):latest
	@echo "$(GREEN)✅ Docker image built!$(NC)"

.PHONY: up
up: ## Start all services
	@echo "$(BLUE)Starting all services...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✅ Services started!$(NC)"

.PHONY: down
down: ## Stop all services
	@echo "$(BLUE)Stopping all services...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✅ Services stopped!$(NC)"

.PHONY: logs
logs: ## View logs
	@echo "$(BLUE)Viewing logs...$(NC)"
	$(DOCKER_COMPOSE) logs -f

.PHONY: ps
ps: ## Show running containers
	@echo "$(BLUE)Running containers:$(NC)"
	$(DOCKER_COMPOSE) ps

# ===== DEPLOYMENT =====
.PHONY: deploy-staging
deploy-staging: ## Deploy to staging
	@echo "$(BLUE)Deploying to staging...$(NC)"
	./scripts/deploy.sh staging
	@echo "$(GREEN)✅ Deployed to staging!$(NC)"

.PHONY: deploy-prod
deploy-prod: ## Deploy to production
	@echo "$(YELLOW)⚠️  Deploying to production...$(NC)"
	./scripts/deploy.sh production
	@echo "$(GREEN)✅ Deployed to production!$(NC)"

# ===== MAINTENANCE =====
.PHONY: clean
clean: ## Clean up temporary files
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	@echo "$(GREEN)✅ Cleanup completed!$(NC)"

.PHONY: clean-docker
clean-docker: ## Clean Docker resources
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	$(DOCKER) system prune -f
	$(DOCKER) volume prune -f
	@echo "$(GREEN)✅ Docker cleanup completed!$(NC)"

.PHONY: backup
backup: ## Backup database
	@echo "$(BLUE)Creating database backup...$(NC)"
	./scripts/backup.sh
	@echo "$(GREEN)✅ Backup created!$(NC)"

.PHONY: restore
restore: ## Restore database from backup
	@echo "$(YELLOW)⚠️  Restoring database...$(NC)"
	./scripts/restore.sh $(BACKUP_FILE)
	@echo "$(GREEN)✅ Database restored!$(NC)"

# ===== DOCUMENTATION =====
.PHONY: docs
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	mkdocs build
	@echo "$(GREEN)✅ Documentation generated!$(NC)"

.PHONY: docs-serve
docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	mkdocs serve

# ===== MONITORING =====
.PHONY: health
health: ## Check application health
	@echo "$(BLUE)Checking application health...$(NC)"
	curl -f http://localhost:8000/health || echo "$(RED)❌ Health check failed$(NC)"

.PHONY: metrics
metrics: ## View application metrics
	@echo "$(BLUE)Application metrics:$(NC)"
	curl -s http://localhost:8000/metrics

# ===== UTILITIES =====
.PHONY: version
version: ## Show version information
	@echo "$(BLUE)Version Information:$(NC)"
	@echo "Project: $(PROJECT_NAME)"
	@echo "Version: $(VERSION)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Docker: $(shell $(DOCKER) --version)"

.PHONY: env
env: ## Show environment information
	@echo "$(BLUE)Environment Information:$(NC)"
	@echo "ENVIRONMENT: $(shell echo $$ENVIRONMENT)"
	@echo "DEBUG: $(shell echo $$DEBUG)"
	@echo "DATABASE_URL: $(shell echo $$DATABASE_URL | sed 's/:[^@]*@/:***@/')"

# ===== QUICK COMMANDS =====
.PHONY: quick-start
quick-start: install models dev ## Quick start for development

.PHONY: quick-test
quick-test: format-check lint test ## Quick test pipeline

.PHONY: quick-deploy
quick-deploy: quality test build deploy-staging ## Quick deployment pipeline

# ===== ADVANCED =====
.PHONY: profile
profile: ## Profile application performance
	@echo "$(BLUE)Profiling application...$(NC)"
	$(PYTHON) -m cProfile -o profile.stats -m app.main
	@echo "$(GREEN)✅ Profile saved to profile.stats$(NC)"

.PHONY: benchmark
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	pytest tests/performance/ --benchmark-only
	@echo "$(GREEN)✅ Benchmarks completed!$(NC)"

.PHONY: load-test
load-test: ## Run load tests
	@echo "$(BLUE)Running load tests...$(NC)"
	locust -f tests/performance/locustfile.py --host=http://localhost:8000
	@echo "$(GREEN)✅ Load tests completed!$(NC)"

# ===== FILE TARGETS =====
requirements.txt: requirements/base.txt requirements/prod.txt
	@echo "$(BLUE)Updating requirements.txt...$(NC)"
	cat requirements/base.txt requirements/prod.txt | sort | uniq > requirements.txt

VERSION:
	@echo "$(VERSION)" > VERSION

# ===== PHONY TARGETS =====
.PHONY: all
all: setup quality test build ## Run complete pipeline

# Check if required tools are installed
check-tools:
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "$(RED)❌ Python is not installed$(NC)"; exit 1; }
	@command -v $(DOCKER) >/dev/null 2>&1 || { echo "$(RED)❌ Docker is not installed$(NC)"; exit 1; }
	@command -v $(DOCKER_COMPOSE) >/dev/null 2>&1 || { echo "$(RED)❌ Docker Compose is not installed$(NC)"; exit 1; }
	@echo "$(GREEN)✅ All required tools are installed$(NC)"
