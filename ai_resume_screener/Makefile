# =============================================================================
# AI Resume Screener - Makefile
# =============================================================================
# Common commands for development, testing, and deployment
# =============================================================================

.PHONY: help install dev test lint format clean build up down logs shell backup restore deploy

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
DOCKER_COMPOSE := docker-compose
DOCKER_COMPOSE_DEV := docker-compose -f docker-compose.yml -f docker-compose.dev.yml
PROJECT_NAME := ai-resume-screener
VERSION := $(shell grep -E '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Help target
help: ## Show this help message
	@echo "$(BLUE)AI Resume Screener - Available Commands$(NC)"
	@echo "======================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# ===== DEVELOPMENT SETUP =====
install: ## Install dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements/dev.txt

install-prod: ## Install production dependencies only
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements/prod.txt

setup: ## Run initial setup
	@echo "$(BLUE)Running initial setup...$(NC)"
	chmod +x scripts/*.sh
	./scripts/setup.sh

download-models: ## Download spaCy models
	@echo "$(BLUE)Downloading spaCy models...$(NC)"
	$(PYTHON) scripts/download_models.py

# ===== DEVELOPMENT =====
dev: ## Start development server
	@echo "$(BLUE)Starting development server...$(NC)"
	./scripts/run_dev.sh

dev-docker: ## Start development environment with Docker
	@echo "$(BLUE)Starting development environment with Docker...$(NC)"
	$(DOCKER_COMPOSE_DEV) up --build

shell: ## Open shell in running container
	@echo "$(BLUE)Opening shell in app container...$(NC)"
	$(DOCKER_COMPOSE) exec app /bin/bash

shell-dev: ## Open shell in development container
	@echo "$(BLUE)Opening shell in development container...$(NC)"
	$(DOCKER_COMPOSE_DEV) exec app /bin/bash

# ===== TESTING =====
test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/integration/ -v

test-docker: ## Run tests in Docker
	@echo "$(BLUE)Running tests in Docker...$(NC)"
	$(DOCKER_COMPOSE) --profile testing run --rm test-runner

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	pytest-watch tests/ -- -v

# ===== CODE QUALITY =====
lint: ## Run linting
	@echo "$(BLUE)Running linting...$(NC)"
	flake8 app/ tests/
	mypy app/

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	black app/ tests/
	isort app/ tests/

format-check: ## Check code formatting
	@echo "$(BLUE)Checking code formatting...$(NC)"
	black --check app/ tests/
	isort --check-only app/ tests/

security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	bandit -r app/
	safety check

pre-commit: ## Run pre-commit hooks
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

# ===== DOCKER OPERATIONS =====
build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	$(DOCKER_COMPOSE) build

build-no-cache: ## Build Docker images without cache
	@echo "$(BLUE)Building Docker images without cache...$(NC)"
	$(DOCKER_COMPOSE) build --no-cache

up: ## Start all services
	@echo "$(BLUE)Starting all services...$(NC)"
	$(DOCKER_COMPOSE) up -d

up-build: ## Start all services and build
	@echo "$(BLUE)Starting all services with build...$(NC)"
	$(DOCKER_COMPOSE) up -d --build

down: ## Stop all services
	@echo "$(BLUE)Stopping all services...$(NC)"
	$(DOCKER_COMPOSE) down

down-volumes: ## Stop all services and remove volumes
	@echo "$(YELLOW)Stopping all services and removing volumes...$(NC)"
	$(DOCKER_COMPOSE) down -v

restart: ## Restart all services
	@echo "$(BLUE)Restarting all services...$(NC)"
	$(DOCKER_COMPOSE) restart

logs: ## Show logs for all services
	@echo "$(BLUE)Showing logs...$(NC)"
	$(DOCKER_COMPOSE) logs -f

logs-app: ## Show logs for app service only
	@echo "$(BLUE)Showing app logs...$(NC)"
	$(DOCKER_COMPOSE) logs -f app

# ===== DATABASE OPERATIONS =====
db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	$(DOCKER_COMPOSE) exec app alembic upgrade head

db-migration: ## Create new database migration
	@echo "$(BLUE)Creating new migration...$(NC)"
	@read -p "Migration message: " message; \
	$(DOCKER_COMPOSE) exec app alembic revision --autogenerate -m "$$message"

db-reset: ## Reset database
	@echo "$(YELLOW)Resetting database...$(NC)"
	$(DOCKER_COMPOSE) exec app alembic downgrade base
	$(DOCKER_COMPOSE) exec app alembic upgrade head

db-shell: ## Open database shell
	@echo "$(BLUE)Opening database shell...$(NC)"
	$(DOCKER_COMPOSE) exec db psql -U postgres -d resume_screener

# ===== BACKUP AND RESTORE =====
backup: ## Create database backup
	@echo "$(BLUE)Creating database backup...$(NC)"
	mkdir -p backups
	$(DOCKER_COMPOSE) exec db pg_dump -U postgres resume_screener > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)Backup created in backups/ directory$(NC)"

restore: ## Restore database from backup
	@echo "$(YELLOW)Restoring database from backup...$(NC)"
	@read -p "Backup file path: " backup_file; \
	$(DOCKER_COMPOSE) exec -T db psql -U postgres -d resume_screener < $$backup_file

backup-files: ## Backup uploaded files
	@echo "$(BLUE)Creating files backup...$(NC)"
	tar -czf backups/files_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz uploads/ data/ logs/

# ===== MONITORING =====
stats: ## Show container stats
	@echo "$(BLUE)Container statistics:$(NC)"
	docker stats --no-stream

health: ## Check service health
	@echo "$(BLUE)Checking service health...$(NC)"
	curl -f http://localhost:8000/health || echo "$(RED)Health check failed$(NC)"

monitoring: ## Open monitoring dashboards
	@echo "$(BLUE)Opening monitoring dashboards...$(NC)"
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"
	@echo "Redis Commander: http://localhost:8082"
	@echo "Adminer: http://localhost:8081"

# ===== DEPLOYMENT =====
deploy-staging: ## Deploy to staging
	@echo "$(BLUE)Deploying to staging...$(NC)"
	./scripts/deploy.sh staging

deploy-prod: ## Deploy to production
	@echo "$(YELLOW)Deploying to production...$(NC)"
	./scripts/deploy.sh production

# ===== CLEANUP =====
clean: ## Clean up development environment
	@echo "$(BLUE)Cleaning up...$(NC)"
	$(DOCKER_COMPOSE) down -v --remove-orphans
	docker system prune -f
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage

clean-all: ## Clean everything including images
	@echo "$(YELLOW)Cleaning everything...$(NC)"
	$(DOCKER_COMPOSE) down -v --remove-orphans --rmi all
	docker system prune -af --volumes

# ===== UTILITIES =====
requirements: ## Update requirements.txt
	@echo "$(BLUE)Updating requirements.txt...$(NC)"
	$(PIP) freeze > requirements.txt

check: ## Run all checks (lint, format, security, test)
	@echo "$(BLUE)Running all checks...$(NC)"
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) security
	$(MAKE) test

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	mkdocs build

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	mkdocs serve

version: ## Show current version
	@echo "$(BLUE)Current version: $(VERSION)$(NC)"

# ===== CI/CD =====
ci: ## Run CI pipeline locally
	@echo "$(BLUE)Running CI pipeline...$(NC)"
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) security
	$(MAKE) test
	$(MAKE) build

# ===== ENVIRONMENT INFO =====
info: ## Show environment information
	@echo "$(BLUE)Environment Information:$(NC)"
	@echo "Project: $(PROJECT_NAME)"
	@echo "Version: $(VERSION)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Docker: $(shell docker --version)"
	@echo "Docker Compose: $(shell $(DOCKER_COMPOSE) --version)"
	@echo ""
	@echo "$(BLUE)Running Services:$(NC)"
	@$(DOCKER_COMPOSE) ps 2>/dev/null || echo "No services running"
