.DEFAULT_GOAL := help

# All python targets call the venv directly, so `make <target>` works
# from a fresh shell without activating anything.

.PHONY: help install dev test lint format compose-up compose-monitoring compose-down health clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Create .venv and install the package with dev + mcp extras
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev,mcp]"

dev: ## Run the API dev server with reload on http://localhost:8000
	.venv/bin/uvicorn app.main:create_app --factory --reload --host 0.0.0.0 --port 8000

test: ## Run the full test suite (80% coverage gate enforced)
	.venv/bin/pytest

lint: ## Ruff lint + format check + strict mypy (same shape as CI)
	.venv/bin/ruff check app/
	.venv/bin/ruff format --check app/
	.venv/bin/mypy app/

format: ## Auto-format app/ with ruff
	.venv/bin/ruff format app/

compose-up: ## Start the stack (Postgres, Redis, Qdrant, Neo4j, commerce provider, API)
	docker compose up -d

compose-monitoring: ## Start the stack plus Prometheus + Grafana
	docker compose --profile monitoring up -d

compose-down: ## Stop the stack
	docker compose down

health: ## Check the local API health endpoint
	curl -fsS http://localhost:8000/health && echo

clean: ## Remove test and tool caches
	rm -rf .pytest_cache .ruff_cache htmlcov coverage.xml
	find . -path ./.venv -prune -o -type d -name __pycache__ -prune -exec rm -rf {} +
