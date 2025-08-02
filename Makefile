# Makefile for MIP-MCP project

.PHONY: help install test test-unit test-integration test-basic coverage lint format clean dev-install

# Default target
help:
	@echo "Available commands:"
	@echo "  install       Install dependencies"
	@echo "  dev-install   Install development dependencies"
	@echo "  test          Run all tests"
	@echo "  test-basic    Run basic tests only"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  coverage      Run tests with coverage report"
	@echo "  lint          Run linting with ruff"
	@echo "  format        Format code with ruff"
	@echo "  clean         Clean cache and build artifacts"

# Installation
install:
	uv sync

dev-install:
	uv sync --dev

# Testing
test:
	uv run pytest tests/unit/ -v

test-all:
	uv run pytest -v

test-basic:
	uv run pytest tests/unit/test_basic.py tests/unit/test_models.py -v

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

# Coverage
coverage:
	uv run pytest tests/unit/test_basic.py tests/unit/test_models.py --cov=src/mip_mcp --cov-report=html --cov-report=term-missing

coverage-all:
	uv run pytest --cov=src/mip_mcp --cov-report=html --cov-report=term-missing

# Code quality
lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

lint-check:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

# Cleanup
clean:
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/

# Development
run-server:
	uv run mip-mcp

check:
	@echo "Running development checks..."
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test-basic