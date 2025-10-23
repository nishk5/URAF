.PHONY: install dev test lint format check clean run demo help

# Default target
.DEFAULT_GOAL := help

## Install production dependencies
install:
	uv sync

## Install with dev dependencies
dev:
	uv sync --extra dev

## Run all tests
test:
	uv run pytest tests/ -v

## Run tests with coverage
test-cov:
	uv run pytest tests/ --cov=uraf --cov-report=html --cov-report=term

## Run specific test file
test-file:
	uv run pytest $(FILE) -v

## Run ruff linter
lint:
	uv run ruff check uraf/ tests/ examples/

## Run ruff formatter
format:
	uv run ruff format uraf/ tests/ examples/

## Run type checker
typecheck:
	uv run mypy uraf/

## Run all checks (lint + typecheck + test)
check: lint typecheck test

## Fix common issues automatically
fix:
	uv run ruff check --fix uraf/ tests/ examples/
	uv run ruff format uraf/ tests/ examples/

## Clean generated files
clean:
	rm -rf .venv
	rm -rf **/__pycache__
	rm -rf **/*.pyc
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +

## Run the demo script
demo:
	uv run python examples/demo_new_features.py

## Run evaluation with default config
run:
	uv run python -m uraf.cli --run --config config.yaml

## Run evaluation with advanced config
run-advanced:
	uv run python -m uraf.cli --run --config examples/advanced-config.yaml

## Show evaluation history
history:
	uv run python -m uraf.cli --history

## Compare models
compare:
	uv run python -m uraf.cli --compare

## Export results
export:
	uv run python -m uraf.cli --export

## Start interactive Python shell with uraf loaded
shell:
	uv run ipython -i -c "import sys; sys.path.insert(0, '.'); from uraf import *; print('URAF loaded!')"

## Build package
build:
	uv build

## Show disk usage
disk:
	@echo "Virtual environment size:"
	@du -sh .venv 2>/dev/null || echo "No .venv found"
	@echo "\nCache size:"
	@du -sh ~/.cache/uv 2>/dev/null || echo "No uv cache"

## Update dependencies
update:
	uv sync --upgrade

## Install pre-commit hooks
hooks:
	uv run pre-commit install

## Show this help message
help:
	@echo "URAF - Modern Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  install      Install production dependencies"
	@echo "  dev          Install with dev dependencies"
	@echo "  update       Update all dependencies"
	@echo ""
	@echo "Development:"
	@echo "  test         Run all tests"
	@echo "  test-cov     Run tests with coverage report"
	@echo "  lint         Run ruff linter"
	@echo "  format       Format code with ruff"
	@echo "  typecheck    Run mypy type checker"
	@echo "  check        Run all checks (lint+type+test)"
	@echo "  fix          Auto-fix linting issues and format"
	@echo ""
	@echo "Running:"
	@echo "  demo         Run feature demonstration"
	@echo "  run          Run evaluation with default config"
	@echo "  run-advanced Run with all advanced features"
	@echo "  history      Show evaluation history"
	@echo "  compare      Compare model performances"
	@echo "  shell        Start interactive Python shell"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean        Remove generated files"
	@echo "  build        Build distribution package"
	@echo "  disk         Show disk usage"
	@echo ""
	@echo "For more info: https://github.com/your-repo/URAF"
