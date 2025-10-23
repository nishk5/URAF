# URAF Development Commands with just
# https://github.com/casey/just

# Default recipe (show help)
default:
    @just --list

# Install production dependencies
install:
    uv sync

# Install with dev dependencies
dev:
    uv sync --extra dev

# Run all tests
test:
    uv run pytest tests/ -v

# Run tests with coverage
test-cov:
    uv run pytest tests/ --cov=uraf --cov-report=html --cov-report=term

# Run specific test file
test-file FILE:
    uv run pytest {{FILE}} -v

# Run tests matching pattern
test-match PATTERN:
    uv run pytest -k {{PATTERN}} -v

# Run ruff linter
lint:
    uv run ruff check uraf/ tests/ examples/

# Run ruff formatter
format:
    uv run ruff format uraf/ tests/ examples/

# Run type checker
typecheck:
    uv run mypy uraf/

# Run all checks (lint + typecheck + test)
check: lint typecheck test

# Fix common issues automatically
fix:
    uv run ruff check --fix uraf/ tests/ examples/
    uv run ruff format uraf/ tests/ examples/

# Clean generated files
clean:
    rm -rf .venv
    rm -rf .pytest_cache
    rm -rf .ruff_cache
    rm -rf htmlcov
    rm -rf .coverage
    rm -rf dist
    rm -rf build
    rm -rf *.egg-info
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete

# Run the demo script
demo:
    uv run python examples/demo_new_features.py

# Run evaluation with default config
run:
    uv run python -m uraf.cli --run --config config.yaml

# Run evaluation with advanced config
run-advanced:
    uv run python -m uraf.cli --run --config examples/advanced-config.yaml

# Show evaluation history
history:
    uv run python -m uraf.cli --history

# Compare models
compare:
    uv run python -m uraf.cli --compare

# Export results
export:
    uv run python -m uraf.cli --export

# Start interactive Python shell with uraf loaded
shell:
    uv run ipython -i -c "import sys; sys.path.insert(0, '.'); from uraf import *; print('URAF loaded!')"

# Build package
build:
    uv build

# Show disk usage
disk:
    @echo "Virtual environment size:"
    @du -sh .venv 2>/dev/null || echo "No .venv found"
    @echo ""
    @echo "Cache size:"
    @du -sh ~/.cache/uv 2>/dev/null || echo "No uv cache"

# Update dependencies
update:
    uv sync --upgrade

# Install pre-commit hooks
hooks:
    uv run pre-commit install

# Run PRM demo
demo-prm:
    uv run python -c "from uraf.process_reward_model import ProcessRewardModel; prm = ProcessRewardModel(); print('✅ PRM ready!')"

# Run tool system demo
demo-tools:
    uv run python -c "from uraf.tool_system import ToolRegistry; registry = ToolRegistry(); print('✅ Tools:', [t['name'] for t in registry.list_tools()])"

# Run memory system demo
demo-memory:
    uv run python -c "import asyncio; from uraf.memory_system import AgentMemory; asyncio.run(AgentMemory().get_memory_statistics()).then(print)"

# Quick CI check (for pre-commit)
ci: fix lint test

# Install just command runner (for first-time users)
install-just:
    @echo "Installing just command runner..."
    @command -v cargo >/dev/null && cargo install just || pip install just-install

# Show environment info
env:
    @echo "Python version:"
    @python --version
    @echo ""
    @echo "uv version:"
    @uv --version
    @echo ""
    @echo "ruff version:"
    @uv run ruff --version
    @echo ""
    @echo "Virtual environment:"
    @echo "  Path: .venv"
    @echo "  Exists: $(if [ -d .venv ]; then echo 'Yes'; else echo 'No'; fi)"
