# Modern Python Tooling for URAF 🚀

URAF uses **cutting-edge Python tooling** for maximum developer productivity and code quality.

---

## 🔧 Tech Stack

| Tool | Purpose | Why? |
|------|---------|------|
| **[uv](https://github.com/astral-sh/uv)** | Package manager | 10-100x faster than pip/poetry, written in Rust |
| **[ruff](https://github.com/astral-sh/ruff)** | Linter & Formatter | Replaces black, flake8, isort, pyupgrade - 10-100x faster |
| **[pytest](https://pytest.org)** | Testing framework | Industry standard, great async support |
| **[mypy](https://mypy-lang.org)** | Type checker | Catch bugs before runtime |
| **Make** | Task runner | Simple, universal, no extra dependencies |

---

## 📦 Installation & Setup

### Quick Start (3 commands!)

```bash
# 1. Install uv (if not already installed)
pip install uv

# 2. Install all dependencies
uv sync --extra dev

# 3. You're ready!
make test
```

### Detailed Setup

```bash
# Clone repository
git clone https://github.com/your-repo/URAF.git
cd URAF

# Install uv globally
pip install --user uv

# Sync dependencies (creates .venv automatically)
uv sync --extra dev

# Verify installation
uv run python -c "import uraf; print('✅ URAF ready!')"
```

---

## ⚡ Developer Workflow

### Essential Commands

```bash
# Development
make dev          # Install with dev dependencies
make test         # Run all tests
make lint         # Check code quality
make format       # Format code
make fix          # Auto-fix all issues
make check        # Run lint + typecheck + tests

# Running URAF
make demo         # Run feature demonstration
make run          # Run evaluation with default config
make run-advanced # Run with all features enabled

# Maintenance
make clean        # Remove generated files
make update       # Update dependencies
make help         # Show all commands
```

### Manual Commands (without Make)

```bash
# Install dependencies
uv sync                           # Production deps
uv sync --extra dev              # + Development deps
uv sync --upgrade                # Update all packages

# Run code
uv run python -m uraf.cli --run
uv run python examples/demo_new_features.py
uv run pytest tests/

# Code quality
uv run ruff check uraf/          # Lint
uv run ruff format uraf/         # Format
uv run ruff check --fix uraf/    # Auto-fix
uv run mypy uraf/                # Type check

# Testing
uv run pytest tests/ -v                      # Verbose
uv run pytest tests/test_file.py            # Single file
uv run pytest -k "test_name"                 # Match pattern
uv run pytest --cov=uraf --cov-report=html  # Coverage
```

---

## 📁 Project Structure

```
URAF/
├── .python-version          # Python version (3.11)
├── pyproject.toml           # Modern project config (PEP 621)
├── uv.lock                  # Locked dependencies
├── Makefile                 # Development tasks
├── .venv/                   # Virtual environment (auto-created)
├── uraf/                    # Main package
│   ├── __init__.py
│   ├── process_reward_model.py
│   ├── tool_system.py
│   ├── memory_system.py
│   └── ... (10 new modules!)
├── tests/                   # Test suite
│   └── test_new_features.py
└── examples/                # Example scripts
    ├── demo_new_features.py
    └── advanced-config.yaml
```

---

## 🎯 Code Quality Standards

### Configured Checks

**Ruff** (replaces multiple tools):
- ✅ pycodestyle (E, W) - PEP 8 enforcement
- ✅ pyflakes (F) - Logical errors
- ✅ isort (I) - Import sorting
- ✅ flake8-bugbear (B) - Common bugs
- ✅ flake8-comprehensions (C4) - Better comprehensions
- ✅ pyupgrade (UP) - Modern Python syntax
- ✅ flake8-unused-arguments (ARG) - Unused code
- ✅ flake8-simplify (SIM) - Simplification suggestions

### Auto-fixing

```bash
# Fix everything automatically
make fix

# Or manually
uv run ruff check --fix uraf/
uv run ruff format uraf/
```

---

## 🧪 Testing

### Running Tests

```bash
# All tests
make test

# With coverage
make test-cov
open htmlcov/index.html  # View coverage report

# Specific test
uv run pytest tests/test_new_features.py::TestPRM -v

# Watch mode (requires pytest-watch)
uv run pytest-watch
```

### Test Configuration

Tests are configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short --strict-markers"
asyncio_mode = "auto"  # Automatic async test support
```

---

## 📊 Performance Comparison

### uv vs Poetry

| Task | Poetry | uv | Speedup |
|------|--------|-----|---------|
| **Install 211 pkgs** | ❌ Disk full (9.2GB cache) | ✅ ~5 min | ∞ |
| **Resolve deps** | ~3 min | ~3 sec | ~60x |
| **Add package** | ~30 sec | ~1 sec | ~30x |

### ruff vs black+flake8+isort

| Task | Old Tools | ruff | Speedup |
|------|-----------|------|---------|
| **Lint project** | ~5 sec | ~0.05 sec | ~100x |
| **Format project** | ~3 sec | ~0.03 sec | ~100x |
| **Auto-fix** | Manual | Automatic | ∞ |

---

## 🔄 Migration from Poetry

If you have an old Poetry setup:

```bash
# 1. Remove poetry artifacts
rm poetry.lock
rm -rf .venv

# 2. Install uv
pip install uv

# 3. Sync with uv
uv sync --extra dev

# 4. Done! pyproject.toml is already compatible
```

---

## 💡 Tips & Tricks

### Fast Package Management

```bash
# Add new dependency
uv add package-name

# Add dev dependency
uv add --dev package-name

# Remove dependency
uv remove package-name

# Update specific package
uv sync --upgrade-package package-name

# Show installed packages
uv pip list

# Show outdated packages
uv pip list --outdated
```

### IDE Integration

**VS Code** (`.vscode/settings.json`):
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll.ruff": true,
      "source.organizeImports.ruff": true
    }
  },
  "ruff.enable": true,
  "ruff.lint.enable": true
}
```

**PyCharm/IntelliJ**:
1. Set interpreter: `.venv/bin/python`
2. Enable Ruff plugin from marketplace
3. Configure external tool for `make` commands

### Pre-commit Hooks

```bash
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

# Install hooks
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files
```

---

## 🐛 Troubleshooting

### uv not found

```bash
# Install uv
pip install --user uv

# Add to PATH (if needed)
export PATH="$HOME/.local/bin:$PATH"
```

### Permission errors

```bash
# Don't use sudo! Use user installs
pip install --user uv
```

### Slow downloads

```bash
# Use faster mirror (China users)
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# Or set in config
uv config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Disk space issues

```bash
# Clean uv cache
uv cache clean

# Clean all caches
make clean
rm -rf ~/.cache/uv
```

---

## 📚 Resources

- **uv Documentation**: https://github.com/astral-sh/uv
- **ruff Documentation**: https://docs.astral.sh/ruff
- **PEP 621** (pyproject.toml): https://peps.python.org/pep-0621/
- **Modern Python Packaging**: https://packaging.python.org/

---

## 🎓 Learning More

### Example Workflow

```bash
# Day 1: Setup
git clone <repo>
cd URAF
make dev
make test

# Day 2: Development
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Write code
vim uraf/my_module.py

# 3. Check quality
make check  # lint + typecheck + test

# 4. Fix issues
make fix

# 5. Commit
git add .
git commit -m "Add my feature"

# 6. Push
git push origin feature/my-feature
```

### Continuous Integration

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: uv sync --extra dev

      - name: Run checks
        run: make check
```

---

**Built with ❤️ using modern Python tooling**
