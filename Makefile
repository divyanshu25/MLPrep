.PHONY: install sync run clean test lint format

# Install dependencies and create virtual environment
install:
	uv sync

# Sync dependencies (install/update)
sync:
	uv sync

# Run the main script
run:
	uv run python src/multiply2.py

# Clean virtual environment and cache
clean:
	uv cache clean
	rm -rf .venv

# Run tests (if you add them later)
test:
	uv run python -m pytest

# Lint code (if you add linting tools later)
lint:
	uv run ruff check src/

# Format code (if you add formatting tools later)
format:
	uv run ruff format src/

# Show project info
info:
	uv tree
