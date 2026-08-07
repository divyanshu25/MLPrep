.PHONY: install sync run clean test lint format study verify-notebooks

# Launch the LeetCode-style study app (http://localhost:8888)
study:
	uv run python app/server.py

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

# Execute every ML curriculum notebook headlessly (fails on any cell error)
verify-notebooks:
	@set -e; for nb in ml/*/*.ipynb; do \
		echo "== $$nb"; \
		(cd $$(dirname $$nb) && uv run jupyter execute --timeout 300 $$(basename $$nb)); \
	done

# Show project info
info:
	uv tree
