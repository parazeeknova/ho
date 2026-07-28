.PHONY: check check-types

check:
	uv run ruff format . && uv run ruff check . --fix

check-types:
	uv run mypy . --ignore-missing-imports --exclude 'refs/'
