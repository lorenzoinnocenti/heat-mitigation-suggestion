.ONESHELL:

.PHONY: help
help:				## This help screen
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

.PHONY: check-uv
check-uv:			## Check if uv is installed
	@if ! command -v uv > /dev/null; then echo "uv is not installed. Install it first (https://docs.astral.sh/uv/)"; exit 1; fi

.PHONY: init
init: check-uv			## Initialize the project template
	@uv run init.py

.PHONY: show
show: check-uv				## Show the current environment.
	@echo "Current environment:"
	@uv run python -V
	@uv run python -m site

.PHONY: install
install: check-uv		## Install the project in dev mode.
	@uv sync --all-extras

.PHONY: fmt
fmt: check-uv			## Format code using ruff.
	@uv run ruff format -v .

.PHONY: lint
lint: check-uv		## Run ruff and mypy (optional).
	@uv run ruff check .
	@uv run ruff format --check .
	@uv run mypy src/ || echo "mypy failed or not installed"

.PHONY: test
test: lint			## Run tests and generate coverage report.
	@uv run pytest --cov-report=xml -o console_output_style=progress

.PHONY: clean
clean:				## Clean unused files (VENV=true to also remove the virtualenv).
	@find ./ -name '*.pyc' -exec rm -f {} \;
	@find ./ -name '__pycache__' -exec rm -rf {} \;
	@find ./ -name 'Thumbs.db' -exec rm -f {} \;
	@find ./ -name '*~' -exec rm -f {} \;
	@rm -rf .cache
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf .ruff_cache
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -rf htmlcov
	@rm -rf .tox/
	@rm -rf docs/_build
	@if [ "$(VENV)" = "true" ]; then echo "Removing virtualenv..."; rm -rf $(PY_ENV); fi
