# colors
GREEN := \e[0;32m
RESET := \e[0m

FLAKE8_SUCCESS := printf '%b\n' "$(GREEN)Success: flake8$(RESET)"

SRC_DIRECTORIES := .
DIRS := . src $(addprefix src/,$(SRC_DIRECTORIES))
MAIN := src.main
ARGS ?=
VENV := .venv
VENV_STAMP := $(VENV)/stamp

UV_LOCK := uv.lock
PYPROJECT_TOML := pyproject.toml

PYCACHES = $(addsuffix /__pycache__,$(DIRS))
MYPYCACHES = $(addsuffix /.mypy_cache,$(DIRS))
EXCLUDE = --exclude $(VENV)

UV := uv
FLAKE8 := $(UV) run python -m flake8 $(EXCLUDE)
MYPY := $(UV) run python -m mypy $(EXCLUDE)

MYPY_FLAGS := \
		--check-untyped-defs \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--warn-return-any \
		--disallow-untyped-defs

install: $(UV_LOCK)
	$(UV) sync --no-install-project

install-dev: $(UV_LOCK)
	$(UV) sync --group dev --no-install-project

run: install
	$(UV) run -m student

clean:
	rm -rf $(PYCACHES) $(MYPYCACHES)
	rm -rf $(VENV)

debug: install-dev
	$(UV) run python -m pdb -m student

lint: install-dev
	$(FLAKE8) student && $(FLAKE8_SUCCESS)
	$(MYPY) student $(MYPY_FLAGS)

$(UV_LOCK): $(PYPROJECT_TOML) | $(PYTHON)
	$(UV) lock

.PHONY: install install-dev run debug lint clean
