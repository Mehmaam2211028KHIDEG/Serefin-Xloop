.PHONY: install format lint flake pylint black run clean

VENV_DIR := .venv
VENV_BIN := $(VENV_DIR)/bin
PYTHON_VERSION := python3.11
PYTHON := $(VENV_BIN)/python
PIP := $(VENV_BIN)/pip

$(VENV_DIR):
	$(PYTHON_VERSION) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install: $(VENV_DIR)

format: $(VENV_DIR)
	$(PYTHON) -m black .
	$(PYTHON) -m isort ./src --profile black

black: $(VENV_DIR)
	$(PYTHON) -m black --check .

flake: $(VENV_DIR)
	$(PYTHON) -m flake8 .

pylint: $(VENV_DIR)
	$(PYTHON) -m pylint ./src

lint: black flake pylint

run: $(VENV_DIR)
	$(PYTHON) ./src/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf $(VENV_DIR)