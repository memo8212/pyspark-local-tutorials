.PHONY: help setup data all test clean

help:
	@echo "make setup  - create venv + install requirements"
	@echo "make data   - generate sample data into ./data"
	@echo "make all    - generate data and run every tutorial"
	@echo "make test   - run pytest"
	@echo "make clean  - remove local run artifacts"

setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

data:
	python common/data.py

all: data
	python run_all.py

test:
	pytest -q

clean:
	rm -rf spark-warehouse metastore_db derby.log output .pytest_cache
	find . -name '__pycache__' -type d -exec rm -rf {} +
