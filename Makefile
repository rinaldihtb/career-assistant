PYTHON ?= python

.PHONY: help clean-data seed-database seed-vector seed-all run test sample-sql

help:
	@echo "Available commands:"
	@echo "  make clean-data     Clean raw_dataset.json into dataset.json"
	@echo "  make seed-database  Seed dataset.json into SQLite"
	@echo "  make seed-vector    Seed dataset.json into Qdrant"
	@echo "  make seed-all       Run clean data, SQLite seed, and vector seed"
	@echo "  make run             Run the Streamlit app"
	@echo "  make test            Run unit tests (no API key needed)"
	@echo "  make sample-sql      Check SQL tools on the real database (no LLM)"
	@echo "  make sample-sql-llm  Smoke test the SQL agent with the real LLM"

clean-data:
	$(PYTHON) -m scripts.clean_data

seed-database:
	$(PYTHON) -m scripts.seed_database

seed-vector:
	$(PYTHON) -m scripts.seed_vector

seed-all: clean-data seed-database seed-vector

run:
	PYTHONPATH=. $(PYTHON) -m streamlit run src/app.py

test:
	$(PYTHON) -m pytest

sample-sql:
	PYTHONPATH=.:src $(PYTHON) -m agents.sql_sample

sample-sql-llm:
	PYTHONPATH=.:src $(PYTHON) -m agents.sql_sample --llm
