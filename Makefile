PYTHON ?= python

.PHONY: help clean-data seed-database seed-vector seed-all

help:
	@echo "Available commands:"
	@echo "  make clean-data     Clean raw_dataset.json into dataset.json"
	@echo "  make seed-database  Seed dataset.json into SQLite"
	@echo "  make seed-vector    Seed dataset.json into Qdrant"
	@echo "  make seed-all       Run clean data, SQLite seed, and vector seed"

clean-data:
	$(PYTHON) -m scripts.clean_data

seed-database:
	$(PYTHON) -m scripts.seed_database

seed-vector:
	$(PYTHON) -m scripts.seed_vector

seed-all: clean-data seed-database seed-vector
