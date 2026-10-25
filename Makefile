PYTHON ?= python

.PHONY: help clean-data seed-database seed-vector seed-all run test lint sample-sql sample-sql-llm sample-rag sample-rag-llm sample-supervisor supervisor-graph

help:
	@echo "Available commands:"
	@echo "  make clean-data         Clean raw_dataset.json into dataset.json"
	@echo "  make seed-database      Seed dataset.json into SQLite"
	@echo "  make seed-vector        Seed dataset.json into Qdrant"
	@echo "  make seed-all           Run clean data, SQLite seed, and vector seed"
	@echo "  make run                Run the Streamlit app"
	@echo "  make test               Run unit tests (no API key needed)"
	@echo "  make lint               Ruff check for agent, repository, model, and test files"
	@echo "  make sample-sql         Check SQL tools on the real database (no LLM)"
	@echo "  make sample-sql-llm     Smoke test the SQL agent with the real LLM"
	@echo "  make sample-rag         Check RAG retrieval on real Qdrant (embedding only)"
	@echo "  make sample-rag-llm     Smoke test the RAG agent with the real LLM"
	@echo "  make sample-supervisor  Smoke test supervisor routing with the real LLM"
	@echo "  make supervisor-graph   Print the supervisor LangGraph as Mermaid (no API key)"

clean-data:
	$(PYTHON) -m scripts.clean_data

seed-database:
	$(PYTHON) -m scripts.seed_database

seed-vector:
	$(PYTHON) -m scripts.seed_vector

seed-all: clean-data seed-database seed-vector

run:
	$(PYTHON) -m streamlit run src/app.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src/agents src/repositories src/models src/cv src/evaluation tests

sample-sql:
	$(PYTHON) -m src.agents.sql_sample

sample-sql-llm:
	$(PYTHON) -m src.agents.sql_sample --llm

sample-rag:
	$(PYTHON) -m src.agents.rag_sample

sample-rag-llm:
	$(PYTHON) -m src.agents.rag_sample --llm

sample-supervisor:
	$(PYTHON) -m src.agents.supervisor_sample

supervisor-graph:
	$(PYTHON) -m src.agents.supervisor_sample --graph

.PHONY: eval

eval:
	$(PYTHON) -m src.evaluation.run_eval $(ARGS)

.PHONY: sample-cv

sample-cv:
	$(PYTHON) -m src.cv.profile_sample $(ARGS)

.PHONY: sample-match

sample-match:
	$(PYTHON) -m src.cv.match_sample $(ARGS)

.PHONY: eval-cv

eval-cv:
	$(PYTHON) -m src.evaluation.run_cv_eval $(ARGS)
