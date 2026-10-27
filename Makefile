PYTHON ?= python

.PHONY: help clean-data seed-database seed-vector seed-all run api test lint
.PHONY: eval eval-cv eval-cv-chat sample-cv sample-plan sample-letter

help:
	@echo "Available commands:"
	@echo "  make clean-data         Clean raw_dataset.json into dataset.json"
	@echo "  make seed-database      Seed dataset.json into SQLite"
	@echo "  make seed-vector        Seed dataset.json into Qdrant"
	@echo "  make seed-all           Run clean data, SQLite seed, and vector seed"
	@echo "  make run                Run the Streamlit app"
	@echo "  make api                Run the FastAPI backend used by the Asisten page"
	@echo "  make test               Run unit tests (no API key needed)"
	@echo "  make lint               Ruff check for src and tests"
	@echo "  make eval               Assistant eval set without CV (real LLM, ARGS=\"--label x\")"
	@echo "  make eval-cv            Cocokkan CV eval set on the fictional CVs (real LLM)"
	@echo "  make eval-cv-chat       Assistant eval set with an active CV (real LLM)"
	@echo "  make sample-cv          Profile extraction on your own CV (ARGS=path/to/cv.pdf)"
	@echo "  make sample-plan        Skill Gap learning plan metrics (real LLM)"
	@echo "  make sample-letter      Cover letter metrics (real LLM)"

clean-data:
	$(PYTHON) -m scripts.clean_data

seed-database:
	$(PYTHON) -m scripts.seed_database

seed-vector:
	$(PYTHON) -m scripts.seed_vector

seed-all: clean-data seed-database seed-vector

run:
	$(PYTHON) -m streamlit run src/app.py

api:
	$(PYTHON) -m uvicorn src.api:app --reload --port 8000

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

eval:
	$(PYTHON) -m src.evaluation.run_eval $(ARGS)

eval-cv:
	$(PYTHON) -m src.evaluation.run_cv_eval $(ARGS)

eval-cv-chat:
	$(PYTHON) -m src.evaluation.run_cv_chat_eval $(ARGS)

sample-cv:
	$(PYTHON) -m src.cv.profile_sample $(ARGS)

sample-plan:
	$(PYTHON) -m src.cv.plan_sample $(ARGS)

sample-letter:
	$(PYTHON) -m src.cv.letter_sample $(ARGS)
