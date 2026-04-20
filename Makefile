.PHONY: setup run pipeline-data index clean

VENV := .venv
PYTHON := $(VENV)/bin/python
UV := uv

# One-time setup
setup: $(VENV)
	$(UV) pip install -e .
	$(MAKE) index

$(VENV):
	$(UV) venv $(VENV)

# Run chat UI  
run:
	$(UV) run python -m odoo_rag.app

# Data pipeline: scrape docs → build index
# Note: QA generation disabled (RAG approach, not fine-tuning)
pipeline-data:
	$(UV) run python -m odoo_rag.ingest
	$(MAKE) index
	# $(UV) run python -m odoo_rag.generate_qa  # Enable only for fine-tuning custom models

# Build FAISS index from data
index:
	$(UV) run python -m odoo_rag.indexer

clean:
	rm -rf $(VENV) data/faiss_index *.csv
