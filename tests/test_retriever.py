"""Tests for retriever module."""

import json
import tempfile
from pathlib import Path

import faiss
import numpy as np
import pytest

from odoo_rag.retriever import Retriever


def test_retriever_module_import():
    """Test that retriever module can be imported."""
    import odoo_rag.retriever

    assert odoo_rag.retriever is not None
    assert hasattr(odoo_rag.retriever, "Retriever")


@pytest.fixture
def temp_index_dir():
    """Create temporary directory with test FAISS index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create minimal FAISS index
        dimension = 384  # all-MiniLM-L6-v2 dimension
        index = faiss.IndexFlatIP(dimension)

        # Add 3 dummy vectors
        vectors = np.random.rand(3, dimension).astype("float32")
        faiss.normalize_L2(vectors)
        index.add(vectors)

        # Save index
        index_path = tmpdir_path / "index.faiss"
        faiss.write_index(index, str(index_path))

        # Create corpus
        corpus = [
            {"text": "How to create a model in Odoo?", "source": "odoo_orm"},
            {"text": "How to add a field to a view?", "source": "odoo_views"},
            {"text": "How to configure access rights?", "source": "odoo_security"},
        ]
        corpus_path = tmpdir_path / "corpus.json"
        with open(corpus_path, "w") as f:
            json.dump(corpus, f)

        yield tmpdir_path


def test_retriever_init(temp_index_dir):
    """Test retriever initialization."""
    retriever = Retriever(index_dir=str(temp_index_dir))
    assert retriever.index is not None
    assert retriever.corpus is not None
    assert len(retriever.corpus) == 3
    assert retriever.dimension == 384
    assert retriever.k == 5


def test_retriever_query(temp_index_dir):
    """Test retriever query functionality."""
    retriever = Retriever(index_dir=str(temp_index_dir))
    results = retriever.query("create model", k=2)

    assert len(results) <= 2
    assert all("document" in r for r in results)
    assert all("score" in r for r in results)
    assert all("source" in r for r in results)


def test_retriever_query_default_k(temp_index_dir):
    """Test retriever uses default k when not specified."""
    retriever = Retriever(index_dir=str(temp_index_dir))
    # Default k is 5, but we only have 3 documents
    results = retriever.query("test query")
    assert len(results) == 3


def test_retriever_query_custom_k(temp_index_dir):
    """Test retriever respects custom k parameter."""
    retriever = Retriever(index_dir=str(temp_index_dir))

    results_1 = retriever.query("test query", k=1)
    assert len(results_1) == 1

    results_2 = retriever.query("test query", k=2)
    assert len(results_2) == 2


def test_retriever_missing_index():
    """Test retriever handles missing index gracefully."""
    with pytest.raises(Exception):
        Retriever(index_dir="/nonexistent/path")


def test_retriever_env_var(temp_index_dir, monkeypatch):
    """Test retriever uses FAISS_INDEX_PATH environment variable."""
    monkeypatch.setenv("FAISS_INDEX_PATH", str(temp_index_dir))
    retriever = Retriever()  # No index_dir specified
    assert retriever.index is not None
    assert len(retriever.corpus) == 3


def test_retriever_build_prompt(temp_index_dir):
    """Test build_prompt method."""
    retriever = Retriever(index_dir=str(temp_index_dir))

    # Get some context chunks first
    results = retriever.query("test query", k=2)

    # Extract document strings from results
    context_chunks = [r["document"] for r in results]

    # Build prompt with context chunks
    prompt = retriever.build_prompt("How to create a model?", context_chunks)

    assert "How to create a model?" in prompt
    assert len(prompt) > 0
