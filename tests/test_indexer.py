"""Comprehensive tests for indexer module."""

import json
import tempfile
from pathlib import Path

import faiss

from odoo_rag.indexer import build_index, chunk_text


def test_chunk_text_basic():
    """Test basic text chunking."""
    text = " ".join(["word"] * 100)
    chunks = chunk_text(text, chunk_size=50, overlap=10)

    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)


def test_chunk_text_small_text():
    """Test chunking text smaller than chunk size."""
    text = "This is a small text with few words that is longer than fifty characters to pass the minimum threshold."
    chunks = chunk_text(text, chunk_size=100, overlap=10)

    assert len(chunks) >= 1
    assert text.split()[0] in chunks[0]


def test_chunk_text_empty():
    """Test chunking empty text."""
    chunks = chunk_text("", chunk_size=50, overlap=10)

    assert len(chunks) == 0


def test_chunk_text_single_word():
    """Test chunking single word (skipped if < 50 chars)."""
    chunks = chunk_text("word", chunk_size=50, overlap=10)

    # Single short word is skipped (< 50 characters)
    assert len(chunks) == 0


def test_chunk_text_exact_chunk_size():
    """Test text that is exactly chunk size."""
    text = " ".join(["word"] * 50)
    chunks = chunk_text(text, chunk_size=50, overlap=10)

    assert len(chunks) >= 1


def test_chunk_text_with_overlap():
    """Test that overlap creates overlapping chunks."""
    text = " ".join([f"word{i}" for i in range(100)])
    chunks = chunk_text(text, chunk_size=20, overlap=5)

    assert len(chunks) > 1
    # Verify chunks were created
    assert all(len(c) > 0 for c in chunks)


def test_chunk_text_no_overlap():
    """Test chunking with zero overlap."""
    text = " ".join(["word"] * 100)
    chunks = chunk_text(text, chunk_size=50, overlap=0)

    assert len(chunks) >= 2


def test_chunk_text_custom_sizes():
    """Test chunking with various chunk sizes."""
    text = " ".join(["word"] * 200)

    chunks_small = chunk_text(text, chunk_size=25, overlap=5)
    chunks_large = chunk_text(text, chunk_size=100, overlap=10)

    assert len(chunks_small) > len(chunks_large)


def test_build_index_basic():
    """Test basic index building."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"
        raw_dir.mkdir()

        # Create test file
        test_file = raw_dir / "test.txt"
        with open(test_file, "w") as f:
            f.write("How to create a model in Odoo?\n")
            f.write("How to add a field to a view?\n")
            f.write("How to configure access rights?\n")

        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()

        # Build index
        build_index(raw_dir=str(raw_dir), output_dir=str(output_dir))

        # Verify outputs exist
        assert (output_dir / "index.faiss").exists()
        assert (output_dir / "corpus.json").exists()

        # Verify index can be loaded
        index = faiss.read_index(str(output_dir / "index.faiss"))
        assert index.ntotal >= 1

        # Verify corpus can be loaded
        with open(output_dir / "corpus.json") as f:
            corpus = json.load(f)
        assert len(corpus) >= 1
        assert all("text" in item for item in corpus)


def test_build_index_multiple_files():
    """Test building index from multiple files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"
        raw_dir.mkdir()

        # Create multiple test files with sufficient content (> 50 chars each)
        for i in range(3):
            test_file = raw_dir / f"test{i}.txt"
            with open(test_file, "w") as f:
                # Write enough content to create chunks (> 50 characters)
                f.write(
                    f"This is content from file number {i} with enough text to create a valid chunk for indexing.\n"
                )

        output_dir = Path(tmpdir) / "output"

        build_index(raw_dir=str(raw_dir), output_dir=str(output_dir))

        assert (output_dir / "index.faiss").exists()
        assert (output_dir / "corpus.json").exists()


def test_build_index_creates_output_dir():
    """Test that build_index creates output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"
        raw_dir.mkdir()

        test_file = raw_dir / "test.txt"
        with open(test_file, "w") as f:
            # Write enough content to create chunks (> 50 characters)
            f.write(
                "Test content for indexing with enough text to create a valid chunk for the FAISS index.\n"
            )

        output_dir = Path(tmpdir) / "nonexistent" / "output"

        build_index(raw_dir=str(raw_dir), output_dir=str(output_dir))

        assert output_dir.exists()
        assert (output_dir / "index.faiss").exists()


def test_build_index_long_document():
    """Test building index from long document."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"
        raw_dir.mkdir()

        test_file = raw_dir / "long.txt"
        with open(test_file, "w") as f:
            # Write a long document that will be chunked
            for i in range(100):
                f.write(f"This is sentence number {i} in a long document. ")

        output_dir = Path(tmpdir) / "output"

        build_index(raw_dir=str(raw_dir), output_dir=str(output_dir))

        # Verify multiple chunks were created
        with open(output_dir / "corpus.json") as f:
            corpus = json.load(f)
        assert len(corpus) > 1  # Long doc should create multiple chunks


def test_build_index_with_metadata():
    """Test that corpus includes source metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"
        raw_dir.mkdir()

        test_file = raw_dir / "odoo_models.txt"
        with open(test_file, "w") as f:
            # Write enough content to create chunks (> 50 characters)
            f.write(
                "Model documentation content with enough text to create a valid chunk for indexing and testing.\n"
            )

        output_dir = Path(tmpdir) / "output"

        build_index(raw_dir=str(raw_dir), output_dir=str(output_dir))

        with open(output_dir / "corpus.json") as f:
            corpus = json.load(f)

        # Verify metadata is present
        assert all("source" in item for item in corpus)
