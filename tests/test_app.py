"""Comprehensive tests for app module."""

import numpy as np
from unittest.mock import MagicMock, patch

from odoo_rag.app import calculate_context_recall


def test_calculate_context_recall_with_valid_inputs():
    """Test context recall calculation with valid inputs."""
    embedder = MagicMock()
    embedder.encode.return_value = np.array([[0.5] * 384, [0.6] * 384])

    score = calculate_context_recall(
        answer="This is a test answer",
        context="This is test context",
        embedder=embedder,
    )

    assert 0.0 <= score <= 1.0
    assert isinstance(score, float)


def test_calculate_context_recall_empty_answer():
    """Test context recall returns 0 for empty answer."""
    embedder = MagicMock()

    score = calculate_context_recall(
        answer="", context="Some context", embedder=embedder
    )

    assert score == 0.0


def test_calculate_context_recall_empty_context():
    """Test context recall returns 0 for empty context."""
    embedder = MagicMock()

    score = calculate_context_recall(
        answer="Some answer", context="", embedder=embedder
    )

    assert score == 0.0


def test_calculate_context_recall_both_empty():
    """Test context recall returns 0 when both inputs are empty."""
    embedder = MagicMock()

    score = calculate_context_recall(answer="", context="", embedder=embedder)

    assert score == 0.0


def test_calculate_context_recall_identical_texts():
    """Test context recall with identical answer and context."""
    embedder = MagicMock()
    # Return identical embeddings for identical texts
    embedder.encode.return_value = np.array([[1.0] * 384, [1.0] * 384])

    score = calculate_context_recall(
        answer="Identical text", context="Identical text", embedder=embedder
    )

    assert score > 0.9  # Should be very high similarity


def test_calculate_context_recall_calls_embedder():
    """Test that calculate_context_recall calls embedder correctly."""
    embedder = MagicMock()
    embedder.encode.return_value = np.array([[0.5] * 384, [0.6] * 384])

    calculate_context_recall(
        answer="Test answer", context="Test context", embedder=embedder
    )

    # Verify embedder was called
    assert embedder.encode.called
    # Should be called with both answer and context
    call_args = embedder.encode.call_args[0][0]
    assert len(call_args) == 2


def test_create_app_module_imports():
    """Test that create_app can be imported."""
    from odoo_rag.app import create_app

    assert create_app is not None
    assert callable(create_app)


def test_sentence_transformer_import():
    """Test that SentenceTransformer can be imported."""
    from odoo_rag.app import SentenceTransformer

    assert SentenceTransformer is not None


def test_calculate_context_recall_with_different_lengths():
    """Test context recall with different text lengths."""
    embedder = MagicMock()
    embedder.encode.return_value = np.array([[0.7] * 384, [0.8] * 384])

    short_answer = "Short"
    long_context = "This is a much longer context with many more words."

    score = calculate_context_recall(
        answer=short_answer, context=long_context, embedder=embedder
    )

    assert 0.0 <= score <= 1.0


def test_calculate_context_recall_whitespace_handling():
    """Test context recall handles whitespace correctly."""
    embedder = MagicMock()
    embedder.encode.return_value = np.array([[0.5] * 384, [0.6] * 384])

    answer_with_spaces = "  Answer with spaces  "
    context_with_newlines = "Context\nwith\nnewlines"

    score = calculate_context_recall(
        answer=answer_with_spaces, context=context_with_newlines, embedder=embedder
    )

    assert 0.0 <= score <= 1.0


def test_download_data_from_s3_import():
    """Test that download_data_from_s3 can be imported."""
    from odoo_rag.app import download_data_from_s3

    assert download_data_from_s3 is not None
    assert callable(download_data_from_s3)


@patch("odoo_rag.app.os.environ.get")
def test_download_data_from_s3_no_bucket(mock_env_get):
    """Test download_data_from_s3 when S3_BUCKET is not set."""
    from odoo_rag.app import download_data_from_s3

    mock_env_get.return_value = None

    # Should not raise exception
    download_data_from_s3()


def test_main_function_import():
    """Test that main function can be imported."""
    from odoo_rag.app import main

    assert main is not None
    assert callable(main)
