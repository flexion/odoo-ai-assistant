"""Comprehensive tests for llm module."""

from unittest.mock import MagicMock, patch

from odoo_rag.llm import ModelResponse, generate


def test_model_response_creation():
    """Test ModelResponse dataclass creation."""
    response = ModelResponse(
        text="Test answer",
        model="test-model-id",
        latency_sec=1.5,
        cost_usd=0.001,
        tokens_in=100,
        tokens_out=50,
    )

    assert response.text == "Test answer"
    assert response.model == "test-model-id"
    assert response.latency_sec == 1.5
    assert response.cost_usd == 0.001
    assert response.tokens_in == 100
    assert response.tokens_out == 50


def test_model_response_defaults():
    """Test ModelResponse with default token values."""
    response = ModelResponse(text="Test", model="model", latency_sec=1.0, cost_usd=0.0)

    assert response.tokens_in == 0
    assert response.tokens_out == 0


def test_generate_echo_mode(monkeypatch):
    """Test generate in echo mode (no AWS required)."""
    monkeypatch.setenv("ODOO_LLM_BACKEND", "echo")

    result = generate(prompt="Test prompt", model_id="test-model")

    assert isinstance(result, ModelResponse)
    assert "[echo]" in result.text
    assert "Test prompt" in result.text
    assert result.model == "echo"
    assert result.latency_sec == 0.0
    assert result.cost_usd == 0.0


def test_generate_echo_mode_truncates_long_prompt(monkeypatch):
    """Test that echo mode truncates long prompts."""
    monkeypatch.setenv("ODOO_LLM_BACKEND", "echo")

    long_prompt = "word " * 100  # 500 characters
    result = generate(prompt=long_prompt, model_id="test-model")

    assert len(result.text) < len(long_prompt)
    assert "..." in result.text


@patch("odoo_rag.llm.boto3.client")
@patch("odoo_rag.llm.time.time")
def test_generate_bedrock_success(mock_time, mock_boto_client):
    """Test successful Bedrock API call."""
    # Mock time to prevent timeout
    mock_time.return_value = 0.0

    mock_client = MagicMock()
    mock_boto_client.return_value = mock_client

    # Mock Bedrock response
    mock_client.converse.return_value = {
        "output": {"message": {"content": [{"text": "Generated answer"}]}},
        "usage": {"inputTokens": 100, "outputTokens": 50},
    }

    # Use real model ID from pricing dict
    response = generate(
        prompt="Test prompt",
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        timeout=5,
    )

    assert response.text == "Generated answer"
    assert response.model == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert response.tokens_in == 100
    assert response.tokens_out == 50
    assert response.cost_usd >= 0


@patch("odoo_rag.llm.boto3.client")
@patch("odoo_rag.llm.time.time")
def test_generate_with_system_prompt(mock_time, mock_boto_client):
    """Test generate with system prompt."""
    # Mock time to prevent timeout
    mock_time.return_value = 0.0

    mock_client = MagicMock()
    mock_boto_client.return_value = mock_client

    mock_client.converse.return_value = {
        "output": {"message": {"content": [{"text": "Answer"}]}},
        "usage": {"inputTokens": 50, "outputTokens": 25},
    }

    # Use real model ID from pricing dict
    generate(
        prompt="Question",
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        system="Be concise",
        timeout=5,
    )

    # Verify system prompt was passed
    call_kwargs = mock_client.converse.call_args[1]
    assert "system" in call_kwargs
    assert call_kwargs["system"][0]["text"] == "Be concise"


@patch("odoo_rag.llm.boto3.client")
def test_generate_with_custom_region(mock_boto_client, monkeypatch):
    """Test generate with custom AWS region."""
    monkeypatch.setenv("ODOO_LLM_BACKEND", "bedrock")

    mock_client = MagicMock()
    mock_boto_client.return_value = mock_client
    mock_client.converse.return_value = {
        "output": {"message": {"content": [{"text": "Answer"}]}},
        "usage": {"inputTokens": 10, "outputTokens": 10},
        "stopReason": "end_turn",
    }

    generate(prompt="Test", model_id="test-model", region="us-west-2")

    # Verify client was created with correct region
    mock_boto_client.assert_called_with("bedrock-runtime", region_name="us-west-2")


@patch("odoo_rag.llm.boto3.client")
def test_generate_error_handling(mock_boto_client, monkeypatch):
    """Test generate handles API errors gracefully."""
    monkeypatch.setenv("ODOO_LLM_BACKEND", "bedrock")

    mock_client = MagicMock()
    mock_boto_client.return_value = mock_client
    mock_client.converse.side_effect = Exception("API Error")

    result = generate(prompt="Test", model_id="test-model", max_retries=1)

    assert isinstance(result, ModelResponse)
    assert "error" in result.text.lower() or "failed" in result.text.lower()
    assert result.tokens_in == 0
    assert result.tokens_out == 0
    assert result.cost_usd == 0.0


@patch("odoo_rag.llm.boto3.client")
@patch("odoo_rag.llm.time.sleep")
@patch("odoo_rag.llm.time.time")
def test_generate_retry_logic(mock_time, mock_sleep, mock_boto_client):
    """Test retry logic on transient failures."""
    # Mock time to prevent timeout
    mock_time.return_value = 0.0

    mock_client = MagicMock()
    mock_boto_client.return_value = mock_client

    # First two calls fail, third succeeds
    mock_client.converse.side_effect = [
        Exception("Throttling"),
        Exception("Throttling"),
        {
            "output": {"message": {"content": [{"text": "Success"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5},
        },
    ]

    # Use real model ID from pricing dict
    response = generate(
        prompt="Test", model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0", timeout=5
    )

    # Should succeed after retries
    assert response.text == "Success"
    # Verify sleep was called for backoff
    assert mock_sleep.called


def test_generate_default_parameters(monkeypatch):
    """Test generate uses correct default parameters."""
    monkeypatch.setenv("ODOO_LLM_BACKEND", "echo")

    result = generate(prompt="Test", model_id="test-model")

    # Should work with defaults
    assert result is not None
    assert isinstance(result, ModelResponse)
