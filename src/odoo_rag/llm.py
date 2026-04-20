"""Multi-model LLM (Large Language Model) backend with cost and latency tracking.

This module handles communication with AWS Bedrock to generate human-readable answers
from retrieved documentation context.

Flow:
1. Receive prompt (question + retrieved context from retriever.py)
2. Send to AWS Bedrock Converse API
3. Track latency (response time) and cost (API pricing)
4. Return generated answer with metrics

Supported models (via AWS Bedrock):
- Claude: Haiku (fast, cheap)
- Llama: 3.1 8B, 3.1 70B, 3.3 70B, 4 Maverick 17B
- Mistral: 7B, Large
- DeepSeek: R1
- Amazon Nova: Pro

Note: Hugging Face code exists but is unused (available for future expansion)
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

import boto3
from botocore.exceptions import ClientError

# Semaphore to limit concurrent Bedrock API calls (prevents throttling)
# AWS Bedrock has rate limits (requests per second)
# Semaphore ensures max 3 concurrent calls to avoid hitting limits
# Prevents "TooManyRequestsException" errors
_bedrock_semaphore = asyncio.Semaphore(3)

# ============================================================================
# HUGGING FACE INFERENCE API SUPPORT (CURRENTLY UNUSED)
# ============================================================================
# The following code provides Hugging Face Inference API integration as an
# alternative to AWS Bedrock. This code is kept for reference in case the
# project needs to support Hugging Face models in the future.
#
# To enable: Uncomment the code below and call _generate_hf_api() instead of
# generate() when using HF models.
#
# Current status: NOT USED - project uses only AWS Bedrock
# ============================================================================

# from huggingface_hub import InferenceClient
# from huggingface_hub.utils import HfHubHTTPError

# # Hugging Face (HF) Inference API pricing - PRO TIER (production scale)
# # Format: (input_cost_per_1k_tokens, output_cost_per_1k_tokens) in USD
# # Based on HF Pro pricing: https://huggingface.co/pricing (~$0.10-0.50 per 1K requests)
# # Per-token costs are approximate estimates for small models
# # Actual costs may vary based on HF subscription tier and usage
# _HF_PRICING: dict[str, tuple[float, float]] = {
#     # Meta Llama
#     "meta-llama/Llama-3.2-3B-Instruct": (0.0001, 0.0002),  # ~$0.10-0.30/1K requests
#     # Google
#     "google/gemma-2-2b-it": (0.0001, 0.0002),              # ~$0.10-0.30/1K requests
#     # Microsoft
#     "microsoft/Phi-3-mini-4k-instruct": (0.0001, 0.0002),   # ~$0.10-0.30/1K requests
#     # Alibaba
#     "Qwen/Qwen2.5-3B-Instruct": (0.0001, 0.0002),           # ~$0.10-0.30/1K requests
# }


@dataclass
class ModelResponse:
    """Response from LLM with performance metrics for comparison.

    Contains both the generated text and metadata about the API call.
    Used to compare different models on cost, speed, and quality.
    """

    text: str  # Generated answer from LLM
    model: str  # Model ID (e.g., "us.anthropic.claude-haiku-4-5-20251001-v1:0")
    latency_sec: float  # Time taken to generate response (seconds)
    cost_usd: float  # Estimated cost in USD based on token usage
    tokens_in: int = 0  # Input tokens (prompt + context)
    tokens_out: int = 0  # Output tokens (generated answer)


# AWS Bedrock pricing (us-east-1 region, per 1K tokens) - Apr 2026
# Format: (input_cost_per_1k_tokens, output_cost_per_1k_tokens) in USD
# Reference: https://aws.amazon.com/bedrock/pricing/
#
# Note: Prices are approximate and subject to change
# Check AWS console for current rates in specific region
# Different regions may have different pricing
#
# Input tokens = prompt + context (what is sent to LLM)
# Output tokens = generated answer (what LLM returns)
_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic Claude
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": (0.00025, 0.00125),  # Claude Haiku
    # Meta Llama (on-demand pricing)
    "us.meta.llama3-1-8b-instruct-v1:0": (0.00022, 0.00022),  # Llama 3.1 8B
    "us.meta.llama3-1-70b-instruct-v1:0": (0.0009, 0.0009),  # Llama 3.1 70B
    "us.meta.llama3-3-70b-instruct-v1:0": (0.0009, 0.0009),  # Llama 3.3 70B
    "us.meta.llama4-maverick-17b-instruct-v1:0": (0.0005, 0.001),  # Llama 4 Maverick
    # Mistral
    "mistral.mistral-7b-instruct-v0:2": (0.00015, 0.0002),  # Mistral 7B
    "mistral.mistral-large-2402-v1:0": (0.004, 0.012),  # Mistral Large
    # DeepSeek
    "us.deepseek.r1-v1:0": (0.001, 0.005),  # DeepSeek R1
    # Amazon Nova
    "us.amazon.nova-pro-v1:0": (0.0008, 0.0032),  # Nova Pro
}


def _calculate_cost(model_id: str, tokens_in: int, tokens_out: int) -> float:
    """Calculate API call cost from actual token counts.

    Args:
        model_id: AWS Bedrock model identifier
        tokens_in: Number of input tokens (prompt + context)
        tokens_out: Number of output tokens (generated answer)

    Returns:
        Cost in USD, rounded to 6 decimal places
    """
    # Look up pricing for this model (case-insensitive)
    lookup = model_id.lower()

    # Fail fast if pricing not defined - better than returning wrong cost
    if lookup not in _PRICING:
        raise ValueError(
            f"Pricing not defined for model: {model_id}\n"
            f"Add pricing to _PRICING dictionary in llm.py"
        )

    price = _PRICING[lookup]

    # Calculate cost: (tokens / 1000) * price_per_1k_tokens
    cost = (tokens_in / 1000) * price[0] + (tokens_out / 1000) * price[1]
    return round(cost, 6)


# def _generate_hf_api(prompt: str, model_id: str, max_tokens: int = 512) -> ModelResponse:
#     """Generate response using Hugging Face Inference API (cloud-based alternative).
#
#     Alternative to AWS Bedrock for testing or when AWS is unavailable.
#     Requires HF_TOKEN environment variable with Hugging Face API token.
#
#     Args:
#         prompt: Full prompt (question + context)
#         model_id: Hugging Face model identifier (e.g., "meta-llama/Llama-3.2-3B-Instruct")
#         max_tokens: Maximum tokens to generate (default: 512)
#
#     Returns:
#         ModelResponse with generated text and metrics
#     """
#     # Check for Hugging Face API token in environment variables
#     # Token required for authentication with HF Inference API
#     # Get token from: https://huggingface.co/settings/tokens
#     token = os.environ.get("HF_TOKEN")
#     if not token:
#         return ModelResponse(
#             text="ERROR: HF_TOKEN not set",
#             model=model_id,
#             latency_sec=0.0,
#             cost_usd=0.0,
#             tokens_in=0,
#             tokens_out=0,
#         )
#
#     try:
#         # Create HF Inference API client with authentication
#         client = InferenceClient(model=model_id, token=token)
#
#         # Track request start time for latency measurement
#         start = time.time()
#
#         # Format prompt as chat message (required by chat_completion API)
#         # Use chat_completion instead of text_generation for better compatibility
#         messages = [{"role": "user", "content": prompt}]
#
#         # Call Hugging Face API to generate response
#         response = client.chat_completion(
#             messages=messages,
#             max_tokens=max_tokens,      # Limit response length
#             temperature=0.4,             # Creativity (0.0=deterministic, 1.0=creative)
#         )
#
#         # Calculate latency (time taken to get response)
#         latency = time.time() - start
#
#         # Extract generated text from API response
#         text = response.choices[0].message.content if response.choices else ""
#
#         # Estimate token counts (HF API doesn't always return exact counts)
#         # Rough estimate: 1 token ≈ 1 word (actual: 1 token ≈ 0.75 words)
#         tokens_in = len(prompt.split())
#         tokens_out = len(text.split()) if text else 0
#
#         # Fail fast if pricing not defined
#         if model_id not in _HF_PRICING:
#             raise ValueError(
#                 f"Pricing not defined for HF model: {model_id}\n"
#                 f"Add pricing to _HF_PRICING dictionary in llm.py"
#             )
#
#         # Calculate approximate cost based on estimated tokens
#         price = _HF_PRICING[model_id]
#         cost = (tokens_in / 1000) * price[0] + (tokens_out / 1000) * price[1]
#
#         return ModelResponse(
#             text=text,
#             model=model_id,
#             latency_sec=round(latency, 3),
#             cost_usd=round(cost, 6),
#             tokens_in=tokens_in,
#             tokens_out=tokens_out,
#         )
#     except Exception as e:
#         error_type = type(e).__name__
#         error_msg = str(e) if str(e) else "No error message provided"
#         return ModelResponse(
#             text=f"ERROR: {error_type}: {error_msg}",
#             model=model_id,
#             latency_sec=0.0,
#             cost_usd=0.0,
#             tokens_in=0,
#             tokens_out=0,
#         )


def generate(
    prompt: str,
    model_id: str,
    region: str = "us-east-1",
    system: str | None = None,
    max_retries: int = 6,
    timeout: int = 30,
) -> ModelResponse:
    """Generate response using AWS Bedrock LLM models.

    Main function to get LLM-generated answers for RAG system.
    Handles retries, timeouts, and error recovery automatically.

    Args:
        prompt: Full prompt (question + retrieved context)
        model_id: AWS Bedrock model identifier (e.g., "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        region: AWS region (default: "us-east-1")
        system: Optional system prompt (instructions for LLM behavior)
        max_retries: Maximum retry attempts on failure (default: 6)
        timeout: Maximum seconds to wait for response (default: 30)

    Returns:
        ModelResponse with generated text, latency, cost, and token counts
    """

    # Check backend mode (for debugging and testing)
    # ODOO_LLM_BACKEND=echo: returns prompt without API call (for debugging/testing, no AWS needed)
    # ODOO_LLM_BACKEND=bedrock: uses AWS Bedrock (default, production)
    backend = os.environ.get("ODOO_LLM_BACKEND", "bedrock")
    if backend == "echo":
        # Echo mode: return prompt without API call (for testing)
        return ModelResponse(
            text=f"[echo] {prompt[:200]}...",
            model="echo",
            latency_sec=0.0,
            cost_usd=0.0,
        )

    # Create AWS Bedrock client for specified region
    # Requires AWS credentials configured (via aws configure or IAM role)
    client = boto3.client("bedrock-runtime", region_name=region)

    # Build messages in Converse API format (unified format for all Bedrock models)
    # Converse API automatically handles different model formats (Claude, Llama, etc.)
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    # System prompt as separate parameter (optional)
    # System prompt = instructions for LLM behavior (e.g., "Be concise and accurate")
    # Some models (like Mistral) don't support system prompts
    system_list = [{"text": system}] if system else []

    # Inference configuration (controls LLM generation behavior)
    inference_config = {
        "maxTokens": 512,  # Maximum tokens to generate (limits response length)
        "temperature": 0.4,  # Creativity (0.0=deterministic, 1.0=creative)
        # Note: Claude models don't support topP parameter with temperature
    }

    # Track request start time for latency and timeout measurement
    start = time.time()

    # Exponential backoff for retries (2s, 4s, 8s, 16s, ...)
    # Prevents overwhelming API with rapid retry attempts
    backoff = 2.0

    # Retry loop: attempt up to max_retries + 1 times (7 attempts total by default)
    for attempt in range(max_retries + 1):
        try:
            # Check if total elapsed time exceeds timeout limit
            elapsed = time.time() - start
            if elapsed > timeout:
                # Return timeout error with diagnostic information
                return ModelResponse(
                    text=f"ERROR: Request timeout after {timeout}s. Possible causes:\n"
                    f"1. AWS credentials expired (run 'aws sts get-caller-identity' to check)\n"
                    f"2. Network connectivity issues\n"
                    f"3. Model taking too long to respond\n"
                    f"4. AWS Bedrock throttling (too many concurrent requests)\n"
                    f"Try: Refresh AWS credentials and retry",
                    model=model_id,
                    latency_sec=elapsed,
                    cost_usd=0.0,
                    tokens_in=0,
                    tokens_out=0,
                )

            # Use Converse API - unified interface for all Bedrock models
            # Handles model-specific formatting automatically (Claude, Llama, Mistral, etc.)
            kwargs = {
                "modelId": model_id,
                "messages": messages,
                "inferenceConfig": inference_config,
            }
            # Only include system prompt if provided (some models don't support it)
            if system_list:
                kwargs["system"] = system_list

            # Call AWS Bedrock API to generate response
            response = client.converse(**kwargs)

            # Extract generated text from API response
            # Response format: {"output": {"message": {"content": [{"text": "..."}]}}}
            text = ""
            output_message = response.get("output", {}).get("message", {})
            for content in output_message.get("content", []):
                if "text" in content:
                    text += content["text"]

            # Get actual token usage from response metadata
            # AWS Bedrock returns exact token counts (not estimates)
            usage = response.get("usage", {})
            tokens_in = usage.get("inputTokens", 0)  # Prompt + context tokens
            tokens_out = usage.get("outputTokens", 0)  # Generated answer tokens

            # Calculate latency (total time from request to response)
            latency = time.time() - start

            # Calculate cost based on actual token usage and model pricing
            cost = _calculate_cost(model_id, tokens_in, tokens_out)

            return ModelResponse(
                text=text,
                model=model_id,
                latency_sec=round(latency, 3),
                cost_usd=cost,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )

        except ClientError as e:
            # AWS Bedrock API error (authentication, throttling, validation, etc.)
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))

            # Handle models that don't support system messages (like Mistral)
            # Retry without system prompt if this error occurs
            if (
                "doesn't support system messages" in error_msg.lower()
                or "system message" in error_msg.lower()
            ):
                if system_list and attempt < max_retries:
                    system_list = []  # Remove system prompt
                    continue  # Retry without system prompt

            # Return error response instead of raising exception (graceful degradation)
            # ValidationException: Invalid parameters (wrong model ID, bad config)
            # ResourceNotFoundException: Model not available in region
            if error_code in ["ValidationException", "ResourceNotFoundException"]:
                return ModelResponse(
                    text=f"ERROR: {error_msg}",
                    model=model_id,
                    latency_sec=0.0,
                    cost_usd=0.0,
                    tokens_in=0,
                    tokens_out=0,
                )

            # If max retries reached, return error instead of crashing
            if attempt == max_retries:
                return ModelResponse(
                    text=f"ERROR: Max retries exceeded - {error_msg}",
                    model=model_id,
                    latency_sec=0.0,
                    cost_usd=0.0,
                    tokens_in=0,
                    tokens_out=0,
                )

            # Wait before retry (exponential backoff: 2s, 4s, 8s, 16s, ...)
            # Prevents overwhelming API with rapid retry attempts
            time.sleep(backoff)
            backoff *= 2  # Double wait time for next retry

        except Exception as e:
            # Catch-all for non-AWS errors (network issues, timeouts, unexpected errors)
            # Return error response instead of raising
            if attempt == max_retries:
                return ModelResponse(
                    text=f"ERROR: {str(e)}",
                    model=model_id,
                    latency_sec=0.0,
                    cost_usd=0.0,
                    tokens_in=0,
                    tokens_out=0,
                )
            time.sleep(backoff)
            backoff *= 2

    return ModelResponse(
        text="ERROR: Unexpected end of retry loop",
        model=model_id,
        latency_sec=0.0,
        cost_usd=0.0,
        tokens_in=0,
        tokens_out=0,
    )
