"""Generate Q&A pairs from documentation using Claude.

NOTE: This module is currently DISABLED in the pipeline.

Purpose:
    Generates synthetic question-answer pairs from scraped Odoo documentation.
    These pairs can be used for:
    - Fine-tuning custom models on Odoo domain
    - Creating automated evaluation datasets
    - Building test suites for CI/CD

Why Disabled:
    This project uses pre-trained Bedrock models (Claude, Llama, etc.) with RAG
    instead of fine-tuning. QA generation is not required for the RAG approach.

    Rationale for RAG over fine-tuning:
    - Multi-model comparison is the core feature (fine-tuning locks to single model)
    - Documentation changes frequently (rebuild index in 2 min vs retrain for hours)
    - No training data preparation time required

    See docs/3_reflections.md for detailed decision rationale.

When to Enable:
    Enable in Makefile and infrastructure/lambda_pipeline.py when fine-tuning
    a custom model on the Odoo troubleshooting domain.
"""

from __future__ import annotations

import json
import os

import boto3


def generate_qa_from_chunk(chunk: str, source: str) -> list[dict]:
    """Generate Q&A pairs from a document chunk using Claude.

    Takes a text chunk from Odoo documentation and uses Claude to generate
    realistic question-answer pairs that users might ask.

    Args:
        chunk: Text excerpt from Odoo documentation
        source: URL or file path where chunk came from (for tracking)

    Returns:
        List of dicts with 'question', 'answer', 'source' keys
    """
    # Create AWS Bedrock client to call Claude API
    # Requires AWS credentials configured (via aws configure or IAM role)
    client = boto3.client("bedrock-runtime", region_name="us-east-1")

    # Build prompt asking Claude to generate question-answer pairs
    # Limit chunk to 2000 chars to avoid token limits
    prompt = f"""Generate 2-3 question-answer pairs based on this Odoo documentation excerpt.

Excerpt:
{chunk[:2000]}

For each pair, output in this format:
Q: <question>
A: <concise answer>

Focus on practical how-to questions that engineers or business users would ask."""

    try:
        # Call Claude API using legacy invoke_model (not Converse API)
        # Note: This uses older Anthropic-specific format, not unified Converse API
        response = client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",  # Claude Haiku (fast, cheap)
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",  # Anthropic API version
                    "max_tokens": 512,  # Limit response length
                    "temperature": 0.3,  # Low creativity (factual QA generation)
                    "messages": [{"role": "user", "content": prompt}],
                }
            ).encode(),
            contentType="application/json",
        )

        # Read and parse JSON response from Claude
        raw = response["body"].read().decode("utf-8")
        result = json.loads(raw)

        # Extract text from response (Claude returns list of content blocks)
        text = ""
        for block in result.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        # Parse Q&A pairs from Claude's response
        # Expected format:
        # Q: How do I create a sales order?
        # A: Navigate to Sales > Orders > Create
        qa_pairs = []
        lines = text.strip().split("\n")
        current_q = None  # Track current question being processed

        for line in lines:
            line = line.strip()
            # Found a question line
            if line.startswith("Q:"):
                current_q = line[2:].strip()  # Remove "Q: " prefix
            # Found answer line (and have a question waiting)
            elif line.startswith("A:") and current_q:
                qa_pairs.append(
                    {
                        "question": current_q,
                        "answer": line[2:].strip(),  # Remove "A: " prefix
                        "source": source,  # Track where this came from
                    }
                )
                current_q = None  # Reset for next Q&A pair

        return qa_pairs

    except Exception as e:
        # If Claude API fails, return empty list (don't crash entire pipeline)
        print(f"  Error generating QA: {e}")
        return []


def generate_all(
    corpus_path: str = "data/faiss_index/corpus.json",
    output_path: str = "data/qa_pairs.jsonl",
) -> None:
    """Generate Q&A pairs from corpus chunks for fine-tuning or evaluation.

    Reads corpus.json (created by indexer.py), samples a subset of chunks,
    and generates synthetic question-answer pairs using Claude.

    Note: This function generates a single unsplit dataset. It does NOT create
    train/validation/test splits. Manual splitting would be required for model
    training and evaluation.

    Args:
        corpus_path: Path to corpus.json file (default: data/faiss_index/corpus.json)
        output_path: Where to save QA pairs in JSONL format (default: data/qa_pairs.jsonl)
    """
    # Load corpus (all documentation chunks with metadata)
    with open(corpus_path) as f:
        corpus = json.load(f)

    # Sample subset of chunks (generating from all chunks would be expensive)
    # Using random sampling provides probabilistic diverse coverage
    # TODO: Implement stratified sampling for guaranteed topic diversity
    #   Ideas:
    #   1. Extract topics from source URLs (e.g., /sales/, /inventory/, /accounting/)
    #   2. Group chunks by topic and sample equally from each group
    #   3. Use embedding-based clustering to identify semantic topics
    #   4. Ensure minimum samples per topic (e.g., 10 chunks from each of 10 topics)
    import random

    random.seed(42)  # Reproducible sampling
    sample_size = min(100, len(corpus))  # Max 100 chunks (avoid high API costs)
    sampled = random.sample(corpus, sample_size)

    print(f"Generating QA for {sample_size} chunks...")

    # Generate Q&A pairs from each sampled chunk
    all_qa = []
    for i, doc in enumerate(sampled):
        # Progress indicator (print every 10 chunks)
        if i % 10 == 0:
            print(f"  {i}/{sample_size}...")

        # Call Claude to generate 2-3 Q&A pairs from this chunk
        qa = generate_qa_from_chunk(doc["text"], doc["source"])
        all_qa.extend(qa)  # Add to master list

    # Save all Q&A pairs to JSONL file (one JSON object per line)
    # JSONL format is standard for training data (easy to stream/process)
    with open(output_path, "w") as f:
        for qa in all_qa:
            f.write(json.dumps(qa) + "\n")

    print(f"Generated {len(all_qa)} Q&A pairs → {output_path}")


def main():
    """Entry point when running this script directly.

    Checks for AWS credentials and generates Q&A pairs from corpus.
    """
    # Warn if AWS credentials not configured (Claude API will fail)
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        print(
            "Warning: AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
        )

    # Generate Q&A pairs and save to data/qa_pairs.jsonl
    generate_all()
    print("Done.")


if __name__ == "__main__":
    main()
