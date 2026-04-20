"""List and test available Bedrock foundation models."""

import boto3

bedrock = boto3.client("bedrock", region_name="us-east-1")

# List models
response = bedrock.list_foundation_models(byOutputModality="TEXT")
models = response["modelSummaries"]

print(f"\n{'Model ID':<60} {'Provider':<15}")
print("-" * 80)

instruct_models = [m for m in models if "instruct" in m["modelId"].lower()]
for model in sorted(instruct_models, key=lambda x: x["modelId"]):
    print(f"{model['modelId']:<60} {model.get('providerName', 'Unknown'):<15}")

print(f"\n\nFound {len(instruct_models)} instruction models")

# Test specific models
print("\n\n--- Testing Quick Prompt ---")
from odoo_rag.llm import generate

test_models = [
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "mistral.mistral-7b-instruct-v0:2",
    "mistral.mistral-large-2402-v1:0",
    "meta.llama3-8b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0",
]

for model_id in test_models:
    try:
        result = generate("Say 'OK'", model_id=model_id)
        print(f"✓ {model_id:<50} OK ({result.latency_sec:.2f}s)")
    except Exception as e:
        print(f"✗ {model_id:<50} FAILED: {str(e)[:40]}")
