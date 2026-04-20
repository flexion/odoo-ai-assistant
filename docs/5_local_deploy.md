# Local Deployment Guide

**For AWS cloud deployment, see [6_cloud_deploy.md](6_cloud_deploy.md)**

## Prerequisites

- Python 3.12+
- `uv` package manager: `pip install uv`
- AWS credentials configured (for Bedrock API access)

## Deployment Steps

```bash
# 1. Navigate to project
cd odoo-ai-assistant

# 2. Configure AWS credentials
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_SESSION_TOKEN="your-token"  # If using temporary credentials
export AWS_REGION=us-east-1

# Verify credentials
aws sts get-caller-identity

# 3. Install dependencies and build index
make setup
make pipeline-data

# 4. Launch Gradio UI
make run

# 5. Open browser
# http://localhost:7860
```

**Expected time:** 5-10 minutes (embedding model download)

## What Gets Deployed

- Gradio UI at `http://localhost:7860`
- 9 AWS Bedrock models available for comparison
- FAISS vector index with Odoo documentation
- RAG metrics: Faithfulness Score, Context Precision

## Common Issues

| Problem | Solution |
|---------|----------|
| `No module named 'odoo_rag'` | Run `make setup` first |
| Bedrock access denied | Verify AWS credentials have `bedrock:InvokeModel` permission |
| Slow first run | First run downloads embedding model (~80MB) |

## Quick Commands

```bash
# Rebuild index
make pipeline-data

# Run UI
make run

# Update documentation sources
# Edit data/sources.json, then run:
make pipeline-data
```
