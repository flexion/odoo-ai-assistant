# Infrastructure Overview

AWS CDK infrastructure for deploying the Odoo RAG system with automated data pipeline and ECS Fargate hosting.

> **📚 For detailed deployment instructions, see [`../docs/6_cloud_deploy.md`](../docs/6_cloud_deploy.md)**

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AWS Cloud (us-east-1)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐         ┌─────────────────────────────────┐       │
│  │  S3 Bucket   │ trigger │  Lambda (Data Pipeline)         │       │
│  │              ├────────►│  - Scrapes Odoo docs            │       │
│  │ input/       │         │  - Builds FAISS index           │       │
│  │  sources.json│◄────────┤  - Uploads to S3 output/        │       │
│  │              │         └─────────────────────────────────┘       │
│  │ output/      │                                                    │
│  │  index.faiss │                                                    │
│  │  corpus.json │                                                    │
│  └──────┬───────┘                                                    │
│         │ Downloads on startup                                       │
│         ▼                                                            │
│  ┌─────────────────────────────────────────┐                        │
│  │  ECS Fargate (Gradio App)               │                        │
│  │  - 2 vCPU, 4GB RAM                      │                        │
│  │  - Loads FAISS index from S3            │                        │
│  │  - Calls Bedrock for 9 models           │                        │
│  └─────────────┬───────────────────────────┘                        │
│                ▼                                                     │
│  ┌─────────────────────────────────────────┐                        │
│  │  Application Load Balancer (ALB)        │                        │
│  │  - Public HTTP endpoint                 │                        │
│  │  - WebSocket support for Gradio         │                        │
│  └─────────────────────────────────────────┘                        │
│                │                                                     │
└────────────────┼─────────────────────────────────────────────────────┘
                 │
                 ▼
            End Users
```

## Data Pipeline Flow

```
1. Upload sources.json to S3
        ↓
2. S3 Event triggers Lambda
        ↓
3. Lambda Pipeline executes:
   ├─ odoo_rag.ingest      → Scrape Odoo docs
   ├─ odoo_rag.indexer     → Build FAISS index (2130 chunks)
   └─ Upload to S3         → index.faiss, corpus.json
        ↓
4. ECS Fargate downloads new index
        ↓
5. Users query via Gradio UI
        ↓
6. FAISS retrieves top-k chunks
        ↓
7. Bedrock generates responses (9 models)
        ↓
8. UI displays comparison charts
```

**Note:** QA generation (`generate_qa.py`) is disabled - RAG approach doesn't require synthetic training data. See [`../docs/3_reflections.md`](../docs/3_reflections.md) for rationale.

## Files in This Directory

| File | Purpose |
|------|---------|
| `app.py` | CDK app entry point |
| `cdk_stack.py` | Stack definitions (Pipeline + ECS App) |
| `lambda_pipeline.py` | Lambda handler for data pipeline |
| `Dockerfile.pipeline` | Docker image for pipeline Lambda |

## CDK Stacks

### 1. OdooRagPipelineStack
- **S3 Bucket** - Stores documentation and FAISS index (versioned)
- **Lambda Function** - Triggered by S3 uploads, runs data pipeline (2GB RAM, 15 min timeout)
- **IAM Roles** - S3 read/write, Bedrock access

### 2. OdooRagAppStack
- **ECS Fargate Service** - Runs Gradio app (2 vCPU, 4GB RAM)
- **Application Load Balancer** - Public HTTP endpoint with WebSocket support
- **IAM Roles** - S3 read, Bedrock access

## Quick Deploy

```bash
# 1. Bootstrap CDK (one-time)
cd infrastructure/
cdk bootstrap

# 2. Deploy pipeline
cdk deploy OdooRagPipelineStack

# 3. Trigger pipeline (builds index)
aws s3 cp ../data/sources.json s3://odoo-rag-data-<ACCOUNT-ID>/input/sources.json

# 4. Deploy app
cdk deploy OdooRagAppStack

# 5. Get ALB URL from outputs
# Visit: http://OdooR-OdooR-XXXXX.us-east-1.elb.amazonaws.com
```

> **📚 For detailed deployment steps, troubleshooting, and cost optimization, see:**
> - [`../docs/6_cloud_deploy.md`](../docs/6_cloud_deploy.md) - AWS deployment guide
> - [`../docs/5_local_deploy.md`](../docs/5_local_deploy.md) - Local testing
> - [`../docs/4_future_improvements.md`](../docs/4_future_improvements.md) - Future enhancements

## Updating Documentation Index

Upload new `sources.json` to trigger automatic rebuild:

```bash
# Edit sources
vim ../data/sources.json

# Upload (triggers Lambda pipeline automatically)
aws s3 cp ../data/sources.json s3://odoo-rag-data-<ACCOUNT-ID>/input/sources.json

# Monitor progress
aws logs tail /aws/lambda/OdooRagPipelineStack-Pipeline --follow
```

ECS Fargate will download the new index on next startup.

## Key Design Decisions

**Why ECS Fargate instead of Lambda?**
- Gradio requires WebSocket support (Lambda Function URLs don't support WebSockets properly)
- Sentence-transformers model loading exceeds Lambda's 10s init timeout
- See [`../docs/3_reflections.md`](../docs/3_reflections.md) for full rationale

**Why is QA generation disabled?**
- RAG uses documentation directly - no need for synthetic training data
- With new or updated sources, fine-tuning would require regenerating QA pairs and retraining (hours vs 2 min index rebuild)
- Documentation changes frequently - fine-tuning adds significant overhead for each update
- Additionally, multi-model comparison requires pre-trained models (can't fine-tune 9 different models)
- See [`../docs/3_reflections.md`](../docs/3_reflections.md) for RAG vs fine-tuning decision rationale

## Cleanup

```bash
# Delete all stacks
cdk destroy --all

# Manually delete S3 bucket (retained by default)
aws s3 rb s3://odoo-rag-data-<ACCOUNT-ID> --force
```
