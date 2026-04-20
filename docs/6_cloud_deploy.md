# AWS Cloud Deployment Guide

**For local development and testing, see [5_local_deploy.md](5_local_deploy.md)**

## Prerequisites

- Node.js 14+ and AWS CDK: `npm install -g aws-cdk`
- AWS CLI configured: `aws configure`
- AWS Account with Bedrock access (us-east-1) and IAM permissions

## Deployment Steps

### Step 1: Configure AWS Credentials

```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_SESSION_TOKEN="your-token"  # If using temporary credentials
export AWS_REGION=us-east-1

# Verify credentials
aws sts get-caller-identity
```

### Step 2: Install Dependencies

```bash
# From project root (odoo-ai-assistant/)
uv pip install -e ".[deploy]"
```

### Step 3: Bootstrap CDK (One-Time, 5 min)

```bash
# Navigate to infrastructure folder
cd infrastructure/

# Get AWS account ID
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Bootstrap CDK (creates S3 bucket for CDK assets)
uv run cdk bootstrap aws://${AWS_ACCOUNT_ID}/us-east-1
```

**Expected Output:**
```
✅  Environment aws://123456789012/us-east-1 bootstrapped
```

---

### Step 4: Deploy Pipeline Stack (10 min)

**What it creates:**
- S3 bucket for data storage
- Lambda function for pipeline execution
- S3 event trigger on `sources.json` upload

```bash
# Still in infrastructure/ folder
uv run cdk deploy OdooRagPipelineStack

# Confirm with 'y' when prompted
```

**Expected Output:**
```
✅  OdooRagPipelineStack

Outputs:
OdooRagPipelineStack.DataBucketName = odoo-rag-data-2026-cohort-v2
OdooRagPipelineStack.PipelineFunctionName = OdooRagPipelineStack-OdooRagPipelineFunction...
```

**Save the bucket name for next steps!**

---

### Step 5: Trigger Data Pipeline (5 min upload + 7 min processing)

```bash
# Go back to project root
cd ..

# Upload sources.json (triggers pipeline automatically!)
aws s3 cp data/sources.json s3://odoo-rag-data-2026-cohort-v2/input/sources.json
```

**Monitor pipeline progress (optional):**
```bash
# Get function name
FUNCTION_NAME=$(aws lambda list-functions --query 'Functions[?contains(FunctionName, `OdooRagPipelineFunction`)].FunctionName' --output text | awk '{print $1}')

# Tail logs
aws logs tail /aws/lambda/$FUNCTION_NAME --follow
```

**Expected Log Output:**
```
Step 1: Ingesting documentation... (5-7 min)
Step 2: Building FAISS index... (30s)
✓ Uploaded output/index.faiss
✓ Uploaded output/corpus.json
Pipeline completed successfully
```

**Verify artifacts exist:**
```bash
aws s3 ls s3://odoo-rag-data-2026-cohort-v2/output/
```

**Expected:**
```
2026-04-16 11:00:00     512345 corpus.json
2026-04-16 11:00:01     123456 index.faiss
```

**⚠️ DO NOT proceed to Step 6 until you see both files!**

---

### Step 6: Deploy Gradio App (10 min)

**What it creates:**
- ECS Fargate cluster
- Application Load Balancer (ALB)
- ECS service running Gradio app
- Public HTTP endpoint

```bash
# Go back to infrastructure folder
cd infrastructure/

# Deploy app stack
uv run cdk deploy OdooRagAppStack

# Confirm with 'y' when prompted
```

**Expected Output:**
```
✅  OdooRagAppStack

Outputs:
OdooRagAppStack.AppUrl = http://OdooR-OdooR-XXXXX.us-east-1.elb.amazonaws.com
```

**🎉 App is live!**

## Verification

```bash
# Open the URL from Step 6 output
open http://OdooR-OdooR-XXXXX.us-east-1.elb.amazonaws.com
```

**Expected:** Gradio UI loads with Chat and Model Comparison tabs. First request may take 10-20 seconds.

## Updating Documentation

```bash
# Edit sources.json
vim data/sources.json

# Upload (triggers pipeline automatically)
aws s3 cp data/sources.json s3://odoo-rag-data-2026-cohort-v2/input/sources.json

# Wait ~10 minutes, then app uses new index automatically
```

## Common Issues

| Problem | Solution |
|---------|----------|
| 502 Bad Gateway | Check ECS logs: `aws ecs list-tasks --cluster OdooRagCluster` |
| Index not found | Verify pipeline completed: `aws s3 ls s3://odoo-rag-data-2026-cohort-v2/output/` |

## Cleanup

```bash
cd infrastructure/
uv run cdk destroy OdooRagAppStack
uv run cdk destroy OdooRagPipelineStack
```

## Cost Estimate

**~$50/month** (ECS Fargate $30 + ALB $16 + Lambda $1 + Bedrock $2 + S3 $0.002)
