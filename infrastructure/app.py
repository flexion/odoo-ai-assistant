#!/usr/bin/env python3
"""CDK app entry point for Odoo RAG deployment."""

import os
import aws_cdk as cdk
from cdk_stack import OdooRagPipelineStack, OdooRagAppStack

app = cdk.App()

# Get AWS account and region from environment or CDK context
env = cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
)

# Stack 1: Data pipeline (S3 + Lambda trigger)
pipeline_stack = OdooRagPipelineStack(
    app,
    "OdooRagPipelineStack",
    env=env,
)

# Stack 2: Gradio app (ECS Fargate + ALB)
app_stack = OdooRagAppStack(
    app,
    "OdooRagAppStack",
    data_bucket_name=pipeline_stack.data_bucket_name,
    env=env,
)

# App stack depends on pipeline stack (needs bucket to exist)
app_stack.add_dependency(pipeline_stack)

app.synth()
