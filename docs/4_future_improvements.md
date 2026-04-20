# Future Improvements

## Current implementation

The RAG system demonstrates multi-model comparison (9 AWS Bedrock models), faithfulness scoring, cost/latency tracking, baseline comparison, automated pipeline (S3-triggered Lambda), ECS Fargate deployment, and Gradio UI.

Documentation coverage: Odoo 18 developer and user documentation including ORM, Sales, CRM, Accounting, Inventory, Website, Security, and Tutorials from odoo.com.

## Improvements for additional time

**CI/CD pipeline:** Add GitHub Actions workflow for automated testing and deployment. Auto-upload `sources.json` changes to S3 to trigger Lambda pipeline, then force ECS Fargate deployment on Lambda completion. Currently using manual `aws s3 cp` and `cdk deploy`.

**HTTPS with custom domain:** Register domain, request ACM certificate, configure Route 53, update ALB to HTTPS. Currently using HTTP-only ALB endpoint. Required for production and may fix clipboard copy issues.

**Migrate pipeline to ECS Fargate:** Remove Lambda timeout and memory limits. Supports larger documentation sets.

**Expand documentation:** Add more user guides, admin documentation, and additional modules. Add URLs to `data/sources.json` and upload to S3 for automatic rebuild.

**Improve RAG quality:** Explore additional approaches to enhance retrieval accuracy and context relevance.

**Multi-turn conversations:** Add conversation history for context-aware responses. Requires session management.

**Fix clipboard copy:** DataFrame copy button doesn't work reliably. CSV export works as workaround.

**Model selection for comparison:** Add multi-select dropdown to choose which models to compare. Reduces evaluation time and API costs.

## Priority

**Production readiness:**
1. Enable HTTPS with custom domain
2. Add CI/CD pipeline
3. Expand documentation coverage

**Scalability and UX:**
4. Migrate pipeline to ECS Fargate
5. Improve RAG quality (larger chunks, hybrid retrieval)
6. Multi-turn conversations
7. Model selection for comparison

**Nice to have:**
8. Additional evaluation metrics
9. Fix clipboard copy after HTTPS migration
