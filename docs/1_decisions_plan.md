## Action Plan

### Project Motivation

Working on Odoo development at Flexion, I frequently encountered AI-generated responses that were hallucinated or not grounded in official documentation. Over the past year, our team has handled 30+ Odoo issues, with 30+ more currently in the backlog. This project aims to build a tool that provides **grounded, documentation-based answers** to help developers get accurate information quickly. The multi-model comparison feature enables identifying which models perform best for Odoo questions, and the RAG architecture allows easy improvement when discrepancies are found—simply update the documentation index rather than retrain a model.

### Objective

Build a RAG-based Odoo 18 documentation assistant with multi-model comparison capabilities. This plan outlines the initial architectural decisions made at project start to enable side-by-side evaluation of AWS Bedrock foundation models on cost, latency, and answer quality metrics.

### Decision 1: RAG vs Fine-Tuning

**Decision**: Use **Retrieval-Augmented Generation (RAG)** instead of fine-tuning foundation models.

**Options**:
- RAG with Foundation Models: Use FAISS + sentence-transformers to retrieve relevant context from Odoo docs, then pass context + query to foundation models via Bedrock Converse API.
- Fine-tuning Custom Model: Import custom Llama 3.2 3B via Bedrock Custom Model Import with balanced training data across 80+ Odoo troubleshooting categories.

#### Pros:
- RAG: Faster deployment (hours vs weeks), no training data preparation needed, instant knowledge updates when docs change, enables multi-model comparison, zero training cost.
- Fine-tuning: Potentially faster inference (no retrieval step), can work offline if model deployed locally, model learns domain-specific patterns.

#### Cons:
- RAG: Slightly higher latency (retrieval overhead), context window limits, depends on retrieval quality.
- Fine-tuning: Data preparation burden (thousands of balanced QA pairs needed across 80+ categories), training time, single model lock-in (can't compare multiple models), update friction (retrain when docs change), training data imbalance issues.

**Rationale**: Initially considered fine-tuning Llama 3.2 3B on Odoo troubleshooting data, but training data imbalance across 80+ categories presented challenges. RAG enables the core feature: comparing multiple foundation models side-by-side. Fine-tuning would lock the project to a single model and require weeks of data preparation. The ability to update knowledge when documentation changes is valuable for a documentation assistant.

### Decision 2: Vector Storage Architecture

**Decision**: Use **Amazon S3 for FAISS index persistence** with runtime download.

**Options**:
- S3 + Local FAISS: Store `index.faiss` and `corpus.json` in S3 bucket, download to `/tmp` or container storage at startup.
- Amazon OpenSearch Serverless: Managed vector search with auto-scaling and pay-per-query pricing.
- EFS Shared Volume: Mount shared storage across containers for fast access.

#### Pros:
- S3 + Local FAISS: Simple implementation, cheap storage (~$0.02/GB), easy CI/CD integration, works well for small indices.
- OpenSearch Serverless: No file management, built-in hybrid search, auto-scaling.
- EFS: Fast access with no download latency, shared across containers.

#### Cons:
- S3 + Local FAISS: Cold-start latency (1-5s for small indices), requires download on each container start.
- OpenSearch Serverless: Higher cost (~$0.40/OCU), more complex setup for MVP.
- EFS: Higher cost (~$0.30/GB), requires VPC setup and configuration.

**Rationale**: S3 provides simple storage for initial deployment. Index files are small and rebuildable via automated pipeline. Easy to share across team and redeploy to different AWS accounts by changing bucket name.

### Decision 3: Application Deployment Platform

**Decision**: Start with **AWS Lambda + Function URL** for simplicity, with **ECS Fargate** as fallback if needed.

**Options**:
- AWS Lambda + Function URL: Serverless container deployment with auto-scaling and pay-per-request pricing.
- ECS Fargate with ALB: Container service with load balancer for HTTP/WebSocket traffic.
- SageMaker Endpoint: ML-focused hosting with built-in monitoring.

#### Pros:
- Lambda: Serverless with auto-scaling, pay-per-request pricing, no server management.
- ECS Fargate: No timeout limits, WebSocket support for Gradio, persistent connections, health checks.
- SageMaker: Built-in monitoring and A/B testing capabilities.

#### Cons:
- Lambda: 15-minute timeout limit, 10-second init timeout (may be tight for ML model loading), WebSocket limitations.
- ECS Fargate: Always-on cost (~$20/month minimum), more complex setup than Lambda.
- SageMaker: Overkill for non-ML workloads, expensive for simple web apps.

**Rationale**: Lambda provides serverless deployment with auto-scaling. If init timeout issues occur with sentence-transformers loading or WebSocket problems with Gradio, can pivot to ECS Fargate.

### Decision 4: Model Access Strategy

**Decision**: Use **AWS Bedrock Converse API** as primary backend, with **Hugging Face Inference API** as fallback.

**Options**:
- Bedrock Converse API: Unified API for Claude, Llama, Mistral, DeepSeek, Nova with automatic format handling.
- Bedrock InvokeModel: Provider-specific API requiring different payload formats (Claude uses Anthropic format, Llama uses Meta format, etc.).
- Hugging Face Inference API: Cloud-hosted models via Hugging Face API with HF_TOKEN authentication.

#### Pros:
- Bedrock Converse API: Single code path for all models, unified token count format, automatic format handling, reduces code complexity.
- Bedrock InvokeModel: More control over payload structure and parameters.
- Hugging Face API: Access to full HF Hub model catalog, alternative if Bedrock has connectivity issues or regional availability problems.

#### Cons:
- Bedrock Converse API: Requires AWS credentials and model access grants, regional availability varies.
- Bedrock InvokeModel: Requires more lines of format-specific code per provider, provider-specific token count parsing, error-prone.
- Hugging Face API: Requires HF_TOKEN, different pricing model, network-dependent latency.

**Rationale**: Converse API reduces code complexity and provides accurate token-based cost tracking. Suitable for comparing multiple foundation models without managing format differences. Hugging Face support will be added as fallback in case of AWS Bedrock connectivity issues.

### Decision 5: Data Pipeline & Knowledge Base Updates

**Decision**: Use **S3-triggered Lambda** for automated index rebuilds when sources change.

**Options**:
- S3 Event + Lambda: Upload new `sources.json` to S3 triggers Lambda that runs ingest → chunk → embed → build index → upload to S3.
- ECS Fargate Task: Run pipeline as containerized Fargate task if Lambda timeout limits are hit.
- Manual Pipeline: Run pipeline locally and upload to S3.

#### Pros:
- S3 Event + Lambda: Automated, versioned, reproducible, triggered by S3 upload.
- ECS Fargate Task: No time limits, can handle large documentation sites, containerized.
- Manual Pipeline: Simple for initial setup and testing.

#### Cons:
- S3 Event + Lambda: 15-minute timeout may limit processing for large doc sites.
- ECS Fargate Task: More complex setup, requires container orchestration.
- Manual Pipeline: Not reproducible, requires local AWS credentials, manual process.

**Rationale**: S3-triggered Lambda provides automation. Update `sources.json` → new index automatically available to all deployed instances. The 15-minute timeout should be sufficient for Odoo documentation processing. If timeout limits are hit, can pivot to ECS Fargate task for longer processing.

### Decision 6: Evaluation Metrics for RAG

**Decision**: Use **RAG-specific metrics** (Faithfulness, Answer Quality) with transparent reporting of all results.

**Options**:
- RAG-Specific Metrics: Faithfulness (answer grounding in context), Answer Quality (documentation alignment), calculated via semantic similarity with sentence-transformers.
- Generic Accuracy Metrics: Compare to ground truth QA pairs using BLEU, ROUGE, or semantic similarity.
- LLM-as-Judge: Use another LLM to evaluate answer quality.

#### Pros:
- RAG-Specific Metrics: Demonstrates RAG value with clear gap between RAG and Baseline, no ground truth needed (uses retrieved context as reference), actionable insights about retriever quality and hallucination risk.
- Generic Accuracy: Measures absolute correctness against known answers.
- LLM-as-Judge: Can evaluate nuanced aspects like helpfulness and tone.

#### Cons:
- RAG-Specific Metrics: Doesn't measure absolute correctness (only grounding in retrieved docs).
- Generic Accuracy: Requires ground truth QA pairs, doesn't show RAG-specific value.
- LLM-as-Judge: Expensive (additional API calls), can be biased, less reproducible.

**Rationale**: For a RAG system, showing that answers are **grounded in retrieved context** is important. Faithfulness shows the difference between RAG (high score) and Baseline (low score), proving the system works as intended. Answer Quality measures how well answers align with official documentation.

### Decision 7: Text Chunking Strategy

**Decision**: Use **word-based sliding window chunking with 500 words per chunk and 50-word overlap (10%)**.

**Options**:
- Word-based sliding window with overlap: 500 words per chunk, 50-word overlap to preserve context across boundaries.
- Character-based fixed size: 1000-1500 characters per chunk with no overlap.
- Semantic chunking: Split at sentence or paragraph boundaries.

#### Pros:
- Word-based sliding window: Balanced size (~750-1000 tokens with all-MiniLM-L6-v2), context preservation via overlap, fast processing, predictable chunk sizes for batching.
- Character-based: Exact byte control, simpler implementation.
- Semantic chunking: Natural breaks at sentence/paragraph boundaries, preserves meaning units.

#### Cons:
- Word-based sliding window: May split mid-sentence (mitigated by overlap), not semantically aware (might split code blocks).
- Character-based: May cut words mid-character, no context preservation across chunks.
- Semantic chunking: Variable chunk sizes (problematic for batch embedding), slower processing.

**Rationale**: For technical documentation like Odoo, word-based chunking provides speed, predictability, and context preservation. The 500-word chunk size with 10% overlap preserves context at chunk boundaries for technical troubleshooting questions spanning multiple paragraphs.

### Decision 8: Infrastructure as Code

**Decision**: Use **AWS CDK (Python)** for reproducible deployments.

**Options**:
- AWS CDK (Python): Python-based infrastructure definition with type safety and IDE support.
- Terraform: Industry-standard IaC tool with HCL syntax and multi-cloud support.
- CloudFormation (YAML): Native AWS service with direct integration.

#### Pros:
- AWS CDK: Type-safe with IDE autocomplete, testing support, reproducible across accounts, matches application language (Python).
- Terraform: Industry-standard tool, state management, multi-cloud portability.
- CloudFormation: Direct AWS integration, no extra tooling required.

#### Cons:
- AWS CDK: Learning curve if new to CDK, generates CloudFormation templates.
- Terraform: Separate tool and language (HCL), requires state file management.
- CloudFormation: Verbose YAML, harder to maintain for complex stacks.

**Rationale**: CDK provides Python-native infrastructure code, matching the application language. Parameterizable for different AWS accounts via context variables. Type safety catches errors at development time.

### Decision 9: Deployment Strategy

**Decision**: Use **AWS deployment from local machine** with documented steps.

**Options**:
- Local deployment only: Run application locally for development and testing.
- AWS deployment from local: Local Docker build → push to ECR → deploy via CDK to AWS.
- GitHub Actions automation: Automated build and deploy on PR merge.

#### Pros:
- Local deployment only: Simple setup, no AWS costs, fast iteration.
- AWS deployment from local: Cloud deployment with manual control, no CI/CD complexity, documented steps.
- GitHub Actions automation: Automated workflow, reproducible builds, no manual steps.

#### Cons:
- Local deployment only: Not accessible to team, no production environment.
- AWS deployment from local: Manual steps for each deployment, requires local AWS credentials.
- GitHub Actions automation: Requires GitHub secrets setup, additional complexity.

**Rationale**: AWS deployment from local machine enables cloud deployment while keeping focus on RAG functionality. Avoids complexity of CI/CD pipeline setup for initial release. Future improvement: automate with GitHub Actions if needed.

---

## Planned Implementation Phases

### Phase 1: Core RAG Foundation
- [ ] Set up FAISS retriever with sentence-transformers (all-MiniLM-L6-v2)
- [ ] Integrate AWS Bedrock Converse API for multiple models
- [ ] Build data pipeline: ingest → chunk → embed → index
- [ ] Create basic Gradio UI with Chat tab
- [ ] Test with 2-3 models initially

### Phase 2: Multi-Model Comparison
- [ ] Add Model Comparison tab to Gradio UI
- [ ] Implement Faithfulness and Answer Quality metrics
- [ ] Add Baseline vs RAG comparison mode
- [ ] Create visualization charts for cost, latency, quality
- [ ] Test with full model suite (target: 9 models)

### Phase 3: Automation & Deployment
- [ ] Set up S3-triggered Lambda for automated index rebuilds
- [ ] Deploy to AWS (Lambda or ECS based on testing results)
- [ ] Configure public URL for team access
- [ ] Limit concurrent model testing to avoid rate limits

### Phase 4: Polish & Documentation
- [ ] Refine UI based on testing feedback
- [ ] Complete documentation (README, DEVELOPMENT, REFLECTIONS)
- [ ] Prepare presentation materials

---

## Target Model Suite

Planning to support 9 AWS Bedrock foundation models across different providers:

**Anthropic**: Claude Haiku 4.5
**Meta**: Llama 3.1 8B, 3.1 70B, 3.3 70B, 4 Maverick
**Mistral AI**: Mistral 7B, Mistral Large
**DeepSeek**: DeepSeek R1
**Amazon**: Nova Pro

*Note: Actual model availability may vary by AWS account and region. Will select available models using `scripts/list_models.py`.*
