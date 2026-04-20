## Odoo RAG Multi-Model Comparison

- Main file: `src/odoo_rag/app.py`
  - Decision: A RAG-based documentation assistant with side-by-side comparison of 9 AWS Bedrock foundation models.
  - Models: Claude Haiku 4.5, Llama 3.1 8B/70B, Llama 3.3 70B, Llama 4 Maverick, Mistral 7B/Large, DeepSeek R1, Nova Pro.
  - Backend: `src/odoo_rag/llm.py` provides Bedrock Converse API integration with unified token tracking.

- Retrieval: FAISS vector index built from Odoo 18 documentation chunks (500 words, 10% overlap).
  - Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-dim).
  - Storage: S3 bucket with runtime download to container.

### Why this approach
- RAG enables multi-model comparison without fine-tuning. Fine-tuning would lock the system to a single model.
- Bedrock Converse API provides a unified interface for all models, eliminating provider-specific payload formats.
- Faithfulness and Answer Quality metrics demonstrate RAG value by comparing grounded answers (RAG) vs ungrounded answers (Baseline).
- S3-triggered Lambda automates index rebuilds when documentation changes, ensuring answers stay current.

### What was done in `src/odoo_rag/app.py`
- `calculate_faithfulness()` measures semantic similarity between answer and retrieved context (0.0 for Baseline, ~0.7 for RAG).
- `calculate_answer_quality()` measures semantic similarity between answer and documentation chunks.
- `run_comparison_eval()` runs both RAG and Baseline modes for each selected model, collecting cost, latency, and quality metrics.
- `create_charts()` generates 4 visualizations: Cost vs Latency (3D), Token Usage, Answer Quality, Response Latency.
- Gradio UI provides Chat tab (single model) and Model Comparison tab (multi-model evaluation).

### Data pipeline
- `src/odoo_rag/ingest.py` fetches Odoo 18 documentation from URLs in `data/sources.json`.
- `src/odoo_rag/indexer.py` chunks text (500 words, 50-word overlap), embeds with sentence-transformers, builds FAISS index.
- `infrastructure/lambda/handler.py` runs pipeline on S3 upload, triggered by `input/sources.json` changes.
- Output: `output/index.faiss` and `output/corpus.json` uploaded to S3.

### Deployment
- Platform: ECS Fargate with Application Load Balancer (WebSocket support for Gradio).
- Infrastructure: AWS CDK (Python) for reproducible deployments across AWS accounts.
- Initial attempt with Lambda failed due to 10-second init timeout (sentence-transformers loading) and WebSocket incompatibility.

### How to run locally
- Install dependencies:
  - `uv sync`
- Set AWS credentials and configure S3 bucket:
  - `export AWS_PROFILE=your-profile`
  - Update `S3_BUCKET` in `src/odoo_rag/config.py`
- Build FAISS index:
  - `uv run python -m odoo_rag.indexer`
- Run the app:
  - `uv run python -m odoo_rag.app`
- Open `http://127.0.0.1:7860` in a browser.

### Key results
- Answer Quality improvement: RAG shows consistent improvement over Baseline across all 9 models.
- Faithfulness gap: RAG scores ~0.7, Baseline scores ~0.0 (demonstrates grounding in retrieved context).
- Cost-latency tradeoff: Llama 3.1 8B provides lowest cost, Claude Haiku 4.5 provides best latency.
- Transparent reporting: All improvements (positive and negative) shown with true median, no cherry-picking.

### Documentation
- `docs/1_decisions_plan.md` - Initial architectural decisions (RAG vs fine-tuning, vector storage, deployment platform).
- `docs/2_project_summary.md` - High-level overview (this document).
- `docs/3_reflections.md` - Lessons learned (Lambda → ECS Fargate pivot, Hugging Face exploration, model availability).
- `docs/4_future_improvements.md` - Planned enhancements (model selection for comparison, caching, hybrid search).
- `docs/5_local_deploy.md` - Local setup and troubleshooting guide.
- `docs/6_cloud_deploy.md` - AWS deployment guide (ECS Fargate + CDK).