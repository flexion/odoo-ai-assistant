# Reflections

## Summary (what was done)
- Built a RAG-based chatbot for Odoo 18 documentation with multi-model comparison.
- Used AWS Bedrock Converse API to compare 9 foundation models side-by-side.
- Implemented FAISS vector search with 500-word chunks and 50-word overlap.
- Created Gradio UI with chat mode and comparison mode.
- Deployed to AWS ECS Fargate with Application Load Balancer.
- Wrote 63 tests achieving 88% coverage (excluding Gradio UI code).
- Tracked cost, latency, and faithfulness metrics for each model.

## How things went

**RAG vs fine-tuning decision:** The plan initially considered fine-tuning Llama 3.2 3B on Odoo data, but training data imbalance across 80+ categories presented challenges. Switching to RAG enabled faster deployment and the core feature: multi-model comparison. Fine-tuning would have locked the project to a single model.

**Bedrock Converse API:** Using the unified Converse API instead of provider-specific `invoke_model` reduced code by 40% and provided actual token counts. Adding new models required only adding the model ID to configuration.

**Deployment platform migration:** Initial Lambda deployment failed due to 10-second init timeout (sentence-transformers loading exceeded limit) and WebSocket incompatibility (Gradio requires persistent connections). Migrated to ECS Fargate with ALB, which supports long-lived processes and full WebSocket functionality.

**Hugging Face exploration:** Added HF Inference API support as alternative backend when AWS Bedrock showed connectivity issues. The code exists in `llm.py` but remains unused after Bedrock issues were resolved. Kept as fallback option.

**Chunking strategy:** Word-based sliding window (500 words, 50-word overlap) produced 54 chunks from Odoo docs. Semantic chunking was tested but created 2130 small fragments vs 74 coherent chunks, degrading performance.

**Timeout handling:** Long-running comparisons (90 API calls for 9 models × 5 questions × 2 modes) hung for 25+ minutes when AWS credentials expired. Added 30-second timeout per request with helpful error messages listing 4 possible causes and actionable fixes.

**Single-question strategy:** Multi-question evaluation showed high variance due to LLM non-determinism. Same model + same question produced different quality on different runs. Switched to single-question default for stable, interpretable results. One question × 9 models = 18 API calls instead of 90.

## What worked well
- Bedrock Converse API simplified multi-model backend
- FAISS + S3 provided simple, cheap vector storage
- RAG enabled fast deployment and instant updates vs fine-tuning
- Test coverage at 88% with proper mocking

## Potential improvements

See `4_future_improvements.md` for enhancements that could be implemented with additional time.

## What was learned

- RAG provides a faster path to deployment than fine-tuning when comparing multiple models. The unified Converse API simplified multi-model support significantly.
- Lambda works well for stateless functions but not for interactive web apps requiring WebSockets or heavy model initialization. ECS Fargate proved to be a better option than Lambda for Gradio applications.
- LLM variance across questions makes single-question evaluation more stable and interpretable than multi-question averaging. One well-chosen question produces clearer results than averaging across multiple questions with high variance.
- Simple word-based chunking outperformed semantic chunking for technical documentation. Starting with simple approaches and only adding complexity when needed proved effective.
- Timeout handling and progress feedback are essential for long-running operations. Clear error messages with actionable fixes improve the user experience significantly.
