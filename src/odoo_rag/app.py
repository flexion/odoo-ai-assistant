"""Gradio web UI for Odoo RAG chatbot with multi-model comparison.

This module provides:
1. Interactive chat interface - Ask questions about Odoo 18 documentation
2. Model comparison - Test multiple LLMs (Claude, Llama, Mistral, etc.) side-by-side
3. Evaluation framework - Compare models on cost, latency, and answer quality
4. Visualizations - Charts showing performance metrics across models

Flow:
1. User asks question → Retriever finds relevant docs → LLM generates answer
2. Comparison mode: Run same question through multiple models
3. Evaluation mode: Batch test on question set, generate performance charts

Deployment:
- Local: python -m odoo_rag.app (uses local data/faiss_index)
- AWS: Deployed on ECS Fargate (downloads data from S3)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Suppress HuggingFace Hub warnings and telemetry
# These environment variables must be set BEFORE importing sentence_transformers
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = (
    "1"  # Suppress symlink warnings when downloading models
)
os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # Avoid tokenizer deadlocks in multi-threaded apps
)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"  # Disable usage tracking to HuggingFace

# Suppress Python warnings about unauthenticated HuggingFace requests
import warnings

warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

import gradio as gr
import pandas as pd
import plotly.express as px

from odoo_rag.llm import generate
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from odoo_rag.retriever import Retriever


def download_data_from_s3():
    """Download FAISS index and corpus from S3 if running on ECS Fargate.

    When deployed to AWS ECS, the app downloads data files from S3 to /tmp (writable storage).
    For local development, this function does nothing (uses local data/ directory).

    Environment variables:
        S3_BUCKET: S3 bucket name (e.g., "odoo-rag-data-bucket")
        S3_PREFIX: Prefix path in bucket (default: "data/")
    """
    # Check if running in cloud environment (S3_BUCKET set by CDK deployment)
    s3_bucket = os.environ.get("S3_BUCKET")
    s3_prefix = os.environ.get("S3_PREFIX", "data/")

    # Local development: use local data files
    if not s3_bucket:
        print("S3_BUCKET not set, using local data files")
        return

    # Create S3 client for downloading files
    import boto3

    s3 = boto3.client("s3")

    # Create data directory in /tmp (only writable location in ECS containers)
    # ECS containers have read-only filesystems except /tmp
    data_dir = Path("/tmp/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # List of files to download from S3
    # index.faiss: FAISS vector database for similarity search
    # corpus.json: Original text chunks with metadata
    files_to_download = [
        (f"{s3_prefix}index.faiss", data_dir / "index.faiss"),
        (f"{s3_prefix}corpus.json", data_dir / "corpus.json"),
    ]

    # Download each file from S3 (skip if already cached in /tmp)
    for s3_key, local_path in files_to_download:
        # Check if file already exists (cached from previous container restart)
        if not local_path.exists():
            print(f"Downloading s3://{s3_bucket}/{s3_key} to {local_path}")
            try:
                s3.download_file(s3_bucket, s3_key, str(local_path))
                print(f"✓ Downloaded {s3_key}")
            except Exception as e:
                print(f"✗ Failed to download {s3_key}: {e}")
        else:
            # File exists from previous container restart (ECS task reuse)
            print(f"✓ Using cached {local_path}")

    # Update environment variable so Retriever knows where to find data
    # Retriever reads FAISS_INDEX_PATH to locate index.faiss and corpus.json
    os.environ["FAISS_INDEX_PATH"] = str(data_dir)


# Download data on module import (runs once per ECS task startup)
# Task startup = first run after deployment or container restart
# Subsequent requests reuse same container and skip download (cached in /tmp)
download_data_from_s3()


def calculate_context_recall(
    answer: str, context: str, embedder: SentenceTransformer
) -> float:
    """Calculate context recall - does retrieved context contain info needed for answer?

    Measures if the retriever found relevant documentation.

    Method: Semantic similarity between answer and retrieved context
    - 1.0 = Context contains all info in answer (excellent retrieval)
    - 0.5 = Context partially relevant (okay retrieval)
    - 0.0 = Context doesn't support answer (poor retrieval)

    Args:
        answer: LLM-generated answer
        context: Retrieved documentation chunks
        embedder: SentenceTransformer model for encoding text

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not answer or not context:
        return 0.0

    embeddings = embedder.encode([answer, context], show_progress_bar=False)
    similarity = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
    return max(0.0, min(1.0, similarity))


# AWS Bedrock models (>3B parameters, all available and working)
# This dictionary maps model names to their respective model IDs.
# These models are used for generating answers to user queries.
AVAILABLE_MODELS = {
    # ========== ANTHROPIC ==========
    "Claude Haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",  # Fast, efficient
    # ========== META LLAMA ==========
    "Llama 3.1 8B": "us.meta.llama3-1-8b-instruct-v1:0",  # Balanced
    "Llama 3.1 70B": "us.meta.llama3-1-70b-instruct-v1:0",  # High quality
    "Llama 3.3 70B": "us.meta.llama3-3-70b-instruct-v1:0",  # Latest Llama
    "Llama 4 Maverick": "us.meta.llama4-maverick-17b-instruct-v1:0",  # Newest Llama
    # ========== MISTRAL ==========
    "Mistral 7B": "mistral.mistral-7b-instruct-v0:2",  # Efficient mid-size
    "Mistral Large": "mistral.mistral-large-2402-v1:0",  # High quality
    # ========== DEEPSEEK ==========
    "DeepSeek R1": "us.deepseek.r1-v1:0",  # Reasoning model
    # ========== AMAZON NOVA ==========
    "Nova Pro": "us.amazon.nova-pro-v1:0",  # Amazon's best
}

# This is the system prompt used for all models.
# It provides context and instructions for the model to generate accurate answers.
SYSTEM_PROMPT = """You are an Odoo 18 expert assistant. Use the provided context to answer accurately.
If the context doesn't contain the answer, say so. Be concise and helpful."""


def create_app() -> gr.Blocks:
    """Create Gradio app with model selector."""

    # Load retriever
    try:
        retriever = Retriever()
    except Exception as e:
        print(f"Warning: Could not load FAISS index: {e}")
        retriever = None

    def chat(
        message: str, model_name: str, history: list
    ) -> tuple[str, list, dict, dict]:
        """Process a chat message and return updated history."""
        if retriever is None:  # ← Check if FAISS index failed to load
            history.append({"role": "user", "content": message})
            history.append(
                {"role": "assistant", "content": "Error: FAISS index not found."}
            )
            return "", history, gr.update(interactive=True), gr.update(interactive=True)

        # Add user message to history
        history.append({"role": "user", "content": message})

        # Retrieve context
        chunks = retriever.query(message, k=3)
        context = [c["document"] for c in chunks]

        # Build RAG prompt
        prompt = retriever.build_prompt(message, context) if context else message

        # Generate response
        try:
            result = generate(prompt, model_id=AVAILABLE_MODELS[model_name])
            response = f"{result.text}\n\n---\n💰 ${result.cost_usd:.6f} | ⏱️ {result.latency_sec:.2f}s | 🎯 {model_name}"
            history.append({"role": "assistant", "content": response})
            return "", history, gr.update(interactive=True), gr.update(interactive=True)
        except Exception as e:
            history.append({"role": "assistant", "content": f"Error: {e}"})
            return "", history, gr.update(interactive=True), gr.update(interactive=True)

    def compare_models(message: str) -> str:
        """Run query through all models and show comparison."""
        if retriever is None:
            return "Error: FAISS index not found."

        chunks = retriever.query(message, k=3)
        context = [c["document"] for c in chunks]
        prompt = retriever.build_prompt(message, context) if context else message

        results = []
        for name, model_id in AVAILABLE_MODELS.items():
            try:
                result = generate(prompt, model_id=model_id, system=SYSTEM_PROMPT)
                results.append(
                    f"## {name}\n{result.text}\n\n*{result.latency_sec:.2f}s | ${result.cost_usd:.6f}*\n---\n"
                )
            except Exception as e:
                results.append(f"## {name}\nError: {e}\n---\n")

        return "\n".join(results)

    def run_comparison_eval(
        eval_questions: str,
        selected_models: list[str],
        include_baseline: bool = False,
        demo_mode: bool = False,
    ) -> tuple[str, pd.DataFrame, Any, Any, Any]:
        """Run evaluation and return results with visualizations.

        Args:
            eval_questions: Questions to evaluate (one per line)
            selected_models: List of model names to compare
            include_baseline: If True, also run without RAG context for comparison
            demo_mode: If True, limit to 3 fastest models to prevent API throttling
        """
        if retriever is None:
            return (
                "Error: FAISS index not found.",
                pd.DataFrame(),
                None,
                None,
                None,
                None,
                None,
            )

        # Parse questions (one per line)
        questions = [q.strip() for q in eval_questions.strip().split("\n") if q.strip()]
        if not questions:
            return (
                "No questions provided.",
                pd.DataFrame(),
                None,
                None,
                None,
                None,
                None,
            )

        # Initialize embedder for faithfulness scoring
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Get model IDs
        model_ids = {
            name: AVAILABLE_MODELS[name]
            for name in selected_models
            if name in AVAILABLE_MODELS
        }
        if not model_ids:
            return "No valid models selected.", pd.DataFrame(), None, None, None

        # Demo mode: strictly limit to 3 fastest models to prevent throttling
        if demo_mode and len(model_ids) > 3:
            # Prioritize faster/cheaper models (order matters: first = highest priority)
            priority_models = [
                "Claude Haiku",
                "Llama 3.1 8B",
                "Llama 3.3 70B",
                "Nova Pro",
                "Llama 4 Maverick",
                "Llama 3.1 70B",
                "DeepSeek R1",
                "Mistral 7B",
                "Mistral Large",
            ]
            filtered = []
            for name in priority_models:
                if name in model_ids:
                    filtered.append(name)
                if len(filtered) >= 3:
                    break
            # Strictly enforce 3 model limit - truncate any excess
            model_ids = {name: model_ids[name] for name in filtered[:3]}

        results = []
        unavailable_models = set()

        for q_idx, question in enumerate(questions, 1):
            # Retrieve context for RAG
            chunks = retriever.query(question, k=3)
            context = [c["document"] for c in chunks]
            rag_prompt = (
                retriever.build_prompt(question, context) if context else question
            )

            # Join context chunks for metric calculations
            context_text = "\n\n".join(context)

            for model_name, model_id in model_ids.items():
                if model_name in unavailable_models:
                    continue

                print(f"[{q_idx}/{len(questions)}] Processing {model_name}...")

                # Run BASELINE (no RAG) if requested
                if include_baseline:
                    try:
                        result = generate(
                            question, model_id=model_id, system=SYSTEM_PROMPT
                        )

                        # Calculate metrics for baseline
                        answer_quality = calculate_context_recall(
                            result.text, context_text, embedder
                        )  # How well answer aligns with docs

                        results.append(
                            {
                                "question": question,
                                "model": f"{model_name} (Baseline)",
                                "mode": "Baseline (No RAG)",
                                "response": result.text,
                                "faithfulness": 0.0,  # Baseline doesn't use retrieved context (RAG-specific metric)
                                "answer_quality": answer_quality,  # How well answer matches official docs
                                "context_recall": 0.0,  # N/A for baseline (no retrieval)
                                "latency_sec": result.latency_sec,
                                "cost_usd": result.cost_usd,
                                "tokens_in": result.tokens_in,
                                "tokens_out": result.tokens_out,
                            }
                        )
                    except Exception as e:
                        results.append(
                            {
                                "question": question,
                                "model": f"{model_name} (Baseline)",
                                "mode": "Baseline (No RAG)",
                                "response": f"ERROR: {e}",
                                "faithfulness": 0.0,
                                "answer_quality": 0.0,
                                "context_recall": 0.0,
                                "latency_sec": 0,
                                "cost_usd": 0,
                                "tokens_in": 0,
                                "tokens_out": 0,
                            }
                        )

                # Run RAG
                try:
                    result = generate(
                        rag_prompt, model_id=model_id, system=SYSTEM_PROMPT
                    )

                    # Calculate RAG metric: answer quality (alignment with documentation)
                    answer_quality = calculate_context_recall(
                        result.text, context_text, embedder
                    )

                    results.append(
                        {
                            "question": question,
                            "model": model_name,
                            "mode": "RAG",
                            "response": result.text,
                            "faithfulness": answer_quality,  # How well answer is grounded in context
                            "answer_quality": answer_quality,  # How well answer matches docs (Documentation Alignment)
                            "latency_sec": result.latency_sec,
                            "cost_usd": result.cost_usd,
                            "tokens_in": result.tokens_in,
                            "tokens_out": result.tokens_out,
                        }
                    )
                except Exception as e:
                    error_msg = str(e)
                    if (
                        "inference profile" in error_msg.lower()
                        or "throughput" in error_msg.lower()
                    ):
                        unavailable_models.add(model_name)
                        continue
                    results.append(
                        {
                            "question": question,
                            "model": model_name,
                            "mode": "RAG",
                            "response": f"ERROR: {e}",
                            "faithfulness": 0.0,
                            "answer_quality": 0.0,
                            "latency_sec": 0,
                            "cost_usd": 0,
                            "tokens_in": 0,
                            "tokens_out": 0,
                        }
                    )

        df = pd.DataFrame(results)

        # Create summary
        summary_text = (
            f"Evaluated {len(questions)} questions across {len(model_ids)} models.\n"
        )
        if unavailable_models:
            summary_text += f"Skipped (unavailable): {', '.join(unavailable_models)}\n"

        # Extract base model name (without "(Baseline)" suffix) for color grouping
        df["base_model"] = df["model"].apply(
            lambda x: x.replace(" (Baseline)", "") if "(Baseline)" in x else x
        )
        df["is_baseline"] = df["model"].apply(
            lambda x: "Baseline (No RAG)" if "(Baseline)" in x else "RAG"
        )

        # Aggregate by model - focus on key metrics
        summary = (
            df.groupby(["base_model", "is_baseline"])
            .agg(
                {
                    "latency_sec": "mean",
                    "cost_usd": "sum",
                    "tokens_in": "sum",
                    "tokens_out": "sum",
                    "faithfulness": "mean",  # Groundedness
                    "answer_quality": "mean",  # Documentation alignment
                }
            )
            .reset_index()
        )

        # Add base_model to summary for color grouping
        summary["base_model"] = summary["base_model"]
        summary["is_baseline"] = summary["is_baseline"]

        # Calculate cost efficiency (cost per 1000 tokens)
        summary["total_tokens"] = summary["tokens_in"] + summary["tokens_out"]
        summary["cost_per_1k_tokens"] = summary.apply(
            lambda row: (
                (row["cost_usd"] / row["total_tokens"] * 1000)
                if row["total_tokens"] > 0
                else 0
            ),
            axis=1,
        )

        # Chart 1: Cost vs Latency scatter with faithfulness as size
        # Create a simplified label combining model and faithfulness for hover
        summary["hover_label"] = summary.apply(
            lambda row: (
                f"{row['base_model']}<br>{row['is_baseline']}<br>Faithfulness: {row['faithfulness']:.2f}"
            ),
            axis=1,
        )

        # Create 3D scatter plot with Faithfulness as Z-axis for better visualization
        fig1 = px.scatter_3d(
            summary,
            x="cost_usd",
            y="latency_sec",
            z="faithfulness",  # Faithfulness as 3rd dimension
            color="base_model",
            symbol="is_baseline",
            size_max=10,  # Smaller markers for 3D
            title="Cost vs Latency vs Faithfulness (3D)<br><sub>Rotate to explore: RAG models (circles) vs Baseline (triangles). Higher Z-axis = Better grounding</sub>",
            labels={
                "cost_usd": "Cost (USD)",
                "latency_sec": "Latency (s)",
                "faithfulness": "Faithfulness",
                "base_model": "",
                "is_baseline": "Mode",
            },
            color_discrete_sequence=px.colors.qualitative.Set1,
            category_orders={"is_baseline": ["RAG", "Baseline"]},
            hover_data=["tokens_in", "tokens_out"],
        )
        # Configure 3D layout with legend outside
        fig1.update_layout(
            scene=dict(
                xaxis_title="Cost ($)",
                yaxis_title="Latency (s)",
                zaxis_title="Faithfulness",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)  # Good viewing angle
                ),
            ),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.02,  # Move legend outside to right
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="lightgray",
                borderwidth=1,
                font=dict(size=9),  # Smaller font to fit more entries
            ),
            title_font_size=14,
            height=600,  # Taller for 3D view
            margin=dict(r=180),  # Right margin for legend
        )

        # Create winner summary table
        rag_summary = summary[summary["is_baseline"] == "RAG"]
        if len(rag_summary) > 0:
            best_cost = rag_summary.loc[rag_summary["cost_usd"].idxmin()]
            best_latency = rag_summary.loc[rag_summary["latency_sec"].idxmin()]
            best_faithfulness = rag_summary.loc[rag_summary["faithfulness"].idxmax()]

            winners_df = pd.DataFrame(
                [
                    {
                        "Metric": "💰 Lowest Cost",
                        "Winner": best_cost["base_model"],
                        "Value": f"${best_cost['cost_usd']:.4f}",
                    },
                    {
                        "Metric": "⚡ Fastest",
                        "Winner": best_latency["base_model"],
                        "Value": f"{best_latency['latency_sec']:.2f}s",
                    },
                    {
                        "Metric": "🎯 Best Faithfulness",
                        "Winner": best_faithfulness["base_model"],
                        "Value": f"{best_faithfulness['faithfulness']:.2f}",
                    },
                ]
            )
        else:
            winners_df = pd.DataFrame()

        # Chart 2: Token Usage - Compare RAG vs Baseline
        token_data = []
        for _, row in summary.iterrows():
            mode = row["is_baseline"]
            token_data.append(
                {
                    "model": row["base_model"],
                    "mode": mode,
                    "type": "Input",
                    "count": row["tokens_in"],
                }
            )
            token_data.append(
                {
                    "model": row["base_model"],
                    "mode": mode,
                    "type": "Output",
                    "count": row["tokens_out"],
                }
            )
        token_df = pd.DataFrame(token_data)

        # Create combined category for better grouping
        token_df["category"] = token_df["type"] + " - " + token_df["mode"]

        fig2 = px.bar(
            token_df,
            x="model",
            y="count",
            color="category",
            title="Token Usage: RAG vs Baseline<br><sub>💡 RAG: Higher input (context) but lower output (concise vs verbose)</sub>",
            labels={"count": "Token Count", "model": "", "category": ""},
            color_discrete_map={
                "Input - RAG": "#1a5f7a",
                "Output - RAG": "#2E86AB",
                "Input - Baseline": "#7d3c5d",
                "Output - Baseline": "#A23B72",
            },
            barmode="group",
        )
        fig2.update_layout(
            legend=dict(
                orientation="h", yanchor="top", y=-0.4, xanchor="center", x=0.5
            ),
            xaxis_tickangle=-45,
            margin=dict(b=180, t=110),
            height=550,
        )

        # Chart 3: Answer Quality - Fair Comparison Metric
        if include_baseline:
            # Calculate per-question improvements (averaged across all models for each question)
            import statistics

            all_improvements = []
            for question in questions:
                q_df = df[df["question"] == question]
                rag_q = q_df[q_df["mode"] == "RAG"]["answer_quality"].mean()
                baseline_q = q_df[q_df["mode"] == "Baseline (No RAG)"][
                    "answer_quality"
                ].mean()
                if baseline_q > 0:
                    improvement = (rag_q - baseline_q) / baseline_q * 100
                    all_improvements.append(improvement)

            if all_improvements:
                # Show ALL improvements (positive and negative) for transparency
                median_improvement = statistics.median(all_improvements)
                improvements_str = ", ".join(
                    [f"{imp:+.0f}%" for imp in all_improvements]
                )
                num_positive = sum(1 for imp in all_improvements if imp > 0)
                num_total = len(all_improvements)

                # Simplified subtitle for single question, detailed for multiple
                if num_total == 1:
                    subtitle = f"<sub>RAG improvement: {improvements_str} (median across models) | Results may vary ±5-10% between runs due to LLM sampling</sub>"
                else:
                    subtitle = f"<sub>RAG improvements: {improvements_str} (median: {median_improvement:+.0f}%) | {num_positive}/{num_total} questions improved | Results vary due to LLM sampling</sub>"
            else:
                subtitle = "<sub>No improvements detected. Try different questions or models.</sub>"
        else:
            subtitle = "<sub>Measures how well answers align with official Odoo 18 documentation</sub>"

        fig3 = px.bar(
            summary,
            x="base_model",
            y="answer_quality",
            color="is_baseline",
            barmode="group",
            title=f"Answer Quality (Documentation Alignment)<br>{subtitle}",
            labels={
                "answer_quality": "Answer Quality Score (0-1)",
                "base_model": "",
                "is_baseline": "Mode",
            },
            color_discrete_map={"RAG": "#2E86AB", "Baseline": "#A23B72"},
            range_y=[0, 1],
            text_auto=".2f",
        )
        fig3.update_layout(
            legend=dict(
                orientation="h", yanchor="top", y=-0.4, xanchor="center", x=0.5
            ),
            xaxis_tickangle=-45,
            margin=dict(b=180, t=80),
            height=550,
        )
        fig3.update_traces(textposition="outside")

        # Chart 4: Response Latency - RAG vs Baseline
        fig4 = px.bar(
            summary,
            x="base_model",
            y="latency_sec",
            color="is_baseline",
            barmode="group",
            title="Response Latency: RAG vs Baseline<br><sub>Total response time. RAG = Retrieval (~0.5-2s) + LLM generation. Baseline = LLM only.</sub>",
            labels={
                "latency_sec": "Latency (seconds)",
                "base_model": "",
                "is_baseline": "Mode",
            },
            color_discrete_map={"RAG": "#2E86AB", "Baseline": "#A23B72"},
            text_auto=".2f",
        )
        fig4.update_layout(
            legend=dict(
                orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5
            ),
            xaxis_tickangle=-45,
            margin=dict(b=150, t=100),
            height=600,  # Match 3D chart height
        )
        fig4.update_traces(textposition="outside")

        # Filter DataFrame columns for display - focus on business value metrics
        display_columns = [
            "question",
            "model",
            "mode",
            "response",
            "faithfulness",
            "answer_quality",
            "latency_sec",
            "cost_usd",
            "tokens_in",
            "tokens_out",
        ]
        df_display = df[[col for col in display_columns if col in df.columns]]

        return summary_text, df_display, fig1, fig2, fig3, fig4, winners_df

    with gr.Blocks(title="Odoo AI Assistant - Model Performance Analyzer") as app:
        gr.Markdown("# 🤖 Odoo AI Assistant - Model Performance Analyzer")

        with gr.Tabs():
            # Tab 1: Chat
            with gr.Tab("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_selector = gr.Dropdown(
                            choices=list(AVAILABLE_MODELS.keys()),
                            value=list(AVAILABLE_MODELS.keys())[0],
                            label="Model",
                        )
                        gr.Markdown("**Available Models**")
                        gr.Markdown("""- **Claude Haiku**: Fast & efficient
- **Llama 3.1 8B**: Balanced performance
- **Llama 3.1 70B**: High quality
- **Llama 3.3 70B**: Latest Llama
- **Llama 4 Maverick**: Newest Llama
- **Mistral 7B**: Efficient mid-size
- **Mistral Large**: High quality
- **DeepSeek R1**: Reasoning model
- **Nova Pro**: Amazon's best""")

                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(height=500)
                        msg_input = gr.Textbox(
                            placeholder="Ask about Odoo 18...",
                            label="Question",
                        )
                        with gr.Row():
                            send_btn = gr.Button("💬 Send", variant="primary")
                            compare_btn = gr.Button(
                                "🔍 Compare All Models", variant="secondary"
                            )
                        gr.Markdown(
                            "⚠️ **Note**: Comparing all models may take 30-60 seconds depending on API response times."
                        )
                        compare_output = gr.Markdown(visible=False)

            # Tab 2: Model Comparison
            with gr.Tab("📊 Model Comparison"):
                gr.Markdown("""Run questions through multiple models and compare cost/latency.
                
                ⚠️ **Note**: Comparison duration depends on the number of selected models, questions, and whether baseline comparison is enabled. 
                Each model call requires a separate AWS Bedrock API request. More models = longer wait time.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        eval_models = gr.CheckboxGroup(
                            choices=list(AVAILABLE_MODELS.keys()),
                            value=["Llama 3.1 8B", "Claude Haiku"],
                            label="Select Models to Compare",
                        )
                        demo_mode = gr.Checkbox(
                            label="Demo Mode (Limit to 3 models)",
                            value=True,
                            info="Limit to 3 fastest models to prevent API throttling during demos with multiple users",
                        )

                        # Add warning that appears when more than 3 models selected in demo mode
                        demo_warning = gr.Markdown(
                            value="",
                            visible=False,
                        )
                        include_baseline = gr.Checkbox(
                            label="Compare with Baseline (No RAG)",
                            value=False,
                            info="Run models twice: once without context (baseline) and once with RAG. Demonstrates how RAG grounds answers in official docs while baseline may make things up.",
                        )
                        run_eval_btn = gr.Button("Run Comparison", variant="primary")

                    with gr.Column(scale=2):
                        eval_questions = gr.Textbox(
                            value="How do I use the grouped() method to partition recordsets in Odoo 18 without read_group overhead?",
                            label="Evaluation Question",
                            lines=2,
                            placeholder="Enter question...",
                            info="💡 Best practice: Use one question at a time for clearest comparison. One question has higher probability of getting consistent metrics.\n\nTo expand knowledge base: add sources to data/sources.json → run 'make pipeline-data'.",
                        )

                eval_status = gr.Textbox(label="Status", interactive=False)

                # Winners summary table
                with gr.Row():
                    winners_table = gr.DataFrame(
                        label="🏆 Best Performers (RAG Models Only)",
                        headers=["Metric", "Winner", "Value"],
                        interactive=False,
                    )

                with gr.Row():
                    with gr.Column():
                        chart1 = gr.Plot(label="Cost vs Latency vs Faithfulness (3D)")
                    with gr.Column():
                        chart2 = gr.Plot(label="Token Usage")

                with gr.Row():
                    with gr.Column():
                        chart3 = gr.Plot(
                            label="Faithfulness: RAG Grounds Answers in Official Docs"
                        )
                    with gr.Column():
                        chart4 = gr.Plot(label="Response Latency")

                with gr.Row():
                    results_table = gr.DataFrame(label="Detailed Results")

                with gr.Row():
                    export_btn = gr.Button("📥 Export Results to CSV", size="sm")
                    export_file = gr.File(label="Download CSV", visible=False)

        # Event handlers for Chat tab
        def chat_with_button(msg, model, history):
            """Chat wrapper that disables both buttons during processing."""
            # Disable both buttons
            yield None, None, gr.update(interactive=False), gr.update(interactive=False)

            # Process chat
            for result in chat(msg, model, history):
                yield result

            # Re-enable both buttons
            yield (
                result[0],
                result[1],
                gr.update(interactive=True),
                gr.update(interactive=True),
            )

        def show_comparison(msg):
            return {compare_output: gr.update(value=compare_models(msg), visible=True)}

        def compare_with_button(msg):
            """Compare wrapper that disables both buttons during processing."""
            # Disable both buttons
            yield (
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(value="⏳ Comparing models...", visible=True),
            )

            # Run comparison
            result = show_comparison(msg)

            # Re-enable both buttons
            yield gr.update(interactive=True), gr.update(interactive=True), result

        send_btn.click(
            fn=lambda: gr.update(interactive=False),
            outputs=[msg_input],
        ).then(
            fn=chat,
            inputs=[msg_input, model_selector, chatbot],
            outputs=[msg_input, chatbot, send_btn, compare_btn],
        ).then(
            lambda: (
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
            ),
            None,
            [send_btn, compare_btn, msg_input],
        )
        msg_input.submit(
            fn=lambda: gr.update(interactive=False),
            outputs=[msg_input],
        ).then(
            fn=chat,
            inputs=[msg_input, model_selector, chatbot],
            outputs=[msg_input, chatbot, send_btn, compare_btn],
        ).then(
            lambda: (
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
            ),
            None,
            [send_btn, compare_btn, msg_input],
        )
        compare_btn.click(
            fn=lambda: (
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            ),
            outputs=[send_btn, compare_btn, msg_input],
        ).then(
            fn=show_comparison,
            inputs=[msg_input],
            outputs=[compare_output],
        ).then(
            fn=lambda: (
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True, value=""),
            ),
            outputs=[send_btn, compare_btn, msg_input],
        )

        # Export function
        def export_to_csv(df):
            """Export DataFrame to CSV file and show download link."""
            if df is None or len(df) == 0:
                return gr.update(visible=False)

            import tempfile
            from datetime import datetime

            # Create temp file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_file = tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=f"_rag_eval_{timestamp}.csv"
            )
            df.to_csv(temp_file.name, index=False)
            temp_file.close()

            # Return file with visible=True
            return gr.update(value=temp_file.name, visible=True)

        # Event handlers for Comparison tab
        def run_comparison_with_button(
            eval_questions, selected_models, demo_mode, include_baseline
        ):
            """Wrapper to disable button during comparison."""
            # This generator yields: (status, df, chart1, chart2, chart3, chart4, winners_df, button_state)
            # First yield disables button
            yield (
                "Running comparison...",
                pd.DataFrame(),
                None,
                None,
                None,
                None,
                pd.DataFrame(),
                gr.update(interactive=False),
            )

            # Run actual comparison
            result = run_comparison_eval(
                eval_questions, selected_models, include_baseline, demo_mode
            )

            # Final result with button re-enabled
            yield (*result, gr.update(interactive=True))

        run_eval_btn.click(
            fn=run_comparison_with_button,
            inputs=[eval_questions, eval_models, demo_mode, include_baseline],
            outputs=[
                eval_status,
                results_table,
                chart1,
                chart2,
                chart3,
                chart4,
                winners_table,
                run_eval_btn,
            ],
        )

        # Demo mode warning handler
        def update_demo_warning(selected_models, demo_mode):
            if demo_mode and len(selected_models) > 3:
                return gr.update(
                    value="⚠️ **Demo Mode ON**: Only the first 3 fastest models will be used",
                    visible=True,
                )
            return gr.update(value="", visible=False)

        eval_models.change(
            fn=update_demo_warning,
            inputs=[eval_models, demo_mode],
            outputs=[demo_warning],
        )
        demo_mode.change(
            fn=update_demo_warning,
            inputs=[eval_models, demo_mode],
            outputs=[demo_warning],
        )

        # Export button handler
        export_btn.click(
            fn=export_to_csv,
            inputs=[results_table],
            outputs=[export_file],
        )

    return app


def main():
    """Main entry point for running Gradio app.

    Deployment modes:
    - Local: python -m odoo_rag.app (uses local data/faiss_index/)
    - ECS: Runs in Docker container on AWS Fargate (downloads from S3)

    Server configuration:
    - Binds to 0.0.0.0:7860 (accessible from outside container)
    - No Gradio share link (not needed for AWS deployment)
    - Shows errors in UI (helpful for debugging)
    """
    # Create Gradio app with all tabs (chat, compare, evaluation)
    app = create_app()

    # Launch web server
    # server_name="0.0.0.0" allows external access (required for Docker/ECS)
    # server_port=7860 is standard Gradio port (exposed in Dockerfile)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # No public Gradio link (use ALB URL instead)
        show_error=True,  # Display errors in UI for debugging
    )


# Entry point when running: python -m odoo_rag.app
if __name__ == "__main__":
    main()
