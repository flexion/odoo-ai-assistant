"""FAISS retriever for Odoo documentation.

This module loads the pre-built FAISS index and corpus, then performs semantic search
to find the most relevant documentation chunks for a given question.

Flow:
1. Load index.faiss (all vectors) and corpus.json (all text chunks) into memory
2. Convert user question to 384-dimensional vector
3. Search FAISS index using dot product similarity
4. Return top-k most relevant chunks with scores
5. Build prompt with retrieved context for LLM
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Suppress HuggingFace Hub warnings about unauthenticated requests
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faiss
from sentence_transformers import SentenceTransformer


class Retriever:
    """FAISS-based document retriever.

    Loads pre-built FAISS index and corpus into memory, then performs semantic search
    to find relevant documentation chunks for user questions.
    """

    def __init__(self, index_dir: str | Path | None = None):
        """Initialize retriever by loading FAISS index, corpus, and embedding model.

        Args:
            index_dir: Path to directory containing index.faiss and corpus.json
                      Defaults to FAISS_INDEX_PATH env var or 'data/faiss_index'
        """
        # ========== STEP 1: Determine index directory path ==========
        # Use environment variable if set (for AWS Lambda/ECS deployment)
        # Otherwise default to local development path
        if index_dir is None:
            index_dir = os.environ.get("FAISS_INDEX_PATH", "data/faiss_index")
        index_dir = Path(index_dir)

        # ========== STEP 2: Load FAISS index into memory ==========
        # index.faiss contains all normalized vectors (384 dimensions each)
        # File size varies based on number of documentation chunks indexed
        # Format: binary FAISS IndexFlatIP (Inner Product index)
        self.index = faiss.read_index(str(index_dir / "index.faiss"))

        # ========== STEP 3: Load corpus (original text chunks) ==========
        # corpus.json contains all text chunks with metadata
        # File size varies based on amount of documentation scraped
        # Format: [{"text": "...", "source": "odoo_orm"}, ...]
        # Array index matches FAISS vector ID (corpus[0] = vector 0)
        with open(index_dir / "corpus.json") as f:
            self.corpus = json.load(f)

        # ========== STEP 4: Load embedding model ==========
        # Same model used in indexer.py to ensure compatibility
        # Model: all-MiniLM-L6-v2 (384-dimensional embeddings)
        # This model converts text → vectors for similarity search
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Store vector dimension (384) and default top-k results
        self.dimension = self.index.d
        self.k = 5  # Default: return top 5 most relevant chunks

    def query(self, question: str, k: int | None = None) -> list[dict[str, Any]]:
        """Retrieve top-k most relevant documentation chunks for a question.

        Process:
        1. Convert question to 384-dimensional vector
        2. Normalize vector to unit length (same as indexing)
        3. Compute dot product with all stored vectors
        4. Return top-k chunks with highest similarity scores

        Args:
            question: User's question (e.g., "How to filter records?")
            k: Number of results to return (default: 5)

        Returns:
            List of dicts with 'document' (text), 'source', and 'score' (0.0-1.0)
        """
        k = k or self.k

        # ========== STEP 1: Convert question to vector ==========
        # Input: "How to filter records?"
        # Output: numpy array of shape (1, 384) with float64 values
        # show_progress_bar=False: suppress progress output
        query_vec = self.embedder.encode([question], show_progress_bar=False)

        # ========== STEP 2: Convert to float32 ==========
        # FAISS requires float32 format (same as indexing)
        # Converts from float64 → float32 for compatibility
        query_vec = query_vec.astype("float32")

        # ========== STEP 3: Normalize to unit length ==========
        # Same L2 normalization as indexing (length = 1.0)
        # This ensures dot product = cosine similarity
        # After normalization: sqrt(sum of squares) = 1.0
        faiss.normalize_L2(query_vec)

        # ========== STEP 4: Search FAISS index ==========
        # Computes dot product between query_vec and ALL stored vectors
        # Returns top-k results sorted by similarity (highest first)
        #
        # scores: similarity scores (range: -1.0 to 1.0, higher = more similar)
        # indices: chunk IDs (0, 1, 2, ..., N-1, matches corpus array index)
        #
        # Example output:
        # scores = [[0.92, 0.88, 0.85, 0.82, 0.79]]  (top 5 scores)
        # indices = [[1, 3, 12, 45, 67]]              (top 5 chunk IDs)
        scores, indices = self.index.search(query_vec, k)

        # ========== STEP 5: Build results with text and metadata ==========
        results = []
        for score, idx in zip(scores[0], indices[0]):
            # Skip invalid indices (should not happen, but defensive check)
            if idx < 0 or idx >= len(self.corpus):
                continue

            # Fetch original text chunk from corpus using chunk ID
            # corpus[idx] = {"text": "...", "source": "odoo_orm"}
            doc = self.corpus[idx]

            # Build result dictionary with text, source, and similarity score
            results.append(
                {
                    "document": doc.get("text", ""),  # Original text chunk
                    "source": doc.get("source", ""),  # Source file (e.g., "odoo_orm")
                    "score": float(score),  # Similarity score (0.0-1.0)
                }
            )

        return results

    def build_prompt(self, question: str, context_chunks: list[str]) -> str:
        """Build RAG (Retrieval Augmented Generation) prompt with retrieved context.

        Combines retrieved documentation chunks with the user's question into a
        structured prompt for the LLM (Large Language Model).

        Args:
            question: User's question (e.g., "How to filter records?")
            context_chunks: List of retrieved text chunks (top-k results)

        Returns:
            Formatted prompt string ready for LLM

        Example output:
            Use the following Odoo documentation to answer the question.

            Context:
            Use search() method with domains...

            ---

            Domain expressions filter recordsets...

            Question: How to filter records?

            Answer:
        """
        # Join multiple chunks with separator for clarity
        # Separator: \n\n---\n\n (blank line, horizontal rule, blank line)
        context = "\n\n---\n\n".join(context_chunks)

        # Build structured prompt with context and question
        # LLM will use context to generate factual, grounded answer
        return f"""Use the following Odoo documentation to answer the question.

Context:
{context}

Question: {question}

Answer:"""
