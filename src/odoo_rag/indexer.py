"""Build FAISS index from raw documentation.

FAISS = Facebook AI Similarity Search (vector database for fast similarity search)

This module converts raw text files into a searchable vector database:
1. Split text into 500-word chunks (with 50-word overlap)
2. Convert each chunk to a 384-dimensional vector (embedding)
3. Store vectors in FAISS index for fast search
4. Store original text in corpus.json for retrieval

Output:
- data/faiss_index/index.faiss (vector database)
- data/faiss_index/corpus.json (original text)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Suppress HuggingFace Hub warnings about unauthenticated requests
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import faiss  # Vector similarity search library
from sentence_transformers import SentenceTransformer  # Text → vector conversion


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split long text into smaller overlapping chunks.

    Chunking strategy: Word-based with overlap
    - Each chunk: ~500 words (~3000 characters)
    - Overlap: 50 words between consecutive chunks
    - Overlap ensures context continuity across chunk boundaries

    Args:
        text: Raw documentation text (can be very long)
        chunk_size: Number of words per chunk (default: 500)
        overlap: Number of words to overlap between chunks (default: 50)

    Returns:
        List of text chunks, each ~500 words

    Example:
        Input: "word1 word2 word3 ... word1000"
        Output: [
            "word1 word2 ... word500",      # Chunk 0: words 1-500
            "word451 word452 ... word950",  # Chunk 1: words 451-950 (50-word overlap)
            "word901 word902 ... word1000"  # Chunk 2: words 901-1000
        ]
    """
    # ========== ACTIVE: Simple word-based chunking ==========
    # Split text into words (separated by spaces)
    words = text.split()
    chunks = []

    # Create overlapping chunks by sliding window
    # Step size = chunk_size - overlap (e.g., 500 - 50 = 450)
    # This means each chunk shares 50 words with the previous chunk
    for i in range(0, len(words), chunk_size - overlap):
        # Extract chunk_size words starting at position i
        chunk = " ".join(words[i : i + chunk_size])

        # Skip very small chunks (< 50 characters)
        if len(chunk) > 50:
            chunks.append(chunk)

    return chunks

    # ========== DISABLED: Semantic Chunking ==========
    # Attempted to split by headers/sections instead of word count
    # Result: Created 3x more small fragments vs word-based chunking
    # Problem: Too much fragmentation → retrieval found incomplete context
    # Conclusion: Simple word-based chunking works better for technical docs
    #
    # import re
    #
    # # Try semantic chunking first (split by headers)
    # header_pattern = r'\n(#{2,3}\s+.+|[A-Z][^.!?]*:|\d+\.\s+[A-Z][^.!?]*)\n'
    # sections = re.split(header_pattern, text)
    #
    # chunks = []
    # current_header = ""
    #
    # if len(sections) > 2:
    #     for i, section in enumerate(sections):
    #         section = section.strip()
    #         if not section:
    #             continue
    #
    #         if (section.startswith('#') or
    #             section.endswith(':') or
    #             (len(section.split()) < 10 and section[0].isupper())):
    #             current_header = section
    #         else:
    #             chunk_text = f"{current_header}\n\n{section}" if current_header else section
    #
    #             if len(chunk_text.split()) > chunk_size:
    #                 paragraphs = chunk_text.split('\n\n')
    #                 for para in paragraphs:
    #                     para = para.strip()
    #                     if len(para.split()) > 30:
    #                         if len(para.split()) > chunk_size:
    #                             sentences = re.split(r'(?<=[.!?])\s+', para)
    #                             current_chunk = current_header + "\n\n" if current_header else ""
    #                             for sent in sentences:
    #                                 if len((current_chunk + sent).split()) > chunk_size:
    #                                     if current_chunk.strip():
    #                                         chunks.append(current_chunk.strip())
    #                                     current_chunk = (current_header + "\n\n" if current_header else "") + sent + " "
    #                                 else:
    #                                     current_chunk += sent + " "
    #                             if current_chunk.strip():
    #                                 chunks.append(current_chunk.strip())
    #                         else:
    #                             chunks.append(para)
    #             else:
    #                 if chunk_text.strip():
    #                     chunks.append(chunk_text.strip())
    # else:
    #     words = text.split()
    #     for i in range(0, len(words), chunk_size - overlap):
    #         chunk = " ".join(words[i:i + chunk_size])
    #         if len(chunk.split()) > 30:
    #             chunks.append(chunk)
    #
    # return chunks


def build_index(
    raw_dir: str = "data/raw", output_dir: str = "data/faiss_index"
) -> None:
    """Build FAISS vector index from raw documentation files.

    Process:
    1. Read all .txt files from data/raw/
    2. Split each file into 500-word chunks
    3. Convert chunks to 384-dimensional vectors (embeddings)
    4. Build FAISS index for fast similarity search
    5. Save index.faiss (vectors) and corpus.json (text)

    Args:
        raw_dir: Directory containing raw text files (default: data/raw)
        output_dir: Directory to save index files (default: data/faiss_index)

    Output files:
        - index.faiss: Vector database (binary file, ~1-2 MB)
        - corpus.json: Original text + metadata (JSON file, ~2-3 MB)
    """
    # Create output directory if needed
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ========== STEP 1: Load and chunk all documentation files ==========
    all_chunks = []  # Store all text chunks
    sources = []  # Track which file each chunk came from

    raw_path = Path(raw_dir)
    if not raw_path.exists():
        print(f"Error: {raw_dir} not found. Run `make ingest` first.")
        return

    # Process each .txt file in data/raw/
    for doc_file in raw_path.glob("*.txt"):
        print(f"Processing: {doc_file.name}")

        # Read entire file content
        text = doc_file.read_text()

        # Split into 500-word chunks with 50-word overlap
        chunks = chunk_text(text)

        # Add chunks to collection, tracking source file
        for chunk in chunks:
            all_chunks.append(chunk)
            sources.append(doc_file.stem)  # e.g., "odoo_orm_api_reference"

    print(f"Total chunks: {len(all_chunks)}")

    # ========== STEP 2: Convert text chunks to vectors (embeddings) ==========
    print("Embedding...")

    # Load pre-trained model: all-MiniLM-L6-v2
    # This model converts text → 384-dimensional vectors
    # Model size: ~90 MB, downloaded on first run
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Convert all chunks to vectors
    # Input: List of text strings (all chunks from documentation)
    # Output: Array of vectors (N chunks × 384 dimensions)
    embeddings = embedder.encode(all_chunks, show_progress_bar=True)

    # ========== STEP 3: Normalize vectors for cosine similarity ==========
    # Convert to 32-bit floats (saves memory vs 64-bit)
    # float64 (default): 8 bytes per number → N chunks × 384 dims × 8 bytes
    # float32: 4 bytes per number → N chunks × 384 dims × 4 bytes
    # Result: 50% memory savings with negligible accuracy loss for similarity search
    #
    # Why this matters:
    # - FAISS requires float32 (will error with float64)
    # - For small datasets (< 1M vectors), memory difference is minimal
    # - For large datasets (> 1M vectors), this saves GBs of RAM
    # - GPU operations are faster with float32
    embeddings = embeddings.astype("float32")

    # Normalize vectors to unit length (L2 normalization)
    #
    # Analogy: Resize all arrows to same length while keeping their direction
    # - Before: arrows of different lengths pointing in different directions
    # - After: all arrows have length = 1.0, but still point in same directions
    #
    # IMPORTANT: "Length" = sqrt(sum of squares), NOT sum of values!
    # Length formula: sqrt(v[0]² + v[1]² + v[2]² + ... + v[383]²)
    #
    # Example with 2D vector:
    # Before: [3, 4]
    #   - Sum: 3 + 4 = 7
    #   - Length: sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5.0
    # After: [0.6, 0.8]  (divide each number by 5.0)
    #   - Sum: 0.6 + 0.8 = 1.4 (NOT 1.0!)
    #   - Length: sqrt(0.6² + 0.8²) = sqrt(0.36 + 0.64) = sqrt(1.0) = 1.0 ✓
    #
    # Why normalize?
    # - Similarity search cares about DIRECTION (meaning), not LENGTH (magnitude)
    # - Example: "cat" and "kitten" should be similar regardless of text length
    #
    # Math shortcut using dot product:
    # Dot product = multiply corresponding elements and sum them up
    # Example: dot([3, 4], [5, 6]) = (3×5) + (4×6) = 15 + 24 = 39
    #
    # For 384-dim vectors:
    # dot(A, B) = (A[0]×B[0]) + (A[1]×B[1]) + ... + (A[383]×B[383])
    #
    # Cosine similarity formula:
    # - Normal: dot(A, B) / (length(A) × length(B))  ← needs division
    # - After normalization: dot(A, B) / (1.0 × 1.0) = dot(A, B)  ← no division!
    # - Result: Just compute dot product (faster!)
    faiss.normalize_L2(embeddings)

    # ========== STEP 4: Build FAISS index ==========
    # Get vector dimension (384 for all-MiniLM-L6-v2)
    dim = embeddings.shape[1]

    # Create FAISS index: IndexFlatIP = Flat + Inner Product
    #
    # Breaking down "IndexFlatIP":
    # - Index = searchable database structure
    # - Flat = exhaustive/brute-force search (checks ALL vectors)
    # - IP = Inner Product (uses dot product to measure similarity)
    #
    # "Flat" search strategy:
    # - Compares query against EVERY stored vector
    # - Pros: 100% accurate, finds true best matches
    # - Cons: Slower for large datasets (millions of vectors)
    # - For small datasets (< 10K vectors): < 1 millisecond (perfectly fine!)
    #
    # Alternative strategies (not used here):
    # - IndexIVF = approximate search (faster but less accurate)
    # - IndexHNSW = graph-based search (faster but uses more memory)
    #
    # "IP" similarity metric:
    # - Uses dot product: dot(query, chunk) = higher score = more similar
    # - Works because vectors are normalized (length = 1.0)
    # - Range: -1.0 (opposite) to +1.0 (identical)
    #
    # Alternative: L2 distance (Euclidean distance)
    # - Measures straight-line distance between two points
    # - Formula: sqrt((A[0]-B[0])² + (A[1]-B[1])² + ... + (A[383]-B[383])²)
    # - Lower distance = more similar (opposite of dot product!)
    # - Used for: images, spatial data, non-normalized vectors
    # - Not ideal for text: sensitive to vector magnitude
    #
    # Why IP (dot product) for text embeddings?
    # - Standard in NLP (Natural Language Processing)
    # - Measures angle between vectors (direction = meaning)
    # - Ignores magnitude after normalization
    # - Faster to compute (no square root needed)
    #
    # Example comparison:
    # Vector A: [0.6, 0.8]  Vector B: [0.6, 0.8]
    # - IP: dot(A,B) = 1.0 (identical!)
    # - L2: distance = 0.0 (identical!)
    # Both work, but IP is faster and standard for text
    index = faiss.IndexFlatIP(dim)

    # Add all vectors to the index
    # Each vector gets an ID (0, 1, 2, ..., N-1)
    #
    # IMPORTANT: FAISS stores the VECTORS themselves, NOT pre-computed dot products!
    # Storage: N vectors × 384 dimensions × 4 bytes per float32
    #
    # How search works:
    # 1. User asks question: "How to filter records?"
    # 2. Convert question to vector: [0.24, -0.44, 0.68, ..., 0.13]
    # 3. FAISS computes dot product with ALL stored vectors (on-the-fly)
    # 4. Returns top-k chunk IDs with highest dot products
    index.add(embeddings)

    # ========== STEP 5: Save both files ==========
    # Save vector database (binary format, ~1.2 MB)
    faiss.write_index(index, str(output_path / "index.faiss"))

    # Save original text with metadata (JSON format, ~2.4 MB)
    # Format: [{"text": "...", "source": "odoo_orm"}, ...]
    # Array index = chunk ID (matches FAISS index)
    corpus = [{"text": t, "source": s} for t, s in zip(all_chunks, sources)]
    with open(output_path / "corpus.json", "w") as f:
        json.dump(corpus, f)

    print(f"Saved index: {len(all_chunks)} documents, dim={dim}")


def main():
    """Entry point: Build FAISS index from raw documentation.

    Workflow:
    1. Read all .txt files from data/raw/
    2. Split into 500-word chunks (50-word overlap)
    3. Convert chunks to 384-dimensional vectors
    4. Build FAISS index for similarity search
    5. Save index.faiss (vectors) + corpus.json (text)

    Output:
    - index.faiss: Vector database for fast search
    - corpus.json: Original text for retrieval

    Next step: Run app.py to start RAG chatbot
    """
    build_index()
    print("Done. Run `make run` to start the app.")


if __name__ == "__main__":
    main()
