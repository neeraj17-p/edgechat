"""
src/embeddings.py
─────────────────
Phase 2: Embedding model wrapper.

Loads the BAAI/bge-small-en-v1.5 sentence embedding model and provides
two methods:
  - embed_chunks()  → embed document text (no prefix needed)
  - embed_query()   → embed a user question (adds BGE query instruction)

Why two separate methods?
  BGE models are trained with a special query instruction prefix ONLY for
  the query side. Chunks are embedded WITHOUT the prefix. Using the wrong
  method (e.g. adding the prefix to chunks) degrades retrieval quality.

Why normalize vectors?
  We use FAISS IndexFlatIP (inner product). On normalized (unit-length)
  vectors, inner product == cosine similarity. This lets us use a simple,
  exact index with no approximation error.

The model is cached after first load — loading from disk takes ~1 second,
which is fast enough to keep in memory for the session.

Offline mode: after the first download, set HF_HUB_OFFLINE=1 in your
shell and the model loads from cache with no network requests.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import config

logger = logging.getLogger(__name__)

# ── Module-level model cache ──────────────────────────────────────────────────
_model_cache: Optional[SentenceTransformer] = None


def _load_model() -> SentenceTransformer:
    """
    Load (or return cached) SentenceTransformer model.

    On first call: downloads the model from HuggingFace Hub (~90 MB) and
    caches it in ~/.cache/huggingface/hub/
    On subsequent calls: loads from local cache instantly.

    If HF_HUB_OFFLINE=1 is set, no network requests are made.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    model_name: str = config.embedding["model_name"]

    # Apply offline mode if configured
    if config.offline.get("hf_hub_offline", False):
        os.environ["HF_HUB_OFFLINE"] = "1"
        logger.info("HF Hub offline mode enabled")

    logger.info("Loading embedding model: %s", model_name)
    logger.info("(First run downloads ~90 MB; subsequent runs use cache)")

    _model_cache = SentenceTransformer(model_name)
    logger.info(
        "Embedding model loaded. Dimension: %d",
        _model_cache.get_sentence_embedding_dimension(),
    )
    return _model_cache


def get_embedding_dim() -> int:
    """Return the vector dimension of the loaded embedding model."""
    return _load_model().get_sentence_embedding_dimension()


def embed_chunks(texts: list[str]) -> np.ndarray:
    """
    Embed a list of document chunk texts into a 2D numpy array.

    IMPORTANT: Do NOT add any prefix to chunk texts. The BGE model is
    trained to embed document text as-is. Adding the query prefix to
    chunks would harm retrieval quality.

    Args:
        texts: List of chunk text strings.

    Returns:
        numpy array of shape (len(texts), embedding_dim), dtype float32.
        Rows are L2-normalized (unit vectors) for cosine similarity.

    Raises:
        ValueError: If texts is empty.
    """
    if not texts:
        raise ValueError("embed_chunks received an empty list of texts.")

    model = _load_model()
    batch_size: int = config.embedding.get("batch_size", 32)
    normalize: bool = config.embedding.get("normalize_vectors", True)

    logger.info("Embedding %d chunks (batch_size=%d) ...", len(texts), batch_size)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    logger.info("Embeddings complete. Shape: %s", embeddings.shape)
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single user query into a 1D numpy array.

    Adds the BGE query instruction prefix before embedding. This is
    required by the bge-small-en-v1.5 model for query-side embedding —
    it was trained with this prefix and retrieval quality drops without it.

    The prefix is read from config.yaml so it can be changed if the
    embedding model is swapped for a different BGE variant.

    Args:
        query: The user's question string.

    Returns:
        numpy array of shape (embedding_dim,), dtype float32, L2-normalized.

    Raises:
        ValueError: If query is empty or whitespace-only.
    """
    query = query.strip()
    if not query:
        raise ValueError("embed_query received an empty query string.")

    model = _load_model()
    normalize: bool = config.embedding.get("normalize_vectors", True)
    instruction: str = config.embedding.get("query_instruction", "")

    # Prepend the BGE query instruction
    prefixed_query = instruction + query

    embedding = model.encode(
        [prefixed_query],
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )

    return embedding[0].astype(np.float32)  # shape: (embedding_dim,)


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick smoke test: load the model and embed a sample sentence.
    Run from project root:
        .\\venv\\Scripts\\python.exe -m src.embeddings
    """
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("Testing embedding model...")
    print(f"Model: {config.embedding['model_name']}")

    sample_chunks = [
        "Students must maintain a minimum attendance of 75%.",
        "The End Semester Examination will be held from 18 November 2024.",
        "Minimum CGPA required for placement eligibility is 6.0.",
    ]
    sample_query = "What is the minimum attendance required?"

    # Test chunk embedding
    chunk_vecs = embed_chunks(sample_chunks)
    print(f"\nChunk embeddings shape: {chunk_vecs.shape}")
    print(f"First vector norm (should be ~1.0): {np.linalg.norm(chunk_vecs[0]):.6f}")

    # Test query embedding
    query_vec = embed_query(sample_query)
    print(f"Query embedding shape: {query_vec.shape}")
    print(f"Query vector norm (should be ~1.0): {np.linalg.norm(query_vec):.6f}")

    # Sanity check: the attendance chunk should score highest for the attendance query
    scores = chunk_vecs @ query_vec  # dot product = cosine similarity (normalized)
    best_idx = int(np.argmax(scores))
    print(f"\nQuery: '{sample_query}'")
    print("Scores:")
    for i, (text, score) in enumerate(zip(sample_chunks, scores)):
        marker = " <-- best match" if i == best_idx else ""
        print(f"  [{score:.4f}] {text[:60]}{marker}")

    if best_idx == 0:
        print("\nSANITY CHECK PASSED: Correct chunk ranked #1")
    else:
        print("\nWARNING: Expected chunk 0 to rank #1, check model/config")
        sys.exit(1)
