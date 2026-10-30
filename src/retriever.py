"""
src/retriever.py
────────────────
Phase 3: Question → top-k relevant chunks.

Given a user's question, this module:
  1. Embeds the question into a 384-dim vector (with BGE query prefix)
  2. Searches the FAISS index for the most similar chunk vectors
  3. Filters out results below the similarity threshold
  4. Returns ranked RetrievalResult objects with chunk text + source info

If the best score is below the threshold, retrieval returns an empty list
and the RAG pipeline will respond with the "not found" message instead of
asking the LLM to guess.

Design note: We lazily load the index on first call and cache it in memory
for the rest of the session. The app loads once; the index stays loaded.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.chunking import Chunk
from src.config import config
from src.embeddings import embed_query
from src.indexer import load_index

logger = logging.getLogger(__name__)

# ── Module-level index cache ──────────────────────────────────────────────────
# Loaded once on first retrieve() call, reused for the rest of the session.
_index_cache = None
_chunks_cache: Optional[list[Chunk]] = None


def _get_index_and_chunks():
    """Lazy-load the FAISS index and chunks, cached after first call."""
    global _index_cache, _chunks_cache
    if _index_cache is None or _chunks_cache is None:
        logger.info("Loading FAISS index from disk (first call) ...")
        _index_cache, _chunks_cache = load_index()
        logger.info("Index ready: %d vectors", _index_cache.ntotal)
    return _index_cache, _chunks_cache


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    One retrieved chunk with its similarity score and rank.

    Attributes:
        chunk:   The Chunk object (text + doc_name + page_number)
        score:   Cosine similarity score (0.0 to 1.0; higher = more similar)
        rank:    1-based rank (rank=1 is the most relevant chunk)
    """
    chunk: Chunk
    score: float
    rank: int

    @property
    def doc_name(self) -> str:
        return self.chunk.doc_name

    @property
    def page_number(self) -> int:
        return self.chunk.page_number

    @property
    def text(self) -> str:
        return self.chunk.text

    def snippet(self, max_chars: int = 200) -> str:
        """Return a short preview of the chunk text."""
        t = self.text.strip()
        return t[:max_chars] + "..." if len(t) > max_chars else t


# ── Core retrieval ────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    k: Optional[int] = None,
    threshold: Optional[float] = None,
    debug: bool = False,
) -> list[RetrievalResult]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Args:
        query:     The user's question string.
        k:         Number of chunks to return. Defaults to config value.
        threshold: Minimum cosine similarity. Results below this are dropped.
                   If the best result is below threshold, returns [].
                   Defaults to config value.
        debug:     If True, logs all scores to help tune the threshold.

    Returns:
        List of RetrievalResult objects sorted by score descending (best first).
        Returns an empty list if no chunk meets the threshold.

    Raises:
        ValueError:      If query is empty.
        FileNotFoundError: If the index hasn't been built yet.
    """
    query = query.strip()
    if not query:
        raise ValueError("Query must be a non-empty string.")

    # Defaults from config
    if k is None:
        k = config.retrieval["top_k"]
    if threshold is None:
        threshold = config.retrieval["similarity_threshold"]
    debug = debug or config.retrieval.get("debug", False)

    t_start = time.perf_counter()

    # Step 1: Embed the query (with BGE query instruction prefix)
    query_vec = embed_query(query)                     # shape: (384,)
    query_vec = query_vec.reshape(1, -1)               # shape: (1, 384) for FAISS

    # Step 2: Search FAISS index
    index, chunks = _get_index_and_chunks()
    # Request more than k so we can filter by threshold and still return k
    search_k = min(k * 3, index.ntotal)
    scores_2d, indices_2d = index.search(query_vec, search_k)

    scores: np.ndarray = scores_2d[0]    # shape: (search_k,)
    indices: np.ndarray = indices_2d[0]  # shape: (search_k,)

    t_retrieval = time.perf_counter() - t_start

    # Step 3: Debug logging
    if debug:
        logger.debug("Retrieval scores for: %r", query[:80])
        for i, (idx, score) in enumerate(zip(indices, scores)):
            if idx == -1:
                break
            chunk = chunks[idx]
            logger.debug(
                "  rank=%d score=%.4f doc=%s page=%d",
                i + 1, score, chunk.doc_name, chunk.page_number,
            )

    # Step 4: Check threshold (on best score only)
    best_score = float(scores[0]) if len(scores) > 0 and indices[0] != -1 else 0.0
    if best_score < threshold:
        logger.info(
            "Best score %.4f is below threshold %.4f — returning empty results.",
            best_score, threshold,
        )
        return []

    # Step 5: Build result list (apply threshold, cap at k)
    results: list[RetrievalResult] = []
    rank = 1
    for idx, score in zip(indices, scores):
        if idx == -1:
            break
        if float(score) < threshold:
            break
        if rank > k:
            break
        results.append(RetrievalResult(
            chunk=chunks[int(idx)],
            score=float(score),
            rank=rank,
        ))
        rank += 1

    logger.info(
        "Retrieved %d/%d chunks above threshold %.2f in %.3fs",
        len(results), k, threshold, t_retrieval,
    )
    return results


def retrieve_for_eval(
    query: str,
    k: int = 5,
) -> list[RetrievalResult]:
    """
    Retrieve top-k chunks without threshold filtering.

    Used during evaluation (hit@k measurement) so we can count how many
    relevant chunks appear in top-1, top-3, top-5 regardless of scores.
    """
    return retrieve(query, k=k, threshold=0.0, debug=False)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_results(query: str, results: list[RetrievalResult]) -> None:
    """Pretty-print retrieval results to the terminal."""
    print(f"\n{'='*65}")
    print(f"  Query: {query!r}")
    print(f"{'='*65}")

    if not results:
        print("  No results above similarity threshold.")
        print("  Try lowering retrieval.similarity_threshold in config.yaml")
        print(f"{'='*65}\n")
        return

    for r in results:
        print(f"\n  Rank #{r.rank}  |  Score: {r.score:.4f}")
        print(f"  Source: {r.doc_name}  (page {r.page_number})")
        print(f"  {'-'*61}")
        # Print first 400 chars of the chunk, indented
        text_preview = r.text.strip()[:400]
        for line in text_preview.split("\n"):
            print(f"  {line}")
        if len(r.text) > 400:
            print("  [... truncated ...]")

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    """
    CLI: run a retrieval query from the terminal.

    Usage:
        .\\venv\\Scripts\\python.exe -m src.retriever "your question here"
        .\\venv\\Scripts\\python.exe -m src.retriever "your question" --k 5 --debug
    """
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="EdgeChat Retrieval CLI")
    parser.add_argument("query", nargs="?", help="Question to search for")
    parser.add_argument(
        "--k", type=int, default=None,
        help=f"Number of results (default: from config.yaml)"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Similarity threshold (default: from config.yaml)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Show all scores including below-threshold results"
    )

    args = parser.parse_args()

    if not args.query:
        # Interactive mode
        print("EdgeChat Retrieval CLI — type a question, Ctrl+C to quit")
        try:
            while True:
                q = input("\nQuestion> ").strip()
                if not q:
                    continue
                results = retrieve(q, k=args.k, threshold=args.threshold, debug=args.debug)
                _print_results(q, results)
        except KeyboardInterrupt:
            print("\nBye!")
    else:
        results = retrieve(
            args.query,
            k=args.k,
            threshold=args.threshold,
            debug=args.debug,
        )
        _print_results(args.query, results)
