"""
src/indexer.py
──────────────
Phase 2: Build, save, and load the FAISS vector index.

What this does:
  1. Takes chunks from chunking.py and embeddings from embeddings.py
  2. Builds a FAISS IndexFlatIP (exact inner-product / cosine search)
  3. Saves the index + chunk metadata to data/index/ for reuse
  4. Provides a load function so we don't re-embed on every app start

Why IndexFlatIP?
  "Flat" = exact search (no approximation). With 413 pages × ~2 chunks/page
  ≈ ~800-1000 chunks, exact search takes <1ms — no need for approximate
  indexes (HNSW, IVF) which trade accuracy for speed at larger scales.
  "IP" = inner product. On L2-normalized vectors, IP == cosine similarity.

Why save to disk?
  Embedding all chunks takes ~5-30 seconds. Saving the index means the app
  starts instantly after the first build, and we only rebuild when PDFs change
  (using the --rebuild flag).

Files written to data/index/:
  faiss.index   — the FAISS index binary (fast binary format)
  chunks.json   — list of chunk metadata dicts (text + provenance)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import faiss
import numpy as np

from src.chunking import Chunk
from src.config import config
from src.embeddings import embed_chunks

logger = logging.getLogger(__name__)


# ── Build ─────────────────────────────────────────────────────────────────────

def build_index(chunks: list[Chunk]) -> tuple[faiss.IndexFlatIP, list[Chunk]]:
    """
    Embed all chunks and build a FAISS IndexFlatIP.

    Args:
        chunks: List of Chunk objects from chunking.py

    Returns:
        (faiss_index, chunks) — the index and the original chunk list
        (index positions correspond 1-to-1 with chunks list positions)

    Raises:
        ValueError: If chunks is empty.
    """
    if not chunks:
        raise ValueError(
            "Cannot build index from empty chunk list. "
            "Make sure PDFs are in data/docs/ and ingestion ran successfully."
        )

    logger.info("Building FAISS index for %d chunks ...", len(chunks))

    # Embed all chunk texts (no prefix — document side)
    texts = [c.text for c in chunks]
    embeddings: np.ndarray = embed_chunks(texts)  # shape: (N, dim)

    embedding_dim = embeddings.shape[1]
    logger.info("Embedding dimension: %d", embedding_dim)

    # Build FAISS IndexFlatIP (exact cosine search on normalized vectors)
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)

    logger.info("FAISS index built. Total vectors: %d", index.ntotal)
    return index, chunks


# ── Save ──────────────────────────────────────────────────────────────────────

def save_index(
    index: faiss.IndexFlatIP,
    chunks: list[Chunk],
    index_dir: Path | None = None,
) -> None:
    """
    Save the FAISS index and chunk metadata to disk.

    Writes two files:
      <index_dir>/faiss.index  — binary FAISS index
      <index_dir>/chunks.json  — JSON array of chunk metadata

    Args:
        index:     The built FAISS index.
        chunks:    The chunk list (must correspond to index positions).
        index_dir: Directory to save into. Defaults to config value.
    """
    if index_dir is None:
        index_dir = config.get_index_dir()

    index_dir.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    faiss_path = index_dir / "faiss.index"
    faiss.write_index(index, str(faiss_path))
    logger.info("FAISS index saved to %s (%d bytes)", faiss_path, faiss_path.stat().st_size)

    # Save chunk metadata as JSON
    chunks_path = index_dir / "chunks.json"
    chunk_dicts = [
        {
            "chunk_id": c.chunk_id,
            "doc_name": c.doc_name,
            "page_number": c.page_number,
            "text": c.text,
            "token_count": c.token_count,
        }
        for c in chunks
    ]
    with chunks_path.open("w", encoding="utf-8") as f:
        json.dump(chunk_dicts, f, ensure_ascii=False, indent=2)

    logger.info("Chunk metadata saved to %s (%d chunks)", chunks_path, len(chunks))
    print(f"Index saved: {index.ntotal} vectors in {faiss_path}")
    print(f"Metadata saved: {len(chunks)} chunks in {chunks_path}")


# ── Load ──────────────────────────────────────────────────────────────────────

def load_index(
    index_dir: Path | None = None,
) -> tuple[faiss.IndexFlatIP, list[Chunk]]:
    """
    Load the FAISS index and chunk metadata from disk.

    This is the fast path used by the app at startup — no re-embedding needed.

    Args:
        index_dir: Directory containing the saved index. Defaults to config.

    Returns:
        (faiss_index, chunks) ready for retrieval.

    Raises:
        FileNotFoundError: If index files don't exist yet.
                           Message tells user to run --rebuild.
    """
    if index_dir is None:
        index_dir = config.get_index_dir()

    faiss_path = index_dir / "faiss.index"
    chunks_path = index_dir / "chunks.json"

    if not faiss_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"Index not found in {index_dir}.\n"
            "Build it first by running:\n"
            "  .\\venv\\Scripts\\python.exe -m src.indexer --rebuild"
        )

    # Load FAISS index
    logger.info("Loading FAISS index from %s ...", faiss_path)
    index = faiss.read_index(str(faiss_path))
    logger.info("FAISS index loaded. Vectors: %d", index.ntotal)

    # Load chunk metadata
    with chunks_path.open("r", encoding="utf-8") as f:
        chunk_dicts: list[dict] = json.load(f)

    chunks = [
        Chunk(
            chunk_id=d["chunk_id"],
            doc_name=d["doc_name"],
            page_number=d["page_number"],
            text=d["text"],
            token_count=d["token_count"],
        )
        for d in chunk_dicts
    ]
    logger.info("Chunk metadata loaded. Chunks: %d", len(chunks))

    # Sanity check: index size must match chunk count
    if index.ntotal != len(chunks):
        raise RuntimeError(
            f"Index/chunk mismatch: FAISS has {index.ntotal} vectors "
            f"but chunks.json has {len(chunks)} entries. "
            "Run --rebuild to fix."
        )

    return index, chunks


def index_exists(index_dir: Path | None = None) -> bool:
    """Return True if a saved index exists and is ready to load."""
    if index_dir is None:
        index_dir = config.get_index_dir()
    return (index_dir / "faiss.index").exists() and (index_dir / "chunks.json").exists()


# ── CLI entry point ───────────────────────────────────────────────────────────

def run_rebuild() -> dict:
    """Full pipeline: ingest → chunk → embed → build → save → reload, returning stats."""
    from src.ingest import ingest_all
    from src.chunking import chunk_pages
    import src.retriever as retriever_mod

    docs_dir = config.get_docs_dir()
    pages = ingest_all(docs_dir)
    chunks = chunk_pages(pages)
    index, chunks = build_index(chunks)
    save_index(index, chunks)

    # Invalidate retriever memory cache
    retriever_mod._index_cache = None
    retriever_mod._chunks_cache = None

    return {
        "pages": len(pages),
        "docs": len(set(p.doc_name for p in pages)),
        "chunks": len(chunks),
        "vectors": index.ntotal,
        "dim": index.d,
    }


def _run_rebuild() -> None:
    """CLI version of full rebuild."""
    print("=" * 60)
    print("  EdgeChat Index Builder")
    print("=" * 60)
    stats = run_rebuild()
    print(f"\nIndex is ready!")
    print(f"  Documents : {stats['docs']}")
    print(f"  Pages     : {stats['pages']}")
    print(f"  Chunks    : {stats['chunks']}")
    print(f"  Vectors   : {stats['vectors']}")
    print(f"  Dimension : {stats['dim']}")
    print(f"\nRun the app with: streamlit run app.py")



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="EdgeChat FAISS Index Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Build index (first time or after adding new PDFs):
    .\\venv\\Scripts\\python.exe -m src.indexer --rebuild

  Check if index exists:
    .\\venv\\Scripts\\python.exe -m src.indexer --status
        """,
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the index from scratch (required on first run or after PDF changes)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check if the index exists and print its stats",
    )

    args = parser.parse_args()

    if args.rebuild:
        _run_rebuild()

    elif args.status:
        if index_exists():
            idx, cks = load_index()
            print(f"Index EXISTS: {idx.ntotal} vectors, {len(cks)} chunks, dim={idx.d}")
        else:
            print("Index NOT found. Run: python -m src.indexer --rebuild")
            sys.exit(1)

    else:
        parser.print_help()
