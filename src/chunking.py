"""
src/chunking.py
───────────────
Phase 2: Token-aware text chunking.

Takes a list of Page objects from ingest.py and splits each page's text
into overlapping chunks of roughly 300-500 tokens (configurable in config.yaml).

Key design decisions:
  - We NEVER merge text across different pages or documents. Each chunk
    stays strictly within one page so source citations stay accurate.
  - We use tiktoken (the same fast BPE tokenizer OpenAI uses) to count
    tokens, which gives a good approximation for the BGE embedding model.
  - We split on paragraph/sentence boundaries first, falling back to
    smaller units. This is what RecursiveCharacterTextSplitter does.
  - Empty pages (from scans) produce zero chunks — silently skipped.

Each Chunk carries:
    chunk_id    → unique ID: "filename.pdf::p3::c0"
    doc_name    → source filename
    page_number → 1-based page number
    text        → the chunk text
    token_count → approximate token count
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import config
from src.ingest import Page

logger = logging.getLogger(__name__)

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A single text chunk with full provenance metadata.

    chunk_id is formatted as:  "<doc_name>::p<page>::c<index>"
    Example: "UG-Programmes-Academic-Rules.pdf::p5::c2"
    """
    chunk_id: str
    doc_name: str
    page_number: int        # 1-based, same as the source page
    text: str
    token_count: int = field(default=0)

    def __post_init__(self) -> None:
        if self.token_count == 0:
            # Recompute if not provided
            enc = _get_tokenizer()
            self.token_count = len(enc.encode(self.text))


# ── Tokenizer (cached) ────────────────────────────────────────────────────────

_tokenizer_cache: tiktoken.Encoding | None = None

def _get_tokenizer() -> tiktoken.Encoding:
    """
    Return a cached tiktoken tokenizer.

    We use cl100k_base (used by GPT-4 / text-embedding-3).
    It over-counts slightly compared to BERT-based models like BGE,
    so our 400-token target may produce chunks of ~340 BGE tokens —
    which is fine and keeps us safely under the 512-token BGE limit.
    """
    global _tokenizer_cache
    if _tokenizer_cache is None:
        _tokenizer_cache = tiktoken.get_encoding("cl100k_base")
    return _tokenizer_cache


def _token_length(text: str) -> int:
    """Return the number of tokens in text using the cached tokenizer."""
    return len(_get_tokenizer().encode(text))


# ── Core splitter ─────────────────────────────────────────────────────────────

def _make_splitter(chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Build a LangChain RecursiveCharacterTextSplitter configured with:
      - Token-based length function (tiktoken)
      - Paragraph → sentence → word fallback separators
      - Configured chunk_size and overlap from config.yaml

    RecursiveCharacterTextSplitter tries each separator in order and
    only falls back to the next one if the chunk would be too long.
    This means it prefers to split on paragraph breaks, then newlines,
    then sentences, then words — preserving semantic coherence.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=_token_length,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        keep_separator=False,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_pages(pages: list[Page]) -> list[Chunk]:
    """
    Split a list of Page objects into a flat list of Chunk objects.

    Reads chunk_size and overlap from config.yaml.
    Pages with no text (empty/scan pages) are silently skipped.

    Args:
        pages: Output from ingest.ingest_all() or ingest.ingest_pdf()

    Returns:
        List of Chunk objects, ordered by document then page then position.
    """
    chunk_size: int = config.chunking["chunk_size"]
    overlap: int = config.chunking["overlap"]
    min_chars: int = config.chunking["min_chunk_chars"]

    splitter = _make_splitter(chunk_size, overlap)
    enc = _get_tokenizer()

    all_chunks: list[Chunk] = []
    skipped_pages = 0

    for page in pages:
        # Skip empty pages (scans produce no text)
        if page.is_empty or len(page.text.strip()) < min_chars:
            skipped_pages += 1
            continue

        # Split this page's text into sub-chunks
        texts = splitter.split_text(page.text)

        for idx, text in enumerate(texts):
            text = text.strip()
            if len(text) < min_chars:
                continue  # discard tiny leftover fragments

            chunk_id = f"{page.doc_name}::p{page.page_number}::c{idx}"
            token_count = len(enc.encode(text))

            all_chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_name=page.doc_name,
                page_number=page.page_number,
                text=text,
                token_count=token_count,
            ))

    logger.info(
        "Chunking complete: %d pages → %d chunks (skipped %d empty pages)",
        len(pages), len(all_chunks), skipped_pages,
    )
    return all_chunks


def print_chunking_report(chunks: list[Chunk]) -> None:
    """
    Print a summary of the chunking results: count, token distribution,
    and per-document breakdown.
    """
    if not chunks:
        print("No chunks produced.")
        return

    token_counts = [c.token_count for c in chunks]
    docs: dict[str, list[Chunk]] = {}
    for c in chunks:
        docs.setdefault(c.doc_name, []).append(c)

    print("\n" + "=" * 65)
    print("  CHUNKING REPORT")
    print("=" * 65)
    print(f"  Total chunks : {len(chunks)}")
    print(f"  Token stats  : min={min(token_counts)}, "
          f"avg={int(sum(token_counts)/len(token_counts))}, "
          f"max={max(token_counts)}")
    print(f"  Documents    : {len(docs)}")
    print()
    print(f"  {'Document':<55} {'Chunks':>6}  {'Avg tok':>7}")
    print(f"  {'-'*55} {'-'*6}  {'-'*7}")
    for doc_name, doc_chunks in sorted(docs.items()):
        avg_tok = int(sum(c.token_count for c in doc_chunks) / len(doc_chunks))
        # Truncate long filenames for display
        display = doc_name if len(doc_name) <= 55 else doc_name[:52] + "..."
        print(f"  {display:<55} {len(doc_chunks):>6}  {avg_tok:>7}")
    print("=" * 65 + "\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    from src.ingest import ingest_all

    try:
        pages = ingest_all(config.get_docs_dir())
        chunks = chunk_pages(pages)
        print_chunking_report(chunks)

        # Show a sample chunk
        if chunks:
            sample = chunks[10] if len(chunks) > 10 else chunks[0]
            print(f"--- Sample chunk: {sample.chunk_id} ({sample.token_count} tokens) ---")
            print(sample.text[:400])
            print("...")
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
