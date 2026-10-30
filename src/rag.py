"""
src/rag.py
──────────
Phase 5: End-to-End RAG Pipeline (Retrieve → Prompt → Generate → Sources).

Connects:
  1. src/retriever.py: semantic search over FAISS index.
  2. src/prompts.py: strict grounding instructions and context formatting.
  3. src/llm.py: local quantized LLM generation via Ollama.

Guarantees:
  - If retrieval similarity is below threshold: skips LLM and returns exact
    not-found message immediately.
  - Every answer includes verified sources with document name, page number,
    similarity score, and text snippet.
  - Returns complete latency breakdown (retrieval, generation, total).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from src.config import config
from src.llm import generate
from src.prompts import NOT_FOUND_MESSAGE, SYSTEM_PROMPT, build_rag_prompt
from src.retriever import RetrievalResult, retrieve

logger = logging.getLogger(__name__)


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class SourceInfo:
    """Citation metadata for a retrieved document passage."""
    doc_name: str
    page_number: int
    score: float
    snippet: str


@dataclass
class RAGResponse:
    """
    Complete response returned by the EdgeChat RAG pipeline.

    Attributes:
        question:          The original user question.
        answer:            The generated answer text (or not-found message).
        sources:           List of citation sources with document, page, and score.
        retrieval_time:    Time spent in embedding + FAISS search (seconds).
        generation_time:   Time spent in LLM prompt processing + token generation (seconds).
        total_time:        Total end-to-end response time (seconds).
        prompt_tokens:     Number of prompt tokens evaluated by the LLM.
        completion_tokens: Number of completion tokens generated.
        tokens_per_second: Generation throughput speed.
        model:             LLM model name.
        is_refusal:        True if the answer could not be found in the documents.
    """
    question: str
    answer: str
    sources: list[SourceInfo]
    retrieval_time: float
    generation_time: float
    total_time: float
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float
    model: str
    is_refusal: bool


# ── Core Pipeline ─────────────────────────────────────────────────────────────

def answer_question(
    question: str,
    k: Optional[int] = None,
    threshold: Optional[float] = None,
    temperature: Optional[float] = None,
    debug: bool = False,
) -> RAGResponse:
    """
    Answer a question using the local RAG pipeline over college PDFs.

    Args:
        question:    The user's inquiry string.
        k:           Number of context passages to retrieve (default: from config.yaml).
        threshold:   Similarity threshold for retrieval filtering.
        temperature: LLM sampling temperature (default: from config.yaml).
        debug:       If True, outputs verbose retrieval and generation logs.

    Returns:
        RAGResponse with answer, citations, and latency metrics.

    Raises:
        ValueError: If question is empty or whitespace.
    """
def prepare_rag_context(
    question: str,
    k: Optional[int] = None,
    threshold: Optional[float] = None,
    debug: bool = False,
) -> tuple[list[SourceInfo], Optional[str], float]:
    """
    Retrieve relevant passages and build the RAG prompt.

    Returns:
        (sources, prompt, retrieval_time)
        If rejected by threshold, prompt will be None and sources will be [].
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must be a non-empty string.")

    t_start = time.perf_counter()
    retrieved_chunks = retrieve(
        query=question,
        k=k,
        threshold=threshold,
        debug=debug,
    )
    retrieval_time = round(time.perf_counter() - t_start, 3)

    if not retrieved_chunks:
        return [], None, retrieval_time

    sources = [
        SourceInfo(
            doc_name=r.doc_name,
            page_number=r.page_number,
            score=round(r.score, 4),
            snippet=r.snippet(max_chars=220),
        )
        for r in retrieved_chunks
    ]
    prompt = build_rag_prompt(question, retrieved_chunks)
    return sources, prompt, retrieval_time


def answer_question(
    question: str,
    k: Optional[int] = None,
    threshold: Optional[float] = None,
    temperature: Optional[float] = None,
    debug: bool = False,
) -> RAGResponse:
    """
    Answer a question using the local RAG pipeline over college PDFs.

    Args:
        question:    The user's inquiry string.
        k:           Number of context passages to retrieve (default: from config.yaml).
        threshold:   Similarity threshold for retrieval filtering.
        temperature: LLM sampling temperature (default: from config.yaml).
        debug:       If True, outputs verbose retrieval and generation logs.

    Returns:
        RAGResponse with answer, citations, and latency metrics.

    Raises:
        ValueError: If question is empty or whitespace.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must be a non-empty string.")

    t_start = time.perf_counter()
    model_name = config.llm.get("model", "llama3.2:3b")

    # Step 1: Retrieve and build prompt
    sources, prompt, retrieval_time = prepare_rag_context(
        question=question,
        k=k,
        threshold=threshold,
        debug=debug,
    )

    # Step 2: Handle threshold refusal
    if prompt is None:
        total_time = round(time.perf_counter() - t_start, 3)
        return RAGResponse(
            question=question,
            answer=NOT_FOUND_MESSAGE,
            sources=[],
            retrieval_time=retrieval_time,
            generation_time=0.0,
            total_time=total_time,
            prompt_tokens=0,
            completion_tokens=0,
            tokens_per_second=0.0,
            model=model_name,
            is_refusal=True,
        )

    # Step 3: LLM generation
    max_tokens = config.generation.get("max_answer_tokens", 512)
    t_gen_start = time.perf_counter()
    llm_resp = generate(
        prompt=prompt,
        system=SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    generation_time = time.perf_counter() - t_gen_start


    total_time = time.perf_counter() - t_start
    answer_text = llm_resp.text.strip()

    # Determine if answer is a refusal
    is_refusal = (
        NOT_FOUND_MESSAGE.lower() in answer_text.lower()
        or "could not find" in answer_text.lower()
        or "not mentioned in the provided" in answer_text.lower()
    )
    if is_refusal:
        # Standardize exact refusal wording
        answer_text = NOT_FOUND_MESSAGE

    return RAGResponse(
        question=question,
        answer=answer_text,
        sources=sources,
        retrieval_time=round(retrieval_time, 3),
        generation_time=round(generation_time, 3),
        total_time=round(total_time, 3),
        prompt_tokens=llm_resp.prompt_tokens,
        completion_tokens=llm_resp.completion_tokens,
        tokens_per_second=llm_resp.tokens_per_second,
        model=llm_resp.model,
        is_refusal=is_refusal,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def print_rag_response(resp: RAGResponse) -> None:
    """Pretty-print the complete RAG answer, sources, and metrics."""
    print("\n" + "=" * 70)
    print(f"  QUESTION: {resp.question}")
    print("=" * 70)
    print(f"\n  ANSWER:\n  {resp.answer}\n")
    print("-" * 70)

    if resp.sources:
        print("  SOURCES:")
        for idx, s in enumerate(resp.sources, 1):
            print(f"   [{idx}] {s.doc_name} (Page {s.page_number}) | Similarity: {s.score:.4f}")
            print(f"       \"{s.snippet}\"")
    else:
        print("  SOURCES: None (query rejected by similarity threshold)")

    print("-" * 70)
    print(
        f"  PERFORMANCE: Total: {resp.total_time:.2f}s | "
        f"Retrieval: {resp.retrieval_time:.2f}s | "
        f"Generation: {resp.generation_time:.2f}s ({resp.tokens_per_second:.1f} tok/s, {resp.completion_tokens} tokens)"
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="EdgeChat RAG Pipeline CLI")
    parser.add_argument("question", nargs="?", help="Question to ask EdgeChat")
    parser.add_argument("--k", type=int, default=None, help="Number of retrieved chunks")
    parser.add_argument("--threshold", type=float, default=None, help="Similarity threshold")
    parser.add_argument("--debug", action="store_true", help="Print debug scores")

    args = parser.parse_args()

    if not args.question:
        print("EdgeChat RAG CLI — Ask questions about your college PDFs (Ctrl+C to exit)")
        try:
            while True:
                q = input("\nEdgeChat> ").strip()
                if not q:
                    continue
                res = answer_question(q, k=args.k, threshold=args.threshold, debug=args.debug)
                print_rag_response(res)
        except KeyboardInterrupt:
            print("\nGoodbye!")
    else:
        res = answer_question(
            args.question,
            k=args.k,
            threshold=args.threshold,
            debug=args.debug,
        )
        print_rag_response(res)
