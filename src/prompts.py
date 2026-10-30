"""
src/prompts.py
──────────────
Phase 5: Prompt templates and formatting for the EdgeChat RAG pipeline.

Follows Core Design Rules:
  - Strict grounding: Answer ONLY from the provided context.
  - Standard refusal: If not found, reply exactly:
    "I could not find this in the provided documents."
  - Exact quotations: Quote numbers, percentages, and names verbatim.
  - Clear citation anchors: Each passage is labelled with Document and Page.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from src.config import config

if TYPE_CHECKING:
    from src.retriever import RetrievalResult

NOT_FOUND_MESSAGE: str = config.generation.get(
    "not_found_message", "I could not find this in the provided documents."
)

SYSTEM_PROMPT: str = f"""You are EdgeChat, a strict, factual college assistant.

Your task is to answer the user's question using ONLY the provided document excerpts below.

Strict Rules:
1. Answer strictly using information explicitly stated in the provided context.
2. If the answer cannot be found in the provided context, respond with EXACTLY this sentence and nothing else:
   "{NOT_FOUND_MESSAGE}"
3. NEVER assume, extrapolate, or use outside knowledge. If a number, percentage, or policy is not in the text, do not guess it.
4. Quote numbers, dates, marks, percentages, and titles exactly as written in the excerpts.
5. Keep your answer direct, clear, and concise. Do not include introductory filler like "Based on the provided documents...".
"""


def format_context_chunk(result: "RetrievalResult", index: int) -> str:
    """Format a single retrieved chunk with its source document and page number."""
    return (
        f"[Passage {index} | Document: {result.doc_name} | Page: {result.page_number}]\n"
        f"{result.text.strip()}"
    )


def build_rag_prompt(query: str, results: list["RetrievalResult"]) -> str:
    """
    Construct the complete user prompt containing labeled context passages and the question.

    Args:
        query:   The user's question.
        results: List of retrieved chunks.

    Returns:
        Formatted prompt string.
    """
    context_sections = [
        format_context_chunk(r, i + 1) for i, r in enumerate(results)
    ]
    context_block = "\n\n".join(context_sections)

    return (
        f"DOCUMENT CONTEXT:\n"
        f"-----------------\n"
        f"{context_block}\n"
        f"-----------------\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER:"
    )
