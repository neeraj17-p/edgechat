"""
tests/test_rag.py
─────────────────
Tests for src/rag.py and src/prompts.py (Phase 5).

Run from the project root:
    .\\venv\\Scripts\\python.exe -m pytest tests/test_rag.py -v
"""

from __future__ import annotations

import pytest
from src.chunking import Chunk
from src.prompts import NOT_FOUND_MESSAGE, build_rag_prompt, format_context_chunk
from src.rag import RAGResponse, SourceInfo, answer_question
from src.retriever import RetrievalResult


class TestPrompts:

    def test_format_context_chunk(self):
        """format_context_chunk() must include the doc_name, page_number, and chunk text."""
        chunk = Chunk(
            chunk_id="test_doc.pdf_p1_c0",
            doc_name="regulations.pdf",
            page_number=3,
            text="Attendance must be at least 75%.",
            token_count=10,
        )
        res = RetrievalResult(chunk=chunk, score=0.85, rank=1)
        formatted = format_context_chunk(res, 1)

        assert "regulations.pdf" in formatted
        assert "Page: 3" in formatted
        assert "Attendance must be at least 75%." in formatted

    def test_build_rag_prompt_structure(self):
        """build_rag_prompt() must assemble context and question clearly."""
        chunk = Chunk(
            chunk_id="test_doc.pdf_p1_c0",
            doc_name="hostel.pdf",
            page_number=1,
            text="Ragging is strictly prohibited.",
            token_count=8,
        )
        res = RetrievalResult(chunk=chunk, score=0.88, rank=1)
        prompt = build_rag_prompt("What is the ragging policy?", [res])

        assert "DOCUMENT CONTEXT:" in prompt
        assert "hostel.pdf" in prompt
        assert "Page: 1" in prompt
        assert "QUESTION: What is the ragging policy?" in prompt
        assert "ANSWER:" in prompt


class TestRAGPipeline:

    def test_empty_question_raises_value_error(self):
        """Calling answer_question() with an empty string must raise ValueError."""
        with pytest.raises(ValueError, match="Question must be a non-empty string"):
            answer_question("")

        with pytest.raises(ValueError, match="Question must be a non-empty string"):
            answer_question("   ")

    def test_answerable_question_returns_grounded_answer(self):
        """A valid question from the PDFs must return answer, citations, and metrics."""
        resp = answer_question("Who is the Vice-Chancellor of COEP Tech University?")

        assert isinstance(resp, RAGResponse)
        assert len(resp.answer) > 0
        assert not resp.is_refusal
        assert len(resp.sources) > 0
        assert resp.total_time > 0.0
        assert resp.retrieval_time > 0.0
        assert resp.generation_time > 0.0
        assert resp.completion_tokens > 0

        # Check first source
        top_source = resp.sources[0]
        assert isinstance(top_source, SourceInfo)
        assert top_source.doc_name.endswith(".pdf")
        assert top_source.page_number >= 1
        assert top_source.score > 0.60
        assert len(top_source.snippet) > 0

    def test_unanswerable_question_returns_exact_refusal(self):
        """An out-of-domain question below threshold must return exact not-found message."""
        resp = answer_question("What is the orbital velocity of the Saturn V moon rocket?")

        assert isinstance(resp, RAGResponse)
        assert resp.is_refusal
        assert resp.answer == NOT_FOUND_MESSAGE
        assert resp.sources == []
        assert resp.generation_time == 0.0
        assert resp.completion_tokens == 0
