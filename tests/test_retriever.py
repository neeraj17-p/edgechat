"""
tests/test_retriever.py
───────────────────────
Tests for src/retriever.py (Phase 3).

Run from the project root:
    .\\venv\\Scripts\\python.exe -m pytest tests/test_retriever.py -v
"""

from __future__ import annotations

import pytest
from src.retriever import retrieve, retrieve_for_eval, RetrievalResult


class TestRetrieverBasics:

    def test_empty_query_raises_value_error(self):
        """Passing an empty or whitespace-only query should raise ValueError."""
        with pytest.raises(ValueError, match="Query must be a non-empty string"):
            retrieve("")

        with pytest.raises(ValueError, match="Query must be a non-empty string"):
            retrieve("   ")

    def test_retrieve_returns_results_for_valid_query(self):
        """A valid query should return a list of RetrievalResult objects."""
        results = retrieve("Vice Chancellor of COEP", k=3)
        assert isinstance(results, list)
        assert len(results) > 0
        assert len(results) <= 3
        for r in results:
            assert isinstance(r, RetrievalResult)
            assert r.chunk is not None
            assert isinstance(r.score, float)
            assert isinstance(r.rank, int)
            assert r.doc_name.endswith(".pdf")
            assert r.page_number >= 1
            assert len(r.text.strip()) > 0

    def test_results_sorted_descending_by_score(self):
        """Retrieved chunks must be sorted with highest similarity score first."""
        results = retrieve("COEP hostel rules and regulations", k=4)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score
                assert results[i].rank == i + 1

    def test_threshold_filters_irrelevant_query(self):
        """A completely irrelevant query with a high threshold should return an empty list."""
        # Query having nothing to do with COEP or college
        results = retrieve(
            "Quantum entanglement teleportation in interstellar astrophysics",
            threshold=0.85,  # Impractically high threshold
        )
        assert results == []

    def test_retrieve_for_eval_respects_k(self):
        """retrieve_for_eval should retrieve exactly min(k, total) without threshold filtering."""
        results = retrieve_for_eval("attendance", k=5)
        assert len(results) == 5
        assert results[0].rank == 1
        assert results[-1].rank == 5

    def test_snippet_helper(self):
        """RetrievalResult.snippet() returns truncated text with ellipsis if long."""
        results = retrieve("Chancellor of COEP", k=1)
        assert len(results) > 0
        r = results[0]
        snippet = r.snippet(max_chars=50)
        assert len(snippet) <= 53  # 50 chars + "..."
