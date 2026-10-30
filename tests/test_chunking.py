"""
tests/test_chunking.py
───────────────────────
Tests for src/chunking.py

Run from the project root:
    .\\venv\\Scripts\\python.exe -m pytest tests/test_chunking.py -v
"""

from __future__ import annotations

import pytest
from src.ingest import Page
from src.chunking import Chunk, chunk_pages


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_page(text: str, doc: str = "test.pdf", page: int = 1) -> Page:
    """Create a Page object for testing without reading a real PDF."""
    return Page(doc_name=doc, page_number=page, text=text)


def long_text(words: int = 600) -> str:
    """Generate a long text with the given approximate word count."""
    sentence = "This is a sentence about college academic regulations. "
    return (sentence * (words // 10 + 1))[:words * 6]  # ~6 chars per word


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestChunkBasics:

    def test_short_page_produces_one_chunk(self):
        """A short page (< chunk_size tokens, but > min_chunk_chars) should produce 1 chunk."""
        # Text is well above min_chunk_chars=100 but well below chunk_size=400 tokens
        text = (
            "Students must maintain a minimum attendance of 75% in each subject per semester. "
            "This applies separately to theory classes, laboratory sessions, and tutorials. "
            "Students who fail to meet this threshold will be declared detained."
        )
        page = make_page(text)
        chunks = chunk_pages([page])
        assert len(chunks) == 1

    def test_long_page_produces_multiple_chunks(self):
        """A long page (> chunk_size tokens) must be split into multiple chunks."""
        page = make_page(long_text(words=800))  # well over 400 tokens
        chunks = chunk_pages([page])
        assert len(chunks) > 1

    def test_chunk_has_correct_doc_name(self):
        """Every chunk must carry the source document name."""
        page = make_page("Some text about exams.", doc="exam_rules.pdf")
        chunks = chunk_pages([page])
        for c in chunks:
            assert c.doc_name == "exam_rules.pdf"

    def test_chunk_has_correct_page_number(self):
        """Every chunk must carry the correct 1-based page number."""
        page = make_page("Some text about attendance.", doc="attendance.pdf", page=3)
        chunks = chunk_pages([page])
        for c in chunks:
            assert c.page_number == 3

    def test_chunk_id_format(self):
        """chunk_id must follow the format: doc::pN::cM"""
        page = make_page("Attendance policy text here.", doc="policy.pdf", page=2)
        chunks = chunk_pages([page])
        for c in chunks:
            parts = c.chunk_id.split("::")
            assert len(parts) == 3
            assert parts[0] == "policy.pdf"
            assert parts[1].startswith("p")
            assert parts[2].startswith("c")

    def test_token_count_is_positive(self):
        """Every chunk must have a positive token count."""
        page = make_page("The minimum CGPA for placement is 6.0.", doc="placement.pdf")
        chunks = chunk_pages([page])
        for c in chunks:
            assert c.token_count > 0

    def test_chunk_text_is_nonempty(self):
        """No chunk should have empty or whitespace-only text."""
        page = make_page("Text about grading system and SGPA calculation.")
        chunks = chunk_pages([page])
        for c in chunks:
            assert c.text.strip() != ""


class TestNoCrossPageMerging:

    def test_chunks_do_not_merge_across_pages(self):
        """
        Pages are independent — no chunk should contain text from two pages.
        We verify by checking that each chunk's page_number is consistent.
        """
        page1 = make_page("Page one content about attendance.", doc="rules.pdf", page=1)
        page2 = make_page("Page two content about examinations.", doc="rules.pdf", page=2)
        chunks = chunk_pages([page1, page2])

        # All page 1 chunks must have page_number == 1
        p1_chunks = [c for c in chunks if "p1" in c.chunk_id]
        for c in p1_chunks:
            assert c.page_number == 1

        # All page 2 chunks must have page_number == 2
        p2_chunks = [c for c in chunks if "p2" in c.chunk_id]
        for c in p2_chunks:
            assert c.page_number == 2

    def test_two_docs_produce_separate_chunks(self):
        """Chunks from two different documents must not mix doc_name."""
        # Texts must be > min_chunk_chars=100 to pass the noise filter
        text_a = (
            "Document A describes the grading system used at Greenfield College. "
            "Grades range from O (Outstanding) to F (Fail) on a 10-point scale. "
            "A minimum of 40 marks is required to pass any subject."
        )
        text_b = (
            "Document B covers the placement eligibility criteria for students. "
            "A minimum CGPA of 6.0 is required to participate in campus drives. "
            "Students must also have no active backlogs at the time of placement."
        )
        page_a = make_page(text_a, doc="docA.pdf", page=1)
        page_b = make_page(text_b, doc="docB.pdf", page=1)
        chunks = chunk_pages([page_a, page_b])

        doc_a_chunks = [c for c in chunks if c.doc_name == "docA.pdf"]
        doc_b_chunks = [c for c in chunks if c.doc_name == "docB.pdf"]

        assert len(doc_a_chunks) >= 1
        assert len(doc_b_chunks) >= 1

        for c in doc_a_chunks:
            assert c.doc_name == "docA.pdf"
        for c in doc_b_chunks:
            assert c.doc_name == "docB.pdf"


class TestEdgeCases:

    def test_empty_page_produces_no_chunks(self):
        """An empty page (scan suspect) must produce zero chunks."""
        empty_page = make_page("")
        chunks = chunk_pages([empty_page])
        assert len(chunks) == 0

    def test_whitespace_only_page_produces_no_chunks(self):
        """A page with only whitespace must produce zero chunks."""
        page = make_page("   \n\n   \t  \n")
        chunks = chunk_pages([page])
        assert len(chunks) == 0

    def test_empty_page_list_produces_no_chunks(self):
        """chunk_pages([]) must return an empty list without error."""
        chunks = chunk_pages([])
        assert chunks == []

    def test_chunk_text_is_substring_of_page_text(self):
        """
        Each chunk's text must be found (approximately) in the original page text.
        We check the first 40 chars as a substring check.
        """
        original = (
            "The minimum attendance is 75% per subject. "
            "Lab sessions count as 2 periods. "
            "Tutorial sessions count as 1 period. " * 30
        )
        page = make_page(original)
        chunks = chunk_pages([page])
        for c in chunks:
            # The first 40 chars of the chunk must appear in the original page
            snippet = c.text[:40].strip()
            if snippet:
                assert snippet in original, (
                    f"Chunk text '{snippet}...' not found in original page text"
                )


class TestWithRealDocs:

    def test_real_corpus_produces_chunks(self):
        """Full corpus must produce a non-zero number of chunks."""
        from pathlib import Path
        from src.ingest import ingest_all

        docs_dir = Path("data/docs")
        if not docs_dir.exists():
            pytest.skip("data/docs not found")

        pages = ingest_all(docs_dir)
        chunks = chunk_pages(pages)

        assert len(chunks) > 0, "Expected at least 1 chunk from real corpus"

    def test_real_corpus_chunk_sizes_are_reasonable(self):
        """All chunks in the real corpus should be within a reasonable token range."""
        from pathlib import Path
        from src.ingest import ingest_all

        docs_dir = Path("data/docs")
        if not docs_dir.exists():
            pytest.skip("data/docs not found")

        pages = ingest_all(docs_dir)
        chunks = chunk_pages(pages)

        for c in chunks:
            # No chunk should be larger than 600 tokens (our max is 400 + overlap)
            assert c.token_count <= 600, (
                f"Oversized chunk in {c.doc_name} p{c.page_number}: {c.token_count} tokens"
            )
            # No chunk should be empty
            assert c.token_count > 0
