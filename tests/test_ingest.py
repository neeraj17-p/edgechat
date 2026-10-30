"""
tests/test_ingest.py
─────────────────────
Tests for src/ingest.py

Run from the project root:
    .\\venv\\Scripts\\python.exe -m pytest tests/test_ingest.py -v
"""

from __future__ import annotations

import pytest
from pathlib import Path

import fitz  # PyMuPDF


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_pdf_path(tmp_path_factory) -> Path:
    """
    Create a small in-memory PDF with 3 pages for testing:
      - Page 1: normal text content
      - Page 2: empty page (simulates a scan)
      - Page 3: normal text with a known keyword
    """
    tmp = tmp_path_factory.mktemp("pdfs")
    pdf_path = tmp / "test_document.pdf"

    doc = fitz.open()

    # Page 1: normal content
    p1 = doc.new_page()
    p1.insert_text((50, 100), "Attendance Policy Test Document", fontsize=14)
    p1.insert_text((50, 130), "Students must maintain 75% attendance.", fontsize=11)
    p1.insert_text((50, 150), "Failing to do so results in detention.", fontsize=11)

    # Page 2: intentionally empty (simulates a scan page)
    doc.new_page()  # no text inserted

    # Page 3: content with a specific fact for retrieval testing
    p3 = doc.new_page()
    p3.insert_text((50, 100), "Placement Policy", fontsize=14)
    p3.insert_text((50, 130), "Minimum CGPA required: 6.0", fontsize=11)
    p3.insert_text((50, 150), "No active backlogs allowed.", fontsize=11)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture(scope="module")
def ingested_pages(sample_pdf_path):
    """Return Page objects from the sample PDF."""
    from src.ingest import ingest_pdf
    return ingest_pdf(sample_pdf_path)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestIngestPdf:

    def test_returns_correct_page_count(self, ingested_pages):
        """PDF with 3 pages must produce exactly 3 Page objects."""
        assert len(ingested_pages) == 3

    def test_page_numbers_are_one_based(self, ingested_pages):
        """Page numbers must be 1, 2, 3 (not 0-indexed)."""
        numbers = [p.page_number for p in ingested_pages]
        assert numbers == [1, 2, 3]

    def test_doc_name_is_filename_only(self, ingested_pages):
        """doc_name must be the filename, not the full path."""
        for page in ingested_pages:
            assert page.doc_name == "test_document.pdf"
            assert "/" not in page.doc_name
            assert "\\" not in page.doc_name

    def test_page1_contains_expected_text(self, ingested_pages):
        """Page 1 text must contain the words we inserted."""
        text = ingested_pages[0].text
        assert "75%" in text or "75" in text
        assert "ttendance" in text  # 'Attendance' or 'attendance'

    def test_page2_is_detected_as_empty(self, ingested_pages):
        """Page 2 has no text — must be flagged as empty and scan suspect."""
        page2 = ingested_pages[1]
        assert page2.is_empty is True
        assert page2.is_scan_suspect is True
        assert page2.char_count < 50

    def test_page1_is_not_empty(self, ingested_pages):
        """Page 1 has real text — must NOT be flagged as empty."""
        assert ingested_pages[0].is_empty is False

    def test_page3_contains_expected_text(self, ingested_pages):
        """Page 3 must contain the placement CGPA fact."""
        text = ingested_pages[2].text
        assert "6.0" in text or "CGPA" in text

    def test_char_count_matches_text_length(self, ingested_pages):
        """char_count must equal len(text) for every page."""
        for page in ingested_pages:
            assert page.char_count == len(page.text)


class TestIngestAll:

    def test_ingest_all_finds_pdfs(self):
        """ingest_all must find and ingest PDFs in data/docs/."""
        from src.ingest import ingest_all
        docs_dir = Path("data/docs")
        if not docs_dir.exists():
            pytest.skip("data/docs not found")

        pages = ingest_all(docs_dir)
        # Should have ingested pages from the PDF corpus
        assert len(pages) >= 5


    def test_ingest_all_raises_on_missing_dir(self, tmp_path):
        """ingest_all must raise FileNotFoundError for non-existent directory."""
        from src.ingest import ingest_all
        with pytest.raises(FileNotFoundError):
            ingest_all(tmp_path / "nonexistent_dir")

    def test_ingest_all_raises_on_empty_dir(self, tmp_path):
        """ingest_all must raise RuntimeError when directory has no PDFs."""
        from src.ingest import ingest_all
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(RuntimeError, match="No PDF files found"):
            ingest_all(empty_dir)

    def test_all_pages_have_doc_name(self):
        """Every page in the full corpus must have a non-empty doc_name."""
        from src.ingest import ingest_all
        docs_dir = Path("data/docs")
        if not docs_dir.exists():
            pytest.skip("data/docs not found")
        pages = ingest_all(docs_dir)
        for page in pages:
            assert page.doc_name, f"Page {page.page_number} has no doc_name"

    def test_all_pages_have_positive_page_number(self):
        """Every page must have a page_number >= 1."""
        from src.ingest import ingest_all
        docs_dir = Path("data/docs")
        if not docs_dir.exists():
            pytest.skip("data/docs not found")
        pages = ingest_all(docs_dir)
        for page in pages:
            assert page.page_number >= 1


class TestFileNotFound:

    def test_ingest_pdf_raises_on_missing_file(self, tmp_path):
        """ingest_pdf must raise FileNotFoundError for a missing file."""
        from src.ingest import ingest_pdf
        with pytest.raises(FileNotFoundError):
            ingest_pdf(tmp_path / "ghost.pdf")

    def test_ingest_pdf_raises_on_wrong_extension(self, tmp_path):
        """ingest_pdf must raise ValueError for a non-PDF file."""
        from src.ingest import ingest_pdf
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("some text")
        with pytest.raises(ValueError, match="Expected a .pdf file"):
            ingest_pdf(txt_file)
