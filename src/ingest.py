"""
src/ingest.py
─────────────
Phase 1: PDF Ingestion

Reads every PDF in the docs directory page by page and returns a list of
Page objects, each containing the doc name, page number, and cleaned text.

Also produces a clear per-document report so you can catch:
  - Empty pages (no text at all — likely scanned images)
  - Very sparse pages (very few characters — possible scan or cover page)
  - Table-heavy pages (rows of numbers and pipes — may extract garbled)

Why page-by-page?
  We keep text anchored to its page number so that every chunk we create
  later can cite "attendance_policy.pdf, page 2" as its source. Merging
  pages would lose that traceability.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ── Thresholds for page classification ───────────────────────────────────────
# A page with fewer than this many characters is considered "empty" or a scan.
EMPTY_PAGE_CHAR_THRESHOLD = 50
# A page with fewer than this many characters is "sparse" (warning only).
SPARSE_PAGE_CHAR_THRESHOLD = 200
# If more than this fraction of lines look like table rows, flag as table-heavy.
TABLE_LINE_FRACTION_THRESHOLD = 0.35


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Page:
    """
    One page from one PDF document.

    Attributes:
        doc_name:         Filename of the source PDF (e.g. 'attendance_policy.pdf')
        page_number:      1-based page number within the document
        text:             Cleaned text extracted from the page
        char_count:       Number of characters in the cleaned text
        is_empty:         True if page has fewer than EMPTY_PAGE_CHAR_THRESHOLD chars
        is_scan_suspect:  True if page is empty — likely a scanned image
        is_table_heavy:   True if majority of lines look like a table
    """
    doc_name: str
    page_number: int        # 1-based
    text: str
    char_count: int = field(init=False)
    is_empty: bool = field(init=False)
    is_scan_suspect: bool = field(init=False)
    is_table_heavy: bool = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)
        self.is_empty = self.char_count < EMPTY_PAGE_CHAR_THRESHOLD
        self.is_scan_suspect = self.is_empty
        self.is_table_heavy = _is_table_heavy(self.text)


# ── Text cleaning ─────────────────────────────────────────────────────────────

def _clean_text(raw: str) -> str:
    """
    Light cleaning of raw PDF text. Removes noise without destroying content.

    Steps:
      1. Normalise line endings (Windows CRLF → LF)
      2. Collapse runs of more than 2 blank lines into a single blank line
      3. Strip leading/trailing whitespace from each line
      4. Remove lines that are just page numbers or running headers
         (heuristic: line is purely digits, or very short repeated line)
      5. Strip leading/trailing whitespace from the full text
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Strip each line
    lines = [line.strip() for line in text.split("\n")]

    # Remove lines that are just a page number (1–3 digits alone on a line)
    lines = [ln for ln in lines if not re.fullmatch(r"\d{1,3}", ln)]

    # Collapse multiple blank lines into one
    cleaned_lines: list[str] = []
    consecutive_blanks = 0
    for ln in lines:
        if ln == "":
            consecutive_blanks += 1
            if consecutive_blanks <= 1:
                cleaned_lines.append("")
        else:
            consecutive_blanks = 0
            cleaned_lines.append(ln)

    return "\n".join(cleaned_lines).strip()


def _is_table_heavy(text: str) -> bool:
    """
    Heuristic: if more than TABLE_LINE_FRACTION_THRESHOLD of non-blank lines
    contain multiple pipe characters or look like CSV/tab-separated data,
    the page is likely table-heavy and may extract poorly.
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return False
    table_like = sum(
        1 for ln in lines
        if ln.count("|") >= 2 or ln.count("\t") >= 3
    )
    return (table_like / len(lines)) > TABLE_LINE_FRACTION_THRESHOLD


# ── Core extraction ───────────────────────────────────────────────────────────

def _extract_pages(pdf_path: Path) -> Iterator[Page]:
    """
    Open one PDF and yield one Page object per page.

    Uses fitz (PyMuPDF) text extraction with layout preservation.
    Each page's text is extracted independently so we keep page numbers.
    """
    doc_name = pdf_path.name
    try:
        pdf = fitz.open(str(pdf_path))
    except Exception as exc:
        logger.error("Cannot open PDF %s: %s", pdf_path, exc)
        raise

    logger.info("Ingesting %s (%d pages)", doc_name, len(pdf))

    for page_index in range(len(pdf)):
        page_num = page_index + 1  # 1-based
        fitz_page = pdf[page_index]

        # "text" mode gives plain text; "blocks" would give bounding boxes
        raw_text = fitz_page.get_text("text")
        cleaned = _clean_text(raw_text)

        yield Page(
            doc_name=doc_name,
            page_number=page_num,
            text=cleaned,
        )

    pdf.close()


def ingest_pdf(pdf_path: Path) -> list[Page]:
    """
    Ingest a single PDF file and return its pages.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of Page objects, one per page.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        RuntimeError:      If PyMuPDF cannot open the file.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path.name}")

    return list(_extract_pages(pdf_path))


def ingest_all(docs_dir: Path) -> list[Page]:
    """
    Ingest all PDFs found in docs_dir (non-recursive).

    Args:
        docs_dir: Directory containing source PDF files.

    Returns:
        List of Page objects from ALL documents, in document order.

    Raises:
        FileNotFoundError: If docs_dir does not exist.
        RuntimeError:      If docs_dir contains no PDFs.
    """
    if not docs_dir.exists():
        raise FileNotFoundError(
            f"Docs directory not found: {docs_dir}\n"
            "Create it and add your PDFs, or check your config.yaml path."
        )

    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(
            f"No PDF files found in {docs_dir}.\n"
            "Add your PDF documents and run again."
        )

    logger.info("Found %d PDF(s) in %s", len(pdf_files), docs_dir)

    all_pages: list[Page] = []
    for pdf_path in pdf_files:
        pages = ingest_pdf(pdf_path)
        all_pages.extend(pages)

    return all_pages


# ── Report ────────────────────────────────────────────────────────────────────

def print_ingestion_report(pages: list[Page]) -> None:
    """
    Print a clear, human-readable per-document ingestion report.

    For each document:
      - Total pages
      - Characters per page (min / avg / max)
      - Count of empty pages (scan suspects)
      - Count of table-heavy pages
      - List of empty page numbers (so you can inspect them)
    """
    if not pages:
        print("No pages found — nothing to report.")
        return

    # Group pages by document
    docs: dict[str, list[Page]] = {}
    for page in pages:
        docs.setdefault(page.doc_name, []).append(page)

    print("\n" + "=" * 65)
    print("  INGESTION REPORT")
    print("=" * 65)

    total_pages = 0
    total_empty = 0
    total_tables = 0

    for doc_name, doc_pages in docs.items():
        chars = [p.char_count for p in doc_pages]
        empty_pages = [p for p in doc_pages if p.is_empty]
        table_pages = [p for p in doc_pages if p.is_table_heavy]
        sparse_pages = [
            p for p in doc_pages
            if not p.is_empty and p.char_count < SPARSE_PAGE_CHAR_THRESHOLD
        ]

        print(f"\n  Document : {doc_name}")
        print(f"  Pages    : {len(doc_pages)}")
        print(f"  Chars    : min={min(chars)}, avg={int(sum(chars)/len(chars))}, max={max(chars)}")

        if empty_pages:
            pnums = [str(p.page_number) for p in empty_pages]
            print(f"  [WARN] Empty pages (possible scans): {', '.join(pnums)}")
        else:
            print("  [OK] No empty pages detected")

        if sparse_pages:
            pnums = [str(p.page_number) for p in sparse_pages]
            print(f"  [WARN] Sparse pages (<{SPARSE_PAGE_CHAR_THRESHOLD} chars): {', '.join(pnums)}")

        if table_pages:
            pnums = [str(p.page_number) for p in table_pages]
            print(f"  [INFO] Table-heavy pages: {', '.join(pnums)}")

        total_pages += len(doc_pages)
        total_empty += len(empty_pages)
        total_tables += len(table_pages)

    print("\n" + "-" * 65)
    print(f"  TOTAL: {len(docs)} document(s), {total_pages} pages")
    print(f"         {total_empty} empty page(s), {total_tables} table-heavy page(s)")
    print("=" * 65 + "\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick test: run from the project root to see the ingestion report.

    Usage:
        .\\venv\\Scripts\\python.exe -m src.ingest
    """
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config to get the docs directory
    try:
        from src.config import config
        docs_dir = config.get_docs_dir()
    except Exception as exc:
        print(f"Could not load config: {exc}")
        sys.exit(1)

    print(f"Loading PDFs from: {docs_dir}")
    try:
        pages = ingest_all(docs_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print_ingestion_report(pages)

    # Show a text preview of page 1 of the first document
    if pages:
        first = pages[0]
        print(f"--- Preview: {first.doc_name}, page {first.page_number} ---")
        print(first.text[:500])
        print("...")
