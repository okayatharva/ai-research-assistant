"""Functions for reading PDF text and splitting it into smaller passages."""

import pymupdf

from config import CHUNK_OVERLAP, CHUNK_SIZE


def extract_pdf_chunks(pdf_bytes: bytes, filename: str) -> list[dict]:
    """Extract selectable PDF text and return overlapping chunks with page info.

    Each returned item contains text, page, and source. Page numbers are kept
    so the user can go back to the original paper and check the answer.
    """
    chunks = []

    # PyMuPDF reads the uploaded PDF directly from memory; no temporary file
    # needs to be created on disk.
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        for page_index, page in enumerate(document):
            page_text = page.get_text("text").strip()
            if not page_text:
                # A scanned page may have no selectable text. OCR is not added yet.
                continue

            page_number = page_index + 1  # People usually count pages from 1.
            start = 0

            while start < len(page_text):
                end = start + CHUNK_SIZE
                text = page_text[start:end].strip()

                if text:
                    chunks.append({
                        "text": text,
                        "page": page_number,
                        "source": filename,
                    })

                if end >= len(page_text):
                    break

                # Move backwards a little so neighbouring chunks share context.
                start = end - CHUNK_OVERLAP

    return chunks
