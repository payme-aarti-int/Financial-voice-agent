from __future__ import annotations

import re
from pathlib import Path
import pdfplumber
from retrieval.extractors.base import BaseExtractor, ExtractedDocument

H1_SIZE = 17.0
SKIP_SIZE = 10.0
CHUNK_SIZE = 500
OVERLAP = 100


def _extract_lines(page) -> list[dict]:
    lines: dict[float, dict] = {}
    for char in page.chars:
        top = round(char["top"], 1)
        if top not in lines:
            lines[top] = {"text": "", "size": char["size"], "top": top}
        lines[top]["text"] += char["text"]
    return [v for _, v in sorted(lines.items())]


def _classify_line(line: dict) -> str:
    size = line["size"]
    text = line["text"].strip()

    if not text:
        return "skip"
    if size < SKIP_SIZE:
        return "skip"

    # h1 by font size
    if size >= H1_SIZE:
        return "h1"

    # h2 by pattern — e.g. "II.1 LICENSING" or "IV.3.a Credit"
    if re.match(r"^[IVXLC]+\.\d+(\.[a-z])?\s+\w", text):
        return "h2"

    return "body"


def _make_chunks(words: list[str], size: int, overlap: int) -> list[str]:
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += size - overlap
    return chunks


class PDFExtractor(BaseExtractor):

    def extract(self) -> ExtractedDocument:
        pages_out = []
        h1 = h2 = None
        accumulated_words: list[str] = []
        current_page = 1
        total_pages = 0

        try:
            with pdfplumber.open(self.file_path) as pdf:
                total_pages = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages, start=1):
                    current_page = page_num
                    lines = _extract_lines(page)

                    for line in lines:
                        kind = _classify_line(line)
                        text = line["text"].strip()

                        if kind == "skip":
                            continue

                        elif kind == "h1":
                            if accumulated_words:
                                for chunk in _make_chunks(accumulated_words, CHUNK_SIZE, OVERLAP):
                                    if chunk.strip():
                                        pages_out.append({"page": page_num, "h1": h1, "h2": h2, "text": chunk, "ocr_used": False})
                                accumulated_words = []
                            h1 = text
                            h2 = None

                        elif kind == "h2":
                            if accumulated_words:
                                for chunk in _make_chunks(accumulated_words, CHUNK_SIZE, OVERLAP):
                                    if chunk.strip():
                                        pages_out.append({"page": page_num, "h1": h1, "h2": h2, "text": chunk, "ocr_used": False})
                                accumulated_words = []
                            h2 = text

                        else:
                            words = text.split()
                            accumulated_words.extend(words)
                            if len(accumulated_words) >= CHUNK_SIZE:
                                for chunk in _make_chunks(accumulated_words[:CHUNK_SIZE], CHUNK_SIZE, OVERLAP):
                                    if chunk.strip():
                                        pages_out.append({"page": page_num, "h1": h1, "h2": h2, "text": chunk, "ocr_used": False})
                                accumulated_words = accumulated_words[CHUNK_SIZE - OVERLAP:]

                if accumulated_words:
                    for chunk in _make_chunks(accumulated_words, CHUNK_SIZE, OVERLAP):
                        if chunk.strip():
                            pages_out.append({"page": current_page, "h1": h1, "h2": h2, "text": chunk, "ocr_used": False})

        except Exception as exc:
            return ExtractedDocument(
                source_file=self.file_path.name,
                file_type="pdf",
                extracted_at=self._timestamp(),
                metadata={"error": str(exc)},
            )

        return ExtractedDocument(
            source_file=self.file_path.name,
            file_type="pdf",
            extracted_at=self._timestamp(),
            pages=pages_out,
            metadata={"total_pages": total_pages, "total_chunks": len(pages_out)},
        )
