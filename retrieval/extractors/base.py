from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json


@dataclass
class ExtractedDocument:
    source_file: str
    file_type: str
    extracted_at: str
    ocr_used: bool = False
    pages: list[dict] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    text: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "file_type": self.file_type,
            "extracted_at": self.extracted_at,
            "ocr_used": self.ocr_used,
            "pages": self.pages,
            "rows": self.rows,
            "text": self.text,
            "metadata": self.metadata,
        }

    def save(self, output_dir: str | Path) -> Path:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(self.source_file).stem
        out_path = out_dir / f"{stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return out_path


class BaseExtractor(ABC):

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

    @abstractmethod
    def extract(self) -> ExtractedDocument:
        ...

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def run(self, output_dir: str | Path = "data/extracted") -> tuple[ExtractedDocument, Path]:
        doc = self.extract()
        saved_path = doc.save(output_dir)
        return doc, saved_path
