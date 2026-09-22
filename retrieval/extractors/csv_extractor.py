from __future__ import annotations

from pathlib import Path
import pandas as pd

from retrieval.extractors.base import BaseExtractor, ExtractedDocument


class CSVExtractor(BaseExtractor):

    def extract(self) -> ExtractedDocument:
        try:
            df = pd.read_csv(self.file_path)
        except Exception as exc:
            return ExtractedDocument(
                source_file=self.file_path.name,
                file_type="csv",
                extracted_at=self._timestamp(),
                metadata={"error": str(exc)},
            )

        # detect date or year column for metadata
        date_col = next(
            (c for c in df.columns if "date" in c.lower() or "year" in c.lower()),
            None,
        )

        rows = []
        for i, row in df.iterrows():
            row_dict = row.to_dict()

            # build natural language text from row
            parts = []
            for col, val in row_dict.items():
                if pd.notna(val):
                    parts.append(f"{col}: {val}")
            text = f"From {self.file_path.name} — " + ", ".join(parts) + "."

            meta = {
                "source_file": self.file_path.name,
                "file_type": "csv",
                "row_index": int(i),
                "columns": list(df.columns),
            }
            if date_col and pd.notna(row.get(date_col)):
                meta["period"] = str(row[date_col])

            rows.append({"text": text, "metadata": meta})

        return ExtractedDocument(
            source_file=self.file_path.name,
            file_type="csv",
            extracted_at=self._timestamp(),
            rows=rows,
            metadata={
                "total_rows": len(df),
                "columns": list(df.columns),
            },
        )