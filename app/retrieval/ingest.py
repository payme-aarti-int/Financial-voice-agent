from __future__ import annotations

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

from app.retrieval.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EXCEL_PATH,
    HEADER_ROW,
    FIRST_DATA_ROW,
    SHEET_NAME,
)

DATE_COLUMN = "week_ending"


def load_excel_data() -> pd.DataFrame:
    raw = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=None)

    headers = raw.iloc[HEADER_ROW].tolist()
    headers[1] = DATE_COLUMN  # unlabelled in the sheet

    data = raw.iloc[FIRST_DATA_ROW:].copy()
    data.columns = headers

    data = data.drop(columns=[c for c in data.columns if pd.isna(c)], errors="ignore")

    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    data = data[data[DATE_COLUMN].notna()].reset_index(drop=True)

    return data


def create_document(row: pd.Series) -> str:
    
    date = row[DATE_COLUMN]
    iso = date.strftime("%Y-%m-%d")
    spoken = date.strftime("%d %B %Y")
    month_year = date.strftime("%B %Y")

    lines = [
        f"RBI Reserve Money, week ending {spoken} ({iso}, {month_year}). "
        f"All figures in rupees crore."
    ]

    for column, value in row.items():
        if column == DATE_COLUMN or pd.isna(value):
            continue
        lines.append(f"{column}: {float(value):,.2f} crore")

    return "\n".join(lines)


def build_index() -> None:
    print(f"Loading {EXCEL_PATH.name} ...")
    df = load_excel_data()
    print(f"  {len(df)} weekly rows, "
          f"{df[DATE_COLUMN].min().date()} to {df[DATE_COLUMN].max().date()}")
    print(f"  columns: {[c for c in df.columns if c != DATE_COLUMN]}")

    documents, metadatas, ids = [], [], []

    for _, row in df.iterrows():
        text = create_document(row)
        if not text.strip():
            continue

        date = row[DATE_COLUMN]
        documents.append(text)
        metadatas.append(
            {
                "source": EXCEL_PATH.name,
                "week_ending": date.strftime("%Y-%m-%d"),
                "year": int(date.year),
                "month": int(date.month),
                "quarter": f"{date.year}-Q{(date.month - 1) // 3 + 1}",
            }
        )
        ids.append(f"rbi-{date.strftime('%Y-%m-%d')}")

    print(f"Embedding {len(documents)} documents with {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        documents, show_progress_bar=True, normalize_embeddings=True
    ).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "RBI weekly reserve money, components and sources"},
    )

    collection.upsert(
        ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
    )

    print(f"\nIndexed {collection.count()} documents at {CHROMA_PATH}")
    print("\nSample document:")
    print(documents[0])


if __name__ == "__main__":
    build_index()