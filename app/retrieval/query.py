from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer

from app.retrieval.config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL


class RBIQueryEngine:
    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        try:
            self.collection = self.client.get_collection(name=COLLECTION_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' not found at {CHROMA_PATH}. "
                f"Run: python -m app.retrieval.ingest"
            ) from exc

    def search(
        self,
        question: str,
        top_k: int = 5,
        year: int | None = None,
        quarter: str | None = None,
    ) -> list[dict]:
        
        where: dict | None = None
        if year is not None and quarter is not None:
            where = {"$and": [{"year": year}, {"quarter": quarter}]}
        elif year is not None:
            where = {"year": year}
        elif quarter is not None:
            where = {"quarter": quarter}

        embedding = self.model.encode(
            [question], normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=embedding,
            n_results=top_k,
            where=where,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            {"document": d, "metadata": m, "distance": dist}
            for d, m, dist in zip(documents, metadatas, distances)
        ]

    def count(self) -> int:
        return self.collection.count()


if __name__ == "__main__":
    engine = RBIQueryEngine()
    print(f"{engine.count()} documents indexed.\n")
    print("Ask a question, or 'exit'. Prefix with a year to filter, e.g.")
    print("  2020 | currency in circulation\n")

    while True:
        raw = input("> ").strip()
        if raw.lower() in {"exit", "quit", "q"}:
            break
        if not raw:
            continue

        year = None
        question = raw
        if "|" in raw:
            left, question = raw.split("|", 1)
            left, question = left.strip(), question.strip()
            if left.isdigit():
                year = int(left)

        for i, r in enumerate(engine.search(question, year=year), start=1):
            print(f"\n--- {i}  week ending {r['metadata']['week_ending']} "
                  f"(distance {r['distance']:.3f})")
            print(r["document"])
        print()