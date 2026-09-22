from __future__ import annotations

import json
from pathlib import Path

GRAPH_PATH = Path("data/knowledge_graph.json")


class GraphQuery:

    def __init__(self, graph_path: str | Path = GRAPH_PATH):
        self.graph_path = Path(graph_path)
        self._nodes: dict = {}
        self._edges: list = []
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if not self.graph_path.exists():
            self._nodes = {}
            self._edges = []
            self._loaded = True
            return

        with open(self.graph_path, encoding="utf-8") as f:
            data = json.load(f)

        self._nodes = {n["id"]: n["attrs"] for n in data.get("nodes", [])}
        self._edges = data.get("edges", [])
        self._loaded = True

    def _keyword_match(self, text: str, keywords: list[str]) -> bool:
        text_lower = text.lower()
        return any(k.lower() in text_lower for k in keywords)

    def search(self, question: str, top_k: int = 10) -> dict:
        
        self._load()

        if not self._nodes:
            return {
                "available": False,
                "error": "Knowledge graph not built yet. Run graph_builder first.",
                "nodes": [],
                "edges": [],
            }

        stop_words = {"what", "is", "are", "the", "a", "an", "of", "in",
                      "for", "to", "how", "does", "do", "and", "or", "with",
                      "about", "tell", "me", "explain", "describe", "which"}
        keywords = [
            w.strip("?.,")
            for w in question.lower().split()
            if w.strip("?.,") not in stop_words and len(w) > 2
        ]

        matching_nodes = []
        for node_id, attrs in self._nodes.items():
            if self._keyword_match(node_id, keywords):
                matching_nodes.append({
                    "id": node_id,
                    "type": attrs.get("type", "unknown"),
                    "source": attrs.get("source", ""),
                    "h1": attrs.get("h1", ""),
                    "h2": attrs.get("h2", ""),
                })

        matched_ids = {n["id"] for n in matching_nodes}
        matching_edges = []
        for edge in self._edges:
            if edge["from"] in matched_ids or edge["to"] in matched_ids:
                matching_edges.append({
                    "from": edge["from"],
                    "relation": edge["relation"],
                    "to": edge["to"],
                    "source": edge.get("source", ""),
                })

        for edge in self._edges:
            if self._keyword_match(edge["relation"], keywords):
                if edge not in matching_edges:
                    matching_edges.append({
                        "from": edge["from"],
                        "relation": edge["relation"],
                        "to": edge["to"],
                        "source": edge.get("source", ""),
                    })

        matching_nodes = matching_nodes[:top_k]
        matching_edges = matching_edges[:top_k * 2]

        summary_lines = []
        for e in matching_edges:
            summary_lines.append(
                f"{e['from']} --[{e['relation']}]--> {e['to']}"
            )

        return {
            "available": True,
            "question": question,
            "keywords": keywords,
            "nodes_found": len(matching_nodes),
            "edges_found": len(matching_edges),
            "summary": "\n".join(summary_lines),
            "nodes": matching_nodes,
            "edges": matching_edges,
        }


if __name__ == "__main__":
    import sys
    import app

    question = " ".join(sys.argv[1:]) or "NBFC licensing requirements"
    engine = GraphQuery()
    result = engine.search(question)

    print(f"Question: {result['question']}")
    print(f"Keywords: {result['keywords']}")
    print(f"Nodes found: {result['nodes_found']}")
    print(f"Edges found: {result['edges_found']}")
    print()
    print("Relationships:")
    print(result["summary"])