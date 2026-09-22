from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from app.llm.client import LLMClient

GRAPH_PATH = Path("data/knowledge_graph.json")


def _extract_entities_relations(client, chunk: dict) -> dict:
    h1 = chunk.get("h1") or "Unknown"
    h2 = chunk.get("h2") or "Unknown"
    text = chunk.get("text", "")[:1500]

    prompt = (
        "Extract entities and relationships from this financial regulatory text.\n"
        "Return ONLY a JSON object with this exact structure, no other format:\n"
        '{"entities": ["NBFC", "RBI", "Net Owned Fund"], "relations": [{"from": "NBFC", "relation": "regulated_by", "to": "RBI"}]}\n'
        "entities must be a flat list of strings.\n"
        "relations must use keys: from, relation, to.\n"
        f"Section: {h1} > {h2}\n"
        f"Text:\n{text}"
    )

    try:
        completion = client.complete(
            [{"role": "user", "content": prompt}]
        )
        raw = completion.text.strip()

        if not raw:
            return {"entities": [], "relations": [], "error": "empty response"}

        # strip markdown fences
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break

        data = json.loads(raw.strip())

        # normalise entities — handle both flat strings and dicts
        raw_entities = data.get("entities", [])
        entities = []
        for e in raw_entities:
            if isinstance(e, str):
                entities.append(e)
            elif isinstance(e, dict):
                # model returned {"type": "X", "value": "Y"}
                val = e.get("value") or e.get("name") or e.get("entity", "")
                if val:
                    entities.append(str(val))

        # normalise relations — handle both formats
        raw_relations = data.get("relations", [])
        relations = []
        for r in raw_relations:
            if isinstance(r, dict):
                # our format: from/relation/to
                if "from" in r and "to" in r:
                    relations.append({
                        "from": str(r["from"]),
                        "relation": str(r.get("relation", "related_to")),
                        "to": str(r["to"]),
                    })
                # model format: subject/predicate/object
                elif "subject" in r and "object" in r:
                    relations.append({
                        "from": str(r["subject"]),
                        "relation": str(r.get("predicate", "related_to")),
                        "to": str(r["object"]),
                    })

        return {"entities": entities, "relations": relations}

    except Exception as exc:
        return {"entities": [], "relations": [], "error": str(exc)}


def build_graph(extracted_json_path: str | Path) -> nx.DiGraph:
    path = Path(extracted_json_path)
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)

    source_file = doc["source_file"]
    chunks = doc.get("pages") or doc.get("rows") or []
    print(f"Building graph for {source_file} — {len(chunks)} chunks")

    G = nx.DiGraph()

    if GRAPH_PATH.exists():
        with open(GRAPH_PATH, encoding="utf-8") as f:
            existing = json.load(f)
        for node in existing.get("nodes", []):
            G.add_node(node["id"], **node.get("attrs", {}))
        for edge in existing.get("edges", []):
            G.add_edge(edge["from"], edge["to"], relation=edge["relation"], source=edge.get("source", ""))
        print(f"Loaded existing graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    with LLMClient() as client:
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            if not text.strip():
                continue

            h1 = chunk.get("h1") or ""
            h2 = chunk.get("h2") or ""
            print(f"  chunk {i+1}/{len(chunks)} | {h2 or h1 or 'no heading'}")

            if h1:
                G.add_node(h1, type="chapter", source=source_file)
            if h2:
                G.add_node(h2, type="section", source=source_file)
                if h1:
                    G.add_edge(h1, h2, relation="contains", source=source_file)

            result = _extract_entities_relations(client, chunk)

            for entity in result.get("entities", []):
                if entity and len(entity) > 1:
                    G.add_node(entity, type="concept", source=source_file, h1=h1, h2=h2)
                    if h2:
                        G.add_edge(h2, entity, relation="mentions", source=source_file)
                    elif h1:
                        G.add_edge(h1, entity, relation="mentions", source=source_file)

            for rel in result.get("relations", []):
                frm = rel.get("from", "").strip()
                to = rel.get("to", "").strip()
                relation = rel.get("relation", "related_to").strip()
                if frm and to and len(frm) > 1 and len(to) > 1:
                    G.add_node(frm, type="concept", source=source_file)
                    G.add_node(to, type="concept", source=source_file)
                    G.add_edge(frm, to, relation=relation, source=source_file)

    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    graph_data = {
        "nodes": [{"id": n, "attrs": G.nodes[n]} for n in G.nodes],
        "edges": [{"from": u, "to": v, "relation": G[u][v]["relation"], "source": G[u][v].get("source", "")} for u, v in G.edges],
    }
    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    print(f"\nGraph saved to {GRAPH_PATH}")
    print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
    return G


if __name__ == "__main__":
    import sys
    import app
    path = sys.argv[1] if len(sys.argv) > 1 else "data/extracted/data.json"
    build_graph(path)
