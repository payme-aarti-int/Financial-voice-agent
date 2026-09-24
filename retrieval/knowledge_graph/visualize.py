"""Render the knowledge graph (or a query-matched slice of it) with networkx.

Usage:
    python -m retrieval.knowledge_graph.visualize
        Draws the full graph to data/knowledge_graph.png.

    python -m retrieval.knowledge_graph.visualize "NBFC licensing requirements"
        Draws only the nodes/edges GraphQuery would return for that question,
        to data/knowledge_graph_query.png -- useful for seeing exactly what
        search_knowledge_graph hands the agent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from retrieval.knowledge_graph.graph_query import GRAPH_PATH, GraphQuery

NODE_COLORS = {
    "chapter": "#e74c3c",
    "section": "#f39c12",
    "concept": "#3498db",
}
DEFAULT_COLOR = "#95a5a6"


def load_graph(graph_path: str | Path = GRAPH_PATH) -> nx.DiGraph:
    """Load the full graph straight from the JSON GraphQuery/graph_builder share."""
    path = Path(graph_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m retrieval.knowledge_graph.graph_builder"
        )

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    graph = nx.DiGraph()
    for node in data.get("nodes", []):
        graph.add_node(node["id"], **node.get("attrs", {}))
    for edge in data.get("edges", []):
        graph.add_edge(edge["from"], edge["to"], relation=edge["relation"])
    return graph


def build_subgraph(question: str, top_k: int = 10) -> nx.DiGraph:
    result = GraphQuery().search(question, top_k=top_k)
    graph = nx.DiGraph()
    for node in result["nodes"]:
        graph.add_node(node["id"], type=node["type"])
    for edge in result["edges"]:
        graph.add_node(edge["from"], type=graph.nodes.get(edge["from"], {}).get("type", "concept"))
        graph.add_node(edge["to"], type=graph.nodes.get(edge["to"], {}).get("type", "concept"))
        graph.add_edge(edge["from"], edge["to"], relation=edge["relation"])
    return graph


def draw(graph: nx.DiGraph, out_path: str | Path, title: str) -> Path:
    out_path = Path(out_path)

    if graph.number_of_nodes() == 0:
        raise ValueError("graph is empty -- nothing to draw")

    colors = [
        NODE_COLORS.get(graph.nodes[n].get("type", ""), DEFAULT_COLOR)
        for n in graph.nodes
    ]

    fig_size = max(10, graph.number_of_nodes() ** 0.5 * 2.2)
    plt.figure(figsize=(fig_size, fig_size))

    layout = nx.spring_layout(graph, seed=42, k=1.4 / (graph.number_of_nodes() ** 0.4))

    nx.draw_networkx_nodes(graph, layout, node_color=colors, node_size=900, alpha=0.9)
    nx.draw_networkx_labels(graph, layout, font_size=8)
    nx.draw_networkx_edges(
        graph, layout, arrows=True, arrowsize=12, alpha=0.5, connectionstyle="arc3,rad=0.05"
    )
    edge_labels = {(u, v): d["relation"] for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, layout, edge_labels=edge_labels, font_size=6)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=node_type,
                   markerfacecolor=color, markersize=10)
        for node_type, color in NODE_COLORS.items()
    ]
    plt.legend(handles=legend_handles, loc="upper left")
    plt.title(f"{title}  ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")
    plt.axis("off")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


if __name__ == "__main__":
    import app  # noqa: F401  -- triggers load_dotenv() + logging setup

    question = " ".join(sys.argv[1:]).strip()

    if question:
        graph = build_subgraph(question)
        out = draw(graph, "data/knowledge_graph_query.png", f'Query: "{question}"')
    else:
        graph = load_graph()
        out = draw(graph, "data/knowledge_graph.png", "Full knowledge graph")

    print(f"Nodes: {graph.number_of_nodes()} | Edges: {graph.number_of_edges()}")
    print(f"Saved to {out}")
