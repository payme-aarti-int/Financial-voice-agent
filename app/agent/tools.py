from __future__ import annotations

import json
from typing import Any, Callable

from app import observability
from app.financial.financials import METRICS, FinancialsRepository

_METRIC_LIST = ", ".join(sorted(METRICS))

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_financials",
            "description": (
                "Get the value of one or more financial metrics over a period, "
                "optionally broken down by month, quarter or year. "
                f"Valid metrics: {_METRIC_LIST}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metrics": {"type": "array", "items": {"type": "string"}},
                    "start_month": {"type": "string"},
                    "end_month": {"type": "string"},
                    "aggregate": {"type": "string", "enum": ["sum", "mean", "min", "max"]},
                    "group_by": {"type": "string", "enum": ["month", "quarter", "year"]},
                },
                "required": ["metrics"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_periods",
            "description": (
                "Find the best or worst periods for a single metric. "
                f"Valid metrics: {_METRIC_LIST}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string"},
                    "top_n": {"type": "integer"},
                    "ascending": {"type": "boolean"},
                    "group_by": {"type": "string", "enum": ["month", "quarter", "year"]},
                    "start_month": {"type": "string"},
                    "end_month": {"type": "string"},
                },
                "required": ["metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_periods",
            "description": (
                "Compare two specific periods on one or more metrics. "
                f"Valid metrics: {_METRIC_LIST}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metrics": {"type": "array", "items": {"type": "string"}},
                    "period_a_start": {"type": "string"},
                    "period_a_end": {"type": "string"},
                    "period_b_start": {"type": "string"},
                    "period_b_end": {"type": "string"},
                },
                "required": ["metrics", "period_a_start", "period_a_end", "period_b_start", "period_b_end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_rbi_data",
            "description": (
                "Search RBI weekly reserve money data (July 2001 to August 2020). "
                "Use for contextual questions about currency, reserve money, RBI claims. "
                "DO NOT use for exact lookups, superlatives, or arithmetic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "year": {"type": "integer"},
                    "top_k": {"type": "integer"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_graph",
            "description": (
                "Search the knowledge graph for relationships between financial "
                "regulatory concepts. Use for questions about what rules apply "
                "to an entity, what an entity requires, or how concepts relate. "
                "Examples: NBFC licensing requirements, regulations for cooperative banks. "
                "DO NOT use for numeric lookups or time-series data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_overview",
            "description": "Find out what data is available. Takes no arguments.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class ToolRegistry:
    def __init__(self, repository: FinancialsRepository | None = None):
        self.repository = repository or FinancialsRepository()

        try:
            from retrieval.query import RBIQueryEngine
            self._rbi = RBIQueryEngine()
        except Exception:
            self._rbi = None

        try:
            from retrieval.knowledge_graph.graph_query import GraphQuery
            self._graph = GraphQuery()
        except Exception:
            self._graph = None

        self._handlers: dict[str, Callable[..., dict]] = {
            "query_financials": self.repository.query,
            "rank_periods": self.repository.rank_periods,
            "compare_periods": self.repository.compare_periods,
            "get_overview": self.repository.get_overview,
            "search_rbi_data": self._search_rbi,
            "search_knowledge_graph": self._search_graph,
        }

    def _search_rbi(self, question: str, year: int | None = None, top_k: int = 3) -> dict:
        try:
            if self._rbi is None:
                return {"available": False, "error": "RBI index not available"}
            results = self._rbi.search(question, top_k=top_k, year=year)
            if not results:
                return {"available": False, "reason": "No matching weeks found."}
            return {
                "available": True,
                "results": [
                    {
                        "week_ending": r["metadata"]["week_ending"],
                        "passage": r["document"],
                        "distance": round(r["distance"], 3),
                    }
                    for r in results
                ],
                "note": "These are retrieved passages, not exact lookups.",
            }
        except Exception as exc:
            return {"available": False, "error": f"search_rbi_data failed: {exc}"}

    def _search_graph(self, question: str, top_k: int = 10) -> dict:
        try:
            if self._graph is None:
                return {"available": False, "error": "Knowledge graph not available"}
            result = self._graph.search(question, top_k=top_k)
            if not result.get("edges"):
                return {"available": False, "reason": "No relationships found."}
            return {
                "available": True,
                "relationships": result["summary"],
                "nodes_found": result["nodes_found"],
                "edges_found": result["edges_found"],
                "note": (
                    "These are graph relationships from regulatory documents. "
                    "Use them to explain how concepts relate, not as precise numeric facts."
                ),
            }
        except Exception as exc:
            return {"available": False, "error": f"search_knowledge_graph failed: {exc}"}

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def system_context(self) -> str:
        return (
            f"Monthly profit-and-loss data is available for "
            f"{self.repository.first_month} through {self.repository.last_month}. "
            f"Available P&L metrics: {_METRIC_LIST}. "
            "There is no balance sheet, cash flow, headcount, or customer data. "
            "RBI weekly reserve money data is also available (July 2001 to August 2020) "
            "via search_rbi_data. "
            "A knowledge graph of regulatory relationships is also available "
            "via search_knowledge_graph for questions about rules and requirements."
        )

    def execute(self, name: str, raw_arguments: str) -> dict:
        handler = self._handlers.get(name)
        if handler is None:
            return {
                "available": False,
                "error": f"Unknown tool '{name}'",
                "valid_tools": sorted(self._handlers),
            }

        try:
            arguments = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError as exc:
            return {"available": False, "error": f"Malformed tool arguments: {exc}"}

        if not isinstance(arguments, dict):
            return {"available": False, "error": "Tool arguments must be an object"}

        for key in ("metrics",):
            if isinstance(arguments.get(key), str):
                arguments[key] = [arguments[key]]

        with observability.tool_span(name, arguments) as set_output:
            try:
                result = handler(**arguments)
            except TypeError as exc:
                result = {"available": False, "error": f"Bad arguments for {name}: {exc}"}
            except Exception as exc:
                result = {"available": False, "error": f"{name} failed: {exc}"}
            set_output(result)
            return result
