from __future__ import annotations
 
import json
from typing import Any, Callable
 
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
                "Use for questions about how much something was: revenue in a "
                "month, EBITDA for a quarter, total operating expenses for a "
                "year, margins over time. "
                "Do NOT use this to find the best or worst period -- use "
                "rank_periods for that. Do NOT use this to compare two periods "
                "-- use compare_periods for that. "
                f"Valid metrics: {_METRIC_LIST}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "One or more metric names from the valid list. "
                            "Ask for several at once when the question needs "
                            "them, e.g. ['revenue', 'ebitda']."
                        ),
                    },
                    "start_month": {
                        "type": "string",
                        "description": "First month, format YYYY-MM. Omit for the earliest available.",
                    },
                    "end_month": {
                        "type": "string",
                        "description": "Last month inclusive, format YYYY-MM. Omit for the latest available.",
                    },
                    "aggregate": {
                        "type": "string",
                        "enum": ["sum", "mean", "min", "max"],
                        "description": (
                            "How to combine months. Use sum for amounts, mean "
                            "for a typical month. Margin metrics are always "
                            "averaged regardless of this setting, because "
                            "summing a percentage is meaningless."
                        ),
                    },
                    "group_by": {
                        "type": "string",
                        "enum": ["month", "quarter", "year"],
                        "description": (
                            "Break the result down by period. Omit for a "
                            "single total across the whole range."
                        ),
                    },
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
                "Use for ANY superlative question: best month, worst quarter, "
                "top three months, lowest margin, which period was strongest, "
                "when did we perform worst. "
                "Set ascending to true for worst/lowest, false for best/highest. "
                f"Valid metrics: {_METRIC_LIST}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": "A single metric name from the valid list.",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "How many periods to return. Default 3.",
                    },
                    "ascending": {
                        "type": "boolean",
                        "description": (
                            "true = worst/lowest first. false = best/highest "
                            "first. Choose based on the question's wording."
                        ),
                    },
                    "group_by": {
                        "type": "string",
                        "enum": ["month", "quarter", "year"],
                        "description": "Rank by month, quarter or year. Default month.",
                    },
                    "start_month": {"type": "string", "description": "Optional range start, YYYY-MM."},
                    "end_month": {"type": "string", "description": "Optional range end, YYYY-MM."},
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
                "Compare two specific periods on one or more metrics, returning "
                "both values, the absolute change and the percentage change. "
                "Use for growth, decline, and any 'X versus Y' question: this "
                "year vs last year, Q1 vs Q2, first half vs second half, "
                "how much did we grow, are we doing better than last year. "
                "Each period is a month range, so a single month is a range "
                "where start equals end. "
                f"Valid metrics: {_METRIC_LIST}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "One or more metric names from the valid list.",
                    },
                    "period_a_start": {"type": "string", "description": "Earlier period start, YYYY-MM."},
                    "period_a_end": {"type": "string", "description": "Earlier period end, YYYY-MM."},
                    "period_b_start": {"type": "string", "description": "Later period start, YYYY-MM."},
                    "period_b_end": {"type": "string", "description": "Later period end, YYYY-MM."},
                },
                "required": [
                    "metrics",
                    "period_a_start",
                    "period_a_end",
                    "period_b_start",
                    "period_b_end",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_overview",
            "description": (
                "Find out what data is available: the date range covered and "
                "the full list of metrics. Takes no arguments. "
                "Use when the user asks what you can answer, or when you are "
                "unsure whether a metric or period exists before querying it."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
 
 
class ToolRegistry:
    def __init__(self, repository: FinancialsRepository | None = None):
        self.repository = repository or FinancialsRepository()
        self._handlers: dict[str, Callable[..., dict]] = {
            "query_financials": self.repository.query,
            "rank_periods": self.repository.rank_periods,
            "compare_periods": self.repository.compare_periods,
            "get_overview": self.repository.get_overview,
        }
 
    @property
    def schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS
 
    def system_context(self) -> str:
        """Facts injected into the system prompt.
 
        Stating the range up front prevents most out-of-range calls before they
        happen, which is cheaper than handling the rejection afterwards.
        """
        return (
            f"Monthly profit-and-loss data is available for "
            f"{self.repository.first_month} through {self.repository.last_month} "
            f"only. Available metrics: {_METRIC_LIST}. "
            "There is no balance sheet, cash flow, headcount, or customer data."
        )
 
    def execute(self, name: str, raw_arguments: str) -> dict:
        """Run one tool call. Never raises -- errors come back as data.
 
        A raised exception kills the turn and the user hears silence. A returned
        error lets the model recover on the next iteration: correct a metric
        name, fix a date, or explain that it cannot answer.
        """
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
 
        # Models sometimes send a bare string where an array is required.
        # Coercing is safer than rejecting: the intent is unambiguous.
        for key in ("metrics",):
            if isinstance(arguments.get(key), str):
                arguments[key] = [arguments[key]]
 
        try:
            return handler(**arguments)
        except TypeError as exc:
            return {"available": False, "error": f"Bad arguments for {name}: {exc}"}
        except Exception as exc:  # noqa: BLE001 - deliberate boundary catch
            return {"available": False, "error": f"{name} failed: {exc}"}
 
