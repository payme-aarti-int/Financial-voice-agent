
from __future__ import annotations
 
import json
from typing import Any, Callable
 
from app.financial.repository import RevenueRepository
 
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_revenue_range",
            "description": (
                "Get revenue for a single month or an inclusive range of months. "
                "Use for questions about how much revenue was earned in a period: "
                "a specific month, a quarter, a half-year, or the whole year. "
                "For a single month, pass the same value for both arguments. "
                "Do NOT use this to compare two periods -- use get_growth for that."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_month": {
                        "type": "string",
                        "description": "First month, format YYYY-MM (e.g. 2025-01).",
                    },
                    "end_month": {
                        "type": "string",
                        "description": (
                            "Last month, inclusive, format YYYY-MM. "
                            "Same as start_month for a single month."
                        ),
                    },
                },
                "required": ["start_month", "end_month"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_growth",
            "description": (
                "Compare revenue between two specific months and get the absolute "
                "and percentage change. Use for questions about growth, decline, "
                "increase, or how much things changed between two points in time. "
                "Do NOT use this to get a total for a period -- use "
                "get_revenue_range for that."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "from_month": {
                        "type": "string",
                        "description": "Earlier month, format YYYY-MM.",
                    },
                    "to_month": {
                        "type": "string",
                        "description": "Later month, format YYYY-MM.",
                    },
                },
                "required": ["from_month", "to_month"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_summary",
            "description": (
                "Get an overview of all available revenue data: total, monthly "
                "average, best and worst month, and the overall trend. Use for "
                "broad questions like how revenue is doing overall, what the "
                "best month was, or whether revenue is growing. Takes no "
                "arguments and always covers the full available period."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
 
 
class ToolRegistry:
 
    def __init__(self, repository: RevenueRepository | None = None):
        self.repository = repository or RevenueRepository()
        self._handlers: dict[str, Callable[..., dict]] = {
            "get_revenue_range": self.repository.get_revenue_range,
            "get_growth": self.repository.get_growth,
            "get_summary": self.repository.get_summary,
        }
 
    @property
    def schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS
 
    def system_context(self) -> str:
        
        return (
            f"Revenue data is available for {self.repository.first_month} "
            f"through {self.repository.last_month} only."
        )
 
    def execute(self, name: str, raw_arguments: str) -> dict:
       
        handler = self._handlers.get(name)
        if handler is None:
            return {"error": f"Unknown tool '{name}'", "available": False}
 
        try:
            arguments = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError as exc:
            return {"error": f"Malformed tool arguments: {exc}", "available": False}
 
        if not isinstance(arguments, dict):
            return {"error": "Tool arguments must be an object", "available": False}
 
        try:
            return handler(**arguments)
        except TypeError as exc:
            # Wrong or missing argument names -- a schema-description problem,
            # so surface it clearly rather than swallowing it.
            return {"error": f"Bad arguments for {name}: {exc}", "available": False}
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all at boundary
            return {"error": f"{name} failed: {exc}", "available": False}
 
