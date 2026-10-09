"""Tavily-backed search_web tool: registered alongside the local data tools,
degrades to {"available": False} without a key or on API failure, and never
raises. No network calls -- the Tavily client is stubbed.
"""

import json

import pytest

from app.agent import tools as agent_tools
from app.tools import web_search


@pytest.fixture(autouse=True)
def _fresh_client(monkeypatch):
    monkeypatch.setattr(web_search, "_client", None)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)


class _StubClient:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def test_missing_key_is_reported_not_raised():
    result = web_search.search_web("latest RBI repo rate")
    assert result["available"] is False
    assert "TAVILY_API_KEY" in result["error"]


def test_empty_question_rejected(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    assert web_search.search_web("   ")["available"] is False


def test_success_shapes_results_and_clamps_max_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    stub = _StubClient(
        response={
            "answer": "The repo rate is 6.5 percent.",
            "results": [
                {"title": "RBI", "url": "https://rbi.org.in/x", "content": "x" * 2000, "score": 0.9},
            ],
        }
    )
    monkeypatch.setattr(web_search, "_get_client", lambda: stub)

    result = web_search.search_web("latest RBI repo rate", max_results=50)

    assert result["available"] is True
    assert result["summary"] == "The repo rate is 6.5 percent."
    assert result["results"][0]["url"] == "https://rbi.org.in/x"
    assert len(result["results"][0]["snippet"]) == web_search.SNIPPET_CHARS
    assert stub.calls[0]["max_results"] == web_search.MAX_RESULTS_CAP
    assert stub.calls[0]["query"] == "latest RBI repo rate"


def test_no_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    monkeypatch.setattr(web_search, "_get_client", lambda: _StubClient(response={"results": []}))
    result = web_search.search_web("something obscure")
    assert result["available"] is False
    assert "reason" in result


def test_api_error_is_reported_not_raised(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    monkeypatch.setattr(
        web_search, "_get_client", lambda: _StubClient(error=RuntimeError("quota exceeded"))
    )
    result = web_search.search_web("anything")
    assert result["available"] is False
    assert "quota exceeded" in result["error"]


# ---- registry integration --------------------------------------------------


def test_registry_registers_search_web_and_keeps_existing_tools():
    registry = agent_tools.ToolRegistry()
    names = {s["function"]["name"] for s in registry.schemas}
    assert {"query_financials", "rank_periods", "compare_periods", "get_overview",
            "search_rbi_data", "search_knowledge_graph", "search_web"} <= names
    assert "search_web" in registry._handlers
    assert "search_web" in registry.system_context()


def test_registry_executes_search_web_without_key():
    registry = agent_tools.ToolRegistry()
    result = registry.execute("search_web", json.dumps({"question": "test"}))
    assert result["available"] is False
    assert "TAVILY_API_KEY" in result["error"]
