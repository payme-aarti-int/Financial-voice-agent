"""Web search via Tavily (https://tavily.com) for current information that
is not in the local P&L / RBI datasets -- recent RBI circulars, regulatory
updates, current market rates.

Reads TAVILY_API_KEY from the environment. Like every other tool handler,
`search_web` never raises: a missing key, network failure or API error comes
back as {"available": False, "error": ...} so the agent can tell the user
plainly rather than crash mid-turn.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

DEFAULT_MAX_RESULTS = 3
MAX_RESULTS_CAP = 10
# Keep each passage short: the result is read aloud via the LLM, and long
# snippets blow up the prompt for the small local models.
SNIPPET_CHARS = 600

_client = None


def _get_client():
    """Lazily build one TavilyClient; re-reads the key if it was unset before."""
    global _client
    if _client is None:
        from tavily import TavilyClient

        _client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return _client


def search_web(question: str, max_results: int = DEFAULT_MAX_RESULTS) -> dict:
    question = (question or "").strip()
    if not question:
        return {"available": False, "error": "search_web needs a non-empty question"}

    if not os.getenv("TAVILY_API_KEY", "").strip():
        return {
            "available": False,
            "error": "TAVILY_API_KEY not set -- web search is disabled. Add it to .env to enable.",
        }

    try:
        max_results = max(1, min(int(max_results or DEFAULT_MAX_RESULTS), MAX_RESULTS_CAP))
    except (TypeError, ValueError):
        max_results = DEFAULT_MAX_RESULTS

    try:
        response = _get_client().search(
            query=question,
            max_results=max_results,
            search_depth="basic",
            include_answer=True,
        )
    except Exception as exc:  # noqa: BLE001 - network/auth/quota: report, don't raise
        log.warning("search_web failed for %r: %s", question, exc)
        return {"available": False, "error": f"search_web failed: {exc}"}

    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": (r.get("content") or "")[:SNIPPET_CHARS],
            "published": r.get("published_date"),
        }
        for r in (response.get("results") or [])
    ]
    if not results:
        return {"available": False, "reason": "No web results found."}

    out: dict = {
        "available": True,
        "results": results,
        "note": (
            "These are web search snippets, not verified data. Attribute them to "
            "their source and prefer the local tools for anything they cover."
        ),
    }
    if response.get("answer"):
        out["summary"] = response["answer"]
    return out
