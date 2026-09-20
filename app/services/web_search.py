import requests

from app.config import settings

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Web search fallback for questions the uploaded document can't answer.

    Returns [] on missing credentials, network errors, or a non-2xx response
    rather than raising — a failed fallback should let the caller abstain,
    not crash the request."""
    if not settings.tavily_api_key:
        return []

    try:
        response = requests.post(
            TAVILY_SEARCH_URL,
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
            },
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException:
        return []

    results = response.json().get("results", [])
    return [
        {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content": result.get("content", ""),
        }
        for result in results
    ]
