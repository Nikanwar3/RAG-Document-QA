"""Unit tests for app.services.web_search — the Tavily-backed fallback used
by qa_agent.abstain_node when the document has no answer. Mocks requests
outright so these never make a real network call."""
import requests

from app.services import web_search


def test_search_web_returns_empty_without_api_key(monkeypatch):
    monkeypatch.setattr(web_search.settings, "tavily_api_key", None)

    assert web_search.search_web("some question") == []


def test_search_web_parses_results(monkeypatch):
    monkeypatch.setattr(web_search.settings, "tavily_api_key", "fake-key")

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    {"title": "T1", "url": "https://example.com/1", "content": "C1"},
                    {"title": "T2", "url": "https://example.com/2", "content": "C2"},
                ]
            }

    monkeypatch.setattr(web_search.requests, "post", lambda *a, **k: FakeResponse())

    results = web_search.search_web("some question")

    assert results == [
        {"title": "T1", "url": "https://example.com/1", "content": "C1"},
        {"title": "T2", "url": "https://example.com/2", "content": "C2"},
    ]


def test_search_web_returns_empty_on_request_error(monkeypatch):
    monkeypatch.setattr(web_search.settings, "tavily_api_key", "fake-key")

    def raise_error(*args, **kwargs):
        raise requests.RequestException("network error")

    monkeypatch.setattr(web_search.requests, "post", raise_error)

    assert web_search.search_web("some question") == []
