"""Unit tests for app.services.vector_store's Pinecone-facing helpers.

_get_index/_get_model are monkeypatched so these never touch a real Pinecone
index or download a SentenceTransformer model, matching the same
import-safety reasoning documented on those lazy singletons.
"""
from app.services import vector_store


class _FakeVector:
    """Stands in for the numpy array SentenceTransformer.encode() returns —
    real code calls .tolist() on it before handing it to Pinecone."""

    def tolist(self):
        return [0.0]


class _FakeIndex:
    def __init__(self, matches):
        self._matches = matches

    def query(self, vector, top_k, include_metadata, namespace):
        return {"matches": self._matches[:top_k]}


def _fake_model():
    return type("M", (), {"encode": lambda self, texts: [_FakeVector()]})()


def test_retrieve_chunks_keeps_id_text_and_score(monkeypatch):
    matches = [
        {"id": "doc-chunk-0", "score": 0.91, "metadata": {"text": "grace period is 30 days"}},
        {"id": "doc-chunk-2", "score": 0.77, "metadata": {"text": "premium is due monthly"}},
    ]
    monkeypatch.setattr(vector_store, "_get_model", _fake_model)
    monkeypatch.setattr(vector_store, "_get_index", lambda: _FakeIndex(matches))

    chunks = vector_store.retrieve_chunks("what is the grace period?", "ns-1", top_k=2)

    assert chunks == [
        {"chunk_id": "doc-chunk-0", "text": "grace period is 30 days", "score": 0.91},
        {"chunk_id": "doc-chunk-2", "text": "premium is due monthly", "score": 0.77},
    ]


def test_query_top_chunks_still_returns_joined_text(monkeypatch):
    """query_top_chunks (used by the legacy /hackrx/run endpoint) keeps its
    original plain-text contract even though retrieve_chunks now shares the
    same underlying _query_index call."""
    matches = [
        {"id": "doc-chunk-0", "score": 0.91, "metadata": {"text": "grace period is 30 days"}},
        {"id": "doc-chunk-1", "score": 0.5, "metadata": {"text": "premium is due monthly"}},
    ]
    monkeypatch.setattr(vector_store, "_get_model", _fake_model)
    monkeypatch.setattr(vector_store, "_get_index", lambda: _FakeIndex(matches))

    context = vector_store.query_top_chunks("what is the grace period?", "ns-1", top_k=2)

    assert context == "grace period is 30 days\npremium is due monthly"
