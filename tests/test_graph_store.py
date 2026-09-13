"""Unit tests for app.services.graph_store - the Neo4j-backed knowledge-graph
layer. Both the LLM extraction chain and the Neo4j driver are monkeypatched
directly (same pattern as test_qa_agent.py's _get_grader_chain), so none of
this touches a live LLM or a live Neo4j instance.
"""
from app.services import graph_store
from app.services.graph_store import ExtractedRelations, Relation


class _FakeExtractionChain:
    def __init__(self, relations):
        self._relations = relations

    def invoke(self, _inputs):
        return ExtractedRelations(relations=self._relations)


def test_extract_relations_caps_to_max_relations(monkeypatch):
    relations = [Relation(subject=f"s{i}", predicate="p", object=f"o{i}") for i in range(5)]
    monkeypatch.setattr(graph_store, "_get_extraction_chain", lambda: _FakeExtractionChain(relations))

    result = graph_store.extract_relations("some document text", max_relations=3)

    assert result == relations[:3]


def test_extract_relations_truncates_input_text(monkeypatch):
    seen = {}

    class _CapturingChain:
        def invoke(self, inputs):
            seen["text"] = inputs["text"]
            return ExtractedRelations(relations=[])

    monkeypatch.setattr(graph_store, "_get_extraction_chain", lambda: _CapturingChain())

    graph_store.extract_relations("x" * 20000)

    assert len(seen["text"]) == graph_store.MAX_EXTRACTION_CHARS


class _FakeResult:
    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)


class _FakeSession:
    def __init__(self, run_results=None):
        self.queries = []
        self._run_results = run_results or []

    def run(self, query, **params):
        self.queries.append((query, params))
        return _FakeResult(self._run_results)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeDriver:
    def __init__(self, session):
        self._session = session

    def session(self):
        return self._session


def test_store_relations_writes_one_merge_per_relation(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr(graph_store, "_get_driver", lambda: _FakeDriver(session))

    relations = [
        Relation(subject="Policyholder", predicate="must pay", object="Premium"),
        Relation(subject="Insurer", predicate="must pay", object="Claim"),
    ]
    graph_store.store_relations("ns-1", relations)

    assert len(session.queries) == 2
    _, params = session.queries[0]
    assert params == {"subject": "Policyholder", "object": "Premium", "predicate": "must pay", "namespace": "ns-1"}


def test_store_relations_skips_neo4j_entirely_when_no_relations(monkeypatch):
    def fail_get_driver():
        raise AssertionError("should not touch the driver for an empty relation list")

    monkeypatch.setattr(graph_store, "_get_driver", fail_get_driver)

    graph_store.store_relations("ns-1", [])  # must not raise


def test_get_relations_maps_records_to_relation_models(monkeypatch):
    records = [{"subject": "A", "predicate": "p", "object": "B"}]
    session = _FakeSession(run_results=records)
    monkeypatch.setattr(graph_store, "_get_driver", lambda: _FakeDriver(session))

    result = graph_store.get_relations("ns-1")

    assert result == [Relation(subject="A", predicate="p", object="B")]


def test_query_related_passes_entity_and_limit(monkeypatch):
    session = _FakeSession(run_results=[])
    monkeypatch.setattr(graph_store, "_get_driver", lambda: _FakeDriver(session))

    graph_store.query_related("ns-1", "Policyholder", limit=2)

    _, params = session.queries[0]
    assert params == {"namespace": "ns-1", "entity": "Policyholder", "limit": 2}
