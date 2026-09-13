"""Unit tests for the LangGraph corrective-RAG agent (app.services.qa_agent).

These test the graph's control flow — who gets called, how many times, and
in what order — by monkeypatching the node functions directly. None of this
touches Groq, Pinecone, sentence-transformers, or Neo4j: grade_node,
rewrite_node and graph_augment_node are the only nodes with an LLM call
inside them, and here they're replaced outright, same as
llm_client.generate_answer is monkeypatched in test_query.py rather than
hitting a real model. graph_augment_node's own tool-calling logic is
exercised directly in TestGraphAugmentNode below.
"""
from app.services import qa_agent


def _stub_graph_augment_not_augmented(state):
    return {"graph_augmented": False}


def test_relevant_on_first_try_never_rewrites(monkeypatch):
    monkeypatch.setattr(qa_agent, "retrieve_node", lambda s: {
        "context": "ctx", "retrieval_attempts": s.get("retrieval_attempts", 0) + 1,
    })
    monkeypatch.setattr(qa_agent, "grade_node", lambda s: {"relevant": True})
    monkeypatch.setattr(qa_agent, "rewrite_node", lambda s: pytest_fail_if_called())
    monkeypatch.setattr(qa_agent, "graph_augment_node", _stub_graph_augment_not_augmented)
    monkeypatch.setattr(qa_agent, "generate_node", lambda s: {"answer": "30 days"})

    result = qa_agent.answer_question("what is the notice period?", "ns-1")

    assert result == {
        "answer": "30 days",
        "retrieval_attempts": 1,
        "query_rewritten": False,
        "graph_augmented": False,
    }


def test_irrelevant_then_relevant_rewrites_once_and_retries_retrieval(monkeypatch):
    grade_calls = {"n": 0}

    def grade(state):
        grade_calls["n"] += 1
        return {"relevant": grade_calls["n"] >= 2}

    monkeypatch.setattr(qa_agent, "retrieve_node", lambda s: {
        "context": f"ctx-for-{s['question']}", "retrieval_attempts": s.get("retrieval_attempts", 0) + 1,
    })
    monkeypatch.setattr(qa_agent, "grade_node", grade)
    monkeypatch.setattr(qa_agent, "rewrite_node", lambda s: {
        "question": s["question"] + "-rewritten", "retries": s.get("retries", 0) + 1,
    })
    monkeypatch.setattr(qa_agent, "graph_augment_node", _stub_graph_augment_not_augmented)
    monkeypatch.setattr(qa_agent, "generate_node", lambda s: {"answer": f"FINAL:{s['context']}"})

    result = qa_agent.answer_question("vague question", "ns-1")

    assert grade_calls["n"] == 2
    assert result["retrieval_attempts"] == 2
    assert result["query_rewritten"] is True
    # The answer used context from the *second* (post-rewrite) retrieval.
    assert result["answer"] == "FINAL:ctx-for-vague question-rewritten"


def test_never_relevant_stops_after_max_retries_and_still_answers(monkeypatch):
    monkeypatch.setattr(qa_agent, "retrieve_node", lambda s: {
        "context": "ctx", "retrieval_attempts": s.get("retrieval_attempts", 0) + 1,
    })
    monkeypatch.setattr(qa_agent, "grade_node", lambda s: {"relevant": False})
    monkeypatch.setattr(qa_agent, "rewrite_node", lambda s: {
        "question": s["question"] + "-r", "retries": s.get("retries", 0) + 1,
    })
    monkeypatch.setattr(qa_agent, "graph_augment_node", _stub_graph_augment_not_augmented)
    monkeypatch.setattr(qa_agent, "generate_node", lambda s: {"answer": "Not mentioned in the document."})

    result = qa_agent.answer_question("unanswerable question", "ns-1")

    # One initial retrieval + one retry per allowed retry.
    assert result["retrieval_attempts"] == qa_agent.MAX_RETRIES + 1
    assert result["answer"] == "Not mentioned in the document."


class _FakeToolCallResponse:
    """Stands in for the AIMessage a tool-bound ChatGroq call would return."""

    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


class _FakeBoundLLM:
    """Stands in for `ChatGroq(...).bind_tools([...])` — graph_augment_node
    only ever calls `.invoke(messages)` on it."""

    def __init__(self, response):
        self._response = response

    def invoke(self, _messages):
        return self._response


class _FakeGraphAugmentLLM:
    """Stands in for the base ChatGroq instance `_get_graph_augment_llm()`
    returns — `.bind_tools([...])` is the only method graph_augment_node
    calls on it."""

    def __init__(self, response):
        self._response = response

    def bind_tools(self, _tools):
        return _FakeBoundLLM(self._response)


class TestGraphAugmentNode:
    """graph_augment_node's own tool-calling decision, isolated from the rest
    of the graph — mirrors how grade_node/rewrite_node's LLM calls are never
    hit directly in the control-flow tests above."""

    def test_no_tool_call_leaves_context_untouched(self, monkeypatch):
        monkeypatch.setattr(
            qa_agent, "_get_graph_augment_llm",
            lambda: _FakeGraphAugmentLLM(_FakeToolCallResponse(tool_calls=[])),
        )

        result = qa_agent.graph_augment_node(
            {"original_question": "q", "context": "some context", "namespace": "ns-1"}
        )

        assert result == {"graph_augmented": False}

    def test_tool_call_appends_graph_context(self, monkeypatch):
        monkeypatch.setattr(
            qa_agent, "_get_graph_augment_llm",
            lambda: _FakeGraphAugmentLLM(
                _FakeToolCallResponse(
                    tool_calls=[{"name": "lookup_related_entities", "args": {"entity": "Policyholder"}}]
                )
            ),
        )
        monkeypatch.setattr(
            qa_agent.graph_store, "query_related",
            lambda namespace, entity, limit=5: [
                qa_agent.graph_store.Relation(subject="Policyholder", predicate="is defined as", object="the Insured")
            ],
        )

        result = qa_agent.graph_augment_node(
            {"original_question": "who is the Policyholder?", "context": "some context", "namespace": "ns-1"}
        )

        assert result["graph_augmented"] is True
        assert "some context" in result["context"]
        assert "Policyholder --is defined as--> the Insured" in result["context"]


def test_route_after_grade():
    assert qa_agent._route_after_grade({"relevant": True, "retries": 0}) == "generate"
    assert qa_agent._route_after_grade({"relevant": False, "retries": 0}) == "rewrite"
    # Out of retries overrides an irrelevant grade — don't loop forever.
    assert qa_agent._route_after_grade({"relevant": False, "retries": qa_agent.MAX_RETRIES}) == "generate"


def pytest_fail_if_called():
    raise AssertionError("rewrite_node should not be called when the first retrieval is already relevant")
