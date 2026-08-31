import uuid

import pytest


async def _auth_headers(client, email="query-user@example.com", password="supersecret1"):
    await client.post("/auth/register", json={"email": email, "password": password})
    login = await client.post("/auth/login", json={"email": email, "password": password})
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def _create_ready_document(client, headers, db_session, monkeypatch, source_url="https://example.com/policy.pdf"):
    from app.models import Document, DocumentStatus
    from app.routers import documents as documents_router

    monkeypatch.setattr(documents_router.ingest_document_task, "delay", lambda *a, **kw: None)

    create_resp = await client.post("/documents", json={"source_url": source_url}, headers=headers)
    document_id = create_resp.json()["id"]

    async with db_session() as session:
        document = await session.get(Document, uuid.UUID(document_id))
        document.status = DocumentStatus.READY
        await session.commit()

    return document_id


@pytest.mark.asyncio
async def test_query_flow_with_cache(client, db_session, monkeypatch):
    from app.routers import query as query_router

    # Fake Redis: exercises the same cache-hit/cache-miss branching in the
    # route without requiring a live Redis instance for this unit test.
    fake_cache = {}

    async def fake_get_cached_answer(document_id, question):
        return fake_cache.get((document_id, question))

    async def fake_set_cached_answer(document_id, question, answer):
        fake_cache[(document_id, question)] = answer

    monkeypatch.setattr(query_router, "get_cached_answer", fake_get_cached_answer)
    monkeypatch.setattr(query_router, "set_cached_answer", fake_set_cached_answer)
    monkeypatch.setattr(
        query_router.vector_store, "query_top_chunks", lambda q, ns, top_k=3: "grace period is 30 days"
    )
    monkeypatch.setattr(query_router.llm_client, "generate_answer", lambda q, c: "30 days")

    headers = await _auth_headers(client)
    document_id = await _create_ready_document(client, headers, db_session, monkeypatch)

    payload = {"document_id": document_id, "question": "What is the grace period?"}

    first = await client.post("/query", json=payload, headers=headers)
    assert first.status_code == 200
    body = first.json()
    assert body["answer"] == "30 days"
    assert body["cache_hit"] is False

    second = await client.post("/query", json=payload, headers=headers)
    assert second.status_code == 200
    assert second.json()["cache_hit"] is True


@pytest.mark.asyncio
async def test_query_agent_flow_with_cache(client, db_session, monkeypatch):
    from app.routers import query as query_router

    fake_cache = {}

    async def fake_get_cached_answer(document_id, question):
        return fake_cache.get((document_id, question))

    async def fake_set_cached_answer(document_id, question, answer):
        fake_cache[(document_id, question)] = answer

    monkeypatch.setattr(query_router, "get_cached_answer", fake_get_cached_answer)
    monkeypatch.setattr(query_router, "set_cached_answer", fake_set_cached_answer)
    # The graph itself is unit-tested in test_qa_agent.py — here we only need
    # to confirm the route wires qa_agent.answer_question's dict through to
    # QueryAgentResponse correctly, so the agent entry point is monkeypatched
    # as a single unit rather than its four internal nodes.
    monkeypatch.setattr(
        query_router.qa_agent,
        "answer_question",
        lambda q, ns: {"answer": "30 days", "retrieval_attempts": 2, "query_rewritten": True},
    )

    headers = await _auth_headers(client, email="agent-query-user@example.com")
    document_id = await _create_ready_document(client, headers, db_session, monkeypatch)

    payload = {"document_id": document_id, "question": "What is the grace period?"}

    first = await client.post("/query/agent", json=payload, headers=headers)
    assert first.status_code == 200
    body = first.json()
    assert body["answer"] == "30 days"
    assert body["cache_hit"] is False
    assert body["retrieval_attempts"] == 2
    assert body["query_rewritten"] is True

    second = await client.post("/query/agent", json=payload, headers=headers)
    assert second.status_code == 200
    assert second.json()["cache_hit"] is True


@pytest.mark.asyncio
async def test_query_rejects_document_not_ready(client, monkeypatch):
    from app.routers import documents as documents_router

    monkeypatch.setattr(documents_router.ingest_document_task, "delay", lambda *a, **kw: None)

    headers = await _auth_headers(client, email="notready@example.com")
    create_resp = await client.post(
        "/documents", json={"source_url": "https://example.com/x.pdf"}, headers=headers
    )
    document_id = create_resp.json()["id"]

    response = await client.post(
        "/query", json={"document_id": document_id, "question": "anything?"}, headers=headers
    )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_query_unknown_document(client):
    headers = await _auth_headers(client, email="unknowndoc@example.com")
    response = await client.post(
        "/query",
        json={"document_id": str(uuid.uuid4()), "question": "anything?"},
        headers=headers,
    )
    assert response.status_code == 404
