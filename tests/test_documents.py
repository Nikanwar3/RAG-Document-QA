import pytest


async def _auth_headers(client, email="doc-user@example.com", password="supersecret1"):
    await client.post("/auth/register", json={"email": email, "password": password})
    login = await client.post("/auth/login", json={"email": email, "password": password})
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_create_and_fetch_document(client, monkeypatch):
    from app.routers import documents as documents_router

    # Don't require a real Celery broker for this test — the ingest task's
    # correctness is covered separately; here we're testing the API contract.
    monkeypatch.setattr(documents_router.ingest_document_task, "delay", lambda *a, **kw: None)

    headers = await _auth_headers(client)

    create_resp = await client.post(
        "/documents",
        json={"source_url": "https://example.com/policy.pdf", "filename": "policy.pdf"},
        headers=headers,
    )
    assert create_resp.status_code == 202
    document = create_resp.json()
    assert document["status"] == "pending"

    get_resp = await client.get(f"/documents/{document['id']}", headers=headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == document["id"]

    list_resp = await client.get("/documents", headers=headers)
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1


@pytest.mark.asyncio
async def test_get_document_not_found(client):
    headers = await _auth_headers(client, email="doc-user2@example.com")
    response = await client.get("/documents/00000000-0000-0000-0000-000000000000", headers=headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_documents_are_scoped_per_user(client, monkeypatch):
    from app.routers import documents as documents_router

    monkeypatch.setattr(documents_router.ingest_document_task, "delay", lambda *a, **kw: None)

    owner_headers = await _auth_headers(client, email="owner@example.com")
    other_headers = await _auth_headers(client, email="other@example.com")

    create_resp = await client.post(
        "/documents", json={"source_url": "https://example.com/a.pdf"}, headers=owner_headers
    )
    document_id = create_resp.json()["id"]

    response = await client.get(f"/documents/{document_id}", headers=other_headers)
    assert response.status_code == 404
