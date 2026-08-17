import pytest


@pytest.mark.asyncio
async def test_register_and_login(client):
    payload = {"email": "test@example.com", "password": "supersecret1"}

    register_resp = await client.post("/auth/register", json=payload)
    assert register_resp.status_code == 201
    assert register_resp.json()["email"] == payload["email"]

    duplicate_resp = await client.post("/auth/register", json=payload)
    assert duplicate_resp.status_code == 400

    login_resp = await client.post("/auth/login", json=payload)
    assert login_resp.status_code == 200
    assert login_resp.json()["access_token"]

    bad_login_resp = await client.post("/auth/login", json={**payload, "password": "wrong-password"})
    assert bad_login_resp.status_code == 401


@pytest.mark.asyncio
async def test_documents_requires_auth(client):
    response = await client.get("/documents")
    assert response.status_code == 401
