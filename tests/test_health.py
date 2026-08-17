import pytest


@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"healthy", "degraded"}
    assert set(body.keys()) == {"status", "database", "redis"}
