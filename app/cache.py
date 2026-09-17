import hashlib
import json

import redis.asyncio as redis

from app.config import settings

redis_client = redis.from_url(settings.redis_url, decode_responses=True)


def build_cache_key(document_id: str, question: str) -> str:
    """Cache key scoped per document, hashing the question so we don't leak raw
    user text into Redis key names."""
    digest = hashlib.sha256(question.strip().lower().encode()).hexdigest()
    return f"qa:{document_id}:{digest}"


async def get_cached_answer(document_id: str, question: str) -> dict | None:
    """Returns {"answer": str, "sources": list[dict]} on a cache hit, else None."""
    cached = await redis_client.get(build_cache_key(document_id, question))
    return json.loads(cached) if cached is not None else None


async def set_cached_answer(document_id: str, question: str, answer: str, sources: list[dict]) -> None:
    await redis_client.set(
        build_cache_key(document_id, question),
        json.dumps({"answer": answer, "sources": sources}),
        ex=settings.query_cache_ttl_seconds,
    )
