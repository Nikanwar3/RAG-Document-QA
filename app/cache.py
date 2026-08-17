import hashlib

import redis.asyncio as redis

from app.config import settings

redis_client = redis.from_url(settings.redis_url, decode_responses=True)


def build_cache_key(document_id: str, question: str) -> str:
    """Cache key scoped per document, hashing the question so we don't leak raw
    user text into Redis key names."""
    digest = hashlib.sha256(question.strip().lower().encode()).hexdigest()
    return f"qa:{document_id}:{digest}"


async def get_cached_answer(document_id: str, question: str) -> str | None:
    return await redis_client.get(build_cache_key(document_id, question))


async def set_cached_answer(document_id: str, question: str, answer: str) -> None:
    await redis_client.set(
        build_cache_key(document_id, question),
        answer,
        ex=settings.query_cache_ttl_seconds,
    )
