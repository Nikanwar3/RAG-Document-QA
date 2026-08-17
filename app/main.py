from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from app.cache import redis_client
from app.database import Base, engine
from app.routers import auth, documents, hackrx, query
from app.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Dev convenience: create tables if they don't exist yet. Real deployments
    # should run `alembic upgrade head` instead (see docker-compose.yml).
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="RAG Document QA", version="2.0.0", lifespan=lifespan)

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(hackrx.router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    db_status = "ok"
    redis_status = "ok"

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        db_status = "unavailable"

    try:
        await redis_client.ping()
    except Exception:
        redis_status = "unavailable"

    overall = "healthy" if db_status == "ok" and redis_status == "ok" else "degraded"
    return HealthResponse(status=overall, database=db_status, redis=redis_status)
