
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central app configuration, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Database (Postgres) ---
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_qa"
    # Sync URL used by Alembic and the Celery worker (both run outside the event loop).
    database_url_sync: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/rag_qa"

    # --- Redis (cache + Celery broker/backend) ---
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    query_cache_ttl_seconds: int = 3600

    # --- Auth ---
    jwt_secret: str = "change-me-in-prod"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    # --- LLM / vector store ---
    groq_api_key: str | None = None
    pinecone_api_key: str | None = None
    pinecone_index: str | None = None
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Legacy HackRx grader endpoint (kept for backward compatibility) ---
    hackrx_token: str | None = None

    # --- AWS (optional; falls back to direct URL download when unset) ---
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str = "ap-south-1"
    s3_bucket_name: str | None = None


settings = Settings()
