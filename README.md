# RAG Document QA

**Author: Nidhi Kanwar — Junior AI Engineer | Generative AI | RAG Systems**

An LLM-powered document question-answering service. Upload a document, it's
processed asynchronously and embedded into Pinecone, and questions against it
are answered by an LLM with retrieved context — cached in Redis and logged to
Postgres for every request.

-----

## Architecture

```
User
 │
 ▼
FastAPI  ──────────────────────────────────────────────────────────
 │
 ▼
Authentication (JWT, bcrypt-hashed passwords)
 │
 ▼
PostgreSQL  (users, documents, query_logs)
 │
 ▼
Redis  (per-document answer cache, keyed by question hash)
 │  cache miss
 ▼
Background Worker  (Celery, broker/backend on Redis — document ingestion:
 │                   download → extract → chunk → embed)
 ▼
Pinecone  (vector search, namespaced per document)
 │
 ▼
LLM  (Groq / Llama3-8B, via the OpenAI-compatible client)
```

Document **ingestion** (download → extract → chunk → embed → upsert) runs on
a Celery worker, off the request path, so uploading a large PDF doesn't block
the API. **Querying** a ready document is synchronous: check the Redis cache
first, and on a miss, do a Pinecone similarity search + LLM call, then cache
and log the result.

-----

## Tech Stack

| Layer | Technology |
| --- | --- |
| API framework | FastAPI, Pydantic, async/await |
| Auth | JWT (python-jose), bcrypt password hashing (passlib) |
| Database | PostgreSQL, SQLAlchemy (async), Alembic migrations |
| Cache | Redis |
| Background jobs | Celery (Redis broker/backend) |
| Vector search | Pinecone, SentenceTransformers (`all-MiniLM-L6-v2`) |
| LLM | Groq Cloud (Llama3-8B-8192) via the OpenAI-compatible client |
| Object storage | AWS S3 (boto3) — optional, falls back to direct URL download |
| Containerization | Docker, docker-compose |
| Testing | pytest, pytest-asyncio, httpx |
| CI | GitHub Actions |

-----

## Project Layout

```
app/
├── main.py                 # FastAPI app, router registration, /health
├── config.py                # Settings (env-driven, pydantic-settings)
├── database.py               # Async SQLAlchemy engine/session
├── models.py                 # User, Document, QueryLog ORM models
├── schemas.py                 # Pydantic request/response models
├── security.py                 # Password hashing, JWT issue/verify
├── deps.py                     # get_current_user, shared dependencies
├── cache.py                     # Redis-backed query cache
├── routers/
│   ├── auth.py                  # POST /auth/register, /auth/login
│   ├── documents.py             # POST/GET /documents — ingestion + status
│   ├── query.py                  # POST /query — cached Q&A
│   └── hackrx.py                  # Legacy /hackrx/run (grader compatibility)
├── services/
│   ├── document_processor.py       # PDF/DOCX/EML text extraction + chunking
│   ├── vector_store.py              # Pinecone embed/query, per-doc namespace
│   ├── llm_client.py                 # Groq LLM call + prompt template
│   └── storage.py                     # S3 upload/download, URL fallback
└── worker/
    ├── celery_app.py                    # Celery app (Redis broker/backend)
    └── tasks.py                          # ingest_document_task

alembic/                # DB migrations
tests/                  # pytest suite (SQLite in-memory, mocked externals)
scripts/                # Manual smoke-test scripts (not run by CI)
```

-----

## Running Locally (Docker Compose)

```bash
cp .env.example .env
# fill in GROQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX at minimum

docker compose up --build
```

This starts Postgres, Redis, the API (after running `alembic upgrade head`),
and a Celery worker. The API is live at `http://localhost:8000`, with
interactive docs at `/docs`.

### Running without Docker

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt

# Postgres + Redis need to be running locally (or point DATABASE_URL /
# REDIS_URL at hosted instances).
alembic upgrade head

uvicorn app.main:app --reload          # terminal 1: API
celery -A app.worker.celery_app worker --loglevel=info   # terminal 2: worker
```

-----

## API

### `POST /auth/register` / `POST /auth/login`

Register with `{email, password}`; login returns a JWT `access_token`. Every
route below requires `Authorization: Bearer <token>`.

### `POST /documents`

```json
{ "source_url": "https://example.com/policy.pdf", "filename": "policy.pdf" }
```

Returns `202` with the document in `pending` status and hands ingestion off
to the Celery worker.

### `GET /documents/{id}` / `GET /documents`

Poll for status: `pending` → `processing` → `ready` (or `failed`, with
`error_message` set).

### `POST /query`

```json
{ "document_id": "<uuid>", "question": "What is the grace period for premium payment?" }
```

Returns `409` until the document's status is `ready`. Answers are cached in
Redis per `(document_id, question)` and every call is logged to the
`query_logs` table with a `cache_hit` flag.

### `GET /health`

Reports liveness of the API plus reachability of Postgres and Redis.

### `POST /hackrx/run` (legacy)

Preserved for backward compatibility with an existing hackathon grader
contract — single-shot bearer-token auth (`HACKRX_TOKEN`), synchronous
ingestion, no Postgres/Redis involved. New integrations should use the
`/documents` + `/query` flow above instead.

-----

## Testing

```bash
pytest -v
```

Tests run against an in-memory SQLite database and mock out Pinecone, the
Groq LLM call, Redis, and the Celery `.delay()` call — no external services
or credentials required. CI (`.github/workflows/ci.yml`) additionally spins
up real Postgres and Redis containers and does a Docker image build.

-----

## Database Migrations

```bash
alembic upgrade head                       # apply
alembic revision -m "add some_column"       # generate a new migration
```

-----

## Supported Document Formats

| Format | Extensions | Notes |
| --- | --- | --- |
| PDF | `.pdf` | PyMuPDF, block-fallback for scanned pages |
| Word | `.docx`, `.doc` | Paragraphs + tables |
| Email | `.eml`, `.msg` | Headers, multipart, HTML-stripped body |

-----

## License

Open-source components: FastAPI (MIT), SQLAlchemy (MIT), Celery (BSD),
PyMuPDF (AGPL/Commercial), Pinecone (Commercial), Groq (Commercial).
