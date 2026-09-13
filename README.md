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
FastAPI  ── structured JSON request logs (app/observability.py) ──────
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
 │                   download → extract → chunk → embed → upsert, plus a
 │                   best-effort knowledge-graph extraction pass)
 ├──────────────────────────┐
 ▼                          ▼
Pinecone                    Neo4j
(vector search,             (entities/relationships extracted per
 namespaced per document)    document, namespaced the same way)
 │                          │
 └────────────┬─────────────┘
              ▼
             LLM  (Groq / Llama3-8B, via LangChain's ChatGroq)
```

Document **ingestion** (download → extract → chunk → embed → upsert) runs on
a Celery worker, off the request path, so uploading a large PDF doesn't block
the API. Ingestion also makes a best-effort pass to extract a small
knowledge graph of the document's entities and relationships into Neo4j
(`app/services/graph_store.py`) — this is enrichment, not a hard dependency:
a failure here (LLM extraction error, Neo4j unreachable) is logged and
swallowed, and the document is still fully usable for Q&A without it.
**Querying** a ready document is synchronous: check the Redis cache first,
and on a miss, do a Pinecone similarity search + LLM call, then cache and
log the result.

Two query paths share that Pinecone/Groq foundation but differ in how much
reasoning happens around the retrieval:

- **`POST /query`** — one retrieve, one generate. Cheapest path.
- **`POST /query/agent`** — a LangGraph agent (`app/services/qa_agent.py`)
  with two extra steps beyond retrieve-then-generate:

  1. **Corrective retrieval** — an LLM grades whether the retrieved chunks
     actually answer the question before generating, and rewrites the
     search query and retries retrieval (capped at 2 attempts) if they
     don't, instead of confidently answering off a bad first retrieval.
  2. **Graph augmentation** — once the context is graded relevant, a second
     LLM call *decides*, via tool-calling (not a hardcoded rule), whether
     the document's knowledge graph has a directly related fact worth
     folding in first — e.g. the retrieved clause refers to "the
     Policyholder" by name without defining it, and the graph knows what
     that resolves to elsewhere in the document.

  ```
  retrieve → grade →[relevant, or out of retries]→ graph_augment → generate → done
               │
               └──[not relevant, retries left]──→ rewrite query → retrieve (loop)
  ```

  Every node's latency and outcome is logged individually
  (`app/observability.py`'s `log_node_timing`), so a slow or failing
  `/query/agent` call is debuggable node-by-node, not just as one opaque
  request.

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
| Knowledge graph | Neo4j — LLM-extracted (subject, predicate, object) relations, one graph per document |
| LLM | Groq Cloud (Llama3-8B-8192) via LangChain (`ChatGroq`, LCEL) |
| Agent orchestration | LangGraph — corrective-RAG + tool-calling graph-augmentation `StateGraph` behind `POST /query/agent` |
| Structured outputs / tool calling | Pydantic-typed LLM outputs (`with_structured_output`) for relevance grading; LangChain `@tool` + `bind_tools` for the graph lookup |
| Evaluation | Keyword-recall harness (`app/services/evaluation.py`, `scripts/run_evaluation.py`) |
| Observability | Structured JSON logs (per-request + per-agent-node, `app/observability.py`); optional LangSmith tracing via `LANGCHAIN_TRACING_V2` |
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
├── observability.py              # Structured JSON logging, request + agent-node timing
├── routers/
│   ├── auth.py                  # POST /auth/register, /auth/login
│   ├── documents.py             # POST/GET /documents, GET /documents/{id}/graph
│   ├── query.py                  # POST /query, POST /query/agent — cached Q&A
│   └── hackrx.py                  # Legacy /hackrx/run (grader compatibility)
├── services/
│   ├── document_processor.py       # PDF/DOCX/EML text extraction + chunking
│   ├── vector_store.py              # Pinecone embed/query, per-doc namespace
│   ├── graph_store.py                 # Neo4j knowledge graph, per-doc namespace
│   ├── llm_client.py                 # ChatGroq call (LangChain LCEL) + prompt
│   ├── qa_agent.py                    # LangGraph corrective-RAG + graph-augment agent
│   ├── evaluation.py                   # Keyword-recall scoring for offline eval
│   └── storage.py                     # S3 upload/download, URL fallback
└── worker/
    ├── celery_app.py                    # Celery app (Redis broker/backend)
    └── tasks.py                          # ingest_document_task

alembic/                # DB migrations
tests/                  # pytest suite (SQLite in-memory, mocked externals)
scripts/                # Manual smoke-test / evaluation scripts (not run by CI)
```

-----

## Running Locally (Docker Compose)

```bash
cp .env.example .env
# fill in GROQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX at minimum

docker compose up --build
```

This starts Postgres, Redis, Neo4j, the API (after running `alembic upgrade
head`), and a Celery worker. The API is live at `http://localhost:8000`,
with interactive docs at `/docs`, and the Neo4j browser at
`http://localhost:7474`. Neo4j is an enrichment layer, not a hard
dependency — see [Architecture](#architecture) — so the stack still answers
questions normally even before any document has a graph.

### Running without Docker

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt

# Postgres + Redis need to be running locally (or point DATABASE_URL /
# REDIS_URL at hosted instances). Neo4j is optional — skip it and ingestion
# just logs a warning and skips graph extraction for that document.
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

### `GET /documents/{id}/graph`

```json
{
  "relations": [
    { "subject": "the Policyholder", "predicate": "must pay", "object": "the Premium" }
  ]
}
```

The knowledge-graph relations extracted from this document at ingestion
time (see [Architecture](#architecture)) — an empty list if extraction was
skipped or unavailable when the document was ingested, not an error.

### `POST /query`

```json
{ "document_id": "<uuid>", "question": "What is the grace period for premium payment?" }
```

Returns `409` until the document's status is `ready`. Answers are cached in
Redis per `(document_id, question)` and every call is logged to the
`query_logs` table with a `cache_hit` flag.

### `POST /query/agent`

Same request/response shape as `POST /query`, answered by the LangGraph
agent instead of a single retrieve-then-generate pass:

```json
{
  "question": "...",
  "answer": "...",
  "cache_hit": false,
  "retrieval_attempts": 2,
  "query_rewritten": true,
  "graph_augmented": false
}
```

`retrieval_attempts` and `query_rewritten` surface whether the agent had to
grade the first retrieval as irrelevant and self-correct; `graph_augmented`
surfaces whether it decided (via tool-calling) that the document's
knowledge graph had a relevant fact worth folding in — see
`app/services/qa_agent.py` for the graph.

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

Tests run against an in-memory SQLite database and mock out Pinecone, Neo4j,
the Groq LLM call, Redis, and the Celery `.delay()` call — no external
services or credentials required. CI (`.github/workflows/ci.yml`)
additionally spins up real Postgres and Redis containers and does a Docker
image build.

-----

## Evaluation

```bash
python scripts/run_evaluation.py --document-id <uuid> --token <jwt> [--agent]
```

Runs a small fixed set of questions against a ready document through
`POST /query` (or `POST /query/agent` with `--agent`) and scores each answer
by keyword recall against an expected-keywords list — a deliberately cheap,
deterministic proxy metric (no LLM-judge call, no extra API cost) rather
than a full RAGAS-style faithfulness/relevance evaluation. The scoring logic
itself (`app/services/evaluation.py`) is unit-tested in
`tests/test_evaluation.py`; the script needs a live server + real document
+ credentials, so it isn't run by CI, the same as
`scripts/manual_hackrx_smoke_test.py`.

-----

## Observability

Every request gets one structured JSON log line (method, path, status,
latency, a request id echoed back as `X-Request-ID`), and every node in the
`/query/agent` graph (retrieve/grade/rewrite/graph_augment/generate) gets
its own timing + success/failure log line — see `app/observability.py`.
That's enough to answer "was this request slow, and if so which node" from
logs alone, without an APM agent.

For LLM-call-level tracing (prompts, token counts, latency per Groq call)
across every LangChain/LangGraph invocation, set these and LangChain picks
them up automatically — no code changes needed:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=rag-document-qa
```

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
PyMuPDF (AGPL/Commercial), Pinecone (Commercial), Groq (Commercial),
Neo4j Community Edition (GPLv3).
