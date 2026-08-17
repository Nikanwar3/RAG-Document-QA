import os
import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models import Document, DocumentStatus
from app.services import document_processor, storage, vector_store
from app.worker.celery_app import celery_app

# Celery tasks run outside the asyncio event loop, so the worker gets its own
# plain synchronous SQLAlchemy engine (psycopg2) rather than sharing the API's
# async engine.
_sync_engine = create_engine(settings.database_url_sync, future=True)
SyncSessionLocal = sessionmaker(bind=_sync_engine, expire_on_commit=False)


@celery_app.task(name="app.worker.tasks.ingest_document_task", bind=True, max_retries=2)
def ingest_document_task(self, document_id: str):
    """Background job: download -> extract text -> chunk -> embed -> upsert into
    Pinecone. Runs off the request path so uploading a large document doesn't
    block the API."""
    session = SyncSessionLocal()
    tmp_path = None
    try:
        # SQLAlchemy's generic Uuid type (used so the same models work against
        # both Postgres and SQLite) expects an actual uuid.UUID for session.get,
        # not the plain string Celery serialized the id as.
        pk = uuid.UUID(document_id)
        document = session.get(Document, pk)
        if document is None:
            return {"status": "not_found", "document_id": document_id}

        document.status = DocumentStatus.PROCESSING
        session.commit()

        suffix = os.path.splitext(document.filename or "")[1] or ".pdf"
        tmp_path = storage.download_document(document.source_url, document.s3_key, suffix)

        text = document_processor.extract_text_from_document(tmp_path)
        if not text.strip():
            raise ValueError("No text could be extracted from the document")

        chunks = document_processor.chunk_text(text)
        if not chunks:
            raise ValueError("No chunks created from document text")

        vector_store.embed_and_store_chunks(chunks, namespace=document.namespace)

        document.status = DocumentStatus.READY
        document.error_message = None
        session.commit()
        return {"status": "ready", "document_id": document_id, "chunks": len(chunks)}

    except Exception as exc:
        session.rollback()
        document = session.get(Document, uuid.UUID(document_id))
        if document is not None:
            document.status = DocumentStatus.FAILED
            document.error_message = str(exc)
            session.commit()
        raise

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        session.close()
