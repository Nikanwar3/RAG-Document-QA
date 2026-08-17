from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import get_cached_answer, set_cached_answer
from app.database import get_db
from app.deps import get_current_user
from app.models import Document, DocumentStatus, QueryLog, User
from app.schemas import QueryRequest, QueryResponse
from app.services import llm_client, vector_store

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def ask_question(
    payload: QueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Document).where(Document.id == payload.document_id, Document.owner_id == current_user.id)
    )
    document = result.scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if document.status != DocumentStatus.READY:
        raise HTTPException(
            status_code=409, detail=f"Document is not ready yet (status={document.status.value})"
        )

    document_id = str(document.id)

    cached = await get_cached_answer(document_id, payload.question)
    if cached is not None:
        db.add(QueryLog(document_id=document.id, question=payload.question, answer=cached, cache_hit=True))
        await db.commit()
        return QueryResponse(question=payload.question, answer=cached, cache_hit=True)

    # Both calls are blocking (network + CPU-bound embedding), so run them off
    # the event loop instead of stalling every other concurrent request.
    context = await run_in_threadpool(vector_store.query_top_chunks, payload.question, document.namespace)
    answer = await run_in_threadpool(llm_client.generate_answer, payload.question, context)

    await set_cached_answer(document_id, payload.question, answer)
    db.add(QueryLog(document_id=document.id, question=payload.question, answer=answer, cache_hit=False))
    await db.commit()

    return QueryResponse(question=payload.question, answer=answer, cache_hit=False)
