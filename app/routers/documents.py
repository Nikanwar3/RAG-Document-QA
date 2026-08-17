import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import get_current_user
from app.models import Document, DocumentStatus, User
from app.schemas import DocumentCreate, DocumentOut
from app.worker.tasks import ingest_document_task

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
async def create_document(
    payload: DocumentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Register a document and hand it off to the background worker. Returns
    immediately with status=pending; poll GET /documents/{id} for readiness."""
    document = Document(
        owner_id=current_user.id,
        source_url=payload.source_url,
        filename=payload.filename,
        status=DocumentStatus.PENDING,
        namespace=str(uuid.uuid4()),
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    ingest_document_task.delay(str(document.id))

    return document


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Document).where(Document.id == document_id, Document.owner_id == current_user.id)
    )
    document = result.scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.get("", response_model=list[DocumentOut])
async def list_documents(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Document).where(Document.owner_id == current_user.id))
    return result.scalars().all()
