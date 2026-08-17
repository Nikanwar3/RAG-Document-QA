import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, String, Text, Uuid
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import relationship

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=_utcnow, nullable=False)

    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    source_url = Column(Text, nullable=True)
    s3_key = Column(String(512), nullable=True)
    filename = Column(String(255), nullable=True)
    status = Column(SAEnum(DocumentStatus), default=DocumentStatus.PENDING, nullable=False)
    error_message = Column(Text, nullable=True)
    # Pinecone namespace this document's chunks live under, so documents never bleed into
    # each other's vector search results.
    namespace = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    owner = relationship("User", back_populates="documents")
    queries = relationship("QueryLog", back_populates="document", cascade="all, delete-orphan")


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(Uuid(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    cache_hit = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=_utcnow, nullable=False)

    document = relationship("Document", back_populates="queries")

    __table_args__ = (Index("ix_query_logs_document_id_created_at", "document_id", "created_at"),)
