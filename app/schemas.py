import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    # bcrypt silently ignores/rejects bytes past 72, so cap here for a clear
    # 422 instead of a confusing hash-time failure.
    password: str = Field(min_length=8, max_length=72)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class DocumentCreate(BaseModel):
    source_url: str
    filename: str | None = None


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str | None = None
    status: str
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class QueryRequest(BaseModel):
    document_id: uuid.UUID
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    cache_hit: bool


class QueryAgentResponse(BaseModel):
    question: str
    answer: str
    cache_hit: bool
    retrieval_attempts: int
    query_rewritten: bool


class HealthResponse(BaseModel):
    status: str
    database: str
    redis: str
