"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-08-17

"""
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # create_type=False: we create the type explicitly below, so the column
    # definition in create_table() below must not also try to auto-create it
    # (that would emit a second CREATE TYPE and fail with "already exists").
    document_status = postgresql.ENUM(
        "pending", "processing", "ready", "failed", name="documentstatus", create_type=False
    )
    document_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("s3_key", sa.String(length=512), nullable=True),
        sa.Column("filename", sa.String(length=255), nullable=True),
        sa.Column("status", document_status, nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("namespace", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "query_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("cache_hit", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_query_logs_document_id_created_at", "query_logs", ["document_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_query_logs_document_id_created_at", table_name="query_logs")
    op.drop_table("query_logs")
    op.drop_table("documents")
    postgresql.ENUM(name="documentstatus").drop(op.get_bind(), checkfirst=True)
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
