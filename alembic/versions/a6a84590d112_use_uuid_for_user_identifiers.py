"""use native PostgreSQL UUID types for user ownership

Revision ID: a6a84590d112
Revises: d4e8f19a2b70
Create Date: 2026-09-20 22:20:00.000000

The prior ownership migration deliberately left legacy documents unowned.
Their nullable user_id values remain NULL during this conversion, while
generated UUID strings for newly registered accounts are converted losslessly.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "a6a84590d112"
down_revision: Union[str, Sequence[str], None] = "d4e8f19a2b70"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint("fk_documents_user_id_users", "documents", type_="foreignkey")
    op.alter_column(
        "users",
        "id",
        existing_type=sa.String(length=36),
        type_=postgresql.UUID(as_uuid=False),
        postgresql_using="id::uuid",
    )
    op.alter_column(
        "documents",
        "user_id",
        existing_type=sa.String(length=36),
        type_=postgresql.UUID(as_uuid=False),
        postgresql_using="user_id::uuid",
    )
    op.create_foreign_key(
        "fk_documents_user_id_users",
        "documents",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint("fk_documents_user_id_users", "documents", type_="foreignkey")
    op.alter_column(
        "documents",
        "user_id",
        existing_type=postgresql.UUID(as_uuid=False),
        type_=sa.String(length=36),
        postgresql_using="user_id::text",
    )
    op.alter_column(
        "users",
        "id",
        existing_type=postgresql.UUID(as_uuid=False),
        type_=sa.String(length=36),
        postgresql_using="id::text",
    )
    op.create_foreign_key(
        "fk_documents_user_id_users",
        "documents",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
