"""create chat_history table

Revision ID: 001
Revises:
Create Date: 2025-01-01 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "chat_history",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=50), nullable=True, comment="qdrant | postgres"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_chat_history_user_created",
        "chat_history",
        ["user_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_chat_history_user_created", table_name="chat_history")
    op.drop_table("chat_history")