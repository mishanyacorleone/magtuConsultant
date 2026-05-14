"""create data tables

Revision ID: 002
Revises: 001
Create Date: 2025-01-01 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Проходные баллы за прошлые годы
    op.create_table(
        "marks_last_years",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("code", sa.Text(), nullable=True),
        sa.Column("profile_spec_name", sa.Text(), nullable=True),
        sa.Column("mark", sa.Integer(), nullable=True),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_marks_spec_name", "marks_last_years", ["profile_spec_name"])
    op.create_index("idx_marks_year", "marks_last_years", ["year"])

    # Минимальные и максимальные баллы по предметам
    op.create_table(
        "min_max_marks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("vi_name", sa.Text(), nullable=True),
        sa.Column("min_marks", sa.Integer(), nullable=True),
        sa.Column("max_marks", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # Стоимость обучения
    op.create_table(
        "prices",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("code", sa.Text(), nullable=True),
        sa.Column("spec_name", sa.Text(), nullable=True),
        sa.Column("ed_form", sa.Text(), nullable=True),
        sa.Column("price", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_prices_spec_name", "prices", ["spec_name"])

    # Информация о специальностях
    op.create_table(
        "spec_info",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("code", sa.Text(), nullable=True),
        sa.Column("profile_spec_name", sa.Text(), nullable=True),
        sa.Column("lvl", sa.Text(), nullable=True),
        sa.Column("link", sa.Text(), nullable=True),
        sa.Column("FormEd", sa.Text(), nullable=True),
        sa.Column("Plan_Budg", sa.Integer(), nullable=True),
        sa.Column("Plan_Comm", sa.Integer(), nullable=True),
        sa.Column("Years", sa.Integer(), nullable=True),
        sa.Column("Months", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_spec_info_spec_name", "spec_info", ["profile_spec_name"])

    # Вступительные испытания в магистратуру
    op.create_table(
        "vi_mag",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("code", sa.Text(), nullable=True),
        sa.Column("profile_spec_name", sa.Text(), nullable=True),
        sa.Column("vi_name", sa.Text(), nullable=True),
        sa.Column("min_marks", sa.Integer(), nullable=True),
        sa.Column("max_marks", sa.Integer(), nullable=True),
        sa.Column("vi_language", sa.Text(), nullable=True),
        sa.Column("vi_form", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_vi_mag_spec_name", "vi_mag", ["profile_spec_name"])

    # Вступительные испытания после школы (ЕГЭ)
    op.create_table(
        "vi_soo_vo",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("code", sa.Text(), nullable=True),
        sa.Column("profile_spec_name", sa.Text(), nullable=True),
        sa.Column("required_vi", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("optional_vi_ege", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("optional_vi_vuz", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_vi_soo_vo_spec_name", "vi_soo_vo", ["profile_spec_name"])

    # Вступительные испытания после колледжа (СПО)
    op.create_table(
        "vi_spo",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("code", sa.Text(), nullable=True),
        sa.Column("profile_spec_name", sa.Text(), nullable=True),
        sa.Column("required_vi", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_vi_spo_spec_name", "vi_spo", ["profile_spec_name"])

    # Соответствие специальностей СПО и ВО
    op.create_table(
        "vo_spo_comb",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("profile", sa.Text(), nullable=True),
        sa.Column("vo_spec", sa.Text(), nullable=True),
        sa.Column("spo_spec", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("vo_spo_comb")
    op.drop_table("vi_spo")
    op.drop_table("vi_soo_vo")
    op.drop_table("vi_mag")
    op.drop_table("spec_info")
    op.drop_table("prices")
    op.drop_table("min_max_marks")
    op.drop_table("marks_last_years")