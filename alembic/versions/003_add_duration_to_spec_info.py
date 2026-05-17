"""Replace Years and Months with Duration in spec_info

Revision ID: 003
Revises: 002
Create Date: 2026-05-17
"""

from alembic import op
import sqlalchemy as sa

revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('spec_info', sa.Column('Duration', sa.String(), nullable=True))
    op.drop_column('spec_info', 'Years')
    op.drop_column('spec_info', 'Months')


def downgrade() -> None:
    op.add_column('spec_info', sa.Column('Years', sa.Integer(), nullable=True))
    op.add_column('spec_info', sa.Column('Months', sa.Integer(), nullable=True))
    op.drop_column('spec_info', 'Duration')
