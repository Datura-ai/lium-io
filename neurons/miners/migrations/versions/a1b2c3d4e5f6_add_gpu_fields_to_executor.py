"""Add gpu_type and gpu_count to executor

Revision ID: a1b2c3d4e5f6
Revises: 33be5b1944a8
Create Date: 2026-02-25 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '33be5b1944a8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('executor', sa.Column('gpu_type', sa.String(), nullable=True))
    op.add_column('executor', sa.Column('gpu_count', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('executor', 'gpu_count')
    op.drop_column('executor', 'gpu_type')
