"""add unmarked_recovery_stop

Revision ID: 20261105_recovery_stop
Revises: 20251022_attestation
Create Date: 2026-11-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes

# revision identifiers, used by Alembic.
revision: str = '20261105_recovery_stop'
down_revision: Union[str, None] = '20251022_attestation'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'unmarked_recovery_stop',
        sa.Column('pod_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('pod_id'),
    )


def downgrade() -> None:
    op.drop_table('unmarked_recovery_stop')
