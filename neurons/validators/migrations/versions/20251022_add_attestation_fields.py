"""add attestation fields and whitelist

Revision ID: 20251022_attestation
Revises: ba52e4725227
Create Date: 2025-10-22

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '20251022_attestation'
down_revision: Union[str, None] = 'ba52e4725227'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add attestation fields to executor table
    op.add_column('executor', sa.Column('attestation_digest', sa.String(), nullable=True))
    op.add_column('executor', sa.Column('tee_type', sa.String(), nullable=True))
    op.add_column('executor', sa.Column('attestation_verified', sa.Boolean(), nullable=True))
    op.add_column('executor', sa.Column('attestation_verified_at', sa.String(), nullable=True))

    # Create attestation_whitelist table
    op.create_table(
        'attestation_whitelist',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('tee_type', sa.String(), nullable=False),
        sa.Column('attestation_digest', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('added_at', sa.String(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create index on tee_type and attestation_digest for faster lookups
    op.create_index('ix_attestation_whitelist_digest', 'attestation_whitelist', ['tee_type', 'attestation_digest'])


def downgrade() -> None:
    # Drop whitelist table and index
    op.drop_index('ix_attestation_whitelist_digest', table_name='attestation_whitelist')
    op.drop_table('attestation_whitelist')

    # Drop attestation fields from executor
    op.drop_column('executor', 'attestation_verified_at')
    op.drop_column('executor', 'attestation_verified')
    op.drop_column('executor', 'tee_type')
    op.drop_column('executor', 'attestation_digest')
