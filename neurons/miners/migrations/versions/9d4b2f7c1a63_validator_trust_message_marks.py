"""validator_trust_message: the network of the build that wrote each row and when its signer was chain-confirmed

Revision ID: 9d4b2f7c1a63
Revises: 5c1e7a2d9b40
Create Date: 2026-11-23 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '9d4b2f7c1a63'
down_revision: str | None = '5c1e7a2d9b40'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # existing rows stay unmarked: unconfirmed, kept only while their signer is the release or the chain owner
    op.add_column('validator_trust_message', sa.Column('network', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
    op.add_column('validator_trust_message', sa.Column('owner_confirmed_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column('validator_trust_message', 'owner_confirmed_at')
    op.drop_column('validator_trust_message', 'network')
