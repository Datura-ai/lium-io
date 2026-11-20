"""validator_trust_message: the validator messages the subnet owner signed that this miner took

Revision ID: 5c1e7a2d9b40
Revises: 33be5b1944a8
Create Date: 2026-11-20 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = '5c1e7a2d9b40'
down_revision: Union[str, None] = '33be5b1944a8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'validator_trust_message',
        sa.Column('revision', sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column('message', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('signature', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('signer', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.PrimaryKeyConstraint('revision'),
    )


def downgrade() -> None:
    op.drop_table('validator_trust_message')
