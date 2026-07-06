"""Executor tier column

Adds a ``tier`` column to the executor table so the miner CLI can surface
whether each executor is ``secure`` or ``spot`` (DAH-1645). Defaults to
``"secure"`` to keep legacy rows valid.

Revision ID: a7c3d1f80e21
Revises: 33be5b1944a8
Create Date: 2026-05-14 08:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "a7c3d1f80e21"
down_revision: Union[str, None] = "33be5b1944a8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "executor",
        sa.Column(
            "tier",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'secure'"),
        ),
    )


def downgrade() -> None:
    op.drop_column("executor", "tier")
