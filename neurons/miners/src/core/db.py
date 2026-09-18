from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from sqlmodel import Session, create_engine

from core.config import settings

# Same pool options as the backend, the portal and the validator (validators/src/core/db.py): a connection Postgres or
# a NAT closed while it sat idle is checked with a ping before it is handed out, and no connection is reused past 30 min.
POOL_RECYCLE_SECONDS = 1800

engine = create_engine(
    str(settings.SQLALCHEMY_DATABASE_URI),
    pool_pre_ping=True,
    pool_recycle=POOL_RECYCLE_SECONDS,
)


def get_db() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_db)]
