from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from sqlalchemy import Engine
from sqlmodel import Session, create_engine

from core.config import settings

# A pooled connection is retired after this long — the value the backend and the portal use (the validator recycles hourly).
POOL_RECYCLE_SECONDS = 1800


def make_engine(uri: str) -> Engine:
    """The miner's engine with the pool options the backend and the portal use (`pool_pre_ping=True, pool_recycle=1800`;
    the validator pings too and recycles hourly). A connection Postgres or a NAT closed while it sat idle is pinged before
    it is handed out and replaced when the ping fails; no connection is reused past POOL_RECYCLE_SECONDS."""
    return create_engine(uri, pool_pre_ping=True, pool_recycle=POOL_RECYCLE_SECONDS)


engine = make_engine(str(settings.SQLALCHEMY_DATABASE_URI))


def get_db() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_db)]
