from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from sqlalchemy import Engine
from sqlmodel import Session, create_engine

from core.config import settings

# A pooled connection is retired after this long — the value the backend and the portal use (the validator recycles hourly).
POOL_RECYCLE_SECONDS = 1800


def make_engine(uri: str) -> Engine:
    return create_engine(uri, pool_pre_ping=True, pool_recycle=POOL_RECYCLE_SECONDS)


engine = make_engine(str(settings.SQLALCHEMY_DATABASE_URI))


def get_db() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_db)]
