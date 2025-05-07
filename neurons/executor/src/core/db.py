from collections.abc import Generator
from typing import Annotated
from contextlib import contextmanager

from fastapi import Depends
from sqlmodel import Session, SQLModel, create_engine

from core.config import settings

engine = create_engine(
    settings.DB_URI,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_reset_on_return="rollback",
    pool_timeout=30,
    pool_recycle=1800,
    pool_use_lifo=True,
)


SQLModel.metadata.create_all(engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
