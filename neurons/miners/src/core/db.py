from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from sqlmodel import Session, create_engine

from core.config import settings

# Local DB engine (for standard mode and local miner operations)
engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI))

# Production DB engine (for central mode - connects to lium-miner-portal prod DB)
prod_engine = None
if settings.PROD_DATABASE_URL:
    prod_engine = create_engine(str(settings.PROD_DATABASE_URL))


def get_db() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session

def get_prod_db() -> Generator[Session, None, None]:
    """Get production database session (for central mode)"""
    if not prod_engine:
        raise RuntimeError("Production database not configured. Set PROD_DATABASE_URL environment variable.")
    with Session(prod_engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_db)]
