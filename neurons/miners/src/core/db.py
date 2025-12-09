from collections.abc import AsyncGenerator, Generator
from typing import Annotated, Optional

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import AsyncAdaptedQueuePool
from sqlmodel import Session, create_engine

from core.config import settings

# Local DB engine (for standard mode and local miner operations)
engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI))

# Production DB engine (for central mode - connects to lium-miner-portal prod DB)
prod_engine = None
if settings.PROD_DATABASE_URL:
    prod_engine = create_engine(str(settings.PROD_DATABASE_URL))

# Async engine and session maker for production DB
_async_prod_engine = None
_async_prod_session_maker: Optional[async_sessionmaker[AsyncSession]] = None

POOL_SIZE = 128


def get_async_prod_engine():
    """Get or create async engine for production database"""
    global _async_prod_engine

    if _async_prod_engine is None:
        if not settings.PROD_DATABASE_URL:
            raise RuntimeError("Production database not configured. Set PROD_DATABASE_URL environment variable.")

        # Convert postgresql:// to postgresql+asyncpg://
        async_url = str(settings.PROD_DATABASE_URL).replace(
            "postgresql://", "postgresql+asyncpg://"
        )

        _async_prod_engine = create_async_engine(
            async_url,
            echo=False,
            future=True,
            poolclass=AsyncAdaptedQueuePool,
            pool_size=POOL_SIZE,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections every hour to prevent stale connections
            pool_timeout=30,  # Timeout for getting connection from pool
            max_overflow=256,
        )

    return _async_prod_engine

def get_async_prod_session_maker() -> async_sessionmaker[AsyncSession]:
    """Get or create async session maker for production database"""
    global _async_prod_session_maker

    if _async_prod_session_maker is None:
        engine = get_async_prod_engine()
        _async_prod_session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    return _async_prod_session_maker

async def get_async_prod_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async session for production database"""
    session_maker = get_async_prod_session_maker()
    async with session_maker() as session:
        yield session

async def close_async_engine():
    """Close the async engine (call on shutdown)"""
    global _async_prod_engine
    if _async_prod_engine is not None:
        await _async_prod_engine.dispose()
        _async_prod_engine = None

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
