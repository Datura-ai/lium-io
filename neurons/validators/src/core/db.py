import logging
import time
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import AsyncAdaptedQueuePool, NullPool
from sqlmodel.ext.asyncio.session import AsyncSession

from core.config import settings

logger = logging.getLogger(__name__)

POOL_SIZE = 20

# Create engine with conditional parameters based on environment
if settings.ENV == "test":
    engine = create_async_engine(
        str(settings.ASYNC_SQLALCHEMY_DATABASE_URI),
        echo=False,
        future=True,
        poolclass=NullPool
    )
else:
    engine = create_async_engine(
        str(settings.ASYNC_SQLALCHEMY_DATABASE_URI),
        echo=settings.ENV == "dev",
        future=True,
        poolclass=AsyncAdaptedQueuePool,
        pool_size=POOL_SIZE,
        max_overflow=30
    )

AsyncSessionMaker = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionMaker() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_db)]


def get_pool_status() -> str:
    """Get current pool status for debugging."""
    if hasattr(engine.pool, 'size'):
        pool_size = engine.pool.size()
        checked_out = engine.pool.checkedout()
        overflow = engine.pool.overflow()
        checked_in = engine.pool.checkedin()
        return f"Pool size: {pool_size}, Checked out: {checked_out}, Overflow: {overflow}, Checked in: {checked_in}"
    return "Pool status unavailable"


# Event listeners for debugging connection pool usage
def checkout_listener(dbapi_con, con_record):
    """Log when connection is taken from pool."""
    con_record.checkout_time = time.time()
    if settings.ENV == "dev":
        logger.debug(f"Connection checked out from pool. {get_pool_status()}")


def checkin_listener(dbapi_con, con_record):
    """Log when connection is returned to pool."""
    if hasattr(con_record, 'checkout_time'):
        usage_time = (time.time() - con_record.checkout_time) * 1000
        if settings.ENV == "dev":
            logger.debug(f"Connection returned to pool after {usage_time:.2f}ms. {get_pool_status()}")


def invalidate_listener(dbapi_con, con_record, exception):
    """Log when connection is invalidated."""
    logger.warning(f"Connection invalidated: {exception}. {get_pool_status()}")


# Register event listeners only for non-NullPool engines and non-async engines
# For async engines, we need to listen to sync_engine events
if hasattr(engine, 'sync_engine'):
    event.listen(engine.sync_engine, 'connect', checkout_listener)
    event.listen(engine.sync_engine, 'close', checkin_listener)
    event.listen(engine.sync_engine, 'invalidate', invalidate_listener)
else:
    event.listen(engine, 'connect', checkout_listener)
    event.listen(engine, 'close', checkin_listener)
    event.listen(engine, 'invalidate', invalidate_listener)
