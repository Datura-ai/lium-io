import logging
from contextlib import asynccontextmanager
from typing import TypeVar

from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from core.db import AsyncSessionMaker, get_pool_status

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=SQLModel)


class BaseDao:
    """Base DAO with per-operation session management to avoid coroutine conflicts."""

    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        """Context manager for database sessions - uses global session maker with connection pooling."""
        logger.debug(f"Acquiring database session. {get_pool_status()}")
        async with AsyncSessionMaker() as session:
            try:
                yield session
                await session.commit()
                logger.debug(f"Database session committed successfully. {get_pool_status()}")
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}. {get_pool_status()}", exc_info=True)
                raise
