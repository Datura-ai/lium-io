"""Database fixtures for validators tests."""

import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """Create an async engine for testing with SQLite in-memory database."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=True,
        future=True,
    )

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest_asyncio.fixture
async def test_db_session(test_engine):
    """Create a test database session for each test."""
    async_session_maker = sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_maker() as session:
        yield session
        await session.rollback()
