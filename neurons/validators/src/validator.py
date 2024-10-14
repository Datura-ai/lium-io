import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from connector import start_connector_process
from fastapi import FastAPI

from core.config import settings
from core.validator import Validator
from routes.apis import apis_router

# Set up logging
logging.basicConfig(level=logging.INFO)


def wait_for_services_sync(timeout=30):
    """Wait until Redis and PostgreSQL connections are working."""
    import time

    import psycopg2
    from redis import Redis
    from redis.exceptions import ConnectionError as RedisConnectionError

    from core.config import settings

    # Initialize Redis client
    redis_client = Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)

    start_time = time.time()

    logging.info("Waiting for services to be available...")

    while True:
        try:
            # Check Redis connection
            redis_client.ping()
            logging.info("Connected to Redis.")

            # Check PostgreSQL connection using SQLAlchemy
            from sqlalchemy import create_engine, text
            from sqlalchemy.exc import SQLAlchemyError

            engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)
            try:
                with engine.connect() as connection:
                    connection.execute(text("SELECT 1"))
                logging.info("Connected to PostgreSQL.")
            except SQLAlchemyError as e:
                logging.error("Failed to connect to PostgreSQL.")
                raise e

            break  # Exit loop if both connections are successful
        except (psycopg2.OperationalError, RedisConnectionError) as e:
            if time.time() - start_time > timeout:
                logging.error("Timeout while waiting for services to be available.")
                raise e
            logging.warning("Waiting for services to be available...")
            time.sleep(1)


wait_for_services_sync()


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    validator = Validator()
    # Run the miner in the background
    task = asyncio.create_task(validator.start())
    connector_process = start_connector_process()

    try:
        yield
    finally:
        await validator.stop()  # Ensure proper cleanup
        await task  # Wait for the background task to complete
        connector_process.terminate()
        logging.info("Validator exited successfully.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=app_lifespan,
)

app.include_router(apis_router)

reload = True if settings.ENV == "dev" else False

if __name__ == "__main__":
    uvicorn.run("validator:app", host="0.0.0.0", port=settings.INTERNAL_PORT, reload=reload)
