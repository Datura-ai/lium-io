import logging

logger = logging.getLogger(__name__)


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

    logger.info("Waiting for services to be available...")

    while True:
        try:
            # Check Redis connection
            redis_client.ping()
            logger.info("Connected to Redis.")

            # Check PostgreSQL connection using SQLAlchemy
            from sqlalchemy import create_engine, text
            from sqlalchemy.exc import SQLAlchemyError

            engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)
            try:
                with engine.connect() as connection:
                    connection.execute(text("SELECT 1"))
                logger.info("Connected to PostgreSQL.")
            except SQLAlchemyError as e:
                logger.error("Failed to connect to PostgreSQL.")
                raise e

            break  # Exit loop if both connections are successful
        except (psycopg2.OperationalError, RedisConnectionError) as e:
            if time.time() - start_time > timeout:
                logger.error("Timeout while waiting for services to be available.")
                raise e
            logger.warning("Waiting for services to be available...")
            time.sleep(1)
