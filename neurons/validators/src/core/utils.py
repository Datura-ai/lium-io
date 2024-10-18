import asyncio
import contextvars
import logging

from pythonjsonlogger import jsonlogger

logger = logging.getLogger(__name__)

# Create a ContextVar to hold the context information
context = contextvars.ContextVar("context", default="TaskService")
context.set("TaskService")


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


def get_logger(name):
    logger = logging.getLogger(name)
    stdout = logging.StreamHandler()
    fmt = jsonlogger.JsonFormatter(
        "%(name)s %(asctime)s %(levelname)s %(filename)s %(funcName)s %(lineno)s %(process)d %(message)s"
    )
    stdout.setFormatter(fmt)
    logger.setLevel(logging.INFO)
    logger.addHandler(stdout)
    return logger


def get_extra_info(extra: dict) -> dict:
    task = asyncio.current_task()
    coro_name = task.get_coro().__name__ if task else "NoTask"
    task_id = id(task) if task else "NoTaskID"
    extra_info = {
        "coro_name": coro_name,
        "task_id": task_id,
        **extra,
    }
    return extra_info


def configure_logs_of_other_modules():
    sqlalchemy_logger = logging.getLogger("sqlalchemy.engine.Engine")
    sqlalchemy_logger.setLevel(logging.WARNING)

    class ContextFilter(logging.Filter):
        """
        This is a filter which injects contextual information into the log.
        """

        def filter(self, record):
            record.context = context.get() or "Default"
            return True

    # Create a custom formatter that adds the context to the log messages
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            try:
                task = asyncio.current_task()
                coro_name = task.get_coro().__name__ if task else "NoTask"
                task_id = id(task) if task else "NoTaskID"
                return f"{getattr(record, 'context', 'Default')} | {coro_name} | {task_id} | {super().format(record)}"
            except Exception:
                return ""

    asyncssh_logger = logging.getLogger("asyncssh")
    asyncssh_logger.setLevel(logging.INFO)

    # Add the filter to the logger
    asyncssh_logger.addFilter(ContextFilter())

    # Create a handler for the logger
    handler = logging.StreamHandler()

    # Add the handler to the logger
    asyncssh_logger.handlers = []
    asyncssh_logger.addHandler(handler)

    # Set the formatter for the handler
    handler.setFormatter(
        CustomFormatter("%(name)s %(asctime)s %(levelname)s %(filename)s %(process)d %(message)s")
    )
