import asyncio
import contextvars
import json
import logging
from logging.config import dictConfig  # noqa
from tenacity import retry, stop_after_attempt, wait_fixed
import asyncssh

from core.config import settings
from celium_collateral_contracts import CollateralContract

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


def get_extra_info(extra: dict) -> dict:
    try:
        task = asyncio.current_task()
        coro_name = task.get_coro().__name__ if task else "NoTask"
        task_id = id(task) if task else "NoTaskID"
    except Exception:
        coro_name = "NoTask"
        task_id = "NoTaskID"
    extra_info = {
        "coro_name": coro_name,
        "task_id": task_id,
        **extra,
    }
    return extra_info


def configure_logs_of_other_modules():
    validator_hotkey = settings.get_bittensor_wallet().get_hotkey().ss58_address

    logging.basicConfig(
        level=logging.INFO,
        format=f"Validator: {validator_hotkey} | Name: %(name)s | Time: %(asctime)s | Level: %(levelname)s | File: %(filename)s | Function: %(funcName)s | Line: %(lineno)s | Process: %(process)d | Message: %(message)s",
    )

    sqlalchemy_logger = logging.getLogger("sqlalchemy")
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
    asyncssh_logger.setLevel(logging.WARNING)

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


def get_logger(name: str):
    LOGGING = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "verbose": {
                "format": "%(levelname)-8s %(asctime)s --- "
                "%(lineno)-8s [%(name)s] %(funcName)-24s : %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "verbose",
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["console"],
        },
        "loggers": {
            "connector": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "asyncssh": {
                "level": "WARNING",
                "propagate": True,
            },
        },
    }

    dictConfig(LOGGING)
    logger = logging.getLogger(name)
    return logger


class StructuredMessage:
    def __init__(self, message, extra: dict):
        self.message = message
        self.extra = extra

    def __str__(self):
        return "%s >>> %s" % (self.message, json.dumps(self.extra))  # noqa


_m = StructuredMessage


async def retry_ssh_command(
    ssh_client: asyncssh.SSHClientConnection,
    command: str,
    tag: str,
    max_attempts: int = 5,
    wait_seconds: int = 10,
):
    @retry(stop=stop_after_attempt(max_attempts), wait=wait_fixed(wait_seconds))
    async def execute_command():
        result = await ssh_client.run(command)
        if result.exit_status != 0:
            raise Exception(f"[{tag}] command: {command} exit_code {result.exit_status}, stderr: {result.stderr.strip()}")

    await execute_command()


def get_collateral_contract(version: str = "1.0.2") -> CollateralContract:
    """
    Initializes and returns a CollateralContract instance.

    Args:
        network (str): The blockchain network to use ('local', 'test', 'finney', etc.).
        contract_address (str): Address of the collateral contract.
        owner_key (str): Ethereum owner key.
        miner_key (str): Optional miner key required for contract operations.

    Returns:
        CollateralContract: The initialized contract instance.
    """
    network = settings.BITTENSOR_NETWORK
    contract_address = settings.COLLATERAL_CONTRACT_ADDRESS
    if version and settings.CONTRACT_VERSIONS.get(version):
        contract_address = settings.CONTRACT_VERSIONS.get(version)["address"]

    rpc_url = settings.SUBTENSOR_EVM_RPC_URL

    return CollateralContract(
        network=network,
        contract_address=contract_address,
        rpc_url=rpc_url,
    )
