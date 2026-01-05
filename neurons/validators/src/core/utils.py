import asyncio
import contextvars
import json
import logging
from logging.config import dictConfig  # noqa
from tenacity import retry, stop_after_attempt, wait_fixed
import asyncssh

from core.config import settings
from celium_collateral_contracts import CollateralContract

# Create a ContextVar to hold the context information
context = contextvars.ContextVar("context", default="TaskService")
context.set("TaskService")


class JSONFormatter(logging.Formatter):
    """Reusable JSON formatter for structured logging across all modules."""

    def __init__(self, include_validator_hotkey=True):
        super().__init__()
        self.include_validator_hotkey = include_validator_hotkey
        self._validator_hotkey = None

    def _get_validator_hotkey(self):
        """Lazy load validator hotkey to avoid initialization issues."""
        if self._validator_hotkey is None and self.include_validator_hotkey:
            try:
                self._validator_hotkey = settings.get_bittensor_wallet().get_hotkey().ss58_address
            except Exception:
                self._validator_hotkey = "unknown"
        return self._validator_hotkey

    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "file": record.filename,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
        }

        # Add validator hotkey if enabled
        if self.include_validator_hotkey:
            hotkey = self._get_validator_hotkey()
            if hotkey:
                log_data["validator"] = hotkey

        # Add context if available
        if hasattr(record, 'context'):
            log_data["context"] = record.context

        # Extract extra data from _StructuredMessage if present
        if hasattr(record, 'msg') and hasattr(record.msg, 'extra') and record.msg.extra:
            log_data["extra"] = record.msg.extra

        # Add exception info if present
        if record.exc_info:
            log_data["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


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
    # Configure root logger with JSON formatter
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add new handler with JSON formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)

    sqlalchemy_logger = logging.getLogger("sqlalchemy")
    sqlalchemy_logger.setLevel(logging.WARNING)

    class ContextFilter(logging.Filter):
        """
        This is a filter which injects contextual information into the log.
        """

        def filter(self, record):
            record.context = context.get() or "Default"
            return True

    asyncssh_logger = logging.getLogger("asyncssh")
    asyncssh_logger.setLevel(logging.WARNING)

    # Add the filter to the logger
    asyncssh_logger.addFilter(ContextFilter())


def get_logger(name: str):
    LOGGING = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": JSONFormatter,
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
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


class _StructuredMessage:
    """Holds message and extra data for structured logging."""
    def __init__(self, message: str, extra: dict = None):
        self.message = message
        self.extra = extra or {}

    def __str__(self):
        # Just return the message, not the concatenated version
        return self.message


def _m(message: str, extra: dict = None):
    """Helper to create structured log messages."""
    return _StructuredMessage(message, extra)


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
