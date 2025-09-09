import asyncio
import contextvars
import json
import logging

from core.config import settings
from celium_collateral_contracts import CollateralContract

logger = logging.getLogger(__name__)

# Create a ContextVar to hold the context information
context = contextvars.ContextVar("context", default="ValidatorService")
context.set("ValidatorService")


def wait_for_services_sync(timeout=30):
    """Wait until PostgreSQL connections are working."""
    from sqlalchemy import create_engine, text

    from core.config import settings

    logger.info("Waiting for services to be available...")

    while True:
        try:
            # Check PostgreSQL connection using SQLAlchemy
            engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            logger.info("Connected to PostgreSQL.")

            break
        except Exception as e:
            logger.error("Failed to connect to PostgreSQL.")
            raise e


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
    miner_hotkey = settings.get_bittensor_wallet().get_hotkey().ss58_address

    logging.basicConfig(
        level=logging.INFO,
        format=f"Miner: {miner_hotkey} | Name: %(name)s | Time: %(asctime)s | Level: %(levelname)s | File: %(filename)s | Function: %(funcName)s | Line: %(lineno)s | Process: %(process)d | Message: %(message)s",
    )

    sqlalchemy_logger = logging.getLogger("sqlalchemy")
    sqlalchemy_logger.setLevel(logging.WARNING)

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

    # Create a handler for the logger
    handler = logging.StreamHandler()

    # Set the formatter for the handler
    handler.setFormatter(
        CustomFormatter("%(name)s %(asctime)s %(levelname)s %(filename)s %(process)d %(message)s")
    )


class StructuredMessage:
    def __init__(self, message, extra: dict):
        self.message = message
        self.extra = extra

    def __str__(self):
        return "%s >>> %s" % (self.message, json.dumps(self.extra))  # noqa


_m = StructuredMessage


def get_collateral_contract(
    miner_key: str = None,
    version: str = "1.0.1",
) -> CollateralContract:
    """
    Initializes and returns a CollateralContract instance.

    Args:
        version (str): Contract version to use (defaults to 1.0.1).
        miner_key (str): Optional miner key required for contract operations.

    Returns:
        CollateralContract: The initialized contract instance.
    """
    network = settings.BITTENSOR_NETWORK
    contract_address = settings.COLLATERAL_CONTRACT_ADDRESS  # Default address
    
    # Use version-specific address if available
    if version and settings.CONTRACT_VERSIONS.get(version):
        contract_address = settings.CONTRACT_VERSIONS.get(version)["address"]
    
    rpc_url = settings.SUBTENSOR_EVM_RPC_URL

    return CollateralContract(
        network=network,
        contract_address=contract_address,
        rpc_url=rpc_url,
        miner_key=miner_key,
    )
