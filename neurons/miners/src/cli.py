import logging
import uuid

import click

from core.db import get_db
from daos.executor import ExecutorDao
from models.executor import Executor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--address", prompt="IP Address", help="IP address of executor")
@click.option("--port", type=int, prompt="Port", help="Port of executor")
@click.option(
    "--validator", prompt="Validator Hotkey", help="Validator hotkey that executor opens to."
)
def add_executor(address: str, port: int, validator: str):
    """Add executor machine to the database"""
    logger.info("Add an new executor (%s:%d) that opens to validator(%s)", address, port, validator)
    # Get the session from the generator
    session = next(get_db())

    executor_dao = ExecutorDao(session=session)
    executor = executor_dao.save(
        Executor(uuid=uuid.uuid4(), address=address, port=port, validator=validator)
    )
    logger.info("Added an executor(id=%s)", str(executor.uuid))


if __name__ == "__main__":
    cli()
