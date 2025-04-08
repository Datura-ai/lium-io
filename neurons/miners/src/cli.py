import asyncio
import logging
import uuid

import click
import sqlalchemy

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
    executor_dao = ExecutorDao(session=next(get_db()))
    try:
        executor = executor_dao.save(
            Executor(uuid=uuid.uuid4(), address=address, port=port, validator=validator)
        )
    except Exception as e:
        logger.error("Failed in adding an executor: %s", str(e))
    else:
        logger.info("Added an executor(id=%s)", str(executor.uuid))


@cli.command()
@click.option("--address", prompt="IP Address", help="IP address of executor")
@click.option("--port", type=int, prompt="Port", help="Port of executor")
def remove_executor(address: str, port: int):
    """Remove executor machine to the database"""
    if click.confirm('Are you sure you want to remove this executor? This may lead to unexpected results'):
        logger.info("Removing executor (%s:%d)", address, port)
        executor_dao = ExecutorDao(session=next(get_db()))
        try:
            executor_dao.delete_by_address_port(address, port)
        except Exception as e:
            logger.error("Failed in removing an executor: %s", str(e))
        else:
            logger.info("Removed an executor(%s:%d)", address, port)
    else:
        logger.info("Executor removal cancelled.")


@cli.command()
@click.option("--address", prompt="IP Address", help="IP address of executor")
@click.option("--port", type=int, prompt="Port", help="Port of executor")
@click.option(
    "--validator", prompt="Validator Hotkey", help="Validator hotkey that executor opens to."
)
def switch_validator(address: str, port: int, validator: str):
    """Switch validator"""
    if click.confirm('Are you sure you want to switch validator? This may lead to unexpected results'):
        logger.info("Switching validator(%s) of an executor (%s:%d)", validator, address, port)
        executor_dao = ExecutorDao(session=next(get_db()))
        try:
            executor_dao.update(
                Executor(uuid=uuid.uuid4(), address=address, port=port, validator=validator)
            )
        except Exception as e:
            logger.error("Failed in switching validator: %s", str(e))
        else:
            logger.info("Validator switched")
    else:
        logger.info("Cancelled.")


@cli.command()
def show_executors():
    """Show executors to the database"""
    executor_dao = ExecutorDao(session=next(get_db()))
    try:
        for executor in executor_dao.get_all_executors():
            logger.info("%s:%d -> %s", executor.address, executor.port, executor.validator)
    except Exception as e:
        logger.error("Failed in showing an executor: %s", str(e))


if __name__ == "__main__":
    cli()
