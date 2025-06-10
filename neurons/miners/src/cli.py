import asyncio
import logging
import uuid

import click
import bittensor
import aiohttp

from core.db import get_db
from daos.executor import ExecutorDao
from models.executor import Executor
from core.config import settings
from core.utils import get_collateral_contract

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
@click.option(
    "--deposit_amount", type=float, prompt="Deposit Amount", help="Amount of TAO to deposit as collateral"
)
def add_executor(address: str, port: int, validator: str, deposit_amount: float):
    """Add executor machine to the database"""
    logger.info("Adding a new executor (%s:%d) that opens to validator(%s)", address, port, validator)
    executor_dao = ExecutorDao(session=next(get_db()))
    executor_uuid = uuid.uuid4()
    try:
        executor = executor_dao.save(
            Executor(uuid=executor_uuid, address=address, port=port, validator=validator)
        )
    except Exception as e:
        logger.error("Failed to add executor: %s", str(e))
    else:
        logger.info("Added executor (id=%s)", str(executor.uuid))

    if deposit_amount < settings.REQUIRED_TAO_COLLATERAL:
        logger.error("Error: Minimum deposit amount is %f TAO.", settings.REQUIRED_TAO_COLLATERAL)
        return

    async def async_add_executor():
        try:
            collateral_contract = get_collateral_contract()
            my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()

            balance = await collateral_contract.get_balance(collateral_contract.miner_address)
            logger.info(f"Miner balance: {balance} TAO for miner hotkey {my_key.ss58_address}")

            if balance < deposit_amount:
                logger.error("Error: Insufficient balance in miner's address.")
                return

            message = (
                f"Deposit amount {deposit_amount} for this executor UUID: {executor_uuid}"
                f" since miner {my_key.ss58_address} is adding this executor"
            )
            logger.info(message)
            await collateral_contract.deposit_collateral(deposit_amount, str(executor_uuid))
        except Exception as e:
            logger.error("Failed to deposit collateral: %s", str(e))
        else:
            logger.info("Deposited collateral successfully.")
    asyncio.run(async_add_executor())


@cli.command()
@click.option("--address", prompt="IP Address", help="IP address of executor")
@click.option("--port", type=int, prompt="Port", help="Port of executor")
@click.option(
    "--deposit_amount", type=float, prompt="Deposit Amount", help="Amount of TAO to deposit as collateral"
)
def deposit_collateral(address: str, port: int, deposit_amount: float):
    """You can deposit collateral for an existing executor on database"""
    if deposit_amount < settings.REQUIRED_TAO_COLLATERAL:
        logger.error("Error: Minimum deposit amount is %f TAO.", settings.REQUIRED_TAO_COLLATERAL)
        return

    executor_dao = ExecutorDao(session=next(get_db()))

    async def async_deposit_collateral():
        try:
            executor = executor_dao.findOne(address, port)
            executor_uuid = executor.uuid
            collateral_contract = get_collateral_contract()
            my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
            balance = await collateral_contract.get_balance(collateral_contract.miner_address)

            logger.info(f"Miner balance: {balance} TAO for miner hotkey {my_key.ss58_address}")

            if balance < deposit_amount:
                logger.error("Error: Insufficient balance in miner's address.")
                return

            message = (
                f"Deposit amount {deposit_amount} for this executor UUID: {executor_uuid}"
                f" since miner {my_key.ss58_address} is going to add this executor"
            )
            logger.info(message)

            await collateral_contract.deposit_collateral(deposit_amount, str(executor_uuid))
        except Exception as e:
            logger.error("Failed to deposit collateral: %s", str(e))
        else:
            logger.info("Deposited collateral successfully.")
    asyncio.run(async_deposit_collateral())


@cli.command()
@click.option("--address", prompt="IP Address", help="IP address of executor")
@click.option("--port", type=int, prompt="Port", help="Port of executor")
@click.option("--reclaim_description", type=str, prompt="Reclaim Description", help="Reclaim Description")
def remove_executor(address: str, port: int, reclaim_description: str):
    """Remove executor machine to the database"""
    if click.confirm('Are you sure you want to remove this executor? This may lead to unexpected results'):
        logger.info("Removing executor (%s:%d)", address, port)
        executor_dao = ExecutorDao(session=next(get_db()))

        async def async_remove_executor():
            collateral_contract = get_collateral_contract()
            my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
            try:
                executor = executor_dao.findOne(address, port)
                executor_uuid = executor.uuid

                eligible_executors = await collateral_contract.get_eligible_executors()
                logger.info(f"All eligible executors: {eligible_executors}")
                logger.info(f"Current executor is eligible: {str(executor_uuid) in eligible_executors}")
                if not str(executor_uuid) in eligible_executors:
                    executor_dao.delete_by_address_port(address, port)
                    logger.info("Removed executor (%s:%d)", address, port)
                else:
                    logger.info(
                        "Need to reclaim deposited collateral from the collateral contract before removing the executor. "
                        "This ensures that any TAO collateral associated with the executor is properly reclaimed to avoid loss."
                    )
            except Exception as e:
                logger.error("Failed to remove executor: %s", str(e))
                return

            try:
                balance = await collateral_contract.get_balance(collateral_contract.miner_address)
                logger.info("Miner balance: %f TAO", balance)

                reclaim_amount = await collateral_contract.get_executor_collateral(executor_uuid)

                logger.info(
                    f"Executor {executor_uuid} is being removed by miner {my_key.ss58_address}. "
                    f"The total collateral of {reclaim_amount} TAO will be reclaimed from the collateral contract."
                )

                await collateral_contract.reclaim_collateral(reclaim_amount, reclaim_description, str(executor_uuid))
            except Exception as e:
                logger.error("Failed to reclaim collateral: %s", str(e))
        asyncio.run(async_remove_executor())
    else:
        logger.info("Executor removal cancelled.")


@cli.command()
def show_executors():
    """Show executors to the database"""
    executor_dao = ExecutorDao(session=next(get_db()))
    try:
        for executor in executor_dao.get_all_executors():
            logger.info("%s %s:%d -> %s", executor.uuid, executor.address, executor.port, executor.validator)
    except Exception as e:
        logger.error("Failed in showing an executor: %s", str(e))


@cli.command()
def get_miner_collateral():
    """Get miner collateral from the collateral contract"""

    async def async_get_miner_collateral():
        try:
            collateral_contract = get_collateral_contract()

            final_collateral = await collateral_contract.get_miner_collateral()

            logger.info("Miner collateral: %f TAO", final_collateral)

        except Exception as e:
            logger.error("Failed in getting miner collateral: %s", str(e))
    asyncio.run(async_get_miner_collateral())


@cli.command()
def get_eligible_executors():
    """Get eligible executors from the collateral contract"""
    async def async_get_eligible_executors():
        try:
            collateral_contract = get_collateral_contract()

            eligible_executors = await collateral_contract.get_eligible_executors()

            if not eligible_executors:
                logger.info("No eligible executors found.")
                return

            for executor in eligible_executors:
                logger.info("Eligible executor: %s", executor)

        except Exception as e:
            logger.error("Failed in getting eligible executors: %s", str(e))
    asyncio.run(async_get_eligible_executors())

@cli.command()
@click.option("--address", prompt="IP Address", help="IP address of executor")
@click.option("--port", type=int, prompt="Port", help="Port of executor")
def get_executor_collateral(address: str, port: int):
    """Get collateral amount for a specific executor by address and port"""
    executor_dao = ExecutorDao(session=next(get_db()))
    try:
        executor = executor_dao.findOne(address, port)
        executor_uuid = str(executor.uuid)
    except Exception as e:
        logger.error("Failed to find executor: %s", str(e))
        return

    async def async_get_executor_collateral():
        try:
            collateral_contract = get_collateral_contract()
            collateral = await collateral_contract.get_executor_collateral(executor_uuid)
            logger.info("Executor %s collateral: %f TAO from collateral contract", executor_uuid, collateral)
        except Exception as e:
            logger.error("Failed to get executor collateral: %s", str(e))
    asyncio.run(async_get_executor_collateral())


if __name__ == "__main__":
    cli()
