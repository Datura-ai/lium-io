import asyncio
import logging
import uuid

import click
import bittensor

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
@click.option("--executor_uuid", prompt="Executor UUID", help="UUID of the executor to reclaim collateral from")
def reclaim_collateral(executor_uuid: str):
    """Reclaim collateral for a specific executor from the contract"""
    async def async_reclaim_collateral():
        collateral_contract = get_collateral_contract()
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        try:
            balance = await collateral_contract.get_balance(collateral_contract.miner_address)
            logger.info("Miner balance: %f TAO", balance)

            reclaim_amount = await collateral_contract.get_executor_collateral(executor_uuid)

            logger.info(
                f"Executor {executor_uuid} is being removed by miner {my_key.ss58_address}. "
                f"The total collateral of {reclaim_amount} TAO will be reclaimed from the collateral contract."
            )

            await collateral_contract.reclaim_collateral("Manual reclaim", executor_uuid)
        except Exception as e:
            logger.error("Failed to reclaim collateral: %s", str(e))
    asyncio.run(async_reclaim_collateral())


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
    """Get miner collateral by summing up collateral from all registered executors"""

    async def async_get_miner_collateral():
        try:
            executor_dao = ExecutorDao(session=next(get_db()))
            executors = executor_dao.get_all_executors()
            collateral_contract = get_collateral_contract()
            total_collateral = 0.0

            for executor in executors:
                executor_uuid = str(executor.uuid)
                collateral = await collateral_contract.get_executor_collateral(executor_uuid)
                total_collateral += float(collateral)
                logger.info("Executor %s collateral: %f TAO", executor_uuid, collateral)

            logger.info("Total miner collateral from all executors: %f TAO", total_collateral)

        except Exception as e:
            logger.error("Failed in getting miner collateral: %s", str(e))
    asyncio.run(async_get_miner_collateral())


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


@cli.command()
def get_reclaim_requests():
    """Get reclaim requests for the current miner from the collateral contract"""
    import json
    async def async_get_reclaim_requests():
        try:
            collateral_contract = get_collateral_contract()
            reclaim_requests = await collateral_contract.get_reclaim_events()
            if not reclaim_requests:
                print(json.dumps([]))
                return
            # Convert each reclaim request to dict if possible
            def to_dict(obj):
                if hasattr(obj, "__dict__"):
                    return dict(obj.__dict__)
                elif hasattr(obj, "_asdict"):  # namedtuple
                    return obj._asdict()
                else:
                    return dict(obj)
            json_output = [to_dict(req) for req in reclaim_requests]
            print(json.dumps(json_output, indent=4))
        except Exception as e:
            logger.error("Failed to get miner reclaim requests: %s", str(e))
    asyncio.run(async_get_reclaim_requests())


@cli.command()
@click.option("--reclaim_request_id", prompt="Reclaim Request ID", type=int, help="ID of the reclaim request to finalize")
def finalize_reclaim_request(reclaim_request_id: int):
    """Finalize a reclaim request by its ID"""
    async def async_finalize_reclaim_request():
        try:
            collateral_contract = get_collateral_contract()
            result = await collateral_contract.finalize_reclaim(reclaim_request_id)
            logger.info("Successfully finalized reclaim request: %s", reclaim_request_id)
            print(result)
        except Exception as e:
            logger.error("Failed to finalize reclaim request: %s", str(e))
    asyncio.run(async_finalize_reclaim_request())


if __name__ == "__main__":
    cli()
