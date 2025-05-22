import asyncio
import logging
import uuid

import click
import sqlalchemy

from core.db import get_db
from daos.executor import ExecutorDao
from models.executor import Executor
from celium_collateral_contracts import CollateralContract
from core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


def initialize_collateral_contract():
    """Helper function to initialize CollateralContract"""
    network = "test" if settings.DEBUG_COLLATERAL_CONTRACT else "finney"
    collateral_contract = CollateralContract(
        network,
        settings.COLLATERAL_CONTRACT_ADDRESS,
        "",
        settings.ETHEREUM_MINER_KEY
    )
    
    return collateral_contract

def get_eth_address_from_hotkey(hotkey: str):
    # logger.info(f"Ethereum address for {hotkey}: {ethereum_address})
    return ""

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
        return
    else:
        logger.info("Added executor (id=%s)", str(executor.uuid))

    if deposit_amount < settings.REQUIRED_TAO_COLLATERAL:
        logger.error("Error: Minimum deposit amount is %f TAO.", settings.REQUIRED_TAO_COLLATERAL)
        return

    async def async_add_executor():
        try:
            collateral_contract = initialize_collateral_contract()
            my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
            collateral_contract.validator_address = get_eth_address_from_hotkey(validator)
            
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
    "--validator", prompt="Validator Hotkey", help="Validator hotkey that executor opens to."
)
@click.option(
    "--deposit_amount", type=float, prompt="Deposit Amount", help="Amount of TAO to deposit as collateral"
)
def deposit_collateral(address: str, port: int, validator: str, deposit_amount: float):
    """You can deposit collateral for an existing executor on database"""
    if deposit_amount < settings.REQUIRED_TAO_COLLATERAL:
        logger.error("Error: Minimum deposit amount is %f TAO.", settings.REQUIRED_TAO_COLLATERAL)
        return

    executor_dao = ExecutorDao(session=next(get_db()))

    async def async_deposit_collateral():
        try:
            executor = executor_dao.findOne(address, port)
            executor_uuid = executor.uuid

            collateral_contract = initialize_collateral_contract()
            my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
            collateral_contract.validator_address = get_eth_address_from_hotkey(validator)
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
@click.option(
    "--reclaim_amount", type=float, prompt="Reclaim Amount", help="Amount of TAO to reclaim collateral"
)
@click.option("--reclaim_description", type=str, prompt="Reclaim Description", help="Reclaim Description")
def remove_executor(address: str, port: int, reclaim_amount:float, reclaim_description: str, eth_private_key: str):
    """Remove executor machine to the database"""
    if click.confirm('Are you sure you want to remove this executor? This may lead to unexpected results'):
        logger.info("Removing executor (%s:%d)", address, port)
        executor_dao = ExecutorDao(session=next(get_db()))

        async def async_remove_executor():
            collateral_contract = initialize_collateral_contract()
            my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
            try:
                executor = executor_dao.findOne(address, port)
                executor_uuid = executor.uuid

                eligible_executors = await collateral_contract.get_eligible_executors(
                    [executor_uuid]
                )

                if len(eligible_executors) == 0:
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

                message = (
                    f"Reclaim amount {reclaim_amount} for this executor UUID: {executor_uuid}"
                    f" since miner {my_key.ss58_address} is removing this executor"
                )
                logger.info(message)
                await collateral_contract.reclaim_collateral(reclaim_amount, reclaim_description, str(executor_uuid))
            except Exception as e:
                logger.error("Failed to reclaim collateral: %s", str(e))
        asyncio.run(async_remove_executor())
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

            async def async_switch_validator():
                try:
                    collateral_contract = initialize_collateral_contract()
                    my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
                    new_validator_address = get_eth_address_from_hotkey(validator)

                    await collateral_contract.update_validator_for_miner(
                        new_validator=new_validator_address
                    )

                    logger.info("Switched validator on collateral contract successfully.")

                    validator_of_miner = await collateral_contract.get_validator_of_miner()
                    
                    logger.info(f"Updated validator of miner from collateral contract: {validator_of_miner}")
                except Exception as e:
                    logger.error("Failed in switching validator on collateral contract: %s", str(e))
            asyncio.run(async_switch_validator())
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
            logger.info("%s %s:%d -> %s", executor.uuid, executor.address, executor.port, executor.validator)
    except Exception as e:
        logger.error("Failed in showing an executor: %s", str(e))


@cli.command()
def get_miner_collateral():
    """Get miner collateral from the collateral contract"""

    async def async_get_miner_collateral():
        try:
            collateral_contract = initialize_collateral_contract()

            final_collateral = await collateral_contract.get_miner_collateral()

            final_collateral_in_tao = collateral_contract.w3.from_wei(final_collateral, "ether")

            logger.info("Miner collateral: %f TAO", final_collateral_in_tao)

        except Exception as e:
            logger.error("Failed in getting miner collateral: %s", str(e))
    asyncio.run(async_get_miner_collateral())


@cli.command()
def get_eligible_executors():
    """Get eligible executors from the collateral contract"""
    async def async_get_eligible_executors():
        try:
            executor_dao = ExecutorDao(session=next(get_db()))
            executors = executor_dao.get_all_executors()
            executor_uuids = [str(executor.uuid) for executor in executors]  # Convert to list of UUID strings

            collateral_contract = initialize_collateral_contract()

            my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()

            eligible_executors = await collateral_contract.get_eligible_executors(executor_uuids)

            for executor in eligible_executors:
                logger.info("Eligible executor: %s", executor)

        except Exception as e:
            logger.error("Failed in getting eligible executors: %s", str(e))
    asyncio.run(async_get_eligible_executors())


if __name__ == "__main__":
    cli()
