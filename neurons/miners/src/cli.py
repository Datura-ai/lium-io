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


@cli.command()
@click.option("--address", prompt="IP Address", help="IP address of executor")
@click.option("--port", type=int, prompt="Port", help="Port of executor")
@click.option(
    "--validator", prompt="Validator Hotkey", help="Validator hotkey that executor opens to."
)
@click.option(
    "--deposit_amount", type=float, prompt="Deposit Amount", help="Amount of TAO to deposit as collateral"
)
@click.option("--eth_private_key", type=str, prompt="Ethereum Private Key", help="Ethereum private key of the miner")
def add_executor(address: str, port: int, validator: str, deposit_amount: float, eth_private_key: str):
    """Add executor machine to the database"""
    if deposit_amount < settings.REQUIRED_TAO_COLLATERAL:
        logger.error("Error: Minimum deposit amount is %f TAO.", REQUIRED_TAO_COLLATERAL)
        return

    logger.info("Add an new executor (%s:%d) that opens to validator(%s)", address, port, validator)
    executor_dao = ExecutorDao(session=next(get_db()))
    executor_uuid = uuid.uuid4()
    try:
        
        executor = executor_dao.save(
            Executor(executor_uuid, address=address, port=port, validator=validator)
        )
        
    except Exception as e:
        logger.error("Failed in adding an executor: %s", str(e))
    else:
        logger.info("Added an executor(id=%s)", str(executor.uuid))

    try:
        network = "test" if settings.DEBUG_COLLATERAL_CONTRACT else "finney"
        # Check if executor is eligible using collateral contract
        collateral_contract = CollateralContract(
            network,
            settings.COLLATERAL_CONTRACT_ADDRESS,
            "",
            eth_private_key
        )

        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()

        collateral_contract.map_hotkey_to_ethereum(
            account=collateral_contract.miner_account,
            hotkey=my_key.ss58_address
        )

        logger.info("Hotkey mapped to Ethereum address successfully.")

        collateral_contract.validator_address = collateral_contract.get_eth_address_from_hotkey(validator)
        logger.info(f"Validator address: {collateral_contract.validator_address} mapped to {validator}")
        balance = collateral_contract.get_balance(collateral_contract.miner_address)

        logger.info(f"Miner balance: {balance} TAO for miner hotkey {my_key.ss58_address}")

        if balance < deposit_amount:
            logger.error("Error: Insufficient balance in miner's address.")
            return

        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()

        message = (
            f"Deposit amount {deposit_amount} for this executor UUID: {executor_uuid}"
            f" since miner {my_key.ss58_address} is going to add this executor"
        )
        logger.info(message)

        collateral_contract.deposit_collateral(deposit_amount, str(executor_uuid))
    except Exception as e:
        logger.error("Failed in depositing collateral: %s", str(e))
    else:
        logger.info("Deposited collateral successfully.")

@cli.command()
@click.option("--address", prompt="IP Address", help="IP address of executor")
@click.option("--port", type=int, prompt="Port", help="Port of executor")
@click.option(
    "--reclaim_amount", type=float, prompt="Reclaim Amount", help="Amount of TAO to reclaim collateral"
)
@click.option("--reclaim_description", type=str, prompt="Reclaim Description", help="Reclaim Description")
@click.option("--eth_private_key", type=str, prompt="Ethereum Private Key", help="Ethereum private key of the miner")
def remove_executor(address: str, port: int, reclaim_amount:float, reclaim_description: str, eth_private_key: str):
    """Remove executor machine to the database"""
    if click.confirm('Are you sure you want to remove this executor? This may lead to unexpected results'):
        logger.info("Removing executor (%s:%d)", address, port)
        executor_dao = ExecutorDao(session=next(get_db()))
        try:
            executor_dao.delete_by_address_port(address, port)

            # Check if executor is eligible using collateral contract3
        except Exception as e:
            logger.error("Failed in removing an executor: %s", str(e))
        else:
            logger.info("Removed an executor(%s:%d) and initiated reclaim", address, port)

        try:
            network = "test" if settings.DEBUG_COLLATERAL_CONTRACT else "finney"

            collateral_contract = CollateralContract(
                network,
                settings.COLLATERAL_CONTRACT_ADDRESS,
                "",
                eth_private_key
            )

            my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()

            eth_address_from_hotkey = collateral_contract.get_eth_address_from_hotkey(my_key.ss58_address)
            logger.info(f"Miner address: {eth_address_from_hotkey} mapped to {my_key.ss58_address}")

            if eth_address_from_hotkey != collateral_contract.miner_address:
                logger.error(
                    "Error: The Ethereum address used for reclaiming collateral does not match the Ethereum address originally used for depositing collateral. "
                    "Please ensure you are using the correct Ethereum key associated with the deposit."
                )
                return

            balance = collateral_contract.get_balance(collateral_contract.miner_address)

            logger.info("Miner balance: %f TAO", balance)

            executor = executor_dao.findOne(address, port)
            executor_uuid = executor.uuid
            logger.info(f"Executor UUID: {executor_uuid}")
            message = (
                f"Reclaim amount {reclaim_amount} for this executor UUID: {executor_uuid}"
                f" since miner {my_key.ss58_address} is going to remove this executor"
            )
            logger.info(message)

            collateral_contract.reclaim_collateral(reclaim_amount, reclaim_description, str(executor.uuid))
        except Exception as e:
            logger.error("Failed in reclaiming collateral: %s", str(e))
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
            logger.info("%s %s:%d -> %s", executor.uuid, executor.address, executor.port, executor.validator)
    except Exception as e:
        logger.error("Failed in showing an executor: %s", str(e))


@cli.command()
@click.option("--eth_private_key", type=str, prompt="Ethereum Private Key", help="Ethereum private key of the miner")
def get_miner_collateral(eth_private_key: str):
    """Get miner collateral from the collateral contract"""

    try:
        network = "test" if settings.DEBUG_COLLATERAL_CONTRACT else "finney"
        # Check if executor is eligible using collateral contract
        collateral_contract = CollateralContract(
            network,
            settings.COLLATERAL_CONTRACT_ADDRESS,
            "",
            eth_private_key
        )

        final_collateral = collateral_contract.get_miner_collateral()

        final_collateral_in_tao = collateral_contract.w3.from_wei(final_collateral, "ether")

        logger.info("Miner collateral: %f TAO", final_collateral_in_tao)

    except Exception as e:
        logger.error("Failed in getting miner collateral: %s", str(e))


@cli.command()
def get_eligible_executors():
    """Get eligible executors from the collateral contract"""
    try:
        network = "test" if settings.DEBUG_COLLATERAL_CONTRACT else "finney"
        # Check if executor is eligible using collateral contract
        collateral_contract = CollateralContract(
            network,
            settings.COLLATERAL_CONTRACT_ADDRESS,
            "",
            ""
        )

        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        collateral_contract.miner_address = collateral_contract.get_eth_address_from_hotkey(my_key.ss58_address)
        eligible_executors = collateral_contract.get_eligible_executors()
        for executor in eligible_executors:
            logger.info("Eligible executor: %s", executor)

    except Exception as e:
        logger.error("Failed in getting eligible executors: %s", str(e))


@cli.command()
def get_eth_address_from_hotkey():
    """Get ethereum address from bittensor hotkey from the collateral contract"""
    try:
        network = "test" if settings.DEBUG_COLLATERAL_CONTRACT else "finney"
        collateral_contract = CollateralContract(
            network,
            settings.COLLATERAL_CONTRACT_ADDRESS,
            "",
            ""
        )

        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        collateral_contract.miner_address = collateral_contract.get_eth_address_from_hotkey(my_key.ss58_address)
        logger.info(f"Ethereum address: {collateral_contract.miner_address} mapped to hotkey {my_key.ss58_address}")

    except Exception as e:
        logger.error("Failed in getting ethereum address mapped to hotkey: %s", str(e))

if __name__ == "__main__":
    cli()
