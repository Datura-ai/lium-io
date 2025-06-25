import asyncio
import logging
import click
from services.cli_service import CliService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--private-key", prompt="Ethereum Private Key", hide_input=True, help="Ethereum private key")
def associate_eth(private_key: str):
    """Associate a miner's ethereum address with their hotkey."""
    logger.info("Please enter your Bittensor wallet password to unlock the hotkey for Ethereum association.")
    cli_service = CliService(private_key=private_key)
    success = cli_service.associate_ethereum_address()
    if success:
        logger.info("✅ Successfully associated ethereum address with hotkey")
    else:
        logger.error("❌ Failed to associate ethereum address with hotkey")


@cli.command()
@click.option("--address", prompt="IP Address", help="IP address of executor")
@click.option("--port", type=int, prompt="Port", help="Port of executor")
@click.option(
    "--validator", prompt="Validator Hotkey", help="Validator hotkey that executor opens to."
)
@click.option(
    "--deposit_amount", type=float, prompt="Deposit Amount", help="Amount of TAO to deposit as collateral"
)
@click.option("--private-key", prompt="Ethereum Private Key", hide_input=True, help="Ethereum private key")
def add_executor(address: str, port: int, validator: str, deposit_amount: float, private_key: str):
    """Add executor machine to the database"""
    cli_service = CliService(private_key=private_key, with_executor_db=True)
    success = asyncio.run(
        cli_service.add_executor(address, port, validator, deposit_amount)
    )
    if success:
        logger.info("✅ Added executor and deposited collateral successfully.")
    else:
        logger.error("❌ Failed to add executor or deposit collateral.")


@cli.command()
@click.option("--address", prompt="IP Address", help="IP address of executor")
@click.option("--port", type=int, prompt="Port", help="Port of executor")
@click.option(
    "--deposit_amount", type=float, prompt="Deposit Amount", help="Amount of TAO to deposit as collateral"
)
@click.option("--private-key", prompt="Ethereum Private Key", hide_input=True, help="Ethereum private key")
def deposit_collateral(address: str, port: int, deposit_amount: float, private_key: str):
    """You can deposit collateral for an existing executor on database"""
    cli_service = CliService(private_key=private_key, with_executor_db=True)
    success = asyncio.run(
        cli_service.deposit_collateral(address, port, deposit_amount)
    )
    if success:
        logger.info("✅ Deposited collateral successfully.")
    else:
        logger.error("❌ Failed to deposit collateral.")


@cli.command()
@click.option("--address", prompt="IP Address", help="IP address of executor")
@click.option("--port", type=int, prompt="Port", help="Port of executor")
def remove_executor(address: str, port: int):
    """Remove executor machine to the database"""
    if click.confirm('Are you sure you want to remove this executor? This may lead to unexpected results'):
        cli_service = CliService(with_executor_db=True)
        success = asyncio.run(cli_service.remove_executor(address, port))
        if success:
            logger.info(f"✅ Removed executor ({address}:{port})")
        else:
            logger.error(f"❌ Failed in removing an executor.")
    else:
        logger.info("Executor removal cancelled.")


@cli.command()
@click.option("--executor_uuid", prompt="Executor UUID", help="UUID of the executor to reclaim collateral from")
@click.option("--private-key", prompt="Ethereum Private Key", hide_input=True, help="Ethereum private key")
def reclaim_collateral(executor_uuid: str, private_key: str):
    """Reclaim collateral for a specific executor from the contract"""
    cli_service = CliService(private_key=private_key)
    success = asyncio.run(
        cli_service.reclaim_collateral(executor_uuid)
    )
    if success:
        logger.info("✅ Reclaimed collateral successfully.")
    else:
        logger.error("❌ Failed to reclaim collateral.")


@cli.command()
def show_executors():
    """Show executors to the database"""
    cli_service = CliService(with_executor_db=True)
    success = asyncio.run(cli_service.show_executors())
    if not success:
        logger.error("Failed in showing executors.")


@cli.command()
def get_miner_collateral():
    """Get miner collateral by summing up collateral from all registered executors"""
    cli_service = CliService(with_executor_db=True)
    success = asyncio.run(cli_service.get_miner_collateral())
    if not success:
        logger.error("❌ Failed in getting miner collateral.")


@cli.command()
@click.option("--address", prompt="IP Address", help="IP address of executor")
@click.option("--port", type=int, prompt="Port", help="Port of executor")
def get_executor_collateral(address: str, port: int):
    """Get collateral amount for a specific executor by address and port"""
    cli_service = CliService(with_executor_db=True)
    success = asyncio.run(cli_service.get_executor_collateral(address, port))
    if not success:
        logger.error("❌ Failed to get executor collateral.")


@cli.command()
def get_reclaim_requests():
    """Get reclaim requests for the current miner from the collateral contract"""
    cli_service = CliService(with_executor_db=True)
    success = asyncio.run(cli_service.get_reclaim_requests())
    if not success:
        logger.error("❌ Failed to get miner reclaim requests.")


@cli.command()
@click.option("--reclaim-request-id", prompt="Reclaim Request ID", type=int, help="ID of the reclaim request to finalize")
@click.option("--private-key", prompt="Ethereum Private Key", hide_input=True, help="Ethereum private key")
def finalize_reclaim_request(reclaim_request_id: int, private_key: str):
    """Finalize a reclaim request by its ID"""
    cli_service = CliService(private_key=private_key)
    success = asyncio.run(cli_service.finalize_reclaim_request(reclaim_request_id))
    if not success:
        logger.error("❌ Failed to finalize reclaim request.")


if __name__ == "__main__":
    cli()
