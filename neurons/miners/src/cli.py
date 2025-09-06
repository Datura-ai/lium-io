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
    cli_service = CliService(private_key=private_key)
    success = cli_service.associate_ethereum_address()
    if success:
        logger.info("✅ Successfully associated ethereum address with hotkey")
    else:
        logger.error("❌ Failed to associate ethereum address with hotkey")


@cli.command()
def get_associated_evm_address():
    """Get the associated EVM address for the Bittensor hotkey."""
    cli_service = CliService()
    cli_service.get_associated_evm_address()


@cli.command()
@click.option("--private-key", prompt="Ethereum Private Key", hide_input=True, help="Ethereum private key")
def get_eth_ss58_address(private_key: str):
    """Associate a miner's ethereum address with their hotkey."""
    cli_service = CliService(private_key=private_key)
    ss58_address = cli_service.get_eth_ss58_address()
    print(ss58_address)


@cli.command()
@click.option(
    "--amount", type=float, required=False, help="Amount of TAO to transfer"
)
@click.option("--private-key", prompt="Ethereum Private Key", hide_input=True, help="Ethereum private key")
def transfer_tao_to_eth_address(private_key: str, amount: float):
    """Associate a miner's ethereum address with their hotkey."""
    cli_service = CliService(private_key=private_key)
    cli_service.transfer_tao_to_eth_address(amount)


@cli.command()
@click.option("--private-key", prompt="Ethereum Private Key", hide_input=True, help="Ethereum private key")
def get_balance_of_eth_address(private_key: str):
    """Get the balance of the Eth address for the Bittensor hotkey."""
    cli_service = CliService(private_key=private_key)
    asyncio.run(cli_service.get_balance_of_eth_address())


@cli.command()
@click.option("--address", prompt="IP Address", help="IP address of executor")
@click.option("--port", type=int, prompt="Port", help="Port of executor")
@click.option(
    "--validator", prompt="Validator Hotkey", help="Validator hotkey that executor opens to."
)
@click.option(
    "--gpu-type", help="Type of GPU", required=False
)
@click.option(
    "--gpu-count", type=int, help="Number of GPUs", required=False
)
@click.option(
    "--deposit-amount", type=float, required=False, help="Amount of TAO to deposit as collateral (optional)"
)
@click.option("--private-key", required=False, hide_input=True, help="Ethereum private key")
def add_executor(
    address: str,
    port: int,
    validator: str,
    gpu_type: str | None = None,
    gpu_count: int | None = None,
    private_key: str | None = None,
    deposit_amount: float | None = None,
):
    """Add executor machine to the database"""
    if gpu_type is not None or gpu_count is not None or deposit_amount is not None:
        if not private_key:
            logger.error("Private key is required to deposit collateral.")
            return

    cli_service = CliService(private_key=private_key, with_executor_db=True)
    success = asyncio.run(
        cli_service.add_executor(address, port, validator, deposit_amount, gpu_type, gpu_count)
    )
    if success:
        logger.info("✅ Added executor and deposited collateral successfully.")
    else:
        logger.error("❌ Failed to add executor or deposit collateral.")


@cli.command()
@click.option("--address", prompt="IP Address", help="IP address of executor")
@click.option("--port", type=int, prompt="Port", help="Port of executor")
@click.option(
    "--gpu-type", prompt="GPU Type", help="Type of GPU"
)
@click.option(
    "--gpu-count", type=int, prompt="GPU Count", help="Number of GPUs"
)
@click.option(
    "--deposit-amount", type=float, required=False, help="Amount of TAO to deposit as collateral (optional)"
)
@click.option("--private-key", prompt="Ethereum Private Key", hide_input=True, help="Ethereum private key")
def deposit_collateral(address: str, port: int, gpu_type: str, gpu_count: int, private_key: str, deposit_amount: float = None):
    """You can deposit collateral for an existing executor on database"""
    cli_service = CliService(private_key=private_key, with_executor_db=True)
    success = asyncio.run(
        cli_service.deposit_collateral(address, port, deposit_amount, gpu_type, gpu_count)
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
@click.option("--require-old-contract", is_flag=True, default=False, help="Require old collateral contract (optional, default: False)")
def reclaim_collateral(executor_uuid: str, private_key: str, require_old_contract: bool):
    """Reclaim collateral for a specific executor from the contract"""
    cli_service = CliService(private_key=private_key, require_old_contract=require_old_contract)
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
@click.option("--address", prompt="IP Address", help="IP address of executor")
@click.option("--port", type=int, prompt="Port", help="Port of executor")
@click.option(
    "--validator", prompt="Validator Hotkey", help="Validator hotkey that executor opens to."
)
def switch_validator(address: str, port: int, validator: str):
    """Switch validator"""
    if click.confirm('Are you sure you want to switch validator? This may lead to unexpected results'):
        logger.info("Switching validator(%s) of an executor (%s:%d)", validator, address, port)
        cli_service = CliService(with_executor_db=True)
        asyncio.run(cli_service.switch_validator(address, port, validator))
    else:
        logger.info("Cancelled.")


@cli.command()
@click.option("--require-old-contract", is_flag=True, default=False, help="Require old collateral contract (optional, default: False)")
def get_miner_collateral(require_old_contract: bool):
    """Get miner collateral by summing up collateral from all registered executors"""
    cli_service = CliService(with_executor_db=True, require_old_contract=require_old_contract)
    success = asyncio.run(cli_service.get_miner_collateral())
    if not success:
        logger.error("❌ Failed in getting miner collateral.")


@cli.command()
@click.option("--address", prompt="IP Address", help="IP address of executor")
@click.option("--port", type=int, prompt="Port", help="Port of executor")
@click.option("--require-old-contract", is_flag=True, default=False, help="Require old collateral contract (optional, default: False)")
def get_executor_collateral(address: str, port: int, require_old_contract: bool):
    """Get collateral amount for a specific executor by address and port"""
    cli_service = CliService(with_executor_db=True, require_old_contract=require_old_contract)
    success = asyncio.run(cli_service.get_executor_collateral(address, port))
    if not success:
        logger.error("❌ Failed to get executor collateral.")


@cli.command()
@click.option("--require-old-contract", is_flag=True, default=False, help="Require old collateral contract (optional, default: False)")
def get_reclaim_requests(require_old_contract: bool):
    """Get reclaim requests for the current miner from the collateral contract"""
    cli_service = CliService(with_executor_db=True, require_old_contract=require_old_contract)
    success = asyncio.run(cli_service.get_reclaim_requests())
    if not success:
        logger.error("❌ Failed to get miner reclaim requests.")


@cli.command()
@click.option("--reclaim-request-id", prompt="Reclaim Request ID", type=int, help="ID of the reclaim request to finalize")
@click.option("--private-key", prompt="Ethereum Private Key", hide_input=True, help="Ethereum private key")
@click.option("--require-old-contract", is_flag=True, default=False, help="Require old collateral contract (optional, default: False)")
def finalize_reclaim_request(reclaim_request_id: int, private_key: str, require_old_contract: bool):
    """Finalize a reclaim request by its ID"""
    cli_service = CliService(private_key=private_key, require_old_contract=require_old_contract)
    success = asyncio.run(cli_service.finalize_reclaim_request(reclaim_request_id))
    if not success:
        logger.error("❌ Failed to finalize reclaim request.")


if __name__ == "__main__":
    cli()
