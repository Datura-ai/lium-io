import asyncio
import logging
import sys
import click
from eth_account import Account
from services.cli_service import CliService
from core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def miner_account(private_key: str):
    """The key's account, or None after logging an error that leaves the key out."""
    try:
        return Account.from_key(private_key)
    except Exception:
        logger.error("❌ The Ethereum private key is malformed; expected 32 bytes as 64 hex characters.")
        return None


@click.group()
def cli():
    pass


@cli.command()
@click.option("--private-key", prompt="Ethereum Private Key", hide_input=True, help="Ethereum private key")
def associate_eth(private_key: str):
    """Associate a miner's ethereum address with their hotkey."""
    if miner_account(private_key) is None:
        sys.exit(1)
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
@click.option("--address", prompt="IP Address", help="IP address of executor")
@click.option("--port", type=int, prompt="Port", help="Port of executor")
@click.option(
    "--price", type=float, prompt="GPU price per hour (USD)", help="GPU price per hour in USD"
)
@click.option(
    "--validator", required=False, help="Validator hotkey that executor opens to."
)
def add_executor(
    address: str,
    port: int,
    price: float,
    validator: str | None = None,
):
    """Add executor machine to the database"""
    cli_service = CliService(with_executor_db=True)
    success = asyncio.run(cli_service.add_executor(address, port, price, validator))
    if success:
        logger.info("✅ Added executor successfully.")
    else:
        logger.error("❌ Failed to add executor.")


@cli.command()
@click.option("--address", prompt="IP Address", help="IP address of executor")
@click.option("--port", type=int, prompt="Port", help="Port of executor")
def remove_executor(address: str, port: int):
    """Remove executor machine from the database"""
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
@click.option("--address", prompt="IP Address", help="IP address of executor")
@click.option("--port", type=int, prompt="Port", help="Port of executor")
@click.option("--price", type=float, prompt="GPU price per hour (USD)", help="New price per GPU in USD")
def update_executor_price(address: str, port: int, price: float):
    """Update the price per GPU for an executor in USD"""
    if price < 0:
        logger.error("❌ Price cannot be negative.")
        return
    
    cli_service = CliService(with_executor_db=True)
    success = asyncio.run(cli_service.update_executor_price(address, port, price_per_gpu=price))
    if success:
        logger.info("✅ Price update saved. Check the marketplace after the next sync for the listed price.")
    else:
        logger.error("❌ Failed to update executor price.")


@cli.command()
@click.option(
    "--from-hotkey",
    "old_hotkey",
    default=None,
    help="The validator hotkey the rows are keyed to today. Default: VALIDATOR_NEXT_HOTKEY, which after the "
    "swap-day pair swap (DEFAULT_VALIDATOR_HOTKEY=<new>, VALIDATOR_NEXT_HOTKEY=<old>) is the previous hotkey.",
)
@click.option(
    "--to-hotkey",
    "new_hotkey",
    default=None,
    help="The validator hotkey the rows move to. Default: DEFAULT_VALIDATOR_HOTKEY (the active one).",
)
@click.option("--dry-run", is_flag=True, default=False, help="Count the rows that would move; change nothing.")
def migrate_validator_hotkey(old_hotkey: str | None, new_hotkey: str | None, dry_run: bool):
    """Re-key this miner's executor rows from one validator hotkey to another (the validator hotkey rotation:
    run once per miner DB after the config pair swap; the central miner has no rows of its own)."""
    from core.db import get_db
    from daos.executor import ExecutorDao
    from services.validator_service import migrate_validator_hotkey_rows

    old_hotkey = old_hotkey or settings.VALIDATOR_NEXT_HOTKEY
    new_hotkey = new_hotkey or settings.DEFAULT_VALIDATOR_HOTKEY
    executor_dao = ExecutorDao(session=next(get_db()))
    try:
        result = migrate_validator_hotkey_rows(executor_dao, old_hotkey, new_hotkey, dry_run=dry_run)
    except ValueError as e:
        raise click.UsageError(str(e)) from e

    logger.info(f"Found {result.found} executor rows keyed to validator {old_hotkey}")
    if dry_run:
        logger.info(f"Dry run: {result.found} rows would move to validator {new_hotkey}; nothing changed")
    else:
        logger.info(f"Migration completed. Moved {result.updated} executor rows to validator {new_hotkey}")


if __name__ == "__main__":
    cli()
