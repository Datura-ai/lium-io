import asyncio
import logging
import click
from eth_account import Account
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from services.cli_service import CliService
from core.config import settings
from core.utils import versions_holding_collateral, versions_with_open_reclaim

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def select_contract_version(prompt_title: str = "Available Contract Versions - Choose One") -> str:
    """
    Display contract versions table and prompt user to select one.
    
    Args:
        prompt_title (str): Title to display above the versions table
        
    Returns:
        str: Selected contract version (e.g., "1.0.2")
    """
    console = Console()
    
    table = Table(title=prompt_title)
    table.add_column("Option", justify="center", style="cyan", no_wrap=True)
    table.add_column("Version", justify="center", style="cyan", no_wrap=True)
    table.add_column("Address", style="magenta")
    table.add_column("Description", style="green")
    
    versions_list = list(settings.CONTRACT_VERSIONS.keys())
    for i, (version, details) in enumerate(settings.CONTRACT_VERSIONS.items(), 1):
        address = details["address"]
        info = details["info"]
        table.add_row(str(i), version, address, info)
    
    console.print(table)
    
    # Prompt user to select version by number
    while True:
        try:
            choice = click.prompt(f"\nSelect contract version (1-{len(versions_list)})", type=int)
            if 1 <= choice <= len(versions_list):
                selected_version = versions_list[choice - 1]
                logger.info(f"Selected version: {selected_version}")
                return selected_version
            else:
                logger.error(f"Invalid choice. Please select between 1 and {len(versions_list)}")
        except (ValueError, click.ClickException):
            logger.error(f"Invalid input. Please enter a number between 1 and {len(versions_list)}")


def display_contract_versions_table(title: str = "Available Contract Versions", highlight_current: bool = True) -> None:
    """
    Display a formatted table of all available contract versions.
    
    Args:
        title (str): Title to display above the table
        highlight_current (bool): Whether to highlight the current default version with a star
    """
    console = Console()
    
    table = Table(title=title)
    table.add_column("Version", justify="center", style="cyan", no_wrap=True)
    table.add_column("Address", style="magenta")
    table.add_column("Description", style="green")
    
    current_version = "1.0.2"  # Default version
    for version, details in settings.CONTRACT_VERSIONS.items():
        address = details["address"]
        info = details["info"]
        
        # Mark the current default version if requested
        if highlight_current:
            version_display = f"{version} ⭐" if version == current_version else version
        else:
            version_display = version
            
        table.add_row(version_display, address, info)
    
    console.print(table)


contract_option = click.option(
    "--contract",
    "contract_version",
    type=click.Choice(list(settings.CONTRACT_VERSIONS)),
    default=None,
    help="Collateral contract version (see show-contract-versions).",
)


def resolve_contract_version(
    contract_version: str | None, prompt_title: str, detected: list[str] | None = None
) -> str:
    """--contract when given, else the single detected version, else the interactive choice."""
    if contract_version:
        return contract_version
    if detected and len(detected) == 1:
        address = settings.CONTRACT_VERSIONS[detected[0]]["address"]
        logger.info("Using contract version %s (%s)", detected[0], address)
        return detected[0]
    return select_contract_version(prompt_title)


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
def show_contract_versions():
    """Show available contract versions with their addresses and information."""
    display_contract_versions_table()


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
def current_contract_version():
    """Get the current contract version"""
    cli_service = CliService()

    console = Console()
    contract_address = cli_service.collateral_contract.contract_address
    console.print(
        Panel.fit(
            f"[bold green]Current Contract Address:[/bold green]\n[bold yellow]{contract_address}[/bold yellow]",
            title="Contract Version Info",
            border_style="blue"
        )
    )


@cli.command()
@click.option("--address", prompt="IP Address", help="IP address of executor")
@click.option("--port", type=int, prompt="Port", help="Port of executor")
def remove_executor(address: str, port: int):
    """Remove executor machine to the database (once it holds no collateral on any contract version)"""
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
@contract_option
def reclaim_collateral(executor_uuid: str, private_key: str, contract_version: str | None):
    """Reclaim collateral for a specific executor from the contract that holds it"""
    detected = None
    if not contract_version:
        detected = asyncio.run(versions_holding_collateral(executor_uuid))
        if not detected:
            logger.error("❌ Executor %s holds no collateral on any contract version.", executor_uuid)
            return
    selected_version = resolve_contract_version(
        contract_version, "Contract Version Selection for Reclaim Collateral", detected
    )

    cli_service = CliService(private_key=private_key, version=selected_version)
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
        logger.info("✅ Successfully updated executor price.")
    else:
        logger.error("❌ Failed to update executor price.")


@cli.command()
@contract_option
def get_miner_collateral(contract_version: str | None):
    """Get miner collateral by summing up collateral from all registered executors"""
    
    selected_version = resolve_contract_version(contract_version, "Contract Version Selection for Miner Collateral")
    
    cli_service = CliService(with_executor_db=True, version=selected_version)
    success = asyncio.run(cli_service.get_miner_collateral())
    if not success:
        logger.error("❌ Failed in getting miner collateral.")


@cli.command()
@click.option("--address", prompt="IP Address", help="IP address of executor")
@click.option("--port", type=int, prompt="Port", help="Port of executor")
@contract_option
def get_executor_collateral(address: str, port: int, contract_version: str | None):
    """Get collateral amount for a specific executor by address and port"""
    
    selected_version = resolve_contract_version(contract_version, "Contract Version Selection for Executor Collateral")
    
    cli_service = CliService(with_executor_db=True, version=selected_version)
    success = asyncio.run(cli_service.get_executor_collateral(address, port))
    if not success:
        logger.error("❌ Failed to get executor collateral.")


@cli.command()
@contract_option
def get_reclaim_requests(contract_version: str | None):
    """Get reclaim requests for the current miner from the collateral contract"""
    
    selected_version = resolve_contract_version(contract_version, "Contract Version Selection for Reclaim Requests")
    
    cli_service = CliService(with_executor_db=True, version=selected_version)
    success = asyncio.run(cli_service.get_reclaim_requests())
    if not success:
        logger.error("❌ Failed to get miner reclaim requests.")


@cli.command()
@click.option("--reclaim-request-id", prompt="Reclaim Request ID", type=int, help="ID of the reclaim request to finalize")
@click.option("--private-key", prompt="Ethereum Private Key", hide_input=True, help="Ethereum private key")
@contract_option
def finalize_reclaim_request(reclaim_request_id: int, private_key: str, contract_version: str | None):
    """Finalize a reclaim request by its ID on the contract that holds it"""
    detected = None
    if not contract_version:
        miner_address = Account.from_key(private_key).address
        detected = asyncio.run(versions_with_open_reclaim(reclaim_request_id, miner_address))
        if not detected:
            logger.error("❌ No open reclaim request %d for this key on any contract version.", reclaim_request_id)
            return
    selected_version = resolve_contract_version(
        contract_version, "Contract Version Selection for Finalizing Reclaim Request", detected
    )

    cli_service = CliService(private_key=private_key, version=selected_version)
    success = asyncio.run(cli_service.finalize_reclaim_request(reclaim_request_id))
    if not success:
        logger.error("❌ Failed to finalize reclaim request.")
        
@cli.command()
def migrate_validator_hotkey():
    from core.config import settings
    from core.db import get_db
    from models.executor import Executor
    from sqlmodel import func, select, update

    old_validator_hotkey = "5E1nK3myeWNWrmffVaH76f2mCFCbe9VcHGwgkfdcD7k3E8D1"

    session = next(get_db())
    filters = [
        Executor.validator == old_validator_hotkey,
    ]
    select_stmt = select(func.count(Executor.uuid)).where(*filters)
    count = session.exec(select_stmt).one()
    logger.info(f"Found {count} miners with validator {old_validator_hotkey}")
    
    update_stmt = (
        update(Executor)
        .where(*filters)
        .values(validator=settings.DEFAULT_VALIDATOR_HOTKEY)
    )
    result = session.exec(update_stmt)
    updated_count = result.rowcount
    session.commit()
    logger.info(f"Migration completed. Updated {updated_count} miners")


if __name__ == "__main__":
    cli()
