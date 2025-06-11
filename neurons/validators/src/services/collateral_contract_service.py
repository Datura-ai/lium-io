from celium_collateral_contracts import CollateralContract
import logging
import aiohttp
from core.utils import get_collateral_contract
from core.config import settings
from core.utils import _m, context, get_extra_info
from datura.requests.miner_requests import ExecutorSSHInfo

logger = logging.getLogger(__name__)

async def get_available_machines():
    """Get Ethereum address for a given hotkey"""
    url = f"{settings.COMPUTE_REST_API_URL}/machines"
 
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
            if response.status != 200:
                error_msg = await response.text()
                logger.error(f"Error {response.status}: Unable to retrieve available machines. Details: {error_msg}")
                return None
            data = await response.json()
            return data
            
class CollateralContractService:
    collateral_contract: CollateralContract

    def __init__(self):
        self.collateral_contract = get_collateral_contract()
        self.validator_hotkey = settings.get_bittensor_wallet().get_hotkey().ss58_address

    async def is_eligible_executor(self, miner_hotkey: str, executor_info: ExecutorSSHInfo, gpu_model: str):
        """
            Check if it is eligible for a specific executor.

            Args:
                miner_hotkey: The hotkey of the hotkey.
                executor_info: The executor information.
                gpu_model: The gpu model name
        """

        default_extra = {
            "collateral_contract_address": self.collateral_contract.contract_address,
            "owner_address": self.collateral_contract.owner_address,
            "miner_address": executor_info.ethereum_address,
            "miner_hotkey": miner_hotkey,
            "executor_uuid": executor_info.uuid,
        }

        try:
            self.collateral_contract.miner_address = executor_info.ethereum_address

            eligible_executors = await self.collateral_contract.get_eligible_executors()

            if executor_info.uuid not in eligible_executors:
                logger.error(_m(
                    "Executor is not eligible based on collateral contract",
                    extra=get_extra_info(default_extra),
                ))
                return False
            else:
                logger.info(_m(
                    "Executor is eligible based on collateral contract",
                    extra=get_extra_info(default_extra),
                ))
            
            machines = await get_available_machines()
            if not machines:
                logger.error("Could not fetch available machines for GPU pricing.")
                return False

            gpu_entry = next((g for g in machines if g.get("name") == gpu_model), None)

            if not gpu_entry:
                logger.error(f"GPU '{gpu_model}' not found in available machines.")
                return False

            deposit_amount = gpu_entry.get("base_price")
            executor_collateral = await self.collateral_contract.get_executor_collateral(executor_info.uuid)
            if executor_collateral < deposit_amount:
                logger.error(
                    _m(
                        "Executor collateral is less than required deposit amount",
                        extra=get_extra_info({
                            **default_extra,
                            "executor_collateral": str(executor_collateral),
                            "required_deposit_amount": str(deposit_amount),
                        })
                    )
                )
                return False
                
            return True
        except Exception as e:
            logger.error(
                _m(
                    "Error checking executor eligibility",
                    extra=get_extra_info(
                        {
                            **default_extra,
                            "error": str(e),
                        }
                    ),
                ),
                exc_info=True,
            )
            return False

    async def handle_reclaim_requests(self, executor_info: ExecutorSSHInfo, is_rented: bool):
        """
            Handle reclaim requests for a specific executor.

            Args:
                executor_info: Information about the executor.
        """
        try:
            reclaim_requests = await self.collateral_contract.get_reclaim_requests()

            message = (
                f"Total reclaim requests count: {len(reclaim_requests)} "
            )
            logger.info(message)

            for request in reclaim_requests:
                default_extra = {
                    "collateral_contract_address": self.collateral_contract.contract_address,
                    "validator_hotkey": self.validator_hotkey,
                    "executor_uuid": executor_info.uuid,
                    "request_id": request.reclaim_request_id,
                    "request_executor_uuid": request.executor_uuid,
                }

                logger.info(
                    _m(
                        "Reclaim request",
                        extra={
                            "validator_hotkey": self.validator_hotkey,
                            "executor_uuid": executor_info.uuid,
                            "request_id": request.reclaim_request_id,
                            "request_executor_uuid": request.executor_uuid,
                        },
                    )
                )

                if request.executor_uuid == executor_info.uuid.replace("-", ""):
                    if is_rented:
                        message = (
                            f"Validator {self.validator_hotkey} denied this reclaim request "
                            f"since executor is rented for this executor UUID: {executor_info.uuid}"
                        )
                        logger.info(message)
                        await self.collateral_contract.deny_reclaim_request(
                            request.reclaim_request_id, message
                        )
        except Exception as e:
            logger.error(
                _m(
                    "Error handling reclaim requests",
                    extra=get_extra_info(
                        {
                            **default_extra,
                            "error": str(e),
                        }
                    ),
                ),
                exc_info=True,
            )

    async def slash_collateral(self, miner_hotkey: str, executor_info: ExecutorSSHInfo):
        """
            Slash collateral for a specific executor.

            Args:
                miner_hotkey: The hotkey of the hotkey.
                executor_info: The executor information.
        """
        default_extra = {
            "collateral_contract_address": self.collateral_contract.contract_address,
            "owner_address": self.collateral_contract.owner_address,
            "miner_address": self.collateral_contract.miner_address,
            "miner_hotkey": miner_hotkey,
            "executor_uuid": executor_info.uuid,
        }
        try:
            self.collateral_contract.miner_address = executor_info.ethereum_address

            # Log the miner's balance
            balance = await self.collateral_contract.get_balance(self.collateral_contract.miner_address)
            logger.info(f"Miner balance: {balance} TAO")

            executor_collateral = await self.collateral_contract.get_executor_collateral(executor_info.uuid)
            logger.info(f"Collateral amount: {executor_collateral} TAO for this executor {executor_info.uuid}")

            # Log and perform the collateral slashing
            logger.info(
                _m(
                    f"Validator {self.validator_hotkey} is slashing collateral of this executor UUID: {executor_info.uuid}",
                    extra=get_extra_info(default_extra),
                )
            )

            await self.collateral_contract.slash_collateral(executor_collateral, "slashit", executor_info.uuid)
            logger.info(
                _m(
                    f"Validator {self.validator_hotkey} slashed collateral of this executor UUID: {executor_info.uuid}",
                    extra=get_extra_info(default_extra),
                )
            )
        except Exception as e:
            logger.error(
                _m(
                    "Error slashing collateral",
                    extra=get_extra_info(
                        {
                            **default_extra,
                            "error": str(e),
                        }
                    ),
                ),
                exc_info=True,
            )
