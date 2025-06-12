import logging
import aiohttp
from datura.requests.miner_requests import ExecutorSSHInfo
from typing import Optional, List, Dict, Any
from core.utils import _m, context, get_extra_info
from core.utils import get_collateral_contract
from core.config import settings
from celium_collateral_contracts import CollateralContract

logger = logging.getLogger(__name__)


async def get_available_machines() -> Optional[List[Dict[str, Any]]]:
    """Get available machines from the compute REST API."""
    url = f"{settings.COMPUTE_REST_API_URL}/machines"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status != 200:
                    error_msg = await response.text()
                    logger.error(f"Error {response.status}: Unable to retrieve available machines. Details: {error_msg}")
                    return None
                data = await response.json()
                return data
    except Exception as e:
        logger.error(f"Exception while fetching available machines: {e}", exc_info=True)
        return None


class CollateralContractService:
    collateral_contract: CollateralContract

    def __init__(self):
        self.collateral_contract = get_collateral_contract()
        self.validator_hotkey = settings.get_bittensor_wallet().get_hotkey().ss58_address

    async def is_eligible_executor(
        self, 
        miner_hotkey: str,
        executor_info: ExecutorSSHInfo, 
        gpu_model: str
    ) -> bool:
        """
            Check if a specific executor is eligible.

            Args:
                miner_hotkey: The hotkey of the hotkey.
                executor_info: The executor information.
                gpu_model: The GPU model name.
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
            if not isinstance(eligible_executors, list):
                logger.error(_m(
                    "Eligible executors response is not a list",
                    extra=get_extra_info(default_extra),
                ))
                return False

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
            if deposit_amount is None or not isinstance(deposit_amount, (int, float)):
                logger.error(
                    _m(
                        "Deposit amount for GPU is invalid or missing",
                        extra=get_extra_info({
                            **default_extra,
                            "gpu_model": gpu_model,
                            "deposit_amount": str(deposit_amount),
                        })
                    )
                )
                return False

            executor_collateral = await self.collateral_contract.get_executor_collateral(executor_info.uuid)
            if executor_collateral is None:
                logger.error(
                    _m(
                        "Executor collateral is invalid or missing",
                        extra=get_extra_info({
                            **default_extra,
                            "executor_collateral": str(executor_collateral),
                        })
                    )
                )
                return False

            if float(executor_collateral) < float(deposit_amount):
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