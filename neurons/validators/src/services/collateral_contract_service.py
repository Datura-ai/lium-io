import logging
import aiohttp
from typing import Optional, List, Dict, Any

from datura.requests.miner_requests import ExecutorSSHInfo
from core.utils import _m, get_extra_info, get_collateral_contract
from core.config import settings

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
                try:
                    return await response.json()
                except aiohttp.ContentTypeError:
                    logger.error("Invalid JSON response from machine API.")
                    return None
    except Exception as e:
        logger.error(f"Exception while fetching available machines: {e}", exc_info=True)
        return None


class CollateralContractService:
    def __init__(self):
        self.collateral_contract = get_collateral_contract()
        self.validator_hotkey = settings.get_bittensor_wallet().get_hotkey().ss58_address

    async def is_eligible_executor(
        self, 
        miner_hotkey: str,
        executor_info: ExecutorSSHInfo, 
        gpu_model: str
    ) -> bool:
        """Check if a specific executor is eligible."""
        self.collateral_contract.miner_address = executor_info.ethereum_address
        default_extra = {
            "collateral_contract_address": self.collateral_contract.contract_address,
            "owner_address": self.collateral_contract.owner_address,
            "miner_address": executor_info.ethereum_address,
            "miner_hotkey": miner_hotkey,
            "executor_uuid": executor_info.uuid,
        }

        try:
            # Get deposit requirement for GPU model
            required_deposit_amount = await self._get_gpu_required_deposit(gpu_model, default_extra)
            if required_deposit_amount is None:
                return False

            # Check executor's actual collateral
            executor_collateral = await self.collateral_contract.get_executor_collateral(executor_info.uuid)
            if executor_collateral is None:
                self._log_error("Executor collateral is invalid or missing", default_extra)
                return False

            if float(executor_collateral) < float(required_deposit_amount):
                self._log_error(
                    "Executor collateral is less than required deposit amount",
                    default_extra,
                    executor_collateral=str(executor_collateral),
                    required_deposit_amount=str(required_deposit_amount),
                )
                return False
            else:
                self._log_error(
                    "Executor collateral meets or exceeds required deposit amount",
                    default_extra,
                    executor_collateral=str(executor_collateral),
                    required_deposit_amount=str(required_deposit_amount),
                )

            return True

        except Exception as e:
            self._log_error("Error checking executor eligibility", default_extra, error=str(e), exc_info=True)
            return False


    async def _get_gpu_required_deposit(self, gpu_model: str, extra: Dict[str, Any]) -> Optional[float]:
        machines = await get_available_machines()
        if machines is None:
            logger.error("Could not fetch available machines for GPU pricing.")
            return None

        gpu_entry = next((g for g in machines if g.get("name") == gpu_model), None)
        if not gpu_entry:
            logger.error(f"GPU '{gpu_model}' not found in available machines.")
            return None

        required_deposit_amount = gpu_entry.get("base_price")
        if required_deposit_amount is None or not isinstance(required_deposit_amount, (int, float)):
            self._log_error(
                "Deposit amount for GPU is invalid or missing",
                extra,
                gpu_model=gpu_model,
                deposit_amount=required_deposit_amount
            )
            return None

        return float(required_deposit_amount)

    def _log_error(self, message: str, extra: Dict[str, Any], exc_info: bool = False, **kwargs):
        full_extra = get_extra_info({**extra, **kwargs})
        logger.error(_m(message, extra=full_extra), exc_info=exc_info)
