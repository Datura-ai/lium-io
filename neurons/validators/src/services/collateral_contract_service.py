import logging
import requests

from typing import Optional, List, Dict, Any
from datura.requests.miner_requests import ExecutorSSHInfo
from core.utils import _m, get_extra_info, get_collateral_contract
from core.config import settings
from services.const import REQUIRED_DEPOSIT_AMOUNT

logger = logging.getLogger(__name__)


class CollateralContractService:
    def __init__(self):
        self.collateral_contract = get_collateral_contract()
        self.validator_hotkey = settings.get_bittensor_wallet().get_hotkey().ss58_address

    async def is_eligible_executor(
        self,
        miner_hotkey: str,
        executor_info: ExecutorSSHInfo,
        gpu_model: str,
        gpu_count: int
    ) -> bool:
        """Check if a specific executor is eligible."""
        default_extra = {
            "collateral_contract_address": self.collateral_contract.contract_address,
            "owner_address": self.collateral_contract.owner_address,
            "miner_hotkey": miner_hotkey,
            "executor_uuid": executor_info.uuid,
        }

        try:
            from core.validator import Validator
            validator = Validator()
            evm_address = validator.get_associated_evm_address(miner_hotkey)

            if evm_address is None:
                self._log_error(
                    f"No evm address found that is associated to this miner hotkey {miner_hotkey} in subnet",
                    default_extra,
                )
                return False

            self._log_error(
                f"Evm address {evm_address} found that is associated to this miner hotkey {miner_hotkey}",
                default_extra,
            )
            # Get deposit requirement for GPU model
            required_deposit_amount = await self._get_gpu_required_deposit(gpu_model, gpu_count)
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
                    f"This executor {executor_info.uuid} is eligible from collateral contract and therefore can have scores",
                    default_extra,
                    executor_collateral=str(executor_collateral),
                    required_deposit_amount=str(required_deposit_amount),
                )

            return True

        except Exception as e:
            self._log_error("Error checking executor eligibility", default_extra, error=str(e), exc_info=True)
            return False


    async def _get_gpu_required_deposit(self, gpu_model: str, gpu_count:int) -> Optional[float]:
        unit_tao_amount = REQUIRED_DEPOSIT_AMOUNT[gpu_model]
        required_deposit_amount = unit_tao_amount * gpu_count * settings.COLLATERAL_DAYS
        return float(required_deposit_amount)

    def _log_error(self, message: str, extra: Dict[str, Any], exc_info: bool = False, **kwargs):
        full_extra = get_extra_info({**extra, **kwargs})
        logger.error(_m(message, extra=full_extra), exc_info=exc_info)

    def get_tao_price_in_usd(self) -> float:
        """Get tao price in usd."""
        response = requests.get(settings.TAO_PRICE_API_URL)
        rate_float = response.json()["market_data"]["current_price"]["usd"]
        return rate_float