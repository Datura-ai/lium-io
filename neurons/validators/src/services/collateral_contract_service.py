import logging

from typing import Optional, Dict, Any
from core.utils import _m, get_extra_info, get_collateral_contract
from core.config import settings
from services.const import REQUIRED_DEPOSIT_AMOUNT
from clients.subtensor_client import SubtensorClient

logger = logging.getLogger(__name__)


class CollateralContractService:
    def __init__(self):
        # Check for settings misconfiguration and handle gracefully
        self.collateral_contract = get_collateral_contract()
        self.subtensor_client = SubtensorClient.get_instance()

    async def is_eligible_executor(
        self,
        miner_hotkey: str,
        executor_uuid: str,
        gpu_model: str,
        gpu_count: int
    ) -> bool:
        """Check if a specific executor is eligible."""
        default_extra = {
            "collateral_contract_address": self.collateral_contract.contract_address,
            "owner_address": self.collateral_contract.owner_address,
            "miner_hotkey": miner_hotkey,
            "executor_uuid": executor_uuid,
        }

        try:
            evm_address_map = self.subtensor_client.evm_address_map

            self._log_info(
                "Evm Address Map Information of Miner Hotkeys",
                default_extra,
                evm_address_map=evm_address_map,
            )

            if miner_hotkey in evm_address_map:
                evm_address = evm_address_map[miner_hotkey]
                self._log_info(
                    f"Evm address {evm_address} found that is associated to this miner hotkey {miner_hotkey}",
                    default_extra,
                )

                miner_address_on_contract = await self.collateral_contract.get_miner_address_of_executor(executor_uuid)

                if evm_address is None:
                    self._log_info(
                        f"No evm address found in this subnet for this miner {miner_hotkey}",
                        default_extra,
                    )
                    return False
                elif miner_address_on_contract is None:
                    self._log_info(
                        f"No miner address found on contract for executor {executor_uuid}",
                        default_extra,
                    )
                    return False
                elif miner_address_on_contract.lower() == evm_address.lower():
                    self._log_info(
                        f"Miner has deposited with EVM address {evm_address} on contract for executor {executor_uuid}",
                        default_extra,
                        miner_address_on_contract=miner_address_on_contract,
                        evm_address=evm_address,
                    )
                else:
                    self._log_info(
                        f"Miner address on contract ({miner_address_on_contract}) does not match EVM address ({evm_address}) for executor {executor_uuid}",
                        default_extra,
                        miner_address_on_contract=miner_address_on_contract,
                        evm_address=evm_address,
                    )
                    return False
            else:
                self._log_info(
                    f"No evm address found that is associated to this miner hotkey {miner_hotkey} in subnet",
                    default_extra,
                )
                return False

            # Get deposit requirement for GPU model
            required_deposit_amount = await self._get_gpu_required_deposit(gpu_model, gpu_count)
            if required_deposit_amount is None:
                self._log_info(
                    f"No required deposit amount found for GPU model {gpu_model}",
                    default_extra,
                )
                return False

            # Check executor's actual collateral
            executor_collateral = await self.collateral_contract.get_executor_collateral(executor_uuid)
            if executor_collateral is None:
                self._log_info("Executor collateral is invalid or missing", default_extra)
                return False

            # Type check and conversion for collateral values
            try:
                executor_collateral_float = float(executor_collateral)
                required_deposit_amount_float = float(required_deposit_amount)
            except (TypeError, ValueError) as e:
                self._log_error(
                    "Error converting collateral values to float",
                    default_extra,
                    error=str(e),
                )
                return False

            if executor_collateral_float < required_deposit_amount_float:
                self._log_info(
                    "Executor collateral is less than required deposit amount",
                    default_extra,
                    executor_collateral=str(executor_collateral),
                    required_deposit_amount=str(required_deposit_amount),
                )
                return False
            else:
                self._log_info(
                    f"This executor {executor_uuid} is eligible from collateral contract and therefore can have scores",
                    default_extra,
                    executor_collateral=str(executor_collateral),
                    required_deposit_amount=str(required_deposit_amount),
                )

            return True

        except KeyError as e:
            self._log_error("KeyError encountered during eligibility check", default_extra, error=str(e), exc_info=True)
            return False
        except Exception as e:
            self._log_error("❌ Error checking executor eligibility", default_extra, error=str(e), exc_info=True)
            return False

    async def _get_gpu_required_deposit(self, gpu_model: str, gpu_count:int) -> Optional[float]:
        # Handle missing GPU model gracefully
        unit_tao_amount = REQUIRED_DEPOSIT_AMOUNT.get(gpu_model)
        if unit_tao_amount is None:
            return None
        required_deposit_amount = unit_tao_amount * gpu_count * settings.COLLATERAL_DAYS
        return float(required_deposit_amount)

    def _log_error(self, message: str, extra: Dict[str, Any], exc_info: bool = False, **kwargs):
        full_extra = get_extra_info({**extra, **kwargs})
        logger.error(_m(message, extra=full_extra), exc_info=exc_info)

    def _log_info(self, message: str, extra: Dict[str, Any], exc_info: bool = False, **kwargs):
        full_extra = get_extra_info({**extra, **kwargs})
        logger.info(_m(message, extra=full_extra), exc_info=exc_info)