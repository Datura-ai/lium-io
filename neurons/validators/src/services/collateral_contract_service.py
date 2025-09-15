import logging

from typing import Optional, Dict, Any
from core.utils import _m, get_extra_info, get_collateral_contract
from core.config import settings
from services.const import REQUIRED_DEPOSIT_AMOUNT
from clients.subtensor_client import SubtensorClient
from celium_collateral_contracts import CollateralContract

logger = logging.getLogger(__name__)


class CollateralContractService:
    def __init__(self):
        # Check for settings misconfiguration and handle gracefully
        self.collateral_contracts = {
            version: get_collateral_contract(version=version) for version in settings.CONTRACT_VERSIONS.keys()
        }
        self.subtensor_client = SubtensorClient.get_instance()

    async def _check_executor_collateral(
        self, 
        collateral_contract: CollateralContract,
        miner_hotkey: str,
        executor_uuid: str,
        gpu_model: str,
        gpu_count: int,
        default_extra: dict,
    ) -> tuple[bool, str | None]:
        evm_address = self.subtensor_client.get_evm_address_for_hotkey(miner_hotkey)

        if evm_address is None:
            error_message = f"No evm address found that is associated to this miner hotkey {miner_hotkey} in subnet"
            return False, error_message

        miner_address_on_contract = await collateral_contract.get_miner_address_of_executor(executor_uuid)

        if miner_address_on_contract is None:
            error_message = f"No miner address found on contract for executor {executor_uuid}"
            return False, error_message
        elif miner_address_on_contract.lower() != evm_address.lower():
            error_message = f"Miner address on contract ({miner_address_on_contract}) does not match EVM address ({evm_address}) for executor {executor_uuid}"
            return False, error_message

        self._log_info(
            f"Miner has deposited with EVM address {evm_address} on contract for executor {executor_uuid}",
            default_extra,
            miner_address_on_contract=miner_address_on_contract,
            evm_address=evm_address,
        )

        # Get deposit requirement for GPU model
        required_deposit_amount = await self._get_gpu_required_deposit(gpu_model, gpu_count)
        if required_deposit_amount is None:
            error_message = f"No required deposit amount found for GPU model {gpu_model}"
            return False, error_message

        # Check executor's actual collateral
        executor_collateral = await collateral_contract.get_executor_collateral(executor_uuid)
        if executor_collateral is None:
            error_message = "Executor doesn't have any collateral deposited on the contract"
            return False, error_message

        # Type check and conversion for collateral values
        try:
            executor_collateral_float = float(executor_collateral)
            required_deposit_amount_float = float(required_deposit_amount)
        except (TypeError, ValueError) as e:
            error_message = "Error converting collateral values to float"
            return False, error_message

        if executor_collateral_float < required_deposit_amount_float:
            error_message = f"{gpu_count} x {gpu_model} requires {required_deposit_amount} TAO, but the executor has only {executor_collateral} TAO deposited"
            return False, error_message
        else:
            self._log_info(
                f"This executor {executor_uuid} is eligible from collateral contract and therefore can have scores",
                default_extra,
                executor_collateral=str(executor_collateral),
                required_deposit_amount=str(required_deposit_amount),
            )

        return True, None


    async def is_eligible_executor(
        self,
        miner_hotkey: str,
        executor_uuid: str,
        gpu_model: str,
        gpu_count: int
    ) -> tuple[bool, str | None, str | None]:
        """
        Check if a specific executor is eligible.
        :param miner_hotkey: Miner hotkey
        :param executor_uuid: Executor UUID
        :param gpu_model: GPU model
        :param gpu_count: GPU count
        :return: Tuple containing eligibility status, error message, and contract version
        """
        error_message = None
        if gpu_model in settings.COLLATERAL_EXCLUDED_GPU_TYPES:
            logger.info(f"GPU model {gpu_model} is excluded from collateral checks")
            return True, None, settings.get_latest_contract_version()

        default_extra = {
            "miner_hotkey": miner_hotkey,
            "executor_uuid": executor_uuid,
        }
        for version in self.collateral_contracts.keys():
            default_extra[f"collateral_contract_address_{version}"] = getattr(self.collateral_contracts[version], "contract_address", None)

        try:
            error_message = ""
            for version in self.collateral_contracts.keys():
                try:
                    collateral_deposited, collateral_contract_error_message = await self._check_executor_collateral(
                        collateral_contract=self.collateral_contracts[version],
                        miner_hotkey=miner_hotkey,
                        executor_uuid=executor_uuid,
                        gpu_model=gpu_model,
                        gpu_count=gpu_count,
                        default_extra=default_extra,
                    )
                    if collateral_deposited:
                        return True, None, version
                    
                    error_message += f"Version: {version} — {collateral_contract_error_message} \n\n"
                except Exception as e:
                    error_message += f"Version: {version} — {str(e)} \n\n"

            return False, error_message, None
        except KeyError as e:
            error_message = f"KeyError encountered during eligibility check: {str(e)}"
            return False, error_message, None
        except Exception as e:
            error_message = f"❌ Error checking executor eligibility: {str(e)}"
            return False, error_message, None

    async def _get_gpu_required_deposit(self, gpu_model: str, gpu_count: int) -> Optional[float]:
        # Handle missing GPU model gracefully
        unit_tao_amount = REQUIRED_DEPOSIT_AMOUNT.get(gpu_model)
        if unit_tao_amount is None:
            return None
        required_deposit_amount = unit_tao_amount * gpu_count * settings.COLLATERAL_DAYS
        return round(required_deposit_amount, 6)

    def _log_error(self, message: str, extra: Dict[str, Any], exc_info: bool = False, **kwargs):
        full_extra = get_extra_info({**extra, **kwargs})
        logger.error(_m(message, extra=full_extra), exc_info=exc_info)

    def _log_info(self, message: str, extra: Dict[str, Any], exc_info: bool = False, **kwargs):
        full_extra = get_extra_info({**extra, **kwargs})
        logger.info(_m(message, extra=full_extra), exc_info=exc_info)
