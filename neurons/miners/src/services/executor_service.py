import asyncio
import json
import logging
from typing import Annotated, Optional, Union
from uuid import UUID

import aiohttp
import bittensor
from datura.requests.miner_requests import ExecutorSSHInfo, PodLog
from fastapi import Depends

from core.config import settings
from core.utils import _m, get_extra_info
from daos.executor import ExecutorDao
from models.executor import Executor

from protocol.miner_portal_request import (
    ExecutorAdded,
    AddExecutorFailed,
    SyncExecutorMinerPortalRequest,
    SyncExecutorMinerPortalSuccess,
    SyncExecutorMinerPortalFailed,
    SyncExecutorCentralMinerSuccess,
    SyncExecutorCentralMinerFailed,
    UpdateExecutorRequest,
    ExecutorUpdated,
    ExecutorUpdateFailed,
    DeleteExecutorRequest,
    ExecutorDeleted,
    ExecutorDeleteFailed,
)

logger = logging.getLogger(__name__)


class ExecutorService:
    def __init__(self, executor_dao: Annotated[ExecutorDao, Depends(ExecutorDao)]):
        self.executor_dao = executor_dao

    def create(self, executor: Executor) -> Union[ExecutorAdded, AddExecutorFailed]:
        try:
            self.executor_dao.save(executor)
            logger.info("Added executor (id=%s)", str(executor.uuid))
            return ExecutorAdded(
                executor_id=executor.uuid,
            )
        except Exception as e:
            log_text = _m(
                "❌ Failed to add executor",
                extra={
                    "executor_id": str(executor.uuid),
                    "address": executor.address,
                    "port": executor.port,
                    "validator": executor.validator,
                    "error": str(e),
                }
            )
            logger.error(log_text)
            return AddExecutorFailed(
                executor_id=executor.uuid,
                error=str(log_text),
            )

    def update(self, payload: UpdateExecutorRequest) -> Union[ExecutorUpdated, ExecutorUpdateFailed]:
        try:
            executor = Executor(
                uuid=payload.executor.uuid,
                validator=payload.executor.validator,
                address=payload.executor.address,
                port=payload.executor.port,
                price_per_hour=payload.executor.price_per_hour,
                price_per_gpu=payload.executor.price_per_gpu,
            )
            self.executor_dao.update_by_uuid(executor.uuid, executor)

            logger.info("Updated for executor (id=%s)", str(executor.uuid))
            return ExecutorUpdated(
                executor_id=executor.uuid,
            )
        except Exception as e:
            return ExecutorUpdateFailed(
                executor_id=executor.uuid,
                error=str(e),
            )

    def delete(self, payload: DeleteExecutorRequest) -> Union[ExecutorDeleted, ExecutorDeleteFailed]:
        executor_uuid = payload.executor.uuid
        try:
            self.executor_dao.delete_by_address_port(payload.executor.address, payload.executor.port)

            logger.info("delete for executor (id=%s)", str(executor_uuid))
            return ExecutorDeleted(
                executor_id=executor_uuid,
            )
        except Exception as e:
            return ExecutorDeleteFailed(
                executor_id=executor_uuid,
                error=str(e),
            )

    def sync_executor_miner_portal(self, request: SyncExecutorMinerPortalRequest) -> Union[SyncExecutorMinerPortalSuccess, SyncExecutorMinerPortalFailed]:
        try:
            for executor_payload in request.payload:
                executor = self.executor_dao.find_by_uuid(executor_payload.uuid)
                if executor:
                    executor.validator = executor_payload.validator
                    executor.address = executor_payload.address
                    executor.port = executor_payload.port
                    executor.price_per_hour = executor_payload.price_per_hour or executor.price_per_hour
                    executor.price_per_gpu = executor_payload.price_per_gpu or executor.price_per_gpu
                    self.executor_dao.update_by_uuid(executor.uuid, executor)
                    logger.info("Updated executor (id=%s)", str(executor.uuid))
                else:
                    logger.warning("Executor not found: %s:%s, adding new executor", executor_payload.address, executor_payload.port)
                    self.executor_dao.save(
                        Executor(
                            uuid=executor_payload.uuid,
                            validator=executor_payload.validator,
                            address=executor_payload.address,
                            port=executor_payload.port,
                            price_per_hour=executor_payload.price_per_hour,
                            price_per_gpu=executor_payload.price_per_gpu,
                        )
                    )

            return SyncExecutorMinerPortalSuccess()
        except Exception as e:
            log_text = _m(
                "Failed to sync executor miner portal",
                extra={
                    "error": str(e),
                }
            )
            logger.error(log_text)
            return SyncExecutorMinerPortalFailed(
                error=str(log_text),
            )

    def sync_executor_central_miner(self) -> Union[SyncExecutorCentralMinerSuccess, SyncExecutorCentralMinerFailed]:
        try:
            executors = self.executor_dao.get_all_executors()
            return SyncExecutorCentralMinerSuccess(
                payload=executors,
            )
        except Exception as e:
            log_text = _m("Failed to sync executor central miner", extra={"error": str(e)})
            logger.error(log_text)
            return SyncExecutorCentralMinerFailed(
                error=str(log_text),
            )

    async def get_executors_for_validator(self, validator_hotkey: str, miner_hotkey: str, executor_id: Optional[str] = None)  -> list[Executor]:
        # Standard mode: use local DB, in Standard mode
        if not settings.CENTRAL_MODE:
            return self.executor_dao.get_executors_for_validator(validator_hotkey, executor_id)

        # Central mode: fetch from portal
        from clients.miner_portal_api import MinerPortalAPI

        # Properly await the async HTTP call
        data = await MinerPortalAPI.fetch_executors(miner_hotkey, executor_id)
        logger.info(
            _m(
                "Fetched executors from portal",
                extra=get_extra_info({"miner_hotkey": miner_hotkey, "executor_id": executor_id, "data": data}),
            ),
        )

        # Expected fields per executor from portal: uuid, validator, address, port, price_per_hour
        result: list[Executor] = []
        for item in data:
            try:
                if item.get("validator_hotkey") != validator_hotkey:
                    continue

                result.append(
                    Executor(
                        uuid=UUID(item.get("id")),
                        validator=item.get("validator_hotkey"),
                        address=item.get("executor_ip_address"),
                        port=int(item.get("executor_ip_port")),
                        price_per_hour=item.get("price_per_hour"),
                        price_per_gpu=item.get("price_per_gpu"),
                    )
                )
            except Exception as e:
                logger.error(
                    _m(
                        "Failed to parse executor from portal",
                        extra=get_extra_info({"error": str(e), "item": str(item)}),
                    )
                )

        return result

    async def send_pubkey_to_executor(
        self, executor: Executor, pubkey: str
    ) -> ExecutorSSHInfo | None:
        """TODO: Send API request to executor with pubkey

        Args:
            executor (Executor): Executor instance that register validator hotkey
            pubkey (str): SSH public key from validator

        Return:
            response (ExecutorSSHInfo | None): Executor SSH connection info.
        """
        timeout = aiohttp.ClientTimeout(total=10)  # 5 seconds timeout
        url = f"http://{executor.address}:{executor.port}/upload_ssh_key"
        keypair: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        payload = {
            "public_key": pubkey,
            "data_to_sign": pubkey,
            "signature": f"0x{keypair.sign(pubkey).hex()}"
        }
        
        base_log_extra = {
            "executor_id": str(executor.uuid),
            "executor_address": executor.address,
            "executor_port": executor.port,
            "url": url,
        }
        logger.info(
            _m(
                "Sending pubkey to executor",
                extra=get_extra_info(base_log_extra),
            ),
        )
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error(
                            _m(
                                "API request failed to register SSH key - HTTP error",
                                extra=get_extra_info({**base_log_extra, "status": response.status, "error": text}),
                            ),
                        )
                        return None
                    response_obj: dict = await response.json()
                    logger.info(
                        _m(
                            "Received response from executor",
                            extra=get_extra_info({
                                **base_log_extra,
                                "response": response_obj,
                            }),
                        ),
                    )
                    response_obj = {
                        **response_obj,
                        **executor.model_dump(mode="json"),
                        "price": executor.price_per_hour,
                    }
                    return ExecutorSSHInfo.parse_obj(response_obj)
            except Exception as e:
                logger.error(
                    _m(
                        "API request failed to register SSH key - request exception",
                        extra=get_extra_info({
                            **base_log_extra,
                            "error": str(e),
                        }),
                    ),
                )
                return None

    async def remove_pubkey_from_executor(self, executor: Executor, pubkey: str):
        """TODO: Send API request to executor to cleanup pubkey

        Args:
            executor (Executor): Executor instance that needs to remove pubkey
        """
        timeout = aiohttp.ClientTimeout(total=10)  # 5 seconds timeout
        url = f"http://{executor.address}:{executor.port}/remove_ssh_key"
        keypair: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        payload = {
            "public_key": pubkey,
            "data_to_sign": pubkey,
            "signature": f"0x{keypair.sign(pubkey).hex()}"
        }
        base_log_extra = {
            "executor_id": str(executor.uuid),
            "executor_address": executor.address,
            "executor_port": executor.port,
            "url": url,
        }
        
        logger.info(
            _m(
                "Removing pubkey from executor",
                extra=get_extra_info(base_log_extra),
            ),
        )
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        logger.error(
                            _m(
                                "API request failed to register SSH key",
                                extra=get_extra_info({**base_log_extra, "status": response.status}),
                            ),
                        )
                        return None
            except Exception as e:
                logger.error(
                    _m(
                        "API request failed to register SSH key",
                        extra=get_extra_info({**base_log_extra, "error": str(e)}),
                    ),
                )

    async def register_pubkey(self, validator_hotkey: str, miner_hotkey: str, pubkey: bytes, executor_id: Optional[str] = None):
        """Register pubkeys to executors for given validator.

        Args:
            validator_hotkey (str): Validator hotkey
            pubkey (bytes): SSH pubkey from validator.

        Return:
            List[dict/object]: Executors SSH connection infos that accepted validator pubkey.
        """
        executors = await self.get_executors_for_validator(validator_hotkey, miner_hotkey, executor_id)
        tasks = [
            asyncio.create_task(
                self.send_pubkey_to_executor(executor, pubkey.decode("utf-8")),
                name=f"{executor}.send_pubkey_to_executor",
            )
            for executor in executors
        ]

        results = [
            result for result in await asyncio.gather(*tasks, return_exceptions=True) if result
        ]
        logger.info(
            _m( 
                "Sent pubkey register to executors",
                extra=get_extra_info({
                    "miner_hotkey": miner_hotkey,
                    "executor_id": executor_id,
                    "executor_ids": [str(executor.uuid) for executor in executors],
                    "executors": len(executors),
                    "accepted_executors": len(results),
                }),
            ),
        )
        return results

    async def deregister_pubkey(self, validator_hotkey: str, miner_hotkey: str, pubkey: bytes, executor_id: Optional[str] = None):
        """Deregister pubkey from executors.

        Args:
            validator_hotkey (str): Validator hotkey
            pubkey (bytes): validator pubkey
        """
        executors = await self.get_executors_for_validator(validator_hotkey, miner_hotkey, executor_id)
        tasks = [
            asyncio.create_task(
                self.remove_pubkey_from_executor(executor, pubkey.decode("utf-8")),
                name=f"{executor}.remove_pubkey_from_executor",
            )
            for executor in executors
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def get_pod_logs(
        self, validator_hotkey: str, miner_hotkey: str, executor_id: str, container_name: str
    ) -> list[PodLog]:
        executors = await self.get_executors_for_validator(validator_hotkey, miner_hotkey, executor_id)
        if len(executors) == 0:
            raise Exception('[get_pod_logs] Error: not found executor')

        executor = executors[0]

        timeout = aiohttp.ClientTimeout(total=20)  # 5 seconds timeout
        url = f"http://{executor.address}:{executor.port}/pod_logs"
        keypair: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        payload = {
            "container_name": container_name,
            "data_to_sign": container_name,
            "signature": f"0x{keypair.sign(container_name).hex()}"
        }
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    raise Exception('[get_pod_logs] Error: API request failed')

                response_obj: list[dict] = await response.json()
                pod_logs = [PodLog.parse_obj(item) for item in response_obj]
                return pod_logs
