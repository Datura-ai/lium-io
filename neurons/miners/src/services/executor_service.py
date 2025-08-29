import asyncio
import json
import logging
from typing import Annotated, Optional, Union

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
    SyncExecutorCentralMinerRequest,
    SyncExecutorCentralMinerSuccess,
    SyncExecutorCentralMinerFailed,
)

logging.basicConfig(level=logging.INFO)
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

    def sync_executor_miner_portal(self, request: SyncExecutorMinerPortalRequest) -> Union[SyncExecutorMinerPortalSuccess, SyncExecutorMinerPortalFailed]:
        try:
            for executor_payload in request.payload:
                executor = self.executor_dao.find_by_uuid(executor_payload.uuid)
                if executor:
                    executor.validator = executor_payload.validator
                    executor.address = executor_payload.address
                    executor.port = executor_payload.port
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

    def sync_executor_central_miner(self, miner_hotkey: str, request: SyncExecutorCentralMinerRequest) -> Union[SyncExecutorCentralMinerSuccess, SyncExecutorCentralMinerFailed]:
        try:
            executors = self.executor_dao.get_all_executors()
            return SyncExecutorCentralMinerSuccess(
                miner_hotkey=miner_hotkey,
                payload=executors,
            )
        except Exception as e:
            log_text = _m("Failed to sync executor central miner", extra={"error": str(e)})
            logger.error(log_text)
            return SyncExecutorCentralMinerFailed(
                error=str(log_text),
            )

    def get_executors_for_validator(self, validator_hotkey: str, executor_id: Optional[str] = None):
        return self.executor_dao.get_executors_for_validator(validator_hotkey, executor_id)

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
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        logger.error("API request failed to register SSH key. url=%s", url)
                        return None
                    response_obj: dict = await response.json()
                    logger.info(
                        "Get response from Executor(%s:%s): %s",
                        executor.address,
                        executor.port,
                        json.dumps(response_obj),
                    )
                    response_obj["uuid"] = str(executor.uuid)
                    response_obj["address"] = executor.address
                    response_obj["port"] = executor.port
                    return ExecutorSSHInfo.parse_obj(response_obj)
            except Exception as e:
                logger.error(
                    "API request failed to register SSH key. url=%s, error=%s", url, str(e)
                )

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
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        logger.error("API request failed to register SSH key. url=%s", url)
                        return None
            except Exception as e:
                logger.error(
                    "API request failed to register SSH key. url=%s, error=%s", url, str(e)
                )

    async def register_pubkey(self, validator_hotkey: str, pubkey: bytes, executor_id: Optional[str] = None):
        """Register pubkeys to executors for given validator.

        Args:
            validator_hotkey (str): Validator hotkey
            pubkey (bytes): SSH pubkey from validator.

        Return:
            List[dict/object]: Executors SSH connection infos that accepted validator pubkey.
        """
        tasks = [
            asyncio.create_task(
                self.send_pubkey_to_executor(executor, pubkey.decode("utf-8")),
                name=f"{executor}.send_pubkey_to_executor",
            )
            for executor in self.get_executors_for_validator(validator_hotkey, executor_id)
        ]

        total_executors = len(tasks)
        results = [
            result for result in await asyncio.gather(*tasks, return_exceptions=True) if result
        ]
        logger.info(
            "Send pubkey register API requests to %d executors and received results from %d executors",
            total_executors,
            len(results),
        )
        return results

    async def deregister_pubkey(self, validator_hotkey: str, pubkey: bytes, executor_id: Optional[str] = None):
        """Deregister pubkey from executors.

        Args:
            validator_hotkey (str): Validator hotkey
            pubkey (bytes): validator pubkey
        """
        tasks = [
            asyncio.create_task(
                self.remove_pubkey_from_executor(executor, pubkey.decode("utf-8")),
                name=f"{executor}.remove_pubkey_from_executor",
            )
            for executor in self.get_executors_for_validator(validator_hotkey, executor_id)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def get_pod_logs(
        self, validator_hotkey: str, executor_id: str, container_name: str
    ) -> list[PodLog]:
        executors = self.executor_dao.get_executors_for_validator(validator_hotkey, executor_id)
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
