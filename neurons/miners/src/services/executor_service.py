import asyncio
import logging
from typing import Annotated

from fastapi import Depends

from daos.executor import ExecutorDao
from models.executor import Executor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutorService:
    def __init__(self, executor_dao: Annotated[ExecutorDao, Depends(ExecutorDao)]):
        self.executor_dao = executor_dao

    def get_executors_for_validator(self, validator_hotkey: str):
        return self.executor_dao.get_executors_for_validator(validator_hotkey)

    async def send_pubkey_to_executor(self, executor: Executor, pubkey: str):
        """TODO: Send API request to executor with pubkey

        Args:
            executor (Executor): Executor instance that register validator hotkey
            pubkey (str): SSH public key from validator

        Return:
            response (str): Executor SSH connection info.
        """
        pass

    async def remove_pubkey_from_executor(self, executor: Executor):
        """TODO: Send API request to executor to cleanup pubkey

        Args:
            executor (Executor): Executor instance that needs to remove pubkey
        """
        pass

    async def register_pubkey(self, validator_hotkey: str, pubkey: bytes):
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
            for executor in self.get_executors_for_validator(validator_hotkey)
        ]

        total_executors = len(tasks)
        results = [result for result in await asyncio.gather(*tasks, return_exceptions=True)]
        logger.info(
            "Send pubkey register API requests to %d executors and received results from %d executors",
            total_executors,
            len(results),
        )
        return results

    async def deregister_pubkey(self, validator_hotkey: str):
        """Deregister pubkey from executors.

        Args:
            validator_hotkey (str): Validator hotkey
        """
        tasks = [
            asyncio.create_task(
                self.remove_pubkey_from_executor(executor),
                name=f"{executor}.remove_pubkey_from_executor",
            )
            for executor in self.get_executors_for_validator(validator_hotkey)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
