import logging

from sqlalchemy import select

from daos.base import BaseDao
from models.executor import Executor

logger = logging.getLogger(__name__)


class ExecutorDao(BaseDao):
    async def upsert(self, executor: Executor) -> Executor:
        try:
            existing_executor = await self.get_executor(
                executor_id=executor.executor_id, miner_hotkey=executor.miner_hotkey
            )

            if existing_executor:
                # Update the fields of the existing executor
                existing_executor.miner_address = executor.miner_address
                existing_executor.miner_port = executor.miner_port
                existing_executor.executor_ip_address = executor.executor_ip_address
                existing_executor.executor_ssh_username = executor.executor_ssh_username
                existing_executor.executor_ssh_port = executor.executor_ssh_port
                existing_executor.attestation_digest = executor.attestation_digest
                existing_executor.tee_type = executor.tee_type
                existing_executor.attestation_verified = executor.attestation_verified
                existing_executor.attestation_verified_at = executor.attestation_verified_at

                await self.session.commit()
                await self.session.refresh(existing_executor)
                return existing_executor
            else:
                # Insert the new executor
                self.session.add(executor)
                await self.session.commit()
                await self.session.refresh(executor)

                return executor
        except Exception as e:
            await self.session.rollback()
            logger.error("Error upsert executor: %s", e)
            raise

    async def rent(self, executor_id: str, miner_hotkey: str) -> Executor:
        try:
            executor = await self.get_executor(executor_id=executor_id, miner_hotkey=miner_hotkey)
            if executor:
                executor.rented = True
                await self.session.commit()
                await self.session.refresh(executor)

            return executor
        except Exception as e:
            await self.session.rollback()
            logger.error("Error rent executor: %s", e)
            raise

    async def unrent(self, executor_id: str, miner_hotkey: str) -> Executor:
        try:
            executor = await self.get_executor(executor_id=executor_id, miner_hotkey=miner_hotkey)
            if executor:
                executor.rented = False
                await self.session.commit()
                await self.session.refresh(executor)

            return executor
        except Exception as e:
            await self.session.rollback()
            logger.error("Error unrent executor: %s", e)
            raise

    async def get_executor(self, executor_id: str, miner_hotkey: str) -> Executor:
        try:
            statement = select(Executor).where(
                Executor.miner_hotkey == miner_hotkey, Executor.executor_id == executor_id
            )
            result = await self.session.exec(statement)
            return result.scalar_one_or_none()
        except Exception as e:
            await self.session.rollback()
            logger.error("Error get executor: %s", e)
            raise
