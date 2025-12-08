from typing import Optional
from uuid import UUID

from sqlmodel import select
from daos.base import BaseDao
from models.miner import Miner
from models.miner_executor import MinerExecutor


class MinerExecutorDao(BaseDao):
    """DAO for MinerExecutor (production DB in central mode)"""

    def get_by_miner_hotkey(
        self, miner_hotkey: str, executor_id: Optional[str] = None
    ) -> list[MinerExecutor]:
        """Get executors by miner hotkey

        Args:
            miner_hotkey: The miner's hotkey
            executor_id: Optional specific executor ID to filter

        Returns:
            List of MinerExecutor objects
        """
        stmt = (
            select(MinerExecutor)
            .join(Miner, Miner.id == MinerExecutor.miner_id)
            .where(Miner.miner_hotkey == miner_hotkey)
        )
        if executor_id:
            stmt = stmt.where(MinerExecutor.id == executor_id)

        return list(self.session.exec(stmt).all())

    def get_by_miner_and_validator(
        self, miner_hotkey: str, validator_hotkey: str, executor_id: Optional[str] = None
    ) -> list[MinerExecutor]:
        """Get executors by miner hotkey and validator hotkey

        Args:
            miner_hotkey: The miner's hotkey
            validator_hotkey: The validator's hotkey
            executor_id: Optional specific executor ID to filter

        Returns:
            List of MinerExecutor objects filtered by validator
        """
        stmt = (
            select(MinerExecutor)
            .join(Miner, Miner.id == MinerExecutor.miner_id)
            .where(
                Miner.miner_hotkey == miner_hotkey,
                MinerExecutor.validator_hotkey == validator_hotkey
            )
        )
        if executor_id:
            stmt = stmt.where(MinerExecutor.id == executor_id)

        return list(self.session.exec(stmt).all())
