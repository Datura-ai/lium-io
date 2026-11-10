from sqlalchemy import delete
from sqlmodel import select

from daos.base import BaseDao
from models.recovery_stop import UnmarkedRecoveryStop


class UnmarkedRecoveryStopDao(BaseDao):
    async def all_pod_ids(self) -> set[str]:
        async with self.get_session() as session:
            return set((await session.exec(select(UnmarkedRecoveryStop.pod_id))).all())

    async def add(self, pod_id: str) -> bool:
        async with self.get_session() as session:
            await session.merge(UnmarkedRecoveryStop(pod_id=pod_id))
        return True

    async def remove(self, pod_id: str) -> bool:
        async with self.get_session() as session:
            await session.execute(delete(UnmarkedRecoveryStop).where(UnmarkedRecoveryStop.pod_id == pod_id))
        return True
