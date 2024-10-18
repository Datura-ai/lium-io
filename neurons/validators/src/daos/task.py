from datetime import datetime, timedelta

import sqlalchemy
from pydantic import BaseModel
from sqlalchemy import func, select

from daos.base import BaseDao
from models.task import Task, TaskStatus


class MinerScore(BaseModel):
    miner_hotkey: str
    total_score: float


class TaskDao(BaseDao):
    async def save(self, task: Task) -> Task:
        try:
            self.session.add(task)
            await self.session.commit()
            await self.session.refresh(task)
            return task
        except Exception as e:
            await self.session.rollback()
            raise e

    async def update(self, uuid: str, **kwargs) -> Task:
        task = await self.get_task_by_uuid(uuid)
        if not task:
            return None  # Or raise an exception if task is not found

        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)

        try:
            await self.session.commit()
            await self.session.refresh(task)
            return task
        except Exception as e:
            await self.session.rollback()
            raise e

    async def get_scores_for_last_epoch(self, tempo: int) -> list[MinerScore]:
        last_epoch = datetime.utcnow() - timedelta(seconds=tempo * 12)

        statement = (
            select(Task.miner_hotkey, func.sum(Task.score).label("total_score"))
            .where(
                Task.task_status.in_([TaskStatus.Finished, TaskStatus.Failed]),
                Task.created_at >= last_epoch,
            )
            .group_by(Task.miner_hotkey)
        )
        results: sqlalchemy.engine.result.ChunkedIteratorResult = await self.session.exec(statement)
        results = results.all()

        return [
            MinerScore(
                miner_hotkey=result[0],
                total_score=result[1],
            )
            for result in results
        ]

    async def get_task_by_uuid(self, uuid: str) -> Task:
        statement = select(Task).where(Task.uuid == uuid)
        results = await self.session.exec(statement)
        return results.scalar_one_or_none()
