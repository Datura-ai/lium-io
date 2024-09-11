from datetime import datetime, timedelta

from pydantic import BaseModel
from sqlalchemy import func

from daos.base import BaseDao
from models.task import Task, TaskStatus


class MinerScore(BaseModel):
    miner_hotkey: str
    total_score: float


class TaskDao(BaseDao):
    def save(self, task: Task) -> Task:
        self.session.add(task)
        self.session.commit()
        self.session.refresh(task)
        return task

    def update(self, uuid: str, **kwargs) -> Task:
        task = self.get_task_by_uuid(uuid)
        if not task:
            return None  # Or raise an exception if task is not found

        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)

        self.session.commit()
        self.session.refresh(task)
        return task

    def get_scores_for_last_epoch(self, tempo: int) -> list[MinerScore]:
        last_epoch = datetime.now() - timedelta(seconds=tempo * 12)

        results = (
            self.session.query(Task.miner_hotkey, func.sum(Task.score).label("total_score"))
            .filter(
                Task.task_status.in_([TaskStatus.Finished, TaskStatus.Failed]),
                Task.created_at >= last_epoch,
            )
            .group_by(Task.miner_hotkey)
            .all()
        )

        return [
            MinerScore(
                miner_hotkey=result[0],
                total_score=result[1],
            )
            for result in results
        ]

    def get_task_by_uuid(self, uuid: str) -> Task:
        return self.session.query(Task).filter_by(uuid=uuid).first()
