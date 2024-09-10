from datetime import datetime, timedelta
from sqlalchemy import func
from pydantic import BaseModel
from typing import List

from models.task import Task, TaskStatus

from daos.base import BaseDao


class MinerAvgScore(BaseModel):
    miner_hotkey: str
    avg_score: float


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

    def get_avg_scores_in_hours(self, hours: int) -> List[MinerAvgScore]:
        hours_ago = datetime.now() - timedelta(hours=hours)

        subquery = self.session.query(
            Task.miner_hotkey,
            Task.executor_id,
            func.avg(Task.score).label('avg_score')
        ).filter(
            Task.task_status.in_([TaskStatus.Finished, TaskStatus.Failed]),
            Task.created_at >= hours_ago
        ).group_by(Task.miner_hotkey, Task.executor_id).subquery()
        
        results = self.session.query(
            subquery.c.miner_hotkey,
            func.sum(subquery.c.avg_score).label('score')
        ).group_by(subquery.c.miner_hotkey).all()

        return [
            MinerAvgScore(
                miner_hotkey=result[0],
                avg_score=result[1],
            )
            for result in results
        ]

    def get_task_by_uuid(self, uuid: str) -> Task:
        return self.session.query(Task).filter_by(uuid=uuid).first()
