from sqlmodel import select, Session

from daos.base import BaseDao
from models.pod_log import PodLog


class PodLogDao(BaseDao[PodLog]):
    def find_by_continer_name(self, session: Session, container_name: str) -> list[PodLog]:
        statement = select(PodLog).where(
            PodLog.container_name == container_name or PodLog.container_name is None
        ).order_by(PodLog.created_at)
        return session.exec(statement).all()
