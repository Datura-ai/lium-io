import logging

from typing import Annotated
from fastapi import Depends

from core.db import get_session
from daos.pod_log import PodLogDao


logger = logging.getLogger(__name__)


class PodLogService:
    def __init__(
        self,
        pod_log_dao: Annotated[PodLogDao, Depends(PodLogDao)]
    ):
        self.pod_log_dao = pod_log_dao

    async def find_by_continer_name(self, container_name: str):
        with get_session() as session:
            return self.pod_log_dao.find_by_continer_name(session, container_name)
