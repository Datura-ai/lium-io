import logging

from typing import Annotated
from fastapi import Depends

from core.db import SessionDep
from daos.pod_log import PodLogDao

from payloads.miner import MinerAuthPayload

logger = logging.getLogger(__name__)


class PodLogService:
    def __init__(
        self,
        session: SessionDep,
        pod_log_dao: Annotated[PodLogDao, Depends(PodLogDao)]
    ):
        self.session = session
        self.pod_log_dao = pod_log_dao

    async def find_by_continer_name(self, payload: MinerAuthPayload):
        return self.pod_log_dao.find_by_continer_name(self.session, '')

