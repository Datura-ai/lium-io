import asyncio

from models.pod_log import PodLog
from services import pod_log_store


class PodLogService:
    async def find_by_continer_name(self, container_name: str) -> list[PodLog]:
        # file read happens off the event loop; a hanging disk must not block /ping
        return await asyncio.to_thread(pod_log_store.find_by_container_name, container_name)
