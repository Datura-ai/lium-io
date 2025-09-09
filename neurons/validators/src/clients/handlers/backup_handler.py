from typing import TYPE_CHECKING
from payload_models.payloads import BackupContainerRequest, RestoreContainerRequest
if TYPE_CHECKING:
    from clients.compute_client import ComputeClient

class BackupHandler:
    def __init__(self, compute_client: "ComputeClient"):
        self.compute_client = compute_client

    async def handle_backup_container_req(self, job_request: BackupContainerRequest):
        """Handle backup container request."""
        await self.compute_client.miner_service.handle_container(job_request)

    async def handle_restore_container_req(self, job_request: RestoreContainerRequest):
        """Handle restore container request."""
        await self.compute_client.miner_service.handle_container(job_request)