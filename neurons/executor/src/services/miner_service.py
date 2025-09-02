import asyncio
import sys
import logging
from pathlib import Path

from typing import Annotated
from fastapi import Depends

from core.config import settings
from services.ssh_service import SSHService

from payloads.miner import UploadSShKeyPayload

logger = logging.getLogger(__name__)


class MinerService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
    ):
        self.ssh_service = ssh_service

    async def upload_ssh_key(self, paylod: UploadSShKeyPayload):
        self.ssh_service.add_pubkey_to_host(paylod.public_key)

        return {
            "ssh_username": self.ssh_service.get_current_os_user(),
            "ssh_port": settings.SSH_PUBLIC_PORT or settings.SSH_PORT,
            "python_path": sys.executable,
            "root_dir": str(Path(__file__).resolve().parents[2]),
            "port_range": settings.RENTING_PORT_RANGE,
            "port_mappings": settings.RENTING_PORT_MAPPINGS,
        }

    async def remove_ssh_key(self, paylod: UploadSShKeyPayload):
        return self.ssh_service.remove_pubkey_from_host(paylod.public_key)
