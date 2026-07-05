import asyncio
import sys
import logging
from pathlib import Path

from typing import Annotated
from fastapi import Depends

from core.config import settings
from services.gpu_attestation_service import GPUAttestationService
from services.ssh_service import SSHService
from services.tdx_service import TDXQuoteService

from payloads.miner import UploadSShKeyPayload

logger = logging.getLogger(__name__)


class MinerService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
        tdx_service: Annotated[TDXQuoteService, Depends(TDXQuoteService)],
        gpu_attestation_service: Annotated[GPUAttestationService, Depends(GPUAttestationService)],
    ):
        self.ssh_service = ssh_service
        self.tdx_service = tdx_service
        self.gpu_attestation_service = gpu_attestation_service

    async def upload_ssh_key(self, paylod: UploadSShKeyPayload):
        self.ssh_service.add_pubkey_to_host(paylod.public_key)

        host_key = self.ssh_service.get_host_public_key()
        # The validator nonce (when present) binds the TDX quote and the GPU
        # evidence to this attestation event; both fall back to their legacy /
        # self-nonce shapes when it is absent.
        tdx_quote = await self.tdx_service.get_quote(host_key, nonce=paylod.nonce)
        nvidia_payload = await self.gpu_attestation_service.collect(nonce=paylod.nonce)

        response = {
            "ssh_username": self.ssh_service.get_current_os_user(),
            "ssh_port": settings.SSH_PUBLIC_PORT or settings.SSH_PORT,
            "python_path": sys.executable,
            "root_dir": str(Path(__file__).resolve().parents[2]),
            "port_range": settings.RENTING_PORT_RANGE,
            "port_mappings": settings.RENTING_PORT_MAPPINGS,
        }
        if host_key:
            response["ssh_host_key"] = host_key
        if tdx_quote:
            response["tdx_quote"] = tdx_quote
        if nvidia_payload:
            response["nvidia_payload"] = nvidia_payload
        return response

    async def remove_ssh_key(self, paylod: UploadSShKeyPayload):
        return self.ssh_service.remove_pubkey_from_host(paylod.public_key)
