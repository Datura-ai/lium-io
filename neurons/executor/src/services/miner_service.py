import asyncio
import sys
import logging
from pathlib import Path

from typing import Annotated
from fastapi import Depends, HTTPException
import bittensor

from core.config import settings
from services.ssh_service import SSHService
from services.tdx_service import TDXQuoteService

from payloads.miner import UploadSShKeyPayload

logger = logging.getLogger(__name__)


class MinerService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
        tdx_quote_service: Annotated[TDXQuoteService, Depends(TDXQuoteService)],
    ):
        self.ssh_service = ssh_service
        self.tdx_service = tdx_quote_service

    def verify_validator_signature(self, payload: UploadSShKeyPayload):
        if not settings.VALIDATOR_HOTKEY_SS58_ADDRESS:
            logger.info("Validator verification skipped")
            return

        # Verify validator signature only if VALIDATOR_HOTKEY_SS58_ADDRESS is configured
        expected_data_format = f"SSH_PUBKEY_INJECTION:{payload.public_key}"
        if not payload.validator_signature:
            logger.error("Validator signature is missing")
            raise HTTPException(
                status_code=401, detail="Validator signature is missing")
        validator_keypair = bittensor.Keypair(
            ss58_address=settings.VALIDATOR_HOTKEY_SS58_ADDRESS)
        validator_verified = validator_keypair.verify(
            expected_data_format, payload.validator_signature)

        if not validator_verified:
            logger.error("Validator auth failed. incorrect signature")
            raise HTTPException(
                status_code=401, detail="Validator signature invalid")
        logger.info("Validator signature verification successful")

    async def upload_ssh_key(self, payload: UploadSShKeyPayload):
        self.verify_validator_signature(payload)
        # Add the SSH public key to authorized_keys
        self.ssh_service.add_pubkey_to_host(payload.public_key)

        host_key = self.ssh_service.get_host_public_key()
        tdx_quote = await self.tdx_service.get_quote(host_key)

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
        return response

    async def remove_ssh_key(self, paylod: UploadSShKeyPayload):
        return self.ssh_service.remove_pubkey_from_host(paylod.public_key)
