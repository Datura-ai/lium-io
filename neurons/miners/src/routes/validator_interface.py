import logging
from typing import Annotated

from core.config import settings
from datura.requests.validator_requests import (
    SimpleValidatorRequest,
    ExecutorInfo,
    ExecutorListResponse,
    SSHPubKeySubmitRequest,
    SSHPubKeyRemoveRequest,
    GetPodLogsRequest,
)
from datura.requests.miner_requests import (
    AcceptSSHKeyRequest,
    FailedRequest,
    SSHKeyRemoved,
    PodLogsResponse,
)
from fastapi import APIRouter, Depends, HTTPException, status

from consumers.validator_consumer import ValidatorConsumer
from dependencies.auth import verify_simple_validator_signature, verify_validator_auth_from_headers
from services.executor_service import ExecutorService
from services.ssh_service import MinerSSHService
from services.validator_service import ValidatorService

logger = logging.getLogger(__name__)

validator_router = APIRouter()


@validator_router.websocket("/websocket/{validator_key}")
async def validator_interface(consumer: Annotated[ValidatorConsumer, Depends(ValidatorConsumer)]):
    await consumer.connect()
    await consumer.handle()


@validator_router.post("/executors", response_model=ExecutorListResponse)
async def get_executors_for_validator(
    request: SimpleValidatorRequest,
    authenticated_validator: Annotated[str, Depends(verify_simple_validator_signature)],
    executor_service: Annotated[ExecutorService, Depends(ExecutorService)],
) -> ExecutorListResponse:
    """Get list of executors available for authenticated validator.
    Requires simple signature authentication: validator signs their own hotkey.
    Returns 200 with empty list if no executors found (not 404).

    Args:
        request: Request with signature and validator_hotkey
        authenticated_validator: Validated hotkey (after signature verification)
        executor_service: Service to retrieve executor information

    Returns:
        ExecutorListResponse: List of executors available for the validator

    Raises:
        HTTPException: 500 if database error or data serialization fails
    """
    try:
        miner_hotkey = settings.get_bittensor_wallet().get_hotkey().ss58_address
        executors = await executor_service.get_executors_for_validator(authenticated_validator, miner_hotkey)
    except Exception as e:
        logger.error(
            "Failed to retrieve executors for validator %s: %s",
            authenticated_validator,
            str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve executor list. Please try again later.",
        )

    return ExecutorListResponse(
        validator_hotkey=authenticated_validator,
        executors=[
            ExecutorInfo(
                uuid=str(executor.uuid),
                address=executor.address,
                port=executor.port
            )
            for executor in executors
        ],
    )


@validator_router.post("/api/validator/ssh-pubkey-submit", response_model=AcceptSSHKeyRequest | FailedRequest)
async def submit_ssh_pubkey(
    request: SSHPubKeySubmitRequest,
    authenticated_validator: Annotated[str, Depends(verify_validator_auth_from_headers)],
    executor_service: Annotated[ExecutorService, Depends(ExecutorService)],
    ssh_service: Annotated[MinerSSHService, Depends(MinerSSHService)],
) -> AcceptSSHKeyRequest | FailedRequest:
    """Submit SSH public key to miner (REST API version).
    
    This endpoint mirrors the WebSocket SSHPubKeySubmitRequest handling.
    Requires authentication via headers: X-Validator-Hotkey, X-Miner-Hotkey, X-Timestamp, X-Signature.
    
    Args:
        request: SSHPubKeySubmitRequest containing public_key and other info
        authenticated_validator: Validated validator hotkey (from headers)
        executor_service: Service to register pubkey
        ssh_service: SSH service for key management
        
    Returns:
        AcceptSSHKeyRequest with executors or FailedRequest on error
    """
    try:
        logger.info("Validator %s sent SSH Pubkey via REST API.", authenticated_validator)
        
        executors = await executor_service.register_pubkey(
            authenticated_validator, request.miner_hotkey, request.public_key, request.executor_id
        )
        
        if request.is_rental_request and len(executors) == 1:
            # Invoke rental request hook if configured
            if settings.RENTAL_REQUEST_HOOK:
                import aiohttp
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            settings.RENTAL_REQUEST_HOOK,
                            json={
                                "executor_address": executors[0].address,
                                "executor_port": executors[0].port,
                                "uuid": str(executors[0].uuid),
                            },
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as resp:
                            if resp.status != 200:
                                logger.error(f"RENTAL_REQUEST_HOOK returned status {resp.status}: {await resp.text()}")
                except Exception as e:
                    logger.error(f"Exception in RENTAL_REQUEST_HOOK: {e}")
        
        logger.info("Sent AcceptSSHKeyRequest to validator %s via REST API", authenticated_validator)
        return AcceptSSHKeyRequest(executors=executors)
    except Exception as e:
        logger.error("Storing SSH key or Sending AcceptSSHKeyRequest failed: %s", str(e))
        ssh_service.remove_pubkey_from_host(request.public_key)
        return FailedRequest(details=str(e))


@validator_router.post("/api/validator/ssh-pubkey-remove", response_model=SSHKeyRemoved | FailedRequest)
async def remove_ssh_pubkey(
    request: SSHPubKeyRemoveRequest,
    authenticated_validator: Annotated[str, Depends(verify_validator_auth_from_headers)],
    executor_service: Annotated[ExecutorService, Depends(ExecutorService)],
) -> SSHKeyRemoved | FailedRequest:
    """Remove SSH public key from miner (REST API version).
    
    This endpoint mirrors the WebSocket SSHPubKeyRemoveRequest handling.
    Requires authentication via headers: X-Validator-Hotkey, X-Miner-Hotkey, X-Timestamp, X-Signature.
    
    Args:
        request: SSHPubKeyRemoveRequest containing public_key to remove
        authenticated_validator: Validated validator hotkey (from headers)
        executor_service: Service to deregister pubkey
        
    Returns:
        SSHKeyRemoved on success or FailedRequest on error
    """
    try:
        logger.info("Validator %s sent remove SSH Pubkey via REST API.", authenticated_validator)
        
        await executor_service.deregister_pubkey(
            authenticated_validator, request.miner_hotkey, request.public_key, request.executor_id
        )
        logger.info("Sent SSHKeyRemoved to validator %s via REST API", authenticated_validator)
        return SSHKeyRemoved()
    except Exception as e:
        logger.error("Failed SSHKeyRemoved request: %s", str(e))
        return FailedRequest(details=str(e))


@validator_router.post("/api/validator/pod-logs", response_model=PodLogsResponse | FailedRequest)
async def get_pod_logs(
    request: GetPodLogsRequest,
    authenticated_validator: Annotated[str, Depends(verify_validator_auth_from_headers)],
    executor_service: Annotated[ExecutorService, Depends(ExecutorService)],
) -> PodLogsResponse | FailedRequest:
    """Get pod logs from miner (REST API version).
    
    This endpoint mirrors the WebSocket GetPodLogsRequest handling.
    Requires authentication via headers: X-Validator-Hotkey, X-Miner-Hotkey, X-Timestamp, X-Signature.
    
    Args:
        request: GetPodLogsRequest containing container_name and executor_id
        authenticated_validator: Validated validator hotkey (from headers)
        executor_service: Service to get pod logs
        
    Returns:
        PodLogsResponse with logs or FailedRequest on error
    """
    try:
        logger.info("Validator %s get pod logs for container %s via REST API.", authenticated_validator, request.container_name)
        
        logs = await executor_service.get_pod_logs(
            authenticated_validator, request.miner_hotkey, request.executor_id, request.container_name
        )
        logger.info("Sent GetPodLogs to validator %s via REST API", authenticated_validator)
        return PodLogsResponse(logs=logs)
    except Exception as e:
        logger.error("Failed GetPodLogs request: %s", str(e))
        return FailedRequest(details=str(e))