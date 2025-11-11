import logging
from typing import Annotated

from core.config import settings
from datura.requests.validator_requests import (
    SimpleValidatorRequest,
    ExecutorInfo,
    ExecutorListResponse,
)
from fastapi import APIRouter, Depends, HTTPException, status

from consumers.validator_consumer import ValidatorConsumer
from dependencies.auth import verify_simple_validator_signature
from services.executor_service import ExecutorService

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
        executors = executor_service.get_executors_for_validator(authenticated_validator, miner_hotkey)
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