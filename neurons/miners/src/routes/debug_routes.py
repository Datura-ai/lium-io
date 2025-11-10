from typing import Annotated

from fastapi import APIRouter, Depends

from core.config import settings
from services.executor_service import ExecutorService

debug_apis_router = APIRouter()


@debug_apis_router.get("/debug/get-executors-for-validator/{validator_hotkey}")
async def get_executors_for_validator(
    validator_hotkey: str, executor_service: Annotated[ExecutorService, Depends(ExecutorService)]
):
    if not settings.DEBUG:
        return None
    return await executor_service.get_executors_for_validator(validator_hotkey)


@debug_apis_router.post("/debug/register_pubkey/{validator_hotkey}")
async def register_pubkey(
    validator_hotkey: str, executor_service: Annotated[ExecutorService, Depends(ExecutorService)]
):
    if not settings.DEBUG:
        return None
    pub_key = "Test Pubkey"
    return await executor_service.register_pubkey(validator_hotkey, pub_key.encode("utf-8"))


@debug_apis_router.post("/debug/remove_pubkey/{validator_hotkey}")
async def remove_pubkey_from_executor(
    validator_hotkey: str, executor_service: Annotated[ExecutorService, Depends(ExecutorService)]
):
    if not settings.DEBUG:
        return None
    pub_key = "Test Pubkey"
    await executor_service.deregister_pubkey(validator_hotkey, pub_key.encode("utf-8"))
