from fastapi import APIRouter, Response
from services.miner_service import MinerRequestPayload

from services.miner_service import MinerServiceDep
from services.task_service import TaskServiceDep

apis_router = APIRouter()


@apis_router.post("/miner_request")
async def request_resource_to_miner(
    miner_payload: MinerRequestPayload, miner_service: MinerServiceDep
):
    """Requesting resource to miner."""
    await miner_service.request_resource_to_miner(miner_payload)


@apis_router.get("/tasks/{uuid}/download")
async def download_private_key_for_task(uuid: str, task_service: TaskServiceDep):
    """Download private key for given task."""
    private_key: str = task_service.get_decrypted_private_key_for_task(uuid)
    if not private_key:
        return Response(content="No private key found", media_type="text/plain", status_code=404)
    return Response(
        content=private_key,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": "attachment; filename=private_key",
        },
    )
