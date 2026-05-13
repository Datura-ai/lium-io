import asyncio
import logging
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from payload_models.forced_validation import (
    ForceValidationCreateRequest,
    ForceValidationRequestRecord,
)

from core.config import settings
from core.utils import configure_logs_of_other_modules, wait_for_services_sync
from core.validator import Validator
from services.forced_validation import ForceValidationConflict, ForceValidationNotFound

configure_logs_of_other_modules()
wait_for_services_sync()


async def app_lifespan(app: FastAPI):
    validator = Validator()
    app.state.validator = validator
    # Run the miner in the background
    task = asyncio.create_task(validator.start())

    try:
        yield
    finally:
        await validator.stop()  # Ensure proper cleanup
        await task  # Wait for the background task to complete
        logging.info("Validator exited successfully.")


async def run_dry_run():
    """Run validator once in DRY_RUN mode without FastAPI server."""
    validator = Validator()
    await validator.start()
    await validator.stop()


app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=app_lifespan,
)


def validate_internal_token(
    x_internal_token: Annotated[str | None, Header(alias="X-Internal-Token")] = None,
) -> None:
    expected_token = settings.FORCED_VALIDATION_INTERNAL_TOKEN
    if expected_token:
        if x_internal_token != expected_token:
            raise HTTPException(status_code=401, detail="Invalid internal token")
        return
    if settings.ENV != "dev" and settings.DEPLOY_ENV in {"PROD", "STAGE"}:
        raise HTTPException(
            status_code=503,
            detail="Forced validation internal token is not configured",
        )


def get_force_validation_service(request: Request):
    validator = getattr(request.app.state, "validator", None)
    if validator is None or not hasattr(validator, "force_validation_service"):
        raise HTTPException(status_code=503, detail="Validator services are not ready")
    return validator.force_validation_service


@app.post(
    "/internal/forced-validations",
    response_model=ForceValidationRequestRecord,
    dependencies=[Depends(validate_internal_token)],
)
async def create_forced_validation(
    payload: ForceValidationCreateRequest,
    service=Depends(get_force_validation_service),
):
    try:
        return await service.create_request(
            executor_id=payload.executor_id,
            miner_hotkey=payload.miner_hotkey,
        )
    except ForceValidationConflict:
        raise HTTPException(
            status_code=409,
            detail="Forced validation is already running for this executor",
        )


@app.get(
    "/internal/forced-validations/{request_id}",
    response_model=ForceValidationRequestRecord,
    dependencies=[Depends(validate_internal_token)],
)
async def get_forced_validation(
    request_id: str,
    service=Depends(get_force_validation_service),
):
    try:
        return await service.get_request(request_id)
    except ForceValidationNotFound:
        raise HTTPException(status_code=404, detail="Forced validation request not found")


reload = True if settings.ENV == "dev" else False

if __name__ == "__main__":
    if settings.DRY_RUN:
        asyncio.run(run_dry_run())
    else:
        uvicorn.run("validator:app", host="0.0.0.0", port=settings.INTERNAL_PORT, reload=reload)
