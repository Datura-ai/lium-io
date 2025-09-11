import asyncio
import logging

import uvicorn
from fastapi import FastAPI, HTTPException

from core.config import settings
from core.utils import configure_logs_of_other_modules, wait_for_services_sync
from core.validator import Validator
from clients.subtensor_client import SubtensorClient

configure_logs_of_other_modules()
wait_for_services_sync()


async def app_lifespan(app: FastAPI):
    validator = Validator()
    # Run the miner in the background
    task = asyncio.create_task(validator.start())

    try:
        yield
    finally:
        await validator.stop()  # Ensure proper cleanup
        await task  # Wait for the background task to complete
        logging.info("Validator exited successfully.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=app_lifespan,
)

reload = True if settings.ENV == "dev" else False

if __name__ == "__main__":
    uvicorn.run("validator:app", host="0.0.0.0", port=settings.INTERNAL_PORT, reload=reload)
