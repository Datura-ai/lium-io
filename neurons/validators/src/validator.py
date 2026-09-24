import asyncio
import logging

import uvicorn
from fastapi import FastAPI

from core.config import settings
from core.utils import (
    configure_logs_of_other_modules,
    wait_for_services_sync,
    widen_default_thread_pool,
)
from core.validator import Validator

configure_logs_of_other_modules()
wait_for_services_sync()


async def app_lifespan(app: FastAPI):
    widen_default_thread_pool(asyncio.get_running_loop())
    validator = Validator()
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

reload = True if settings.ENV == "dev" else False

if __name__ == "__main__":
    if settings.DRY_RUN:
        asyncio.run(run_dry_run())
    else:
        uvicorn.run("validator:app", host="0.0.0.0", port=settings.INTERNAL_PORT, reload=reload)
