import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from core.config import settings
from core.utils import wait_for_services_sync
from core.validator import Validator
from routes.apis import apis_router

# Set up logging
logging.basicConfig(level=logging.INFO)


wait_for_services_sync()


@asynccontextmanager
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

app.include_router(apis_router)

reload = True if settings.ENV == "dev" else False

if __name__ == "__main__":
    uvicorn.run("validator:app", host="0.0.0.0", port=settings.INTERNAL_PORT, reload=reload)
