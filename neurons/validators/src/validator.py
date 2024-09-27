import logging
import asyncio

from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn

from core.config import settings
from core.validator import Validator
from routes.apis import apis_router
from connector import start_connector_process

# Set up logging
logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    validator = Validator()
    # Run the miner in the background
    task = asyncio.create_task(validator.start())
    connector_process = start_connector_process()
    
    try:
        yield
    finally:
        await validator.stop()  # Ensure proper cleanup
        await task  # Wait for the background task to complete
        connector_process.terminate()
        logging.info("Validator exited successfully.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=app_lifespan,
)

app.include_router(apis_router)

reload = True if settings.ENV == "dev" else False

if __name__ == "__main__":
    uvicorn.run("validator:app", host="0.0.0.0", port=settings.INTERNAL_PORT, reload=reload)