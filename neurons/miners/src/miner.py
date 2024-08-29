import logging
import asyncio

from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn

from core.config import settings
from routes.validator_interface import validator_router
from core.miner import Miner

# Set up logging
logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    miner = Miner()
    # Run the miner in the background
    task = asyncio.create_task(miner.start())
    
    try:
        yield
    finally:
        await miner.stop()  # Ensure proper cleanup
        await task  # Wait for the background task to complete
        logging.info("Miner exited successfully.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=app_lifespan,
)

app.include_router(validator_router)

if __name__ == "__main__":
    uvicorn.run("miner:app", host="0.0.0.0", port=settings.INTERNAL_PORT, reload=True)