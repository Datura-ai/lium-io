import logging

from fastapi import FastAPI
import uvicorn

from core.config import settings
from routes.validator_interface import validator_router
from core.miner import Miner

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title=settings.PROJECT_NAME,
)

app.include_router(validator_router)

if __name__ == "__main__":
    with Miner():
        uvicorn.run("miner:app", host="0.0.0.0", port=settings.PORT, reload=True)