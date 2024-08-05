import logging

from fastapi import FastAPI
import uvicorn

from core.config import settings
from routes.validator_interface import validator_router

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title=settings.PROJECT_NAME,
)

app.include_router(validator_router)

if __name__ == "__main__":
    uvicorn.run("miner:app", host="0.0.0.0", port=8000, reload=True)