import logging

from fastapi import FastAPI
import uvicorn

from core.config import settings
from middlewares.miner import MinerMiddleware
from routes.apis import apis_router

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title=settings.PROJECT_NAME,
)

app.add_middleware(MinerMiddleware)
app.include_router(apis_router)

reload = True if settings.ENV == "dev" else False

if __name__ == "__main__":
    uvicorn.run("executor:app", host="0.0.0.0", port=settings.INTERNAL_PORT, reload=reload)
