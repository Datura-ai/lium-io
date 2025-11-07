import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import uvicorn

from core.config import settings
from core.logger import get_logger
from middlewares.miner import MinerMiddleware
from routes.apis import apis_router

# Set up logging
logging.basicConfig(level=logging.INFO)

logger = get_logger(__name__)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom handler for request validation errors.
    Logs at DEBUG level to reduce noise in production while maintaining dev observability.
    """
    logger.debug(
        f"Validation error on {request.method} {request.url.path}: {exc.errors()}"
    )
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


app = FastAPI(
    title=settings.PROJECT_NAME,
)

app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_middleware(MinerMiddleware)
app.include_router(apis_router)

reload = True if settings.ENV == "dev" else False

if __name__ == "__main__":
    uvicorn.run("executor:app", host="0.0.0.0", port=settings.INTERNAL_PORT, reload=reload)
