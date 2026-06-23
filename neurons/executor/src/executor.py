import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import uvicorn

from core.config import settings
from core.logger import get_logger
from middlewares.miner import MinerMiddleware
from routes.apis import apis_router
from services.cache_template_service import run_cache_template_prefetch

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start pulling this host's cache template image as soon as the executor
    # boots, and keep it fresh in the background, without blocking startup.
    prefetch_task = asyncio.create_task(run_cache_template_prefetch())
    try:
        yield
    finally:
        prefetch_task.cancel()
        try:
            await prefetch_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"cache template pre-pull task ended with error: {e}")


app = FastAPI(
    title=settings.PROJECT_NAME,
    docs_url=None,  # Disable /docs
    redoc_url=None,  # Disable /redoc
    openapi_url=None,  # Disable /openapi.json
    lifespan=lifespan,
)

app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_middleware(MinerMiddleware)
app.include_router(apis_router)

reload = True if settings.ENV == "dev" else False

if __name__ == "__main__":
    uvicorn.run("executor:app", host="0.0.0.0", port=settings.INTERNAL_PORT, reload=reload)
