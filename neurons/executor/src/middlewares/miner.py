import logging

import bittensor
from fastapi.responses import JSONResponse
from payloads.miner import MinerAuthPayload
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from core.config import settings

logger = logging.getLogger(__name__)


class MinerMiddleware(BaseHTTPMiddleware):
    def __init__(self, app) -> None:
        super().__init__(app)

    async def dispatch(self, request, call_next):
        try:
            body_bytes = await request.body()
            miner_ip = request.client.host

            # Parse it into the Pydantic model
            payload = MinerAuthPayload.model_validate_json(body_bytes)
            logger.info(f"miner ip: {miner_ip}")

            keypair = bittensor.Keypair(ss58_address=settings.MINER_HOTKEY_SS58_ADDRESS)
            if not keypair.verify(payload.public_key, payload.signature):
                logger.error("Auth failed. incorrect signature")
                return JSONResponse(status_code=401, content={"detail": "Auth failed"})

            response = await call_next(request)

            return response
        except ValidationError as e:
            # Handle validation error if needed
            logger.error(f"Validation Error: {e}")

            return JSONResponse(status_code=422, content={"detail": e.errors()})
