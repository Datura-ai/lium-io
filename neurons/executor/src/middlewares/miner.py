import bittensor
from fastapi.responses import JSONResponse
from payloads.miner import MinerAuthPayload
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from core.config import settings
from core.logger import _m, get_logger

logger = get_logger(__name__)


class MinerMiddleware(BaseHTTPMiddleware):
    def __init__(self, app) -> None:
        super().__init__(app)

    async def dispatch(self, request, call_next):
        try:
            body_bytes = await request.body()
            miner_ip = request.client.host
            default_extra = {"miner_ip": miner_ip}

            # Parse it into the Pydantic model
            payload = MinerAuthPayload.model_validate_json(body_bytes)

            logger.info(_m("miner ip", extra=default_extra))

            keypair = bittensor.Keypair(ss58_address=settings.MINER_HOTKEY_SS58_ADDRESS)
            if not keypair.verify(payload.public_key, payload.signature):
                logger.error(
                    _m(
                        "Auth failed. incorrect signature",
                        extra={
                            **default_extra,
                            "signature": payload.signature,
                            "public_key": payload.public_key,
                            "miner_hotkey": settings.MINER_HOTKEY_SS58_ADDRESS,
                        },
                    )
                )
                return JSONResponse(status_code=401, content="Unauthorized")

            response = await call_next(request)
            return response
        except ValidationError as e:
            # Handle validation error if needed
            error_message = str(_m("Validation Error", extra={"errors": str(e.errors())}))
            logger.error(error_message)
            return JSONResponse(status_code=422, content=error_message)
