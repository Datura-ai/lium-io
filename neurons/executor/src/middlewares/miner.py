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

            # Try verifying with both the configured miner hotkey and the default portal hotkey
            hotkeys_to_verify = [
                settings.MINER_HOTKEY_SS58_ADDRESS,
                settings.DEFAULT_MINER_HOTKEY,
            ]

            verified = False
            for hotkey in hotkeys_to_verify:
                keypair = bittensor.Keypair(ss58_address=hotkey)
                if keypair.verify(payload.data_to_sign, payload.signature):
                    verified = True
                    logger.info(
                        _m(
                            "Auth successful",
                            extra={
                                **default_extra,
                                "verified_with_hotkey": hotkey,
                            },
                        )
                    )
                    break

            if not verified:
                logger.error(
                    _m(
                        "Auth failed. incorrect signature",
                        extra={
                            **default_extra,
                            "signature": payload.signature,
                            "data_to_sign": payload.data_to_sign,
                            "tried_hotkeys": hotkeys_to_verify,
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
