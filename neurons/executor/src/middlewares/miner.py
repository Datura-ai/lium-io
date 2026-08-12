import asyncio
import re
import time

import bittensor
from fastapi.responses import JSONResponse
from payloads.miner import MinerAuthPayload
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from core.config import settings
from core.logger import _m, get_logger

logger = get_logger(__name__)

AUTHENTICATED_REQUEST_TIMEOUT_SECONDS = 30
# /vast market ops legitimately block up to ~90s waiting for offers to appear
# after unlist→list; everything else keeps the tight cap
VAST_REQUEST_TIMEOUT_SECONDS = 180


def request_timeout_seconds(path: str) -> int:
    # per-path timeout for authenticated requests
    if path.startswith("/vast"):
        return VAST_REQUEST_TIMEOUT_SECONDS
    return AUTHENTICATED_REQUEST_TIMEOUT_SECONDS


class MinerMiddleware(BaseHTTPMiddleware):
    def __init__(self, app) -> None:
        super().__init__(app)

    async def dispatch(self, request, call_next):
        # Skip middleware for GET requests as they have no body to validate.
        # Prevents false validation errors when GET requests arrive (returns proper 404 instead of 422).
        if request.method == "GET":
            return await call_next(request)

        # Skip middleware for endpoints with their own signature verification
        if request.url.path in ["/hardware_utilization", "/ping"]:
            return await call_next(request)
        
        # Skip for specific container hardware utilization endpoint
        # Pattern: /containers/{container_name}
        if re.match(r'^/containers(/[^/]+)?/?$', request.url.path):
            return await call_next(request)
            
        default_extra = {
            'url': request.url.path,
            'client_host': request.client.host,
        }
        try:
            body_bytes = await request.body()
            # miner_ip = request.client.host

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
                try:
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
                except ValueError as e:
                    logger.error(f"Error verifying signature: {e}")
                    verified = False

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

            started = time.monotonic()
            try:
                response = await asyncio.wait_for(
                    call_next(request),
                    timeout=request_timeout_seconds(request.url.path),
                )
            except TimeoutError:
                logger.error(
                    _m(
                        "Authenticated request timed out before route response",
                        extra={
                            **default_extra,
                            "stage": "call_next",
                            "duration_ms": int((time.monotonic() - started) * 1000),
                        },
                    )
                )
                return JSONResponse(
                    status_code=504,
                    content={
                        "status": "failed",
                        "failure_code": "EXECUTOR_AUTHENTICATED_REQUEST_TIMEOUT",
                        "stage": "call_next",
                        "message": "Request authenticated but executor route did not return before timeout",
                    },
                )

            logger.info(
                _m(
                    "Authenticated request completed",
                    extra={
                        **default_extra,
                        "status_code": response.status_code,
                        "duration_ms": int((time.monotonic() - started) * 1000),
                    },
                )
            )
            return response
        except ValidationError as e:
            # Handle validation error if needed
            error_message = str(_m("Validation Error", extra={**default_extra, "errors": str(e.errors())}))
            logger.error(error_message)
            return JSONResponse(status_code=422, content=error_message)
