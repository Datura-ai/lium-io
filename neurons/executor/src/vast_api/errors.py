import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Machine-readable error codes shared by refusals and failures (see plan-vastctl.md
# + plan-key-split.md): g0_failed, state_mount_missing, gpu_broken, install_rollback,
# identify_rejected, report_failed, rental_running, run_in_progress, host_command_failed.

logger = logging.getLogger(__name__)


class ApiRefusal(Exception):
    """Safety-rail refusal — HTTP 409, overridable with force where allowed."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class ApiFailure(Exception):
    """Real failure while executing an operation — HTTP 500."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def safe_error(exc: Exception) -> str:
    # only our own error types carry curated text; anything else may embed internals
    # (host paths, command lines, stack info) — log it, expose the class name alone
    if isinstance(exc, (ApiRefusal, ApiFailure)):
        return f"{exc.code}: {exc.detail}"[:300]
    logger.warning("suppressed error detail in API response: %r", exc)
    return type(exc).__name__


def _error_response(status_code: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "detail": detail}},
    )


async def refusal_handler(request: Request, exc: ApiRefusal) -> JSONResponse:
    return _error_response(409, exc.code, exc.detail)


async def failure_handler(request: Request, exc: ApiFailure) -> JSONResponse:
    return _error_response(500, exc.code, exc.detail)


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiRefusal, refusal_handler)
    app.add_exception_handler(ApiFailure, failure_handler)
