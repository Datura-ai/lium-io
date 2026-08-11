from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Machine-readable error codes shared by refusals and failures (see plan-vastctl.md):
# g0_failed, state_mount_missing, gpu_broken, install_rollback, identify_rejected,
# gpu_busy, rental_running, run_in_progress.


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
