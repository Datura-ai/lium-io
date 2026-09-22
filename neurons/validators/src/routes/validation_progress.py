"""`GET /validation-progress` — where each node's verification stands on this validator.

Who reads it: support, from the operator's box, with the token the validator's environment
holds (`VALIDATION_PROGRESS_TOKEN`). The validator's port is published by docker-compose
(`${EXTERNAL_PORT}:${INTERNAL_PORT}`), so the route is registered only when that token is set
(`validator.py`), every request must carry it in `X-Validation-Progress-Token`, and the payload
names executors by uuid with their steps and timings only — no hotkeys, no exception text (the
error CLASS and the step; the text stays in the validator's own log line). The same transitions
are logged as `[progress]` lines for Loki.
"""

from __future__ import annotations

import hmac
from typing import Any

from core.validation_progress import progress
from fastapi import APIRouter, Header, HTTPException, Query

from core.config import settings

router = APIRouter(tags=["validation-progress"])

TOKEN_HEADER = "X-Validation-Progress-Token"


def require_token(token: str | None) -> None:
    """The header must equal VALIDATION_PROGRESS_TOKEN (constant-time); no token configured = no route."""
    expected = settings.VALIDATION_PROGRESS_TOKEN
    if not expected:
        raise HTTPException(status_code=404, detail="Not found")
    if token is None or not hmac.compare_digest(token.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized")


def list_progress(phase: str | None = None, lane: str | None = None) -> dict[str, Any]:
    records = progress.snapshot()
    if phase:
        records = [r for r in records if r["phase"] == phase]
    if lane:
        records = [r for r in records if r["lane"] == lane]
    records.sort(key=lambda r: r["updated_at"], reverse=True)
    return {"count": len(records), "executors": records}


def get_progress(executor_uuid: str) -> dict[str, Any]:
    record = progress.get(executor_uuid)
    if record is None:
        raise HTTPException(status_code=404, detail="No verification of this executor is known to this validator")
    return record


@router.get("/validation-progress")
async def list_validation_progress(
    phase: str | None = Query(default=None, description="Only records in this phase"),
    lane: str | None = Query(default=None, description="Only records of this lane: express or cycle"),
    token: str | None = Header(default=None, alias=TOKEN_HEADER),
) -> dict[str, Any]:
    require_token(token)
    return list_progress(phase, lane)


@router.get("/validation-progress/{executor_uuid}")
async def get_validation_progress(
    executor_uuid: str,
    token: str | None = Header(default=None, alias=TOKEN_HEADER),
) -> dict[str, Any]:
    require_token(token)
    return get_progress(executor_uuid)
