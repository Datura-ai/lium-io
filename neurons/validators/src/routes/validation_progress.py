"""`GET /validation-progress` — where each node's verification stands on this validator.

For support: which check a node is on and since when, the last reason code and the last failure,
and the express lane's waits (asking the miner, retry scheduled, left to the cycle). The data is
the in-memory registry `core.validation_progress.progress`; the same transitions are logged as
`[progress]` lines for Loki. Bound to the validator's internal port like every other route here.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from core.validation_progress import progress

router = APIRouter(tags=["validation-progress"])


@router.get("/validation-progress")
async def list_validation_progress(
    phase: str | None = Query(default=None, description="Only records in this phase"),
    lane: str | None = Query(default=None, description="Only records of this lane: express or cycle"),
) -> dict[str, Any]:
    records = progress.snapshot()
    if phase:
        records = [r for r in records if r["phase"] == phase]
    if lane:
        records = [r for r in records if r["lane"] == lane]
    records.sort(key=lambda r: r["updated_at"], reverse=True)
    return {"count": len(records), "executors": records}


@router.get("/validation-progress/{executor_uuid}")
async def get_validation_progress(executor_uuid: str) -> dict[str, Any]:
    record = progress.get(executor_uuid)
    if record is None:
        raise HTTPException(status_code=404, detail="No verification of this executor is known to this validator")
    return record
