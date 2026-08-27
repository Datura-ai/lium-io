"""DAH-2787 — the published spec carries only visits into THIS executor's filler containers.

The scrape reads the whole host: it sees every `filler_*` container there, including one that
belongs to another executor on the same box. The backend's snapshot says which containers are
this executor's, and only those visits may reach the incentive gate.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from neurons.validators.src.protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from neurons.validators.src.services.task.result_handler import ResultHandler

from tests.helpers import build_state, default_executor

OWN_CONTAINER = "filler_own"
NEIGHBOUR_CONTAINER = "filler_neighbour"


def _visit(container: str) -> dict[str, Any]:
    return {
        "container": container,
        "kind": "docker_exec",
        "pid": None,
        "seconds_after_start": 3600.0,
        "command": "kill -STOP 1584",
    }


def _miner_info() -> SimpleNamespace:
    return SimpleNamespace(miner_hotkey="miner-hotkey", job_batch_id="batch-1")


def _context(context_factory, *, entries: list[Any], rented_data: RentedExecutorsResponse | None):
    state = build_state(
        specs={"gpu": {"count": 1}, "filler_entries": entries},
        rented_data=rented_data,
    )
    return context_factory(
        state=state,
        tdx_attestation_passed=False,
        score=1.0,
        job_score=1.0,
        collateral_deposited=False,
        ssh_pub_keys=[],
        rented=False,
    )


async def _published_entries(context) -> list[Any]:
    handler = ResultHandler(redis_service=None, dry_run=True)
    result = await handler.handle_result(
        context=context,
        miner_info=_miner_info(),
        executor_info=context.executor,
        verified_job_info={},
        log_text="ok",
        success=True,
    )
    return result.spec["filler_entries"]


def _snapshot_with_own_container() -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={},
        all_filler_containers_by_executor={default_executor().uuid: [OWN_CONTAINER]},
    )


@pytest.mark.asyncio
async def test_a_visit_into_this_executors_filler_is_published(context_factory):
    # Arrange
    ctx = _context(
        context_factory,
        entries=[_visit(OWN_CONTAINER)],
        rented_data=_snapshot_with_own_container(),
    )

    # Act
    entries = await _published_entries(ctx)

    # Assert
    assert [entry["container"] for entry in entries] == [OWN_CONTAINER]


@pytest.mark.asyncio
async def test_a_visit_into_another_executors_filler_is_dropped(context_factory):
    # A host can run more than one executor. The neighbour's breach must not cost this one.
    ctx = _context(
        context_factory,
        entries=[_visit(NEIGHBOUR_CONTAINER), _visit(OWN_CONTAINER)],
        rented_data=_snapshot_with_own_container(),
    )

    entries = await _published_entries(ctx)

    assert [entry["container"] for entry in entries] == [OWN_CONTAINER]


@pytest.mark.asyncio
async def test_a_container_the_backend_never_issued_is_dropped(context_factory):
    # DAH-2757: our own name on a container the backend disowns proves nothing.
    ctx = _context(
        context_factory,
        entries=[_visit("filler_forged")],
        rented_data=_snapshot_with_own_container(),
    )

    assert await _published_entries(ctx) == []


@pytest.mark.asyncio
async def test_no_backend_snapshot_publishes_no_visit(context_factory):
    # Nothing can be attributed to an executor this cycle, so nothing is judged.
    ctx = _context(context_factory, entries=[_visit(OWN_CONTAINER)], rented_data=None)

    assert await _published_entries(ctx) == []
