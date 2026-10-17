import json
import logging
from unittest.mock import AsyncMock

import pytest
from neurons.validators.src.services import redis_service
from neurons.validators.src.services.redis_service import RedisService
from neurons.validators.src.services.task import result_handler
from neurons.validators.src.services.task.result_handler import ResultHandler
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from tests.helpers import build_state

VERIFIED_RECORD = {"count": 7, "failed": 0, "spec": "s", "uuids": "GPU-a"}


@pytest.fixture
def service():
    service = RedisService.__new__(RedisService)
    service.hset = AsyncMock()
    service.publish = AsyncMock()
    return service


@pytest.fixture
def enforced(monkeypatch):
    monkeypatch.setattr(redis_service.settings, "VERIFIED_RECORD_OWNER_ENFORCEMENT_ENABLED", True)


def _written_verified_record(service) -> dict:
    return json.loads(service.hset.await_args.args[2])


def _logs(caplog, key: str) -> int:
    return sum(key in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_a_foreign_reset_is_logged_and_still_written_by_default(service, caplog):
    with caplog.at_level(logging.WARNING, logger=redis_service.logger.name):
        await service.clear_verified_job_info(
            "other-hotkey", "exec-1", prev_info=VERIFIED_RECORD, record_owner="owner-hotkey"
        )

    assert _logs(caplog, "VERIFIED_RECORD_FOREIGN_WRITER") == 1
    assert _written_verified_record(service) == {**VERIFIED_RECORD, "count": 0}
    service.publish.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_foreign_write_or_reset_is_refused_under_enforcement(service, enforced):
    await service.clear_verified_job_info(
        "other-hotkey", "exec-1", prev_info=VERIFIED_RECORD, record_owner="owner-hotkey"
    )
    await service.set_verified_job_info(
        "other-hotkey", "exec-1", prev_info=VERIFIED_RECORD, record_owner="owner-hotkey"
    )

    service.hset.assert_not_awaited()
    service.publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_backend_owner_still_writes_under_enforcement(service, enforced, caplog):
    with caplog.at_level(logging.WARNING, logger=redis_service.logger.name):
        await service.set_verified_job_info(
            "owner-hotkey", "exec-1", prev_info=VERIFIED_RECORD, record_owner="owner-hotkey"
        )

    assert _written_verified_record(service)["count"] == 8
    assert _logs(caplog, "VERIFIED_RECORD_FOREIGN_WRITER") == 0


@pytest.mark.asyncio
async def test_no_backend_owner_lets_every_write_through_under_enforcement(service, enforced):
    # a new node: the backend creates its owner row on the first report
    await service.set_verified_job_info("any-hotkey", "exec-1", prev_info={}, spec="s")
    await service.clear_verified_job_info("any-hotkey", "exec-1", prev_info=VERIFIED_RECORD)

    assert service.hset.await_count == 2
    service.publish.assert_awaited_once()


def test_owner_enforcement_is_off_by_default():
    fields = type(redis_service.settings).model_fields
    assert fields["VERIFIED_RECORD_OWNER_ENFORCEMENT_ENABLED"].default is False


def _context(context_factory, owners: dict | None):
    rented = None
    if owners is not None:
        rented = RentedExecutorsResponse(executors={}, executor_owner_hotkeys=owners)
    ctx = context_factory(state=build_state(rented_data=rented))
    return ctx.model_copy(update={"clear_verified_job_info": True})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owners, expected",
    [
        ({"executor-123": "owner-hotkey"}, "owner-hotkey"),
        ({"other-executor": "someone"}, None),
        ({}, None),
        (None, None),
    ],
)
async def test_the_result_handler_passes_the_backend_owner(context_factory, owners, expected):
    redis = AsyncMock()

    await ResultHandler(redis_service=redis, dry_run=False)._persist_verification_data(
        "miner-hotkey",
        "executor-123",
        VERIFIED_RECORD,
        _context(context_factory, owners),
        success=False,
    )

    assert redis.clear_verified_job_info.await_args.kwargs["record_owner"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("owners, logged", [({"other-executor": "someone"}, 1), ({}, 0)])
async def test_an_executor_missing_from_a_sent_map_is_logged(
    context_factory, caplog, owners, logged
):
    # an older backend sends no map: nothing to log on every node of every cycle
    with caplog.at_level(logging.INFO, logger=result_handler.logger.name):
        ResultHandler._get_record_owner(_context(context_factory, owners))

    assert _logs(caplog, "VERIFIED_RECORD_OWNER_UNKNOWN") == logged


@pytest.mark.parametrize(
    "sent_id",
    [
        "0b6f8a2e-1c3d-4e5f-8a9b-0c1d2e3f4a5b",
        "0B6F8A2E1C3D4E5F8A9B0C1D2E3F4A5B",
        "{0b6f8a2e-1c3d-4e5f-8a9b-0c1d2e3f4a5b}",
    ],
)
def test_every_form_of_an_executor_id_finds_the_same_owner(sent_id):
    assert result_handler._executor_key(sent_id) == "0b6f8a2e-1c3d-4e5f-8a9b-0c1d2e3f4a5b"


@pytest.mark.asyncio
async def test_the_reset_carries_the_address_this_cycle_checked(context_factory):
    redis = AsyncMock()
    ctx = _context(context_factory, None)
    ctx = ctx.model_copy(update={"clear_verified_job_evidence": {"reason_code": "SPEC_CHANGED"}})

    await ResultHandler(redis_service=redis, dry_run=False)._persist_verification_data(
        "miner-hotkey", "executor-123", VERIFIED_RECORD, ctx, success=False
    )

    evidence = redis.clear_verified_job_info.await_args.kwargs["evidence"]
    assert evidence == {
        "reason_code": "SPEC_CHANGED",
        "executor_ip": ctx.executor.address,
        "executor_port": ctx.executor.port,
    }
