"""The seeded offline executor in a cycle with UNREACHABLE_EXECUTOR_REPORT_ENABLED on: it gets its own failed row
instead of only dropping out of the results."""

import pytest
from services.task.availability import AvailabilityErrorCode

from core.config import settings
from services.ioc import ioc
from tests import lib
from tests.cycle.test_cycle import _one_cycle

pytestmark = pytest.mark.timeout(900)


def test_an_executor_the_miner_could_not_reach_gets_an_unreachable_by_miner_row(monkeypatch):
    """The miner lists the dead executor in known_executor_ids but cannot install the key there; with the
    report on, the cycle returns one failed result for it, EXECUTOR_UNREACHABLE_BY_MINER at score 0, beside
    the live executor's own verdict (lium-io#1577)."""
    monkeypatch.setattr(settings, "UNREACHABLE_EXECUTOR_REPORT_ENABLED", True)
    payload, result, _ = lib.run(_one_cycle(ioc))
    by_uuid = {str(r.executor_info.uuid).lower(): r for r in result["results"]}
    lib.write_artifact(
        "cycle-unreachable-result.json", [r.model_dump(mode="json") for r in result["results"]]
    )
    assert set(by_uuid) == {lib.EXECUTOR_UUID, lib.DEAD_EXECUTOR_UUID}, list(by_uuid)
    dead = by_uuid[lib.DEAD_EXECUTOR_UUID]
    code = AvailabilityErrorCode.EXECUTOR_UNREACHABLE_BY_MINER.value
    assert dead.validation_event and dead.validation_event.reason_code == code, dead.log_text
    assert dead.score == 0 and dead.log_status == "error" and not dead.is_rented
    assert dead.job_batch_id == payload.job_batch_id
    assert [e["reason_code"] for e in dead.availability_errors or []] == [code], (
        dead.availability_errors
    )
