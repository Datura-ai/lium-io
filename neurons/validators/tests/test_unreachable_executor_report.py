"""An unrented node the miner knows but could not reach gets a failed row, not silence.

The miner answers each wave with the executors that accepted the validator's key (`executors`)
and every executor it lists (`known_executor_ids`, DAH-3338). An id in the second list and not the
first is a node the miner tried and could not reach. For a rented node the wave already writes
RENTED_EXECUTOR_NOT_LISTED (DAH-3558); for an unrented one it wrote nothing, so the node only
"stopped reporting". Prod, 12 Oct 20:24Z on: 5CAJ2tqg's 83963038, 11ac1c1c and 117215b3 got
"failed to register SSH key - executor did not answer" from the central miner on every wave and
not one validation row. With UNREACHABLE_EXECUTOR_REPORT_ENABLED the wave writes one failed row
per such node (EXECUTOR_UNREACHABLE_BY_MINER). Flag off: nothing is added.
"""

from unittest.mock import patch

import pytest
from datura.requests.miner_requests import AcceptSSHKeyRequest
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.task.availability import AvailabilityErrorCode, ReachSource, ReachTarget
from tests.test_rented_executor_not_listed import (
    LISTED,
    MISSING,
    make_payload,
    make_service,
    miner_returned,
    rented,
    request,
    rest_service,  # noqa: F401 - fixture
)
from tests.test_verification_started_report import _FakeMinerClient

UNREACHABLE = "83963038-0a21-41ab-8d4a-b915cd40702c"


@pytest.fixture
def flag_on(monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "UNREACHABLE_EXECUTOR_REPORT_ENABLED", True)


def accepted(listed_ids, known_ids) -> AcceptSSHKeyRequest:
    return AcceptSSHKeyRequest(executors=miner_returned(*listed_ids), known_executor_ids=known_ids)


def build(msg: AcceptSSHKeyRequest, rented_data=None, executor_id=None):
    return make_service()._build_unreachable_known_results(make_payload(), rented_data, msg, executor_id)


def test_known_unrented_executor_the_miner_could_not_reach_gets_one_failed_row(flag_on):
    msg = accepted([LISTED], [LISTED, UNREACHABLE])

    results = build(msg, RentedExecutorsResponse(executors={}))

    [result] = results
    assert result.executor_info.uuid == UNREACHABLE
    assert result.score == 0 and result.job_score == 0
    assert result.log_status == "error"
    assert result.spec is None
    assert result.is_rented is False
    assert result.failure_reason_code == AvailabilityErrorCode.EXECUTOR_UNREACHABLE_BY_MINER
    assert result.validation_event.reason_code == "EXECUTOR_UNREACHABLE_BY_MINER"
    [error] = result.availability_errors
    assert error["reach_source"] == ReachSource.MINER
    assert error["reach_target"] == ReachTarget.EXECUTOR_API
    assert "EXECUTOR_UNREACHABLE_BY_MINER" in result.log_text


def test_flag_off_adds_nothing(monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "UNREACHABLE_EXECUTOR_REPORT_ENABLED", False)
    msg = accepted([], [UNREACHABLE])

    results = build(msg)

    assert results == []


def test_miner_without_known_executor_ids_adds_nothing(flag_on):
    msg = accepted([LISTED], None)

    results = build(msg)

    assert results == []


def test_rented_executors_are_left_to_the_not_listed_row(flag_on):
    msg = accepted([], [MISSING])

    results = build(msg, RentedExecutorsResponse(executors={MISSING.upper(): rented()}))

    assert results == []


def test_single_executor_request_reports_nothing(flag_on):
    msg = accepted([], [UNREACHABLE])

    results = build(msg, executor_id=UNREACHABLE)

    assert results == []


def test_uuid_case_does_not_split_a_returned_executor_from_its_known_id(flag_on):
    msg = accepted([LISTED], [LISTED.upper()])

    results = build(msg)

    assert results == []


@pytest.mark.asyncio
async def test_rest_zero_executors_falls_through_to_the_unreachable_node_row(rest_service, flag_on):  # noqa: F811
    async def _make_rest_request(method, url, json_data, headers, timeout, log_extra, operation_name):
        if url.endswith("ssh-pubkey-submit"):
            return 200, accepted([], [UNREACHABLE]).model_dump(mode="json")
        return 200, {"message_type": "SSHKeyRemoved"}

    rest_service._make_rest_request = _make_rest_request

    out = await request(rest_service, RentedExecutorsResponse(executors={}))

    assert [r.executor_info.uuid for r in out["results"]] == [UNREACHABLE]
    assert out["results"][0].failure_reason_code == "EXECUTOR_UNREACHABLE_BY_MINER"


@pytest.mark.asyncio
async def test_websocket_cycle_carries_listed_verdict_and_unreachable_row(rest_service, flag_on, monkeypatch):  # noqa: F811
    from core.config import settings

    monkeypatch.setattr(settings, "USE_REST_API", False)
    msg = accepted([LISTED], [LISTED, UNREACHABLE])

    with patch("services.miner_service.MinerClient", return_value=_FakeMinerClient(msg)):
        out = await request(rest_service, RentedExecutorsResponse(executors={}))

    by_uuid = {r.executor_info.uuid: r for r in out["results"]}
    assert set(by_uuid) == {LISTED, UNREACHABLE}
    assert by_uuid[LISTED].score == 1.0
    assert by_uuid[UNREACHABLE].failure_reason_code == "EXECUTOR_UNREACHABLE_BY_MINER"
