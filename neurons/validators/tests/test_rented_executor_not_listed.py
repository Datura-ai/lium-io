"""DAH-3558: a rented node the miner left out of its answer gets a failed row, not silence.

A rented node the miner does not put in AcceptSSHKeyRequest.executors gets no pipeline, and
the wave writes nothing about it (the measurement is in the PR body). With
RENTED_EXECUTOR_NOT_LISTED_REPORT_ENABLED the wave writes one failed result
(score 0, RENTED_EXECUTOR_NOT_LISTED, availability error) per rented executor of the miner that
the backend lists and the miner did not return. Flag off: today's behaviour, nothing is added.
On main (no flag, no methods) every test here fails; that is the fail-old proof.

The helper tests drive `_build_not_listed_rented_results` directly (pure, like the manual-rental
synthesis next to it); the last tests drive the REST path end to end, so the empty-answer guard
and the result list are checked where the cycle reads them.
"""

from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from datura.requests.miner_requests import AcceptSSHKeyRequest, ExecutorSSHInfo
from payload_models.payloads import MinerJobRequestPayload
from protocol.vc_protocol.compute_requests import (
    ManualRentalInfo,
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)
from services.miner_service import MinerService
from services.task.availability import (
    AvailabilityErrorCode,
    ReachSource,
    ReachTarget,
    silence_availability_errors_on_our_own_outage,
)
from tests.test_express_lane import _cycle_inputs, _executor_info, _job_result
from tests.test_verification_started_report import _accepted, _FakeMinerClient

MINER_HOTKEY = "5MinerHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
OTHER_MINER_HOTKEY = "5OtherMinerHotkeyBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
MISSING = "700fea51-83d4-4404-88d9-167bb320c960"
LISTED = "2b02e4e0-e0cc-4833-bc62-98f018404d69"


def make_payload() -> MinerJobRequestPayload:
    return MinerJobRequestPayload(
        job_batch_id="2026-09-16 13:21:48",
        miner_hotkey=MINER_HOTKEY,
        miner_coldkey="5ColdkeyCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
        miner_address="192.0.2.10",
        miner_port=8091,
    )


def rented(*, miner_hotkey: str = MINER_HOTKEY, address: str = "198.51.100.7", port: str = "8001") -> RentedExecutor:
    return RentedExecutor(
        miner_hotkey=miner_hotkey,
        executor_ip_address=address,
        executor_ip_port=port,
        pods=[RentedPod(pod_id="pod-1", container_name="pod_pod-1")],
    )


def miner_returned(*executor_ids: str) -> list[ExecutorSSHInfo]:
    return [_executor_info(executor_id) for executor_id in executor_ids]


def make_service() -> MinerService:
    return MinerService(
        ssh_service=Mock(), task_service=Mock(), redis_service=Mock(), attestation_service=Mock()
    )


@pytest.fixture
def flag_on(monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "RENTED_EXECUTOR_NOT_LISTED_REPORT_ENABLED", True)


def build(service: MinerService, rented_data: RentedExecutorsResponse, *, listed_ids=(LISTED,), executor_id=None):
    return service._build_not_listed_rented_results(make_payload(), rented_data, miner_returned(*listed_ids), executor_id)


# --- the helper -----------------------------------------------------------------------------


def test_flag_off_adds_nothing(monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "RENTED_EXECUTOR_NOT_LISTED_REPORT_ENABLED", False)
    rented_data = RentedExecutorsResponse(executors={MISSING: rented(), LISTED: rented()})

    assert build(make_service(), rented_data) == []
    assert make_service()._has_not_listed_rented_executors(make_payload(), rented_data, [], None) is False


def test_rented_executor_the_miner_left_out_gets_one_failed_row(flag_on):
    """The regression: 700fea51, 16 Sep 13:21Z. The backend lists it as rented; the miner answered
    with its other executors only. Today the cycle writes nothing about it."""
    rented_data = RentedExecutorsResponse(executors={MISSING: rented(), LISTED: rented()})

    results = build(make_service(), rented_data)

    assert [r.executor_info.uuid for r in results] == [MISSING]
    result = results[0]
    assert result.score == 0 and result.job_score == 0
    assert result.log_status == "error"
    assert result.failure_reason_code == AvailabilityErrorCode.RENTED_EXECUTOR_NOT_LISTED
    assert result.is_rented is True
    # the backend skips its executor upsert on a null spec: a synthetic spec would flip the row
    assert result.spec is None
    # address and port come from the backend's rented list: the miner handed over nothing
    assert result.executor_info.address == "198.51.100.7"
    assert result.executor_info.port == 8001
    [error] = result.availability_errors
    assert error["reason_code"] == "RENTED_EXECUTOR_NOT_LISTED"
    assert error["reach_source"] == ReachSource.MINER
    assert error["reach_target"] == ReachTarget.EXECUTOR_API
    assert result.validation_event is not None
    assert result.validation_event.reason_code == "RENTED_EXECUTOR_NOT_LISTED"
    assert "RENTED_EXECUTOR_NOT_LISTED" in result.log_text
    assert "pod-1" in result.log_text


def test_listed_other_miners_and_manual_rentals_are_left_alone(flag_on):
    other = str(uuid4())
    manual = str(uuid4())
    rented_data = RentedExecutorsResponse(
        executors={
            LISTED: rented(),
            other: rented(miner_hotkey=OTHER_MINER_HOTKEY),
            manual: rented(),
        },
        manual_rental_executors={manual: ManualRentalInfo(gpu_model="NVIDIA H200", gpu_count=8)},
    )

    assert build(make_service(), rented_data) == []


def test_uuid_case_does_not_split_a_listed_executor_from_its_rental(flag_on):
    rented_data = RentedExecutorsResponse(executors={LISTED.upper(): rented()})

    assert build(make_service(), rented_data, listed_ids=(LISTED,)) == []


def test_single_executor_request_reports_nothing(flag_on):
    """The express lane and a rental key-submit ask the miner for one executor and keep their own
    bounded retry when the miner does not return it (DAH-2958); only the cycle's whole-miner
    request writes the outage row."""
    rented_data = RentedExecutorsResponse(executors={MISSING: rented()})

    assert build(make_service(), rented_data, listed_ids=(), executor_id=MISSING) == []
    assert make_service()._has_not_listed_rented_executors(make_payload(), rented_data, [], MISSING) is False


def test_probe_agrees_with_the_builder(flag_on):
    service = make_service()
    rented_data = RentedExecutorsResponse(executors={MISSING: rented(), LISTED: rented()})

    assert service._has_not_listed_rented_executors(make_payload(), rented_data, miner_returned(LISTED), None) is True
    assert service._has_not_listed_rented_executors(make_payload(), rented_data, miner_returned(LISTED, MISSING), None) is False
    assert service._has_not_listed_rented_executors(make_payload(), None, [], None) is False


# --- the DAH-2748 outage silencer ------------------------------------------------------------


def _ssh_unreachable_result(executor_id: str):
    result = _job_result(executor_id)
    result.availability_errors = [{"reason_code": "EXECUTOR_SSH_UNREACHABLE", "reach_source": "validator"}]
    return result


def _reached_result(executor_id: str):
    result = _job_result(executor_id)
    result.availability_errors = []
    return result


def test_not_listed_rows_do_not_tip_the_outage_ratio(flag_on):
    """Regression (review round 2): the not-listed rows carry an availability error the validator
    never measured, and they counted on both sides of the DAH-2748 ratio. One miner dropping eight
    rented nodes at once then read as our own outage and silenced a real SSH failure elsewhere."""
    rented_data = RentedExecutorsResponse(executors={str(uuid4()): rented() for _ in range(8)})
    not_listed = build(make_service(), rented_data, listed_ids=())
    assert len(not_listed) == 8
    ssh_failure = _ssh_unreachable_result(str(uuid4()))
    cycle = not_listed + [ssh_failure] + [_reached_result(str(uuid4())) for _ in range(5)]

    silenced = silence_availability_errors_on_our_own_outage(cycle)

    assert silenced == 0
    assert ssh_failure.availability_errors == [
        {"reason_code": "EXECUTOR_SSH_UNREACHABLE", "reach_source": "validator"}
    ]
    assert all(row.availability_errors for row in not_listed)


def test_not_listed_rows_are_not_silenced_and_do_not_pad_the_checked_count(flag_on):
    """The other side of the ratio: eight validator connects failed out of ten, our own outage.
    The not-listed rows neither dilute that share nor lose their error: the miner's answer is not
    a reading our egress can explain."""
    rented_data = RentedExecutorsResponse(executors={str(uuid4()): rented() for _ in range(20)})
    not_listed = build(make_service(), rented_data, listed_ids=())
    ssh_failures = [_ssh_unreachable_result(str(uuid4())) for _ in range(8)]
    reached = [_reached_result(str(uuid4())) for _ in range(2)]

    silenced = silence_availability_errors_on_our_own_outage(not_listed + ssh_failures + reached)

    assert silenced == 10
    assert all(result.availability_errors is None for result in ssh_failures + reached)
    assert all(row.availability_errors[0]["reason_code"] == "RENTED_EXECUTOR_NOT_LISTED" for row in not_listed)


# --- through the REST path -------------------------------------------------------------------


@pytest.fixture
def rest_service(mocker, monkeypatch, tmp_path):
    import services.file_encrypt_service as file_encrypt_service
    from core.config import settings

    # the cycle inputs name a job-files directory that must exist; keep it under pytest's tmp_path
    monkeypatch.setattr(file_encrypt_service, "JOB_FILES_ROOT", tmp_path)
    monkeypatch.setattr(settings, "USE_REST_API", True)
    my_key = Mock(ss58_address="validator-hotkey")
    my_key.sign.return_value = b"\x01\x02\x03"
    mocker.patch(
        "core.config.Settings.get_bittensor_wallet",
        return_value=Mock(get_hotkey=Mock(return_value=my_key)),
    )
    ssh_service = mocker.Mock()
    ssh_service.generate_ssh_key.return_value = (b"---PRIV---", b"ssh-ed25519 pub")
    ssh_service.decrypt_payload.return_value = "---DECRYPTED-PRIV---"
    task_service = mocker.Mock()

    async def create_task(miner_info, executor_info, **_):
        return _job_result(executor_info.uuid)

    task_service.create_task = AsyncMock(side_effect=create_task)
    service = MinerService(
        ssh_service=ssh_service,
        task_service=task_service,
        redis_service=mocker.AsyncMock(),
        attestation_service=Mock(maybe_issue_nonce=AsyncMock(return_value=None)),
    )
    mocker.patch("services.miner_service.measure_and_attach", AsyncMock())

    def miner_returns(*executor_ids: str):
        async def _make_rest_request(method, url, json_data, headers, timeout, log_extra, operation_name):
            if url.endswith("ssh-pubkey-submit"):
                return 200, AcceptSSHKeyRequest(executors=miner_returned(*executor_ids)).model_dump(mode="json")
            return 200, {"message_type": "SSHKeyRemoved"}

        service._make_rest_request = _make_rest_request

    service.miner_returns = miner_returns
    return service


async def request(service: MinerService, rented_data: RentedExecutorsResponse):
    return await service.request_job_to_miner(
        payload=make_payload(),
        encrypted_files=_cycle_inputs().encrypted_files,
        rented_data=rented_data,
        default_docker_image_digests={},
    )


@pytest.mark.asyncio
async def test_cycle_result_carries_the_listed_verdict_and_the_missing_rented_node(rest_service, flag_on):
    rest_service.miner_returns(LISTED)
    rented_data = RentedExecutorsResponse(executors={MISSING: rented(), LISTED: rented()})

    out = await request(rest_service, rented_data)

    by_uuid = {r.executor_info.uuid: r for r in out["results"]}
    assert set(by_uuid) == {LISTED, MISSING}
    assert by_uuid[LISTED].score == 1.0
    assert by_uuid[MISSING].failure_reason_code == "RENTED_EXECUTOR_NOT_LISTED"
    # the pipeline ran for the listed executor only: the missing node is never contacted
    assert rest_service.task_service.create_task.await_count == 1


@pytest.mark.asyncio
async def test_a_miner_whose_only_rented_node_is_down_gets_a_row_for_the_node_not_a_miner_failure(rest_service, flag_on):
    """Zero executors used to end as one miner-level failure row (uuid 1111-…) and nothing about
    the node; with the flag the node gets its own row."""
    rest_service.miner_returns()
    rented_data = RentedExecutorsResponse(executors={MISSING: rented()})

    out = await request(rest_service, rented_data)

    assert [r.executor_info.uuid for r in out["results"]] == [MISSING]
    assert out["results"][0].failure_reason_code == "RENTED_EXECUTOR_NOT_LISTED"


@pytest.mark.asyncio
async def test_websocket_path_zero_executors_falls_through_to_the_rented_node_row(
    rest_service, flag_on, monkeypatch
):
    """The same fall-through on the WebSocket path, the default transport."""
    from core.config import settings

    monkeypatch.setattr(settings, "USE_REST_API", False)
    rented_data = RentedExecutorsResponse(executors={MISSING: rented()})

    with patch("services.miner_service.MinerClient", return_value=_FakeMinerClient(_accepted())):
        out = await request(rest_service, rented_data)

    assert [r.executor_info.uuid for r in out["results"]] == [MISSING]
    assert out["results"][0].failure_reason_code == "RENTED_EXECUTOR_NOT_LISTED"


@pytest.mark.asyncio
async def test_flag_off_zero_executors_is_still_a_miner_failure(rest_service, monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "RENTED_EXECUTOR_NOT_LISTED_REPORT_ENABLED", False)
    rest_service.miner_returns()
    rented_data = RentedExecutorsResponse(executors={MISSING: rented()})

    out = await request(rest_service, rented_data)

    [result] = out["results"]
    assert result.executor_info.uuid == "11111111-1111-1111-1111-111111111111"
    assert "zero executors" in result.log_text
