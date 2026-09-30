"""An executor the miner lists twice is validated, and counted in its idle tier, once per cycle.

Regression: from 23 to 25 Sep 2026 one 1x B300 executor ran the full validation pipeline twice in
every cycle (two pipeline ids 2 ms apart). The 1x B300 idle tier (cap 4) counted one GPU more than
it held, and every node in the tier was paid a smaller share. `_claim_for_cycle` handed the
miner's list to the wave as is, and each entry started its own task.

The wave now keeps the first entry per executor uuid, with the express lane on and off.
"""

import logging
from unittest.mock import AsyncMock, Mock

import pytest
from datura.requests.miner_requests import AcceptSSHKeyRequest, ExecutorSSHInfo
from incentive.config import IncentiveConfig
from incentive.rental_price import RentalPriceIncentive
from payload_models.payloads import MinerJobEnryptedFiles, MinerJobRequestPayload
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.miner_service import CYCLE_DONE, EXPRESS_LANE, MinerService
from services.task.models import JobResult

B300 = "NVIDIA B300 SXM6 AC"
LISTED_TWICE = "Executor listed twice by miner; scored once"
EXPRESS_LANE_ON_AND_OFF = pytest.mark.parametrize("express_lane", [False, True], ids=["lane-off", "lane-on"])


def _executor_info(executor_id: str) -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid=executor_id,
        address="198.51.100.7",
        port=8001,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python3",
        root_dir="/root/app",
    )


def _idle_b300_1x(executor_id: str) -> JobResult:
    return JobResult(
        executor_info=_executor_info(executor_id),
        score=1.0,
        job_score=1.0,
        job_batch_id="2026-09-24 16:17:00",
        log_status="info",
        log_text="Validation task completed",
        gpu_model=B300,
        gpu_count=1,
        collateral_deposited=True,
        sysbox_runtime=True,
    )


def _payload() -> MinerJobRequestPayload:
    return MinerJobRequestPayload(
        job_batch_id="2026-09-24 16:17:00",
        miner_hotkey="miner-a",
        miner_coldkey="miner-coldkey",
        miner_address="192.0.2.10",
        miner_port=8091,
    )


@pytest.fixture
def miner_service(mocker, monkeypatch):
    """A MinerService over REST whose miner returns the executor list a test hands it; every task
    resolves to a passing idle 1x B300 result."""
    from core.config import settings

    my_key = Mock(ss58_address="validator-hotkey")
    my_key.sign.return_value = b"\x01\x02\x03"
    mocker.patch(
        "core.config.Settings.get_bittensor_wallet",
        return_value=Mock(get_hotkey=Mock(return_value=my_key)),
    )
    monkeypatch.setattr(settings, "USE_REST_API", True)
    ssh_service = mocker.Mock()
    ssh_service.generate_ssh_key.return_value = (b"---PRIV---", b"ssh-ed25519 pub")
    ssh_service.decrypt_payload.return_value = "---DECRYPTED-PRIV---"
    task_service = mocker.Mock()

    async def create_task(miner_info, executor_info, **_):
        return _idle_b300_1x(executor_info.uuid)

    task_service.create_task = AsyncMock(side_effect=create_task)
    service = MinerService(
        ssh_service=ssh_service,
        task_service=task_service,
        redis_service=mocker.AsyncMock(),
        attestation_service=Mock(maybe_issue_nonce=AsyncMock(return_value=None)),
    )
    mocker.patch("services.miner_service.measure_and_attach", AsyncMock())

    def miner_returns(*executor_ids: str) -> None:
        async def _make_rest_request(method, url, json_data, headers, timeout, log_extra, operation_name):
            if url.endswith("ssh-pubkey-submit"):
                return 200, AcceptSSHKeyRequest(
                    executors=[_executor_info(e) for e in executor_ids]
                ).model_dump(mode="json")
            return 200, {"message_type": "SSHKeyRemoved"}

        service._make_rest_request = _make_rest_request

    service.miner_returns = miner_returns
    return service


async def _request(service: MinerService) -> dict:
    return await service.request_job_to_miner(
        payload=_payload(),
        encrypted_files=MinerJobEnryptedFiles(
            encrypt_key="k",
            all_keys={},
            tmp_directory="/tmp",
            machine_scrape_file_name="scrape",
            machine_scrape_source="",
        ),
        rented_data=RentedExecutorsResponse(executors={}),
        default_docker_image_digests={},
    )


def _verified(service: MinerService) -> list[str]:
    return [c.kwargs["executor_info"].uuid for c in service.task_service.create_task.call_args_list]


async def _score(job_results: dict[str, list[JobResult]]) -> RentalPriceIncentive:
    redis = AsyncMock()
    redis.get_portion_per_gpu_type = AsyncMock(return_value=0.3)
    redis.get_executor_uptime = AsyncMock(return_value=9999)
    incentive = RentalPriceIncentive(
        IncentiveConfig(), redis, job_results,
        total_gpu_model_count_map={B300: sum(j.gpu_count for jobs in job_results.values() for j in jobs)},
    )
    price_provider = AsyncMock()
    price_provider.get_tao_price.return_value = 500.0
    price_provider.get_alpha_rate.return_value = 0.5
    incentive.price_provider = price_provider
    await incentive.calculate_mining_scores()
    return incentive


@EXPRESS_LANE_ON_AND_OFF
@pytest.mark.parametrize(
    ("listed", "verified", "listed_counts"),
    [
        (("exec-twin", "exec-twin", "exec-other"), ["exec-twin", "exec-other"], {"exec-twin": 2}),
        (("exec-a", "exec-b", "exec-c"), ["exec-a", "exec-b", "exec-c"], {}),
    ],
    ids=["listed-twice", "distinct"],
)
@pytest.mark.asyncio
async def test_each_executor_uuid_gets_one_validation_task(
    miner_service, monkeypatch, caplog, express_lane, listed, verified, listed_counts
):
    from core.config import settings

    monkeypatch.setattr(settings, "EXPRESS_LANE_ENABLED", express_lane)
    miner_service.miner_returns(*listed)

    with caplog.at_level(logging.WARNING):
        job = await _request(miner_service)

    assert _verified(miner_service) == verified
    assert [r.executor_info.uuid for r in job["results"]] == verified
    repeats = [r for r in caplog.records if LISTED_TWICE in r.getMessage()]
    assert {r.msg.extra["executor_uuid"]: r.msg.extra["listed_count"] for r in repeats} == listed_counts
    if express_lane:
        assert miner_service.in_flight == dict.fromkeys(verified, CYCLE_DONE)
    else:
        assert miner_service.in_flight == {}


@EXPRESS_LANE_ON_AND_OFF
@pytest.mark.asyncio
async def test_an_executor_listed_twice_is_one_gpu_in_its_idle_tier(miner_service, monkeypatch, express_lane):
    """End to end: the 23-25 Sep shape (a repeated 1x B300 beside three distinct ones, cap 4)
    fills the tier with 4 GPUs, not 5, so every node is paid in full."""
    from core.config import settings

    monkeypatch.setattr(settings, "EXPRESS_LANE_ENABLED", express_lane)
    miner_service.miner_returns("exec-twin", "exec-twin", "exec-b", "exec-c", "exec-d")

    job = await _request(miner_service)
    incentive = await _score({"miner-a": job["results"]})

    assert incentive.unrented_count_by_bucket[("B300", 1)] == 4
    assert incentive.cap_multiplier_by_bucket[("B300", 1)] == pytest.approx(1.0)


@EXPRESS_LANE_ON_AND_OFF
def test_the_claim_both_miner_paths_share_keeps_the_first_entry_per_uuid(
    miner_service, monkeypatch, caplog, express_lane
):
    """The WebSocket and the REST path both take the wave's executors from `_claim_for_cycle`."""
    from core.config import settings

    monkeypatch.setattr(settings, "EXPRESS_LANE_ENABLED", express_lane)
    first, repeat, other = _executor_info("exec-twin"), _executor_info("exec-twin"), _executor_info("exec-other")
    repeat.address = "203.0.113.9"

    with caplog.at_level(logging.WARNING):
        claimed = miner_service._claim_for_cycle(_payload(), [first, other, repeat, repeat], {})

    assert claimed == [first, other]
    assert claimed[0] is first
    repeats = [r for r in caplog.records if LISTED_TWICE in r.getMessage()]
    assert [r.msg.extra["listed_count"] for r in repeats] == [3]


def test_a_repeat_of_an_executor_the_express_lane_holds_stays_with_the_lane(miner_service, monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "EXPRESS_LANE_ENABLED", True)
    miner_service.in_flight["exec-new"] = EXPRESS_LANE
    listed = [_executor_info("exec-new"), _executor_info("exec-new"), _executor_info("exec-known")]

    claimed = miner_service._claim_for_cycle(_payload(), listed, {})

    assert [e.uuid for e in claimed] == ["exec-known"]
    assert miner_service.in_flight["exec-new"] == EXPRESS_LANE


def test_flag_off_a_list_without_repeats_is_handed_on_unchanged(miner_service, monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "EXPRESS_LANE_ENABLED", False)
    listed = [_executor_info("exec-a"), _executor_info("exec-b")]

    assert miner_service._claim_for_cycle(_payload(), listed, {}) == listed
