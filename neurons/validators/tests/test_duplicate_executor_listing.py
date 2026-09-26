"""An executor the miner lists twice is validated, and counted in its idle tier, once per cycle.

Regression: from 23 to 25 Sep 2026 one 1x B300 executor ran the full validation pipeline twice in
every cycle (two pipeline ids 2 ms apart). The 1x B300 idle tier (cap 4) counted one GPU more than
it held, and every node in the tier was paid a smaller share. `_claim_for_cycle` handed the
miner's list to the wave as is, and each entry started its own task.

The wave now keeps the first entry per executor uuid, with the express lane on and off, and the
idle-tier count adds each executor uuid once per tier as a guard. Input without repeats is scored
exactly as before (the table at the end).
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
H100 = "NVIDIA H100 80GB HBM3"
H200 = "NVIDIA H200"
B200 = "NVIDIA B200"
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


async def _score(job_results: dict[str, list[JobResult]], disable_guard: bool = False) -> RentalPriceIncentive:
    redis = AsyncMock()
    redis.get_portion_per_gpu_type = AsyncMock(return_value=0.3)
    redis.get_executor_uptime = AsyncMock(return_value=9999)
    incentive = RentalPriceIncentive(
        IncentiveConfig(), redis, job_results,
        total_gpu_model_count_map={
            gpu_model: sum(j.gpu_count for jobs in job_results.values() for j in jobs if j.gpu_model == gpu_model)
            for gpu_model in {j.gpu_model for jobs in job_results.values() for j in jobs}
        },
    )
    price_provider = AsyncMock()
    price_provider.get_tao_price.return_value = 500.0
    price_provider.get_alpha_rate.return_value = 0.5
    incentive.price_provider = price_provider
    if disable_guard:
        # the pre-guard scoring, for the table's baseline
        incentive._mark_repeated_idle_copies = lambda: None
    await incentive.calculate_mining_scores()
    return incentive


def _idle_pay(incentive: RentalPriceIncentive, jobs: dict[str, list[JobResult]]) -> float:
    return sum(r.incentive or 0.0 for results in jobs.values() for r in results) / incentive.rental_share


def _reasons(result: JobResult) -> list[str]:
    return [reason.reason for reason in result.zero_incentive_reasons]


# --- The wave: one validation task per executor uuid ------------------------------------------


@EXPRESS_LANE_ON_AND_OFF
@pytest.mark.asyncio
async def test_an_executor_listed_twice_gets_one_validation_task(miner_service, monkeypatch, caplog, express_lane):
    from core.config import settings

    monkeypatch.setattr(settings, "EXPRESS_LANE_ENABLED", express_lane)
    miner_service.miner_returns("exec-twin", "exec-twin", "exec-other")

    with caplog.at_level(logging.WARNING):
        job = await _request(miner_service)

    assert _verified(miner_service) == ["exec-twin", "exec-other"]
    assert [r.executor_info.uuid for r in job["results"]] == ["exec-twin", "exec-other"]
    repeats = [r for r in caplog.records if LISTED_TWICE in r.getMessage()]
    assert len(repeats) == 1
    assert repeats[0].msg.extra["executor_uuid"] == "exec-twin"
    assert repeats[0].msg.extra["listed_count"] == 2
    if express_lane:
        assert miner_service.in_flight == {"exec-twin": CYCLE_DONE, "exec-other": CYCLE_DONE}
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
@pytest.mark.asyncio
async def test_distinct_executors_are_all_verified_and_nothing_is_logged(
    miner_service, monkeypatch, caplog, express_lane
):
    from core.config import settings

    monkeypatch.setattr(settings, "EXPRESS_LANE_ENABLED", express_lane)
    miner_service.miner_returns("exec-a", "exec-b", "exec-c")

    with caplog.at_level(logging.WARNING):
        job = await _request(miner_service)

    assert _verified(miner_service) == ["exec-a", "exec-b", "exec-c"]
    assert [r.executor_info.uuid for r in job["results"]] == ["exec-a", "exec-b", "exec-c"]
    assert LISTED_TWICE not in caplog.text


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


def test_flag_off_a_list_without_repeats_is_handed_on_as_is(miner_service, monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "EXPRESS_LANE_ENABLED", False)
    listed = [_executor_info("exec-a"), _executor_info("exec-b")]

    assert miner_service._claim_for_cycle(_payload(), listed, {}) is listed


# --- The guard: an executor is counted and paid once per cycle -----------------------------------


@pytest.mark.asyncio
async def test_a_uuid_repeated_within_a_miner_is_paid_one_share_and_counted_once(caplog):
    jobs = {
        "miner-a": [_idle_b300_1x("exec-twin"), _idle_b300_1x("exec-twin"), _idle_b300_1x("exec-b")],
        "miner-b": [_idle_b300_1x("exec-c"), _idle_b300_1x("exec-d")],
    }

    with caplog.at_level(logging.WARNING):
        incentive = await _score(jobs)

    first, repeat, distinct = jobs["miner-a"]
    assert incentive.unrented_count_by_bucket[("B300", 1)] == 4
    assert incentive.cap_multiplier_by_bucket[("B300", 1)] == pytest.approx(1.0)
    assert incentive._weighted_rate_sum_by_bucket[("B300", 1)] == pytest.approx(4 * distinct.hourly_rate)
    assert _idle_pay(incentive, jobs) == pytest.approx(1.0)
    assert first.incentive == pytest.approx(distinct.incentive)
    assert first.incentive == pytest.approx(incentive.rental_share / 4)
    assert repeat.incentive == 0.0
    assert repeat.eligible_for_rental_share is False
    assert _reasons(repeat) == ["duplicate_executor_in_cycle"]
    assert _reasons(first) == []
    assert incentive.miner_incentives["miner-a"] == pytest.approx(2 * incentive.rental_share / 4)
    assert "No unrented incentive for this copy" in caplog.text


@pytest.mark.parametrize("listed_first", ["miner-a", "miner-b"])
@pytest.mark.asyncio
async def test_a_uuid_listed_by_two_hotkeys_is_paid_once_under_the_lowest_hotkey(listed_first):
    """The cycle collects miners' results in completion order; the paid copy does not depend on it."""
    others = {"miner-c": [_idle_b300_1x(f"exec-{n}") for n in ("c", "d", "e", "f")]}
    twins = {"miner-a": [_idle_b300_1x("exec-twin")], "miner-b": [_idle_b300_1x("exec-twin")]}
    order = [listed_first, "miner-b" if listed_first == "miner-a" else "miner-a"]
    jobs = {hotkey: twins[hotkey] for hotkey in order} | others

    incentive = await _score(jobs)

    paid, unpaid = jobs["miner-a"][0], jobs["miner-b"][0]
    distinct = others["miner-c"][0]
    assert incentive.unrented_count_by_bucket[("B300", 1)] == 5
    assert _idle_pay(incentive, jobs) == pytest.approx(1.0)
    assert paid.incentive == pytest.approx(distinct.incentive)
    assert paid.incentive == pytest.approx(incentive.rental_share / 5)
    assert unpaid.incentive == 0.0
    assert _reasons(unpaid) == ["duplicate_executor_in_cycle"]
    assert _reasons(paid) == []
    assert incentive.miner_incentives.get("miner-b", 0.0) == 0.0


@pytest.mark.asyncio
async def test_a_uuid_reported_with_two_gpu_counts_is_counted_and_paid_once():
    jobs = {
        "miner-a": [_idle_b300_1x("exec-twin"), _idle_b300_1x("exec-b")],
        "miner-b": [_job("exec-twin", B300, 8)],
    }

    incentive = await _score(jobs)

    assert incentive.unrented_count_by_bucket[("B300", 1)] == 2
    assert incentive.unrented_count_by_bucket.get(("B300", 8), 0) == 0
    assert _idle_pay(incentive, jobs) == pytest.approx(1.0)
    assert jobs["miner-b"][0].incentive == 0.0
    assert _reasons(jobs["miner-b"][0]) == ["duplicate_executor_in_cycle"]


@pytest.mark.asyncio
async def test_a_split_nodes_rented_and_free_portions_are_not_duplicates():
    jobs = {
        "miner-a": [
            _job(
                "exec-split", B300, 4, is_rented=True, rented_gpu_count=2,
                supports_gpu_splitting=True, gpu_splitting_min_count=1,
            ),
        ],
    }

    incentive = await _score(jobs)

    assert incentive.unrented_count_by_bucket[("B300", 1)] == 2
    assert _reasons(jobs["miner-a"][0]) == []
    assert jobs["miner-a"][0].incentive_idle > 0


# --- Input without repeats: scores identical to the pre-guard counting ----------------------------


def _job(executor_id: str, gpu_model: str, gpu_count: int, **fields) -> JobResult:
    result = _idle_b300_1x(executor_id)
    result.gpu_model = gpu_model
    result.gpu_count = gpu_count
    for name, value in fields.items():
        setattr(result, name, value)
    return result


def _fleet_17_sep() -> dict[str, list[JobResult]]:
    jobs = {f"miner-{i}": [_job(f"exec-1x-{i:02d}", B300, 1)] for i in range(12)}
    jobs["miner-8x"] = [_job("exec-8x", B300, 8)]
    return jobs


def _fleet_rented_and_idle() -> dict[str, list[JobResult]]:
    return {
        "miner-a": [_job(f"exec-idle-{i}", B300, 1) for i in range(6)],
        "miner-b": [_job(f"exec-rented-{i}", B300, 1, is_rented=True) for i in range(2)],
    }


def _fleet_mixed_models() -> dict[str, list[JobResult]]:
    return {
        "miner-a": [_job(f"exec-b300-{i}", B300, 1) for i in range(3)] + [_job("exec-h100", H100, 1)],
        "miner-b": [_job("exec-h200-8x", H200, 8), _job("exec-b200-1x", B200, 1, sysbox_runtime=False)],
    }


def _fleet_split_nodes() -> dict[str, list[JobResult]]:
    """A partially rented split node's free portion shares its executor uuid with the rented
    portion; an idle split node may move to its split tier."""
    return {
        "miner-a": [
            _job(
                "exec-split-mixed", B300, 4, is_rented=True, rented_gpu_count=2,
                supports_gpu_splitting=True, gpu_splitting_min_count=1,
            ),
            _job("exec-b300-1x", B300, 1),
        ],
        "miner-b": [
            _job(
                f"exec-split-idle-{i}", B200, 8,
                supports_gpu_splitting=True, gpu_splitting_min_count=1,
            )
            for i in range(9)
        ],
    }


@pytest.mark.parametrize(
    ("fleet", "expected_counts"),
    [
        (_fleet_17_sep, {("B300", 1): 12, ("B300", 8): 8}),
        (_fleet_rented_and_idle, {("B300", 1): 6}),
        (_fleet_mixed_models, {("B300", 1): 3, ("H100", 1): 1, ("H200", 8): 8, ("B200", 1): 1}),
        (_fleet_split_nodes, {("B300", 1): 3, ("B200", 8): 64, ("B200", 1): 8}),
    ],
    ids=["17-sep-b300", "rented-and-idle", "mixed-models", "split-nodes"],
)
@pytest.mark.asyncio
async def test_input_without_repeats_scores_exactly_as_before(fleet, expected_counts):
    guarded_jobs, baseline_jobs = fleet(), fleet()

    guarded = await _score(guarded_jobs)
    baseline = await _score(baseline_jobs, disable_guard=True)

    assert {k: v for k, v in guarded.unrented_count_by_bucket.items() if v} == expected_counts
    assert guarded.unrented_count_by_bucket == baseline.unrented_count_by_bucket
    assert guarded.cap_multiplier_by_bucket == baseline.cap_multiplier_by_bucket
    assert guarded.total_rental_cost == baseline.total_rental_cost
    assert guarded.rental_share == baseline.rental_share
    for hotkey, results in guarded_jobs.items():
        for result, before in zip(results, baseline_jobs[hotkey], strict=True):
            assert result.executor_info.uuid == before.executor_info.uuid
            assert result.count_bucket == before.count_bucket
            assert result.effective_rate == before.effective_rate
            assert result.incentive == before.incentive
            assert result.incentive_idle == before.incentive_idle
            assert result.incentive_rented == before.incentive_rented
