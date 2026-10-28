"""An executor the miner lists twice is validated, and counted in its idle tier, once per cycle.

Regression: from 23 to 25 Sep 2026 one 1x B300 executor ran the full validation pipeline twice in
every cycle (two pipeline ids 2 ms apart). The 1x B300 idle tier (cap 4) counted one GPU more than
it held, and every node in the tier was paid a smaller share. `_claim_for_cycle` handed the
miner's list to the wave as is, and each entry started its own task.

The wave now keeps the first entry per executor uuid, with the express lane on and off.
"""

from unittest.mock import AsyncMock

import pytest
from fixtures.rest_miner_fixtures import executor_info as _executor_info
from incentive.config import IncentiveConfig
from incentive.rental_price import RentalPriceIncentive
from payload_models.payloads import MinerJobEnryptedFiles, MinerJobRequestPayload
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.miner_service import MinerService
from services.task.models import JobResult

pytest_plugins = ["fixtures.rest_miner_fixtures"]

B300 = "NVIDIA B300 SXM6 AC"
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
def miner_task_result():
    return _idle_b300_1x


async def _request(service: MinerService, **kwargs) -> dict:
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
        **kwargs,
    )


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


@pytest.mark.asyncio
async def test_an_executor_listed_twice_is_one_gpu_in_its_idle_tier(rest_miner_service):
    """End to end: the 23-25 Sep shape (a repeated 1x B300 beside three distinct ones, cap 4)
    fills the tier with 4 GPUs, not 5, so every node is paid in full."""
    rest_miner_service.miner_returns("exec-twin", "exec-twin", "exec-b", "exec-c", "exec-d")

    job = await _request(rest_miner_service)
    incentive = await _score({"miner-a": job["results"]})

    assert incentive.unrented_count_by_bucket[("B300", 1)] == 4
    assert incentive.cap_multiplier_by_bucket[("B300", 1)] == pytest.approx(1.0)


def test_a_list_without_repeats_is_handed_on_unchanged(rest_miner_service):
    listed = [_executor_info("exec-a"), _executor_info("exec-b")]

    assert rest_miner_service._claim_for_cycle(_payload(), listed, {}) == listed
