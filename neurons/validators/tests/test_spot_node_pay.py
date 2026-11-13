"""Spot-node pay and the secure floor.

Spot: an idle spot node running Lium fillers is paid min(1.0 x its GPU configuration's average
filler revenue per GPU-hour, 0.8 x its secure rate before secure cap dilution), outside the secure
buckets, diluted above the secure cap of its GPU model and bucket in a spot pool of its own
(DAH-5944), and a GPU model's spot nodes together at most what its fillers earn per hour (DAH-5128).
Secure floor: an idle secure node's cap-diluted rate is raised to 1.0 x that average, even
above its listed rate. The two factors are separate constants. No cap (owner rule): spot pay and floor top-ups are paid on top of the
burn-capped rental share, up to the whole incentive pool.
"""

import json
import logging
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from core.config import settings, shared_client
from core.validator import settle_cycle_results
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.config import DEFAULT_PRICE, IncentiveConfig
from incentive import rental_price
from incentive.miner_incentive_log import MinerLogLine, ZeroIncentiveReason
from incentive.rental_price import ExecutorEstimateParams, RentalPriceIncentive
from incentive.utils import log_for_monitoring
from services.const import FIXED_RATIO, SECONDS_PER_BLOCK, TEMPO
from protocol.vc_protocol.compute_requests import FillerRevenueByGpuConfig, RentedExecutorsResponse
from services.task.result_handler import ResultHandler
from services.task_service import JobResult

from tests.helpers import build_state, default_executor

H100 = "NVIDIA H100 80GB HBM3"
HOURLY_RATE = 10.0
BUCKET_CAP = 8
# large enough that the rental share stays under the burn-emission cap, except where a test says so
TAO_PRICE = 1_000_000.0
# small enough that the rental share is capped at the burn emission
AT_CAP_TAO_PRICE = 1e-6
ALPHA_RATE = 1.0
# the owner's factors, written out rather than imported so that a change to either constant fails here
SPOT_FACTOR = 1.0
FLOOR_FACTOR = 1.0
SPOT_RENTAL_FACTOR = 0.5


def _config(cap: int = BUCKET_CAP) -> IncentiveConfig:
    return IncentiveConfig(
        rental_incentive_gpu_types=["H100"],
        max_unrented_gpus={"H100": {8: cap}},
        rental_prices_per_hour={H100: HOURLY_RATE},
        gpu_count_custom_prices={"*": {"*": DEFAULT_PRICE}},
    )


def _node(uuid: str, **overrides) -> JobResult:
    fields: dict = dict(
        executor_info=ExecutorSSHInfo(
            uuid=uuid,
            address="192.0.2.1",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python3",
            root_dir="/tmp",
        ),
        score=1.0,
        job_score=1.0,
        job_batch_id="2026-09-28 00:00:00",
        log_status="success",
        log_text="ok",
        gpu_model=H100,
        gpu_count=8,
        collateral_deposited=True,
        sysbox_runtime=True,
    )
    fields.update(overrides)
    return JobResult(**fields)


# what the GPU model's fillers earn per hour, more than any test's spot nodes are paid together
AMPLE_FILLER_EARNINGS = 1_000_000.0


def _spot(uuid: str = "spot-1", **overrides) -> JobResult:
    return _node(
        uuid,
        **{
            "is_spot": True,
            "is_on_spot_pay_list": True,
            "has_lium_filler": True,
            "filler_revenue_usd_per_hour": AMPLE_FILLER_EARNINGS,
            "filler_revenue_feed_ok": True,
            **overrides,
        },
    )


async def _run(
    results: list[JobResult],
    config: IncentiveConfig | None = None,
    tao_price: float = TAO_PRICE,
) -> RentalPriceIncentive:
    incentive = RentalPriceIncentive(config or _config(), AsyncMock(), {"hk": results}, {})
    incentive.price_provider = AsyncMock()
    incentive.price_provider.get_tao_price.return_value = tao_price
    incentive.price_provider.get_alpha_rate.return_value = ALPHA_RATE
    await incentive.calculate_mining_scores()
    return incentive


def _without_filler_averages(results: list[JobResult]) -> list[JobResult]:
    # the same cycle with no spot pay and no floor top-up: no floor without an average, and the spot
    # nodes off the pay list (an unmeasured one is seeded with its secure rate)
    for result in results:
        result.filler_revenue_per_gpu_hour = None
        result.is_on_spot_pay_list = False
    return results


def _codes(result: JobResult) -> list[str]:
    return [reason.reason for reason in result.zero_incentive_reasons]


def _paid_rate(result: JobResult) -> float:
    # a floored node: its listed or diluted rate plus the top-up; a spot node: its spot rate
    if result.spot_pay_candidate:
        return result.effective_rate or 0.0
    return (result.effective_rate or 0.0) + (result.floor_top_up_rate or 0.0)


def _paid_cost(incentive: RentalPriceIncentive) -> float:
    return sum(r.gpu_count * _paid_rate(r) for r in incentive.job_results["hk"])


def _share_per_usd_hour(tao_price: float) -> float:
    return (TEMPO * SECONDS_PER_BLOCK) / 3600 / FIXED_RATIO / (TEMPO * tao_price * ALPHA_RATE)


def _assert_pool_pays_exactly(incentive: RentalPriceIncentive) -> None:
    # the buckets pay from the rental share, spot pay and top-ups from the share on top of it
    assert incentive.total_rental_cost + incentive._unbucketed_rental_cost == pytest.approx(
        _paid_cost(incentive)
    )
    assert sum(r.incentive for r in incentive.job_results["hk"]) == pytest.approx(
        incentive.rental_share + incentive.unbucketed_share
    )


RENTAL_LINE = "Rental price incentive for executor is calculated successfully"
SPOT_LINE = "Spot-tier incentive for executor is calculated successfully"
FLOOR_APPLIED = "secure_filler_revenue_floor_applied"
FLOOR_NOT_PAID = "secure_filler_revenue_floor_not_paid"


def _logged(result: JobResult, marker: str) -> list[dict]:
    """The fields of every miner-facing line whose message starts with, or whose event is, `marker`."""
    lines: list[dict] = []
    for line in result.incentive_logs:
        message, _, fields = line.partition(" >>> ")
        parsed: dict = json.loads(fields)
        if message.startswith(marker) or parsed.get("event") == marker:
            lines.append(parsed)
    return lines


def _incentive_from_logged_formulas(result: JobResult) -> float:
    """The incentive a miner recomputes from the formula lines alone."""
    total: float = 0.0
    for f in _logged(result, RENTAL_LINE):
        total += f["rental_share"] * f["gpu_count"] * f["effective_rate"] / f["total_rental_cost"]
    for f in _logged(result, SPOT_LINE):
        total += f["unbucketed_share"] * f["gpu_count"] * f["effective_rate"] / f["unbucketed_rental_cost"]
    for f in _logged(result, FLOOR_APPLIED):
        total += f["unbucketed_share"] * f["gpu_count"] * f["floor_top_up_rate"] / f["unbucketed_rental_cost"]
    return total


def _half_clamp_tao_price() -> float:
    # for _over_cap_cycle(8.0) plus one spot node: the buckets alone stay under the burn cap, but
    # the extras would take the rental side past the pool by half their own share
    burn_emission = RentalPriceIncentive(_config(), AsyncMock(), {}, {}).total_burn_emission
    bucket_cost = 2 * 8 * HOURLY_RATE * 0.5
    extra_cost = 8 * SPOT_FACTOR * 2.0 + 2 * 8 * (FLOOR_FACTOR * 8.0 - HOURLY_RATE * 0.5)
    return _share_per_usd_hour(1.0) * (bucket_cost + extra_cost / 2) / burn_emission


def _top_up_pay(incentive: RentalPriceIncentive, result: JobResult) -> float:
    if result.floor_top_up_rate is None:
        return 0.0
    return (
        incentive.unbucketed_share
        * result.gpu_count
        * result.floor_top_up_rate
        / incentive._unbucketed_rental_cost
    )


@pytest.fixture
def spot_rental_on(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_SPOT_RENTAL_INCENTIVE", True)


@pytest.fixture(autouse=True)
def flags_default_off(monkeypatch):
    monkeypatch.setattr(settings, "SPOT_IDLE_PAY_SECURE_DECISION_ENABLED", False)
    monkeypatch.setattr(settings, "ENABLE_SPOT_RENTAL_INCENTIVE", False)


# ── spot-node pay ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_spot_is_paid_the_filler_side_when_it_is_lower():
    spot = _spot(filler_revenue_per_gpu_hour=2.0)
    secure = _node("secure-1")

    incentive = await _run([spot, secure])

    assert spot.effective_rate == pytest.approx(SPOT_FACTOR * 2.0)
    assert spot.incentive > 0
    assert spot.incentive / secure.incentive == pytest.approx(SPOT_FACTOR * 2.0 / HOURLY_RATE)
    assert spot.incentive_formula_version == "rental_price_v2"
    assert spot.mining_score == 0
    assert _codes(spot) == []
    _assert_pool_pays_exactly(incentive)


@pytest.mark.asyncio
async def test_one_spot_node_keeps_its_rate_while_the_model_earns_more():
    spot = _spot(filler_revenue_per_gpu_hour=2.0, filler_revenue_usd_per_hour=8 * 2.0)

    incentive = await _run([spot])

    assert spot.spot_pool_multiplier == pytest.approx(1.0)
    assert spot.effective_rate == pytest.approx(SPOT_FACTOR * 2.0)
    _assert_pool_pays_exactly(incentive)


@pytest.mark.asyncio
async def test_spot_nodes_of_a_model_share_what_its_fillers_earn(caplog):
    # Fish: n $/h of B300 filler earnings is split across all spot nodes, not paid per node
    earnings = 8 * 2.0
    spots = [_spot(f"spot-{i}", filler_revenue_per_gpu_hour=2.0, filler_revenue_usd_per_hour=earnings) for i in range(4)]
    secure = _node("secure-1")

    # a cap that fits all four spot nodes, so only the filler-earnings pool scales them
    incentive = await _run([*spots, secure], _config(cap=4 * 8))

    assert sum(s.gpu_count * s.effective_rate for s in spots) == pytest.approx(earnings)
    for spot in spots:
        assert spot.spot_pool_multiplier == pytest.approx(0.25)
        assert spot.effective_rate == pytest.approx(SPOT_FACTOR * 2.0 / 4)
    assert secure.effective_rate == pytest.approx(HOURLY_RATE)
    assert _logged(spots[0], SPOT_LINE)[0]["spot_pool_multiplier"] == pytest.approx(0.25)
    [model_line] = [r for r in caplog.records if r.getMessage().startswith("Spot pay shares the GPU model's filler earnings")]
    assert model_line.msg.extra["spot_pool_multiplier"] == pytest.approx(0.25)
    assert model_line.msg.extra["filler_revenue_usd_per_hour"] == pytest.approx(earnings)
    assert model_line.msg.extra["spot_gpu_count"] == 4 * 8
    _assert_pool_pays_exactly(incentive)


@pytest.mark.asyncio
async def test_a_model_s_spot_pool_does_not_dilute_another_model():
    a100 = "NVIDIA A100-SXM4-80GB"
    config = IncentiveConfig(
        rental_incentive_gpu_types=["H100", "A100"],
        max_unrented_gpus={"H100": {8: 2 * 8}, "A100": {8: BUCKET_CAP}},
        rental_prices_per_hour={H100: HOURLY_RATE, a100: HOURLY_RATE},
        gpu_count_custom_prices={"*": {"*": DEFAULT_PRICE}},
    )
    crowded = [_spot(f"h-{i}", filler_revenue_per_gpu_hour=2.0, filler_revenue_usd_per_hour=8.0) for i in range(2)]
    alone = _spot("a-1", gpu_model=a100, filler_revenue_per_gpu_hour=2.0)

    await _run([*crowded, alone], config)

    assert crowded[0].spot_pool_multiplier == pytest.approx(8.0 / 32.0)
    assert alone.spot_pool_multiplier == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_spot_without_measured_model_earnings_gets_zero():
    spot = _spot(filler_revenue_per_gpu_hour=2.0, filler_revenue_usd_per_hour=None)

    await _run([spot])

    assert spot.effective_rate == 0
    assert spot.incentive == 0


def test_model_earnings_add_every_usable_configuration():
    rented = RentedExecutorsResponse(
        executors={},
        filler_revenue_by_gpu_config=[
            FillerRevenueByGpuConfig(base_model="B300", gpu_count=8, usd_per_gpu_hour=3.0, gpu_hours=480.0),
            FillerRevenueByGpuConfig(base_model="B300", gpu_count=1, usd_per_gpu_hour=2.0, gpu_hours=48.0),
            FillerRevenueByGpuConfig(base_model="B300", gpu_count=2, usd_per_gpu_hour=9.0, gpu_hours=1.0),
            FillerRevenueByGpuConfig(base_model="H100", gpu_count=8, usd_per_gpu_hour=1.0, gpu_hours=480.0),
        ],
    )

    # 3 x 480 / 24 + 2 x 48 / 24; the 1 GPU-hour sample is under the minimum
    assert rented.get_filler_revenue_usd_per_hour("B300", 24.0) == pytest.approx(60.0 + 4.0)
    assert rented.get_filler_revenue_usd_per_hour("B200", 24.0) is None


@pytest.mark.asyncio
async def test_spot_is_paid_the_secure_side_when_it_is_lower():
    spot = _spot(filler_revenue_per_gpu_hour=4 * HOURLY_RATE)
    secure = _node("secure-1")

    await _run([spot, secure])

    assert spot.effective_rate == pytest.approx(rental_price.SPOT_IDLE_SECURE_RATE_SHARE * HOURLY_RATE)
    assert spot.incentive == pytest.approx(rental_price.SPOT_IDLE_SECURE_RATE_SHARE * secure.incentive)
    assert "secure_rate = 0.8 * hourly_rate" in _messages(spot, SPOT_LINE)[0]


@pytest.mark.asyncio
async def test_spot_secure_side_carries_the_nodes_own_sysbox_and_driver(monkeypatch):
    monkeypatch.setattr(settings, "PORTION_FOR_SYSBOX_UNRENTED", 0.5)
    spot = _spot(filler_revenue_per_gpu_hour=4 * HOURLY_RATE, sysbox_runtime=False)

    await _run([spot])

    assert spot.effective_rate == pytest.approx(rental_price.SPOT_IDLE_SECURE_RATE_SHARE * 0.5 * HOURLY_RATE)


@pytest.mark.asyncio
async def test_spot_secure_side_is_zero_below_the_minimum_driver():
    spot = _spot(filler_revenue_per_gpu_hour=2.0, nvidia_driver_version="535.104.05")

    await _run([spot, _node("secure-1")])

    assert spot.driver_multiplier == 0
    assert spot.effective_rate == 0
    assert spot.incentive == 0
    assert _codes(spot) == [ZeroIncentiveReason.NVIDIA_DRIVER_BELOW_MINIMUM]


@pytest.mark.asyncio
async def test_spot_without_a_lium_filler_gets_zero():
    spot = _spot(has_lium_filler=False, filler_revenue_per_gpu_hour=2.0)

    await _run([spot])

    assert spot.incentive == 0
    assert _codes(spot) == [ZeroIncentiveReason.SPOT_WITHOUT_LIUM_FILLER]


@pytest.mark.asyncio
async def test_spot_whose_config_earned_zero_is_not_seeded_with_its_secure_rate():
    spot = _spot(filler_revenue_per_gpu_hour=0.0, filler_revenue_usd_per_hour=0.0)

    await _run([spot])

    assert spot.incentive == 0.0
    assert _logged(spot, "Spot-tier rate seeded") == []
    # a measured zero records its own reason, as before seeding
    assert _codes(spot) == [ZeroIncentiveReason.SPOT_NO_FILLER_REVENUE_FOR_GPU_CONFIG]
    assert any("fillers earned nothing" in line for line in spot.incentive_logs)


@pytest.mark.asyncio
async def test_spot_with_no_average_from_a_failed_feed_is_not_seeded():
    # no usable average for any configuration (failed fetch, backend outage, older backend): keep the old result
    spot = _spot(filler_revenue_per_gpu_hour=None, filler_revenue_usd_per_hour=None, filler_revenue_feed_ok=False)

    await _run([spot])

    assert spot.incentive == 0
    assert _logged(spot, "Spot-tier rate seeded") == []
    assert _codes(spot) == [ZeroIncentiveReason.SPOT_NO_FILLER_REVENUE_FOR_GPU_CONFIG]


@pytest.mark.asyncio
async def test_secure_decision_does_not_seed_from_a_failed_feed(secure_decision_on):
    spot = _spot(has_lium_filler=False, filler_revenue_per_gpu_hour=None, filler_revenue_feed_ok=False)

    await _run([spot])

    assert spot.incentive == 0
    assert _codes(spot) == [ZeroIncentiveReason.SPOT_NO_FILLER_REVENUE_FOR_GPU_CONFIG]


@pytest.mark.asyncio
async def test_spot_whose_config_has_no_average_is_seeded_with_its_spot_ceiling():
    # Fish 23 Oct: every GPU model gets idle pay; the seed is not scaled down by the model's (tiny) earnings.
    # The seed is the spot ceiling, SPOT_IDLE_SECURE_RATE_SHARE of the secure rate, like any spot rate
    spot = _spot(filler_revenue_per_gpu_hour=None, filler_revenue_usd_per_hour=None)
    measured = _spot("spot-2", filler_revenue_per_gpu_hour=2.0, filler_revenue_usd_per_hour=4.0)

    incentive = await _run([spot, measured])

    seed = rental_price.SPOT_IDLE_SECURE_RATE_SHARE * HOURLY_RATE
    # the seeded node takes a Spot cap slot (lium-io#2133): 16 idle Spot GPUs in the 8-GPU cap halve both rates
    assert incentive.spot_count_by_bucket[("H100", 8)] == 16
    assert spot.spot_cap_multiplier == measured.spot_cap_multiplier == pytest.approx(0.5)
    assert spot.effective_rate == pytest.approx(seed * 0.5)
    assert spot.spot_pool_multiplier == pytest.approx(1.0)
    assert spot.incentive > 0
    assert _codes(spot) == []
    assert _logged(spot, "Spot-tier rate seeded")[0]["spot_rate_seed"] == pytest.approx(seed)
    # the measured node still shares only its model's filler earnings
    assert measured.spot_pool_multiplier == pytest.approx(4.0 / 8.0)
    _assert_pool_pays_exactly(incentive)


@pytest.mark.asyncio
async def test_spot_over_its_cap_is_diluted_and_does_not_dilute_secure_nodes():
    # DAH-5944: four 8x spot nodes in the 8-GPU cap of their own pool; one 8x secure node in its own
    spots = [_spot(f"spot-{i}", filler_revenue_per_gpu_hour=2.0) for i in range(4)]
    secure = _node("secure-1")

    incentive = await _run([*spots, secure])

    assert secure.unrented_cap_multiplier == 1.0
    assert secure.effective_rate == HOURLY_RATE
    assert incentive.unrented_count_by_bucket[("H100", 8)] == 8
    assert incentive.spot_count_by_bucket[("H100", 8)] == 32
    for spot in spots:
        assert spot.spot_cap_multiplier == pytest.approx(8 / 32)
        # the secure tier fields stay unset, so no reader counts a spot node as a secure tier member
        assert (spot.max_cap, spot.total_unrented_by_gpu_type, spot.unrented_cap_multiplier) == (None, None, None)
        assert spot.effective_rate == pytest.approx(SPOT_FACTOR * 2.0 * 8 / 32)
    [line] = _logged(spots[0], SPOT_LINE)
    assert (line["spot_gpus_in_bucket"], line["spot_cap"], line["count_bucket"]) == (
        32,
        BUCKET_CAP,
        8,
    )
    assert line["spot_cap_multiplier"] == pytest.approx(0.25)
    assert any("Spot cap dilution applied: 32 idle Spot GPUs" in m for m in _messages(spots[0], SPOT_LINE))
    _assert_pool_pays_exactly(incentive)


@pytest.mark.asyncio
@pytest.mark.parametrize("filler_revenue_per_gpu_hour", [100.0, None], ids=["measured", "unmeasured"])
async def test_four_8x_spot_nodes_over_the_cap_are_paid_the_capped_total(filler_revenue_per_gpu_hour):
    # 4 x 8 GPUs at 0.8 x the 10 $/h Secure rate is 256 $/h uncapped; the 8-GPU cap pays 8 x 8 = 64 $/h
    spots = [_spot(f"spot-{i}", filler_revenue_per_gpu_hour=filler_revenue_per_gpu_hour) for i in range(4)]

    await _run(spots)

    # a node with no measured average is unpaid, or once it is paid at its Secure-side rate it is in the cap pool too
    seeded: bool = hasattr(MinerLogLine, "spot_rate_seeded_from_secure_rate")
    paid: float = sum(s.gpu_count * (s.effective_rate or 0.0) for s in spots)
    assert paid == pytest.approx(64.0 if filler_revenue_per_gpu_hour is not None or seeded else 0.0)


@pytest.mark.asyncio
async def test_unpaid_spot_gpus_take_no_cap_slot():
    # a zero driver multiplier or a 0 filler average pays 0, so neither dilutes the paid node in the bucket
    paid = _spot("spot-paid", filler_revenue_per_gpu_hour=2.0)
    old_driver = _spot("spot-driver", filler_revenue_per_gpu_hour=2.0, nvidia_driver_version="535.104.05")
    no_filler_pay = _spot("spot-zero", filler_revenue_per_gpu_hour=0.0)

    incentive = await _run([paid, old_driver, no_filler_pay])

    assert incentive.spot_count_by_bucket[("H100", 8)] == 8
    assert paid.effective_rate == pytest.approx(SPOT_FACTOR * 2.0)
    assert old_driver.effective_rate == 0
    assert not no_filler_pay.effective_rate  # a measured zero is not priced at all (DAH-5721)


@pytest.mark.asyncio
async def test_spot_at_its_cap_is_not_diluted():
    spot = _spot(filler_revenue_per_gpu_hour=2.0)

    await _run([spot])

    assert spot.spot_cap_multiplier == 1.0
    assert spot.effective_rate == pytest.approx(SPOT_FACTOR * 2.0)
    assert not any("Spot cap dilution applied" in m for m in _messages(spot, SPOT_LINE))


@pytest.mark.asyncio
async def test_secure_nodes_over_their_cap_do_not_dilute_spot():
    secures = [_node(f"secure-{i}") for i in range(4)]
    spot = _spot(filler_revenue_per_gpu_hour=2.0)

    incentive = await _run([*secures, spot])

    assert secures[0].unrented_cap_multiplier == pytest.approx(8 / 32)
    assert incentive.spot_count_by_bucket[("H100", 8)] == 8
    assert spot.spot_cap_multiplier == 1.0
    assert spot.effective_rate == pytest.approx(SPOT_FACTOR * 2.0)


@pytest.mark.asyncio
async def test_the_filler_earnings_pool_scales_the_cap_diluted_spot_rates():
    # cap 8 of 16 spot GPUs halves each rate to 1.0; 16 GPUs x 1.0 = 16 $/h against 4 $/h of filler earnings
    spots = [
        _spot(f"spot-{i}", filler_revenue_per_gpu_hour=2.0, filler_revenue_usd_per_hour=4.0)
        for i in range(2)
    ]

    incentive = await _run(spots)

    for spot in spots:
        assert spot.spot_cap_multiplier == pytest.approx(0.5)
        assert spot.spot_pool_multiplier == pytest.approx(4.0 / 16.0)
        assert spot.effective_rate == pytest.approx(SPOT_FACTOR * 2.0 * 0.5 * 4.0 / 16.0)
    assert sum(s.gpu_count * s.effective_rate for s in spots) == pytest.approx(4.0)
    _assert_pool_pays_exactly(incentive)


@pytest.mark.asyncio
async def test_spot_in_a_config_with_no_secure_idle_pay_gets_zero():
    spot = _spot(filler_revenue_per_gpu_hour=2.0)

    await _run([spot], _config(cap=0))

    assert spot.incentive == 0
    assert ZeroIncentiveReason.NO_UNRENTED_CAPACITY_FOR_GPU_COUNT in _codes(spot)


@pytest.mark.asyncio
async def test_rented_spot_stays_out_of_both_pools():
    spot = _spot(is_rented=True, filler_revenue_per_gpu_hour=2.0)

    await _run([spot])

    assert spot.incentive == 0
    assert _codes(spot) == [ZeroIncentiveReason.SPOT_TIER]


# ── DAH-4299: the secure decision on whether a spot node is paid ─────────────


@pytest.fixture
def secure_decision_on(monkeypatch):
    monkeypatch.setattr(settings, "SPOT_IDLE_PAY_SECURE_DECISION_ENABLED", True)


@pytest.mark.asyncio
async def test_secure_decision_pays_an_idle_spot_node_no_lium_filler_runs_on(secure_decision_on):
    spot = _spot(has_lium_filler=False, filler_revenue_per_gpu_hour=2.0)

    incentive = await _run([spot, _node("secure-1")])

    assert spot.effective_rate == pytest.approx(SPOT_FACTOR * 2.0)
    assert spot.incentive > 0
    assert _codes(spot) == []
    _assert_pool_pays_exactly(incentive)


@pytest.mark.asyncio
async def test_secure_decision_seeds_a_spot_node_with_no_average(secure_decision_on):
    spot = _spot(has_lium_filler=False, filler_revenue_per_gpu_hour=None)

    await _run([spot])

    assert spot.effective_rate == pytest.approx(rental_price.SPOT_IDLE_SECURE_RATE_SHARE * HOURLY_RATE)
    assert spot.incentive > 0
    assert _codes(spot) == []


@pytest.mark.asyncio
async def test_secure_decision_refuses_a_spot_node_an_idle_pool_gate_refuses(
    secure_decision_on, monkeypatch
):
    p90_config = shared_client.config.model_copy(update={"machine_prices_p90": {H100: 1.0}})
    monkeypatch.setattr(shared_client, "_config", p90_config)
    spot = _spot(filler_revenue_per_gpu_hour=2.0)
    spot.executor_info.price_per_gpu = 5.0

    await _run([spot])

    assert spot.incentive == 0
    assert spot.spot_pay_candidate is False
    assert _codes(spot) == [ZeroIncentiveReason.PRICE_ABOVE_MARKET_P90_SOFT_LIMIT]


@pytest.mark.asyncio
async def test_secure_decision_refuses_a_spot_gpu_model_outside_the_program(secure_decision_on):
    spot = _spot(filler_revenue_per_gpu_hour=2.0)
    config = _config()
    config.rental_incentive_gpu_types = []

    await _run([spot], config)

    assert spot.incentive == 0
    assert _codes(spot) == [ZeroIncentiveReason.GPU_MODEL_NOT_ELIGIBLE_FOR_UNRENTED_INCENTIVE]


@pytest.mark.asyncio
async def test_secure_decision_leaves_a_rented_spot_node_out_of_both_pools(secure_decision_on):
    spot = _spot(is_rented=True, has_lium_filler=False, filler_revenue_per_gpu_hour=2.0)

    await _run([spot])

    assert spot.incentive == 0
    assert _codes(spot) == [ZeroIncentiveReason.SPOT_TIER]


@pytest.mark.asyncio
async def test_rented_spot_earns_half_of_a_rented_secure_node(spot_rental_on):
    spot = _spot(is_rented=True)
    secure = _node("secure-1", is_rented=True)
    redis_service = AsyncMock()
    redis_service.get_portion_per_gpu_type.return_value = 1.0
    incentive = RentalPriceIncentive(_config(), redis_service, {"hk": [spot, secure]}, {H100: 16})
    incentive.price_provider = AsyncMock()
    incentive.price_provider.get_tao_price.return_value = TAO_PRICE
    incentive.price_provider.get_alpha_rate.return_value = ALPHA_RATE

    await incentive.calculate_mining_scores()

    assert secure.incentive > 0
    assert spot.incentive == pytest.approx(SPOT_RENTAL_FACTOR * secure.incentive)
    assert spot.mining_score == pytest.approx(SPOT_RENTAL_FACTOR * secure.mining_score)
    assert _codes(spot) == []


@pytest.mark.asyncio
async def test_rented_spot_counts_in_the_gpu_denominator_only_when_it_earns(monkeypatch):
    results = {"hk": [_spot(is_rented=True), _node("secure-1", is_rented=True)]}

    monkeypatch.setattr(settings, "ENABLE_SPOT_RENTAL_INCENTIVE", False)
    assert RentalPriceIncentive(_config(), AsyncMock(), results, {}).get_snapshot().mining.total_gpu_model_count_map == {
        H100: 8
    }

    monkeypatch.setattr(settings, "ENABLE_SPOT_RENTAL_INCENTIVE", True)
    assert RentalPriceIncentive(_config(), AsyncMock(), results, {}).get_snapshot().mining.total_gpu_model_count_map == {
        H100: 16
    }


@pytest.mark.parametrize("flag_on, gpus", [(False, 8), (True, 16)], ids=["flag-off", "flag-on"])
def test_the_live_gpu_denominator_counts_rented_spot_only_when_it_earns(monkeypatch, flag_on, gpus):
    monkeypatch.setattr(settings, "ENABLE_SPOT_RENTAL_INCENTIVE", flag_on)
    spot = _spot(is_rented=True)
    spot.executor_info.port = 8081
    spot.executor_info.ssh_port = 2222  # its own machine: one SSH endpoint under two hotkeys is a duplicate
    results = {"hk-spot": [spot], "hk-secure": [_node("secure-1", is_rented=True)]}

    assert settle_cycle_results(results, {}) == {H100: gpus}


@pytest.mark.asyncio
async def test_the_rented_spot_score_log_carries_the_factor_and_the_final_score(spot_rental_on):
    spot = _spot(is_rented=True)
    secure = _node("secure-1", is_rented=True)
    redis_service = AsyncMock()
    redis_service.get_portion_per_gpu_type.return_value = 1.0
    incentive = RentalPriceIncentive(_config(), redis_service, {"hk": [spot, secure]}, {H100: 16})

    await incentive.calculate_executor_score(spot)
    await incentive.calculate_executor_score(secure)

    spot_line = next(line for line in spot.incentive_logs if "Mining score is calculated" in line)
    secure_line = next(line for line in secure.incentive_logs if "Mining score is calculated" in line)
    assert "spot_rental_factor" in spot_line
    assert f'"mining_score": {spot.mining_score}' in spot_line
    assert "spot_rental_factor" not in secure_line


@pytest.mark.asyncio
async def test_rented_spot_as_a_restriction_still_earns_nothing(spot_rental_on):
    spot = _spot(is_rented=True, is_on_spot_pay_list=False)

    await _run([spot, _node("secure-1", is_rented=True)])

    assert spot.incentive == 0
    assert _codes(spot) == [ZeroIncentiveReason.SPOT_TIER]


@pytest.mark.asyncio
async def test_spot_rental_flag_leaves_an_idle_spot_node_on_spot_pay(spot_rental_on):
    spot = _spot(filler_revenue_per_gpu_hour=2.0)

    await _run([spot, _node("secure-1")])

    assert spot.spot_pay_candidate is True
    assert spot.effective_rate == pytest.approx(SPOT_FACTOR * 2.0)
    assert _codes(spot) == []


def _split_config() -> IncentiveConfig:
    return IncentiveConfig(
        rental_incentive_gpu_types=["H100"],
        max_unrented_gpus={"H100": {8: BUCKET_CAP, 1: BUCKET_CAP}},
        rental_prices_per_hour={H100: HOURLY_RATE},
        gpu_count_custom_prices={"*": {"*": DEFAULT_PRICE}},
    )


def _split_spot(**overrides) -> JobResult:
    # 4 of 8 GPUs rented in 1-GPU splits; the average is the node's 8x configuration's
    return _spot(
        "split-spot",
        **{
            "is_rented": True,
            "rented_gpu_count": 4,
            "supports_gpu_splitting": True,
            "gpu_splitting_min_count": 1,
            "filler_revenue_per_gpu_hour": 2.0,
            **overrides,
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("secure_decision", [False, True])
async def test_the_free_remainder_of_a_partially_rented_spot_node_takes_spot_idle_pay(monkeypatch, secure_decision):
    monkeypatch.setattr(settings, "SPOT_IDLE_PAY_SECURE_DECISION_ENABLED", secure_decision)
    split = _split_spot()

    incentive = await _run([split], _split_config())

    # only the 4 free GPUs are a spot pay candidate, rated at the 1-GPU split tier
    [(_, free)] = incentive._spot_pay_candidates
    assert free.is_split_remainder and free.gpu_count == 4
    assert free.effective_rate == pytest.approx(SPOT_FACTOR * 2.0)
    assert split.incentive_idle > 0
    # the rented part earns no rental incentive with the spot rental flag off
    assert split.incentive_rented == 0
    assert split.incentive == pytest.approx(split.incentive_idle)
    assert split.gpu_count == 8
    assert ZeroIncentiveReason.SPOT_NO_FILLER_REVENUE_FOR_GPU_CONFIG not in _codes(split)


@pytest.mark.asyncio
async def test_a_spot_remainder_and_its_rented_part_are_each_paid_once(spot_rental_on):
    split = _split_spot()
    redis_service = AsyncMock()
    redis_service.get_portion_per_gpu_type.return_value = 1.0
    incentive = RentalPriceIncentive(_split_config(), redis_service, {"hk": [split]}, {H100: 8})
    incentive.price_provider = AsyncMock()
    incentive.price_provider.get_tao_price.return_value = TAO_PRICE
    incentive.price_provider.get_alpha_rate.return_value = ALPHA_RATE

    await incentive.calculate_mining_scores()

    # the rented 4 GPUs earn rental incentive only, the free 4 spot idle pay only
    [(_, free)] = incentive._spot_pay_candidates
    assert free.gpu_count == 4 and not free.is_rented
    assert rental_price.earns_spot_rental_incentive(free) is False
    assert incentive._unbucketed_rental_cost == pytest.approx(4 * SPOT_FACTOR * 2.0)
    assert split.incentive_rented > 0
    assert split.incentive_idle > 0
    assert split.incentive == pytest.approx(split.incentive_rented + split.incentive_idle)


@pytest.mark.asyncio
async def test_a_spot_remainder_without_a_lium_filler_is_not_paid():
    split = _split_spot(has_lium_filler=False)

    incentive = await _run([split], _split_config())

    assert incentive._spot_pay_candidates == []
    assert split.incentive == 0
    assert ZeroIncentiveReason.SPOT_WITHOUT_LIUM_FILLER in _codes(split)


@pytest.mark.asyncio
@pytest.mark.parametrize("secure_decision", [False, True])
async def test_a_spot_remainder_below_the_port_floor_is_not_paid(monkeypatch, secure_decision):
    monkeypatch.setattr(settings, "SPOT_IDLE_PAY_SECURE_DECISION_ENABLED", secure_decision)
    split = _split_spot(spec={"available_port_count": 2})

    incentive = await _run([split], _split_config())

    assert incentive._spot_pay_candidates == []
    assert split.incentive == 0
    assert ZeroIncentiveReason.PORT_LIMITED_REMAINDER in _codes(split)


@pytest.mark.asyncio
async def test_a_spot_remainder_is_paid_only_for_its_splits_with_a_free_port():
    # 4 free 1-GPU splits, 3 free ports: 3 GPUs paid, the merged row still reports 8
    split = _split_spot(spec={"available_port_count": 3})

    incentive = await _run([split], _split_config())

    [(_, free)] = incentive._spot_pay_candidates
    assert free.gpu_count == 3
    assert split.gpu_count == 8
    assert split.incentive_idle > 0


@pytest.mark.asyncio
async def test_spot_blocked_by_another_exclusion_is_not_paid():
    spot = _spot(is_provider_banned=True, filler_revenue_per_gpu_hour=2.0)

    await _run([spot])

    assert spot.incentive == 0
    assert spot.spot_pay_candidate is False
    assert _codes(spot) == [ZeroIncentiveReason.BANNED_NETWORK_ABUSE]


@pytest.mark.asyncio
async def test_a_spot_node_off_the_pay_list_is_scored_as_before():
    # a banned, pinned or no-incentive-rental node: spot, but not on the spot pay list
    def cycle() -> list[JobResult]:
        return [
            _spot("restricted", is_on_spot_pay_list=False, filler_revenue_per_gpu_hour=2.0),
            _node("secure-1"),
        ]

    off = _without_filler_averages(cycle())
    await _run(off)
    on = cycle()
    await _run(on)

    assert on[0].incentive == 0
    assert on[0].spot_pay_candidate is False
    assert _codes(on[0]) == [ZeroIncentiveReason.SPOT_TIER]
    assert on[0].incentive_logs == off[0].incentive_logs
    assert on[1].incentive == pytest.approx(off[1].incentive)


@pytest.mark.asyncio
async def test_a_spot_result_built_without_the_providers_choice_is_not_paid():
    spot = _node("spot-1", is_spot=True, has_lium_filler=True, filler_revenue_per_gpu_hour=2.0)

    await _run([spot])

    assert spot.is_on_spot_pay_list is False
    assert spot.incentive == 0
    assert _codes(spot) == [ZeroIncentiveReason.SPOT_TIER]


# ── secure floor ─────────────────────────────────────────────────────────────


def _over_cap_cycle(filler_revenue: float | None) -> list[JobResult]:
    # two 8x nodes in an 8-GPU bucket: cap multiplier 0.5, diluted rate HOURLY_RATE / 2
    return [
        _node("secure-1", filler_revenue_per_gpu_hour=filler_revenue),
        _node("secure-2", filler_revenue_per_gpu_hour=filler_revenue),
    ]


@pytest.mark.asyncio
async def test_floor_lifts_a_diluted_rate_after_dilution():
    incentive = await _run(_over_cap_cycle(8.0))

    for node in incentive.job_results["hk"]:
        assert node.unrented_cap_multiplier == 0.5
        assert node.effective_rate == HOURLY_RATE * 0.5
        assert node.floor_top_up_rate == pytest.approx(FLOOR_FACTOR * 8.0 - HOURLY_RATE * 0.5)
        assert _paid_rate(node) == pytest.approx(FLOOR_FACTOR * 8.0)
        [line] = _logged(node, FLOOR_APPLIED)
        assert line["diluted_rate"] == HOURLY_RATE * 0.5
        assert line["floored_rate"] == pytest.approx(FLOOR_FACTOR * 8.0)
    _assert_pool_pays_exactly(incentive)


@pytest.mark.asyncio
async def test_floor_lifts_above_the_listed_rate():
    incentive = await _run(_over_cap_cycle(4 * HOURLY_RATE))

    for node in incentive.job_results["hk"]:
        assert node.hourly_rate == HOURLY_RATE
        assert node.effective_rate == HOURLY_RATE * 0.5
        assert _paid_rate(node) == pytest.approx(FLOOR_FACTOR * 4 * HOURLY_RATE)
    _assert_pool_pays_exactly(incentive)


@pytest.mark.asyncio
async def test_floor_leaves_a_bucket_with_no_capacity_at_zero():
    config = IncentiveConfig(
        rental_incentive_gpu_types=["H100"],
        max_unrented_gpus={"H100": {8: BUCKET_CAP, 1: 0}},
        rental_prices_per_hour={H100: HOURLY_RATE},
        gpu_count_custom_prices={"*": {"*": DEFAULT_PRICE}},
    )
    no_capacity = _node("secure-1x", gpu_count=1, filler_revenue_per_gpu_hour=8.0)

    incentive = await _run([*_over_cap_cycle(8.0), no_capacity], config)

    assert no_capacity.max_cap == 0
    assert no_capacity.effective_rate == 0
    assert no_capacity.incentive == 0
    _assert_pool_pays_exactly(incentive)


@pytest.mark.asyncio
async def test_floor_below_the_diluted_rate_changes_nothing():
    incentive = await _run(_over_cap_cycle(2.0))

    for node in incentive.job_results["hk"]:
        assert node.effective_rate == HOURLY_RATE * 0.5
        assert node.floor_top_up_rate is None
        assert not _logged(node, FLOOR_APPLIED)
    _assert_pool_pays_exactly(incentive)


@pytest.mark.asyncio
async def test_floor_without_an_average_changes_nothing():
    incentive = await _run(_over_cap_cycle(None))

    for node in incentive.job_results["hk"]:
        assert node.effective_rate == HOURLY_RATE * 0.5


@pytest.mark.asyncio
async def test_floor_keeps_the_nodes_own_penalties(monkeypatch):
    monkeypatch.setattr(settings, "PORTION_FOR_SYSBOX_UNRENTED", 0.5)
    incentive = await _run(
        [
            _node("secure-1", filler_revenue_per_gpu_hour=8.0, sysbox_runtime=False),
            _node("secure-2", filler_revenue_per_gpu_hour=8.0),
        ]
    )

    penalised, full = incentive.job_results["hk"]
    assert penalised.effective_rate == pytest.approx(HOURLY_RATE * 0.5 * 0.5)
    assert _paid_rate(penalised) == pytest.approx(FLOOR_FACTOR * 8.0 * 0.5)
    assert _paid_rate(full) == pytest.approx(FLOOR_FACTOR * 8.0)
    _assert_pool_pays_exactly(incentive)


@pytest.mark.asyncio
async def test_a_floored_node_below_the_minimum_driver_adds_no_floor_cost():
    incentive = await _run(
        [
            _node("old-driver", filler_revenue_per_gpu_hour=8.0, nvidia_driver_version="535.104.05"),
            _node("secure-2", filler_revenue_per_gpu_hour=8.0),
        ]
    )

    old_driver, full = incentive.job_results["hk"]
    assert old_driver.driver_multiplier == 0
    assert old_driver.incentive == 0
    assert _paid_rate(full) == pytest.approx(FLOOR_FACTOR * 8.0)
    assert incentive._unbucketed_rental_cost == pytest.approx(8 * (FLOOR_FACTOR * 8.0 - HOURLY_RATE * 0.5))
    _assert_pool_pays_exactly(incentive)


@pytest.mark.asyncio
async def test_floor_skips_the_free_remainder_of_a_partially_rented_split_node(
    monkeypatch
):
    config = IncentiveConfig(
        rental_incentive_gpu_types=["H100"],
        max_unrented_gpus={"H100": {8: BUCKET_CAP, 1: 1}},
        rental_prices_per_hour={H100: HOURLY_RATE},
        gpu_count_custom_prices={"*": {"*": DEFAULT_PRICE}},
    )
    # the average is the whole 8x configuration's; the 4 free GPUs are rated at the 1x tier
    split = _node(
        "split-1",
        is_rented=True,
        rented_gpu_count=4,
        supports_gpu_splitting=True,
        gpu_splitting_min_count=1,
        filler_revenue_per_gpu_hour=8.0,
    )

    incentive = await _run([split], config)

    free = split.incentive_formula_inputs["unrented"]
    assert free["unrented_cap_multiplier"] == 0.25
    assert free["effective_rate"] == HOURLY_RATE * 0.25
    # no top-up on the remainder: nothing paid from the unbucketed share, nothing in its cost
    assert "unbucketed_share" not in free
    assert "floor_top_up_rate" not in free
    assert split.floor_top_up_rate is None
    assert incentive._unbucketed_rental_cost == 0
    assert incentive.unbucketed_share == 0


@pytest.mark.asyncio
async def test_spot_pay_and_the_floor_apply_together():
    both = await _run([*_over_cap_cycle(8.0), _spot(filler_revenue_per_gpu_hour=2.0)])
    assert [_paid_rate(n) for n in both.job_results["hk"]] == pytest.approx(
        [FLOOR_FACTOR * 8.0, FLOOR_FACTOR * 8.0, SPOT_FACTOR * 2.0]
    )
    _assert_pool_pays_exactly(both)


def _messages(result: JobResult, marker: str) -> list[str]:
    return [
        line.partition(" >>> ")[0]
        for line in result.incentive_logs
        if line.startswith(marker) or f'"event": "{marker}"' in line
    ]


@pytest.mark.asyncio
async def test_the_spot_factor_moves_spot_pay_alone(monkeypatch):
    monkeypatch.setattr(rental_price, "FILLER_REVENUE_PAY_FACTOR", 0.5)

    incentive = await _run([*_over_cap_cycle(8.0), _spot(filler_revenue_per_gpu_hour=2.0)])

    secure_1, secure_2, spot = incentive.job_results["hk"]
    assert spot.effective_rate == pytest.approx(0.5 * 2.0)
    assert _logged(spot, SPOT_LINE)[0]["filler_rate"] == pytest.approx(0.5 * 2.0)
    assert "min(0.5 * filler_revenue_per_gpu_hour" in _messages(spot, SPOT_LINE)[0]
    assert [_paid_rate(secure_1), _paid_rate(secure_2)] == pytest.approx([FLOOR_FACTOR * 8.0] * 2)
    assert f"below {FLOOR_FACTOR:g} x the" in _messages(secure_1, FLOOR_APPLIED)[0]
    _assert_pool_pays_exactly(incentive)


@pytest.mark.asyncio
async def test_the_floor_factor_moves_the_floor_alone(monkeypatch):
    monkeypatch.setattr(rental_price, "SECURE_FILLER_REVENUE_FLOOR_FACTOR", 0.8)

    incentive = await _run([*_over_cap_cycle(8.0), _spot(filler_revenue_per_gpu_hour=2.0)])
    at_cap = await _run(_over_cap_cycle(8.0), tao_price=AT_CAP_TAO_PRICE)

    secure_1, secure_2, spot = incentive.job_results["hk"]
    assert [_paid_rate(secure_1), _paid_rate(secure_2)] == pytest.approx([0.8 * 8.0] * 2)
    assert "below 0.8 x the" in _messages(secure_1, FLOOR_APPLIED)[0]
    assert spot.effective_rate == pytest.approx(SPOT_FACTOR * 2.0)
    assert f"min({SPOT_FACTOR:g} * filler_revenue_per_gpu_hour" in _messages(spot, SPOT_LINE)[0]
    assert _messages(at_cap.job_results["hk"][0], FLOOR_NOT_PAID)[0].startswith("Unrented incentive: 0.8 x the")
    _assert_pool_pays_exactly(incentive)


# ── no cap: paid on top of the burn-capped rental share ──────────────────────


def _mixed_cycle() -> list[JobResult]:
    # 8-GPU bucket: three 8x nodes against a cap of 8 (multiplier 1/3); two have a filler average
    return [
        _node("secure-1x", gpu_count=1),
        _node("secure-a", filler_revenue_per_gpu_hour=8.0),
        _node("secure-b", filler_revenue_per_gpu_hour=8.0),
        _node("secure-c"),
        _spot(filler_revenue_per_gpu_hour=2.0),
    ]


def _mixed_config() -> IncentiveConfig:
    return IncentiveConfig(
        rental_incentive_gpu_types=["H100"],
        max_unrented_gpus={"H100": {8: BUCKET_CAP, 1: BUCKET_CAP}},
        rental_prices_per_hour={H100: HOURLY_RATE},
        gpu_count_custom_prices={"*": {"*": DEFAULT_PRICE}},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tao_price", [TAO_PRICE, AT_CAP_TAO_PRICE], ids=["below-cap", "at-cap"])
@pytest.mark.parametrize("spot_rental_incentive", [False, True], ids=["spot-rental-off", "spot-rental-on"])
async def test_non_spot_idle_pay_is_identical_with_or_without_spot_pay_and_the_floor(
    monkeypatch, spot_rental_incentive, tao_price
):
    monkeypatch.setattr(settings, "ENABLE_SPOT_RENTAL_INCENTIVE", spot_rental_incentive)
    off = await _run(_without_filler_averages(_mixed_cycle()), _mixed_config(), tao_price=tao_price)
    on = await _run(_mixed_cycle(), _mixed_config(), tao_price=tao_price)

    assert on.total_rental_cost == off.total_rental_cost
    assert on.rental_share == off.rental_share
    if tao_price == AT_CAP_TAO_PRICE:
        assert on.rental_share_raw > on.total_burn_emission
        assert on.rental_share == on.total_burn_emission
    for before, after in zip(off.job_results["hk"][:4], on.job_results["hk"][:4], strict=True):
        if after.floor_top_up_rate is None:
            assert after.incentive == before.incentive
            # the extras are paid out of the burn remainder; every other input is unchanged
            inputs_after, inputs_before = (
                after.incentive_formula_inputs,
                before.incentive_formula_inputs,
            )
            assert inputs_after.pop("burn_share") == pytest.approx(
                inputs_before.pop("burn_share") - on.unbucketed_share
            )
            assert inputs_after == inputs_before
        else:
            # a floored node keeps its diluted pay; only the top-up is added, from the share on top
            assert after.incentive - _top_up_pay(on, after) == pytest.approx(before.incentive)
    _assert_pool_pays_exactly(on)


@pytest.mark.asyncio
async def test_the_paid_share_rises_by_exactly_spot_pay_and_floor_top_ups(monkeypatch):
    off = await _run(_without_filler_averages(_mixed_cycle()), _mixed_config())
    on = await _run(_mixed_cycle(), _mixed_config())

    spot_pay = 8 * SPOT_FACTOR * 2.0
    floor_top_ups = 2 * 8 * (FLOOR_FACTOR * 8.0 - HOURLY_RATE / 3)
    assert on._unbucketed_rental_cost == pytest.approx(spot_pay + floor_top_ups)
    extra_share = (spot_pay + floor_top_ups) * _share_per_usd_hour(TAO_PRICE)
    assert on.rental_share == off.rental_share
    assert on.unbucketed_share == pytest.approx(extra_share)
    assert on.rental_share + on.unbucketed_share - off.rental_share == pytest.approx(extra_share)
    assert on.burn_share == pytest.approx(off.burn_share - extra_share)
    _assert_pool_pays_exactly(on)


@pytest.mark.asyncio
async def test_extras_past_the_whole_pool_are_clamped_and_only_they_shrink(monkeypatch, caplog):
    def cycle() -> list[JobResult]:
        return [*_over_cap_cycle(8.0), _spot(filler_revenue_per_gpu_hour=2.0)]

    burn_emission = RentalPriceIncentive(_config(), AsyncMock(), {}, {}).total_burn_emission
    tao_price = _half_clamp_tao_price()

    off = await _run(_without_filler_averages(cycle()), tao_price=tao_price)
    with caplog.at_level(logging.WARNING):
        on = await _run(cycle(), tao_price=tao_price)

    clamped = [r for r in caplog.records if "clamped to the incentive pool" in r.message]
    assert len(clamped) == 1
    assert on.rental_share == off.rental_share < burn_emission
    assert on.unbucketed_share == pytest.approx(on.unbucketed_share_raw / 2)
    assert on.mining_share + on.rental_share + on.unbucketed_share == pytest.approx(1.0)
    assert on.burn_share == pytest.approx(0.0, abs=1e-12)
    secure_a, secure_b, spot = on.job_results["hk"]
    per_usd_unclamped = on.unbucketed_share_raw / on._unbucketed_rental_cost
    assert spot.incentive == pytest.approx(per_usd_unclamped * 8 * spot.effective_rate / 2)
    for before, after in zip(off.job_results["hk"][:2], (secure_a, secure_b), strict=True):
        assert after.incentive - _top_up_pay(on, after) == pytest.approx(before.incentive)
        assert _top_up_pay(on, after) == pytest.approx(
            per_usd_unclamped * 8 * after.floor_top_up_rate / 2
        )
    _assert_pool_pays_exactly(on)


@pytest.mark.asyncio
@pytest.mark.parametrize("burn_emission", [None, 0.3], ids=["default-burn", "burn-0.3"])
async def test_at_the_burn_cap_the_pool_has_no_room_left_for_extras(monkeypatch, caplog, burn_emission):
    if burn_emission is not None:
        monkeypatch.setattr("incentive.rental_price.get_total_burn_emission", lambda: burn_emission)
        monkeypatch.setattr("incentive.default.get_total_burn_emission", lambda: burn_emission)

    with caplog.at_level(logging.WARNING):
        on = await _run(_mixed_cycle(), _mixed_config(), tao_price=AT_CAP_TAO_PRICE)

    if burn_emission is not None:
        assert on.total_burn_emission == burn_emission
    # the mining share is 1 - burn emission, so a capped rental share leaves the pool full
    assert on.mining_share + on.total_burn_emission == pytest.approx(1.0)
    assert on.rental_share == on.total_burn_emission
    assert on.unbucketed_share == 0.0
    assert on.burn_share == 0.0
    assert on.unbucketed_paid_fraction == 0.0
    *secure, spot = on.job_results["hk"]
    assert spot.incentive == 0.0
    assert _codes(spot) == [ZeroIncentiveReason.SPOT_NO_HEADROOM_AT_BURN_CAP]
    assert _logged(spot, SPOT_LINE) == []
    floored = [n for n in secure if n.floor_top_up_rate]
    assert len(floored) == 2
    for node in floored:
        # a floored node earns its diluted rate; the top-up has no room and says so
        assert _logged(node, FLOOR_APPLIED) == []
        assert len(_logged(node, FLOOR_NOT_PAID)) == 1
        assert _codes(node) == []
    assert len([r for r in caplog.records if "clamped to the incentive pool" in r.message]) == 1
    _assert_pool_pays_exactly(on)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "zero_rate_spot, own_reason",
    [
        (
            dict(gpu_model="NVIDIA H200"),
            ZeroIncentiveReason.GPU_MODEL_NOT_ELIGIBLE_FOR_UNRENTED_INCENTIVE,
        ),
        (dict(gpu_count=1), ZeroIncentiveReason.NO_UNRENTED_CAPACITY_FOR_GPU_COUNT),
        (dict(nvidia_driver_version="535.104.05"), ZeroIncentiveReason.NVIDIA_DRIVER_BELOW_MINIMUM),
    ],
    ids=["model-outside-the-program", "zero-cap-tier", "low-driver"],
)
async def test_a_zero_rate_spot_node_at_the_cap_keeps_only_its_own_reason(
    zero_rate_spot, own_reason
):
    config = IncentiveConfig(
        rental_incentive_gpu_types=["H100"],
        max_unrented_gpus={"H100": {8: BUCKET_CAP, 1: 0}},
        rental_prices_per_hour={H100: HOURLY_RATE},
        gpu_count_custom_prices={"*": {"*": DEFAULT_PRICE}},
    )
    paid = _spot("spot-paid", filler_revenue_per_gpu_hour=2.0)
    zero_rate = _spot("spot-zero", filler_revenue_per_gpu_hour=2.0, **zero_rate_spot)

    incentive = await _run([_node("secure-1"), paid, zero_rate], config, tao_price=AT_CAP_TAO_PRICE)

    assert incentive.unbucketed_share_raw > 0
    assert incentive.unbucketed_share == 0.0
    assert _codes(paid) == [ZeroIncentiveReason.SPOT_NO_HEADROOM_AT_BURN_CAP]
    assert zero_rate.effective_rate == 0
    assert zero_rate.incentive == 0
    assert _codes(zero_rate) == [own_reason]


@pytest.mark.asyncio
async def test_a_price_outage_reports_no_burn_cap_reason_or_note(caplog):
    with caplog.at_level(logging.WARNING):
        incentive = await _run(
            [*_over_cap_cycle(8.0), _spot(filler_revenue_per_gpu_hour=2.0)], tao_price=None
        )

    # no price: every share is 0, but the rental share is not at the burn cap
    assert incentive.rental_share == 0
    assert incentive.unbucketed_share_raw == incentive.unbucketed_share == 0
    assert incentive._unbucketed_rental_cost > 0
    for node in incentive.job_results["hk"]:
        assert node.incentive == 0
        assert ZeroIncentiveReason.SPOT_NO_HEADROOM_AT_BURN_CAP not in _codes(node)
        assert _logged(node, FLOOR_NOT_PAID) == []
        assert _logged(node, FLOOR_APPLIED) == []
    assert [r for r in caplog.records if "clamped to the incentive pool" in r.message] == []


@pytest.mark.asyncio
async def test_spot_and_floored_nodes_publish_both_terms_of_the_formula():
    incentive = await _run(
        [*_over_cap_cycle(8.0), _node("plain-1x", gpu_count=1), _spot(filler_revenue_per_gpu_hour=2.0)],
        _mixed_config(),
    )
    floored, _, plain, spot = incentive.job_results["hk"]
    unbucketed_cost = 8 * SPOT_FACTOR * 2.0 + 2 * 8 * (FLOOR_FACTOR * 8.0 - HOURLY_RATE * 0.5)

    def rental_term(inputs: dict) -> float:
        return inputs["rental_share"] * inputs["gpu_count"] * inputs["effective_rate"] / inputs["total_rental_cost"]

    def unbucketed_term(inputs: dict) -> float:
        return (
            inputs["unbucketed_share"] * inputs["gpu_count"] * inputs["floor_top_up_rate"]
            / inputs["unbucketed_rental_cost"]
        )

    floored_inputs = floored.incentive_formula_inputs
    assert floored.incentive_formula_version == spot.incentive_formula_version == "rental_price_v2"
    assert floored_inputs["unbucketed_share"] == incentive.unbucketed_share > 0
    assert floored_inputs["unbucketed_rental_cost"] == pytest.approx(unbucketed_cost)
    assert floored_inputs["effective_rate"] == HOURLY_RATE * 0.5
    assert floored_inputs["floor_top_up_rate"] == pytest.approx(FLOOR_FACTOR * 8.0 - HOURLY_RATE * 0.5)
    assert "spot_pay" not in floored_inputs
    assert rental_term(floored_inputs) + unbucketed_term(floored_inputs) == pytest.approx(floored.incentive)

    spot_inputs = spot.incentive_formula_inputs
    assert spot_inputs["spot_pay"] is True
    assert spot_inputs["unbucketed_share"] == incentive.unbucketed_share
    assert spot_inputs["unbucketed_rental_cost"] == pytest.approx(unbucketed_cost)
    assert spot_inputs["floor_top_up_rate"] == spot_inputs["effective_rate"] == pytest.approx(SPOT_FACTOR * 2.0)
    # a spot node is paid the unbucketed term alone
    assert unbucketed_term(spot_inputs) == pytest.approx(spot.incentive)

    for key in ("unbucketed_share", "unbucketed_rental_cost", "floor_top_up_rate", "spot_pay"):
        assert key not in plain.incentive_formula_inputs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tao_price", [TAO_PRICE, "half-clamp", AT_CAP_TAO_PRICE], ids=["below-clamp", "clamped", "at-cap"]
)
async def test_the_logged_formulas_reproduce_the_paid_incentive(tao_price):
    if tao_price == "half-clamp":
        tao_price = _half_clamp_tao_price()
    incentive = await _run([*_over_cap_cycle(8.0), _spot(filler_revenue_per_gpu_hour=2.0)], tao_price=tao_price)

    for node in incentive.job_results["hk"]:
        assert _incentive_from_logged_formulas(node) == pytest.approx(node.incentive, rel=1e-12, abs=1e-15)
        for f in _logged(node, RENTAL_LINE):
            # the rental line is the bucket part alone, at the listed or diluted rate
            assert f["effective_rate"] == node.effective_rate
            assert f["effective_rate"] <= f["hourly_rate"]
        for f in _logged(node, FLOOR_APPLIED):
            assert f["incentive"] == pytest.approx(node.incentive)
            assert f["top_up_incentive"] == pytest.approx(_top_up_pay(incentive, node))
            assert f["paid_fraction"] == pytest.approx(incentive.unbucketed_paid_fraction)
        for f in _logged(node, SPOT_LINE):
            assert f["paid_fraction"] == pytest.approx(incentive.unbucketed_paid_fraction)
    expected_fraction = {TAO_PRICE: 1.0, AT_CAP_TAO_PRICE: 0.0}.get(tao_price, 0.5)
    assert incentive.unbucketed_paid_fraction == pytest.approx(expected_fraction)
    _assert_pool_pays_exactly(incentive)


@pytest.mark.asyncio
@pytest.mark.parametrize("tao_price", [TAO_PRICE, AT_CAP_TAO_PRICE], ids=["below-cap", "at-cap"])
async def test_monitoring_splits_the_rental_pool_by_the_bucket_rate(caplog, tao_price):
    with caplog.at_level(logging.INFO):
        incentive = await _run(_over_cap_cycle(8.0), tao_price=tao_price)

    def logged(prefix: str) -> list[dict]:
        return [r.msg.extra for r in caplog.records if str(r.msg).startswith(prefix)]

    summaries = logged("Unrented_bucket_summary")
    assert [s["bucket_key"] for s in summaries] == ["H100_8"]
    assert summaries[0]["share_of_rental_pool"] == pytest.approx(1.0)
    assert summaries[0]["cost_per_h"] == pytest.approx(incentive.total_rental_cost)
    # the splits add up to the burn emission only with the unbucketed share in them
    assert incentive._unbucketed_rental_cost > 0
    for line in (*logged("Incentive_results"), *logged("Final emission splits calculated")):
        assert line["unbucketed_share"] == incentive.unbucketed_share
        assert line["unbucketed_rental_cost"] == incentive._unbucketed_rental_cost
        assert line["rental_share"] + line["burn_share"] + line["unbucketed_share"] == pytest.approx(
            incentive.total_burn_emission
        )
    assert len(logged("Incentive_results")) == len(logged("Final emission splits calculated")) == 1


@pytest.mark.asyncio
async def test_monitoring_defaults_to_no_unbucketed_share(caplog):
    incentive = await _run(_over_cap_cycle(8.0))

    caplog.clear()
    with caplog.at_level(logging.INFO):
        log_for_monitoring(incentive.job_results, 0.0, incentive.unrented_count_by_bucket)

    (line,) = [r.msg.extra for r in caplog.records if str(r.msg).startswith("Incentive_results")]
    assert (line["unbucketed_share"], line["unbucketed_rental_cost"]) == (0.0, 0.0)


@pytest.mark.asyncio
async def test_an_unpriced_capped_model_gets_no_floor():
    config = IncentiveConfig(
        rental_incentive_gpu_types=["H100", "H200"],
        max_unrented_gpus={"H100": {8: BUCKET_CAP}, "H200": {8: BUCKET_CAP}},
        rental_prices_per_hour={H100: HOURLY_RATE},
        gpu_count_custom_prices={"*": {"*": DEFAULT_PRICE}},
    )
    unpriced = _node("h200-1", gpu_model="NVIDIA H200", filler_revenue_per_gpu_hour=8.0)

    incentive = await _run([*_over_cap_cycle(8.0), unpriced], config)

    assert unpriced.eligible_for_rental_share is True
    assert unpriced.max_cap == BUCKET_CAP
    assert unpriced.hourly_rate == 0
    assert unpriced.floor_top_up_rate is None
    assert unpriced.incentive == 0
    assert incentive._unbucketed_rental_cost == pytest.approx(2 * 8 * (FLOOR_FACTOR * 8.0 - HOURLY_RATE * 0.5))
    _assert_pool_pays_exactly(incentive)


@pytest.mark.asyncio
async def test_snapshot_carries_the_unbucketed_cost():
    incentive = await _run([_spot(filler_revenue_per_gpu_hour=2.0), _node("secure-1")])

    snapshot = incentive.get_snapshot()

    assert snapshot.rental.unbucketed_rental_cost == pytest.approx(8 * SPOT_FACTOR * 2.0)
    assert snapshot.rental.total_rental_cost == pytest.approx(8 * HOURLY_RATE)
    assert snapshot.burn_share == pytest.approx(
        incentive.total_burn_emission - snapshot.rental_share - incentive.unbucketed_share
    )
    seeded = RentalPriceIncentive(_config(), AsyncMock(), {}, {}, snapshot=snapshot)
    assert seeded._unbucketed_rental_cost == snapshot.rental.unbucketed_rental_cost


@pytest.mark.asyncio
async def test_an_estimate_from_the_snapshot_is_not_scaled_by_spot_pay(caplog):
    def estimate_from(incentive: RentalPriceIncentive):
        seeded = RentalPriceIncentive(
            _config(), AsyncMock(), {}, {}, snapshot=incentive.get_snapshot()
        )
        return seeded.estimate_executor(ExecutorEstimateParams(gpu_model=H100, gpu_count=8))

    off = await _run(
        _without_filler_averages([_spot(filler_revenue_per_gpu_hour=2.0), _node("secure-1")]),
        tao_price=AT_CAP_TAO_PRICE,
    )
    on = await _run(
        [_spot(filler_revenue_per_gpu_hour=2.0), _node("secure-1")], tao_price=AT_CAP_TAO_PRICE
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        estimate_on = await estimate_from(on)
    estimate_off = await estimate_from(off)

    assert estimate_on.incentive == estimate_off.incentive
    assert estimate_on.total_rental_cost == estimate_off.total_rental_cost
    # the live cycle logged the clamp once; an estimate seeded from it does not repeat it
    assert [r for r in caplog.records if "clamped to the incentive pool" in r.message] == []


# ── the backend's per-configuration average ──────────────────────────────────


def _rented_data(*entries: FillerRevenueByGpuConfig, complete: bool = True) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={}, filler_revenue_by_gpu_config=list(entries), filler_revenue_complete=complete
    )


def test_average_is_read_for_the_exact_configuration():
    data = _rented_data(
        FillerRevenueByGpuConfig(
            base_model="H100", gpu_count=8, usd_per_gpu_hour=2.0, gpu_hours=100
        ),
        FillerRevenueByGpuConfig(
            base_model="H100", gpu_count=1, usd_per_gpu_hour=3.0, gpu_hours=100
        ),
    )

    assert data.get_filler_revenue_per_gpu_hour("H100", 8, 24) == 2.0
    assert data.get_filler_revenue_per_gpu_hour("H100", 1, 24) == 3.0
    assert data.get_filler_revenue_per_gpu_hour("H100", 4, 24) is None
    assert data.get_filler_revenue_per_gpu_hour("H200", 8, 24) is None
    assert data.get_filler_revenue_per_gpu_hour(None, 8, 24) is None


def test_thin_sample_reads_as_no_average():
    data = _rented_data(
        FillerRevenueByGpuConfig(base_model="H100", gpu_count=8, usd_per_gpu_hour=2.0, gpu_hours=24)
    )

    assert data.get_filler_revenue_per_gpu_hour("H100", 8, 24) == 2.0
    assert data.get_filler_revenue_per_gpu_hour("H100", 8, 24.5) is None


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_unusable_average_reads_as_no_average(value):
    data = _rented_data(
        FillerRevenueByGpuConfig(
            base_model="H100", gpu_count=8, usd_per_gpu_hour=value, gpu_hours=100
        )
    )

    assert data.get_filler_revenue_per_gpu_hour("H100", 8, 24) is None


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        ([], False),  # an older backend's default, or nothing sent
        ([FillerRevenueByGpuConfig(base_model="H100", gpu_count=8, usd_per_gpu_hour=1.0, gpu_hours=10)], False),
        ([FillerRevenueByGpuConfig(base_model="H100", gpu_count=8, usd_per_gpu_hour=math.nan, gpu_hours=100)], False),
        ([FillerRevenueByGpuConfig(base_model="H100", gpu_count=8, usd_per_gpu_hour=0.0, gpu_hours=100)], True),
    ],
)
def test_the_feed_is_live_only_with_a_usable_average(entries, expected):
    data = _rented_data(*entries)

    assert data.has_filler_revenue_averages(24) is expected


def test_a_partial_or_stale_feed_seeds_nobody():
    # usable averages, but the backend did not mark the feed fresh and complete (a stale rate, the TSDB unread, an
    # older backend): a configuration missing from it may be a gap, not unmeasured
    data = _rented_data(
        FillerRevenueByGpuConfig(base_model="H100", gpu_count=8, usd_per_gpu_hour=1.0, gpu_hours=100), complete=False
    )

    assert data.has_filler_revenue_averages(24) is False
    assert RentedExecutorsResponse(executors={}).filler_revenue_complete is False


def test_a_measured_zero_average_is_zero_not_unmeasured():
    data = _rented_data(
        FillerRevenueByGpuConfig(base_model="H100", gpu_count=8, usd_per_gpu_hour=0.0, gpu_hours=100)
    )

    assert data.get_filler_revenue_per_gpu_hour("H100", 8, 24) == 0.0


async def _handled(
    context_factory, rented_data: RentedExecutorsResponse | None, **context_overrides
) -> JobResult:
    state = build_state(gpu_model_count=f"{H100}:8", rented_data=rented_data, sysbox_runtime=True)
    ctx = context_factory(
        state=state,
        tdx_attestation_passed=False,
        score=1.0,
        job_score=1.0,
        collateral_deposited=True,
        ssh_pub_keys=[],
        rented=False,
        **context_overrides,
    )
    return await ResultHandler(redis_service=None, dry_run=True).handle_result(
        context=ctx,
        miner_info=SimpleNamespace(miner_hotkey="miner-hotkey", job_batch_id="batch-1"),
        executor_info=ctx.executor,
        verified_job_info={},
        log_text="ok",
        success=True,
    )


@pytest.mark.asyncio
async def test_result_reads_the_nodes_filler_and_its_config_average(context_factory):
    executor_id = str(default_executor().uuid)
    data = RentedExecutorsResponse(
        executors={},
        all_filler_containers_by_executor={executor_id: ["filler_run-1"]},
        filler_revenue_by_gpu_config=[
            FillerRevenueByGpuConfig(
                base_model="H100", gpu_count=1, usd_per_gpu_hour=3.0, gpu_hours=100
            ),
            FillerRevenueByGpuConfig(
                base_model="H100", gpu_count=8, usd_per_gpu_hour=2.0, gpu_hours=100
            ),
        ],
    )

    result = await _handled(context_factory, data)

    assert result.has_lium_filler is True
    assert result.filler_revenue_per_gpu_hour == 2.0


@pytest.mark.asyncio
async def test_result_without_a_filler_or_backend_data(context_factory):
    no_filler = await _handled(context_factory, RentedExecutorsResponse(executors={}))
    no_data = await _handled(context_factory, None)

    for result in (no_filler, no_data):
        assert result.has_lium_filler is False
        assert result.filler_revenue_per_gpu_hour is None


def _spot_reply(**fields) -> RentedExecutorsResponse:
    # a spot node with a filler and an average: everything spot pay needs but the provider's choice
    executor_id = str(default_executor().uuid)
    return RentedExecutorsResponse.model_validate(
        {
            "executors": {},
            "spot_executor_ids": [executor_id],
            "all_filler_containers_by_executor": {executor_id: ["filler_run-1"]},
            "filler_revenue_by_gpu_config": [_GOOD_ENTRY],
            **fields,
        }
    )


_EXECUTOR_ID = str(default_executor().uuid)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields, context, expected_codes",
    [
        # demoted, force-spot, pinned and no-incentive-rental nodes all arrive this way: in the spot
        # list, not in the provider's
        ({"provider_spot_executor_ids": []}, {}, [ZeroIncentiveReason.SPOT_TIER]),
        ({"provider_spot_executor_ids": ["another-executor"]}, {}, [ZeroIncentiveReason.SPOT_TIER]),
        # an older backend that does not send the field
        ({}, {}, [ZeroIncentiveReason.SPOT_TIER]),
        # a banned provider is out even if the backend listed the node for spot pay
        (
            {"provider_spot_executor_ids": [_EXECUTOR_ID], "banned_hotkeys": ["miner-hotkey"]},
            {"is_provider_banned": True},
            [ZeroIncentiveReason.BANNED_NETWORK_ABUSE],
        ),
        (
            {"banned_hotkeys": ["miner-hotkey"]},
            {"is_provider_banned": True},
            [ZeroIncentiveReason.BANNED_NETWORK_ABUSE, ZeroIncentiveReason.SPOT_TIER],
        ),
    ],
    ids=["off-the-pay-list", "listed-for-another-node", "field-absent", "banned-listed", "banned"],
)
async def test_only_a_spot_node_on_the_pay_list_is_paid(
    context_factory, fields, context, expected_codes
):
    result = await _handled(context_factory, _spot_reply(**fields), **context)

    assert result.is_spot is True
    assert result.has_lium_filler is True
    assert result.filler_revenue_per_gpu_hour == 2.0
    await _run([result, _node("secure-1")])
    assert result.incentive == 0
    assert result.spot_pay_candidate is False
    assert _codes(result) == expected_codes


@pytest.mark.asyncio
async def test_a_spot_node_on_the_pay_list_is_paid(context_factory):
    result = await _handled(
        context_factory, _spot_reply(provider_spot_executor_ids=[_EXECUTOR_ID])
    )

    assert result.is_on_spot_pay_list is True
    await _run([result, _node("secure-1")])
    assert result.spot_pay_candidate is True
    # the model's fillers earned 2.0 x 100 GPU-hours over 24 h: less than 8 GPUs at 2.0 an hour
    assert result.filler_revenue_usd_per_hour == pytest.approx(2.0 * 100 / 24)
    assert result.effective_rate * result.gpu_count == pytest.approx(2.0 * 100 / 24)
    assert result.incentive > 0
    assert _codes(result) == []


@pytest.mark.asyncio
async def test_the_pay_list_alone_does_not_make_a_node_spot(context_factory):
    data = RentedExecutorsResponse(executors={}, provider_spot_executor_ids=[_EXECUTOR_ID])

    result = await _handled(context_factory, data)

    assert result.is_spot is False
    assert result.is_on_spot_pay_list is False


@pytest.mark.parametrize(
    "value, expected, warns",
    [
        (None, [], False),
        ("spot-1", [], True),
        ({"spot-1": True}, [], True),
        (["spot-1", 7, None, "spot-2"], ["spot-1", "spot-2"], False),
    ],
    ids=["null", "string", "dict", "mixed-items"],
)
def test_a_malformed_provider_spot_list_reads_as_empty(value, expected, warns, caplog):
    with caplog.at_level(logging.WARNING):
        data = RentedExecutorsResponse.model_validate(
            {**_reply([_GOOD_ENTRY]), "provider_spot_executor_ids": value}
        )

    assert data.provider_spot_executor_ids == expected
    assert data.spot_executor_ids == ["spot-1"]
    assert data.banned_hotkeys == ["banned-hk"]
    assert any("provider_spot_executor_ids is not a list" in r.message for r in caplog.records) is warns


_GOOD_ENTRY: dict = {
    "base_model": "H100",
    "gpu_count": 8,
    "usd_per_gpu_hour": 2.0,
    "gpu_hours": 100,
}


def _reply(filler_revenue_by_gpu_config) -> dict:
    return {
        "executors": {},
        "banned_hotkeys": ["banned-hk"],
        "spot_executor_ids": ["spot-1"],
        "filler_revenue_by_gpu_config": filler_revenue_by_gpu_config,
    }


def test_null_averages_read_as_none_without_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        data = RentedExecutorsResponse.model_validate(_reply(None))

    assert data.filler_revenue_by_gpu_config == []
    assert data.banned_hotkeys == ["banned-hk"]
    assert caplog.records == []


def test_invalid_average_entries_are_dropped_with_a_warning(caplog):
    malformed = [
        {"base_model": "H100", "gpu_count": 1},
        {**_GOOD_ENTRY, "usd_per_gpu_hour": "lots"},
        "8x H100",
        None,
        _GOOD_ENTRY,
    ]

    with caplog.at_level(logging.WARNING):
        data = RentedExecutorsResponse.model_validate(_reply(malformed))

    assert data.filler_revenue_by_gpu_config == [FillerRevenueByGpuConfig(**_GOOD_ENTRY)]
    assert data.get_filler_revenue_per_gpu_hour("H100", 8, 24) == 2.0
    assert data.spot_executor_ids == ["spot-1"]
    dropped = [
        r for r in caplog.records if "dropped an invalid filler_revenue_by_gpu_config" in r.message
    ]
    assert len(dropped) == 4


@pytest.mark.parametrize("value", [{"H100": 2.0}, "2.0", 2.0])
def test_a_field_that_is_not_a_list_reads_as_no_averages(value, caplog):
    with caplog.at_level(logging.WARNING):
        data = RentedExecutorsResponse.model_validate(_reply(value))

    assert data.filler_revenue_by_gpu_config == []
    assert data.banned_hotkeys == ["banned-hk"]
    assert any("is not a list" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_a_malformed_field_scores_like_an_absent_one(context_factory):
    executor_id = str(default_executor().uuid)
    fillers = {"all_filler_containers_by_executor": {executor_id: ["filler_run-1"]}}
    malformed = RentedExecutorsResponse.model_validate(
        {**_reply([{"gpu_count": "eight"}, 7]), **fillers}
    )
    absent = RentedExecutorsResponse.model_validate({"executors": {}, **fillers})

    handled = [await _handled(context_factory, data) for data in (malformed, absent)]

    assert [r.has_lium_filler for r in handled] == [True, True]
    assert [r.filler_revenue_per_gpu_hour for r in handled] == [None, None]
    new, old = [await _run([result]) for result in handled]
    after, before = new.job_results["hk"][0], old.job_results["hk"][0]
    assert after.incentive == before.incentive
    assert after.effective_rate == before.effective_rate
    assert after.incentive_logs == before.incentive_logs


def test_older_backend_sends_no_averages():
    data = RentedExecutorsResponse(executors={})

    assert data.filler_revenue_by_gpu_config == []
    assert data.get_filler_revenue_per_gpu_hour("H100", 8, 24) is None
    assert data.provider_spot_executor_ids == []


def test_a_lone_one_gpu_node_has_an_average_after_two_filler_hours():
    # a lone 1x node earns from its first 2 Default Job hours, not after a full day
    data = _rented_data(
        FillerRevenueByGpuConfig(
            base_model="RTX 3090", gpu_count=1, usd_per_gpu_hour=0.08, gpu_hours=2.0
        )
    )

    floor = settings.FILLER_REVENUE_MIN_GPU_HOURS
    assert floor == 2.0
    assert data.get_filler_revenue_per_gpu_hour("RTX 3090", 1, floor) == 0.08


@pytest.mark.asyncio
@pytest.mark.parametrize("gpu_hours", [2.0, 23.9])
async def test_secure_keeps_the_24_gpu_hour_threshold_while_spot_uses_2(
    context_factory, gpu_hours
):
    # a sample of 2 to under 24 GPU-hours pays Spot, but gives a secure node no floor average
    sample = {"filler_revenue_by_gpu_config": [{**_GOOD_ENTRY, "gpu_hours": gpu_hours}]}

    secure = await _handled(context_factory, _spot_reply(spot_executor_ids=[], **sample))
    spot = await _handled(context_factory, _spot_reply(**sample))

    assert secure.is_spot is False
    assert secure.filler_revenue_per_gpu_hour is None
    # diluted by a second 8x node in its bucket, it still gets no floor top-up
    await _run([secure, _node("secure-2")])
    assert secure.effective_rate == HOURLY_RATE * 0.5
    assert secure.floor_top_up_rate is None
    assert spot.is_spot is True
    assert spot.filler_revenue_per_gpu_hour == 2.0


@pytest.mark.asyncio
async def test_secure_uses_an_average_taken_over_24_gpu_hours(context_factory):
    sample = {"filler_revenue_by_gpu_config": [{**_GOOD_ENTRY, "gpu_hours": 24.0}]}

    secure = await _handled(context_factory, _spot_reply(spot_executor_ids=[], **sample))

    assert secure.filler_revenue_per_gpu_hour == 2.0

