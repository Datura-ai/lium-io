"""Spot-node pay (ENABLE_SPOT_NODE_PAY) and the secure floor (ENABLE_SECURE_FILLER_REVENUE_FLOOR).

Spot: an idle spot node running Lium fillers is paid min(0.9 x its GPU configuration's average
filler revenue per GPU-hour, its secure rate before cap dilution), outside the buckets.
Secure floor: an idle secure node's cap-diluted rate is raised to min(0.9 x that average, its
undiluted rate). Both flags off reproduce the old numbers exactly.
"""

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from core.config import settings
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.config import DEFAULT_PRICE, IncentiveConfig
from incentive.miner_incentive_log import ZeroIncentiveReason
from incentive.rental_price import RentalPriceIncentive
from protocol.vc_protocol.compute_requests import FillerRevenueByGpuConfig, RentedExecutorsResponse
from services.task.result_handler import ResultHandler
from services.task_service import JobResult

from tests.helpers import build_state, default_executor

H100 = "NVIDIA H100 80GB HBM3"
HOURLY_RATE = 10.0
BUCKET_CAP = 8
# large enough that the rental share stays under the burn-emission cap in every test here
TAO_PRICE = 1_000_000.0
ALPHA_RATE = 1.0


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
            address="10.0.0.1",
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


def _spot(uuid: str = "spot-1", **overrides) -> JobResult:
    return _node(uuid, **{"is_spot": True, "has_lium_filler": True, **overrides})


async def _run(
    results: list[JobResult], config: IncentiveConfig | None = None
) -> RentalPriceIncentive:
    incentive = RentalPriceIncentive(config or _config(), AsyncMock(), {"hk": results}, {})
    incentive.price_provider = AsyncMock()
    incentive.price_provider.get_tao_price.return_value = TAO_PRICE
    incentive.price_provider.get_alpha_rate.return_value = ALPHA_RATE
    await incentive.calculate_mining_scores()
    return incentive


def _codes(result: JobResult) -> list[str]:
    return [reason.reason for reason in result.zero_incentive_reasons]


def _paid_cost(incentive: RentalPriceIncentive) -> float:
    return sum(r.gpu_count * (r.effective_rate or 0.0) for r in incentive.job_results["hk"])


@pytest.fixture
def spot_pay_on(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_SPOT_NODE_PAY", True)


@pytest.fixture
def floor_on(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_SECURE_FILLER_REVENUE_FLOOR", True)


@pytest.fixture(autouse=True)
def flags_default_off(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_SPOT_NODE_PAY", False)
    monkeypatch.setattr(settings, "ENABLE_SECURE_FILLER_REVENUE_FLOOR", False)


# ── spot-node pay ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_spot_is_paid_the_filler_side_when_it_is_lower(spot_pay_on):
    spot = _spot(filler_revenue_per_gpu_hour=2.0)
    secure = _node("secure-1")

    incentive = await _run([spot, secure])

    assert spot.effective_rate == pytest.approx(0.9 * 2.0)
    assert spot.incentive > 0
    assert spot.incentive / secure.incentive == pytest.approx(0.9 * 2.0 / HOURLY_RATE)
    assert spot.incentive_formula_version == "rental_price_v2"
    assert spot.mining_score == 0
    assert _codes(spot) == []
    assert incentive.total_rental_cost == pytest.approx(_paid_cost(incentive))


@pytest.mark.asyncio
async def test_spot_is_paid_the_secure_side_when_it_is_lower(spot_pay_on):
    spot = _spot(filler_revenue_per_gpu_hour=4 * HOURLY_RATE)
    secure = _node("secure-1")

    await _run([spot, secure])

    assert spot.effective_rate == pytest.approx(HOURLY_RATE)
    assert spot.incentive == pytest.approx(secure.incentive)


@pytest.mark.asyncio
async def test_spot_secure_side_carries_the_nodes_own_sysbox_and_driver(spot_pay_on, monkeypatch):
    monkeypatch.setattr(settings, "PORTION_FOR_SYSBOX_UNRENTED", 0.5)
    spot = _spot(filler_revenue_per_gpu_hour=4 * HOURLY_RATE, sysbox_runtime=False)

    await _run([spot])

    assert spot.effective_rate == pytest.approx(0.5 * HOURLY_RATE)


@pytest.mark.asyncio
async def test_spot_without_a_lium_filler_gets_zero(spot_pay_on):
    spot = _spot(has_lium_filler=False, filler_revenue_per_gpu_hour=2.0)

    await _run([spot])

    assert spot.incentive == 0
    assert _codes(spot) == [ZeroIncentiveReason.SPOT_WITHOUT_LIUM_FILLER]


@pytest.mark.asyncio
async def test_spot_whose_config_has_no_average_gets_zero(spot_pay_on):
    spot = _spot(filler_revenue_per_gpu_hour=None)

    await _run([spot])

    assert spot.incentive == 0
    assert _codes(spot) == [ZeroIncentiveReason.SPOT_NO_FILLER_REVENUE_FOR_GPU_CONFIG]


@pytest.mark.asyncio
async def test_spot_has_no_cap_and_does_not_dilute_secure_nodes(spot_pay_on):
    spots = [_spot(f"spot-{i}", filler_revenue_per_gpu_hour=2.0) for i in range(4)]
    secure = _node("secure-1")

    incentive = await _run([*spots, secure])

    # one 8x secure node fills the 8-GPU cap exactly; four 8x spot nodes sit outside it
    assert secure.unrented_cap_multiplier == 1.0
    assert secure.effective_rate == HOURLY_RATE
    for spot in spots:
        assert spot.effective_rate == pytest.approx(0.9 * 2.0)
    assert incentive.unrented_count_by_bucket[("H100", 8)] == 8


@pytest.mark.asyncio
async def test_spot_in_a_config_with_no_secure_idle_pay_gets_zero(spot_pay_on):
    spot = _spot(filler_revenue_per_gpu_hour=2.0)

    await _run([spot], _config(cap=0))

    assert spot.incentive == 0
    assert ZeroIncentiveReason.NO_UNRENTED_CAPACITY_FOR_GPU_COUNT in _codes(spot)


@pytest.mark.asyncio
async def test_rented_spot_stays_out_of_both_pools(spot_pay_on):
    spot = _spot(is_rented=True, filler_revenue_per_gpu_hour=2.0)

    await _run([spot])

    assert spot.incentive == 0
    assert _codes(spot) == [ZeroIncentiveReason.SPOT_TIER]


@pytest.mark.asyncio
async def test_spot_blocked_by_another_exclusion_is_not_paid(spot_pay_on):
    spot = _spot(is_provider_banned=True, filler_revenue_per_gpu_hour=2.0)

    await _run([spot])

    assert spot.incentive == 0
    assert spot.spot_pay_candidate is False
    assert _codes(spot) == [ZeroIncentiveReason.BANNED_NETWORK_ABUSE]


@pytest.mark.asyncio
async def test_spot_flag_off_keeps_spot_at_zero():
    spot = _spot(filler_revenue_per_gpu_hour=2.0)

    await _run([spot])

    assert spot.incentive == 0
    assert _codes(spot) == [ZeroIncentiveReason.SPOT_TIER]


@pytest.mark.asyncio
async def test_spot_flag_leaves_secure_nodes_unchanged(monkeypatch):
    def cycle() -> list[JobResult]:
        return [_spot(filler_revenue_per_gpu_hour=2.0), _node("secure-1"), _node("secure-2")]

    off = await _run(cycle())
    monkeypatch.setattr(settings, "ENABLE_SPOT_NODE_PAY", True)
    on = await _run(cycle())

    for before, after in zip(off.job_results["hk"][1:], on.job_results["hk"][1:], strict=True):
        assert after.effective_rate == before.effective_rate
        assert after.unrented_cap_multiplier == before.unrented_cap_multiplier
        # below the burn cap the rental share grows with the pool, so a secure node's pay holds
        assert after.incentive == pytest.approx(before.incentive)


# ── secure floor ─────────────────────────────────────────────────────────────


def _over_cap_cycle(filler_revenue: float | None) -> list[JobResult]:
    # two 8x nodes in an 8-GPU bucket: cap multiplier 0.5, diluted rate HOURLY_RATE / 2
    return [
        _node("secure-1", filler_revenue_per_gpu_hour=filler_revenue),
        _node("secure-2", filler_revenue_per_gpu_hour=filler_revenue),
    ]


@pytest.mark.asyncio
async def test_floor_off_keeps_the_diluted_rate():
    incentive = await _run(_over_cap_cycle(8.0))

    for node in incentive.job_results["hk"]:
        assert node.effective_rate == HOURLY_RATE * 0.5


@pytest.mark.asyncio
async def test_floor_lifts_a_diluted_rate_after_dilution(floor_on):
    incentive = await _run(_over_cap_cycle(8.0))

    for node in incentive.job_results["hk"]:
        assert node.unrented_cap_multiplier == 0.5
        assert node.effective_rate == pytest.approx(0.9 * 8.0)
    assert incentive.total_rental_cost == pytest.approx(_paid_cost(incentive))


@pytest.mark.asyncio
async def test_floor_never_lifts_above_the_undiluted_rate(floor_on):
    incentive = await _run(_over_cap_cycle(4 * HOURLY_RATE))

    for node in incentive.job_results["hk"]:
        assert node.effective_rate == pytest.approx(HOURLY_RATE)


@pytest.mark.asyncio
async def test_floor_below_the_diluted_rate_changes_nothing(floor_on):
    incentive = await _run(_over_cap_cycle(2.0))

    for node in incentive.job_results["hk"]:
        assert node.effective_rate == HOURLY_RATE * 0.5
    assert incentive.total_rental_cost == pytest.approx(_paid_cost(incentive))


@pytest.mark.asyncio
async def test_floor_without_an_average_changes_nothing(floor_on):
    incentive = await _run(_over_cap_cycle(None))

    for node in incentive.job_results["hk"]:
        assert node.effective_rate == HOURLY_RATE * 0.5


@pytest.mark.asyncio
async def test_floor_keeps_the_nodes_own_penalties(floor_on, monkeypatch):
    monkeypatch.setattr(settings, "PORTION_FOR_SYSBOX_UNRENTED", 0.5)
    incentive = await _run(
        [
            _node("secure-1", filler_revenue_per_gpu_hour=8.0, sysbox_runtime=False),
            _node("secure-2", filler_revenue_per_gpu_hour=8.0),
        ]
    )

    penalised, full = incentive.job_results["hk"]
    assert penalised.effective_rate == pytest.approx(0.9 * 8.0 * 0.5)
    assert full.effective_rate == pytest.approx(0.9 * 8.0)
    assert incentive.total_rental_cost == pytest.approx(_paid_cost(incentive))


@pytest.mark.asyncio
async def test_floor_skips_the_free_remainder_of_a_partially_rented_split_node(
    floor_on, monkeypatch
):
    monkeypatch.setattr(settings, "ENABLE_SPLIT_PARTIAL_RENTAL_SCORING", True)
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

    await _run([split], config)

    free = split.incentive_formula_inputs["unrented"]
    assert free["unrented_cap_multiplier"] == 0.25
    assert free["effective_rate"] == HOURLY_RATE * 0.25


@pytest.mark.asyncio
async def test_the_two_flags_are_independent(monkeypatch):
    def cycle() -> list[JobResult]:
        return [*_over_cap_cycle(8.0), _spot(filler_revenue_per_gpu_hour=2.0)]

    monkeypatch.setattr(settings, "ENABLE_SPOT_NODE_PAY", True)
    spot_only = await _run(cycle())
    assert [n.effective_rate for n in spot_only.job_results["hk"]] == pytest.approx(
        [HOURLY_RATE * 0.5, HOURLY_RATE * 0.5, 0.9 * 2.0]
    )

    monkeypatch.setattr(settings, "ENABLE_SPOT_NODE_PAY", False)
    monkeypatch.setattr(settings, "ENABLE_SECURE_FILLER_REVENUE_FLOOR", True)
    floor_only = await _run(cycle())
    secure_1, secure_2, spot = floor_only.job_results["hk"]
    assert [secure_1.effective_rate, secure_2.effective_rate] == pytest.approx(
        [0.9 * 8.0, 0.9 * 8.0]
    )
    assert spot.incentive == 0
    assert _codes(spot) == [ZeroIncentiveReason.SPOT_TIER]

    monkeypatch.setattr(settings, "ENABLE_SPOT_NODE_PAY", True)
    both = await _run(cycle())
    assert [n.effective_rate for n in both.job_results["hk"]] == pytest.approx(
        [0.9 * 8.0, 0.9 * 8.0, 0.9 * 2.0]
    )
    assert both.total_rental_cost == pytest.approx(_paid_cost(both))


# ── flags off: the old behaviour exactly ─────────────────────────────────────


@pytest.mark.asyncio
async def test_flags_off_ignore_the_new_data_exactly():
    def cycle(with_new_data: bool) -> list[JobResult]:
        extra: dict = (
            {"has_lium_filler": True, "filler_revenue_per_gpu_hour": 8.0} if with_new_data else {}
        )
        return [
            _node("secure-1", **extra),
            _node("secure-2", **extra),
            _node("secure-3", gpu_count=1, **extra),
            _node("spot-1", is_spot=True, **extra),
        ]

    old = await _run(cycle(with_new_data=False))
    new = await _run(cycle(with_new_data=True))

    assert new.total_rental_cost == old.total_rental_cost
    assert new.rental_share == old.rental_share
    for before, after in zip(old.job_results["hk"], new.job_results["hk"], strict=True):
        assert after.incentive == before.incentive
        assert after.effective_rate == before.effective_rate
        assert after.incentive_logs == before.incentive_logs
        assert _codes(after) == _codes(before)


@pytest.mark.asyncio
async def test_snapshot_carries_the_unbucketed_cost(spot_pay_on):
    incentive = await _run([_spot(filler_revenue_per_gpu_hour=2.0), _node("secure-1")])

    snapshot = incentive.get_snapshot()

    assert snapshot.rental.unbucketed_rental_cost == pytest.approx(8 * 0.9 * 2.0)
    seeded = RentalPriceIncentive(_config(), AsyncMock(), {}, {}, snapshot=snapshot)
    assert seeded._unbucketed_rental_cost == snapshot.rental.unbucketed_rental_cost


# ── the backend's per-configuration average ──────────────────────────────────


def _rented_data(*entries: FillerRevenueByGpuConfig) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(executors={}, filler_revenue_by_gpu_config=list(entries))


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


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_unusable_average_reads_as_no_average(value):
    data = _rented_data(
        FillerRevenueByGpuConfig(
            base_model="H100", gpu_count=8, usd_per_gpu_hour=value, gpu_hours=100
        )
    )

    assert data.get_filler_revenue_per_gpu_hour("H100", 8, 24) is None


async def _handled(context_factory, rented_data: RentedExecutorsResponse | None) -> JobResult:
    state = build_state(gpu_model_count=f"{H100}:8", rented_data=rented_data)
    ctx = context_factory(
        state=state,
        tdx_attestation_passed=False,
        score=1.0,
        job_score=1.0,
        collateral_deposited=True,
        ssh_pub_keys=[],
        rented=False,
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


def test_older_backend_sends_no_averages():
    data = RentedExecutorsResponse(executors={})

    assert data.filler_revenue_by_gpu_config == []
    assert data.get_filler_revenue_per_gpu_hour("H100", 8, 24) is None
