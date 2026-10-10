"""Spot-node pay (ENABLE_SPOT_NODE_PAY) and the secure floor (ENABLE_SECURE_FILLER_REVENUE_FLOOR).

Spot: an idle spot node running Lium fillers is paid min(0.95 x its GPU configuration's average
filler revenue per GPU-hour, its secure rate before cap dilution), outside the buckets.
Secure floor: an idle secure node's cap-diluted rate is raised to 0.95 x that average, even
above its listed rate. The two factors are separate constants. No cap (owner rule): spot pay and floor top-ups are paid on top of the
burn-capped rental share, up to the whole incentive pool. Both flags off reproduce the old
numbers exactly.
"""

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from core.config import settings
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.config import DEFAULT_PRICE, IncentiveConfig
from incentive.miner_incentive_log import ZeroIncentiveReason
from incentive.rental_price import RentalPriceIncentive
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
SPOT_FACTOR = 0.95
FLOOR_FACTOR = 0.95


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


def _spot(uuid: str = "spot-1", **overrides) -> JobResult:
    return _node(
        uuid, **{"is_spot": True, "is_provider_chosen_spot": True, "has_lium_filler": True, **overrides}
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


FLOOR_APPLIED = "secure_filler_revenue_floor_applied"


def _logged(result: JobResult, marker: str) -> list[dict]:
    """The fields of every miner-facing line whose message starts with, or whose event is, `marker`."""
    lines: list[dict] = []
    for line in result.incentive_logs:
        message, _, fields = line.partition(" >>> ")
        parsed: dict = json.loads(fields)
        if message.startswith(marker) or parsed.get("event") == marker:
            lines.append(parsed)
    return lines


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

    assert spot.effective_rate == pytest.approx(SPOT_FACTOR * 2.0)
    assert spot.incentive > 0
    assert spot.incentive / secure.incentive == pytest.approx(SPOT_FACTOR * 2.0 / HOURLY_RATE)
    assert spot.incentive_formula_version == "rental_price_v2"
    assert spot.mining_score == 0
    assert _codes(spot) == []
    _assert_pool_pays_exactly(incentive)


@pytest.mark.asyncio
async def test_spot_is_paid_the_secure_side_when_it_is_lower(spot_pay_on):
    spot = _spot(filler_revenue_per_gpu_hour=4 * HOURLY_RATE)
    secure = _node("secure-1")

    await _run([spot, secure])

    assert spot.effective_rate == pytest.approx(HOURLY_RATE)
    assert spot.incentive == pytest.approx(secure.incentive)


@pytest.mark.asyncio
async def test_rented_spot_stays_out_of_both_pools(spot_pay_on):
    spot = _spot(is_rented=True, filler_revenue_per_gpu_hour=2.0)

    await _run([spot])

    assert spot.incentive == 0
    assert _codes(spot) == [ZeroIncentiveReason.SPOT_TIER]


@pytest.mark.asyncio
async def test_spot_flag_off_keeps_spot_at_zero():
    spot = _spot(filler_revenue_per_gpu_hour=2.0)

    await _run([spot])

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
async def test_floor_lifts_a_diluted_rate_after_dilution(floor_on):
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
async def test_floor_keeps_the_nodes_own_penalties(floor_on, monkeypatch):
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
@pytest.mark.parametrize("flag", ["ENABLE_SPOT_NODE_PAY", "ENABLE_SECURE_FILLER_REVENUE_FLOOR"])
async def test_non_spot_idle_pay_is_identical_with_either_flag_on_or_off(
    monkeypatch, flag, tao_price
):
    off = await _run(_mixed_cycle(), _mixed_config(), tao_price=tao_price)
    monkeypatch.setattr(settings, flag, True)
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
async def test_extras_past_the_whole_pool_are_clamped_and_only_they_shrink(monkeypatch, caplog):
    def cycle() -> list[JobResult]:
        return [*_over_cap_cycle(8.0), _spot(filler_revenue_per_gpu_hour=2.0)]

    burn_emission = RentalPriceIncentive(_config(), AsyncMock(), {}, {}).total_burn_emission
    tao_price = _half_clamp_tao_price()

    off = await _run(cycle(), tao_price=tao_price)
    monkeypatch.setattr(settings, "ENABLE_SPOT_NODE_PAY", True)
    monkeypatch.setattr(settings, "ENABLE_SECURE_FILLER_REVENUE_FLOOR", True)
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


# ── flags off: the old behaviour exactly ─────────────────────────────────────


# ── the backend's per-configuration average ──────────────────────────────────


def _rented_data(*entries: FillerRevenueByGpuConfig) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(executors={}, filler_revenue_by_gpu_config=list(entries))


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
        # a banned provider is out even if the backend listed the node as provider-chosen
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
    ids=["demoted-or-pinned", "listed-for-another-node", "field-absent", "banned-listed", "banned"],
)
async def test_only_a_provider_chosen_spot_node_is_paid(
    context_factory, spot_pay_on, fields, context, expected_codes
):
    result = await _handled(context_factory, _spot_reply(**fields), **context)

    assert result.is_spot is True
    assert result.has_lium_filler is True
    assert result.filler_revenue_per_gpu_hour == 2.0
    await _run([result, _node("secure-1")])
    assert result.incentive == 0
    assert result.spot_pay_candidate is False
    assert _codes(result) == expected_codes


_GOOD_ENTRY: dict = {
    "base_model": "H100",
    "gpu_count": 8,
    "usd_per_gpu_hour": 2.0,
    "gpu_hours": 100,
}


