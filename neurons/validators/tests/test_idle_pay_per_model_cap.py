"""DAH-6030 per-model cap: an idle GPU-hour is rated at most k x what its GPU model's Default Jobs earned per idle
GPU-hour, so a model's idle pay (listed rate, secure floor and spot together) stays within k x its Default Job revenue."""

from unittest.mock import AsyncMock

import pytest
from core.config import settings
from incentive.rental_price import SPOT_IDLE_SECURE_RATE_SHARE, ExecutorEstimateParams, RentalPriceIncentive

from tests.test_spot_node_pay import H100, HOURLY_RATE, _config, _node, _paid_rate, _run, _spot


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_SPOT_RENTAL_INCENTIVE", False)
    monkeypatch.setattr(settings, "SPOT_IDLE_PAY_SECURE_DECISION_ENABLED", False)


def _secure(uuid: str, model_usd_per_hour: float | None, per_filled_gpu_hour: float = 5.0):
    return _node(uuid, filler_revenue_usd_per_hour=model_usd_per_hour, filler_revenue_per_gpu_hour=per_filled_gpu_hour)


def _model_paid(incentive) -> float:
    return sum(r.gpu_count * _paid_rate(r) for r in incentive.job_results["hk"])


@pytest.mark.asyncio
async def test_k_unset_pays_the_listed_rate(monkeypatch):
    monkeypatch.setattr(settings, "IDLE_PAY_BUDGET_K", None)
    incentive = await _run([_secure("secure-1", model_usd_per_hour=8.0)])

    assert _paid_rate(incentive.job_results["hk"][0]) == pytest.approx(HOURLY_RATE)


@pytest.mark.asyncio
async def test_the_cap_binds_on_normalized_pay_and_spot_stays_at_80_pct_of_secure(monkeypatch):
    monkeypatch.setattr(settings, "IDLE_PAY_BUDGET_K", 0.9)
    # 16 idle GPUs of one model whose Default Jobs earned 16 USD/hour: 1 USD per idle GPU-hour, cap 0.9 after
    # weight normalization (no mining score here, so the paid weights are divided by the burn-emission mass)
    nodes = [
        _secure("secure-1", model_usd_per_hour=16.0),
        _spot(filler_revenue_usd_per_hour=16.0, filler_revenue_per_gpu_hour=5.0),
    ]
    incentive = await _run(nodes)
    mass = incentive._weight_mass()
    secure, spot = incentive.job_results["hk"]

    assert mass < 1.0
    assert _paid_rate(secure) / mass == pytest.approx(0.9)
    assert _paid_rate(spot) == pytest.approx(SPOT_IDLE_SECURE_RATE_SHARE * _paid_rate(secure))
    assert _model_paid(incentive) / mass <= 0.9 * 16.0 + 1e-9


@pytest.mark.asyncio
async def test_the_secure_floor_never_lifts_above_the_cap(monkeypatch):
    monkeypatch.setattr(settings, "IDLE_PAY_BUDGET_K", 0.9)
    # 16 GPUs in an 8-GPU bucket are diluted to half; the floor (1.0 x 20 per filled GPU-hour) would lift each to 20.
    # Default Jobs earned 32 USD/hour on the model: 2 per idle GPU-hour, so the normalized rate stops at 1.8.
    nodes = [_secure(f"secure-{i}", model_usd_per_hour=32.0, per_filled_gpu_hour=20.0) for i in range(2)]
    incentive = await _run(nodes, _config(cap=8))
    mass = incentive._weight_mass()

    for node in incentive.job_results["hk"]:
        assert node.floor_top_up_rate
        assert _paid_rate(node) / mass == pytest.approx(1.8)
    assert _model_paid(incentive) / mass == pytest.approx(0.9 * 32.0)


@pytest.mark.asyncio
async def test_an_unpaid_spot_node_does_not_lower_the_cap(monkeypatch):
    monkeypatch.setattr(settings, "IDLE_PAY_BUDGET_K", 0.9)
    alone = await _run([_secure("secure-1", model_usd_per_hour=8.0)])
    # a 4-GPU spot node has no capacity in this config (only the 8-GPU bucket), so it is paid 0
    with_unpaid_spot = await _run(
        [
            _secure("secure-1", model_usd_per_hour=8.0),
            _spot(gpu_count=4, filler_revenue_usd_per_hour=8.0, filler_revenue_per_gpu_hour=5.0),
        ]
    )

    assert with_unpaid_spot.job_results["hk"][1].effective_rate == 0
    assert with_unpaid_spot._idle_rate_cap_by_model == pytest.approx(alone._idle_rate_cap_by_model)


@pytest.mark.asyncio
async def test_an_estimate_applies_the_cap_with_its_own_gpus_added(monkeypatch):
    monkeypatch.setattr(settings, "IDLE_PAY_BUDGET_K", 0.9)
    live = await _run([_secure("secure-1", model_usd_per_hour=8.0)], _config(cap=16))
    snapshot = live.get_snapshot()
    assert snapshot.rental.idle_gpus_by_model == {"H100": 8}

    estimator = RentalPriceIncentive(_config(cap=16), AsyncMock(), {}, {}, snapshot=snapshot)
    estimate = await estimator.estimate_executor(ExecutorEstimateParams(gpu_model=H100, gpu_count=8))

    # 16 idle GPUs share the model's 8 USD/hour: 0.45 per GPU-hour after normalization
    assert estimate.hourly_rate == pytest.approx(0.9 * 8.0 * estimator._weight_mass() / 16)


@pytest.mark.asyncio
async def test_a_model_without_a_default_job_sample_keeps_its_listed_rate(monkeypatch):
    monkeypatch.setattr(settings, "IDLE_PAY_BUDGET_K", 0.9)
    incentive = await _run([_secure("secure-1", model_usd_per_hour=None)])

    assert _paid_rate(incentive.job_results["hk"][0]) == pytest.approx(HOURLY_RATE)


@pytest.mark.asyncio
async def test_a_spot_node_seeded_at_the_secure_rate_is_counted_and_capped(monkeypatch):
    monkeypatch.setattr(settings, "IDLE_PAY_BUDGET_K", 0.9)
    # DAH-5721: a spot node with no filler sample for its configuration is paid the secure rate outside the spot pool;
    # another configuration of the model supplies the 16 USD/hour reading
    seeded = _spot("spot-seeded", filler_revenue_usd_per_hour=None, filler_revenue_per_gpu_hour=None)
    seeded.hourly_rate, seeded.sysbox_multiplier, seeded.driver_multiplier = HOURLY_RATE, 1.0, 1.0
    seeded.effective_rate = RentalPriceIncentive._spot_secure_rate(seeded)
    cap_idle_rates = RentalPriceIncentive._cap_idle_rates_at_default_job_yield

    def seed_then_cap(self):
        self._spot_seeded.append(seeded)
        cap_idle_rates(self)

    monkeypatch.setattr(RentalPriceIncentive, "_cap_idle_rates_at_default_job_yield", seed_then_cap)
    incentive = await _run([_secure("secure-1", model_usd_per_hour=16.0)])

    cap = incentive._idle_rate_cap_by_model["H100"]
    assert incentive._idle_gpus_by_model["H100"] == 16
    assert cap == pytest.approx(0.9 * 16.0 * incentive._weight_mass() / 16)
    assert seeded.effective_rate == pytest.approx(SPOT_IDLE_SECURE_RATE_SHARE * cap)


@pytest.mark.asyncio
async def test_the_cap_does_not_depend_on_the_order_of_a_models_readings(monkeypatch):
    monkeypatch.setattr(settings, "IDLE_PAY_BUDGET_K", 0.9)
    caps = []
    for readings in ((8.0, 16.0), (16.0, 8.0)):
        incentive = await _run([_secure(str(i), model_usd_per_hour=usd) for i, usd in enumerate(readings)])
        caps.append(incentive._idle_rate_cap_by_model["H100"])

    assert caps[0] == pytest.approx(caps[1])
