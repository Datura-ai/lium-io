"""DAH-6030 per-model cap: an idle GPU-hour is rated at most k x what its GPU model's Default Jobs earned per idle
GPU-hour, so a model's idle pay (listed rate, secure floor and spot together) stays within k x its Default Job revenue."""

import pytest
from core.config import settings

from tests.test_spot_node_pay import HOURLY_RATE, _config, _node, _paid_rate, _run, _spot


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
async def test_secure_and_spot_share_k_times_the_model_default_job_revenue(monkeypatch):
    monkeypatch.setattr(settings, "IDLE_PAY_BUDGET_K", 0.9)
    # 16 idle GPUs of one model whose Default Jobs earned 16 USD/hour: 1 USD per idle GPU-hour, cap 0.9
    nodes = [
        _secure("secure-1", model_usd_per_hour=16.0),
        _spot(filler_revenue_usd_per_hour=16.0, filler_revenue_per_gpu_hour=5.0),
    ]
    incentive = await _run(nodes)

    for node in incentive.job_results["hk"]:
        assert _paid_rate(node) == pytest.approx(0.9)
    assert _model_paid(incentive) == pytest.approx(0.9 * 16.0)


@pytest.mark.asyncio
async def test_the_secure_floor_never_lifts_above_the_cap(monkeypatch):
    monkeypatch.setattr(settings, "IDLE_PAY_BUDGET_K", 0.9)
    # 16 GPUs in an 8-GPU bucket are diluted to half; the floor (1.0 x 20 per filled GPU-hour) would lift each to 20.
    # Default Jobs earned 32 USD/hour on the model: 2 per idle GPU-hour, so the listed rate plus floor stop at 1.8.
    nodes = [_secure(f"secure-{i}", model_usd_per_hour=32.0, per_filled_gpu_hour=20.0) for i in range(2)]
    incentive = await _run(nodes, _config(cap=8))

    for node in incentive.job_results["hk"]:
        assert node.floor_top_up_rate
        assert _paid_rate(node) == pytest.approx(1.8)
    assert _model_paid(incentive) == pytest.approx(0.9 * 32.0)


@pytest.mark.asyncio
async def test_a_model_without_a_default_job_sample_keeps_its_listed_rate(monkeypatch):
    monkeypatch.setattr(settings, "IDLE_PAY_BUDGET_K", 0.9)
    incentive = await _run([_secure("secure-1", model_usd_per_hour=None)])

    assert _paid_rate(incentive.job_results["hk"][0]) == pytest.approx(HOURLY_RATE)
